"""Binance Futures adapter for Testnet or Mainnet market data and execution."""
from __future__ import annotations

import hashlib
import hmac
import os
import shlex
import socket
import subprocess
import time
from decimal import Decimal, ROUND_DOWN
from typing import Any
from urllib.parse import urlencode

import requests


TESTNET_BASE_URL = "https://testnet.binancefuture.com"
MAINNET_BASE_URL = "https://fapi.binance.com"


class BinanceNetworkError(RuntimeError):
    """A request failed in a way that is safe for the runner to retry later."""


class BinanceFuturesAdapter:
    def __init__(self, *, api_key: str | None = None, api_secret: str | None = None,
                 proxy: str | None = None, timeout: float = 15.0,
                 network: str = "testnet", reconnect_command: str | None = None,
                 reconnect_timeout: float = 45.0, market_network: str | None = None):
        network = str(network).lower().strip()
        if network not in {"testnet", "mainnet"}:
            raise ValueError("network must be testnet or mainnet")
        self.network = network
        self.base_url = TESTNET_BASE_URL if network == "testnet" else MAINNET_BASE_URL
        market_network = str(market_network or network).lower().strip()
        if market_network not in {"testnet", "mainnet"}:
            raise ValueError("market_network must be testnet or mainnet")
        self.market_network = market_network
        self.market_base_url = TESTNET_BASE_URL if market_network == "testnet" else MAINNET_BASE_URL
        key_name = "BINANCE_TESTNET_API_KEY" if network == "testnet" else "BINANCE_MAINNET_API_KEY"
        secret_name = "BINANCE_TESTNET_API_SECRET" if network == "testnet" else "BINANCE_MAINNET_API_SECRET"
        self.api_key = api_key or os.getenv(key_name, "")
        self.api_secret = api_secret or os.getenv(secret_name, "")
        self.key_name, self.secret_name = key_name, secret_name
        self.timeout = timeout
        self.reconnect_command = str(reconnect_command or "").strip()
        self.reconnect_timeout = float(reconnect_timeout)
        self.session = requests.Session()
        if proxy:
            self.session.proxies.update({"http": proxy, "https": proxy})
        self.server_time_offset_ms = 0
        self.market_time_offset_ms = 0

    def close(self) -> None:
        self.session.close()

    def _proxy_is_healthy(self) -> bool:
        if not self.session.proxies:
            return True
        proxy = self.session.proxies.get("https") or self.session.proxies.get("http")
        try:
            host_port = str(proxy).split("://", 1)[-1].rsplit("/", 1)[0]
            host, port = host_port.rsplit(":", 1)
            with socket.create_connection((host.strip("[]"), int(port)), timeout=2):
                return True
        except (OSError, ValueError):
            return False

    def _reconnect_proxy(self) -> None:
        if not self.reconnect_command:
            return
        try:
            subprocess.run(
                shlex.split(self.reconnect_command), check=False,
                timeout=self.reconnect_timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"proxy reconnect command failed or timed out: {self.reconnect_command}"
            ) from exc
        deadline = time.time() + self.reconnect_timeout
        while time.time() < deadline:
            if self._proxy_is_healthy():
                return
            time.sleep(1.0)
        raise RuntimeError(
            "proxy reconnect timed out; 127.0.0.1:1087 is not ready"
        )

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None,
                 signed: bool = False, market: bool = False) -> Any:
        base_params = dict(params or {})
        if signed:
            if not self.api_key or not self.api_secret:
                raise RuntimeError(
                    f"signed Binance {self.network} request requires {self.key_name}/{self.secret_name}"
                )
        headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}
        # GETs are idempotent and can be retried. POST orders are deliberately
        # never replayed: a broken connection can happen after Binance accepted
        # the order, so the caller must reconcile positions before retrying.
        attempts = 4 if method.upper() == "GET" else 1
        for attempt in range(attempts):
            request_params = dict(base_params)
            if signed:
                offset = self.market_time_offset_ms if market else self.server_time_offset_ms
                request_params["timestamp"] = int(time.time() * 1000) + offset
                request_params.setdefault("recvWindow", 5000)
                request_params.pop("signature", None)
                query = urlencode(request_params, doseq=True)
                request_params["signature"] = hmac.new(
                    self.api_secret.encode(), query.encode(), hashlib.sha256
                ).hexdigest()
            try:
                response = self.session.request(
                    method, (self.market_base_url if market else self.base_url) + path, params=request_params,
                    headers=headers, timeout=self.timeout,
                )
                if response.status_code >= 500 or response.status_code in (418, 429):
                    if method.upper() == "GET" and attempt + 1 < attempts:
                        if response.status_code not in (418, 429):
                            try:
                                self._reconnect_proxy()
                            except Exception as reconnect_exc:
                                raise BinanceNetworkError(
                                    f"Binance {self.network} proxy recovery failed for GET {path}"
                                ) from reconnect_exc
                        retry_after = response.headers.get("Retry-After")
                        try:
                            delay = max(1.0, float(retry_after)) if retry_after else 2.0 ** attempt
                        except ValueError:
                            delay = 2.0 ** attempt
                        time.sleep(min(120.0, delay))
                        continue
                break
            except requests.exceptions.RequestException as exc:
                if method.upper() != "GET":
                    raise BinanceNetworkError(
                        f"Binance {self.network} {method} {path} lost connection; "
                        "order status must be reconciled before retrying"
                    ) from exc
                if attempt + 1 >= attempts:
                    raise BinanceNetworkError(
                        f"Binance {self.network} GET {path} failed after {attempts} attempts"
                    ) from exc
                try:
                    self._reconnect_proxy()
                except Exception as reconnect_exc:
                    raise BinanceNetworkError(
                        f"Binance {self.network} proxy recovery failed for GET {path}"
                    ) from reconnect_exc
                time.sleep(min(8.0, 2.0 ** attempt))
        if response.status_code >= 500 or response.status_code in (418, 429):
            detail = ("execution status is unknown for POST and must be reconciled before retrying"
                      if method.upper() == "POST" else "retry later")
            raise BinanceNetworkError(
                f"Binance {self.network} {method} {path} returned HTTP {response.status_code}; {detail}"
            )
        if response.status_code >= 400:
            if response.status_code == 401 and '"code":-2015' in response.text:
                raise RuntimeError(
                    f"Binance {self.network} rejected the API credentials (-2015). "
                    f"Use a {self.network} Futures API key, enable Futures permissions, "
                    "and allow the proxy's public IP if IP restrictions are enabled. "
                    f"Endpoint: {path}"
                )
            raise RuntimeError(f"Binance {self.network} {method} {path} {response.status_code}: {response.text[:500]}")
        return response.json()

    def market_server_time(self) -> int:
        return int(self._request("GET", "/fapi/v1/time", market=True)["serverTime"])

    def sync_server_time(self) -> int:
        local = int(time.time() * 1000)
        server = int(self._request("GET", "/fapi/v1/time")["serverTime"])
        self.server_time_offset_ms = server - local
        return self.server_time_offset_ms

    def sync_market_time(self) -> int:
        local = int(time.time() * 1000)
        server = self.market_server_time()
        self.market_time_offset_ms = server - local
        return self.market_time_offset_ms

    def completed_klines(self, symbol: str, interval: str = "30m", limit: int = 20) -> list[list[Any]]:
        # One local clock read is sufficient for filtering the currently
        # forming bar.  Calling /time once per symbol needlessly burns rate
        # limit and was the cause of repeated 418 public-IP bans.
        now_ms = int(time.time() * 1000) + self.market_time_offset_ms
        rows = self._request("GET", "/fapi/v1/klines", params={
            "symbol": symbol.upper(), "interval": interval, "limit": limit,
        }, market=True)
        # closeTime is inclusive. Never expose the still-forming bar.
        return [row for row in rows if int(row[6]) < now_ms]

    def realtime_factor_klines(self, symbols: list[str], *, interval: str = "30m",
                               limit: int = 20) -> dict[str, list[list[Any]]]:
        return {symbol.upper(): self.completed_klines(symbol, interval, limit)
                for symbol in symbols}

    def mark_prices(self, symbols: list[str]) -> dict[str, float]:
        rows = self._request("GET", "/fapi/v1/premiumIndex", market=True)
        wanted = {x.upper() for x in symbols}
        return {str(x["symbol"]).upper(): float(x["markPrice"])
                for x in rows if str(x.get("symbol", "")).upper() in wanted}

    def one_minute_twap_prices(self, symbols: list[str]) -> dict[str, float]:
        """Return the latest completed 1m OHLC TWAP approximation per symbol.

        Binance's public kline endpoint does not expose a native TWAP field.
        For each completed 1-minute candle we use the standard OHLC average
        ``(open + high + low + close) / 4`` as the bar TWAP approximation.
        The currently forming candle is excluded by ``completed_klines``.
        """
        result: dict[str, float] = {}
        for symbol in symbols:
            rows = self.completed_klines(symbol, interval="1m", limit=2)
            if not rows:
                continue
            row = rows[-1]
            values = [float(row[index]) for index in (1, 2, 3, 4)]
            if all(value > 0 for value in values):
                result[str(symbol).upper()] = sum(values) / 4.0
        return result

    def account(self) -> dict[str, Any]:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def account_equity(self, account: dict[str, Any] | None = None) -> float:
        account = self.account() if account is None else account
        return float(account.get("totalMarginBalance") or account.get("totalWalletBalance") or 0.0)

    def available_balance(self, account: dict[str, Any] | None = None) -> float:
        """Free balance available for opening new cross-margin positions."""
        account = self.account() if account is None else account
        return float(account.get("availableBalance") or 0.0)

    def hedge_mode(self) -> bool:
        payload = self._request("GET", "/fapi/v1/positionSide/dual", signed=True)
        return bool(payload.get("dualSidePosition"))

    def set_leverage(self, symbol: str, leverage: int | float) -> dict[str, Any]:
        """Set the exchange leverage for one USDⓈ-M symbol.

        ``gross_leverage`` controls portfolio sizing in the engine, but does
        not change Binance's per-symbol account setting by itself. Binance
        accepts an integer leverage value, so reject fractional values instead
        of silently applying a different setting.
        """
        value = float(leverage)
        if not value.is_integer() or not 1 <= value <= 125:
            raise ValueError("exchange leverage must be an integer in [1, 125]")
        return self._request("POST", "/fapi/v1/leverage", params={
            "symbol": str(symbol).upper(), "leverage": int(value),
        }, signed=True)

    def position_rows(self) -> list[dict[str, Any]]:
        return list(self.account().get("positions", []))

    def positions(self) -> dict[str, float]:
        rows = self.account().get("positions", [])
        result = {}
        for row in rows:
            qty = float(row.get("positionAmt", 0.0))
            if abs(qty) > 0:
                result[str(row["symbol"]).upper()] = qty
        return result

    def exchange_filters(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        wanted = {x.upper() for x in symbols}
        info = self._request("GET", "/fapi/v1/exchangeInfo", market=True)
        result = {}
        for row in info.get("symbols", []):
            symbol = str(row.get("symbol", "")).upper()
            if symbol not in wanted:
                continue
            filters = {str(x.get("filterType")): x for x in row.get("filters", [])}
            lot = filters.get("LOT_SIZE", {})
            market_lot = filters.get("MARKET_LOT_SIZE", {})
            min_notional = filters.get("MIN_NOTIONAL", {})
            result[symbol] = {
                "step_size": float(lot.get("stepSize", 0.0)),
                "min_qty": float(lot.get("minQty", 0.0)),
                "min_notional": float(min_notional.get("notional", 0.0)),
                "market_step_size": float(market_lot.get("stepSize", 0.0)),
                "market_min_qty": float(market_lot.get("minQty", 0.0)),
                "market_max_qty": float(market_lot.get("maxQty", 0.0)),
            }
        return result

    @staticmethod
    def round_step(value: float, step: float) -> float:
        if step <= 0:
            return value
        value_d, step_d = Decimal(str(value)), Decimal(str(step))
        return float((value_d / step_d).to_integral_value(rounding=ROUND_DOWN) * step_d)

    @staticmethod
    def format_quantity(value: float, step: float) -> str:
        """Serialize quantity with exactly the exchange LOT_SIZE precision."""
        value_d = Decimal(str(value))
        if step > 0:
            step_d = Decimal(str(step))
            value_d = (value_d / step_d).to_integral_value(rounding=ROUND_DOWN) * step_d
            value_d = value_d.quantize(step_d, rounding=ROUND_DOWN)
        # Fixed-point output avoids scientific notation and float tails such
        # as 0.30000000000000004 that Binance rejects with -1111.
        return format(value_d, "f")

    def build_order_plan(self, target_notionals: dict[str, float], prices: dict[str, float],
                         filters: dict[str, dict[str, float]] | None = None) -> list[dict[str, Any]]:
        filters = filters or {}
        plan = []
        for symbol, notional in sorted(target_notionals.items()):
            symbol = symbol.upper()
            price = float(prices.get(symbol, 0.0))
            if price <= 0 or notional == 0:
                continue
            qty = abs(float(notional)) / price
            rule = filters.get(symbol, {})
            step = float(rule.get("market_step_size") or rule.get("step_size", 0.0))
            qty = self.round_step(qty, step)
            actual_notional = qty * price
            reason = ""
            if rule.get("market_max_qty", 0.0) and qty > float(rule["market_max_qty"]):
                reason = "above_market_max_qty"
            elif qty < float(rule.get("market_min_qty") or rule.get("min_qty", 0.0)):
                reason = "below_min_qty"
            elif actual_notional < float(rule.get("min_notional", 0.0)):
                reason = "below_min_notional"
            plan.append({
                "symbol": symbol,
                "side": "BUY" if notional > 0 else "SELL",
                "position_side": "BOTH",
                "quantity": qty,
                "step_size": step,
                "price": price,
                "target_notional": float(notional),
                "actual_notional": actual_notional,
                "rejected": bool(reason),
                "reason": reason,
            })
        return plan

    def build_reconcile_plan(self, target_notionals: dict[str, float], prices: dict[str, float],
                             filters: dict[str, dict[str, float]],
                             position_rows: list[dict[str, Any]], *, hedge_mode: bool,
                             rebalance_tolerance_pct: float = 0.005) -> list[dict[str, Any]]:
        """Build quantity deltas, not full target orders, against exchange positions."""
        current: dict[tuple[str, str], float] = {}
        for row in position_rows:
            symbol = str(row.get("symbol", "")).upper()
            side = str(row.get("positionSide", "BOTH")).upper()
            qty = float(row.get("positionAmt", 0.0))
            if symbol:
                current[(symbol, side)] = qty
        plan: list[dict[str, Any]] = []
        symbols = sorted(set(target_notionals) | {symbol for symbol, _ in current})

        def append(symbol: str, side: str, position_side: str, qty: float,
                   target_notional: float, reduce_only: bool) -> None:
            price = float(prices.get(symbol, 0.0))
            rule = filters.get(symbol, {})
            step = float(rule.get("market_step_size") or rule.get("step_size", 0.0))
            qty = self.round_step(abs(qty), step)
            actual = qty * price
            reason = ""
            if price <= 0:
                reason = "missing_price"
            elif rule.get("market_max_qty", 0.0) and qty > float(rule["market_max_qty"]):
                reason = "above_market_max_qty"
            elif qty <= 0 or qty < float(rule.get("market_min_qty") or rule.get("min_qty", 0.0)):
                reason = "below_min_qty"
            elif actual < float(rule.get("min_notional", 0.0)) and not reduce_only:
                reason = "below_min_notional"
            plan.append({
                "symbol": symbol, "side": side, "position_side": position_side,
                "quantity": qty, "step_size": step,
                "price": price, "target_notional": target_notional,
                "actual_notional": actual, "reduce_only": reduce_only,
                "rejected": bool(reason), "reason": reason,
            })

        for symbol in symbols:
            price = float(prices.get(symbol, 0.0))
            if price <= 0:
                continue
            target_notional = float(target_notionals.get(symbol, 0.0))
            target_qty = target_notional / price
            if not hedge_mode:
                existing = current.get((symbol, "BOTH"), 0.0)
                delta = target_qty - existing
                if abs(delta) > 0:
                    if existing and target_qty and existing * target_qty < 0:
                        append(symbol, "SELL" if existing > 0 else "BUY", "BOTH",
                               abs(existing), target_notional, True)
                        append(symbol, "BUY" if target_qty > 0 else "SELL", "BOTH",
                               abs(target_qty), target_notional, False)
                        continue
                    # Do not churn an already-correct position because mark
                    # price moved by a few ticks between two reconciliations.
                    # A full close (target=0) is never suppressed.
                    if target_notional and existing * target_qty > 0 and abs(delta * price) < max(
                        10.0, abs(target_notional) * rebalance_tolerance_pct
                    ):
                        continue
                    reducing = existing != 0 and (target_qty == 0 or existing * delta < 0) and abs(target_qty) <= abs(existing)
                    append(symbol, "BUY" if delta > 0 else "SELL", "BOTH", delta,
                           target_notional, reducing)
                continue
            long_qty = abs(current.get((symbol, "LONG"), 0.0))
            short_qty = abs(current.get((symbol, "SHORT"), 0.0))
            wanted_long = max(target_qty, 0.0)
            wanted_short = max(-target_qty, 0.0)
            if short_qty and wanted_short == 0:
                append(symbol, "BUY", "SHORT", short_qty, target_notional, True)
            if long_qty and wanted_long == 0:
                append(symbol, "SELL", "LONG", long_qty, target_notional, True)
            long_delta = wanted_long - long_qty
            short_delta = wanted_short - short_qty
            if wanted_long and abs(long_delta) > 0:
                if long_qty and abs(long_delta * price) < max(10.0, abs(target_notional) * rebalance_tolerance_pct):
                    long_delta = 0.0
            if wanted_short and abs(short_delta) > 0:
                if short_qty and abs(short_delta * price) < max(10.0, abs(target_notional) * rebalance_tolerance_pct):
                    short_delta = 0.0
            if wanted_long and abs(long_delta) > 0:
                append(symbol, "BUY" if long_delta > 0 else "SELL", "LONG", long_delta,
                       target_notional, long_delta < 0)
            if wanted_short and abs(short_delta) > 0:
                append(symbol, "SELL" if short_delta > 0 else "BUY", "SHORT", short_delta,
                       target_notional, short_delta < 0)
        # Close/reduce legs first. Opening legs first can temporarily require
        # more margin when a rebalance replaces an existing portfolio.
        return sorted(plan, key=lambda order: (not bool(order.get("reduce_only")), order["symbol"]))

    def submit_market_orders(self, plan: list[dict[str, Any]], *, cycle_id: str = "",
                             leverage: int | float | None = None) -> list[dict[str, Any]]:
        if not self.api_key or not self.api_secret:
            raise RuntimeError(f"{self.network} orders require {self.key_name}/{self.secret_name}")
        results = []
        configured_symbols: set[str] = set()
        for order_index, order in enumerate(plan):
            if order.get("rejected") or order.get("quantity", 0) <= 0:
                continue
            symbol = str(order["symbol"]).upper()
            if leverage is not None and symbol not in configured_symbols:
                leverage_result = self.set_leverage(symbol, leverage)
                configured_symbols.add(symbol)
                order["exchange_leverage"] = int(float(leverage))
                order["leverage_response"] = leverage_result
            params = {
                "symbol": symbol, "side": order["side"], "type": "MARKET",
                "quantity": self.format_quantity(
                    float(order["quantity"]),
                    float(order.get("step_size", 0.0)),
                ),
                "newOrderRespType": "RESULT",
                "positionSide": order.get("position_side", "BOTH"),
            }
            if cycle_id:
                # Keep the ID short and unique across long/short hedge legs.
                suffix = f"{order_index}:{order['symbol']}:{order['side']}:{order.get('position_side','BOTH')}"
                digest = hashlib.sha1(suffix.encode()).hexdigest()[:10]
                params["newClientOrderId"] = ("sh_" + cycle_id + "_" + str(order_index) + "_" + digest)[:36]
            if order.get("reduce_only") and params["positionSide"] == "BOTH":
                params["reduceOnly"] = "true"
            try:
                results.append(self._request("POST", "/fapi/v1/order", params=params, signed=True))
            except BinanceNetworkError:
                raise
            except Exception as exc:
                results.append({
                    "symbol": symbol,
                    "client_order_id": params.get("newClientOrderId"),
                    "error": str(exc),
                    "order_error": True,
                })
        return results


# Historical name used by older scripts and notebooks.
BinanceTestnetAdapter = BinanceFuturesAdapter
