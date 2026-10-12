"""Binance USD-M futures REST client for public data and optional execution.

Public endpoints work without credentials. Private account and order endpoints
use Binance HMAC-SHA256 signing and are only called by the explicit live-trading
entry points.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests


FUTURES_PROD = "https://fapi.binance.com"
FUTURES_TESTNET = "https://demo-fapi.binance.com"


class BinanceClient:
    """One Binance USD-M client for either production or testnet."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        api_secret: str = "",
        timeout: float = 10.0,
        recv_window: int = 5000,
        retries: int = 3,
        backoff: float = 1.0,
        proxy: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        self.recv_window = recv_window
        self.retries = retries
        self.backoff = backoff
        self.time_offset_ms = 0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "crypto-quant/0.1"})
        if api_key:
            self.session.headers.update({"X-MBX-APIKEY": api_key})
        self.proxies = {"http": proxy, "https": proxy} if proxy else None

    def _signed_timestamp_ms(self) -> int:
        return int(time.time() * 1000) + self.time_offset_ms

    def sync_time(self) -> int:
        response = self.session.get(
            self.base_url + "/fapi/v1/time",
            timeout=self.timeout,
            proxies=self.proxies,
        )
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"GET /fapi/v1/time {response.status_code}: {response.text}")
        server_ms = int(response.json()["serverTime"])
        self.time_offset_ms = server_ms - int(time.time() * 1000)
        logging.info("Synced Binance time offset: %+dms", self.time_offset_ms)
        return self.time_offset_ms

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Private endpoints require api_key and api_secret")
        payload = dict(params)
        payload["timestamp"] = self._signed_timestamp_ms()
        payload["recvWindow"] = self.recv_window
        query = urlencode(payload)
        payload["signature"] = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        return payload

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        signed: bool = False,
    ) -> Any:
        raw_params = params or {}
        delay = self.backoff
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                payload = self._sign(raw_params) if signed else dict(raw_params)
                payload = {k: v for k, v in payload.items() if v is not None}
                if method == "GET":
                    response = self.session.get(
                        self.base_url + path,
                        params=payload,
                        timeout=self.timeout,
                        proxies=self.proxies,
                    )
                else:
                    response = self.session.request(
                        method,
                        self.base_url + path,
                        data=payload,
                        timeout=self.timeout,
                        proxies=self.proxies,
                    )
                if response.status_code in (418, 429):
                    wait = float(response.headers.get("Retry-After", delay))
                    logging.warning("Rate limited on %s, waiting %.1fs", path, wait)
                    time.sleep(wait)
                    delay *= 2
                    continue
                if (
                    signed
                    and response.status_code == 400
                    and '"code":-1021' in response.text
                    and attempt < self.retries
                ):
                    self.sync_time()
                    continue
                if not 200 <= response.status_code < 300:
                    raise RuntimeError(
                        f"{method} {path} {response.status_code}: {response.text}"
                    )
                return response.json() if response.content else {}
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt >= self.retries:
                    break
                time.sleep(delay)
                delay *= 2
        raise last_exc or RuntimeError(f"{method} {path} failed after multiple retries")

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Compatibility wrapper retained for the market-data downloader."""
        return self._request("GET", path, params)

    # Public market data. ``futures`` is retained for historical live callers.
    def exchange_info(self, futures: bool = True) -> Dict[str, Any]:
        if not futures:
            raise ValueError("This client supports Binance USD-M futures only")
        return self._get("/fapi/v1/exchangeInfo")

    def ticker_24hr(self, futures: bool = True) -> List[Dict[str, Any]]:
        if not futures:
            raise ValueError("This client supports Binance USD-M futures only")
        return self._get("/fapi/v1/ticker/24hr")

    def klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 500,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        futures: bool = True,
    ) -> List[List[Any]]:
        if not futures:
            raise ValueError("This client supports Binance USD-M futures only")
        return self._get("/fapi/v1/klines", {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
            "startTime": start_ms,
            "endTime": end_ms,
        })

    def funding_history(
        self,
        symbol: str,
        limit: int = 1000,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return self._get("/fapi/v1/fundingRate", {
            "symbol": symbol.upper(),
            "limit": limit,
            "startTime": start_ms,
            "endTime": end_ms,
        })

    def futures_data_history(
        self,
        endpoint: str,
        symbol: str,
        period: str = "5m",
        limit: int = 500,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        allowed = {
            "topLongShortAccountRatio",
            "topLongShortPositionRatio",
            "globalLongShortAccountRatio",
            "takerlongshortRatio",
        }
        if endpoint not in allowed:
            raise ValueError(f"Unsupported futures data endpoint: {endpoint}")
        return self._get(f"/futures/data/{endpoint}", {
            "symbol": symbol.upper(),
            "period": period,
            "limit": min(int(limit), 500),
            "startTime": start_ms,
            "endTime": end_ms,
        })

    # Signed USD-M account and execution endpoints.
    def usdt_balance(self) -> Optional[float]:
        for row in self._request("GET", "/fapi/v2/balance", signed=True):
            if row.get("asset") == "USDT":
                return float(row.get("availableBalance", 0.0))
        return None

    def futures_account(self) -> Dict[str, Any]:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def positions(self) -> List[Dict[str, Any]]:
        rows = self._request("GET", "/fapi/v2/positionRisk", signed=True)
        return [row for row in rows if abs(float(row.get("positionAmt", 0.0))) > 0]

    def hedge_mode_enabled(self) -> bool:
        """Return True when the USD-M account uses dual-side (Hedge) Mode."""
        payload = self._request("GET", "/fapi/v1/positionSide/dual", signed=True)
        value = payload.get("dualSidePosition", False)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() == "true"

    def set_leverage(self, symbol: str, leverage: int) -> Any:
        return self._request("POST", "/fapi/v1/leverage", {
            "symbol": symbol.upper(),
            "leverage": int(leverage),
        }, signed=True)

    def set_margin_type(self, symbol: str, margin_type: str = "CROSSED") -> Any:
        try:
            return self._request("POST", "/fapi/v1/marginType", {
                "symbol": symbol.upper(),
                "marginType": margin_type.upper(),
            }, signed=True)
        except RuntimeError as exc:
            if "-4046" in str(exc):
                return {"msg": "already set"}
            if "-4168" in str(exc):
                logging.warning("Could not change %s margin type: %s", symbol.upper(), exc)
                return {"msg": "margin type not changed"}
            raise

    def market_order(
        self,
        symbol: str,
        side: str,
        quantity: str,
        *,
        reduce_only: bool = False,
        position_side: Optional[str] = None,
    ) -> Any:
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "MARKET",
            "quantity": quantity,
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        if position_side:
            params["positionSide"] = position_side.upper()
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def close(self) -> None:
        self.session.close()


def make_futures_client(testnet: bool = False, **kwargs: Any) -> BinanceClient:
    return BinanceClient(FUTURES_TESTNET if testnet else FUTURES_PROD, **kwargs)
