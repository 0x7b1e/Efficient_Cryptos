"""Deterministic in-memory Binance USD-M Futures for runner tests.

It replaces ``requests.Session`` inside the real adapter, so the adapter's own
signing, retry, reconnect and plan-building code is exercised unchanged.  A
``FakeClock`` drives ``time.time``/``time.sleep`` and ``datetime.now``.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

MINUTE_MS = 60_000
INTERVAL_MS = {"1m": MINUTE_MS, "30m": 30 * MINUTE_MS}


class StopRun(BaseException):
    """Raised by the fake clock to end a runner loop (escapes its except blocks)."""


class FakeClock:
    def __init__(self, start_ms: int, end_ms: int):
        self.ms = float(start_ms)
        self.end_ms = end_ms

    def time(self) -> float:
        return self.ms / 1000.0

    def sleep(self, seconds: float) -> None:
        self.ms += float(seconds) * 1000.0
        if self.ms >= self.end_ms:
            raise StopRun()

    def datetime_class(self):
        clock = self

        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.fromtimestamp(clock.time(), tz=tz or timezone.utc)
        return FakeDatetime


class FakeResponse:
    def __init__(self, payload, status: int = 200):
        self._payload = payload
        self.status_code = status
        self.headers: dict = {}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeExchange:
    """One-way (BOTH) USD-M account with market-order fills at the mark price."""

    def __init__(self, clock: FakeClock, symbols: list[str], *, wallet: float = 10_000.0,
                 crash: tuple[str, int, int, float] | None = None,
                 fail_order_numbers: set[int] = frozenset(), fee_bps: float = 5.0):
        self.clock = clock
        self.symbols = list(symbols)
        self.wallet = wallet
        self.positions: dict[str, dict] = {}
        self.crash = crash                       # (symbol, start_ms, end_ms, multiplier)
        self.fail_order_numbers = set(fail_order_numbers)
        self.fee_bps = fee_bps
        self.order_count = 0
        self.leverage: dict[str, int] = {}
        self.origin_ms = clock.ms - 30 * 24 * 3_600_000      # prices drift relative to this

    # ----------------------------------------------------------------- market
    def price(self, symbol: str, t_ms: float) -> float:
        i = self.symbols.index(symbol)
        hours = (t_ms - self.origin_ms) / 3_600_000.0
        value = (100.0 + 7 * i) * math.exp(0.03 * math.sin(hours / (1.3 + 0.37 * i) + i)
                                           + 0.004 * (i - 4) * hours / 24)
        if self.crash and self.crash[0] in (symbol, "*") and self.crash[1] <= t_ms < self.crash[2]:
            value *= self.crash[3]
        return round(value, 4)

    def quote_volume(self, symbol: str, open_ms: int) -> float:
        i = self.symbols.index(symbol)
        bar = open_ms // (30 * MINUTE_MS)
        return 1e7 * (1 + i) * (1.5 + math.sin(bar * (0.7 + 0.13 * i)))

    def klines(self, symbol: str, interval: str, limit: int) -> list[list]:
        step = INTERVAL_MS[interval]
        now = int(self.clock.ms)
        current_open = (now // step) * step
        rows = []
        for k in range(limit - 1, -1, -1):
            open_ms = current_open - k * step
            o = self.price(symbol, open_ms)
            c = self.price(symbol, open_ms + step - 1)
            h = max(o, c) * 1.001
            low = min(o, c) * 0.999
            qv = self.quote_volume(symbol, open_ms) * step / (30 * MINUTE_MS)
            rows.append([open_ms, str(o), str(h), str(low), str(c), str(qv / c),
                         open_ms + step - 1, str(qv), 100, "0", "0", "0"])
        return rows

    # ---------------------------------------------------------------- account
    def _position_rows(self) -> list[dict]:
        rows = []
        for symbol in self.symbols:
            pos = self.positions.get(symbol)
            qty = pos["qty"] if pos else 0.0
            entry = pos["entry"] if pos else 0.0
            mark = self.price(symbol, self.clock.ms)
            rows.append({
                "symbol": symbol, "positionSide": "BOTH", "positionAmt": f"{qty:.6f}",
                "entryPrice": f"{entry:.6f}", "markPrice": f"{mark:.6f}",
                "notional": f"{qty * mark:.6f}", "unRealizedProfit": f"{qty * (mark - entry):.6f}",
                "leverage": str(self.leverage.get(symbol, 20)),
            })
        return rows

    def account(self) -> dict:
        rows = self._position_rows()
        upnl = sum(float(r["unRealizedProfit"]) for r in rows)
        margin = sum(abs(float(r["notional"])) / int(r["leverage"]) for r in rows)
        total = self.wallet + upnl
        return {"totalMarginBalance": f"{total:.6f}", "totalWalletBalance": f"{self.wallet:.6f}",
                "availableBalance": f"{max(0.0, total - margin):.6f}", "positions": rows}

    def fill(self, params: dict) -> dict:
        self.order_count += 1
        if self.order_count in self.fail_order_numbers:
            raise requests.exceptions.ConnectionError("injected disconnect after order submission")
        symbol, side = params["symbol"], params["side"]
        qty = float(params["quantity"]) * (1 if side == "BUY" else -1)
        mark = self.price(symbol, self.clock.ms)
        pos = self.positions.setdefault(symbol, {"qty": 0.0, "entry": 0.0})
        old = pos["qty"]
        new = old + qty
        if old and old * qty < 0:                      # closing part
            closed = min(abs(qty), abs(old)) * (1 if old > 0 else -1)
            self.wallet += closed * (mark - pos["entry"])
        if new == 0 or abs(new) < 1e-12:
            pos.update(qty=0.0, entry=0.0)
        elif old * new <= 0 or abs(new) > abs(old):     # opened or added
            if old * new > 0:
                pos["entry"] = (old * pos["entry"] + qty * mark) / new
            else:
                pos["entry"] = mark
            pos["qty"] = new
        else:
            pos["qty"] = new
        self.wallet -= abs(qty) * mark * self.fee_bps / 10_000
        return {"orderId": self.order_count, "symbol": symbol, "status": "FILLED",
                "executedQty": params["quantity"], "avgPrice": f"{mark:.4f}",
                "clientOrderId": params.get("newClientOrderId")}

    # ------------------------------------------------------------------- HTTP
    def request(self, method, url, params=None, headers=None, timeout=None):
        path = urlparse(url).path
        params = dict(params or {})
        if path == "/fapi/v1/time":
            return FakeResponse({"serverTime": int(self.clock.ms)})
        if path == "/fapi/v1/klines":
            return FakeResponse(self.klines(params["symbol"], params["interval"], int(params["limit"])))
        if path == "/fapi/v1/premiumIndex":
            return FakeResponse([{"symbol": s, "markPrice": str(self.price(s, self.clock.ms))}
                                 for s in self.symbols])
        if path == "/fapi/v2/account":
            return FakeResponse(self.account())
        if path == "/fapi/v1/positionSide/dual":
            return FakeResponse({"dualSidePosition": False})
        if path == "/fapi/v1/exchangeInfo":
            filters = [{"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                       {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.001", "minQty": "0.001",
                        "maxQty": "100000"},
                       {"filterType": "MIN_NOTIONAL", "notional": "5"}]
            return FakeResponse({"symbols": [{"symbol": s, "filters": filters} for s in self.symbols]})
        if path == "/fapi/v1/leverage" and method == "POST":
            self.leverage[params["symbol"]] = int(params["leverage"])
            return FakeResponse({"symbol": params["symbol"], "leverage": int(params["leverage"])})
        if path == "/fapi/v1/order" and method == "POST":
            return FakeResponse(self.fill(params))
        return FakeResponse({"code": -1, "msg": f"unknown {method} {path}"}, status=404)

    def close(self):
        pass
