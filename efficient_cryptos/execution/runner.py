"""Live runner: signal -> risk engine -> Binance reconciliation, once per poll.

One cycle (``LiveRunner.run_cycle``):

1. **Account** -- read Binance NAV and free margin (signed runs). NAV re-anchors
   the engine; new targets are sized from ``min(NAV, free margin)``.
2. **Signal** -- ask the signal source for the latest completed cross-section.
3. **Mark & step** -- mark open positions with the completed 1m TWAP (risk
   checks every poll); on a *new* signal timestamp also run ``Engine.step``
   (PnL, risk, rebalance).
4. **Targets** -- only the LIVE book becomes exchange targets; in SHADOW the
   targets are empty, so the account is flattened.  Optional exchange-side risk
   controls may override targets from the real positions.
5. **Sync** -- on the first cycle, a new signal or a risk event: set leverage,
   build a quantity-delta reconcile plan (reduce legs first) and submit market
   orders (live mode only).
6. **Audit** -- write ``latest.json``, append ``audit.jsonl``, persist engine
   and runner state atomically.

Failure handling (unchanged from the battle-tested Developer_Crypto runner):

* ``BinanceNetworkError`` -> roll the engine back to the cycle checkpoint, log
  to ``runner_errors.jsonl`` and retry with exponential backoff (5s .. 120s).
  The adapter itself retries GETs and reconnects the proxy; POST orders are
  never replayed -- the next cycle reconciles against actual positions.
* ``ValueError``/``RuntimeError`` (bad cross-section, rejected order) -> same
  rollback and backoff, logged as ``cycle_error``.
* SIGINT/SIGTERM -> finish the current cycle, optionally flatten
  (``close_on_exit``), then exit.
"""
from __future__ import annotations

import hashlib
import json
import math
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from efficient_cryptos.engine.risk_engine import Engine, EngineConfig
from efficient_cryptos.execution.binance_adapter import BinanceNetworkError, BinanceFuturesAdapter
from efficient_cryptos.execution.guards import (
    ManualChangeGuard, apply_exchange_risk_controls, position_snapshot,
)
from efficient_cryptos.execution.sources.base import SignalSource, Sizing

AUDIT_POSITION_KEYS = ("symbol", "positionSide", "positionAmt", "leverage",
                       "entryPrice", "markPrice", "notional", "unRealizedProfit")


@dataclass
class RunnerConfig:
    output: Path
    symbols: list[str]
    network: str = "testnet"
    poll_seconds: float = 0.0
    gross_leverage: float = 3.0
    margin_buffer_pct: float = 0.20
    long_percent: float = 10.0
    short_percent: float = 10.0
    min_assets: int = 8
    live_orders: bool = False
    reset_state: bool = False
    engine: EngineConfig = field(default_factory=EngineConfig)
    # --- optional guards (all off by default) ---
    exchange_risk: dict | None = None          # ``risk`` section of config/default.yaml
    manual_guard_file: Path | None = None      # enable the manual-change guard
    reset_manual_halt: bool = False
    close_on_exit: bool = False

    @property
    def sizing(self) -> Sizing:
        return Sizing(self.gross_leverage, self.margin_buffer_pct, self.long_percent,
                      self.short_percent, self.min_assets)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


@dataclass
class _Account:
    snapshot: dict | None
    equity: float | None
    available: float | None
    sizing_equity: float


class LiveRunner:
    def __init__(self, config: RunnerConfig, source: SignalSource, adapter: BinanceFuturesAdapter):
        self.cfg = config
        self.source = source
        self.adapter = adapter
        self.engine = Engine(config.engine)
        self.signed = bool(adapter.api_key and adapter.api_secret)
        if config.live_orders and not self.signed:
            raise SystemExit(f"--live-orders requires {adapter.key_name} and {adapter.secret_name}")
        config.output.mkdir(parents=True, exist_ok=True)
        self.state_path = config.output / "engine_state.json"
        self.runner_path = config.output / "runner_state.json"
        self.errors_path = config.output / "runner_errors.jsonl"
        self.symbols = sorted({s.strip().upper() for s in config.symbols if s.strip()})
        self.last_timestamp: str | None = None
        self.first_cycle = True
        self.retry_delay = 5.0
        self.stop_requested = False
        self.guard = ManualChangeGuard(config.manual_guard_file) if config.manual_guard_file else None
        if self.guard and config.reset_manual_halt:
            self.guard.reset()

    # ------------------------------------------------------------------ setup
    def start(self) -> None:
        """Load state and anchor a brand-new run to the real account NAV."""
        if self.cfg.reset_state:
            self.state_path.unlink(missing_ok=True); self.runner_path.unlink(missing_ok=True)
        self.engine.load(self.state_path)
        # Existing state is never overwritten by a transient account query.
        account_equity = None
        while True:
            try:
                self.adapter.sync_market_time()
                if self.signed:
                    self.adapter.sync_server_time()
                    account_equity = self.adapter.account_equity()
                break
            except BinanceNetworkError as exc:
                print(json.dumps({"timestamp": _now(), "type": "network_error", "phase": "startup",
                                  "error": str(exc), "retry_in_seconds": 5.0}, ensure_ascii=False), flush=True)
                time.sleep(5.0)
        if account_equity is not None and not self.state_path.exists():
            if account_equity <= 0 or not math.isfinite(account_equity):
                raise SystemExit(f"invalid {self.cfg.network} account equity: {account_equity}")
            self.engine.equity = account_equity
            self.engine.peak_equity = account_equity
        runner_state = json.loads(self.runner_path.read_text()) if self.runner_path.exists() else {}
        self.last_timestamp = runner_state.get("last_signal_timestamp")
        if self.guard and self.guard.halted and self.cfg.live_orders:
            raise SystemExit("Manual-change guard is halted: inspect the account, then restart "
                             "with --reset-manual-halt. Reason: " + str(self.guard.state.get("reason")))

    # -------------------------------------------------------------- one cycle
    def _account(self) -> _Account:
        snapshot = self.adapter.account() if self.signed else None
        equity = self.adapter.account_equity(snapshot) if self.signed else None
        available = self.adapter.available_balance(snapshot) if self.signed else None
        sizing_equity = equity if self.signed else self.engine.equity
        if self.signed:
            # Re-anchor live risk to the exchange NAV: realized PnL, funding,
            # liquidations and manual changes are invisible to the simulation.
            if equity is not None and math.isfinite(equity) and equity > 0:
                self.engine.equity = equity
                self.engine.peak_equity = max(self.engine.peak_equity, equity)
            # Size new targets from free margin, not total NAV.
            if available is not None and math.isfinite(available) and available > 0:
                sizing_equity = min(sizing_equity, available)
        return _Account(snapshot, equity, available, sizing_equity)

    def _mark_and_step(self, timestamp: str, rows: list[dict], bars: list[dict],
                       sizing_equity: float) -> tuple[dict, bool]:
        engine = self.engine
        now = _now()
        mark_result = engine.mark(now, bars) if engine.positions or engine.shadow_positions else None
        is_new_signal = self.last_timestamp is None or timestamp > self.last_timestamp
        if is_new_signal:
            result = engine.step(timestamp, rows, bars, account_equity=sizing_equity)
            self.last_timestamp = timestamp
        else:
            result = mark_result or {
                "timestamp": now, "state": engine.state, "equity": engine.equity,
                "shadow_equity": engine.shadow_equity, "net_return": 0.0,
                "shadow_return": 0.0, "cost": 0.0, "risk_events": "",
                "positions": len(engine.positions),
            }
        return result, is_new_signal

    def _set_leverage(self, symbols: set[str], plan: list[dict]) -> None:
        # Every managed/held symbol, including ones without an order this cycle,
        # so an existing position never keeps a stale exchange leverage.
        responses = {s: self.adapter.set_leverage(s, self.cfg.gross_leverage) for s in sorted(symbols)}
        for order in plan:
            symbol = str(order.get("symbol", "")).upper()
            if symbol in responses:
                order["exchange_leverage"] = int(float(self.cfg.gross_leverage))
                order["leverage_response"] = responses[symbol]

    def run_cycle(self) -> dict:
        adapter, cfg = self.adapter, self.cfg
        account = self._account()
        batch = self.source.fetch(adapter, self.symbols, account.sizing_equity, cfg.sizing)
        timestamp, rows = batch.timestamp, batch.rows
        if batch.symbols is not None:
            self.symbols = batch.symbols
        # Include exchange-held symbols so positions that left the signal can be closed.
        account_rows = list(account.snapshot.get("positions", [])) if self.signed else []
        hedge_mode = adapter.hedge_mode() if self.signed else False
        existing = {str(r.get("symbol", "")).upper() for r in account_rows
                    if abs(float(r.get("positionAmt", 0.0))) > 0}
        market_symbols = sorted(set(self.symbols) | existing)
        # Completed 1m OHLC TWAP for risk marks; live mark price for order sizing.
        prices = adapter.mark_prices(market_symbols)
        risk_prices = adapter.one_minute_twap_prices(market_symbols)
        bars = [{"symbol": s, "close": risk_prices[s],
                 "quote_volume": batch.quote_volumes.get(s, 1_000_000_000.0)}
                for s in market_symbols if s in risk_prices]
        result, is_new_signal = self._mark_and_step(timestamp, rows, bars, account.sizing_equity)

        # Only the real execution book may produce exchange targets; the shadow
        # book is counterfactual and must never become Binance orders.
        targets = ({s: p.notional for s, p in self.engine.positions.items()}
                   if self.engine.state == "LIVE" else {})
        extra: dict[str, Any] = {}
        risk_override = False
        if cfg.exchange_risk is not None and self.signed:
            targets, notes = apply_exchange_risk_controls(targets, account_rows, account.equity, cfg.exchange_risk)
            extra["exchange_risk_notes"] = notes
            risk_override = bool(notes)

        all_order_symbols = sorted(set(targets) | existing)
        filters = adapter.exchange_filters(all_order_symbols) if all_order_symbols else {}
        plan = (adapter.build_reconcile_plan(targets, prices, filters, account_rows, hedge_mode=hedge_mode)
                if self.signed else adapter.build_order_plan(targets, prices, filters))
        submitted: list = []

        if self.guard and cfg.live_orders:
            changes = self.guard.check(position_snapshot(account_rows))
            if changes:
                # Someone else changed the account: flatten and halt.
                plan = adapter.build_reconcile_plan({}, prices, filters, account_rows, hedge_mode=hedge_mode)
                submitted = adapter.submit_market_orders(plan, cycle_id="manualhalt")
                extra["manual_halt"] = changes
                self.stop_requested = True
                return self._finish(timestamp, is_new_signal, hedge_mode, account, result,
                                    {}, plan, submitted, adapter.position_rows(), extra)

        should_sync = self.first_cycle or is_new_signal or bool(result.get("risk_events")) or risk_override
        if cfg.live_orders and should_sync:
            self._set_leverage(set(targets) | existing, plan)
            cycle = hashlib.sha1((timestamp + json.dumps(targets, sort_keys=True)).encode()).hexdigest()[:12]
            submitted = adapter.submit_market_orders(plan, cycle_id=cycle)
            # Re-read so the audit holds Binance's actual leverage and notional.
            if self.signed:
                account_rows = adapter.position_rows()
            if self.guard:
                self.guard.arm(position_snapshot(account_rows))
        return self._finish(timestamp, is_new_signal, hedge_mode, account, result,
                            targets, plan, submitted, account_rows, extra)

    def _finish(self, timestamp, is_new_signal, hedge_mode, account, result, targets,
                plan, submitted, account_rows, extra) -> dict:
        cfg = self.cfg
        exchange_positions = [
            {key: row.get(key) for key in AUDIT_POSITION_KEYS if key in row}
            for row in account_rows if abs(float(row.get("positionAmt", 0.0) or 0.0)) > 0
        ]
        payload = {
            "mode": f"{cfg.network.upper()}_LIVE" if cfg.live_orders else f"{cfg.network.upper()}_DRY_RUN",
            "factor_id": self.source.factor_id, "timestamp": timestamp,
            "signal_is_new": is_new_signal, "hedge_mode": hedge_mode,
            # Binance's raw NAV is kept apart from the (free-margin capped) sizing basis.
            "binance_equity": account.equity,
            "binance_available_balance": account.available,
            "sizing_equity": account.sizing_equity,
            "account_equity": account.equity,
            "engine": result, "targets": targets, "orders": plan,
            "submitted": submitted, "exchange_positions": exchange_positions,
            **extra,
        }
        (cfg.output / "latest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        with (cfg.output / "audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.engine.save(self.state_path)
        self.runner_path.write_text(json.dumps({"last_signal_timestamp": self.last_timestamp}, indent=2),
                                    encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload

    # ------------------------------------------------------------------- loop
    def _sleep(self, seconds: float) -> None:
        """Sleep in <=1s chunks so SIGINT/SIGTERM stop the runner promptly."""
        whole = int(seconds)
        for _ in range(whole):
            if self.stop_requested:
                return
            time.sleep(1.0)
        if seconds - whole > 0 and not self.stop_requested:
            time.sleep(seconds - whole)

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):
            self.stop_requested = True
            print(json.dumps({"timestamp": _now(), "type": "stop_requested", "signal": signum}), flush=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass  # not on the main thread (tests)

    def run(self) -> int:
        self._install_signal_handlers()
        self.start()
        try:
            while not self.stop_requested:
                # A failed request must not leave a half-processed bar in memory.
                checkpoint = self.engine.to_dict()
                checkpoint_timestamp = self.last_timestamp
                try:
                    self.run_cycle()
                    self.first_cycle = False
                    self.retry_delay = 5.0
                    if self.cfg.poll_seconds <= 0:
                        break
                    self._sleep(self.cfg.poll_seconds)
                except BinanceNetworkError as exc:
                    self.engine.load_dict(checkpoint)
                    self.last_timestamp = checkpoint_timestamp
                    _log(self.errors_path, {"timestamp": _now(), "type": "network_error",
                                            "error": str(exc), "retry_in_seconds": self.retry_delay})
                    self._sleep(self.retry_delay)
                    self.retry_delay = min(120.0, self.retry_delay * 2.0)
                except (ValueError, RuntimeError) as exc:
                    _log(self.errors_path, {"timestamp": _now(), "type": "cycle_error", "error": str(exc)})
                    self.engine.load_dict(checkpoint)
                    self.last_timestamp = checkpoint_timestamp
                    self._sleep(min(120.0, self.retry_delay))
                    self.retry_delay = min(120.0, self.retry_delay * 2.0)
            if self.stop_requested and self.cfg.close_on_exit and self.cfg.live_orders \
                    and not (self.guard and self.guard.halted):
                close_all_positions(self.adapter, output=self.cfg.output)
                if self.guard:
                    self.guard.disarm("intentional close_on_exit")
        finally:
            self.adapter.close()
        return 0


def close_all_positions(adapter: BinanceFuturesAdapter, *, output: Path | None = None) -> list:
    """Flatten every open position with reduce-first market orders."""
    rows = adapter.position_rows()
    held = sorted({str(r.get("symbol", "")).upper() for r in rows if abs(float(r.get("positionAmt", 0.0))) > 0})
    if not held:
        return []
    prices = adapter.mark_prices(held)
    plan = adapter.build_reconcile_plan({}, prices, adapter.exchange_filters(held), rows,
                                        hedge_mode=adapter.hedge_mode())
    submitted = adapter.submit_market_orders(plan, cycle_id="closeall")
    record = {"timestamp": _now(), "type": "close_all", "orders": plan, "submitted": submitted}
    if output is not None:
        with (Path(output) / "audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return submitted
