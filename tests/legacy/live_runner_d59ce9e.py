# Unmodified Developer_Crypto live_runner (commit d59ce9e), imports re-pointed.
# Kept only as the reference for tests/test_runner_golden.py, which checks that
# efficient_cryptos.execution.runner behaves identically.
"""Dry-run or confirmed live runner for Binance Futures factor signals.

By default this performs one market snapshot and prints an order plan. Sending
orders requires the selected network's explicit confirmation flags.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from datetime import datetime, timezone
import hashlib

from efficient_cryptos.execution.binance_adapter import BinanceNetworkError, BinanceTestnetAdapter
from efficient_cryptos.engine.risk_engine import Engine, EngineConfig
from efficient_cryptos.execution.sources.liquidity_factor import APPROVED_FACTOR, compute_liquidity_0041_5, targets_from_predictions

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,ZECUSDT,MOVRUSDT,NEARUSDT,QNTUSDT,XRPUSDT,DOGEUSDT,SUIUSDT,UNIUSDT,WLDUSDT,1000PEPEUSDT,BNBUSDT,LINKUSDT,ADAUSDT"


def _resolve_rebalance_bars(*, bars: int, hours: float | None,
                            minutes: float | None, bar_minutes: int = 30) -> int:
    """Resolve an operator supplied interval without silently rounding it.

    The realtime factor is built on completed 30-minute bars.  Therefore an
    arbitrary wall-clock interval is supported as long as it is an integer
    number of those bars (30m, 90m, 4h, 4.5h, 24h, ...).  Invalid values fail
    before the runner can submit an order instead of being rounded to a
    different trading policy.
    """
    if hours is not None and minutes is not None:
        raise ValueError("use only one of --rebalance-hours and --rebalance-minutes")
    value = minutes if minutes is not None else (hours * 60.0 if hours is not None else None)
    if value is None:
        if bars <= 0:
            raise ValueError("--rebalance-bars must be positive")
        return int(bars)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("rebalance interval must be a positive finite number")
    exact = value / float(bar_minutes)
    resolved = int(round(exact))
    if resolved <= 0 or not math.isclose(exact, resolved, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"rebalance interval {value:g} minutes is not a multiple of "
            f"the {bar_minutes}-minute signal bar"
        )
    return resolved


def _signals(path: Path, *, equity: float, gross_leverage: float,
             margin_buffer_pct: float, long_percent: float,
             short_percent: float, min_assets: int = 0) -> tuple[str, list[dict]]:
    # ``utf-8-sig`` also accepts ordinary UTF-8 while removing a BOM that is
    # commonly added by spreadsheet applications (otherwise the first header
    # is silently parsed as ``\ufefftimestamp``).
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("signal CSV is empty; expected header timestamp,symbol,signal,target_notional")
    required = {"timestamp", "symbol"}
    columns = set(rows[0])
    missing = sorted(required - columns)
    if missing:
        actual = ",".join(str(column) for column in rows[0] if column is not None)
        raise ValueError(
            "signal CSV is missing required column(s): " + ", ".join(missing)
            + "; expected timestamp,symbol and either signal,target_notional or prediction"
            + (f"; found: {actual}" if actual else "")
        )
    rows = [row for row in rows if any(str(value or "").strip() for value in row.values())]
    if not rows:
        raise ValueError("signal CSV contains no data rows")
    def parse_ts(value: str) -> datetime:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    latest_dt = max(parse_ts(row["timestamp"]) for row in rows)
    latest = latest_dt.isoformat()
    latest_rows = [row for row in rows if parse_ts(row["timestamp"]) == latest_dt]
    if "target_notional" in columns:
        if "signal" not in columns:
            raise ValueError("target_notional input also requires signal")
        return latest, latest_rows
    if "prediction" not in columns:
        raise ValueError("signal CSV must contain either target_notional or prediction")
    latest_rows = [r for r in latest_rows if str(r.get("prediction", "")).strip() not in {"", "nan", "None"}]
    if len(latest_rows) < min_assets:
        raise ValueError(f"insufficient signal cross-section: {len(latest_rows)} < {min_assets}")
    latest_rows.sort(key=lambda r: float(r["prediction"]))
    n = len(latest_rows)
    nl = max(1, min(n, math.ceil(n * long_percent / 100))) if long_percent else 0
    ns = max(1, min(n, math.ceil(n * short_percent / 100))) if short_percent else 0
    if nl + ns > n:
        raise ValueError(f"long/short selections overlap: {nl}+{ns}>{n}")
    budget = equity * (1.0 - margin_buffer_pct) * gross_leverage / 2.0
    result = []
    for row in latest_rows[-nl:] if nl else []:
        item = dict(row); item["signal"] = item["prediction"]; item["target_notional"] = budget / nl; result.append(item)
    for row in latest_rows[:ns] if ns else []:
        item = dict(row); item["signal"] = item["prediction"]; item["target_notional"] = -budget / ns; result.append(item)
    return latest, result


def main() -> int:
    ap = argparse.ArgumentParser(description="Shadow Testnet factor runner (dry-run by default)")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--signals", type=Path, help="CSV signal/prediction input")
    source.add_argument("--realtime-factor-id", choices=[APPROVED_FACTOR],
                        help="Calculate the approved factor from completed production 30m bars")
    source.add_argument("--realtime-factor-jsonl", type=Path,
                        help="Calculate an arbitrary generated factor from completed 30m bars")
    ap.add_argument("--realtime-factor-record-id",
                    help="factor_id inside --realtime-factor-jsonl; omit only when JSONL has one record")
    ap.add_argument("--realtime-history-bars", type=int, default=320,
                    help="Completed 30m bars fetched per symbol for a JSONL factor")
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--min-assets", type=int, default=8)
    ap.add_argument("--output", type=Path, default=Path("outputs/testnet_shadow_run"))
    ap.add_argument("--proxy", default=None)
    ap.add_argument(
        "--proxy-reconnect-command",
        default=None,
        help="Non-interactive proxy reconnect command. The default uses sudo -n "
             "and Anycast node 30; configure a matching NOPASSWD sudoers rule.",
    )
    ap.add_argument("--network", choices=("testnet", "mainnet"), default="testnet",
                    help="execution environment; defaults to isolated Testnet")
    ap.add_argument("--market-network", choices=("testnet", "mainnet"),
                    help="public market-data environment; defaults to --network")
    ap.add_argument("--poll-seconds", type=float, default=0.0, help="Repeat; 0 means one cycle")
    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    ap.add_argument("--gross-leverage", type=float, default=3.0)
    ap.add_argument("--margin-buffer-pct", type=float, default=0.20)
    ap.add_argument("--long-percent", type=float, default=10.0)
    ap.add_argument("--short-percent", type=float, default=10.0)
    ap.add_argument("--rebalance-bars", type=int, default=24)
    ap.add_argument("--rebalance-hours", type=float,
                    help="Overrides rebalance-bars; must be a multiple of 0.5h")
    ap.add_argument("--rebalance-minutes", type=float,
                    help="Overrides rebalance-bars; must be a multiple of 30")
    ap.add_argument("--symbol-stop", type=float, default=0.05)
    ap.add_argument("--symbol-take", type=float, default=0.30)
    ap.add_argument("--daily-stop", type=float, default=0.05)
    ap.add_argument("--daily-take", type=float, default=0.30)
    ap.add_argument("--portfolio-drawdown", type=float, default=0.10)
    ap.add_argument("--shadow-min-periods", type=int, default=3)
    ap.add_argument("--shadow-positive-periods", type=int, default=2)
    ap.add_argument("--shadow-recovery-pnl-pct", type=float, default=0.01)
    ap.add_argument("--live-orders", action="store_true", help="Send orders in the selected environment")
    ap.add_argument("--confirm-testnet-orders", action="store_true")
    ap.add_argument("--confirm-mainnet-orders", action="store_true",
                    help="second explicit confirmation required for Mainnet orders")
    ap.add_argument("--reset-state", action="store_true")
    args = ap.parse_args()
    if not math.isfinite(args.gross_leverage) or not args.gross_leverage.is_integer() or not 1 <= args.gross_leverage <= 125:
        raise SystemExit("--gross-leverage must be an integer in [1, 125] for Binance leverage")
    if args.proxy and not args.proxy_reconnect_command:
        args.proxy_reconnect_command = "sudo -n /usr/local/sbin/shadow-reconnect-anycast"
    if args.realtime_history_bars < 20:
        raise SystemExit("--realtime-history-bars must be at least 20")
    if args.realtime_factor_record_id and not args.realtime_factor_jsonl:
        raise SystemExit("--realtime-factor-record-id requires --realtime-factor-jsonl")
    if args.network == "testnet" and args.confirm_mainnet_orders:
        raise SystemExit("--confirm-mainnet-orders is only valid with --network mainnet")
    if args.network == "mainnet" and args.confirm_testnet_orders:
        raise SystemExit("--confirm-testnet-orders is only valid with --network testnet")
    if args.live_orders:
        required_confirmation = (
            args.confirm_mainnet_orders if args.network == "mainnet"
            else args.confirm_testnet_orders
        )
        if not required_confirmation:
            flag = "--confirm-mainnet-orders" if args.network == "mainnet" else "--confirm-testnet-orders"
            raise SystemExit(f"Refusing {args.network} orders: add {flag} explicitly")

    try:
        args.rebalance_bars = _resolve_rebalance_bars(
            bars=args.rebalance_bars,
            hours=args.rebalance_hours,
            minutes=args.rebalance_minutes,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = EngineConfig(
        initial_equity=args.initial_equity, gross_leverage=args.gross_leverage,
        margin_buffer_pct=args.margin_buffer_pct, rebalance_bars=args.rebalance_bars,
        per_symbol_stop_loss_pct=args.symbol_stop,
        per_symbol_take_profit_pct=args.symbol_take,
        daily_stop_loss_pct=args.daily_stop, daily_take_profit_pct=args.daily_take,
        portfolio_drawdown_pct=args.portfolio_drawdown,
        shadow_min_periods=args.shadow_min_periods,
        shadow_positive_periods=args.shadow_positive_periods,
        shadow_recovery_pnl_pct=args.shadow_recovery_pnl_pct,
    )
    engine = Engine(config)
    adapter = BinanceTestnetAdapter(
        proxy=args.proxy, network=args.network,
        reconnect_command=args.proxy_reconnect_command,
        market_network=args.market_network,
    )
    signed = bool(adapter.api_key and adapter.api_secret)
    if args.live_orders and not signed:
        raise SystemExit(
            f"--live-orders requires {adapter.key_name} and {adapter.secret_name}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "engine_state.json"
    runner_path = args.output / "runner_state.json"
    if args.reset_state:
        state_path.unlink(missing_ok=True); runner_path.unlink(missing_ok=True)
    engine.load(state_path)
    # On a brand-new signed Testnet run, use the actual account NAV as the
    # initial sizing/risk baseline. Existing state is never overwritten by a
    # transient account query during restart.
    account_equity = None
    while True:
        try:
            adapter.sync_market_time()
            if signed:
                adapter.sync_server_time()
                account_equity = adapter.account_equity()
            break
        except BinanceNetworkError as exc:
            print(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "network_error", "phase": "startup", "error": str(exc),
                "retry_in_seconds": 5.0,
            }, ensure_ascii=False), flush=True)
            time.sleep(5.0)
    if account_equity is not None and not state_path.exists():
        if account_equity <= 0 or not math.isfinite(account_equity):
            raise SystemExit(f"invalid Testnet account equity: {account_equity}")
        engine.equity = account_equity
        engine.peak_equity = account_equity
    runner_state = json.loads(runner_path.read_text()) if runner_path.exists() else {}
    last_timestamp = runner_state.get("last_signal_timestamp")
    symbols = sorted({value.strip().upper() for value in args.symbols.split(",") if value.strip()})
    first_cycle = True
    retry_delay = 5.0
    try:
        while True:
            # A failed request must not leave a half-processed bar in memory.
            # This checkpoint is discarded only after the complete cycle has
            # been written to disk.
            checkpoint = engine.to_dict()
            checkpoint_timestamp = last_timestamp
            try:
                account_snapshot = adapter.account() if signed else None
                binance_equity = adapter.account_equity(account_snapshot) if signed else None
                available_balance = adapter.available_balance(account_snapshot) if signed else None
                sizing_equity = binance_equity if signed else engine.equity
                if signed:
                    # Re-anchor live risk to the exchange NAV each cycle. This
                    # captures realized PnL, funding, liquidation and manual
                    # account changes that the local simulation cannot know.
                    if binance_equity is not None and math.isfinite(binance_equity) and binance_equity > 0:
                        engine.equity = binance_equity
                        engine.peak_equity = max(engine.peak_equity, binance_equity)
                    # Size new targets from free margin, not total NAV. Existing
                    # positions, unrealized PnL and open orders can make free
                    # margin much smaller than totalMarginBalance.
                    if available_balance is not None and math.isfinite(available_balance) and available_balance > 0:
                        sizing_equity = min(sizing_equity, available_balance)
                quote_volumes: dict[str, float] = {}
                if args.realtime_factor_id:
                    snapshot = compute_liquidity_0041_5(
                        adapter.realtime_factor_klines(symbols, interval="30m", limit=20)
                    )
                    timestamp = snapshot.timestamp
                    rows = targets_from_predictions(
                        snapshot, equity=sizing_equity,
                        gross_leverage=args.gross_leverage,
                        margin_buffer_pct=args.margin_buffer_pct,
                        long_percent=args.long_percent,
                        short_percent=args.short_percent,
                        min_assets=args.min_assets,
                    )
                    quote_volumes = snapshot.quote_volumes
                elif args.realtime_factor_jsonl:
                    from efficient_cryptos.execution.sources.jsonl_factor import compute_generic_factor
                    snapshot = compute_generic_factor(
                        adapter.realtime_factor_klines(
                            symbols, interval="30m", limit=args.realtime_history_bars
                        ),
                        factor_jsonl=args.realtime_factor_jsonl,
                        factor_id=args.realtime_factor_record_id,
                    )
                    timestamp = snapshot.timestamp
                    rows = targets_from_predictions(
                        snapshot, equity=sizing_equity,
                        gross_leverage=args.gross_leverage,
                        margin_buffer_pct=args.margin_buffer_pct,
                        long_percent=args.long_percent,
                        short_percent=args.short_percent,
                        min_assets=args.min_assets,
                    )
                    quote_volumes = snapshot.quote_volumes
                else:
                    timestamp, rows = _signals(
                        args.signals, equity=sizing_equity,
                        gross_leverage=args.gross_leverage,
                        margin_buffer_pct=args.margin_buffer_pct,
                        long_percent=args.long_percent,
                        short_percent=args.short_percent,
                        min_assets=args.min_assets,
                    )
                    symbols = sorted({str(row["symbol"]).upper() for row in rows})
                # Include exchange-held symbols before fetching prices. This
                # allows reconciliation to close positions that disappeared
                # from the latest signal cross-section.
                account_rows = list(account_snapshot.get("positions", [])) if signed else []
                hedge_mode = adapter.hedge_mode() if signed else False
                existing_symbols = {
                    str(row.get("symbol", "")).upper()
                    for row in account_rows
                    if abs(float(row.get("positionAmt", 0.0))) > 0
                }
                market_symbols = sorted(set(symbols) | existing_symbols)
                # Use completed 1-minute OHLC TWAP for risk marking. Keep the
                # live mark price separately for order sizing and quantities.
                prices = adapter.mark_prices(market_symbols)
                risk_prices = adapter.one_minute_twap_prices(market_symbols)
                now = datetime.now(timezone.utc).isoformat()
                bars = [{"symbol": symbol, "close": risk_prices[symbol],
                         "quote_volume": quote_volumes.get(symbol, 1_000_000_000.0)}
                        for symbol in market_symbols if symbol in risk_prices]
                mark_result = engine.mark(now, bars) if engine.positions or engine.shadow_positions else None
                is_new_signal = last_timestamp is None or timestamp > last_timestamp
                result = engine.step(timestamp, rows, bars, account_equity=sizing_equity) if is_new_signal else (mark_result or {
                    "timestamp": now, "state": engine.state, "equity": engine.equity,
                    "shadow_equity": engine.shadow_equity, "net_return": 0.0,
                    "shadow_return": 0.0, "cost": 0.0, "risk_events": "",
                    "positions": len(engine.positions),
                })
                if is_new_signal:
                    last_timestamp = timestamp
                # Only the real execution book may produce exchange targets.
                # ``shadow_positions`` is strictly counterfactual and must never
                # be converted into Binance orders while the engine is SHADOW.
                targets = (
                    {symbol: pos.notional for symbol, pos in engine.positions.items()}
                    if engine.state == "LIVE" else {}
                )
                all_order_symbols = sorted(set(targets) | existing_symbols)
                filters = adapter.exchange_filters(all_order_symbols) if all_order_symbols else {}
                plan = adapter.build_reconcile_plan(
                    targets, prices, filters, account_rows, hedge_mode=hedge_mode,
                ) if signed else adapter.build_order_plan(targets, prices, filters)
                submitted = []
                should_sync = first_cycle or is_new_signal or bool(result.get("risk_events"))
                if args.live_orders and should_sync:
                    # Set leverage for every managed/held symbol, including a
                    # symbol whose quantity delta is currently zero. Otherwise an
                    # existing position can keep Binance's stale leverage (or 0)
                    # simply because no new order was needed this cycle.
                    leverage_results = {}
                    for symbol in sorted(set(targets) | existing_symbols):
                        leverage_results[symbol] = adapter.set_leverage(
                            symbol, args.gross_leverage
                        )
                    for order in plan:
                        symbol = str(order.get("symbol", "")).upper()
                        if symbol in leverage_results:
                            order["exchange_leverage"] = int(float(args.gross_leverage))
                            order["leverage_response"] = leverage_results[symbol]
                    cycle = hashlib.sha1((timestamp + json.dumps(targets, sort_keys=True)).encode()).hexdigest()[:12]
                    submitted = adapter.submit_market_orders(
                        plan, cycle_id=cycle
                    )
                    # Re-read after synchronous market orders so the audit file
                    # contains Binance's actual position leverage and notional.
                    if signed:
                        account_rows = adapter.position_rows()
                exchange_positions = [
                    {
                        key: row.get(key)
                        for key in (
                            "symbol", "positionSide", "positionAmt", "leverage",
                            "entryPrice", "markPrice", "notional", "unRealizedProfit",
                        )
                        if key in row
                    }
                    for row in account_rows
                    if abs(float(row.get("positionAmt", 0.0) or 0.0)) > 0
                ]
                active_factor_id = (
                    args.realtime_factor_id
                    or args.realtime_factor_record_id
                    or (str(args.realtime_factor_jsonl) if args.realtime_factor_jsonl else "csv")
                )
                payload = {
                    "mode": f"{args.network.upper()}_LIVE" if args.live_orders else f"{args.network.upper()}_DRY_RUN",
                    "factor_id": active_factor_id, "timestamp": timestamp,
                    "signal_is_new": is_new_signal, "hedge_mode": hedge_mode,
                    # Keep Binance's raw account NAV separate from the sizing
                    # basis, which may be capped by free margin.
                    "binance_equity": binance_equity,
                    "binance_available_balance": available_balance,
                    "sizing_equity": sizing_equity,
                    "account_equity": binance_equity,
                    "engine": result, "targets": targets, "orders": plan,
                    "submitted": submitted, "exchange_positions": exchange_positions,
                }
                (args.output / "latest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                with (args.output / "audit.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                engine.save(state_path)
                runner_path.write_text(json.dumps({"last_signal_timestamp": last_timestamp}, indent=2), encoding="utf-8")
                print(json.dumps(payload, indent=2, ensure_ascii=False))
                first_cycle = False
                retry_delay = 5.0
                if args.poll_seconds <= 0:
                    break
                time.sleep(args.poll_seconds)
            except BinanceNetworkError as exc:
                engine.load_dict(checkpoint)
                last_timestamp = checkpoint_timestamp
                error = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": "network_error", "error": str(exc),
                    "retry_in_seconds": retry_delay,
                }
                with (args.output / "runner_errors.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(error, ensure_ascii=False) + "\n")
                print(json.dumps(error, ensure_ascii=False), flush=True)
                time.sleep(retry_delay)
                retry_delay = min(120.0, retry_delay * 2.0)
                continue
            except (ValueError, RuntimeError) as exc:
                error = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": "cycle_error", "error": str(exc),
                }
                with (args.output / "runner_errors.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(error, ensure_ascii=False) + "\n")
                print(json.dumps(error, ensure_ascii=False), flush=True)
                engine.load_dict(checkpoint)
                last_timestamp = checkpoint_timestamp
                time.sleep(min(120.0, retry_delay))
                retry_delay = min(120.0, retry_delay * 2.0)
                continue
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
