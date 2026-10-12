"""Command line for the live runner: ``python -m efficient_cryptos.execution.cli``.

Dry-run by default: one market snapshot, an order plan, no orders.  Sending
orders needs ``--live-orders`` plus the network's confirmation flag
(``--confirm-testnet-orders`` or ``--confirm-mainnet-orders``) and API keys in
``BINANCE_{TESTNET,MAINNET}_API_KEY`` / ``..._API_SECRET``.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from efficient_cryptos.engine.risk_engine import EngineConfig
from efficient_cryptos.execution.binance_adapter import BinanceFuturesAdapter
from efficient_cryptos.execution.runner import LiveRunner, RunnerConfig, close_all_positions
from efficient_cryptos.execution.sources.base import (
    AlphaSource, CsvSource, JsonlSource, LiquiditySource, MlSource,
)
from efficient_cryptos.execution.sources.liquidity_factor import APPROVED_FACTOR

DEFAULT_SYMBOLS = ("BTCUSDT,ETHUSDT,SOLUSDT,ZECUSDT,MOVRUSDT,NEARUSDT,QNTUSDT,XRPUSDT,DOGEUSDT,"
                   "SUIUSDT,UNIUSDT,WLDUSDT,1000PEPEUSDT,BNBUSDT,LINKUSDT,ADAUSDT")


def resolve_rebalance_bars(*, bars: int, hours: float | None,
                           minutes: float | None, bar_minutes: int = 30) -> int:
    """Resolve an operator supplied interval without silently rounding it.

    Signals are built on completed 30-minute bars, so any interval that is an
    integer number of bars is accepted (30m, 90m, 4h, 4.5h, 24h, ...).  Other
    values fail before any order instead of becoming a different policy.
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Risk-managed Binance Futures factor runner (dry-run by default)")
    src = ap.add_argument_group("signal source (choose one)")
    source = src.add_mutually_exclusive_group()
    source.add_argument("--signals", type=Path, help="CSV with prediction or signal,target_notional")
    source.add_argument("--realtime-factor-id", choices=[APPROVED_FACTOR],
                        help="built-in factor computed from completed 30m bars")
    source.add_argument("--realtime-factor-jsonl", type=Path,
                        help="any generated factor record computed from completed 30m bars")
    source.add_argument("--alpha-config", type=Path,
                        help="Alpha101/191 live scan; YAML with alpha_live/universe/portfolio sections")
    source.add_argument("--ml-model", type=Path, help="trained model directory or checkpoint")
    src.add_argument("--realtime-factor-record-id",
                     help="factor_id inside --realtime-factor-jsonl; omit only when JSONL has one record")
    src.add_argument("--realtime-history-bars", type=int, default=320,
                     help="Completed 30m bars fetched per symbol for a JSONL factor")
    src.add_argument("--ml-device", default="auto")
    src.add_argument("--ml-k-long", type=int, default=2)
    src.add_argument("--ml-m-short", type=int, default=1)
    src.add_argument("--ml-history-bars", type=int, default=260)

    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--min-assets", type=int, default=8)
    ap.add_argument("--output", type=Path, default=Path("outputs/testnet_shadow_run"))

    net = ap.add_argument_group("network and reconnect")
    net.add_argument("--proxy", default=None)
    net.add_argument("--proxy-reconnect-command", default=None,
                     help="Non-interactive proxy reconnect command; with --proxy it defaults to "
                          "'sudo -n /usr/local/sbin/shadow-reconnect-anycast' "
                          "(install with scripts/06_live/install_anycast_reconnect.sh)")
    net.add_argument("--network", choices=("testnet", "mainnet"), default="testnet",
                     help="execution environment; defaults to isolated Testnet")
    net.add_argument("--market-network", choices=("testnet", "mainnet"),
                     help="public market-data environment; defaults to --network")
    net.add_argument("--poll-seconds", type=float, default=0.0, help="Repeat; 0 means one cycle")

    size = ap.add_argument_group("portfolio and engine risk")
    size.add_argument("--initial-equity", type=float, default=10_000.0)
    size.add_argument("--gross-leverage", type=float, default=3.0)
    size.add_argument("--margin-buffer-pct", type=float, default=0.20)
    size.add_argument("--long-percent", type=float, default=10.0)
    size.add_argument("--short-percent", type=float, default=10.0)
    size.add_argument("--rebalance-bars", type=int, default=24)
    size.add_argument("--rebalance-hours", type=float, help="Overrides rebalance-bars; must be a multiple of 0.5h")
    size.add_argument("--rebalance-minutes", type=float, help="Overrides rebalance-bars; must be a multiple of 30")
    size.add_argument("--symbol-stop", type=float, default=0.05)
    size.add_argument("--symbol-take", type=float, default=0.30)
    size.add_argument("--daily-stop", type=float, default=0.05)
    size.add_argument("--daily-take", type=float, default=0.30)
    size.add_argument("--portfolio-drawdown", type=float, default=0.10)
    size.add_argument("--shadow-min-periods", type=int, default=3)
    size.add_argument("--shadow-positive-periods", type=int, default=2)
    size.add_argument("--shadow-recovery-pnl-pct", type=float, default=0.01)

    guard = ap.add_argument_group("exchange-side guards (optional)")
    guard.add_argument("--exchange-risk-config", type=Path,
                       help="YAML whose 'risk' section enables stop/take/daily halts on actual Binance "
                            "positions (state: <output>/exchange_risk_state.json)")
    guard.add_argument("--manual-guard", action="store_true",
                       help="halt and flatten if positions change outside this runner "
                            "(state: <output>/manual_guard_state.json)")
    guard.add_argument("--reset-manual-halt", action="store_true")
    guard.add_argument("--close-on-exit", action="store_true",
                       help="flatten all positions after SIGINT/SIGTERM (live mode)")
    guard.add_argument("--close-all", action="store_true",
                       help="flatten all positions now and exit (needs the live confirmation flags)")

    live = ap.add_argument_group("orders")
    live.add_argument("--live-orders", action="store_true", help="Send orders in the selected environment")
    live.add_argument("--confirm-testnet-orders", action="store_true")
    live.add_argument("--confirm-mainnet-orders", action="store_true",
                      help="second explicit confirmation required for Mainnet orders")
    live.add_argument("--reset-state", action="store_true")
    return ap


def validate(args: argparse.Namespace) -> None:
    sources = [args.signals, args.realtime_factor_id, args.realtime_factor_jsonl, args.alpha_config, args.ml_model]
    if not args.close_all and sum(x is not None for x in sources) != 1:
        raise SystemExit("choose exactly one signal source: --signals, --realtime-factor-id, "
                         "--realtime-factor-jsonl, --alpha-config or --ml-model")
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
    if args.live_orders or args.close_all:
        required = args.confirm_mainnet_orders if args.network == "mainnet" else args.confirm_testnet_orders
        if not (args.live_orders and required):
            flag = "--confirm-mainnet-orders" if args.network == "mainnet" else "--confirm-testnet-orders"
            raise SystemExit(f"Refusing {args.network} orders: add --live-orders and {flag} explicitly")
    try:
        args.rebalance_bars = resolve_rebalance_bars(
            bars=args.rebalance_bars, hours=args.rebalance_hours, minutes=args.rebalance_minutes,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def make_source(args: argparse.Namespace):
    if args.signals:
        return CsvSource(args.signals)
    if args.realtime_factor_id:
        return LiquiditySource()
    if args.realtime_factor_jsonl:
        return JsonlSource(args.realtime_factor_jsonl, args.realtime_factor_record_id, args.realtime_history_bars)
    if args.alpha_config:
        return AlphaSource(args.alpha_config, proxy=args.proxy)
    return MlSource(args.ml_model, device=args.ml_device, k_long=args.ml_k_long,
                    m_short=args.ml_m_short, history_bars=args.ml_history_bars, proxy=args.proxy)


def runner_config(args: argparse.Namespace) -> RunnerConfig:
    exchange_risk = None
    if args.exchange_risk_config:
        from efficient_cryptos.common.config import load_config
        exchange_risk = dict(load_config(args.exchange_risk_config))
        # Keep the day-open NAV and blocked symbols next to this run's other state,
        # so Testnet and Mainnet (or two experiments) can never share them.
        exchange_risk["risk"] = {**exchange_risk.get("risk", {}),
                                 "state_file": str(args.output / "exchange_risk_state.json")}
    return RunnerConfig(
        output=args.output, symbols=args.symbols.split(","), network=args.network,
        poll_seconds=args.poll_seconds, gross_leverage=args.gross_leverage,
        margin_buffer_pct=args.margin_buffer_pct, long_percent=args.long_percent,
        short_percent=args.short_percent, min_assets=args.min_assets,
        live_orders=args.live_orders, reset_state=args.reset_state,
        engine=EngineConfig(
            initial_equity=args.initial_equity, gross_leverage=args.gross_leverage,
            margin_buffer_pct=args.margin_buffer_pct, rebalance_bars=args.rebalance_bars,
            per_symbol_stop_loss_pct=args.symbol_stop, per_symbol_take_profit_pct=args.symbol_take,
            daily_stop_loss_pct=args.daily_stop, daily_take_profit_pct=args.daily_take,
            portfolio_drawdown_pct=args.portfolio_drawdown,
            shadow_min_periods=args.shadow_min_periods,
            shadow_positive_periods=args.shadow_positive_periods,
            shadow_recovery_pnl_pct=args.shadow_recovery_pnl_pct,
        ),
        exchange_risk=exchange_risk,
        manual_guard_file=(args.output / "manual_guard_state.json") if args.manual_guard else None,
        reset_manual_halt=args.reset_manual_halt, close_on_exit=args.close_on_exit,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate(args)
    adapter = BinanceFuturesAdapter(
        proxy=args.proxy, network=args.network,
        reconnect_command=args.proxy_reconnect_command, market_network=args.market_network,
    )
    if args.close_all:
        try:
            adapter.sync_server_time()
            close_all_positions(adapter, output=args.output if args.output.exists() else None)
        finally:
            adapter.close()
        return 0
    return LiveRunner(runner_config(args), make_source(args), adapter).run()


if __name__ == "__main__":
    raise SystemExit(main())
