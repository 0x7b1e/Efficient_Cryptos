# 12 Dashboard

A read-only Streamlit site that reads existing result files under `outputs/` directly. It has three pages:

- **Factor leaderboard**: reads Rust backtest results; filter, sort, compare training vs test periods, and inspect a single factor's NAV, drawdown, decile returns and cost breakdown.
- **Shadow replay**: reads backtests replayed with the live risk engine, showing LIVE/SHADOW switches, the shadow book, costs and margin usage.
- **Live monitor**: reads the live runner's output directory, showing equity, positions, long/short exposure, SHADOW recovery progress, costs, margin usage, orders and reconnect records.

Backtests and live trading use the same conventions:
- **Costs**: fees + slippage, computed from the engine's `fee_bps + slippage_bps` (plus volume impact);
- **Funding**: positive when paid;
- **Cost share of gross return** = (fees + slippage + funding) / return before costs;
- **SHADOW**: highlighted with an orange background.

The dashboard stores no API keys, does not connect to Binance and cannot place orders; all operations are done from the command line.

## Starting

```bash
pip install -e ".[ui]"          # streamlit + plotly
streamlit run app/streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```

On remote machines such as AutoDL, open it in your local browser through an SSH tunnel (replace the port and host with your instance's SSH details):

```bash
# Run on your own computer, then visit http://localhost:8501
ssh -N -L 8501:127.0.0.1:8501 -p <SSH port> root@<SSH host>
```

Streamlit has no login. Do not expose the dashboard to the public internet with `--server.address 0.0.0.0` or a platform's public port, or anyone will be able to see your positions and equity.

## Factor leaderboard

Data comes from the output of `scripts/03_backtest/sweep_rust_rebalance.sh`:

```text
outputs/sweep_train/
├── 1h/<factor>.metrics.json   .timeseries.csv   .layers.csv
├── 2h/ ...
└── 24h/ ...
```

Sidebar settings:
- **Backtest results directory**: automatically lists every directory under `outputs/` that contains backtest results, most recently updated first.
- **Comparison period directory**: pick backtest results for another time range (e.g. the 2026 test period). This adds test-period metric columns and a training vs test scatter plot, and overlays test-period NAV and decile returns on the detail charts.
- **Factor JSONL**: shows formula, category, direction and economic hypothesis, and enables filtering by category.

Page contents:

| Section | Content |
|---|---|
| Filters | Rebalancing period, Sharpe, return, max drawdown, search, category; "best period per factor only"; "profitable in the test period too" |
| Leaderboard | Click a row and the details below switch to that factor and period; the filtered CSV can be downloaded |
| Cross-period stability | For each factor: number of profitable periods, median Sharpe, worst Sharpe, worst drawdown |
| Training vs test | Scatter of the two periods' Sharpe, share positive in both periods, Spearman correlation of Sharpe ranks |
| Factor details | Metrics per period; NAV and drawdown (with the test period overlaid); mean excess return per decile; cumulative curves of Q1, Q10 and Q10−Q1 |
| Cost breakdown | Return before costs, fees + slippage, funding, net return, cost share of gross return, annualized turnover, and cumulative curves |

"Cost as % of capital" and "cost share of gross return %" in the leaderboard include only fees and slippage (metrics.json has no funding); the cost breakdown in the details includes funding. Factors where costs eat more than half the gross return usually do better with a slower rebalancing period.

The Rust backtest holds at fixed leverage with no risk state machine, so there is no SHADOW. To see how risk control and SHADOW affect results, use Shadow replay below.

On the training vs test view, the **Spearman correlation** is the most direct indicator of overfitting:
- Near 0: training-period ranking has almost no predictive power for the test period; the top of the leaderboard is mostly luck;
- Clearly positive (e.g. above 0.2): the training-period ordering still partly holds in the test period.

## Shadow replay

Data comes from `scripts/05_shadow/run_selected_factor_shadow.py --out-dir DIR`, with the same directory layout as the Rust backtest (`DIR/<period>/<factor>.metrics.json`, `.timeseries.csv`). The dashboard distinguishes the two kinds of results by the `final_state` field in metrics, so both kinds of directories can live under `outputs/` without getting mixed up.

| Section | Content |
|---|---|
| Leaderboard | Return, Sharpe, drawdown, final state, number of SHADOW entries / LIVE recoveries, cost as % of capital, funding, cost share of gross return |
| Details · metrics | Final state, share of time in SHADOW, return before costs, fees + slippage, funding, net return |
| Details · charts | Live book equity and shadow book NAV; cumulative PnL breakdown; long/short notional and margin usage %; SHADOW periods highlighted in orange |
| Details · tables | Risk events (stop loss, intraday stop, entering / leaving SHADOW) and this replay's parameters |

The replay script outputs a few extra display-only columns that do not affect trading logic:
- `timeseries.csv`: `total_cost`, `total_funding` (the engine's exact cumulative values), long/short notionals (live book and shadow book), SHADOW period counts;
- `metrics.json`: `config`, recording this replay's parameters.

Margin usage = gross notional / leverage / equity, consistent with Binance's initial-margin convention. Replays from older versions lack these columns: costs are approximately reconstructed from per-step ratios, and positions and margin usage are not shown; rerun the replay to fill them in.

## Live monitor

Data comes from `run_live.py --output DIR`:

| File | Purpose |
|---|---|
| `latest.json` | The most recent cycle: equity, positions, target positions, orders |
| `audit.jsonl` | One line per cycle. The file keeps growing, so the dashboard reads only the last N lines (adjustable in the sidebar) |
| `engine_state.json` | Risk state: LIVE/SHADOW, high-water mark, day-open equity, blocked symbols |
| `runner_errors.jsonl` | Network and cycle errors, plus retry wait times |
| `exchange_risk_state.json`, `manual_guard_state.json` | State of the optional exchange-side risk control and manual-change detection |

Each audit record has a read-only `engine_totals` field containing the engine's cumulative costs, cumulative funding, shadow book state and risk parameters (including recovery conditions). It is for the dashboard; the runner itself never reads it. Audits written by older runners lack this field: costs are estimated from per-cycle ratios, funding is not shown, and recovery conditions use the sidebar settings.

Page contents:
- **Status bar**: factor, mode (TESTNET/MAINNET, LIVE or DRY_RUN), LIVE/SHADOW state, last update time. A red warning appears when `latest.json` has not been updated for more than the configured number of minutes; set this threshold to 2–3 times the poll interval.
- **Metrics**: account equity, available margin, today's PnL, drawdown from peak, gross leverage, net / gross exposure (near 0 when dollar-neutral), cumulative cost.
- **SHADOW state and recovery conditions**: expanded automatically while in SHADOW. Lists each recovery condition's current value and requirement (number of SHADOW periods, how many were profitable, shadow book cumulative return, whether the account is frozen by the intraday stop), plus the shadow book's simulated positions (no orders are placed).
- **Costs and margin usage**: fees + slippage, funding, cost share of gross return, current and maximum margin usage and gross leverage within the read window, and their curves (SHADOW periods in orange). Margin usage = (equity − available balance) / equity, straight from Binance. Costs are the engine's exact cumulative values from its fees + slippage model, the same convention as backtesting; the fees Binance actually charges will differ with VIP tier, BNB fee discounts, etc.
- **Positions**: the exchange's real positions for signed runs, the engine's simulated book for dry-runs; on the right, a bar chart of long/short notionals.
- **Target vs actual**: the difference between target and actual position per symbol, for spotting deviations caused by rounding, minimum order sizes or rejections.
- **Equity and exposure**: equity curve with SHADOW periods highlighted in orange, alongside the shadow book; below, long, short and net exposure.
- **Risk events / recent orders / errors and reconnects / state files**.

The page auto-refreshes every 30 seconds by default; this can be turned off or changed in the sidebar.

## Code structure

| File | Role |
|---|---|
| `efficient_cryptos/dashboard/data.py` | All file reading and computation, no streamlit dependency, covered by `tests/test_dashboard_data.py` (the live part uses data produced by the real runner against a fake exchange) |
| `app/streamlit_app.py` | Entry point and page navigation |
| `app/views/factors.py`, `shadow.py`, `live.py` | The three pages (do not put them in `app/pages/`; that directory name makes Streamlit bypass page navigation) |
| `app/_shared.py` | Path selection, formatting, colors |

When adding a page, put data processing in `dashboard/data.py` with tests; page files should only handle presentation.
