# 06 Live trading: dry-run → Testnet → mainnet

Entry point: `python scripts/06_live/run_live.py` (equivalent to `python -m efficient_cryptos.execution.cli`). **The default is dry-run**: it only fetches market data and prints the order plan, places no orders and needs no API key.

## What each cycle does

```text
1 Account   Read Binance equity (NAV) and available margin; NAV calibrates engine equity,
            new positions are sized by min(NAV, available margin)
2 Signal    The signal source returns the cross-section of the latest completed bar
3 Risk      Mark positions with the average price of completed 1m klines (risk is checked on every poll);
            when a new signal timestamp appears, run step (settle PnL, risk checks, rebalance)
4 Targets   Only the LIVE book produces exchange targets; in SHADOW the target is empty, i.e. flat;
            optional exchange-side risk control can override targets
5 Sync      On the first cycle, a new signal or a risk event: set leverage → build a reconciliation
            plan from quantity differences (reduce first, then open) → market orders
6 Audit     Write latest.json, append to audit.jsonl, atomically save engine and runner state
```

Error handling:

| Situation | Behavior |
|---|---|
| Network error (`BinanceNetworkError`) | The engine rolls back to its state at the start of the cycle, the error is logged to `runner_errors.jsonl`, and it retries with backoff (starting at 5s, doubling, up to 120s). The adapter itself retries GET requests 4 times and reconnects the proxy when needed; **orders (POST) are never resent automatically** — the next cycle reconciles directly against the exchange's real positions |
| Insufficient cross-section, a single order rejected, etc. (`ValueError` / `RuntimeError`) | Likewise rolled back with backoff, logged as `cycle_error` |
| 418 / 429 rate limiting | Wait for `Retry-After`; no proxy reconnect |
| Signed requests | Exchange server time is calibrated at startup; timestamp and signature are regenerated on every retry |
| SIGINT / SIGTERM | Exit after finishing the current cycle; with `--close-on-exit`, close positions first, then exit |

## Signal sources (pick one of five)

| Argument | Signal | Selection and sizing |
|---|---|---|
| `--realtime-factor-id liquidity_crypto_templates_10k_v1_liquidity_0041_5` | Built-in liquidity factor, computed from the latest completed 30m klines | Top/Bottom by `--long-percent`/`--short-percent`, equal weight |
| `--realtime-factor-jsonl F.jsonl [--realtime-factor-record-id ID]` | Any generated factor; uses the same sandbox as backtesting | Same as above |
| `--signals F.csv` | The latest timestamp cross-section in the CSV (`prediction`, or `signal,target_notional`) | `prediction` column: percentile, equal weight; `target_notional` column: used directly |
| `--alpha-config config/default.yaml` | Alpha101/191 live scan (the `alpha_live` section of the YAML) | Keeps Alpha's own rules (inverse-volatility sizing by default), scanned only once per bar |
| `--ml-model outputs/crypto_t1/<run>` | A trained ML model | Keeps ML's own `--ml-k-long`/`--ml-m-short` equal-weight sizing |

Every signal source goes through the same risk engine: the engine caps gross exposure at `equity × (1 − margin_buffer) × gross_leverage`, then applies stop losses, the intraday circuit breaker and SHADOW.

**Sizing formula** (percentile signal sources): per-side budget = `sizing_equity × (1 − margin_buffer_pct) × gross_leverage / 2`, split equally among the symbols selected on that side. For example, with equity 10,000, 3x leverage, a 20% buffer and 2 symbols on each side: each long is +6,000 and each short is −6,000. `--gross-leverage` is also used as the exchange leverage setting for each symbol and must be an integer from 1 to 125.

## Rebalancing period and polling

- `--rebalance-hours 4` or `--rebalance-minutes 240`: must be a whole multiple of 30 minutes, otherwise it fails immediately rather than silently rounding; `--rebalance-bars` specifies it directly in bars.
- `--poll-seconds 60`: only decides how often risk and market data are checked; it does not change the rebalancing period. `0` means run a single cycle.
- State is saved in the `--output` directory: `engine_state.json` (positions, equity, SHADOW state) and `runner_state.json` (the last processed signal timestamp). It is restored automatically on restart, and the same signal timestamp is never processed twice. `--reset-state` clears only local state and **does not touch positions on the exchange**.

## Running

```bash
SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,ZECUSDT,MOVRUSDT,NEARUSDT,QNTUSDT,XRPUSDT,DOGEUSDT,SUIUSDT,UNIUSDT,WLDUSDT,1000PEPEUSDT,BNBUSDT,LINKUSDT,ADAUSDT
COMMON="--realtime-factor-id liquidity_crypto_templates_10k_v1_liquidity_0041_5 --symbols $SYMBOLS \
  --poll-seconds 60 --rebalance-hours 4 --gross-leverage 3 --margin-buffer-pct 0.20 \
  --long-percent 10 --short-percent 10 --symbol-stop 0.10 --symbol-take 0.30 \
  --daily-stop 0.05 --daily-take 0.30 --portfolio-drawdown 0.10 --proxy http://127.0.0.1:1087"

# 1) dry-run: mainnet market data, only print the order plan
python -u scripts/06_live/run_live.py $COMMON --network mainnet --output outputs/dry_run

# 2) Testnet orders; compute signals from mainnet market data to avoid distorted Testnet data
export BINANCE_TESTNET_API_KEY=... BINANCE_TESTNET_API_SECRET=...
python -u scripts/06_live/run_live.py $COMMON --network testnet --market-network mainnet \
  --output outputs/testnet_live --manual-guard --exchange-risk-config config/default.yaml \
  --live-orders --confirm-testnet-orders

# 3) Mainnet: start with 1x leverage and small capital, in a separate output directory
export BINANCE_MAINNET_API_KEY=... BINANCE_MAINNET_API_SECRET=...
python -u scripts/06_live/run_live.py $COMMON --gross-leverage 1 --network mainnet \
  --output outputs/mainnet_live --manual-guard --exchange-risk-config config/default.yaml \
  --close-on-exit --live-orders --confirm-mainnet-orders
```

- Placing orders requires all four of: `--live-orders`; the confirmation flag for the current network (`--confirm-testnet-orders` or `--confirm-mainnet-orders`); that network's API key being set; and the two confirmation flags not being mixed (e.g. `--confirm-mainnet-orders` must not appear on the Testnet network).
- `--network` decides where orders are placed; `--market-network` decides where market data comes from and defaults to `--network`. Testnet klines and volumes differ greatly from mainnet, so volume-type factors computed on them are distorted.

## Exchange-side protection (optional, recommended for live trading)

| Argument | Effect |
|---|---|
| `--exchange-risk-config config/default.yaml` | Reads the `risk` section of the YAML and adds another layer of risk control based on the **exchange's real positions and account equity**: per-symbol stop loss/take profit from the exchange entry price, with no new positions that day after a trigger; intraday stop loss, take profit and drawdown circuit breakers on account equity; reset at `daily_reset_utc_offset_hours` (default UTC+8). It is independent of the engine's risk control and covers cases the engine cannot estimate (real slippage, funding, liquidation). State is saved in `<output>/exchange_risk_state.json` |
| `--manual-guard` | Records a position snapshot after every sync. If positions are changed externally between two syncs (manual close in the Binance app, another program, liquidation), the runner closes all positions and stops. Subsequent restarts are refused; after confirming the account is safe, add `--reset-manual-halt` to continue. State is saved in `<output>/manual_guard_state.json` |
| `--close-on-exit` | On SIGINT/SIGTERM, finish the current cycle, then close all positions and exit |
| `--close-all` | Immediately close all positions and exit (also requires `--live-orders` and the confirmation flag) |

The runner treats the whole futures account as its own: any position not in the targets is closed during reconciliation. **Do not share the same account with other strategies.**

## Proxy reconnection

With `--proxy`, when a GET request hits a connection failure or a 5xx error, the reconnect command runs first and then the proxy port is checked for recovery. The default reconnect command is `sudo -n /usr/local/sbin/shadow-reconnect-anycast`; customize it with `--proxy-reconnect-command`.

First-time install (installs the script and the Anycast runtime directory owned by root, so passwordless sudo cannot be abused through files writable by an ordinary user):

```bash
sudo ANYCAST_SRC=$HOME/Anycast bash scripts/06_live/install_anycast_reconnect.sh
sudo visudo -f /etc/sudoers.d/shadow-anycast
#   <username> ALL=(root) NOPASSWD: /usr/local/sbin/shadow-reconnect-anycast
sudo -n /usr/local/sbin/shadow-reconnect-anycast 30     # test; 30 is the node number
```

`sudo -n` never prompts for a password; if sudoers is not configured correctly it fails immediately instead of hanging for input.

## Outputs and monitoring

| File (under `--output`) | Content |
|---|---|
| `latest.json` / `audit.jsonl` | Each cycle's mode, signal timestamp, Binance equity, available margin, sizing equity, engine result, target positions, order plan, order responses and real exchange positions; plus `exchange_risk_notes` when exchange-side risk control is on |
| `runner_errors.jsonl` | Network errors and per-cycle errors |
| `engine_state.json` / `runner_state.json` | State used for restart recovery |
| `exchange_risk_state.json` / `manual_guard_state.json` | State of the exchange-side protections |

Run it in tmux and keep a log:

```bash
tmux new -s live
python -u scripts/06_live/run_live.py ... 2>&1 | tee -a outputs/testnet_live/runner.log
# Ctrl-b d to detach; tmux attach -t live to reattach
```

Stop the process with Ctrl-C or `kill <pid>` (SIGTERM), not `kill -9`: that skips the cleanup and the close-on-exit.

## Failures and recovery

- **`-2019 Margin is insufficient`**: not enough available margin. New positions are already sized by available margin; you can raise `--margin-buffer-pct` to 0.3. A single rejected order is recorded as `order_error` in `submitted` and does not interrupt the cycle.
- **After a manual-change halt**: first log in to Binance and check positions, open orders and position mode; then confirm the new targets with a dry-run; finally restart with `--reset-manual-halt`. Do not delete the state files to bypass the check.
- **Prolonged network outage**: software stop losses, exchange-side risk control and close-on-exit cannot run; you must check the account manually.

## Go-live checklist

- [ ] In dry-run, the long/short symbols, quantities and directions are all correct
- [ ] Each network and each strategy uses its own `--output` directory
- [ ] Testnet uses `--market-network mainnet` and has been observed continuously for long enough
- [ ] The mainnet API key has only futures trading enabled, withdrawals disabled and an IP allow-list set (including the proxy's egress IP)
- [ ] `--manual-guard` and `--exchange-risk-config` are enabled
- [ ] No other process is operating on the same account
- [ ] A manual emergency close plan is ready (`--close-all` or the Binance app)
- [ ] First mainnet run uses 1x leverage and a very small amount of capital

## Known limitations

1. **No exchange-side conditional stop orders.** Stop losses and take profits are software risk controls that only work while the process and network are healthy. Before going live with real funds, consider pre-placing `STOP_MARKET` or similar conditional orders as a backstop.
2. **Cross-sections around the top of the hour may be incomplete.** Klines are requested symbol by symbol, so across an hour boundary only some symbols may have the new bar. The built-in liquidity factor takes the timestamp covering the most symbols, may signal from an incomplete cross-section, and never recomputes the same timestamp. A timestamp going backwards is not treated as a new signal.
3. **Live JSONL factor inputs are not identical to backtesting.** In backtests, derived features (`log_ret_1`, `funding_rate`, etc.) have already been z-scored per cross-section; the live JSONL signal source computes from raw klines without that step, and without funding or positioning data. Factors that use only raw OHLCV / `quote_volume` (such as the built-in liquidity factor) are unaffected; for factors that use derived features, compare live and backtest predictions before going live.
4. **Testnet does not represent real execution quality**: liquidity, slippage and funding all differ from mainnet.

## Monitoring

Each record in `audit.jsonl` carries a read-only `engine_totals` field: the engine's cumulative costs, cumulative funding, shadow book state and risk parameters. The runner never reads it; the field exists only for the dashboard. See [12 Dashboard](12_dashboard.md) for how to use the dashboard.
