# 10 Meme event strategy

An independent short-horizon event strategy that does not interact with the 30m cross-sectional pipeline:

```text
Dynamic Meme universe → 1m/5m momentum, volume shock, trade intensity, taker flow → event score and cross-sectional percentile
  → DPSK event factors → positions, stop loss / take profit, liquidity-decay exits, cooldown
```

| Module (`efficient_cryptos/meme/`) | Role |
|---|---|
| `universe.py` | Builds a dynamic pool of non-major USDT perps from the 24h ticker |
| `features.py` | 1m, 3m, 5m and 10m momentum, volume shock, trade intensity, order flow imbalance (OFI) |
| `detector.py` | Triggers events by cross-sectional percentile |
| `cooldown.py` | Per-symbol cooldown; stop-loss, take-profit, momentum-decay and liquidity-decay exits |
| `execution.py` | Converts events into target notionals; only outputs targets, never signs or places orders |
| `social_attention.py` | Reads exported, timestamped social data |

## Running

```bash
# Universe snapshot (should be generated daily or hourly; backtests use the snapshot visible at the time, to avoid survivorship bias)
python scripts/10_meme/make_universe.py --tickers-json data/universe/binance_futures_24h.json \
  --top-n 200 --min-quote-volume 1000000 --out data_meme/universe_snapshot.json

# Download 1m data to a separate directory (public endpoints only, resumable)
python scripts/10_meme/download_futures_1m.py --universe data_meme/universe_snapshot.json \
  --start-date 2026-01-01 --end-date 2026-10-01 --out-dir data_meme --proxy http://127.0.0.1:1087 --workers 2

# Features and forward return labels (future_ret_1m/5m/10m/30m)
python scripts/10_meme/build_features.py --kline-dir data_meme/klines --universe data_meme/universe_snapshot.json \
  --start-date 2026-01-01 --end-date 2026-10-01 --with-labels --out data/features/meme_event_1m.parquet

# Event-window backtest: MFE/MAE statistics and exit reasons
python scripts/10_meme/backtest_events.py --features data/features/meme_event_1m.parquet --out-dir outputs/meme_bt \
  --hold-minutes 30 --stop-loss 0.05 --take-profit 0.30 --decay-volume-shock -0.20 --fee-bps 4 --slippage-bps 8

# Generate event factors with the meme prompts
python scripts/10_meme/generate_dpsk_factors.py --provider deepseek --model deepseek-chat --total-candidates 100 --dry-run
```

- **Default triggers**: `event_percentile ≥ 95%`, `volume_shock_5m ≥ 0.5`, `|ret_5m| ≥ 1%`. Thresholds should be determined by scanning within the training period.
- **Backtest rules**: only one event position per symbol at a time. `trades.csv` records each trade's MFE, MAE, holding time and exit reason.
- **Input requirements**: at least `open_time` (or `timestamp`) and `open,high,low,close,quote_volume`; `volume,trades,taker_buy_quote` are optional.
- **Data sources**: do not mix spot 1s klines into futures backtests. For higher frequency, prefer aggregating futures aggTrade or bookTicker to 1s/5s, then generate 1m/5m features.

`MemeEventExecutor` currently only produces `EventTarget`. Before connecting it to live trading, you must confirm that event-window results remain positive after fees and slippage.
