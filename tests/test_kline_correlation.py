import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from efficient_cryptos.data.correlation import resample_close

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "01_data"))
import kline_correlation  # noqa: E402

START_MS = int(pd.Timestamp("2026-03-02", tz="UTC").value // 1_000_000)  # a Monday


def _write(dir_: Path, symbol: str, log_ret: np.ndarray, *, drop: slice | None = None, qv: float = 1e6) -> None:
    n = len(log_ret)
    close = 100 * np.exp(np.cumsum(log_ret))
    frame = pd.DataFrame({
        "open_time": START_MS + np.arange(n, dtype=np.int64) * 60_000,
        "open": close, "high": close, "low": close, "close": close, "volume": 1.0,
        "close_time": START_MS + np.arange(n, dtype=np.int64) * 60_000 + 59_999,
        "quote_volume": qv, "trades": 10, "taker_buy_base": 0.5, "taker_buy_quote": qv / 2,
    })
    if drop is not None:
        frame = frame.drop(index=frame.index[drop])
    (dir_ / "klines").mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dir_ / "klines" / f"{symbol}.parquet", index=False)


@pytest.fixture()
def dataset(tmp_path):
    rng = np.random.default_rng(7)
    n = 14 * 1440
    market = rng.normal(0, 1e-3, n)
    sector = rng.normal(0, 1e-3, n)
    noise = lambda: rng.normal(0, 1e-3, n)  # noqa: E731
    binance, hyper = tmp_path / "data", tmp_path / "data_hl"
    _write(binance, "BTCUSDT", market + 0.3 * noise(), qv=9e6)
    _write(binance, "ETHUSDT", market + 0.6 * noise(), qv=8e6)
    _write(binance, "SOLUSDT", market + sector + noise(), qv=5e6)
    _write(binance, "SUIUSDT", market + sector + noise(), qv=4e6)
    tsla = 0.3 * market + noise()
    _write(binance, "TSLAUSDT", tsla, qv=3e6)
    _write(hyper, "TSLAUSDT", tsla + 0.1 * noise(), qv=1e6)
    _write(binance, "NEWUSDT", market + noise(), drop=slice(0, 10 * 1440), qv=7e6)  # only 4 days
    _write(binance, "GAPUSDT", noise(), drop=slice(500, 520), qv=2e6)               # missing minutes
    return binance, hyper


def test_resample_marks_incomplete_bars(dataset):
    binance, _ = dataset
    frame = pd.read_parquet(binance / "klines" / "GAPUSDT.parquet")
    bars = resample_close(frame, 30)
    # minutes 500..519 fall in the 30m bars starting at minute 480 and 510
    missing = bars.index[bars["close"].isna()]
    expected = pd.to_datetime([START_MS + 480 * 60_000, START_MS + 510 * 60_000], unit="ms", utc=True)
    assert list(missing) == list(expected)
    assert bars["quote_volume"].iloc[0] == pytest.approx(30 * 2e6)


def test_resample_rejects_coarser_source():
    frame = pd.DataFrame({"open_time": START_MS + np.arange(10) * 3_600_000, "close": 1.0, "quote_volume": 1.0})
    with pytest.raises(ValueError):
        resample_close(frame, 30)


def test_cli_outputs_and_pool(dataset, tmp_path, capsys):
    binance, hyper = dataset
    out = tmp_path / "out"
    rc = kline_correlation.main([
        "--data-dir", str(binance), "--data-dir", str(hyper), "--interval", "30m",
        "--min-overlap", "50", "--max-corr", "0.6", "--min-coverage", "0.5",
        "--rolling-days", "7", "--out-dir", str(out), "--no-plots",
    ])
    assert rc == 0
    raw = pd.read_csv(out / "corr_raw.csv", index_col=0)
    resid = pd.read_csv(out / "corr_resid.csv", index_col=0)
    assert set(raw.columns) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "TSLAUSDT",
                                "data_hl/TSLAUSDT", "NEWUSDT", "GAPUSDT"}
    assert np.allclose(np.diag(raw.to_numpy()), 1.0)
    assert raw.loc["BTCUSDT", "ETHUSDT"] > 0.8
    assert raw.loc["TSLAUSDT", "data_hl/TSLAUSDT"] > 0.95
    assert abs(raw.loc["GAPUSDT", "BTCUSDT"]) < 0.15
    # the shared sector shows up after the market move is removed
    assert resid.loc["SOLUSDT", "SUIUSDT"] > 0.3
    assert resid.loc["SOLUSDT", "SUIUSDT"] < raw.loc["SOLUSDT", "SUIUSDT"]

    stats = pd.read_csv(out / "symbol_stats.csv").set_index("symbol")
    assert stats.loc["NEWUSDT", "coverage"] < 0.35
    assert stats.loc["BTCUSDT", "coverage"] == pytest.approx(671 / 672)  # first bar has no prior close
    assert stats.loc["data_hl/TSLAUSDT", "max_corr_peer"] == "TSLAUSDT"
    assert stats.loc["ETHUSDT", "beta_to_ref"] == pytest.approx(1.0, abs=0.1)
    assert stats.loc["BTCUSDT", "median_daily_quote_volume"] == pytest.approx(9e6 * 1440, rel=1e-6)

    decisions = pd.read_csv(out / "pool_decisions.csv").set_index("symbol")
    assert decisions.loc["NEWUSDT", "decision"] == "low_coverage"
    assert decisions.loc["data_hl/TSLAUSDT", "decision"] == "correlated"
    assert decisions.loc["data_hl/TSLAUSDT", "closest_kept"] == "TSLAUSDT"
    pool = json.loads((out / "suggested_pool.json").read_text())["symbols"]
    assert "TSLAUSDT" in pool and "data_hl/TSLAUSDT" not in pool and "BTCUSDT" in pool
    assert not (pd.read_csv(out / "rolling_avg_corr.csv", index_col=0).empty)
    pairs = pd.read_csv(out / "high_corr_pairs.csv")
    assert {"left", "right", "corr", "overlap_bars", "corr_resid"} <= set(pairs.columns)
    assert "suggested pool" in capsys.readouterr().out


def test_session_filter_and_raw_selection(dataset, tmp_path):
    binance, _ = dataset
    out = tmp_path / "out2"
    rc = kline_correlation.main([
        "--data-dir", str(binance), "--symbols", "BTCUSDT,ETHUSDT,SOLUSDT,TSLAUSDT",
        "--weekdays-only", "--utc-hours", "13-20", "--select-on", "raw", "--max-corr", "0.7",
        "--min-overlap", "20", "--out-dir", str(out), "--rolling-days", "0",
    ])
    assert rc == 0
    meta = json.loads((out / "suggested_pool.json").read_text())
    assert "ETHUSDT" not in meta["symbols"]  # raw corr to BTC ~0.78
    stats = pd.read_csv(out / "symbol_stats.csv")
    assert stats["bars"].max() == 10 * 14  # 10 weekdays x 14 half-hour bars in 13:00-20:00
    assert (out / "heatmap_raw.png").exists() and (out / "heatmap_resid.png").exists()
