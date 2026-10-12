"""Regression tests for backtest PnL accounting, funding timing and look-ahead."""
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from efficient_cryptos.backtest.funding import funding_event_rates
from efficient_cryptos.backtest.portfolio import lifecycle_pnl_costed
from efficient_cryptos.factors.sandbox import evaluate_factor_record, validate_factor_code

ROOT = Path(__file__).resolve().parents[1]
RUST_BIN = ROOT / "rust_backtest" / "target" / "release" / "crypto-factor-backtest"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _ms(values) -> list[int]:
    return list((pd.DatetimeIndex(values) - EPOCH) // pd.Timedelta(milliseconds=1))


def _market(bars: int = 6, symbols=("AAAUSDT", "BBBUSDT", "CCCUSDT")) -> pd.DataFrame:
    rows = []
    for bar in range(bars):
        ts = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(minutes=30 * bar)
        for i, symbol in enumerate(symbols):
            rows.append({
                "timestamp": ts, "symbol": symbol, "open": 100.0 + i, "high": 101.0 + i,
                "low": 99.0 + i, "close": 100.0 + i + bar, "volume": 10.0,
                "quote_volume": 1_000.0 * (i + 1), "log_ret_1": 0.0,
                "funding_rate": 0.0, "funding_rate_raw": 0.0,
                "raw_target": 0.0, "target": 0.0,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- funding
def test_funding_events_are_paid_by_the_row_holding_through_them(tmp_path):
    (tmp_path / "funding").mkdir()
    events = pd.to_datetime(["2025-01-01T08:00:00Z", "2025-01-01T12:00:00Z", "2025-01-01T08:10:00Z"])
    pd.DataFrame({"funding_time": _ms(events), "symbol": "AAAUSDT",
                  "funding_rate": [0.001, 0.002, 0.0005]}).to_parquet(tmp_path / "funding" / "AAAUSDT.parquet")
    times = pd.date_range("2025-01-01T06:00:00Z", "2025-01-01T13:00:00Z", freq="30min")
    frame = pd.DataFrame({"timestamp": times, "symbol": "AAAUSDT"})
    rates = funding_event_rates(frame, tmp_path, bar_minutes=30)
    paid = {t.strftime("%H:%M"): r for t, r in zip(frame.timestamp, rates) if r}
    # Row t is held during (t+30m, t+60m]: 08:00 -> row 07:00, 08:10 -> row 07:30,
    # 12:00 (a 4h settlement the old 00/08/16 rule ignored) -> row 11:00.
    assert paid == {"07:00": 0.001, "07:30": 0.0005, "11:00": 0.002}


def test_python_lifecycle_uses_simple_returns_and_actual_settlements():
    ts = pd.Timestamp("2025-01-01T03:00:00Z")   # not a 00/08/16 bar
    frame = pd.DataFrame({
        "timestamp": [ts, ts], "symbol": ["L", "S"], "prediction": [1.0, -1.0],
        "raw_target": [0.10, 0.10], "target": [0.0, 0.0], "quote_volume": [1e12, 1e12],
        "funding_rate": [0.0, 0.0], "funding_event_rate": [0.01, 0.0],
    })
    out = lifecycle_pnl_costed(frame, long_percent=50, short_percent=50, gross_leverage=2.0,
                               include_costs=False, include_funding=True)
    # Equal simple returns on both legs net to zero; only the long pays funding.
    assert math.isclose(out.gross_return.iloc[0], 0.0, abs_tol=1e-15)
    assert math.isclose(out.funding_cost.iloc[0], 0.01)


# ------------------------------------------------------------------ Rust PnL
def _write_zero_drift_csv(path: Path, *, with_raw: bool, seed: int = 3, bars: int = 4000, n: int = 8):
    rng = np.random.default_rng(seed)
    sig = np.linspace(0.004, 0.012, n)
    raw = rng.normal(-sig**2 / 2, sig, size=(bars, n))
    target = raw - raw.mean(axis=1, keepdims=True)
    ts = pd.date_range("2025-01-01", periods=bars, freq="30min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S+00:00")
    pred = -sig[None, :] + 1e-9 * rng.normal(size=(bars, n))   # always short the most volatile coins
    frame = pd.DataFrame({
        "timestamp": np.repeat(ts, n), "symbol": np.tile([f"C{i}USDT" for i in range(n)], bars),
        "prediction": pred.ravel(), "target": target.ravel(), "quote_volume": 1e15, "funding_rate": 0.0,
    })
    if with_raw:
        frame["raw_target"] = raw.ravel()
    frame.to_csv(path, index=False)
    return raw, pred


@pytest.mark.skipif(not RUST_BIN.exists(), reason="build rust_backtest in release mode first")
def test_rust_portfolio_return_is_exact_simple_return(tmp_path):
    raw, pred = _write_zero_drift_csv(tmp_path / "f.csv", with_raw=True)
    subprocess.run([str(RUST_BIN), "--input", str(tmp_path / "f.csv"), "--output", str(tmp_path / "m.json"),
                    "--timeseries-output", str(tmp_path / "ts.csv"), "--long-percent", "25",
                    "--short-percent", "25", "--signal-interval-bars", "8", "--signal-hold-bars", "8",
                    "--gross-leverage", "3"], check=True, capture_output=True)
    gross = pd.read_csv(tmp_path / "ts.csv").gross_return.to_numpy()
    expected = np.zeros(len(raw))
    for t in range(len(raw)):
        if t % 8 == 0:
            order = np.argsort(pred[t]); w = np.zeros(raw.shape[1])
            w[order[-2:]] = 3 / 2 / 2; w[order[:2]] = -3 / 2 / 2
        expected[t] = (w * np.expm1(raw[t])).sum()
    assert np.abs(gross - expected).max() < 1e-12
    metrics = json.loads((tmp_path / "m.json").read_text())
    assert metrics["portfolio_return_basis"] == "sum_w_simple_raw_return"


@pytest.mark.skipif(not RUST_BIN.exists(), reason="build rust_backtest in release mode first")
def test_rust_no_longer_rewards_shorting_volatile_coins(tmp_path):
    """The old exp(sum w*log r) formula turned this zero-alpha book into +300%/yr at 3x."""
    _write_zero_drift_csv(tmp_path / "f.csv", with_raw=True, bars=17520)
    subprocess.run([str(RUST_BIN), "--input", str(tmp_path / "f.csv"), "--output", str(tmp_path / "m.json"),
                    "--timeseries-output", str(tmp_path / "ts.csv"), "--long-percent", "25",
                    "--short-percent", "25", "--signal-interval-bars", "8", "--signal-hold-bars", "8",
                    "--gross-leverage", "3"], check=True, capture_output=True)
    gross = pd.read_csv(tmp_path / "ts.csv").gross_return.to_numpy()
    # Expected value is exactly 0; allow 4 standard errors.
    assert abs(gross.mean()) < 4 * gross.std() / math.sqrt(len(gross))


@pytest.mark.skipif(not RUST_BIN.exists(), reason="build rust_backtest in release mode first")
def test_rust_charges_funding_from_event_column(tmp_path):
    frame = pd.DataFrame({
        "timestamp": ["2025-01-01T03:00:00+00:00"] * 2, "symbol": ["L", "S"],
        "prediction": [1.0, -1.0], "target": [0.0, 0.0], "raw_target": [0.0, 0.0],
        "quote_volume": [1e15, 1e15], "funding_rate": [0.0, 0.0], "funding_event_rate": [0.01, 0.0],
    })
    frame.to_csv(tmp_path / "f.csv", index=False)
    subprocess.run([str(RUST_BIN), "--input", str(tmp_path / "f.csv"), "--output", str(tmp_path / "m.json"),
                    "--long-percent", "50", "--short-percent", "50", "--gross-leverage", "2",
                    "--include-funding"], check=True, capture_output=True)
    metrics = json.loads((tmp_path / "m.json").read_text())
    assert metrics["funding_basis"] == "actual_settlements"
    assert math.isclose(metrics["final_equity"], 10_000 * (1 - 0.01))


# ---------------------------------------------------------------- look-ahead
F = "def compute_factor(df):\n    return {}"


@pytest.mark.parametrize("body", [
    'df["quote_volume"] / df.groupby("symbol")["quote_volume"].transform("mean")',
    'df.groupby("symbol")["close"].transform("zscore")',
    'df.groupby(["ticker", "dataDate"])["high"].transform("max")',
    'df.groupby("symbol")["close"].transform(lambda s: (s - s.mean()) / s.std())',
    'df["close"] - df["close"].mean()',
    'df["close"].rank(pct=True)',
    'df.groupby("symbol")["close"].rank()',
    'df["close"].iloc[::-1].cummax().iloc[::-1]',
    'df["close"].fillna(method="bfill")',
    'df["close"].interpolate()',
])
def test_validator_rejects_look_ahead(body):
    assert validate_factor_code(F.format(body)) is not None


@pytest.mark.parametrize("body", [
    'df.groupby("symbol")["close"].transform(lambda s: s.rolling(12).mean())',
    'df["close"] - df.groupby("timestamp")["close"].transform("mean")',
    'df.groupby("timestamp")["close"].rank(pct=True)',
    'df["close"].rolling(5).max()',
    'df[["open", "close"]].mean(axis=1)',
    'df.groupby("symbol")["close"].transform(lambda s: s.expanding().mean())',
    'rank(df["close"]) + zscore(df["quote_volume"])',
])
def test_validator_accepts_causal_code(body):
    assert validate_factor_code(F.format(body)) is None


def _predictions(frame: pd.DataFrame, body: str) -> pd.Series:
    return evaluate_factor_record(frame, {"factor_id": "t", "code": F.format(body)})["prediction"]


@pytest.mark.parametrize("body", ['df["share"]', 'rank(df["quote_volume"])', 'zscore(df["quote_volume"])'])
def test_aliases_and_helpers_do_not_see_the_future(body):
    frame = _market()
    before = _predictions(frame, body)
    changed = frame.copy()
    future = changed["timestamp"] >= changed["timestamp"].max()
    changed.loc[future & (changed.symbol == "AAAUSDT"), "quote_volume"] = 1e9
    after = _predictions(changed, body)
    past = ~future.to_numpy()
    np.testing.assert_allclose(before.to_numpy()[past], after.to_numpy()[past])


def test_rank_helper_is_cross_sectional():
    preds = _predictions(_market(), 'rank(df["quote_volume"])')
    frame = _market().assign(prediction=preds.to_numpy())
    # Quote volume rises with the symbol index, identically at every bar.
    assert (frame.groupby("symbol")["prediction"].nunique() == 1).all()
    assert list(frame[frame.timestamp == frame.timestamp.min()].prediction.round(6)) == [round(1 / 3, 6), round(2 / 3, 6), 1.0]


def test_factor_code_cannot_read_the_settlement_column():
    frame = _market().assign(funding_event_rate=0.001)
    with pytest.raises(ValueError, match="unsupported column"):
        _predictions(frame, 'df["funding_event_rate"]')
