import pytest
import pandas as pd

from efficient_cryptos.backtest.portfolio import decile_profile, factor_summary, lifecycle_pnl_costed
from efficient_cryptos.common.costs import CostModel
from efficient_cryptos.factors.sandbox import evaluate_factor_record, validate_factor_code


def _frame() -> pd.DataFrame:
    rows = []
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT"]
    for bar in range(8):
        ts = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(minutes=30 * bar)
        for rank, symbol in enumerate(symbols):
            rows.append({
                "timestamp": ts,
                "symbol": symbol,
                "close": 100 + bar + rank,
                "open": 99 + bar + rank,
                "high": 101 + bar + rank,
                "low": 98 + bar + rank,
                "volume": 1000.0,
                "quote_volume": 1_000_000.0,
                "funding_rate": 0.0001,
                "funding_rate_raw": 0.00001,
                "log_ret_1": 0.001 * (rank + 1),
                "raw_target": 0.001 * (rank + 1),
                "target": 0.001 * (rank + 1),
            })
    return pd.DataFrame(rows)


def test_generated_factor_is_executed_and_aligned():
    frame = _frame()
    record = {
        "factor_id": "momentum_0001",
        "direction": "positive",
        "code": (
            "def compute_factor(df):\n"
            "    return df['close'].groupby(df['symbol']).transform("
            "lambda x: x.rolling(2).mean())\n"
        ),
    }
    result = evaluate_factor_record(frame, record)
    assert list(result.columns).count("prediction") == 1
    assert len(result) == len(frame)
    assert result["prediction"].notna().sum() > 0
    assert result["funding_rate"].eq(0.00001).all()


def test_legacy_dpsk_aliases_are_supported():
    frame = _frame()
    record = {
        "factor_id": "legacy_alias",
        "direction": "positive",
        "code": (
            "def compute_factor(df):\n"
            "    return df['closePrice'].groupby(df['ticker']).transform("
            "lambda x: x.rolling(2).mean())\n"
        ),
    }
    result = evaluate_factor_record(frame, record)
    assert result["prediction"].notna().sum() > 0


def test_negative_lookahead_is_rejected():
    code = "def compute_factor(df):\n    return df['close'].shift(-1)\n"
    assert "future-looking" in validate_factor_code(code, {"close"})


def test_future_labels_are_not_exposed_to_generated_factors():
    frame = _frame()
    record = {
        "factor_id": "future_label_probe",
        "direction": "positive",
        "code": "def compute_factor(df):\n    return df['target']\n",
    }
    try:
        evaluate_factor_record(frame, record)
    except ValueError as exc:
        assert "unsupported column: target" in str(exc)
    else:
        raise AssertionError("future target label was exposed to factor code")


def test_live_cpu_and_positioning_feature_detection():
    pytest.importorskip("torch")
    from efficient_cryptos.execution.sources.ml_model import _needs_positioning, choose_device
    assert choose_device("cpu").type == "cpu"
    assert _needs_positioning(["top_account_long_short_ratio_z_12"])
    assert not _needs_positioning(["log_ret_12", "funding_rate"])


def test_costed_lifecycle_charges_turnover():
    frame = _frame()
    frame["prediction"] = frame.groupby("timestamp")["target"].transform(
        lambda values: values if values.name.minute == 0 else -values
    )
    result = lifecycle_pnl_costed(
        frame, interval_bars=1, hold_bars=1, k_long=1, m_short=1,
        gross_leverage=1.0, include_costs=True, include_funding=False,
        cost_model=CostModel(taker_fee_bps=10, slippage_base_bps=0,
                             slippage_impact_coef=0, max_slippage_bps=0),
    )
    assert result["cost"].sum() > 0
    assert result["turnover"].sum() > 0


def test_no_cost_summary_does_not_charge_funding():
    frame = _frame()
    frame["prediction"] = frame["target"]
    summary = factor_summary(
        frame, interval_bars=1, hold_bars=1, k_long=1, m_short=1,
        long_percent=None, short_percent=None, gross_leverage=1.0,
        initial_cash=10000.0, cost_model=CostModel(), include_costs=False,
        include_funding=False,
    )
    assert summary["total_cost"] == 0.0


def test_decile_profile_has_explicit_top_and_bottom_for_small_universe():
    frame = _frame()
    frame["prediction"] = frame["target"]
    profile = decile_profile(frame)
    assert 1 in set(profile["decile"])
    assert 10 in set(profile["decile"])
