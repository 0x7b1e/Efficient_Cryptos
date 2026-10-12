from efficient_cryptos.execution.cli import resolve_rebalance_bars as _resolve_rebalance_bars
from efficient_cryptos.execution.sources.liquidity_factor import compute_liquidity_0041_5


def test_rebalance_interval_must_match_signal_bar():
    assert _resolve_rebalance_bars(bars=24, hours=4, minutes=None) == 8
    assert _resolve_rebalance_bars(bars=24, hours=None, minutes=90) == 3
    try:
        _resolve_rebalance_bars(bars=24, hours=None, minutes=25)
    except ValueError as exc:
        assert "multiple" in str(exc)
    else:
        raise AssertionError("non-bar-aligned interval should be rejected")


def test_realtime_factor_uses_only_last_completed_window():
    data = {}
    for index, symbol in enumerate(("AUSDT", "BUSDT", "CUSDT")):
        rows = []
        for bar in range(13):
            # Binance kline indices: open time is [0], quote volume is [7].
            quote_volume = 100.0 + bar * (index + 1)
            rows.append([bar * 1_800_000, 1, 1, 1, 1, 1, bar * 1_800_000 + 1_799_999, quote_volume])
        data[symbol] = rows
    snapshot = compute_liquidity_0041_5(data)
    assert snapshot.timestamp.endswith("00:00+00:00")
    assert {row["symbol"] for row in snapshot.rows} == {"AUSDT", "BUSDT", "CUSDT"}
