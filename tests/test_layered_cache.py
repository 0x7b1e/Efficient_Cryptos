import pandas as pd

from efficient_cryptos.common.storage import LocalStore
from efficient_cryptos.features.cache import load_layered_feature_frame


def test_layered_cache_reuses_base_for_multiple_horizons(tmp_path):
    store = LocalStore(tmp_path)
    symbols = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")
    for symbol in symbols:
        rows = []
        for index in range(120):
            open_time = index * 60_000
            price = 100.0 + index * 0.01
            rows.append({
                "open_time": open_time,
                "open": price,
                "high": price + 0.1,
                "low": price - 0.1,
                "close": price + 0.05,
                "volume": 10.0,
                "close_time": open_time + 59_999,
                "quote_volume": 1_000.0,
                "trades": 5,
                "taker_buy_base": 5.0,
                "taker_buy_quote": 500.0,
            })
        store.upsert_klines(symbol, pd.DataFrame(rows))

    first, first_features = load_layered_feature_frame(
        tmp_path, symbols, interval="30m", horizon=1, days=None,
        require_positioning=False,
    )
    second, second_features = load_layered_feature_frame(
        tmp_path, symbols, interval="30m", horizon=4, days=None,
        require_positioning=False,
    )
    assert len(first) == len(second)
    assert first_features == second_features
    assert any((tmp_path / "features").glob("base_30m_dall_*.parquet"))
    assert any((tmp_path / "features").glob("label_30m_h1_dall_*.parquet"))
    assert any((tmp_path / "features").glob("label_30m_h4_dall_*.parquet"))
