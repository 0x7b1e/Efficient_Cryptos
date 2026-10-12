import numpy as np
import pandas as pd
import pytest

from efficient_cryptos.execution.sources.alpha_scanner import _generated_factor


FACTOR_ID = "momentum_crypto_templates_10k_v1_momentum_0065_0005"


def test_live_generated_momentum_matches_research_formula():
    index = pd.RangeIndex(60)
    close = pd.DataFrame({"A": np.linspace(100, 120, 60), "B": np.linspace(50, 45, 60)}, index=index)
    volume = pd.DataFrame({"A": np.linspace(10, 20, 60), "B": np.linspace(25, 15, 60)}, index=index)

    actual = _generated_factor({"close": close, "volume": volume}, FACTOR_ID)
    expected = ((close.pct_change() * volume).rolling(48, min_periods=48).sum()
                / volume.rolling(48, min_periods=48).sum())

    pd.testing.assert_frame_equal(actual, expected)


def test_unapproved_generated_factor_is_rejected():
    frame = pd.DataFrame({"A": [1.0, 2.0]})
    with pytest.raises(ValueError, match="not approved"):
        _generated_factor({"close": frame, "volume": frame}, "unknown")
