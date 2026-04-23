import numpy as np
import pandas as pd
import pytest

from feature_engine import assert_no_future_leak


def test_assert_no_future_leak_flags_perfect_future_feature():
    idx = pd.date_range("2024-01-01", periods=120, freq="B", tz="UTC")
    fwd = pd.Series(np.random.normal(size=len(idx)), index=idx)
    df = pd.DataFrame({"forward_return_5d": fwd, "safe_feature": np.random.normal(size=len(idx)), "leaky_feature": fwd})

    with pytest.raises(ValueError):
        assert_no_future_leak(df, ["safe_feature", "leaky_feature"])


def test_feature_cols_constant_matches_add_features_output():
    import numpy as np
    import pandas as pd
    from feature_engine import FEATURE_COLS, add_features

    idx = pd.date_range("2023-01-01", periods=100, freq="B", tz="UTC")
    prices = pd.Series(
        100 + np.cumsum(np.random.normal(0, 1, 100)), index=idx
    )
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.randint(1_000_000, 3_000_000, 100),
        },
        index=idx,
    )
    out = add_features(df)
    for col in FEATURE_COLS:
        assert col in out.columns, (
            f"FEATURE_COLS contains '{col}' but add_features does not produce it"
        )
