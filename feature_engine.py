from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS: list[str] = [
    "ma20",
    "ma50",
    "rsi14",
    "daily_return",
    "volatility_5d",
    "momentum_norm",
    "lag_return_2",
    "volume_zscore",
    "ma_crossover",
    "adx",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = (high.diff()).where(lambda s: (s > 0) & (s > -low.diff()), 0.0)
    minus_dm = (-low.diff()).where(lambda s: (s > 0) & (s > high.diff()), 0.0)
    tr = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return dx.rolling(period).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    req = {"open", "high", "low", "close", "volume"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df.copy()
    ret1 = out["close"].pct_change()
    ma20 = out["close"].rolling(20).mean()
    ma50 = out["close"].rolling(50).mean()

    out["ma20"] = ma20.shift(1)
    out["ma50"] = ma50.shift(1)
    out["rsi14"] = _rsi(out["close"], 14).shift(1)
    out["daily_return"] = ret1.shift(1)
    out["volatility_5d"] = ret1.rolling(5).std().shift(1)
    momentum = out["close"].pct_change(10)
    out["momentum_norm"] = ((momentum - momentum.rolling(60).mean()) / momentum.rolling(60).std()).shift(1)
    out["lag_return_2"] = ret1.shift(2)
    vol_mean = out["volume"].rolling(20).mean()
    vol_std = out["volume"].rolling(20).std()
    out["volume_zscore"] = ((out["volume"] - vol_mean) / vol_std.replace(0, np.nan)).shift(1)
    out["ma_crossover"] = (ma20 > ma50).astype(int).shift(1)
    out["adx"] = _adx(out, 14).shift(1)
    roll60 = out["close"].pct_change(60).shift(1)
    out["regime"] = np.select([roll60 > 0.05, roll60 < -0.05], ["Bull", "Bear"], default="Sideways")

    return out


def assert_no_future_leak(df: pd.DataFrame, feature_cols: list[str], target_col: str = "forward_return_5d") -> None:
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} missing.")
    valid = df[feature_cols + [target_col]].dropna()
    if valid.empty:
        raise ValueError("No non-null rows for leakage assertion.")
    for col in feature_cols:
        corr = abs(valid[col].corr(valid[target_col]))
        if pd.notna(corr) and corr > 0.95:
            raise ValueError(f"Feature {col} suspiciously correlated with future return ({corr:.3f}).")


if __name__ == "__main__":
    idx = pd.date_range("2023-01-01", periods=320, freq="B", tz="UTC")
    prices = pd.Series(100 + np.cumsum(np.random.normal(0, 1, len(idx))), index=idx)
    demo = pd.DataFrame(
        {
            "open": prices * (1 + np.random.normal(0, 0.002, len(idx))),
            "high": prices * (1 + np.random.normal(0.005, 0.002, len(idx))),
            "low": prices * (1 - np.random.normal(0.005, 0.002, len(idx))),
            "close": prices,
            "volume": np.random.randint(1_000_000, 3_000_000, len(idx)),
        },
        index=idx,
    )
    feat = add_features(demo)
    feat["forward_return_5d"] = feat["close"].shift(-5) / feat["close"] - 1
    cols = [
        "ma20",
        "ma50",
        "rsi14",
        "daily_return",
        "volatility_5d",
        "momentum_norm",
        "lag_return_2",
        "volume_zscore",
        "ma_crossover",
        "adx",
    ]
    assert_no_future_leak(feat, cols)
    print(feat[cols + ["regime"]].tail())
