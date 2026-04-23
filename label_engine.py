from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

LABEL_POSITIVE = "Positive"
LABEL_NEUTRAL = "Neutral"
LABEL_NEGATIVE = "Negative"


@dataclass(frozen=True)
class LabelConfig:
    horizon_days: int = 5
    positive_threshold: float = 0.015
    negative_threshold: float = -0.015


def create_labels(df: pd.DataFrame, config: LabelConfig | None = None) -> pd.DataFrame:
    """Create leak-free forward labels from close prices.

    Label at date T uses only close(T+5) / close(T) - 1.
    """
    config = config or LabelConfig()
    if "close" not in df.columns:
        raise ValueError("Input DataFrame must include 'close' column.")

    out = df.copy()
    future_close = out["close"].shift(-config.horizon_days)
    fwd_return = (future_close / out["close"]) - 1.0

    out["forward_return_5d"] = fwd_return
    out["label"] = np.select(
        [
            fwd_return > config.positive_threshold,
            fwd_return < config.negative_threshold,
        ],
        [LABEL_POSITIVE, LABEL_NEGATIVE],
        default=LABEL_NEUTRAL,
    )
    out.loc[future_close.isna(), "label"] = np.nan
    return out


def validate_no_leakage(df: pd.DataFrame, feature_cols: Iterable[str] | None = None) -> None:
    """Assert features do not contain future-filled values.

    We enforce that wherever forward label is unavailable (tail horizon), feature
    values must have at least one NaN if provided by rolling/shift operations.
    """
    if "forward_return_5d" not in df.columns:
        raise ValueError("'forward_return_5d' must exist. Run create_labels first.")

    feature_cols = list(feature_cols or [
        c for c in df.columns
        if c not in {"close", "open", "high", "low", "volume",
                     "label", "forward_return_5d", "regime"}
    ])
    unavailable_mask = df["forward_return_5d"].isna()

    for col in feature_cols:
        if col not in df.columns:
            continue
        if unavailable_mask.any() and df.loc[unavailable_mask, col].notna().all():
            raise AssertionError(
                f"Potential leakage in feature '{col}': complete values present where forward label is unavailable."
            )


if __name__ == "__main__":
    demo = pd.DataFrame(
        {
            "close": [100, 101, 102, 101, 103, 104, 105, 106, 104, 107, 109],
        },
        index=pd.date_range("2024-01-01", periods=11, freq="B", tz="UTC"),
    )
    labeled = create_labels(demo)
    print(labeled[["close", "forward_return_5d", "label"]].tail(8))
    validate_no_leakage(labeled, feature_cols=[])
    print("Leakage validation passed.")
