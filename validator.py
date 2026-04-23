from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    accuracy: float
    confusion: np.ndarray


def walk_forward_validate(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    fit_predict_fn: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray],
    min_train_window: int = 252,
    step_size: int = 21,
) -> dict:
    data = df.dropna(subset=feature_cols + [label_col]).copy()
    if len(data) < min_train_window + step_size:
        raise ValueError("Not enough samples for walk-forward validation.")

    folds: list[FoldResult] = []
    y_true_all: list = []
    y_pred_all: list = []

    fold_idx = 1
    start = min_train_window
    while start + step_size <= len(data):
        train = data.iloc[:start]
        test = data.iloc[start : start + step_size]

        preds = fit_predict_fn(train[feature_cols], train[label_col], test[feature_cols])
        acc = accuracy_score(test[label_col], preds)
        conf = confusion_matrix(test[label_col], preds, labels=["Negative", "Neutral", "Positive"])
        if acc > 0.8:
            logger.warning("Fold %s has suspiciously high accuracy %.3f (possible leakage).", fold_idx, acc)

        folds.append(
            FoldResult(
                fold=fold_idx,
                train_start=train.index.min(),
                train_end=train.index.max(),
                test_start=test.index.min(),
                test_end=test.index.max(),
                accuracy=acc,
                confusion=conf,
            )
        )

        y_true_all.extend(test[label_col].tolist())
        y_pred_all.extend(preds.tolist())
        start += step_size
        fold_idx += 1

    return {
        "folds": folds,
        "avg_accuracy": float(np.mean([f.accuracy for f in folds])) if folds else float("nan"),
        "overall_confusion_matrix": confusion_matrix(y_true_all, y_pred_all, labels=["Negative", "Neutral", "Positive"]),
    }


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier

    idx = pd.date_range("2023-01-01", periods=420, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"f1": rng.normal(size=len(idx)), "f2": rng.normal(size=len(idx))}, index=idx)
    y = np.where(X["f1"] > 0.5, "Positive", np.where(X["f1"] < -0.5, "Negative", "Neutral"))
    demo = X.copy()
    demo["label"] = y

    def fit_predict(xtr, ytr, xte):
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(xtr, ytr)
        return model.predict(xte)

    result = walk_forward_validate(demo, ["f1", "f2"], "label", fit_predict)
    print("Average accuracy:", result["avg_accuracy"])
    print("Folds:", len(result["folds"]))
