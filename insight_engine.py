from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from feature_engine import FEATURE_COLS
from validator import walk_forward_validate

logger = logging.getLogger(__name__)


class ModelUnderperformingError(RuntimeError):
    pass


LABELS = ["Negative", "Neutral", "Positive"]


@dataclass
class EnsembleOutput:
    signal: str
    confidence: float
    probabilities: dict[str, float]


class InsightEngine:
    def __init__(self, model_dir: str | Path = "models") -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.lgbm = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.rf.fit(X, y)
        self.lgbm.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rf_p = self.rf.predict_proba(X)
        lgbm_p = self.lgbm.predict_proba(X)
        return (rf_p + lgbm_p) / 2.0

    def predict(self, X: pd.DataFrame) -> list[EnsembleOutput]:
        probs = self.predict_proba(X)
        outputs = []
        for row in probs:
            idx = int(np.argmax(row))
            outputs.append(
                EnsembleOutput(
                    signal=LABELS[idx],
                    confidence=float(row[idx]),
                    probabilities={LABELS[i]: float(row[i]) for i in range(len(LABELS))},
                )
            )
        return outputs

    def save_versioned(self) -> tuple[Path, Path]:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        rf_path = self.model_dir / f"v{ts}_rf.pkl"
        lgbm_path = self.model_dir / f"v{ts}_lgbm.pkl"
        joblib.dump(self.rf, rf_path)
        joblib.dump(self.lgbm, lgbm_path)
        return rf_path, lgbm_path

    def train_with_walk_forward(
        self, df: pd.DataFrame, feature_cols: list[str], label_col: str
    ) -> tuple[list, "InsightEngine"]:
        logger.info(
            "Label distribution: %s",
            df[label_col].value_counts(normalize=True).round(3).to_dict()
        )

        def fit_predict_fn(xtr: pd.DataFrame, ytr: pd.Series, xte: pd.DataFrame) -> np.ndarray:
            rf_fold = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            lgbm_fold = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            )
            rf_fold.fit(xtr, ytr)
            lgbm_fold.fit(xtr, ytr)
            probs = (rf_fold.predict_proba(xte) + lgbm_fold.predict_proba(xte)) / 2.0
            preds = np.array([LABELS[int(np.argmax(row))] for row in probs])
            return preds

        report = walk_forward_validate(
            df=df,
            feature_cols=feature_cols,
            label_col=label_col,
            fit_predict_fn=fit_predict_fn,
        )
        clean = df.dropna(subset=feature_cols + [label_col])
        self.fit(clean[feature_cols], clean[label_col])
        return report["folds"], self


def compare_to_baselines(df: pd.DataFrame, label_col: str = "label", pred_col: str = "ensemble_pred") -> dict:
    required = {label_col, pred_col, "momentum_norm"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for baseline comparison: {missing}")

    data = df.dropna(subset=[label_col, pred_col, "momentum_norm"]).copy()
    y = data[label_col]
    ensemble_acc = accuracy_score(y, data[pred_col])

    always_neutral = np.array(["Neutral"] * len(data))
    always_positive = np.array(["Positive"] * len(data))
    momentum_follow = np.where(data["momentum_norm"] > 0, "Positive", "Negative")

    scores = {
        "ensemble": ensemble_acc,
        "always_neutral": accuracy_score(y, always_neutral),
        "always_positive": accuracy_score(y, always_positive),
        "momentum_follow": accuracy_score(y, momentum_follow),
    }

    if any(scores["ensemble"] <= scores[k] for k in ["always_neutral", "always_positive", "momentum_follow"]):
        raise ModelUnderperformingError(f"Ensemble did not beat all baselines: {scores}")

    return scores


if __name__ == "__main__":
    np.random.seed(7)
    idx = pd.date_range("2023-01-01", periods=350, freq="B", tz="UTC")
    X = pd.DataFrame(
        {
            "ma20": np.random.normal(size=len(idx)),
            "ma50": np.random.normal(size=len(idx)),
            "rsi14": np.random.uniform(0, 100, size=len(idx)),
            "momentum_norm": np.random.normal(size=len(idx)),
        },
        index=idx,
    )
    y = np.where(X["momentum_norm"] > 0.5, "Positive", np.where(X["momentum_norm"] < -0.5, "Negative", "Neutral"))

    eng = InsightEngine()
    eng.fit(X, pd.Series(y, index=idx))
    preds = [o.signal for o in eng.predict(X)]
    report_df = X.copy()
    report_df["label"] = y
    report_df["ensemble_pred"] = preds
    print(compare_to_baselines(report_df))
    print("Saved models:", eng.save_versioned())
