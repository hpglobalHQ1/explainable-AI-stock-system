from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sqlalchemy import Float, String
from sqlalchemy.orm import Mapped, mapped_column

from db.database import Base, get_session
from validator import walk_forward_validate


class SentimentWeight(Base):
    __tablename__ = "sentiment_weights"

    ticker: Mapped[str] = mapped_column(String, primary_key=True)
    tech_weight: Mapped[float] = mapped_column(Float, nullable=False)
    sentiment_weight: Mapped[float] = mapped_column(Float, nullable=False)


@dataclass
class FusionWeights:
    tech_weight: float
    sentiment_weight: float


class SentimentFusionEngine:
    def fit_and_store(self, ticker: str, df: pd.DataFrame, label_col: str = "label") -> FusionWeights:
        req = {"technical_signal_prob", "sentiment_score", label_col}
        if req.difference(df.columns):
            raise ValueError(f"Missing columns: {req.difference(df.columns)}")

        clean = df.dropna(subset=list(req)).copy()
        if len(clean) < 100:
            weights = FusionWeights(0.5, 0.5)
            self._upsert_weights(ticker, weights)
            return weights

        X = clean[["technical_signal_prob", "sentiment_score"]]
        y = (clean[label_col] == "Positive").astype(int)

        # Use same walk-forward splits to avoid leakage in weight-learning.
        def fit_predict(xtr, ytr, xte):
            lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
            lr.fit(xtr.values, ytr.values)
            return lr.predict(xte.values)

        walk_forward_validate(
            clean.assign(binary_label=y),
            feature_cols=["technical_signal_prob", "sentiment_score"],
            label_col="binary_label",
            fit_predict_fn=fit_predict,
        )

        model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        model.fit(X, y)
        coef = np.abs(model.coef_[0])
        denom = coef.sum() if coef.sum() > 0 else 1.0
        weights = FusionWeights(float(coef[0] / denom), float(coef[1] / denom))
        self._upsert_weights(ticker, weights)
        return weights

    def _upsert_weights(self, ticker: str, weights: FusionWeights) -> None:
        with get_session() as s:
            existing = s.get(SentimentWeight, ticker.upper())
            if existing:
                existing.tech_weight = weights.tech_weight
                existing.sentiment_weight = weights.sentiment_weight
            else:
                s.add(
                    SentimentWeight(
                        ticker=ticker.upper(),
                        tech_weight=weights.tech_weight,
                        sentiment_weight=weights.sentiment_weight,
                    )
                )

    def load_weights(self, ticker: str) -> FusionWeights:
        with get_session() as s:
            row = s.get(SentimentWeight, ticker.upper())
            if not row:
                return FusionWeights(0.5, 0.5)
            return FusionWeights(row.tech_weight, row.sentiment_weight)


if __name__ == "__main__":
    from db.database import init_db

    init_db()
    idx = pd.date_range("2023-01-01", periods=320, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    demo = pd.DataFrame(
        {
            "technical_signal_prob": rng.uniform(0, 1, len(idx)),
            "sentiment_score": rng.uniform(-1, 1, len(idx)),
        },
        index=idx,
    )
    demo["label"] = np.where(demo["technical_signal_prob"] + 0.3 * demo["sentiment_score"] > 0.65, "Positive", "Neutral")

    sfe = SentimentFusionEngine()
    w = sfe.fit_and_store("AAPL", demo)
    print("Learned weights:", w)
