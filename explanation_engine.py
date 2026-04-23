from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from insight_engine import InsightEngine


@dataclass
class ExplanationEngine:
    insight_engine: InsightEngine
    feature_names: list[str]

    def _class_index(self, signal: str) -> int:
        return {"Negative": 0, "Neutral": 1, "Positive": 2}[signal]

    def explain(self, ticker: str, features: pd.DataFrame, sentiment_score: float) -> dict:
        missing = set(self.feature_names) - set(features.columns)
        extra = set(features.columns) - set(self.feature_names) - {"regime"}
        if missing:
            raise ValueError(f"Inference input missing features: {missing}")
        if extra:
            raise ValueError(f"Inference input has unexpected features: {extra}")
        features = features[self.feature_names]

        pred = self.insight_engine.predict(features)[0]
        idx = self._class_index(pred.signal)

        rf_exp = shap.TreeExplainer(self.insight_engine.rf)
        lgbm_exp = shap.TreeExplainer(self.insight_engine.lgbm)

        rf_vals = rf_exp.shap_values(features)
        lgbm_vals = lgbm_exp.shap_values(features)

        if isinstance(rf_vals, list):
            rf_selected = rf_vals[idx][0]
        elif getattr(rf_vals, "ndim", 0) == 3:
            rf_selected = rf_vals[0, :, idx]
        else:
            rf_selected = rf_vals[0]

        if isinstance(lgbm_vals, list):
            lgbm_selected = lgbm_vals[idx][0]
        elif getattr(lgbm_vals, "ndim", 0) == 3:
            lgbm_selected = lgbm_vals[0, :, idx]
        else:
            lgbm_selected = lgbm_vals[0]

        combined = (np.array(rf_selected) + np.array(lgbm_selected)) / 2.0
        shap_map = {f: float(v) for f, v in zip(self.feature_names, combined)}

        regime = str(features.iloc[0].get("regime", "Sideways"))
        narrative = (
            f"Trend: MA crossover ({shap_map.get('ma_crossover', 0):+.2f}). "
            f"Momentum: RSI impact ({shap_map.get('rsi14', 0):+.2f}). "
            f"Sentiment: {'Positive' if sentiment_score >= 0 else 'Negative'} ({sentiment_score:+.2f}). "
            f"Regime: {regime} market. "
            f"Volatility impact ({shap_map.get('volatility_5d', 0):+.2f}). "
            f"Overall bias: {pred.signal} (confidence: {pred.confidence:.2f})"
        )

        return {
            "ticker": ticker.upper(),
            "signal": pred.signal,
            "confidence": pred.confidence,
            "shap_values": shap_map,
            "narrative": narrative,
        }


if __name__ == "__main__":
    idx = pd.date_range("2023-01-01", periods=280, freq="B", tz="UTC")
    X = pd.DataFrame(
        {
            "ma20": np.random.normal(size=len(idx)),
            "ma50": np.random.normal(size=len(idx)),
            "rsi14": np.random.uniform(20, 80, size=len(idx)),
            "daily_return": np.random.normal(size=len(idx)) / 100,
            "volatility_5d": np.random.uniform(0.01, 0.05, size=len(idx)),
            "momentum_norm": np.random.normal(size=len(idx)),
            "lag_return_2": np.random.normal(size=len(idx)) / 100,
            "volume_zscore": np.random.normal(size=len(idx)),
            "ma_crossover": np.random.randint(0, 2, size=len(idx)),
            "adx": np.random.uniform(10, 40, size=len(idx)),
        },
        index=idx,
    )
    y = np.where(X["momentum_norm"] > 0.4, "Positive", np.where(X["momentum_norm"] < -0.4, "Negative", "Neutral"))
    ie = InsightEngine()
    ie.fit(X, pd.Series(y, index=idx))
    ee = ExplanationEngine(ie, feature_names=list(X.columns))
    latest = X.iloc[[-1]].assign(regime="Bull")
    print(ee.explain("AAPL", latest, sentiment_score=0.21))
