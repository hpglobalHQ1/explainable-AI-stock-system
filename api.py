from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from data_engine import DataEngine
from explanation_engine import ExplanationEngine
from feature_engine import FEATURE_COLS, add_features
from insight_engine import InsightEngine
from label_engine import create_labels, validate_no_leakage
from portfolio_engine import CapitalLimitError, PortfolioEngine, PositionInput
from risk_engine import RiskEngine
from schemas.pydantic_models import (
    AnalysisResponse,
    BaselineResponse,
    InsightResponse,
    PortfolioResponse,
    PositionCreateRequest,
    PositionView,
    RiskResponse,
    StockResponse,
)
from sentiment_engine import SentimentEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="Explainable AI Stock Insight System")
data_engine = DataEngine()
risk_engine = RiskEngine(data_engine)
portfolio_engine = PortfolioEngine()
sentiment_engine = SentimentEngine()


def _build_training_frame(ticker: str):
    raw = data_engine.fetch_ohlcv(ticker)
    df = create_labels(add_features(raw))
    validate_no_leakage(df)
    return df


@app.get("/stock/{ticker}", response_model=StockResponse)
def get_stock(ticker: str):
    try:
        df = data_engine.fetch_ohlcv(ticker).tail(30)
        points = [
            {
                "timestamp": idx,
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for idx, r in df.iterrows()
        ]
        return StockResponse(ticker=ticker.upper(), points=points)
    except Exception as e:
        logger.exception("/stock failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/{ticker}", response_model=AnalysisResponse)
def get_analysis(ticker: str):
    try:
        df = add_features(data_engine.fetch_ohlcv(ticker))
        latest = df.iloc[-1]
        sent = sentiment_engine.get_headline_sentiment(ticker)
        indicators = latest[["ma20", "ma50", "rsi14", "volatility_5d", "momentum_norm", "adx", "regime"]].to_dict()
        return AnalysisResponse(ticker=ticker.upper(), sentiment_score=sent["aggregate_score"], indicators=indicators)
    except Exception as e:
        logger.exception("/analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/insight/{ticker}", response_model=InsightResponse)
def get_insight(ticker: str):
    try:
        df = _build_training_frame(ticker)
        train = df.dropna(subset=FEATURE_COLS + ["label"])
        model = InsightEngine()
        model.fit(train[FEATURE_COLS], train["label"])
        explainer = ExplanationEngine(model, FEATURE_COLS)
        sent = sentiment_engine.get_headline_sentiment(ticker)
        latest = df.iloc[[-1]][FEATURE_COLS].assign(regime=df.iloc[-1]["regime"])
        out = explainer.explain(ticker, latest, sent["aggregate_score"])
        out["regime"] = df.iloc[-1]["regime"]
        return InsightResponse(**out)
    except Exception as e:
        logger.exception("/insight failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio", response_model=PortfolioResponse)
def get_portfolio():
    try:
        positions = portfolio_engine.list_positions()
        tickers = sorted(set(p.ticker for p in positions))
        current_prices = {t: float(data_engine.fetch_ohlcv(t)["close"].iloc[-1]) for t in tickers}
        pnl = portfolio_engine.get_pnl_report(current_prices)
        views = [PositionView(id=p.id, ticker=p.ticker, quantity=p.quantity, buy_price=p.buy_price, buy_date=p.buy_date, notes=p.notes) for p in positions]
        return PortfolioResponse(
            positions=views,
            total_value=portfolio_engine.get_portfolio_value(current_prices),
            pnl_rows=pnl.to_dict(orient="records"),
            concentration_risk=portfolio_engine.get_concentration_risk(current_prices),
        )
    except Exception as e:
        logger.exception("/portfolio failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio/add", response_model=dict)
def add_position(req: PositionCreateRequest):
    try:
        pid = portfolio_engine.add_position(PositionInput(**req.model_dump()))
        return {"status": "ok", "position_id": pid}
    except CapitalLimitError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("/portfolio/add failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk/{ticker}", response_model=RiskResponse)
def get_ticker_risk(ticker: str):
    try:
        return RiskResponse(**asdict(risk_engine.ticker_risk(ticker)))
    except Exception as e:
        logger.exception("/risk failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk/portfolio", response_model=RiskResponse)
def get_portfolio_risk():
    try:
        positions = portfolio_engine.list_positions()
        if not positions:
            raise HTTPException(status_code=400, detail="Portfolio empty")
        gross = sum(p.quantity * p.buy_price for p in positions)
        weights = {p.ticker: (p.quantity * p.buy_price) / gross for p in positions}
        return RiskResponse(**asdict(risk_engine.portfolio_risk(weights)))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("/risk/portfolio failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/baselines/{ticker}", response_model=BaselineResponse)
def get_baselines(ticker: str):
    try:
        df = _build_training_frame(ticker)
        model = InsightEngine()
        folds, fitted_model = model.train_with_walk_forward(df, FEATURE_COLS, "label")
        clean = df.dropna(subset=FEATURE_COLS + ["label", "momentum_norm"]).copy()
        y = clean["label"]
        scores = {
            "ensemble": float(sum(f.accuracy for f in folds) / len(folds)) if folds else 0.0,
            "always_neutral": float((y == "Neutral").mean()),
            "always_positive": float((y == "Positive").mean()),
            "momentum_follow": float((np.where(clean["momentum_norm"] > 0, "Positive", "Negative") == y).mean()),
        }
        if any(scores["ensemble"] <= scores[k] for k in ["always_neutral", "always_positive", "momentum_follow"]):
            raise ValueError(f"Ensemble did not beat baselines in walk-forward evaluation: {scores}")
        return BaselineResponse(ticker=ticker.upper(), scores=scores)
    except Exception as e:
        logger.exception("/baselines failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    from db.database import init_db

    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
