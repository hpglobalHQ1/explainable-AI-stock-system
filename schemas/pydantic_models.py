from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class OHLCVPoint(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class StockResponse(BaseModel):
    ticker: str
    points: list[OHLCVPoint]


class AnalysisResponse(BaseModel):
    ticker: str
    sentiment_score: float
    indicators: dict[str, Any]


class InsightResponse(BaseModel):
    ticker: str
    signal: str
    confidence: float = Field(ge=0.0, le=1.0)
    shap_values: dict[str, float]
    narrative: str
    regime: str


class PositionCreateRequest(BaseModel):
    ticker: str
    quantity: float
    buy_price: float
    buy_date: date | None = None
    notes: str = ""


class PositionView(BaseModel):
    id: int
    ticker: str
    quantity: float
    buy_price: float
    buy_date: date
    notes: str


class PortfolioResponse(BaseModel):
    positions: list[PositionView]
    total_value: float
    pnl_rows: list[dict[str, Any]]
    concentration_risk: dict[str, Any]


class RiskResponse(BaseModel):
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    sharpe_ratio: float
    beta_vs_spy: float
    max_drawdown: float
    annualized_volatility: float


class BaselineResponse(BaseModel):
    ticker: str
    scores: dict[str, float]
