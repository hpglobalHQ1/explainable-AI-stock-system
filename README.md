# Explainable AI Stock Insight System

A production-style decision-support platform for stock analysis using leak-safe technical features, FinBERT sentiment, ensemble modeling, SHAP explanations, risk analytics, FastAPI endpoints, and Streamlit dashboards.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
pytest -q
```

## Run API

```bash
python -m db.database
python api.py
```

## Run dashboard

```bash
streamlit run dashboard.py
```

## Project modules

- `label_engine.py`: forward-label creation + leakage checks.
- `data_engine.py`: yfinance OHLCV retrieval with cache/retry.
- `feature_engine.py`: technical indicators and leak checks.
- `validator.py`: expanding walk-forward CV.
- `insight_engine.py`: RF+LGBM ensemble and baselines gate.
- `explanation_engine.py`: per-prediction SHAP narrative.
- `sentiment_engine.py`: FinBERT scoring and recency-weighted sentiment.
- `sentiment_fusion.py`: learned sentiment/technical fusion weights.
- `risk_engine.py`: VaR, CVaR, Sharpe, Beta, drawdown, volatility.
- `portfolio_engine.py`: ORM-backed portfolio management.
- `api.py`: FastAPI service.
- `dashboard.py`: Streamlit multi-page UI.
