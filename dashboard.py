from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_engine import DataEngine
from explanation_engine import ExplanationEngine
from feature_engine import FEATURE_COLS, add_features
from insight_engine import InsightEngine, compare_to_baselines
from label_engine import create_labels
from portfolio_engine import PortfolioEngine
from risk_engine import RiskEngine
from sentiment_engine import SentimentEngine

st.set_page_config(page_title="Explainable AI Stock Insight", layout="wide")

data_engine = DataEngine()
sentiment_engine = SentimentEngine()
portfolio_engine = PortfolioEngine()
risk_engine = RiskEngine(data_engine)

PAGES = ["Stock Overview", "AI Insight", "Sentiment", "Portfolio", "Risk", "Model Health"]
page = st.sidebar.radio("Page", PAGES)
ticker = st.sidebar.text_input("Ticker", "AAPL").upper()

raw = data_engine.fetch_ohlcv(ticker)
feat = add_features(raw)


@st.cache_resource(show_spinner=False)
def _get_trained_model(cache_ticker: str) -> InsightEngine:
    cache_raw = data_engine.fetch_ohlcv(cache_ticker)
    cache_feat = add_features(cache_raw)
    cache_labeled = create_labels(cache_feat)
    cache_train = cache_labeled.dropna(subset=FEATURE_COLS + ["label"])
    cache_model = InsightEngine()
    cache_model.fit(cache_train[FEATURE_COLS], cache_train["label"])
    return cache_model

if page == "Stock Overview":
    st.title("Stock Overview")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw.index, y=raw["close"], name="Close"))
    fig.add_trace(go.Scatter(x=feat.index, y=feat["ma20"], name="MA20"))
    fig.add_trace(go.Scatter(x=feat.index, y=feat["ma50"], name="MA50"))
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.bar(raw.tail(60), y="volume", title="Volume"), use_container_width=True)

elif page == "AI Insight":
    st.title("AI Insight")
    labeled = create_labels(feat)
    train = labeled.dropna(subset=FEATURE_COLS + ["label"]) 
    model = _get_trained_model(ticker)
    sent = sentiment_engine.get_headline_sentiment(ticker)
    exp = ExplanationEngine(model, FEATURE_COLS)
    out = exp.explain(ticker, feat.iloc[[-1]][FEATURE_COLS].assign(regime=feat.iloc[-1]["regime"]), sent["aggregate_score"])
    st.metric("Signal", out["signal"])
    st.progress(float(out["confidence"]))
    st.write(out["narrative"])
    st.write(f"Regime: **{feat.iloc[-1]['regime']}**")

elif page == "Sentiment":
    st.title("Sentiment")
    sent = sentiment_engine.get_headline_sentiment(ticker)
    st.metric("Aggregate sentiment", f"{sent['aggregate_score']:.3f}")
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=sent["aggregate_score"], gauge={"axis": {"range": [-1, 1]}}))
    st.plotly_chart(gauge, use_container_width=True)
    st.dataframe(pd.DataFrame(sent["headlines"]))

elif page == "Portfolio":
    st.title("Portfolio")
    positions = portfolio_engine.list_positions()
    if not positions:
        st.warning("No positions found.")
    else:
        tickers = sorted(set(p.ticker for p in positions))
        prices = {t: float(data_engine.fetch_ohlcv(t)["close"].iloc[-1]) for t in tickers}
        pnl = portfolio_engine.get_pnl_report(prices)
        st.dataframe(pnl)
        st.metric("Total Value", f"${portfolio_engine.get_portfolio_value(prices):,.2f}")

        price_frame = pd.concat({t: data_engine.fetch_ohlcv(t)["close"] for t in tickers}, axis=1).dropna()
        corr = portfolio_engine.get_correlation_matrix(price_frame)
        st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlation Heatmap"), use_container_width=True)

elif page == "Risk":
    st.title("Risk")
    r = risk_engine.ticker_risk(ticker)
    cols = st.columns(4)
    cols[0].metric("VaR 95", f"{r.var_95:.4f}")
    cols[1].metric("CVaR 95", f"{r.cvar_95:.4f}")
    cols[2].metric("Sharpe", f"{r.sharpe_ratio:.2f}")
    cols[3].metric("Beta", f"{r.beta_vs_spy:.2f}")
    cols2 = st.columns(2)
    cols2[0].metric("Max Drawdown", f"{r.max_drawdown:.2%}")
    cols2[1].metric("Annualized Vol", f"{r.annualized_volatility:.2%}")

elif page == "Model Health":
    st.title("Model Health")
    labeled = create_labels(feat)
    model_wf = InsightEngine()
    folds, _ = model_wf.train_with_walk_forward(labeled, FEATURE_COLS, "label")
    fold_df = pd.DataFrame([{"fold": f.fold, "accuracy": f.accuracy} 
                            for f in folds]).set_index("fold")
    st.line_chart(fold_df)
    st.write(f"Average walk-forward accuracy: "
             f"{sum(f.accuracy for f in folds)/len(folds):.3f}")

if __name__ == "__main__":
    print("Run with: streamlit run dashboard.py")
