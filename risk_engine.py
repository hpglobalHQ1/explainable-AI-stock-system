from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from data_engine import DataEngine


@dataclass
class RiskReport:
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    sharpe_ratio: float
    beta_vs_spy: float
    max_drawdown: float
    annualized_volatility: float


class RiskEngine:
    def __init__(self, data_engine: DataEngine | None = None, risk_free_rate: float = 0.045) -> None:
        self.data_engine = data_engine or DataEngine()
        self.risk_free_rate = risk_free_rate

    @staticmethod
    def _compute_metrics(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float) -> RiskReport:
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        r = aligned.iloc[:, 0]
        b = aligned.iloc[:, 1]

        var95 = float(np.percentile(r, 5))
        var99 = float(np.percentile(r, 1))
        cvar95 = float(r[r <= var95].mean())
        cvar99 = float(r[r <= var99].mean())

        ann_vol = float(r.std() * np.sqrt(252))
        excess = r.mean() * 252 - risk_free_rate
        sharpe = float(excess / ann_vol) if ann_vol != 0 else 0.0

        cov = np.cov(r, b)[0, 1]
        beta = float(cov / np.var(b)) if np.var(b) != 0 else 0.0

        equity = (1 + r).cumprod()
        rolling_max = equity.cummax()
        max_dd = float(((equity - rolling_max) / rolling_max).min())

        return RiskReport(var95, var99, cvar95, cvar99, sharpe, beta, max_dd, ann_vol)

    def ticker_risk(self, ticker: str) -> RiskReport:
        px = self.data_engine.fetch_ohlcv(ticker)["close"]
        spy = self.data_engine.fetch_ohlcv("SPY")["close"]
        return self._compute_metrics(px.pct_change().dropna(), spy.pct_change().dropna(), self.risk_free_rate)

    def portfolio_risk(self, weights: dict[str, float]) -> RiskReport:
        series = []
        for t, w in weights.items():
            returns = self.data_engine.fetch_ohlcv(t)["close"].pct_change().dropna() * float(w)
            series.append(returns.rename(t))
        combined = pd.concat(series, axis=1).dropna().sum(axis=1)
        spy = self.data_engine.fetch_ohlcv("SPY")["close"].pct_change().dropna()
        return self._compute_metrics(combined, spy, self.risk_free_rate)


if __name__ == "__main__":
    re = RiskEngine()
    report = re.ticker_risk("AAPL")
    print(asdict(report))
