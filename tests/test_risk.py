import pandas as pd

from risk_engine import RiskEngine


def test_risk_metrics_known_properties():
    rets = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, -0.01] * 60)
    bench = pd.Series([0.008, -0.015, 0.01, -0.004, 0.012, -0.009] * 60)
    report = RiskEngine._compute_metrics(rets, bench, risk_free_rate=0.045)

    assert report.var_99 <= report.var_95
    assert report.cvar_99 <= report.var_99
    assert report.annualized_volatility > 0
    assert -1 <= report.max_drawdown <= 0
