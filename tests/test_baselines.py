import numpy as np
import pandas as pd

from insight_engine import compare_to_baselines


def test_ensemble_beats_baselines():
    idx = pd.date_range("2024-01-01", periods=200, freq="B", tz="UTC")
    mom = np.r_[np.ones(80), np.zeros(40), -np.ones(80)]
    label = np.where(mom > 0.2, "Positive", np.where(mom < -0.2, "Negative", "Neutral"))
    ensemble = label.copy()  # perfect predictions

    df = pd.DataFrame({"momentum_norm": mom, "label": label, "ensemble_pred": ensemble}, index=idx)
    scores = compare_to_baselines(df)
    assert scores["ensemble"] > scores["always_positive"]
    assert scores["ensemble"] > scores["always_neutral"]
    assert scores["ensemble"] > scores["momentum_follow"]
