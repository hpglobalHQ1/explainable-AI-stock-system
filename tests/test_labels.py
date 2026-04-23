import pandas as pd

from label_engine import create_labels


def test_label_computation_forward_5d_only():
    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    close = [100, 100, 100, 100, 100, 102, 98, 100, 103, 97]
    df = pd.DataFrame({"close": close}, index=idx)
    out = create_labels(df)

    # T=0 uses T+5 = 102 => +2% => Positive
    assert out.iloc[0]["label"] == "Positive"
    # T=1 uses T+5 = 98 => -2% => Negative
    assert out.iloc[1]["label"] == "Negative"
    # last 5 rows lack horizon
    assert out["label"].tail(5).isna().all()
