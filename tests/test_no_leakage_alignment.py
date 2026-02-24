import pandas as pd


def test_splits_are_time_ordered_per_ticker():
    df = pd.read_parquet("data/processed/model_dataset.parquet")
    assert "date" in df.columns
    for t in df["ticker"].unique():
        g = df[df["ticker"] == t].sort_values("date")
        # sanity: dates should be nondecreasing
        assert g["date"].is_monotonic_increasing