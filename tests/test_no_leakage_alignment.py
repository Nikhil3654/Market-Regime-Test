from pathlib import Path

import pandas as pd
import pytest


DATA_PATH = Path("data/processed/model_dataset.parquet")


@pytest.mark.skipif(not DATA_PATH.exists(), reason="Local processed dataset not available in CI")
def test_splits_are_time_ordered_per_ticker():
    df = pd.read_parquet(DATA_PATH)

    assert "date" in df.columns
    assert "ticker" in df.columns
    assert "split" in df.columns

    for ticker in df["ticker"].unique():
        g = df[df["ticker"] == ticker].sort_values("date")
        assert g["date"].is_monotonic_increasing

        train_max = g[g["split"] == "train"]["date"].max()
        val_min = g[g["split"] == "val"]["date"].min()
        val_max = g[g["split"] == "val"]["date"].max()
        test_min = g[g["split"] == "test"]["date"].min()

        assert train_max < val_min
        assert val_max < test_min