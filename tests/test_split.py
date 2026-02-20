import pandas as pd
from src.data.split import add_time_split


def test_add_time_split_has_all_splits():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=200, freq="D"),
            "x": range(200),
        }
    )
    out = add_time_split(df, date_col="date", train_frac=0.7, val_frac=0.15)
    assert set(out["split"].unique()) == {"train", "val", "test"}