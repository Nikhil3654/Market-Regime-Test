import pandas as pd
import numpy as np


def make_features_local(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1d"] = df["close"].pct_change()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    return df.dropna()


def test_features_columns_exist():
    n = 30
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "close": np.linspace(100, 120, n),
        }
    )
    out = make_features_local(df)
    assert "ret_1d" in out.columns
    assert "vol_20d" in out.columns
    assert len(out) > 0