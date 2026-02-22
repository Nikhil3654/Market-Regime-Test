import pandas as pd
from src.regimes.label import label_regimes


def test_label_regimes_adds_columns():
    df = pd.DataFrame(
        {
            "ticker": ["X"] * 5,
            "close": [1, 2, 3, 2, 1],
            "sma_20": [1, 1, 2, 2, 1],
            "sma_50": [1, 1, 1, 1, 1],
            "vol_20d": [0.1, 0.2, 0.3, 0.2, 0.1],
        }
    )
    out = label_regimes(df)
    assert "regime" in out.columns
    assert "high_vol" in out.columns
    assert "bull" in out.columns