from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # Remove duplicates by date (keep last)
    df = df.drop_duplicates(subset=["date"], keep="last")

    # Basic numeric safety
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close"])
    return df.reset_index(drop=True)


def build_features_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_ohlcv(df)

    close = df["close"]

    # Returns
    df["ret_1d"] = close.pct_change()
    df["ret_5d"] = close.pct_change(5)
    df["logret_1d"] = np.log(close).diff()

    # Volatility
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()

    # Trend
    df["sma_10"] = close.rolling(10).mean()
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["price_sma20"] = close / df["sma_20"]
    df["sma20_sma50"] = df["sma_20"] / df["sma_50"]

    # Drawdown
    roll_max_60 = close.rolling(60).max()
    df["drawdown_60"] = (close / roll_max_60) - 1.0

    # Targets (shifted forward so no leakage)
    df["y_ret_1d"] = df["ret_1d"].shift(-1)
    df["y_dir_1d"] = (df["y_ret_1d"] > 0).astype(int)

    # Drop rows with NaNs from rolling windows and last target row
    df = df.dropna().reset_index(drop=True)

    return df