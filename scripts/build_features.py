from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    # Basic returns
    df["ret_1d"] = df["close"].pct_change()
    df["logret_1d"] = np.log(df["close"]).diff()

    # Rolling features
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["meanret_20d"] = df["ret_1d"].rolling(20).mean()

    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_ratio_20_50"] = df["sma_20"] / df["sma_50"]

    # Drop rows with insufficient history
    return df.dropna().reset_index(drop=True)


def main() -> None:
    all_rows = []
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        raise RuntimeError("No CSVs found in data/raw. Run download_stooq.py first.")

    for csv_path in csvs:
        symbol = csv_path.stem
        df = pd.read_csv(csv_path)
        feats = make_features(df)
        feats.insert(0, "ticker", symbol)
        all_rows.append(feats)

    full = pd.concat(all_rows, ignore_index=True)
    out_path = OUT_DIR / "features.parquet"
    full.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(full))
    print("Tickers:", sorted(full["ticker"].unique().tolist()))
    print("Columns:", list(full.columns))


if __name__ == "__main__":
    main()