from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathlib import Path

import pandas as pd

from src.features.build_features import build_features_and_targets
from src.data.split import add_time_split


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        raise RuntimeError("No CSVs in data/raw. Run scripts/download_stooq.py first.")

    all_rows = []
    for csv_path in csvs:
        ticker = csv_path.stem
        df = pd.read_csv(csv_path)
        feats = build_features_and_targets(df)
        feats.insert(0, "ticker", ticker)
        all_rows.append(feats)

    full = pd.concat(all_rows, ignore_index=True)

    # Split per ticker to avoid mixing timeline differences
    parts = []
    for ticker, g in full.groupby("ticker", sort=True):
        g = add_time_split(g, date_col="date", train_frac=0.70, val_frac=0.15)
        parts.append(g)

    dataset = pd.concat(parts, ignore_index=True)

    out_path = OUT_DIR / "model_dataset.parquet"
    dataset.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(dataset))
    print("Tickers:", sorted(dataset["ticker"].unique().tolist()))
    print("Split counts:\n", dataset["split"].value_counts())


if __name__ == "__main__":
    main()