from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


DATA_PATH = Path("data/processed/model_dataset.parquet")


def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    required = ["date", "ticker", "close", "y_ret_1d", "y_dir_1d"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("Missing required columns:", missing)
        print("This check needs close and the y columns in the dataset.")
        return

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    ok = 0
    bad = 0
    for t in df["ticker"].unique():
        g = df[df["ticker"] == t].copy().reset_index(drop=True)
        if len(g) < 3:
            continue
        # compute next day return from close
        ret = (g["close"].shift(-1) / g["close"]) - 1.0
        diff = (ret - g["y_ret_1d"]).abs()
        # ignore last row which has no next day
        diff = diff.iloc[:-1]
        bad_here = int((diff > 1e-9).sum())
        bad += bad_here
        ok += int(len(diff) - bad_here)

    print("alignment_ok_rows:", ok)
    print("alignment_bad_rows:", bad)
    if bad == 0:
        print("PASS: y_ret_1d matches next day return from close")
    else:
        print("FAIL: y_ret_1d mismatch. Check target construction.")


if __name__ == "__main__":
    main()