from __future__ import annotations

from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.regimes.label import label_regimes
from src.models.baseline import train_logreg_and_eval


DATA_PATH = Path("data/processed/model_dataset.parquet")
OUT_DIR = Path("outputs/day3")
OUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "ret_1d",
    "ret_5d",
    "logret_1d",
    "vol_10d",
    "vol_20d",
    "price_sma20",
    "sma20_sma50",
    "drawdown_60",
]


def run_for_ticker(df: pd.DataFrame, ticker: str) -> dict:
    g = df[df["ticker"] == ticker].copy()
    g = label_regimes(g)

    train_df = g[g["split"] == "train"]
    val_df = g[g["split"] == "val"]
    test_df = g[g["split"] == "test"]

    val_res = train_logreg_and_eval(train_df, val_df, FEATURES, target_col="y_dir_1d")
    test_res = train_logreg_and_eval(train_df, test_df, FEATURES, target_col="y_dir_1d")

    regime_counts = g["regime"].value_counts().to_dict()

    return {
        "ticker": ticker,
        "val": {"acc": val_res.accuracy, "f1": val_res.f1, "auc": val_res.auc},
        "test": {"acc": test_res.accuracy, "f1": test_res.f1, "auc": test_res.auc},
        "regime_counts": regime_counts,
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise RuntimeError("model_dataset.parquet missing. Run scripts/build_model_dataset.py first.")

    df = pd.read_parquet(DATA_PATH)
    tickers = sorted(df["ticker"].unique().tolist())

    results = [run_for_ticker(df, t) for t in tickers]

    out_path = OUT_DIR / "baseline_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"results": results, "features": FEATURES}, f, indent=2)

    print("Saved:", out_path)
    for r in results:
        print(
            r["ticker"],
            "val_acc", round(r["val"]["acc"], 4),
            "val_auc", round(r["val"]["auc"], 4) if r["val"]["auc"] == r["val"]["auc"] else "nan",
            "test_acc", round(r["test"]["acc"], 4),
        )


if __name__ == "__main__":
    main()