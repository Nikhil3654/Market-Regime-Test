from __future__ import annotations

from pathlib import Path
import sys
import json
import csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.regimes.label import label_regimes
from src.models.baseline import train_logreg_and_eval
from src.models.tree_baseline import train_rf_and_eval
from src.backtest.simple import run_threshold_strategy


DATA_PATH = Path("data/processed/model_dataset.parquet")
OUT_DIR = Path("outputs/day4")
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


def proba_logreg(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
    Xtr = train_df[FEATURES].astype(np.float32).values
    ytr = train_df["y_dir_1d"].astype(int).values
    Xev = eval_df[FEATURES].astype(np.float32).values
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xev)[:, 1]


def proba_rf(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
    Xtr = train_df[FEATURES].astype(np.float32).values
    ytr = train_df["y_dir_1d"].astype(int).values
    Xev = eval_df[FEATURES].astype(np.float32).values

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xev)[:, 1]


def run_for_ticker(df: pd.DataFrame, ticker: str, thresh: float = 0.55) -> dict:
    g = df[df["ticker"] == ticker].copy()
    g = label_regimes(g)

    train_df = g[g["split"] == "train"].copy()
    val_df = g[g["split"] == "val"].copy()
    test_df = g[g["split"] == "test"].copy()

    log_val = train_logreg_and_eval(train_df, val_df, FEATURES, target_col="y_dir_1d")
    log_test = train_logreg_and_eval(train_df, test_df, FEATURES, target_col="y_dir_1d")

    rf_val = train_rf_and_eval(train_df, val_df, FEATURES, target_col="y_dir_1d")
    rf_test = train_rf_and_eval(train_df, test_df, FEATURES, target_col="y_dir_1d")

    val_df["p_logreg"] = proba_logreg(train_df, val_df)
    test_df["p_logreg"] = proba_logreg(train_df, test_df)
    val_df["p_rf"] = proba_rf(train_df, val_df)
    test_df["p_rf"] = proba_rf(train_df, test_df)

    bt_val_log = run_threshold_strategy(val_df, prob_col="p_logreg", ret_col="y_ret_1d", thresh=thresh)
    bt_test_log = run_threshold_strategy(test_df, prob_col="p_logreg", ret_col="y_ret_1d", thresh=thresh)

    bt_val_rf = run_threshold_strategy(val_df, prob_col="p_rf", ret_col="y_ret_1d", thresh=thresh)
    bt_test_rf = run_threshold_strategy(test_df, prob_col="p_rf", ret_col="y_ret_1d", thresh=thresh)

    return {
        "ticker": ticker,
        "threshold": thresh,
        "metrics": {
            "logreg": {
                "val": {"acc": log_val.accuracy, "f1": log_val.f1, "auc": log_val.auc},
                "test": {"acc": log_test.accuracy, "f1": log_test.f1, "auc": log_test.auc},
            },
            "rf": {
                "val": {"acc": rf_val.accuracy, "f1": rf_val.f1, "auc": rf_val.auc},
                "test": {"acc": rf_test.accuracy, "f1": rf_test.f1, "auc": rf_test.auc},
            },
        },
        "backtest": {
            "logreg": {"val": bt_val_log, "test": bt_test_log},
            "rf": {"val": bt_val_rf, "test": bt_test_rf},
        },
        "regime_counts": g["regime"].value_counts().to_dict(),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise RuntimeError("Missing model_dataset.parquet. Run scripts/build_model_dataset.py first.")

    df = pd.read_parquet(DATA_PATH)
    tickers = sorted(df["ticker"].unique().tolist())

    all_results = [run_for_ticker(df, t, thresh=0.55) for t in tickers]

    out_json = OUT_DIR / "day4_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"results": all_results, "features": FEATURES}, f, indent=2)

    out_csv = OUT_DIR / "day4_backtest_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "ticker",
            "model",
            "split",
            "threshold",
            "cum_return",
            "sharpe",
            "max_drawdown",
            "hit_rate",
            "n_days",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            for model in ["logreg", "rf"]:
                for split in ["val", "test"]:
                    bt = r["backtest"][model][split]
                    w.writerow(
                        {
                            "ticker": r["ticker"],
                            "model": model,
                            "split": split,
                            "threshold": r["threshold"],
                            "cum_return": bt["cum_return"],
                            "sharpe": bt["sharpe"],
                            "max_drawdown": bt["max_drawdown"],
                            "hit_rate": bt["hit_rate"],
                            "n_days": bt["n_days"],
                        }
                    )

    print("Saved:", out_json)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()