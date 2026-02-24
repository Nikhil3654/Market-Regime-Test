from __future__ import annotations

from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from src.backtest.simple import equity_curve
from src.viz.plots import save_equity_curve, save_confusion_matrix


DATA_PATH = Path("data/processed/model_dataset.parquet")
OUT_DIR = Path("outputs/day5")
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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


def fit_models(train_df: pd.DataFrame):
    Xtr = train_df[FEATURES].astype(np.float32).values
    ytr = train_df["y_dir_1d"].astype(int).values

    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xtr, ytr)

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(Xtr, ytr)

    return lr, rf


def proba(model, df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].astype(np.float32).values
    return model.predict_proba(X)[:, 1]


def main() -> None:
    if not DATA_PATH.exists():
        raise RuntimeError("Missing model_dataset.parquet")

    df = pd.read_parquet(DATA_PATH)
    tickers = sorted(df["ticker"].unique().tolist())

    summary = {"tickers": {}, "threshold": 0.55}

    for t in tickers:
        g = df[df["ticker"] == t].copy()
        train_df = g[g["split"] == "train"].copy()
        test_df = g[g["split"] == "test"].copy()

        lr, rf = fit_models(train_df)

        test_df = test_df.sort_values("date")
        p_lr = proba(lr, test_df)
        p_rf = proba(rf, test_df)

        thresh = 0.55
        pos_lr = (p_lr >= thresh).astype(np.float64)
        pos_rf = (p_rf >= thresh).astype(np.float64)

        r = test_df["y_ret_1d"].to_numpy(dtype=np.float64)

        eq_lr = equity_curve(pos_lr * r)
        eq_rf = equity_curve(pos_rf * r)

        save_equity_curve(test_df["date"], eq_lr, f"{t} equity curve (logreg)", PLOTS_DIR / f"{t}_equity_logreg.png")
        save_equity_curve(test_df["date"], eq_rf, f"{t} equity curve (rf)", PLOTS_DIR / f"{t}_equity_rf.png")

        y_true = test_df["y_dir_1d"].astype(int).to_numpy()
        yhat_lr = (p_lr >= 0.5).astype(int)
        yhat_rf = (p_rf >= 0.5).astype(int)

        cm_lr = confusion_matrix(y_true, yhat_lr, labels=[0, 1])
        cm_rf = confusion_matrix(y_true, yhat_rf, labels=[0, 1])

        save_confusion_matrix(cm_lr, f"{t} test confusion (logreg)", PLOTS_DIR / f"{t}_cm_logreg.png")
        save_confusion_matrix(cm_rf, f"{t} test confusion (rf)", PLOTS_DIR / f"{t}_cm_rf.png")

        summary["tickers"][t] = {
            "final_equity_logreg": float(eq_lr[-1]) if len(eq_lr) else 1.0,
            "final_equity_rf": float(eq_rf[-1]) if len(eq_rf) else 1.0,
            "cm_logreg": cm_lr.tolist(),
            "cm_rf": cm_rf.tolist(),
        }

    out_json = OUT_DIR / "summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved plots to:", PLOTS_DIR)
    print("Saved summary to:", out_json)


if __name__ == "__main__":
    main()