from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@dataclass
class ModelResult:
    accuracy: float
    f1: float
    auc: float


def _prep_xy(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    X = df[feature_cols].astype(np.float32).values
    y = df[target_col].astype(int).values
    return X, y


def train_rf_and_eval(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "y_dir_1d",
    n_estimators: int = 300,
    max_depth: int | None = None,
    random_state: int = 42,
) -> ModelResult:
    Xtr, ytr = _prep_xy(train_df, feature_cols, target_col)
    Xev, yev = _prep_xy(eval_df, feature_cols, target_col)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xev)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(yev, pred)
    f1 = f1_score(yev, pred)
    try:
        auc = roc_auc_score(yev, proba)
    except Exception:
        auc = float("nan")

    return ModelResult(float(acc), float(f1), float(auc))