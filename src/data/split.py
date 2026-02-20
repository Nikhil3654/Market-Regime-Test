from __future__ import annotations

import pandas as pd


def add_time_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be > 0 and sum to < 1")

    out = df.copy()
    out = out.sort_values(date_col).reset_index(drop=True)

    unique_dates = out[date_col].drop_duplicates().sort_values().reset_index(drop=True)
    n = len(unique_dates)
    if n < 50:
        raise ValueError("Not enough dates to split safely (need at least 50).")

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    d_train_end = unique_dates.iloc[train_end - 1]
    d_val_end = unique_dates.iloc[val_end - 1]

    def label(d):
        if d <= d_train_end:
            return "train"
        if d <= d_val_end:
            return "val"
        return "test"

    out["split"] = out[date_col].apply(label)
    return out