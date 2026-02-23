from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(returns: np.ndarray) -> np.ndarray:
    eq = np.ones_like(returns, dtype=np.float64)
    for i in range(len(returns)):
        eq[i] = (eq[i - 1] if i > 0 else 1.0) * (1.0 + returns[i])
    return eq


def max_drawdown(eq: np.ndarray) -> float:
    peak = -1e9
    mdd = 0.0
    for v in eq:
        peak = max(peak, v)
        dd = (v / peak) - 1.0
        mdd = min(mdd, dd)
    return float(mdd)


def sharpe_daily(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return float("nan")
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd == 0:
        return float("nan")
    return float((mu / sd) * np.sqrt(252))


def run_threshold_strategy(df: pd.DataFrame, prob_col: str, ret_col: str, thresh: float = 0.55) -> dict:
    p = df[prob_col].to_numpy(dtype=np.float64)
    r = df[ret_col].to_numpy(dtype=np.float64)

    pos = (p >= thresh).astype(np.float64)
    strat = pos * r

    eq = equity_curve(strat)
    mdd = max_drawdown(eq)
    sh = sharpe_daily(strat)
    hit = float(np.mean((r > 0) == (p >= 0.5)))

    return {
        "n_days": int(len(strat)),
        "cum_return": float(eq[-1] - 1.0) if len(eq) else 0.0,
        "sharpe": sh,
        "max_drawdown": mdd,
        "hit_rate": hit,
    }