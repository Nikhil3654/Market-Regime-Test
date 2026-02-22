from __future__ import annotations

import pandas as pd


def label_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, interpretable regime labels:
    - trend: close above SMA50 and SMA20 > SMA50
    - high_vol: vol_20d above its median for that ticker
    Combine into 4 regimes:
      bull_lowvol, bull_highvol, bear_lowvol, bear_highvol
    """
    out = df.copy()

    # Trend proxy
    bull = (out["close"] > out["sma_50"]) & (out["sma_20"] > out["sma_50"])

    # Vol threshold per ticker to avoid mixing scale
    def add_vol_flag(g: pd.DataFrame) -> pd.DataFrame:
        thr = g["vol_20d"].median()
        g = g.copy()
        g["high_vol"] = g["vol_20d"] > thr
        return g

    out = out.groupby("ticker", group_keys=False).apply(add_vol_flag)
    out["bull"] = bull.astype(int)

    def regime_row(r):
        if r["bull"] == 1 and r["high_vol"] == 0:
            return "bull_lowvol"
        if r["bull"] == 1 and r["high_vol"] == 1:
            return "bull_highvol"
        if r["bull"] == 0 and r["high_vol"] == 0:
            return "bear_lowvol"
        return "bear_highvol"

    out["regime"] = out.apply(regime_row, axis=1)
    return out