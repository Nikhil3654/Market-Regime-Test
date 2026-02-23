import numpy as np
import pandas as pd
from src.backtest.simple import run_threshold_strategy


def test_backtest_runs():
    df = pd.DataFrame({"p": np.array([0.6, 0.4, 0.9, 0.2]), "r": np.array([0.01, -0.02, 0.03, 0.01])})
    res = run_threshold_strategy(df, prob_col="p", ret_col="r", thresh=0.55)
    assert "cum_return" in res
    assert "max_drawdown" in res