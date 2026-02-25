# Market-Regime-Test

Lightweight market regime detection and directional forecasting baselines with a simple probability threshold strategy backtest.

## What this repo does
- Builds a supervised dataset from daily OHLCV data for a small set of ETFs (example: SPY, QQQ, IWM)
- Trains baseline classifiers for next day direction (Logistic Regression and Random Forest)
- Evaluates predictive metrics (accuracy, F1, ROC AUC)
- Runs a simple trading rule:
  - go long when predicted probability of up move >= 0.55
  - otherwise stay in cash
- Produces equity curves and confusion matrices for quick sanity checks

## Dataset
This project is designed to use publicly available market data (daily bars).  
You can regenerate the dataset locally by running the dataset build script.

Output artifact:
- `data/processed/model_dataset.parquet`

## Quickstart (Windows)
```powershell
cd C:\github\Market-Regime-Test
.\.venv\Scripts\Activate.ps1

# Build dataset
python scripts\build_model_dataset.py

# Day 4: baselines + backtest summary
python scripts\run_day4.py

# Day 5: plots
python scripts\make_day5_plots.py