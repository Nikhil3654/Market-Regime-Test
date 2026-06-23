# Market-Regime-Test

A compact finance ML project that turns daily ETF price data into a complete research pipeline: feature engineering, market regime labeling, baseline modeling, backtesting, and visual diagnostics.

This is not meant to be a production trading system. The goal is to show a clean, skeptical, and reproducible finance workflow where every model result is connected back to time based evaluation and strategy behavior.

## Motivation 

A lot of finance ML demos look impressive but quietly make common mistakes:

- random train test splits on time series
- features that accidentally use future information
- model accuracy without a trading interpretation
- no baseline comparison
- no visual inspection of failure cases

This project keeps the setup intentionally simple and focuses on the fundamentals: clean features, time aware splits, interpretable baselines, and backtest diagnostics.

## Objectives

The pipeline uses daily ETF market data and builds a supervised learning dataset for next day direction prediction.

It includes:

- Feature engineering from daily OHLCV data
- Time based train, validation, and test splits
- Rule based market regime labels
- Logistic Regression baseline
- Random Forest baseline
- Probability threshold strategy backtest
- Equity curve plots
- Confusion matrix diagnostics
- CI safe tests for core logic

## Pipeline overview

```text
Daily OHLCV data
        |
        v
Backward looking features
        |
        v
Time based split
        |
        v
Regime labeling
        |
        v
Model training
        |
        v
Probability threshold backtest
        |
        v
Metrics and plots
```

## Features used

The model uses simple backward looking market features:

- 1 day return
- 5 day return
- log return
- 10 day rolling volatility
- 20 day rolling volatility
- price to 20 day moving average
- 20 day to 50 day moving average ratio
- 60 day drawdown

The target is next day direction:

```text
y_dir_1d = 1 if next day return > 0 else 0
```

The return target is shifted forward by one day, while all input features are computed using information available up to the current day.

## Market regimes

The regime labels are intentionally simple and interpretable. Each day is assigned to one of four regimes:

- bull low volatility
- bull high volatility
- bear low volatility
- bear high volatility

The labels are based on moving average trend and rolling volatility. This gives a simple way to inspect whether model behavior changes across different market conditions.

## Current results

The table below shows the Random Forest strategy on the held out test split.

The strategy rule is simple:

```text
go long if predicted probability of an up day >= 0.55
otherwise stay in cash
```

| Ticker | Model | Cumulative Return | Sharpe | Max Drawdown |
| --- | --- | ---: | ---: | ---: |
| SPY | Random Forest | 0.8151 | 1.7686 | -0.0592 |
| QQQ | Random Forest | 0.5501 | 0.8416 | -0.1663 |
| IWM | Random Forest | 0.8891 | 1.5509 | -0.1020 |

These results are best read as baseline research outputs. A serious trading version would still need transaction costs, slippage, robustness checks, and walk forward retraining.

## Project structure

```text
src/
  backtest/      strategy metrics and equity curve helpers
  data/          time split logic
  features/      feature and target construction
  models/        logistic regression and random forest baselines
  regimes/       rule based market regime labeling
  viz/           plotting utilities

scripts/
  download_stooq.py
  build_model_dataset.py
  train_baselines.py
  run_day4.py
  make_day5_plots.py
  leakage_check.py

reports/
  README ready plots

tests/
  lightweight tests that are safe to run in CI
```

## Reproducibility

Generated data and outputs are intentionally not committed:

```text
data/raw/
data/processed/
outputs/
```

The repo keeps the code, tests, and report graphics under version control. The scripts in `scripts/` can regenerate the local artifacts when the data is available.

## Limitations

This is a baseline finance ML project. The current version does not include:

- transaction costs
- slippage
- position sizing
- benchmark adjusted returns
- rolling walk forward retraining
- macroeconomic features
- a deployed dashboard
