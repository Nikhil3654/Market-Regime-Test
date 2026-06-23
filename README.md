# Market-Regime-Test

A compact finance ML project that turns daily ETF price data into a complete research pipeline: feature engineering, market regime labeling, baseline modeling, backtesting, and visual diagnostics.

This is not meant to be a production trading system. The goal is to show a clean, skeptical, and reproducible finance workflow where every model result is connected back to time based evaluation and strategy behavior.

## Why I built this

A lot of finance ML demos look impressive but quietly make common mistakes:

- random train test splits on time series
- features that accidentally use future information
- model accuracy without a trading interpretation
- no baseline comparison
- no visual inspection of failure cases

This project keeps the setup intentionally simple and focuses on the fundamentals: clean features, time aware splits, interpretable baselines, and backtest diagnostics.

## What the project does

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