from __future__ import annotations

from pathlib import Path
import time
import requests

import pandas as pd


RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Keep it small Day 1
TICKERS = {
    # Stooq uses lowercase, and many tickers have ".us"
    "SPY": "spy.us",
    "QQQ": "qqq.us",
    "IWM": "iwm.us",
}

# Stooq direct daily CSV endpoint
# Example: https://stooq.com/q/d/l/?s=spy.us&i=d
STOOQ_URL = "https://stooq.com/q/d/l/"


def download_ticker(symbol: str, stooq_code: str) -> Path:
    params = {"s": stooq_code, "i": "d"}
    r = requests.get(STOOQ_URL, params=params, timeout=30)
    r.raise_for_status()

    out_path = RAW_DIR / f"{symbol}.csv"
    out_path.write_bytes(r.content)
    return out_path


def sanity_check(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"\nFile: {csv_path}")
    print("Rows:", len(df))
    if "date" in df.columns:
        print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Missing values total:", int(df.isna().sum().sum()))
    print("Columns:", list(df.columns))


def main() -> None:
    print("Downloading from Stooq...")
    for symbol, code in TICKERS.items():
        try:
            path = download_ticker(symbol, code)
            sanity_check(path)
            time.sleep(0.2)
        except Exception as e:
            print(f"Failed {symbol} ({code}): {e}")


if __name__ == "__main__":
    main()