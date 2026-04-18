# ABOUTME: Pulls BTC-USD daily OHLCV from yfinance for 2018-09-01 to 2026-04-15.
# ABOUTME: Output: data/yfinance/btcusd_daily.parquet. Run: source .venv/bin/activate && python scripts/stage1_pull_yfinance.py

import os
import sys
import time
from datetime import datetime

import pandas as pd
import yfinance as yf

PROJECT_ROOT = "/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "yfinance")
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "stage1_pull.log")

START_DATE = "2018-09-01"
END_DATE = "2026-04-16"  # yfinance end is exclusive

os.makedirs(DATA_DIR, exist_ok=True)


def log(msg, file=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if file:
        file.write(line + "\n")
        file.flush()


def main():
    t0 = time.time()

    with open(LOG_FILE, "a") as lf:
        log("=== yfinance BTC-USD pull started ===", lf)

        ticker = yf.Ticker("BTC-USD")
        df = ticker.history(start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True)

        if df is None or len(df) == 0:
            log("STOP: yfinance returned empty for BTC-USD", lf)
            sys.exit(1)

        df = df.reset_index()
        # Normalize column names
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # date column may be 'date' or 'datetime'
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["date"] = pd.to_datetime(df["date"])

        # Select and rename columns
        col_map = {"open": "open", "high": "high", "low": "low", "close": "close",
                   "volume": "volume"}
        keep = ["date"]
        for src, dst in col_map.items():
            if src in df.columns:
                df[dst] = df[src]
                keep.append(dst)

        # adj_close: after auto_adjust=True, 'close' is already adjusted
        df["adj_close"] = df["close"]
        keep.append("adj_close")

        out = df[keep].copy()
        out = out[(out["date"] >= START_DATE) & (out["date"] <= "2026-04-15")].copy()

        # Cross-check on 3 reference dates
        REFERENCE = {
            "2020-01-01": (6000, 10000),
            "2021-11-10": (60000, 70000),
            "2022-06-15": (18000, 25000),
        }
        for ref_date, (lo, hi) in REFERENCE.items():
            row = out[out["date"] == ref_date]
            if len(row) == 0:
                log(f"  WARN: reference date {ref_date} not in data (may be weekend/holiday)", lf)
                continue
            price = float(row["close"].iloc[0])
            status = "OK" if lo <= price <= hi else f"RANGE_CHECK_FAIL expected [{lo},{hi}]"
            log(f"  Cross-check {ref_date}: close={price:.0f} [{status}]", lf)

        out_path = os.path.join(DATA_DIR, "btcusd_daily.parquet")
        out.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

        date_min = out["date"].min().date()
        date_max = out["date"].max().date()
        missing = out["close"].isna().sum()
        log(f"  BTC-USD: {len(out)} rows, {date_min} to {date_max}, {missing} NaN close", lf)

        elapsed = time.time() - t0
        log(f"=== yfinance pull complete: {elapsed:.1f}s ===", lf)

    print(f"\nyfinance done in {elapsed:.1f}s. File: {out_path}")


if __name__ == "__main__":
    main()
