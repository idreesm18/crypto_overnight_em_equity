# ABOUTME: Pulls FRED macro series (VIXCLS, DTWEXBGS, DGS10, DGS2, DFF, T5YIE, T10YIE) via fredapi.
# ABOUTME: Inputs: FRED_API_KEY from .env. Output: data/fred/{SERIES}.parquet + data/fred/fred_all.parquet.
# ABOUTME: Run: source .venv/bin/activate && python scripts/stage1_pull_fred.py

import os
import sys
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

PROJECT_ROOT = "/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "fred")
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "stage1_pull.log")

SERIES = ["VIXCLS", "DTWEXBGS", "DGS10", "DGS2", "DFF", "T5YIE", "T10YIE"]
START_DATE = "2018-09-01"
END_DATE = "2026-04-15"

os.makedirs(DATA_DIR, exist_ok=True)


def log(msg, file=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if file:
        file.write(line + "\n")
        file.flush()


def main():
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("STOP: FRED_API_KEY not found in .env")
        sys.exit(1)

    fred = Fred(api_key=api_key)

    t0 = time.time()
    all_frames = []

    with open(LOG_FILE, "a") as lf:
        log("=== FRED pull started ===", lf)

        for series_id in SERIES:
            try:
                raw = fred.get_series(series_id, observation_start=START_DATE,
                                      observation_end=END_DATE)
                if raw is None or len(raw) == 0:
                    log(f"  WARNING: {series_id} returned empty", lf)
                    continue
            except Exception as e:
                err_str = str(e)
                if "401" in err_str or "403" in err_str:
                    log(f"STOP: FRED API key error on {series_id}: {e}", lf)
                    sys.exit(1)
                log(f"  ERROR fetching {series_id}: {e}", lf)
                continue

            df = raw.reset_index()
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])

            # Forward-fill weekends/holidays then drop leading NaN
            df = df.set_index("date").reindex(
                pd.date_range(df["date"].min(), df["date"].max(), freq="D")
            ).ffill().reset_index()
            df.columns = ["date", "value"]
            df = df.dropna(subset=["value"])

            # Filter to requested range
            df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()

            out_path = os.path.join(DATA_DIR, f"{series_id}.parquet")
            df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

            date_min = df["date"].min().date()
            date_max = df["date"].max().date()
            missing = df["value"].isna().sum()
            log(f"  {series_id}: {len(df)} rows, {date_min} to {date_max}, {missing} NaN", lf)

            long_df = df.copy()
            long_df["series"] = series_id
            all_frames.append(long_df)

        # Merged long-format file
        if all_frames:
            fred_all = pd.concat(all_frames, ignore_index=True)[["date", "series", "value"]]
            fred_all_path = os.path.join(DATA_DIR, "fred_all.parquet")
            fred_all.to_parquet(fred_all_path, engine="pyarrow", compression="snappy", index=False)
            log(f"  fred_all.parquet: {len(fred_all)} rows", lf)

        elapsed = time.time() - t0
        log(f"=== FRED pull complete: {elapsed:.1f}s ===", lf)

    print(f"\nFRED done in {elapsed:.1f}s. Files in {DATA_DIR}")


if __name__ == "__main__":
    main()
