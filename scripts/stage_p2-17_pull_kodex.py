# ABOUTME: Stage P2-17 — Pull 069500 (KODEX 200) daily OHLCV via pykrx get_market_ohlcv endpoint.
# ABOUTME: Inputs: none (pykrx pulls from KRX via krx_tunnel). Output: data/pykrx/etfs/069500_KS_daily.parquet.
#           Columns: date, open, high, low, close, volume. Date range: 2019-2026.
#           Run: source .venv/bin/activate && python3 scripts/stage_p2-17_pull_kodex.py

import os
import sys
import logging
import time
from datetime import datetime

import pandas as pd

PROJECT = "/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity_p2"
LOGS = os.path.join(PROJECT, "logs")
OUT_DIR = os.path.join(PROJECT, "data", "pykrx", "etfs")
OUT_PATH = os.path.join(OUT_DIR, "069500_KS_daily.parquet")
LOG_PATH = os.path.join(LOGS, "stage_p2-17_env.log")

os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
log.info("=" * 70)
log.info("Stage P2-17: 069500 KODEX 200 daily OHLCV pull — started")

# pykrx version check
import pykrx
log.info(f"pykrx version: {pykrx.__version__}")

# Annual chunks to stay inside KRX query limits
CHUNKS = [
    ("20190101", "20191231"),
    ("20200101", "20201231"),
    ("20210101", "20211231"),
    ("20220101", "20221231"),
    ("20230101", "20231231"),
    ("20240101", "20241231"),
    ("20250101", "20251231"),
    ("20260101", "20260430"),
]

# Column rename: pykrx get_market_ohlcv returns Korean columns
RENAME = {
    "시가": "open",
    "고가": "high",
    "저가": "low",
    "종가": "close",
    "거래량": "volume",
}

sys.path.insert(0, os.path.join(PROJECT, "scripts"))
from krx_tunnel import krx_tunnel  # noqa: E402


def pull_chunks():
    frames = []
    with krx_tunnel():
        from pykrx import stock  # must import inside tunnel context
        for fromdate, todate in CHUNKS:
            log.info(f"  Pulling 069500 {fromdate} to {todate}…")
            try:
                df = stock.get_market_ohlcv(fromdate=fromdate, todate=todate, ticker="069500")
                if df is None or df.empty:
                    log.warning(f"  Empty result for {fromdate}-{todate}")
                    continue
                log.info(f"  Got {len(df)} rows")
                frames.append(df)
                time.sleep(0.5)  # polite delay between chunks
            except Exception as exc:
                log.warning(f"  Chunk {fromdate}-{todate} failed: {exc}")
                time.sleep(2)
                try:
                    df = stock.get_market_ohlcv(fromdate=fromdate, todate=todate, ticker="069500")
                    if df is not None and not df.empty:
                        frames.append(df)
                        log.info(f"  Retry OK: {len(df)} rows")
                except Exception as exc2:
                    log.error(f"  Chunk {fromdate}-{todate} retry also failed: {exc2}")
    return frames


frames = pull_chunks()

if not frames:
    log.error("No data retrieved for any chunk. Check tunnel and pykrx.")
    sys.exit(1)

raw = pd.concat(frames)
raw.index = pd.to_datetime(raw.index)
raw.index.name = "date"
raw = raw.reset_index()

# Rename Korean columns
raw = raw.rename(columns=RENAME)
# Drop extra columns (등락률 / pct_change etc.)
keep_cols = ["date", "open", "high", "low", "close", "volume"]
for c in keep_cols:
    if c not in raw.columns:
        raw[c] = float("nan")
raw = raw[keep_cols].copy()

# Dedupe by date, sort ascending
raw = raw.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)

# Verify schema
assert raw.dtypes["date"].kind == "M", "date column must be datetime"
for c in ["open", "high", "low", "close"]:
    assert raw[c].notna().sum() > 0, f"{c} is all NaN"

# Anomaly checks
zero_hl = (raw["high"] == 0) | (raw["low"] == 0)
n_zero = int(zero_hl.sum())
if n_zero > 0:
    log.warning(f"  Rows with high==0 or low==0: {n_zero} (suspension days — kept)")

n_rows = len(raw)
date_min = raw["date"].min()
date_max = raw["date"].max()

log.info(f"069500 OHLCV: {n_rows} rows, {date_min.date()} to {date_max.date()}")
log.info(f"Columns: {raw.columns.tolist()}")
log.info(f"Dtypes:\n{raw.dtypes.to_string()}")

# Verification
assert n_rows > 1500, f"Expected >1500 rows, got {n_rows}"
assert date_min.year == 2019, f"Expected date_min year 2019, got {date_min.year}"
assert set(["date", "open", "high", "low", "close", "volume"]).issubset(raw.columns), \
    "Missing columns"

raw.to_parquet(OUT_PATH, index=False)
log.info(f"Written {OUT_PATH}: {n_rows} rows")
log.info(f"Stage P2-17 pull completed: {n_rows} rows, {date_min.date()} to {date_max.date()}")
