# ABOUTME: Pulls Binance spot/perp 1-min klines, funding rates, and liquidations via bulk archive.
# ABOUTME: Output: data/binance/{spot_klines,perp_klines,funding_rates,liquidations}/. Run: source .venv/bin/activate && python scripts/stage1_pull_binance.py
# ABOUTME: Uses multiprocessing Pool of 8 workers. 45-min budget cap with resumability note.

import io
import os
import sys
import time
import zipfile
import logging
from datetime import datetime, timezone, timedelta, date
from multiprocessing import Pool, Manager
from functools import partial as functools_partial

import pandas as pd
import requests

PROJECT_ROOT = "/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "binance")
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "stage1_pull.log")

START_DATE = date(2018, 9, 1)
END_DATE = date(2026, 4, 15)
BUDGET_SECS = 45 * 60  # 45 minutes

SPOT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
PERP_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
FUNDING_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
LIQ_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

BULK_BASE = "https://data.binance.vision/data"
WORKERS = 8

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

LIQ_COLS = [
    "symbol", "side", "order_type", "time_in_force", "orig_qty",
    "price", "avg_price", "order_status", "last_filled_qty",
    "accumulated_filled_qty", "trade_time",
]


# ─────────────────────────────────────────────────────────────────────────────
# Log helpers (thread/process-safe via print to stdout, consolidated at end)
# ─────────────────────────────────────────────────────────────────────────────

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def logprint(msg):
    print(f"[{ts()}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_zip(url, retries=3) -> bytes | None:
    """Download a zip URL with retries. Returns raw bytes or None on 404/permanent failure."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 404:
                return None
            if resp.status_code == 200:
                return resp.content
            resp.raise_for_status()
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def _parse_zip_csv(raw_bytes, names=None) -> pd.DataFrame | None:
    """Extract CSV from zip bytes and parse into DataFrame."""
    try:
        z = zipfile.ZipFile(io.BytesIO(raw_bytes))
    except zipfile.BadZipFile:
        return None
    csv_name = z.namelist()[0]
    with z.open(csv_name) as f:
        raw_text = f.read().decode(errors="replace")
    if not raw_text.strip():
        return None
    first_char = raw_text.lstrip()[0]
    if names and first_char.isdigit():
        df = pd.read_csv(io.StringIO(raw_text), header=None, names=names)
    else:
        df = pd.read_csv(io.StringIO(raw_text))
    return df


def _parse_open_time(series: pd.Series) -> pd.Series:
    """Handle Binance ms vs μs open_time ambiguity."""
    ot = series.astype(float)
    ot = ot.where(ot < 1e13, ot / 1000)  # μs → ms
    return pd.to_datetime(ot.astype(int), unit="ms", utc=True)


# ─────────────────────────────────────────────────────────────────────────────
# Worker functions (called in Pool)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_spot_day(args):
    """Fetch one day of spot 1m klines. Returns (date_str, df_or_None, gap_minutes)."""
    symbol, date_str = args
    url = (f"{BULK_BASE}/spot/daily/klines/{symbol}/1m/"
           f"{symbol}-1m-{date_str}.zip")
    raw = _fetch_zip(url)
    if raw is None:
        return (date_str, None, 0)
    df = _parse_zip_csv(raw, names=KLINE_COLS)
    if df is None or df.empty:
        return (date_str, None, 0)
    df = df.drop(columns=["ignore"], errors="ignore")
    df["open_time"] = _parse_open_time(df["open_time"])
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_volume", "taker_buy_quote_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Count missing minutes (expected 1440 per day)
    gap = max(0, 1440 - len(df))
    return (date_str, df, gap)


def _fetch_perp_day(args):
    """Fetch one day of perp (fapi) 1m klines."""
    symbol, date_str = args
    url = (f"{BULK_BASE}/futures/um/daily/klines/{symbol}/1m/"
           f"{symbol}-1m-{date_str}.zip")
    raw = _fetch_zip(url)
    if raw is None:
        return (date_str, None, 0)
    df = _parse_zip_csv(raw, names=KLINE_COLS)
    if df is None or df.empty:
        return (date_str, None, 0)
    df = df.drop(columns=["ignore"], errors="ignore")
    df["open_time"] = _parse_open_time(df["open_time"])
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_volume", "taker_buy_quote_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    gap = max(0, 1440 - len(df))
    return (date_str, df, gap)


def _fetch_liq_day(args):
    """Fetch one day of liquidation snapshots."""
    symbol, date_str = args
    url = (f"{BULK_BASE}/futures/um/daily/liquidationSnapshot/{symbol}/"
           f"{symbol}-liquidationSnapshot-{date_str}.zip")
    raw = _fetch_zip(url)
    if raw is None:
        return (date_str, None)
    df = _parse_zip_csv(raw)
    if df is None or df.empty:
        return (date_str, None)
    # Normalize columns
    df.columns = [c.lower().strip() for c in df.columns]
    rename = {
        "symbol": "symbol", "side": "side", "order_type": "order_type",
        "time_in_force": "time_in_force", "original_quantity": "orig_qty",
        "price": "price", "average_price": "avg_price",
        "order_status": "order_status", "last_filled_quantity": "last_filled_qty",
        "filled_accumulated_quantity": "accumulated_filled_qty",
        "trade_time": "trade_time",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    # time column
    time_col = next((c for c in df.columns if "time" in c), None)
    if time_col:
        df["time"] = _parse_open_time(df[time_col])
    # qty and price
    for col in ["price", "avg_price", "orig_qty", "last_filled_qty", "accumulated_filled_qty"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # usd_notional
    if "price" in df.columns and "accumulated_filled_qty" in df.columns:
        df["usd_notional"] = df["price"] * df["accumulated_filled_qty"]
    elif "price" in df.columns and "orig_qty" in df.columns:
        df["usd_notional"] = df["price"] * df["orig_qty"]
    out_cols = [c for c in ["time", "side", "price", "accumulated_filled_qty", "usd_notional"] if c in df.columns]
    if "time" not in df.columns:
        return (date_str, None)
    df = df.rename(columns={"accumulated_filled_qty": "qty"})
    out_cols = [c for c in ["time", "side", "price", "qty", "usd_notional"] if c in df.columns]
    return (date_str, df[out_cols] if out_cols else None)


def _fetch_funding_month(args):
    """Fetch one month of funding rates."""
    symbol, year, month = args
    url = (f"{BULK_BASE}/futures/um/monthly/fundingRate/{symbol}/"
           f"{symbol}-fundingRate-{year}-{month:02d}.zip")
    raw = _fetch_zip(url)
    if raw is None:
        return (f"{year}-{month:02d}", None)
    df = _parse_zip_csv(raw)
    if df is None or df.empty:
        return (f"{year}-{month:02d}", None)
    df.columns = [c.lower().strip() for c in df.columns]
    # Columns vary: calc_time / funding_time, last_funding_rate / funding_rate, mark_price
    time_col = next((c for c in df.columns if "time" in c), None)
    rate_col = next((c for c in df.columns if "rate" in c), None)
    if time_col is None or rate_col is None:
        return (f"{year}-{month:02d}", None)
    df["funding_time"] = _parse_open_time(df[time_col])
    df["funding_rate"] = pd.to_numeric(df[rate_col], errors="coerce")
    out = {"funding_time": df["funding_time"], "funding_rate": df["funding_rate"]}
    if "mark_price" in df.columns:
        out["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce")
    return (f"{year}-{month:02d}", pd.DataFrame(out))


# ─────────────────────────────────────────────────────────────────────────────
# Day-list helpers
# ─────────────────────────────────────────────────────────────────────────────

def date_range_str(start: date, end: date):
    """Yield YYYY-MM-DD strings from start to end inclusive."""
    d = start
    while d <= end:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def month_range(start: date, end: date):
    """Yield (year, month) from start to end month inclusive."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, path: str, lf):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    logprint(f"  Saved: {path} ({len(df)} rows)")
    lf.write(f"[{ts()}]   Saved: {path} ({len(df)} rows)\n")
    lf.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Pull functions
# ─────────────────────────────────────────────────────────────────────────────

def pull_spot_klines(lf, deadline):
    logprint("=== Spot 1m klines ===")
    lf.write(f"[{ts()}] === Spot 1m klines ===\n")

    gap_log = {}  # symbol -> list of (date_str, gap_min)
    coverage = {}

    for symbol in SPOT_SYMBOLS:
        if time.time() > deadline:
            logprint(f"  BUDGET CAP: stopping spot at symbol {symbol}")
            lf.write(f"[{ts()}]   BUDGET CAP: stopping spot at symbol {symbol}\n")
            break

        logprint(f"  Downloading {symbol} spot 1m...")
        all_dates = list(date_range_str(START_DATE, END_DATE))
        args = [(symbol, d) for d in all_dates]

        frames = []
        gaps = []
        found_days = 0

        t_sym = time.time()
        with Pool(WORKERS) as pool:
            for (date_str, df, gap_min) in pool.imap_unordered(_fetch_spot_day, args, chunksize=10):
                if time.time() > deadline:
                    pool.terminate()
                    logprint(f"    BUDGET CAP mid-symbol {symbol} at {date_str}")
                    break
                if df is not None:
                    frames.append(df)
                    found_days += 1
                    if gap_min > 30:
                        gaps.append((date_str, gap_min))

        if not frames:
            logprint(f"  WARN: no data for {symbol}")
            coverage[symbol] = 0.0
            continue

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("open_time").drop_duplicates("open_time")
        out_path = os.path.join(DATA_DIR, "spot_klines", f"{symbol}_1m.parquet")
        save_parquet(combined, out_path, lf)

        total_days = len(all_dates)
        cov = found_days / total_days
        coverage[symbol] = cov
        logprint(f"  {symbol}: {found_days}/{total_days} days ({cov:.1%}), {len(gaps)} days with >30min gap, {time.time()-t_sym:.0f}s")
        lf.write(f"[{ts()}]   {symbol}: {found_days}/{total_days} days ({cov:.1%}), {time.time()-t_sym:.0f}s\n")

        if gaps:
            gap_log[symbol] = gaps
            lf.write(f"[{ts()}]   Binance gaps (>30 min) for {symbol}:\n")
            for gdate, gmin in gaps[:50]:  # log first 50
                lf.write(f"             {gdate}: {gmin} min missing\n")
            lf.flush()

    # Stop condition: coverage < 80% for BTC or ETH
    for sym in ["BTCUSDT", "ETHUSDT"]:
        cov = coverage.get(sym, 0.0)
        if cov < 0.80:
            logprint(f"STOP: {sym} spot coverage {cov:.1%} < 80%")
            lf.write(f"[{ts()}] STOP: {sym} spot coverage {cov:.1%} < 80%\n")
            sys.exit(1)


def pull_perp_klines(lf, deadline):
    logprint("=== Perp 1m klines ===")
    lf.write(f"[{ts()}] === Perp 1m klines ===\n")

    for symbol in PERP_SYMBOLS:
        if time.time() > deadline:
            logprint(f"  BUDGET CAP: stopping perp at symbol {symbol}")
            break

        logprint(f"  Downloading {symbol} perp 1m...")
        all_dates = list(date_range_str(START_DATE, END_DATE))
        args = [(symbol, d) for d in all_dates]

        frames = []
        found_days = 0
        gaps = []

        t_sym = time.time()
        with Pool(WORKERS) as pool:
            for (date_str, df, gap_min) in pool.imap_unordered(_fetch_perp_day, args, chunksize=10):
                if time.time() > deadline:
                    pool.terminate()
                    logprint(f"    BUDGET CAP mid-symbol {symbol}")
                    break
                if df is not None:
                    frames.append(df)
                    found_days += 1
                    if gap_min > 30:
                        gaps.append((date_str, gap_min))

        if not frames:
            logprint(f"  WARN: no perp data for {symbol}")
            continue

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("open_time").drop_duplicates("open_time")
        out_path = os.path.join(DATA_DIR, "perp_klines", f"{symbol}_1m.parquet")
        save_parquet(combined, out_path, lf)

        total_days = len(all_dates)
        cov = found_days / total_days
        logprint(f"  {symbol} perp: {found_days}/{total_days} days ({cov:.1%}), {time.time()-t_sym:.0f}s")
        lf.write(f"[{ts()}]   {symbol} perp: {found_days}/{total_days} days ({cov:.1%}), {time.time()-t_sym:.0f}s\n")

        if gaps:
            lf.write(f"[{ts()}]   Perp gaps (>30 min) for {symbol}:\n")
            for gdate, gmin in gaps[:50]:
                lf.write(f"             {gdate}: {gmin} min missing\n")
            lf.flush()


def pull_funding_rates(lf, deadline):
    logprint("=== Funding rates ===")
    lf.write(f"[{ts()}] === Funding rates ===\n")

    for symbol in FUNDING_SYMBOLS:
        if time.time() > deadline:
            logprint(f"  BUDGET CAP: stopping funding at {symbol}")
            break

        months = list(month_range(START_DATE, END_DATE))
        args = [(symbol, y, m) for (y, m) in months]

        frames = []
        t_sym = time.time()
        with Pool(min(WORKERS, 4)) as pool:
            for (period, df) in pool.imap_unordered(_fetch_funding_month, args):
                if df is not None:
                    frames.append(df)

        if not frames:
            logprint(f"  WARN: no funding data for {symbol}")
            continue

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("funding_time").drop_duplicates("funding_time")
        out_path = os.path.join(DATA_DIR, "funding_rates", f"{symbol}_funding.parquet")
        save_parquet(combined, out_path, lf)
        logprint(f"  {symbol} funding: {len(combined)} rows, {time.time()-t_sym:.0f}s")
        lf.write(f"[{ts()}]   {symbol} funding: {len(combined)} rows\n")


def pull_liquidations(lf, deadline):
    logprint("=== Liquidations ===")
    lf.write(f"[{ts()}] === Liquidations ===\n")

    for symbol in LIQ_SYMBOLS:
        if time.time() > deadline:
            logprint(f"  BUDGET CAP: stopping liq at {symbol}")
            break

        all_dates = list(date_range_str(date(2020, 1, 1), END_DATE))  # liq data from ~2020
        args = [(symbol, d) for d in all_dates]

        frames = []
        found_days = 0
        t_sym = time.time()
        with Pool(WORKERS) as pool:
            for (date_str, df) in pool.imap_unordered(_fetch_liq_day, args, chunksize=10):
                if time.time() > deadline:
                    pool.terminate()
                    break
                if df is not None:
                    frames.append(df)
                    found_days += 1

        if not frames:
            logprint(f"  WARN: no liquidation data for {symbol}")
            continue

        combined = pd.concat(frames, ignore_index=True)
        if "time" in combined.columns:
            combined = combined.sort_values("time")
        out_path = os.path.join(DATA_DIR, "liquidations", f"{symbol}_liq.parquet")
        save_parquet(combined, out_path, lf)
        logprint(f"  {symbol} liq: {len(combined)} rows from {found_days} days, {time.time()-t_sym:.0f}s")
        lf.write(f"[{ts()}]   {symbol} liq: {len(combined)} rows from {found_days} days\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.join(DATA_DIR, "spot_klines"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "perp_klines"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "funding_rates"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "liquidations"), exist_ok=True)

    deadline = time.time() + BUDGET_SECS
    t0 = time.time()

    with open(LOG_FILE, "a") as lf:
        lf.write(f"\n[{ts()}] === Binance pull started ===\n")
        lf.flush()

        pull_funding_rates(lf, deadline)
        pull_liquidations(lf, deadline)
        pull_perp_klines(lf, deadline)
        pull_spot_klines(lf, deadline)

        elapsed = time.time() - t0
        budget_hit = time.time() > deadline
        lf.write(f"[{ts()}] === Binance pull complete: {elapsed:.1f}s {'(BUDGET CAP HIT)' if budget_hit else ''} ===\n")
        lf.flush()

    logprint(f"\nBinance done in {elapsed:.1f}s {'(BUDGET CAP)' if budget_hit else ''}.")
    if budget_hit:
        logprint("RESUMABILITY: Re-run this script; existing parquet files will be overwritten. "
                 "To resume incrementally, filter date args to only missing dates.")


if __name__ == "__main__":
    main()
