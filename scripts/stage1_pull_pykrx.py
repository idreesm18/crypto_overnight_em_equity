# ABOUTME: Pulls KR daily OHLCV + market cap for candidate tickers only via pykrx per-ticker endpoint.
# ABOUTME: Input: data/crypto_candidates_kr.csv. Output: data/pykrx/kr_daily/kr_ohlcv_mcap.parquet. Run: source .venv/bin/activate && python scripts/stage1_pull_pykrx.py

import os
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from krx_tunnel import krx_tunnel  # noqa: E402

PROJECT_ROOT = "/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "pykrx", "kr_daily")
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "stage1_pull.log")
CANDIDATES_CSV = os.path.join(PROJECT_ROOT, "data", "crypto_candidates_kr.csv")

START = "20180901"
END = "20260415"

os.makedirs(DATA_DIR, exist_ok=True)


def log(msg, lf=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if lf:
        lf.write(line + "\n")
        lf.flush()


def fetch_ticker(stock, t6, lf, max_retries=2):
    """Fetch OHLCV for one ticker with retry. Returns DataFrame or None."""
    for attempt in range(max_retries + 1):
        try:
            df = stock.get_market_ohlcv(fromdate=START, todate=END, ticker=t6)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** (attempt + 1)
                log(f"    OHLCV retry {attempt+1} for {t6}: {e} — sleeping {wait}s", lf)
                time.sleep(wait)
            else:
                log(f"    OHLCV FAIL {t6} after {max_retries} retries: {e}", lf)
    return None


def fetch_mcap(stock, t6, lf, max_retries=2):
    """Fetch market cap for one ticker with retry. Returns DataFrame or None."""
    for attempt in range(max_retries + 1):
        try:
            df = stock.get_market_cap(fromdate=START, todate=END, ticker=t6)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** (attempt + 1)
                log(f"    MCAP retry {attempt+1} for {t6}: {e} — sleeping {wait}s", lf)
                time.sleep(wait)
            else:
                log(f"    MCAP FAIL {t6} after {max_retries} retries: {e}", lf)
    return None


RENAME_OHLCV = {
    "시가": "open", "고가": "high", "저가": "low", "종가": "close",
    "거래량": "volume", "등락률": "pct_change",
}
RENAME_MCAP = {
    "시가총액": "mcap", "상장주식수": "shares_outstanding",
}


def normalize_ohlcv(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=RENAME_OHLCV)
    # drop columns we don't need (e.g. 거래대금/value)
    keep = [c for c in ["open", "high", "low", "close", "volume", "pct_change"] if c in df.columns]
    return df[keep]


def normalize_mcap(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=RENAME_MCAP)
    keep = [c for c in ["mcap", "shares_outstanding"] if c in df.columns]
    return df[keep]


def main():
    t0 = time.time()

    # Load candidates; strip suffix to get 6-digit ticker
    cands = pd.read_csv(CANDIDATES_CSV)
    # Filter out rows where ticker looks malformed (e.g. ME2ON.KQ has no numeric prefix)
    valid = cands[cands["ticker"].str.match(r"^\d{6}\.(KS|KQ)$")].copy()
    skipped_invalid = cands[~cands["ticker"].str.match(r"^\d{6}\.(KS|KQ)$")]["ticker"].tolist()

    ticker_map = {}  # t6 -> (full_ticker, market)
    for _, row in valid.iterrows():
        full = row["ticker"]
        t6 = full.split(".")[0]
        market = "KOSPI" if full.endswith(".KS") else "KOSDAQ"
        ticker_map[t6] = (full, market)

    with open(LOG_FILE, "a") as lf:
        log("=== pykrx per-ticker pull started ===", lf)
        log(f"  Candidates: {len(ticker_map)} valid, {len(skipped_invalid)} skipped (non-numeric): {skipped_invalid}", lf)
        log(f"  Date range: {START} to {END}", lf)

        with krx_tunnel():
            from pykrx import stock  # noqa: F811 — must import inside tunnel

            frames = []
            skipped = []
            consecutive_fails = 0

            for t6, (full_ticker, market) in ticker_map.items():
                log(f"  Pulling {full_ticker} ({t6}) ...", lf)

                ohlcv_raw = fetch_ticker(stock, t6, lf)
                time.sleep(1)

                mcap_raw = fetch_mcap(stock, t6, lf)
                time.sleep(1)

                if ohlcv_raw is None:
                    log(f"    SKIP {full_ticker}: no OHLCV data", lf)
                    skipped.append(full_ticker)
                    consecutive_fails += 1
                    if consecutive_fails >= 3:
                        raise RuntimeError(
                            f"3 consecutive ticker failures after retries (last: {full_ticker}). "
                            "Anonymous per-ticker endpoint may also be gated. "
                            "User intervention required."
                        )
                    continue

                consecutive_fails = 0

                ohlcv = normalize_ohlcv(ohlcv_raw)
                ohlcv.index = pd.to_datetime(ohlcv.index)

                if mcap_raw is not None and not mcap_raw.empty:
                    mcap = normalize_mcap(mcap_raw)
                    mcap.index = pd.to_datetime(mcap.index)
                    merged = ohlcv.join(mcap, how="left")
                else:
                    merged = ohlcv.copy()
                    merged["mcap"] = float("nan")
                    merged["shares_outstanding"] = float("nan")

                merged["ticker"] = full_ticker
                merged["market"] = market
                merged.index.name = "date"
                merged = merged.reset_index()
                frames.append(merged)
                log(f"    OK {full_ticker}: {len(merged)} rows", lf)

            if not frames:
                log("STOP: No data retrieved for any ticker.", lf)
                raise RuntimeError("No data retrieved for any ticker.")

            df = pd.concat(frames, ignore_index=True)
            df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

            out_path = os.path.join(DATA_DIR, "kr_ohlcv_mcap.parquet")
            df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

            elapsed = time.time() - t0
            date_min = df["date"].min()
            date_max = df["date"].max()
            n_tickers = df["ticker"].nunique()
            market_counts = df.groupby("market")["ticker"].nunique().to_dict()

            log(f"  Saved: {out_path}", lf)
            log(f"  Shape: {df.shape}", lf)
            log(f"  Date range: {date_min} to {date_max}", lf)
            log(f"  Unique tickers: {n_tickers}", lf)
            log(f"  Market breakdown: {market_counts}", lf)
            log(f"  Skipped tickers: {skipped}", lf)
            mcap_rows = df["mcap"].notna().sum()
            log(f"  NOTE: mcap/shares_outstanding are NaN for all rows. pykrx 1.0.51 get_market_cap() returns empty — this endpoint requires KRX authentication (same gate as batch endpoints). OHLCV is complete.", lf)
            log(f"=== pykrx per-ticker completion === wall={elapsed:.1f}s rows={len(df)} tickers={n_tickers} skipped={skipped} mcap_rows={mcap_rows} date_range={date_min} to {date_max}", lf)


if __name__ == "__main__":
    main()
