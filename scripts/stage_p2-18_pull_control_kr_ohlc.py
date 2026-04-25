# ABOUTME: Stage P2-18 — pulls daily OHLCV for 38 control_kr tickers via pykrx per-ticker endpoint.
# ABOUTME: Input: hardcoded list of 38 control_kr KRX tickers. Output: data/pykrx/kr_daily/kr_control_ohlcv.parquet.
#           Run: source .venv/bin/activate && python3 scripts/stage_p2-18_pull_control_kr_ohlc.py

import os
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from krx_tunnel import krx_tunnel  # noqa: E402

PROJECT_ROOT = "/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity_p2"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "pykrx", "kr_daily")
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "stage_p2-18_env.log")
OUT_PATH = os.path.join(DATA_DIR, "kr_control_ohlcv.parquet")

# Full analysis window — pull in annual chunks (KRX 2-year window limit)
START_YEAR = 2019
END_YEAR   = 2026
END_DATE   = "20260430"

CONTROL_KR_TICKERS = [
    "000990.KS", "002790.KS", "003490.KS", "005380.KS", "009150.KS",
    "012330.KS", "020150.KQ", "024110.KS", "030000.KS", "030520.KQ",
    "032640.KQ", "034020.KS", "034220.KS", "035900.KQ", "036460.KQ",
    "036540.KQ", "041590.KQ", "047810.KS", "052790.KQ", "053280.KQ",
    "053800.KQ", "060150.KQ", "067000.KQ", "069960.KQ", "078890.KQ",
    "079370.KQ", "083930.KQ", "086790.KS", "101000.KQ", "114090.KQ",
    "122870.KQ", "138930.KS", "180640.KQ", "192250.KQ", "194700.KQ",
    "215600.KQ", "218410.KS", "259960.KS",
]

RENAME_OHLCV = {
    "시가": "open",
    "고가": "high",
    "저가": "low",
    "종가": "close",
    "거래량": "volume",
    "등락률": "pct_change",
}

os.makedirs(DATA_DIR, exist_ok=True)


def log(msg, lf=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if lf:
        lf.write(line + "\n")
        lf.flush()


def annual_chunks():
    """Yield (fromdate, todate) string pairs in annual segments."""
    for year in range(START_YEAR, END_YEAR + 1):
        fd = f"{year}0101"
        if year == END_YEAR:
            td = END_DATE
        else:
            td = f"{year}1231"
        yield fd, td


def fetch_ticker_chunked(stock, t6, lf, max_retries=2):
    """Fetch OHLCV for one ticker in annual chunks. Returns combined DataFrame or None."""
    frames = []
    for fd, td in annual_chunks():
        for attempt in range(max_retries + 1):
            try:
                chunk = stock.get_market_ohlcv(fromdate=fd, todate=td, ticker=t6)
                if chunk is not None and not chunk.empty:
                    frames.append(chunk)
                break
            except Exception as e:
                if attempt < max_retries:
                    wait = 2 ** (attempt + 1)
                    log(f"    retry {attempt+1} for {t6} [{fd}-{td}]: {e} — sleeping {wait}s", lf)
                    time.sleep(wait)
                else:
                    log(f"    FAIL {t6} [{fd}-{td}] after {max_retries} retries: {e}", lf)
        time.sleep(1)  # 1-2 second sleep between chunk calls to avoid throttling

    if not frames:
        return None
    return pd.concat(frames)


def normalize_ohlcv(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=RENAME_OHLCV)
    keep = [c for c in ["open", "high", "low", "close", "volume", "pct_change"] if c in df.columns]
    return df[keep]


def main():
    t0 = time.time()

    with open(LOG_FILE, "a") as lf:
        log("=" * 70, lf)
        log("Stage P2-18: control_kr pykrx OHLCV pull — started", lf)

        # Log pykrx version
        try:
            import pykrx as _pk
            log(f"pykrx version: {_pk.__version__}", lf)
        except Exception:
            log("pykrx version: unknown", lf)

        log(f"Tickers to pull: {len(CONTROL_KR_TICKERS)}", lf)
        log(f"Date range: {START_YEAR}-01-01 to {END_DATE}", lf)

        with krx_tunnel() as _:
            log("Tunnel OK — routing through KR confirmed", lf)
            from pykrx import stock  # noqa: F811 — must import inside tunnel

            frames = []
            skipped = []

            for full_ticker in CONTROL_KR_TICKERS:
                t6 = full_ticker.split(".")[0]
                market = "KOSPI" if full_ticker.endswith(".KS") else "KOSDAQ"
                log(f"  Pulling {full_ticker} ({t6}) ...", lf)

                raw = fetch_ticker_chunked(stock, t6, lf)
                time.sleep(2)  # 1-2 second sleep between ticker calls to avoid throttling

                if raw is None or raw.empty:
                    log(f"    SKIP {full_ticker}: no data returned (delisted / unavailable)", lf)
                    skipped.append(full_ticker)
                    continue

                ohlcv = normalize_ohlcv(raw)
                ohlcv.index = pd.to_datetime(ohlcv.index)
                # Dedupe index (annual chunks may overlap on year boundaries)
                ohlcv = ohlcv[~ohlcv.index.duplicated(keep="first")]
                ohlcv = ohlcv.sort_index()

                ohlcv["ticker"] = full_ticker
                ohlcv["market"] = market
                ohlcv.index.name = "date"
                ohlcv = ohlcv.reset_index()

                date_min = ohlcv["date"].min().date()
                date_max = ohlcv["date"].max().date()
                log(f"    OK {full_ticker}: {len(ohlcv)} rows  [{date_min} to {date_max}]", lf)
                frames.append(ohlcv)

            if not frames:
                log("STOP: no data retrieved for any control_kr ticker.", lf)
                raise RuntimeError("No data retrieved for any control_kr ticker.")

            df = pd.concat(frames, ignore_index=True)
            df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

            # Enforce schema: date as datetime64, numeric columns
            df["date"] = pd.to_datetime(df["date"])
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Final column order matching kr_ohlcv_mcap.parquet as closely as possible
            col_order = ["date", "ticker", "open", "high", "low", "close", "volume", "market"]
            if "pct_change" in df.columns:
                col_order.append("pct_change")
            df = df[[c for c in col_order if c in df.columns]]

            df.to_parquet(OUT_PATH, engine="pyarrow", compression="snappy", index=False)

            elapsed = time.time() - t0
            n_tickers = df["ticker"].nunique()
            n_rows = len(df)
            date_min = df["date"].min()
            date_max = df["date"].max()

            log(f"  Saved: {OUT_PATH}", lf)
            log(f"  Shape: {df.shape}", lf)
            log(f"  Unique tickers: {n_tickers}", lf)
            log(f"  Date range: {date_min} to {date_max}", lf)
            log(f"  Skipped ({len(skipped)}): {skipped}", lf)
            log(f"  Verification: n_tickers={n_tickers} >= 38? {n_tickers >= 38}, total_rows={n_rows} > 50000? {n_rows > 50000}", lf)
            log(f"=== stage_p2-18 complete === wall={elapsed:.1f}s rows={n_rows} tickers={n_tickers} skipped={len(skipped)}", lf)


if __name__ == "__main__":
    main()
