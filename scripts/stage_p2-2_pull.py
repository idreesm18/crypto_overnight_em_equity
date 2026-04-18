# ABOUTME: Pulls Pass 2 additional data: HSI, KOSPI, S&P 500 OHLCV, BTC supply schedule, S&P mcap proxy.
# ABOUTME: Inputs: none (network); Outputs: data/yfinance/index/, data/yfinance/sp500/, data/derived/. Run: source .venv/bin/activate && python scripts/stage_p2-2_pull.py

import os
import sys
import time
from datetime import datetime, date, timedelta

import pandas as pd
import yfinance as yf

PROJECT_ROOT = "/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity"
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "stage_p2-2_pull.log")

START_DATE = "2018-09-01"
END_DATE_EXCL = "2026-04-18"   # yfinance end is exclusive; today is 2026-04-17
END_DATE_INCL = "2026-04-17"

# Bitcoin halving schedule (block reward changes)
# Each epoch: (start_date, reward_btc_per_block)
# ~144 blocks/day is the target
BLOCKS_PER_DAY = 144
BTC_EPOCHS = [
    (date(2016, 7, 9),  12.5),   # post-2016 halving
    (date(2020, 5, 11),  6.25),  # post-2020 halving
    (date(2024, 4, 19),  3.125), # post-2024 halving
]
# Reference anchor: 2024-01-01 ~ 19,600,000 BTC
BTC_ANCHOR_DATE = date(2024, 1, 1)
BTC_ANCHOR_SUPPLY = 19_600_000.0

# S&P 500 mcap reference: Siblis Research, 2024-12-31 total market cap ~$52.2 trillion
SP500_REF_DATE = "2024-12-31"
SP500_REF_MCAP = 52.2e12


def log(msg, lf=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if lf:
        lf.write(line + "\n")
        lf.flush()


def pull_yfinance_ohlcv(ticker_sym, lf):
    """Pull daily OHLCV via yfinance. Returns DataFrame with date, open, high, low, close, volume, adj_close."""
    ticker = yf.Ticker(ticker_sym)
    df = ticker.history(start=START_DATE, end=END_DATE_EXCL, interval="1d", auto_adjust=True)

    if df is None or len(df) == 0:
        return None

    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # date column normalization
    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    col_map = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
    keep = ["date"]
    for src, dst in col_map.items():
        if src in df.columns:
            df[dst] = df[src]
            keep.append(dst)
    df["adj_close"] = df["close"]
    keep.append("adj_close")

    out = df[keep].copy()
    out = out[(out["date"] >= START_DATE) & (out["date"] <= END_DATE_INCL)].copy()
    out = out.sort_values("date").reset_index(drop=True)
    return out


def build_btc_supply(lf):
    """Build daily BTC circulating supply from issuance schedule anchored to 2024-01-01."""
    start = date(2018, 9, 1)
    end = date(2026, 4, 17)

    # Generate all calendar dates
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)

    # Compute daily issuance for each epoch
    def reward_for_date(dt):
        reward = 12.5  # default pre-2020 epoch (covers 2018-09-01 through 2020-05-10)
        for epoch_start, epoch_reward in BTC_EPOCHS:
            if dt >= epoch_start:
                reward = epoch_reward
        return reward

    daily_issuance = {d: reward_for_date(d) * BLOCKS_PER_DAY for d in days}

    # Anchor: supply[2024-01-01] = 19,600,000
    # Fill forward from anchor
    supply = {}
    supply[BTC_ANCHOR_DATE] = BTC_ANCHOR_SUPPLY

    # Forward fill
    idx = days.index(BTC_ANCHOR_DATE)
    for i in range(idx + 1, len(days)):
        d = days[i]
        supply[d] = supply[days[i - 1]] + daily_issuance[d]

    # Backward fill
    for i in range(idx - 1, -1, -1):
        d = days[i]
        supply[d] = supply[days[i + 1]] - daily_issuance[days[i + 1]]

    df = pd.DataFrame({"date": pd.to_datetime(list(supply.keys())),
                       "btc_supply": list(supply.values())})
    df = df.sort_values("date").reset_index(drop=True)
    return df


def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "data", "yfinance", "index"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "data", "yfinance", "sp500"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "data", "derived"), exist_ok=True)

    t0 = time.time()

    with open(LOG_FILE, "w") as lf:
        log("=== Stage P2-2 additional data pull started ===", lf)
        log(f"Period: {START_DATE} through {END_DATE_INCL}", lf)

        # ── 1. HSI ──────────────────────────────────────────────────────────────
        log("Pulling ^HSI (Hang Seng Index)...", lf)
        hsi = pull_yfinance_ohlcv("^HSI", lf)
        if hsi is None or len(hsi) == 0:
            log("BLOCK: ^HSI returned empty dataframe. Network issue.", lf)
            sys.exit(1)

        hsi_path = os.path.join(PROJECT_ROOT, "data", "yfinance", "index", "hsi_daily.parquet")
        hsi.to_parquet(hsi_path, engine="pyarrow", compression="snappy", index=False)
        log(f"  ^HSI: {len(hsi)} rows, {hsi['date'].min().date()} to {hsi['date'].max().date()}", lf)

        # Cross-check HSI on known trading dates
        hsi_checks = {
            "2020-01-02": None,
            "2021-12-31": None,
            "2023-01-03": None,
            "2025-01-02": None,
        }
        log("  ^HSI spot-checks (manual review):", lf)
        for ref_date in hsi_checks:
            row = hsi[hsi["date"] == ref_date]
            if len(row) == 0:
                log(f"    {ref_date}: not in data (weekend/holiday)", lf)
            else:
                log(f"    {ref_date}: close={float(row['close'].iloc[0]):,.1f}", lf)

        nan_hsi = hsi["close"].isna().sum()
        if nan_hsi > 0:
            log(f"  WARN: ^HSI has {nan_hsi} NaN close values", lf)

        # ── 2. KOSPI ────────────────────────────────────────────────────────────
        log("Pulling ^KS11 (KOSPI Composite)...", lf)
        kospi = pull_yfinance_ohlcv("^KS11", lf)
        if kospi is None or len(kospi) == 0:
            log("BLOCK: ^KS11 returned empty dataframe. Network issue.", lf)
            sys.exit(1)

        kospi_path = os.path.join(PROJECT_ROOT, "data", "yfinance", "index", "kospi_daily.parquet")
        kospi.to_parquet(kospi_path, engine="pyarrow", compression="snappy", index=False)
        log(f"  ^KS11: {len(kospi)} rows, {kospi['date'].min().date()} to {kospi['date'].max().date()}", lf)

        # Cross-check KOSPI
        log("  ^KS11 spot-checks (manual review):", lf)
        for ref_date in ["2020-01-02", "2021-12-30", "2023-01-02", "2025-01-02"]:
            row = kospi[kospi["date"] == ref_date]
            if len(row) == 0:
                log(f"    {ref_date}: not in data (weekend/holiday)", lf)
            else:
                log(f"    {ref_date}: close={float(row['close'].iloc[0]):,.2f}", lf)

        nan_kospi = kospi["close"].isna().sum()
        if nan_kospi > 0:
            log(f"  WARN: ^KS11 has {nan_kospi} NaN close values", lf)

        # ── 3. S&P 500 ──────────────────────────────────────────────────────────
        log("Pulling ^GSPC (S&P 500)...", lf)
        gspc = pull_yfinance_ohlcv("^GSPC", lf)
        if gspc is None or len(gspc) == 0:
            log("BLOCK: ^GSPC returned empty dataframe. Network issue.", lf)
            sys.exit(1)

        gspc_path = os.path.join(PROJECT_ROOT, "data", "yfinance", "sp500", "gspc_daily.parquet")
        gspc.to_parquet(gspc_path, engine="pyarrow", compression="snappy", index=False)
        log(f"  ^GSPC: {len(gspc)} rows, {gspc['date'].min().date()} to {gspc['date'].max().date()}", lf)

        nan_gspc = gspc["close"].isna().sum()
        if nan_gspc > 0:
            log(f"  WARN: ^GSPC has {nan_gspc} NaN close values", lf)

        # ── 4. BTC circulating supply ────────────────────────────────────────────
        log("Building BTC circulating supply from issuance schedule...", lf)
        btc_supply = build_btc_supply(lf)

        # Validate: monotonically non-decreasing
        diffs = btc_supply["btc_supply"].diff().dropna()
        if (diffs < 0).any():
            log("BLOCK: BTC supply is non-monotonic (has decreases). Calculation error.", lf)
            sys.exit(1)
        if (btc_supply["btc_supply"] < 0).any():
            log("BLOCK: BTC supply has negative values.", lf)
            sys.exit(1)

        btc_path = os.path.join(PROJECT_ROOT, "data", "derived", "btc_supply.parquet")
        btc_supply.to_parquet(btc_path, engine="pyarrow", compression="snappy", index=False)
        log(f"  BTC supply: {len(btc_supply)} rows, {btc_supply['date'].min().date()} to {btc_supply['date'].max().date()}", lf)

        # Spot-check reference dates
        btc_refs = ["2018-09-01", "2020-05-10", "2020-05-11", "2024-01-01", "2024-04-18", "2026-04-17"]
        log("  BTC supply spot-checks:", lf)
        for ref_date in btc_refs:
            row = btc_supply[btc_supply["date"] == ref_date]
            if len(row) == 0:
                log(f"    {ref_date}: not found", lf)
            else:
                log(f"    {ref_date}: {float(row['btc_supply'].iloc[0]):,.0f} BTC", lf)

        # ── 5. S&P 500 market-cap proxy ─────────────────────────────────────────
        log("Building S&P 500 market-cap proxy...", lf)

        # Get reference close for 2024-12-31 from pulled gspc data
        ref_row = gspc[gspc["date"] == SP500_REF_DATE]
        if len(ref_row) == 0:
            log("  WARN: 2024-12-31 not in ^GSPC data; using hardcoded close 5881.63", lf)
            ref_close = 5881.63
        else:
            ref_close = float(ref_row["close"].iloc[0])
            log(f"  Reference close from pulled data ({SP500_REF_DATE}): {ref_close:.2f}", lf)

        scaling_factor = SP500_REF_MCAP / ref_close
        log(f"  Scaling factor: {SP500_REF_MCAP:.3e} / {ref_close:.2f} = {scaling_factor:.6e}", lf)

        sp500_mcap = gspc[["date", "close"]].copy()
        sp500_mcap = sp500_mcap.rename(columns={"close": "sp500_close"})
        sp500_mcap["sp500_mcap_proxy"] = sp500_mcap["sp500_close"] * scaling_factor

        # Validate: positive everywhere
        if (sp500_mcap["sp500_mcap_proxy"] <= 0).any():
            log("BLOCK: S&P 500 mcap proxy has non-positive values.", lf)
            sys.exit(1)

        mcap_path = os.path.join(PROJECT_ROOT, "data", "derived", "sp500_mcap_proxy.parquet")
        sp500_mcap.to_parquet(mcap_path, engine="pyarrow", compression="snappy", index=False)
        log(f"  S&P mcap proxy: {len(sp500_mcap)} rows, {sp500_mcap['date'].min().date()} to {sp500_mcap['date'].max().date()}", lf)

        # Reference date check
        log(f"  S&P mcap on {SP500_REF_DATE}: {sp500_mcap[sp500_mcap['date'] == SP500_REF_DATE]['sp500_mcap_proxy'].values}", lf)

        # ── 6. Skip logs ─────────────────────────────────────────────────────────
        skip_msg_krx = "SKIP: pykrx mcap restoration — KRX_ID/KRX_PW absent (logged in feature_restoration_decisions.log)."
        skip_msg_liq = "SKIP: liquidations pull — no provider key (COINGLASS/LAEVITAS/CCDATA all absent; logged in feature_restoration_decisions.log)."
        log(skip_msg_krx, lf)
        log(skip_msg_liq, lf)

        elapsed = time.time() - t0
        log(f"=== Stage P2-2 pull complete: {elapsed:.1f}s ===", lf)

        # ── Summary block ────────────────────────────────────────────────────────
        log("", lf)
        log("=== SUMMARY ===", lf)
        log(f"hsi_daily.parquet:       {len(hsi)} rows, {hsi['date'].min().date()} to {hsi['date'].max().date()}", lf)
        log(f"kospi_daily.parquet:     {len(kospi)} rows, {kospi['date'].min().date()} to {kospi['date'].max().date()}", lf)
        log(f"gspc_daily.parquet:      {len(gspc)} rows, {gspc['date'].min().date()} to {gspc['date'].max().date()}", lf)
        log(f"btc_supply.parquet:      {len(btc_supply)} rows, {btc_supply['date'].min().date()} to {btc_supply['date'].max().date()}", lf)
        log(f"sp500_mcap_proxy.parquet:{len(sp500_mcap)} rows, {sp500_mcap['date'].min().date()} to {sp500_mcap['date'].max().date()}", lf)
        log(f"S&P 500 scaling factor:  {scaling_factor:.6e}  (reference: {SP500_REF_DATE} close={ref_close:.2f}, mcap=$52.2T)", lf)

        # BTC key reference dates
        for ref_date in ["2024-01-01", "2026-04-17"]:
            row = btc_supply[btc_supply["date"] == ref_date]
            if len(row):
                log(f"BTC supply {ref_date}:    {float(row['btc_supply'].iloc[0]):,.0f} BTC", lf)

        ref_row_dec31 = sp500_mcap[sp500_mcap["date"] == "2024-12-31"]
        if len(ref_row_dec31):
            log(f"S&P mcap proxy 2024-12-31: ${float(ref_row_dec31['sp500_mcap_proxy'].iloc[0]):.3e}", lf)

        log("Skipped: pykrx mcap (KRX creds absent), liquidations (no provider key).", lf)


if __name__ == "__main__":
    main()
