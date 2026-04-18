# ABOUTME: Stage P2-5 — Track A index-level feature engineering for HSI and KOSPI.
# ABOUTME: Inputs: data/yfinance/index/{hsi,kospi}_daily.parquet, data/binance/ (BTC/ETH/SOL 1m), data/fred/.
# ABOUTME: Outputs: output/features_track_a_index.parquet, logs/stage_p2-5_index_features.log.
# Run: source .venv/bin/activate && python3 scripts/stage_p2-5_index_features.py

import sys
import logging
import warnings
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path("/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity_p2")
DATA = ROOT / "data"
OUT  = ROOT / "output"
LOGS = ROOT / "logs"
OUT.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

LOG_FILE = LOGS / "stage_p2-5_index_features.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_START = "2019-01-01"
WINDOW_END   = "2026-04-17"
MIN_PER_YEAR = 525_600

SPOT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
PERP_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

CRYPTO_FEAT_COLS = [
    "btc_ov_log_return", "eth_ov_log_return",
    "btc_ov_realized_vol", "eth_ov_realized_vol",
    "btc_ov_max_drawdown", "btc_ov_volume_usd", "btc_ov_volume_surge",
    "btc_ov_taker_imbalance", "crosspair_dispersion", "btc_eth_spread",
    "btc_funding_rate_latest", "btc_funding_rate_delta",
]
MACRO_FEAT_COLS = [
    "vix_level", "vix_5d_change", "yield_curve_slope",
    "dxy_level", "dxy_5d_change", "breakeven_5y",
]
TARGET_COLS = ["gap_return", "intraday_return", "cc_return"]


def stop(msg: str) -> None:
    log.error(f"[BLOCK] {msg}")
    sys.exit(1)


# ── Load Binance 1m klines ────────────────────────────────────────────────────
def load_spot_klines() -> dict:
    klines = {}
    for sym in SPOT_SYMBOLS:
        p = DATA / "binance" / "spot_klines" / f"{sym}_1m.parquet"
        df = pd.read_parquet(p).set_index("open_time").sort_index()
        klines[sym] = df
        log.info(f"Spot {sym}: {len(df):,} rows, {df.index.min()} to {df.index.max()}")
    return klines


def load_funding_rates() -> dict:
    rates = {}
    for sym in PERP_SYMBOLS:
        p = DATA / "binance" / "funding_rates" / f"{sym}_funding.parquet"
        df = pd.read_parquet(p).sort_values("funding_time").reset_index(drop=True)
        df["funding_time"] = pd.to_datetime(df["funding_time"], utc=True)
        rates[sym] = df
        log.info(f"Funding {sym}: {len(df):,} rows")
    return rates


# ── Load FRED macro ────────────────────────────────────────────────────────────
def load_fred() -> pd.DataFrame:
    series_map = {
        "VIXCLS":   "vix",
        "DTWEXBGS": "dxy",
        "DGS10":    "dgs10",
        "DGS2":     "dgs2",
        "T5YIE":    "t5yie",
        "T10YIE":   "t10yie",
    }
    dfs = []
    for fname, col in series_map.items():
        df = pd.read_parquet(DATA / "fred" / f"{fname}.parquet")
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.rename(columns={"value": col}).set_index("date")
        dfs.append(df[[col]])
    fred = pd.concat(dfs, axis=1).sort_index()
    fred = fred.ffill(limit=7)
    fred["yield_curve_slope"] = fred["dgs10"] - fred["dgs2"]
    fred["vix_5d_change"]     = fred["vix"] - fred["vix"].shift(5)
    fred["dxy_5d_change"]     = fred["dxy"] - fred["dxy"].shift(5)
    log.info(f"FRED: {len(fred)} rows, {fred.index.min()} to {fred.index.max()}")
    return fred


# ── Load index daily OHLCV ────────────────────────────────────────────────────
def load_index(fname: str) -> pd.DataFrame:
    """Load yfinance index parquet; index on date (tz-naive datetime)."""
    df = pd.read_parquet(DATA / "yfinance" / "index" / fname)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date").set_index("date")
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.astype(float)
    return df


# ── Overnight windows ─────────────────────────────────────────────────────────
def build_overnight_windows(market: str) -> pd.DataFrame:
    """
    For each trading session T, compute window_start=close(T), window_end=open(T+1).
    Returns columns: date, next_date, window_start_utc, window_end_utc, is_weekend_gap.
    """
    cal_name = "XHKG" if market == "HK" else "XKRX"
    cal = xcals.get_calendar(cal_name)
    sessions = cal.sessions_in_range(WINDOW_START, WINDOW_END)
    rows = []
    for i, sess in enumerate(sessions[:-1]):
        try:
            w_start = cal.session_close(sess)
        except Exception:
            continue
        next_sess = sessions[i + 1]
        try:
            w_end = cal.session_open(next_sess)
        except Exception:
            continue
        gap_days = (next_sess - sess).days
        rows.append({
            "date":              sess.date(),
            "next_date":         next_sess.date(),
            "window_start_utc":  w_start,
            "window_end_utc":    w_end,
            "is_weekend_gap":    gap_days > 3,
            "gap_calendar_days": gap_days,
        })
    df = pd.DataFrame(rows)
    log.info(f"{market}: {len(df)} overnight windows, {df['is_weekend_gap'].sum()} weekend gaps")
    return df


# ── Crypto feature helpers ────────────────────────────────────────────────────
def slice_klines(klines_df: pd.DataFrame, w_start, w_end) -> pd.DataFrame:
    mask = (klines_df.index >= w_start) & (klines_df.index < w_end)
    return klines_df.loc[mask]


def log_return_from_endpoints(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return np.nan
    return float(np.log(df["close"].iloc[-1] / df["open"].iloc[0]))


def realized_vol(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return np.nan
    r = np.log(df["close"] / df["close"].shift(1)).dropna()
    return float(r.std() * np.sqrt(MIN_PER_YEAR)) if len(r) > 0 else np.nan


def max_drawdown(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return np.nan
    prices = df["high"].values
    peak = prices[0]
    mdd = 0.0
    for p in prices:
        if p > peak:
            peak = p
        dd = p / peak - 1.0
        if dd < mdd:
            mdd = dd
    return float(mdd)


def taker_imbalance(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return np.nan
    total = df["quote_volume"].sum()
    if total == 0:
        return np.nan
    buy = df["taker_buy_quote_volume"].sum()
    return float((buy - (total - buy)) / total)


# ── Crypto feature computation per market ─────────────────────────────────────
def compute_crypto_features(
    windows: pd.DataFrame,
    spot_klines: dict,
    funding_rates: dict,
) -> pd.DataFrame:
    n = len(windows)
    results = {col: np.full(n, np.nan) for col in CRYPTO_FEAT_COLS}

    btc_spot = spot_klines["BTCUSDT"]
    eth_spot = spot_klines["ETHUSDT"]
    sol_spot = spot_klines["SOLUSDT"]
    bnb_spot = spot_klines["BNBUSDT"]
    xrp_spot = spot_klines["XRPUSDT"]
    btc_fund = funding_rates["BTCUSDT"]

    # Temp arrays for dispersion/surge
    sol_rets = np.full(n, np.nan)
    bnb_rets = np.full(n, np.nan)
    xrp_rets = np.full(n, np.nan)
    btc_vols = np.full(n, np.nan)

    log.info(f"Computing crypto features for {n} windows...")
    for i, row in enumerate(windows.itertuples(index=False)):
        ws = row.window_start_utc
        we = row.window_end_utc

        btc_s = slice_klines(btc_spot, ws, we)
        if len(btc_s) > 0:
            results["btc_ov_log_return"][i]      = log_return_from_endpoints(btc_s)
            results["btc_ov_realized_vol"][i]    = realized_vol(btc_s)
            results["btc_ov_max_drawdown"][i]    = max_drawdown(btc_s)
            results["btc_ov_volume_usd"][i]      = float(btc_s["quote_volume"].sum())
            results["btc_ov_taker_imbalance"][i] = taker_imbalance(btc_s)
            btc_vols[i] = float(btc_s["quote_volume"].sum())

        eth_s = slice_klines(eth_spot, ws, we)
        if len(eth_s) > 0:
            results["eth_ov_log_return"][i]   = log_return_from_endpoints(eth_s)
            results["eth_ov_realized_vol"][i] = realized_vol(eth_s)

        sol_s = slice_klines(sol_spot, ws, we)
        if len(sol_s) > 0:
            sol_rets[i] = log_return_from_endpoints(sol_s)

        bnb_s = slice_klines(bnb_spot, ws, we)
        if len(bnb_s) > 0:
            bnb_rets[i] = log_return_from_endpoints(bnb_s)

        xrp_s = slice_klines(xrp_spot, ws, we)
        if len(xrp_s) > 0:
            xrp_rets[i] = log_return_from_endpoints(xrp_s)

        fund_before = btc_fund[btc_fund["funding_time"] < we]
        if len(fund_before) >= 2:
            results["btc_funding_rate_latest"][i] = float(fund_before["funding_rate"].iloc[-1])
            results["btc_funding_rate_delta"][i]  = float(
                fund_before["funding_rate"].iloc[-1] - fund_before["funding_rate"].iloc[-2]
            )
        elif len(fund_before) == 1:
            results["btc_funding_rate_latest"][i] = float(fund_before["funding_rate"].iloc[-1])

        if i % 500 == 0:
            log.info(f"  {i}/{n} windows processed")

    df = pd.DataFrame({k: v for k, v in results.items()}, index=windows.index)

    # Cross-pair dispersion
    return_mat = np.column_stack([
        df["btc_ov_log_return"].values,
        df["eth_ov_log_return"].values,
        sol_rets, bnb_rets, xrp_rets,
    ])
    df["crosspair_dispersion"] = np.nanstd(return_mat, axis=1)
    df["btc_eth_spread"]       = df["btc_ov_log_return"] - df["eth_ov_log_return"]

    # Volume surge: btc volume / rolling 7-day mean (shifted 1)
    dates = pd.to_datetime(windows.reset_index(drop=True)["date"])
    vol_series = pd.Series(btc_vols, index=dates).sort_index()
    rolling_mean = vol_series.rolling(7, min_periods=4).mean().shift(1)
    df["btc_ov_volume_surge"] = btc_vols / rolling_mean.values

    log.info(f"Crypto features computed. BTC return NaN: {np.isnan(results['btc_ov_log_return']).sum()}/{n}")
    return df


# ── Macro feature join ─────────────────────────────────────────────────────────
def attach_macro_features(windows: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    fred_df = fred.copy()
    fred_df.index = pd.to_datetime(fred_df.index)

    macro_cols = {
        "vix":               "vix_level",
        "dxy":               "dxy_level",
        "yield_curve_slope": "yield_curve_slope",
        "vix_5d_change":     "vix_5d_change",
        "dxy_5d_change":     "dxy_5d_change",
        "t5yie":             "breakeven_5y",
    }

    window_dates = pd.to_datetime(windows["date"])
    macro_rows = []
    for d in window_dates:
        lag_date = d - pd.Timedelta(days=1)
        avail = fred_df[fred_df.index <= lag_date]
        if len(avail) == 0:
            macro_rows.append({v: np.nan for v in macro_cols.values()})
        else:
            last = avail.iloc[-1]
            macro_rows.append({v: last[k] for k, v in macro_cols.items()})

    return pd.DataFrame(macro_rows, index=windows.index)


# ── Index targets ─────────────────────────────────────────────────────────────
def compute_index_targets(
    prev_date: pd.Timestamp,
    curr_date: pd.Timestamp,
    index_df: pd.DataFrame,
) -> dict:
    """
    gap_return:      log(open_curr / close_prev)
    intraday_return: log(close_curr / open_curr)
    cc_return:       log(close_curr / close_prev)
    """
    empty = {"gap_return": np.nan, "intraday_return": np.nan, "cc_return": np.nan}

    if curr_date not in index_df.index:
        return empty

    curr_row = index_df.loc[curr_date]

    # Resolve prev_date: use as-is if present, else nearest prior row
    if prev_date in index_df.index:
        prev_row = index_df.loc[prev_date]
    else:
        before = index_df[index_df.index < curr_date]
        if len(before) == 0:
            return empty
        prev_row = before.iloc[-1]

    prev_close = prev_row["close"]
    curr_open  = curr_row["open"]
    curr_close = curr_row["close"]

    if any(v <= 0 or np.isnan(v) for v in [prev_close, curr_open, curr_close]):
        return empty

    return {
        "gap_return":      float(np.log(curr_open  / prev_close)),
        "intraday_return": float(np.log(curr_close / curr_open)),
        "cc_return":       float(np.log(curr_close / prev_close)),
    }


# ── Build index feature rows ───────────────────────────────────────────────────
def build_index_features(
    market: str,
    ticker: str,
    windows: pd.DataFrame,
    crypto_feat: pd.DataFrame,
    macro_feat: pd.DataFrame,
    index_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    n = len(windows)
    log.info(f"{market} ({ticker}): assembling {n} rows...")

    for i, (_, wrow) in enumerate(windows.iterrows()):
        date      = pd.Timestamp(wrow["date"])
        next_date = pd.Timestamp(wrow["next_date"])

        cf  = crypto_feat.iloc[i].to_dict()
        mf  = macro_feat.iloc[i].to_dict()
        tgt = compute_index_targets(date, next_date, index_df)

        row = {
            "date":           next_date.normalize(),
            "ticker":         ticker,
            "is_weekend_gap": wrow["is_weekend_gap"],
            **cf,
            **mf,
            **tgt,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    log.info(f"{market}: {len(df):,} rows assembled")
    return df


# ── Validation ─────────────────────────────────────────────────────────────────
def validate(df: pd.DataFrame) -> bool:
    log.info("\n" + "=" * 60)
    log.info("VALIDATION")
    log.info("=" * 60)

    ok = True

    # Row counts per ticker
    for tkr, grp in df.groupby("ticker"):
        log.info(f"  {tkr}: {len(grp):,} rows, dates {grp['date'].min().date()} to {grp['date'].max().date()}")

    total_rows = len(df)
    log.info(f"  Total rows: {total_rows:,}")

    # BLOCK: < 1800 total rows
    if total_rows < 1800:
        stop(f"BLOCK: only {total_rows} rows total (expected >= 1800)")

    # Ticker values
    tickers_found = set(df["ticker"].unique())
    expected = {"^HSI", "^KS11"}
    if tickers_found != expected:
        log.error(f"VALIDATION FAIL: tickers {tickers_found} != expected {expected}")
        ok = False

    # Feature count: expect exactly 18 (12 crypto + 6 macro)
    feat_cols = [c for c in df.columns if c in CRYPTO_FEAT_COLS + MACRO_FEAT_COLS]
    log.info(f"  Feature columns found: {len(feat_cols)} (expected 18)")
    if len(feat_cols) != 18:
        log.error(f"VALIDATION FAIL: {len(feat_cols)} feature columns (expected 18)")
        ok = False

    # No stock-level feature columns
    stock_cols = [c for c in df.columns if c.startswith("stock_")]
    if stock_cols:
        log.error(f"VALIDATION FAIL: stock-level columns present: {stock_cols}")
        ok = False
    else:
        log.info("  No stock-level feature columns: OK")

    # Crypto feature NaN check
    PERP_COLS = {"btc_funding_rate_latest", "btc_funding_rate_delta"}
    log.info("\nFeature NaN rates:")
    for col in CRYPTO_FEAT_COLS + MACRO_FEAT_COLS:
        if col not in df.columns:
            log.warning(f"  MISSING: {col}")
            continue
        nan_pct = df[col].isna().mean() * 100
        is_perp = col in PERP_COLS
        flag = " [perp — NaN expected pre-2020]" if is_perp else ""
        log.info(f"  {col:35s}: NaN={nan_pct:.1f}%{flag}")
        # BLOCK: > 30% NaN in crypto features (non-perp)
        if col in CRYPTO_FEAT_COLS and not is_perp and nan_pct > 30:
            stop(f"BLOCK: crypto feature {col} has {nan_pct:.1f}% NaN (>30% threshold)")

    # Target NaN check
    log.info("\nTarget NaN rates:")
    for tgt in TARGET_COLS:
        if tgt not in df.columns:
            log.warning(f"  MISSING TARGET: {tgt}")
            ok = False
            continue
        nan_pct = df[tgt].isna().mean() * 100
        log.info(f"  {tgt}: NaN={nan_pct:.1f}%")
        # BLOCK: > 10% NaN in targets
        if nan_pct > 10:
            stop(f"BLOCK: target {tgt} has {nan_pct:.1f}% NaN (>10% threshold)")

    # Outlier check (>10 sigma)
    log.info("\nOutlier check (>10 sigma):")
    for col in CRYPTO_FEAT_COLS + MACRO_FEAT_COLS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 10:
            continue
        mu, sigma = s.mean(), s.std()
        if sigma > 0:
            outliers = (np.abs(s - mu) > 10 * sigma).sum()
            if outliers > 0:
                log.warning(f"  {col}: {outliers} values > 10 sigma (mu={mu:.4f}, sigma={sigma:.4f})")

    return ok


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("Stage P2-5: Index Feature Engineering — Track A")
    log.info(f"Window: {WINDOW_START} to {WINDOW_END}")

    # ── Load Binance data ──
    log.info("\nLoading Binance spot klines...")
    spot_klines   = load_spot_klines()
    log.info("Loading funding rates...")
    funding_rates = load_funding_rates()

    # ── Load FRED ──
    log.info("\nLoading FRED macro...")
    fred = load_fred()

    # ── Load index daily OHLCV ──
    log.info("\nLoading index daily OHLCV...")
    hsi_df   = load_index("hsi_daily.parquet")
    kospi_df = load_index("kospi_daily.parquet")
    log.info(f"HSI:   {len(hsi_df)} rows, {hsi_df.index.min().date()} to {hsi_df.index.max().date()}")
    log.info(f"KOSPI: {len(kospi_df)} rows, {kospi_df.index.min().date()} to {kospi_df.index.max().date()}")

    # ── Build overnight windows ──
    log.info("\nBuilding overnight windows...")
    hk_windows = build_overnight_windows("HK")
    kr_windows = build_overnight_windows("KR")

    # ── Compute crypto features per market ──
    log.info("\nComputing HK crypto features...")
    hk_crypto = compute_crypto_features(hk_windows, spot_klines, funding_rates)

    log.info("\nComputing KR crypto features...")
    kr_crypto = compute_crypto_features(kr_windows, spot_klines, funding_rates)

    # ── Macro features ──
    log.info("\nAttaching macro features...")
    hk_macro = attach_macro_features(hk_windows, fred)
    kr_macro = attach_macro_features(kr_windows, fred)

    # ── Build index feature tables ──
    log.info("\nBuilding HSI feature table...")
    hsi_feat = build_index_features("HK", "^HSI",  hk_windows, hk_crypto, hk_macro, hsi_df)

    log.info("\nBuilding KOSPI feature table...")
    kospi_feat = build_index_features("KR", "^KS11", kr_windows, kr_crypto, kr_macro, kospi_df)

    # ── Combine ──
    combined = pd.concat([hsi_feat, kospi_feat], ignore_index=True)
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Ensure column order: date, ticker, [features], [targets]
    meta_cols  = ["date", "ticker", "is_weekend_gap"]
    feat_order = CRYPTO_FEAT_COLS + MACRO_FEAT_COLS
    tgt_order  = TARGET_COLS
    all_cols   = meta_cols + feat_order + tgt_order
    # Keep only columns that exist
    final_cols = [c for c in all_cols if c in combined.columns]
    combined = combined[final_cols]

    log.info(f"\nCombined shape: {combined.shape}")

    # ── Validate ──
    ok = validate(combined)
    if not ok:
        log.error("Validation failed — see log above. Halting before save.")
        sys.exit(1)

    # ── Save ──
    out_path = OUT / "features_track_a_index.parquet"
    combined.to_parquet(out_path, index=False)
    log.info(f"\nSaved: {out_path}")

    # ── Summary ──
    log.info("\n" + "=" * 70)
    log.info("STAGE P2-5 SUMMARY")
    log.info("=" * 70)
    for tkr, grp in combined.groupby("ticker"):
        log.info(f"  {tkr}: {len(grp):,} rows, {grp['date'].min().date()} to {grp['date'].max().date()}")
    log.info(f"  Total rows: {len(combined):,}")
    feat_count = len([c for c in combined.columns if c in CRYPTO_FEAT_COLS + MACRO_FEAT_COLS])
    log.info(f"  Feature columns: {feat_count} (expected 18)")
    log.info(f"  Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")

    # Rows/features dropped
    hk_w  = len(hk_windows)
    kr_w  = len(kr_windows)
    hk_out = len(combined[combined["ticker"] == "^HSI"])
    kr_out = len(combined[combined["ticker"] == "^KS11"])
    hk_drop = hk_w - hk_out
    kr_drop = kr_w - kr_out
    log.info(f"  HSI  rows dropped (no index data on next_date): {hk_drop}")
    log.info(f"  KS11 rows dropped (no index data on next_date): {kr_drop}")

    gap_nan_hsi   = combined[combined["ticker"] == "^HSI"]["gap_return"].isna().mean() * 100
    gap_nan_ks11  = combined[combined["ticker"] == "^KS11"]["gap_return"].isna().mean() * 100
    log.info(f"  gap_return NaN: ^HSI={gap_nan_hsi:.1f}%, ^KS11={gap_nan_ks11:.1f}%")
    log.info(f"  Validation: {'PASS' if ok else 'FAIL'}")
    log.info("Stage P2-5 complete.")


if __name__ == "__main__":
    main()
