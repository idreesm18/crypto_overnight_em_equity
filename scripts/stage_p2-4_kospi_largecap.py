# ABOUTME: Stage P2-4 — KOSPI large-cap universe construction and Track A feature engineering.
# ABOUTME: Inputs: data/stock_picks/crypto_candidates_kr.csv, data/yfinance/kr_control/ (cache),
#           data/binance/, data/fred/, output/universe_log.csv (KR rebalance date grid).
#           Outputs: output/kospi_largecap_universe_log.csv, output/features_track_a_kospi_largecap.parquet,
#           logs/stage_p2-4_kospi_largecap.log.
# Run: source .venv/bin/activate && python3 scripts/stage_p2-4_kospi_largecap.py

import sys
import logging
import warnings
import time
import numpy as np
import pandas as pd
import exchange_calendars as xcals
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity")
DATA = ROOT / "data"
OUT  = ROOT / "output"
LOGS = ROOT / "logs"
OUT.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

LOG_FILE = LOGS / "stage_p2-4_kospi_largecap.log"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
WINDOW_START   = "2019-01-01"
WINDOW_END     = "2026-04-15"
CORR_WINDOW    = 60    # trailing trading days for BTC correlation (match Pass 1)
ADV_WINDOW     = 20    # trailing trading days for ADV
ADV_THRESH_USD = 50_000_000   # $50M USD equivalent (P2-4 brief)
TOP_N          = 12           # select top 12 by signed BTC correlation
MIN_POOL       = 9            # flat if pool < 9 after ADV filter
KRW_USD        = 1 / 1330     # constant KRW→USD (per P2-4 brief)
MIN_PER_YEAR   = 525_600      # minutes in a year for vol scaling

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
STOCK_FEAT_COLS = ["stock_rv_20d", "stock_ret_20d", "stock_prior_day_return"]
TARGET_COLS     = ["tgt_gap", "tgt_intraday", "tgt_cc"]
PERP_COLS       = ["btc_funding_rate_latest", "btc_funding_rate_delta"]


def stop(msg: str) -> None:
    log.error(f"[BLOCK] {msg}")
    sys.exit(1)


# ── Load KR candidate tickers (.KS only) ──────────────────────────────────────
def load_ks_tickers() -> list:
    df = pd.read_csv(DATA / "stock_picks" / "crypto_candidates_kr.csv")
    tickers = df["ticker"].str.strip().tolist()
    ks_tickers = [t for t in tickers if t.upper().endswith(".KS")]
    log.info(f"KR candidate CSV: {len(tickers)} total, {len(ks_tickers)} with .KS suffix")
    return ks_tickers


# ── Load KR price history via yfinance (with cache reuse) ──────────────────────
def load_ks_prices(tickers: list) -> tuple:
    """
    Pull daily OHLCV for .KS tickers via yfinance, using kr_control cache dir.
    Returns (price_map dict keyed by ticker, load_log dict).
    """
    # Reuse the same cache dir as P2-3 to avoid redundant downloads
    cache_dir = DATA / "yfinance" / "kr_control"
    cache_dir.mkdir(parents=True, exist_ok=True)

    price_map = {}
    load_log = {"found": [], "missing": []}

    for ticker in tickers:
        cache_file = cache_dir / f"{ticker.replace('.', '_')}.parquet"

        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if len(df) > 0:
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    price_map[ticker] = df
                    load_log["found"].append(ticker)
                    log.info(f"{ticker}: loaded from cache ({len(df)} rows)")
                    continue
            except Exception:
                pass

        log.info(f"{ticker}: pulling from yfinance...")
        try:
            t = yf.Ticker(ticker)
            raw = t.history(start=WINDOW_START, end="2026-04-18", interval="1d", auto_adjust=True)
            time.sleep(0.3)

            if raw is None or len(raw) == 0:
                load_log["missing"].append(f"{ticker}: empty response")
                log.warning(f"{ticker}: empty response from yfinance")
                continue

            raw = raw.reset_index()
            raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
            if "datetime" in raw.columns:
                raw = raw.rename(columns={"datetime": "date"})
            raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None).dt.normalize()
            raw = raw[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
            raw = raw.set_index("date").sort_index().astype(float)
            raw["adv_usd_raw"] = raw["close"] * raw["volume"] * KRW_USD
            raw["ret"] = raw["close"].pct_change()

            raw.to_parquet(cache_file)
            price_map[ticker] = raw
            load_log["found"].append(ticker)
            log.info(f"{ticker}: {len(raw)} rows ({raw.index.min().date()} to {raw.index.max().date()})")
        except Exception as e:
            load_log["missing"].append(f"{ticker}: {e}")
            log.warning(f"{ticker}: {e}")

    log.info(f".KS loaded: {len(load_log['found'])}, missing: {len(load_log['missing'])}")
    return price_map, load_log


# ── Load BTC daily returns for correlation computation ─────────────────────────
def load_btc_daily() -> pd.Series:
    """Load BTC-USD daily close returns."""
    btc = pd.read_parquet(DATA / "yfinance" / "btcusd_daily.parquet")[["date", "close"]].copy()
    btc["date"] = pd.to_datetime(btc["date"])
    btc = btc.sort_values("date").set_index("date")
    btc["ret"] = btc["close"].pct_change()
    log.info(f"BTC daily: {len(btc)} rows, {btc.index.min().date()} to {btc.index.max().date()}")
    return btc["ret"].rename("btc_ret")


# ── Load rebalance dates from Pass 1 universe log ─────────────────────────────
def get_rebalance_dates() -> pd.DatetimeIndex:
    univ = pd.read_csv(OUT / "universe_log.csv")
    univ["date"] = pd.to_datetime(univ["date"])
    # Use KR market dates from Pass 1 as the grid
    dates = univ[univ["market"] == "KR"]["date"].sort_values().unique()
    log.info(f"KR rebalance dates from universe_log: {len(dates)} ({pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()})")
    return pd.DatetimeIndex(dates)


# ── KOSPI large-cap universe construction ─────────────────────────────────────
def compute_kospi_largecap_universe(price_map: dict, btc_ret: pd.Series,
                                     rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    For each rebalance date:
      1. ADV filter: trailing 20-day ADV > $50M USD
      2. If < 9 pass ADV: log FLAT, pool empty
      3. Rank by SIGNED BTC correlation (descending, matching Pass 1 convention)
      4. Select top 12
    Returns DataFrame: date, ticker, adv_usd, btc_corr_60d, rank, flat_month_flag
    """
    results = []
    all_ticker_dates = {tkr: pd.DatetimeIndex(df.index) for tkr, df in price_map.items()}

    for reb in rebal_dates:
        ticker_rows = []

        for ticker, df in price_map.items():
            # Data strictly before rebalance date
            hist = df[df.index < reb]
            if len(hist) < max(CORR_WINDOW, ADV_WINDOW):
                continue

            # Trailing 20-day ADV
            adv_series = hist["adv_usd_raw"].iloc[-ADV_WINDOW:].dropna()
            if len(adv_series) < 5:
                continue
            adv_usd = adv_series.mean()
            if adv_usd < ADV_THRESH_USD:
                continue

            # Trailing 60-day BTC correlation (signed, matching Pass 1)
            stock_window = hist.iloc[-CORR_WINDOW:]
            btc_window = btc_ret.reindex(stock_window.index).dropna()
            stock_ret_aligned = hist["ret"].reindex(btc_window.index).dropna()
            common_idx = stock_ret_aligned.index.intersection(btc_window.index)

            if len(common_idx) < 30:
                btc_corr = np.nan
            else:
                btc_corr = stock_ret_aligned.loc[common_idx].corr(btc_window.loc[common_idx])

            if pd.isna(btc_corr):
                continue

            ticker_rows.append({"ticker": ticker, "adv_usd": adv_usd, "btc_corr_60d": btc_corr})

        n_eligible = len(ticker_rows)

        # Flat month: fewer than MIN_POOL tickers pass ADV filter
        if n_eligible < MIN_POOL:
            results.append({
                "date": reb,
                "ticker": np.nan,
                "adv_usd": np.nan,
                "btc_corr_60d": np.nan,
                "rank": np.nan,
                "flat_month_flag": True,
            })
            log.info(f"FLAT month: {reb.date()} — only {n_eligible} eligible tickers (< {MIN_POOL})")
            continue

        # Rank by signed BTC correlation descending (Pass 1 convention)
        month_df = pd.DataFrame(ticker_rows)
        month_df = month_df.sort_values("btc_corr_60d", ascending=False).reset_index(drop=True)
        month_df = month_df.head(TOP_N).copy()
        month_df["rank"] = month_df.index + 1
        month_df.insert(0, "date", reb)
        month_df["flat_month_flag"] = False
        results.append(month_df)

    if not results:
        return pd.DataFrame(columns=["date", "ticker", "adv_usd", "btc_corr_60d", "rank", "flat_month_flag"])

    # Combine — handle both scalar rows and DataFrames
    df_parts = []
    for r in results:
        if isinstance(r, dict):
            df_parts.append(pd.DataFrame([r]))
        else:
            df_parts.append(r)

    return pd.concat(df_parts, ignore_index=True)


# ── Binance klines ─────────────────────────────────────────────────────────────
def load_spot_klines() -> dict:
    klines = {}
    for sym in SPOT_SYMBOLS:
        p = DATA / "binance" / "spot_klines" / f"{sym}_1m.parquet"
        df = pd.read_parquet(p).set_index("open_time").sort_index()
        klines[sym] = df
        log.info(f"Spot {sym}: {len(df):,} rows")
    return klines


def load_perp_klines() -> dict:
    klines = {}
    for sym in PERP_SYMBOLS:
        p = DATA / "binance" / "perp_klines" / f"{sym}_1m.parquet"
        df = pd.read_parquet(p).set_index("open_time").sort_index()
        klines[sym] = df
    return klines


def load_funding_rates() -> dict:
    rates = {}
    for sym in PERP_SYMBOLS:
        p = DATA / "binance" / "funding_rates" / f"{sym}_funding.parquet"
        df = pd.read_parquet(p).sort_values("funding_time").reset_index(drop=True)
        df["funding_time"] = pd.to_datetime(df["funding_time"], utc=True)
        rates[sym] = df
    return rates


# ── FRED macro ─────────────────────────────────────────────────────────────────
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


# ── Overnight windows ──────────────────────────────────────────────────────────
def build_overnight_windows() -> pd.DataFrame:
    cal = xcals.get_calendar("XKRX")
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
    log.info(f"KR: {len(df)} overnight windows, {df['is_weekend_gap'].sum()} weekend gaps")
    return df


# ── Crypto feature computation (identical to P2-3) ────────────────────────────
def slice_klines(klines_df: pd.DataFrame, w_start, w_end) -> pd.DataFrame:
    mask = (klines_df.index >= w_start) & (klines_df.index < w_end)
    return klines_df.loc[mask]


def log_return_from_endpoints(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return np.nan
    return np.log(df["close"].iloc[-1] / df["open"].iloc[0])


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
    return mdd


def taker_imbalance(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return np.nan
    total = df["quote_volume"].sum()
    if total == 0:
        return np.nan
    buy = df["taker_buy_quote_volume"].sum()
    return (buy - (total - buy)) / total


def compute_crypto_features(windows: pd.DataFrame, spot_klines: dict,
                              perp_klines: dict, funding_rates: dict) -> pd.DataFrame:
    n = len(windows)
    results = {col: np.full(n, np.nan) for col in CRYPTO_FEAT_COLS}
    btc_vols = np.full(n, np.nan)

    btc_spot = spot_klines["BTCUSDT"]
    eth_spot  = spot_klines["ETHUSDT"]
    sol_spot  = spot_klines["SOLUSDT"]
    bnb_spot  = spot_klines["BNBUSDT"]
    xrp_spot  = spot_klines["XRPUSDT"]
    btc_fund  = funding_rates["BTCUSDT"]

    sol_tmp = np.full(n, np.nan)
    bnb_tmp = np.full(n, np.nan)
    xrp_tmp = np.full(n, np.nan)

    log.info("Computing crypto features...")
    for i, row in enumerate(windows.itertuples(index=False)):
        ws = row.window_start_utc
        we = row.window_end_utc

        btc_slice = slice_klines(btc_spot, ws, we)
        if len(btc_slice) > 0:
            results["btc_ov_log_return"][i]      = log_return_from_endpoints(btc_slice)
            results["btc_ov_realized_vol"][i]    = realized_vol(btc_slice)
            results["btc_ov_max_drawdown"][i]    = max_drawdown(btc_slice)
            results["btc_ov_volume_usd"][i]      = btc_slice["quote_volume"].sum()
            results["btc_ov_taker_imbalance"][i] = taker_imbalance(btc_slice)
            btc_vols[i] = btc_slice["quote_volume"].sum()

        eth_slice = slice_klines(eth_spot, ws, we)
        if len(eth_slice) > 0:
            results["eth_ov_log_return"][i]   = log_return_from_endpoints(eth_slice)
            results["eth_ov_realized_vol"][i] = realized_vol(eth_slice)

        sol_slice = slice_klines(sol_spot, ws, we)
        if len(sol_slice) > 0:
            sol_tmp[i] = log_return_from_endpoints(sol_slice)

        bnb_slice = slice_klines(bnb_spot, ws, we)
        if len(bnb_slice) > 0:
            bnb_tmp[i] = log_return_from_endpoints(bnb_slice)

        xrp_slice = slice_klines(xrp_spot, ws, we)
        if len(xrp_slice) > 0:
            xrp_tmp[i] = log_return_from_endpoints(xrp_slice)

        fund_before = btc_fund[btc_fund["funding_time"] < we]
        if len(fund_before) >= 2:
            results["btc_funding_rate_latest"][i] = fund_before["funding_rate"].iloc[-1]
            results["btc_funding_rate_delta"][i]  = (
                fund_before["funding_rate"].iloc[-1] - fund_before["funding_rate"].iloc[-2]
            )
        elif len(fund_before) == 1:
            results["btc_funding_rate_latest"][i] = fund_before["funding_rate"].iloc[-1]

        if i % 500 == 0:
            log.info(f"  {i}/{n} windows processed")

    df = pd.DataFrame(results, index=windows.index)

    return_mat = np.column_stack([
        df["btc_ov_log_return"].values,
        df["eth_ov_log_return"].values,
        sol_tmp, bnb_tmp, xrp_tmp,
    ])
    df["crosspair_dispersion"] = np.nanstd(return_mat, axis=1)
    df["btc_eth_spread"]       = df["btc_ov_log_return"] - df["eth_ov_log_return"]

    windows_reset = windows.reset_index(drop=True)
    dates = pd.to_datetime(windows_reset["date"])
    vol_series = pd.Series(btc_vols, index=dates).sort_index()
    rolling_mean = vol_series.rolling(7, min_periods=4).mean().shift(1)
    df["btc_ov_volume_surge"] = btc_vols / rolling_mean.values

    return df


# ── Macro features ─────────────────────────────────────────────────────────────
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


# ── Stock features ─────────────────────────────────────────────────────────────
def compute_stock_features(date: pd.Timestamp, ticker: str, price_map: dict) -> dict:
    empty = {"stock_rv_20d": np.nan, "stock_ret_20d": np.nan, "stock_prior_day_return": np.nan}
    if ticker not in price_map:
        return empty
    hist = price_map[ticker]
    hist_before = hist[hist.index < date]
    if len(hist_before) < 2:
        return empty
    log_rets = np.log(hist_before["close"] / hist_before["close"].shift(1)).dropna()
    if len(log_rets) == 0:
        return empty
    prior_day_ret = float(log_rets.iloc[-1])
    trail = log_rets.iloc[-20:]
    if len(trail) < 5:
        return {"stock_rv_20d": np.nan, "stock_ret_20d": np.nan,
                "stock_prior_day_return": prior_day_ret}
    return {
        "stock_rv_20d":           float(trail.std() * np.sqrt(252)),
        "stock_ret_20d":          float(trail.sum()),
        "stock_prior_day_return": prior_day_ret,
    }


# ── Targets ────────────────────────────────────────────────────────────────────
def compute_targets_crossday(prev_date: pd.Timestamp, curr_date: pd.Timestamp,
                              ticker: str, price_map: dict) -> dict:
    empty = {"tgt_gap": np.nan, "tgt_intraday": np.nan, "tgt_cc": np.nan}
    if ticker not in price_map:
        return empty
    hist = price_map[ticker]
    if curr_date not in hist.index:
        return empty
    if prev_date not in hist.index:
        before = hist[hist.index < curr_date]
        if len(before) == 0:
            return empty
        prev_row = before.iloc[-1]
    else:
        prev_row = hist.loc[prev_date]
    curr_row = hist.loc[curr_date]
    if prev_row["close"] <= 0 or curr_row["open"] <= 0 or curr_row["close"] <= 0:
        return empty
    return {
        "tgt_gap":      float(np.log(curr_row["open"]  / prev_row["close"])),
        "tgt_intraday": float(np.log(curr_row["close"] / curr_row["open"])),
        "tgt_cc":       float(np.log(curr_row["close"] / prev_row["close"])),
    }


# ── Universe map: rebal_date -> ticker list ────────────────────────────────────
def build_universe_map(univ_log: pd.DataFrame) -> dict:
    """Build dict: rebal_date (date) -> list of tickers. Excludes flat months."""
    univ = univ_log[~univ_log["flat_month_flag"]].copy()
    univ["date"] = pd.to_datetime(univ["date"])
    rebal_dates = sorted(univ["date"].unique())
    rebal_map = {}
    for rd in rebal_dates:
        tickers = univ[univ["date"] == rd]["ticker"].tolist()
        rebal_map[pd.Timestamp(rd).date()] = tickers
    return rebal_map


def get_tickers_for_date(date: pd.Timestamp, rebal_map: dict) -> list:
    d = date.date() if hasattr(date, "date") else date
    eligible = [rd for rd in rebal_map.keys() if rd <= d]
    if not eligible:
        return []
    return rebal_map[max(eligible)]


# ── Build full feature table ───────────────────────────────────────────────────
def build_features(windows: pd.DataFrame, crypto_feat: pd.DataFrame,
                   macro_feat: pd.DataFrame, price_map: dict,
                   univ_log: pd.DataFrame) -> pd.DataFrame:
    rebal_map = build_universe_map(univ_log)
    rows = []
    n_windows = len(windows)
    log.info("Expanding windows to (date, ticker) rows...")

    for i, (_, wrow) in enumerate(windows.iterrows()):
        date      = pd.Timestamp(wrow["date"])
        next_date = pd.Timestamp(wrow["next_date"])

        tickers = get_tickers_for_date(date, rebal_map)
        if not tickers:
            continue

        cf = crypto_feat.iloc[i].to_dict()
        mf = macro_feat.iloc[i].to_dict()

        for ticker in tickers:
            sf  = compute_stock_features(next_date, ticker, price_map)
            tgt = compute_targets_crossday(date, next_date, ticker, price_map)
            row = {
                "date":           next_date,
                "window_date":    date,
                "ticker":         ticker,
                "is_weekend_gap": wrow["is_weekend_gap"],
                **cf, **mf, **sf, **tgt,
            }
            rows.append(row)

        if i % 100 == 0:
            log.info(f"  {i}/{n_windows} windows expanded")

    df = pd.DataFrame(rows)
    log.info(f"Feature table: {len(df):,} rows")
    return df


# ── Validation ─────────────────────────────────────────────────────────────────
def validate_universe_log(univ_log: pd.DataFrame) -> bool:
    ok = True
    active = univ_log[~univ_log["flat_month_flag"]].copy()
    active["date"] = pd.to_datetime(active["date"])

    # No .KQ tickers
    kq_found = active[active["ticker"].str.upper().str.endswith(".KQ")]
    if len(kq_found) > 0:
        log.error(f"VALIDATION FAIL: {len(kq_found)} .KQ tickers in universe log: {kq_found['ticker'].unique().tolist()}")
        ok = False
    else:
        log.info("Universe log: no .KQ tickers present (OK)")

    # No duplicate (date, ticker)
    dupes = active.duplicated(["date", "ticker"]).sum()
    if dupes > 0:
        log.error(f"VALIDATION FAIL: {dupes} duplicate (date, ticker) pairs")
        ok = False
    else:
        log.info("Universe log: no duplicate (date, ticker) pairs")

    # Monotonic dates
    all_dates = pd.to_datetime(univ_log["date"])
    if not all_dates.is_monotonic_increasing:
        log.error("VALIDATION FAIL: dates not monotonic")
        ok = False
    else:
        log.info("Universe log: dates monotonic")

    # Pool sizes
    per_month = active.groupby("date")["ticker"].count()
    if len(per_month) > 0:
        log.info(f"Active months pool sizes: min={per_month.min()}, median={per_month.median():.0f}, max={per_month.max()}")
        over_12 = (per_month > 12).sum()
        if over_12 > 0:
            log.warning(f"  {over_12} months with pool > 12 tickers")
    else:
        log.warning("No active months in universe log")

    return ok


def validate_features(df: pd.DataFrame) -> bool:
    ok = True
    log.info(f"\n{'='*60}")
    log.info("VALIDATION: feature parquet")
    log.info(f"{'='*60}")
    log.info(f"Total rows: {len(df):,}")

    # No duplicate (date, ticker)
    dupes = df.duplicated(["date", "ticker"]).sum()
    if dupes > 0:
        log.error(f"VALIDATION FAIL: {dupes} duplicate (date, ticker) pairs")
        ok = False
    else:
        log.info("No duplicate (date, ticker) pairs")

    all_feat = CRYPTO_FEAT_COLS + MACRO_FEAT_COLS + STOCK_FEAT_COLS
    log.info("\nFeature stats:")
    for col in all_feat:
        if col not in df.columns:
            log.warning(f"  MISSING COLUMN: {col}")
            continue
        s = df[col]
        nan_pct = s.isna().mean() * 100
        is_perp = col in PERP_COLS
        log.info(
            f"  {col:35s}: mean={s.mean():+.4f}  std={s.std():.4f}  nan={nan_pct:.1f}%"
            + (" [perp]" if is_perp else "")
        )
        # BLOCK: >20% NaN in any non-perp feature
        if not is_perp and nan_pct > 20:
            log.error(f"  BLOCK threshold: {col} has {nan_pct:.1f}% NaN (>20%)")
            ok = False

    log.info("\nTarget availability:")
    for tgt in TARGET_COLS:
        if tgt in df.columns:
            nn = df[tgt].notna().sum()
            log.info(f"  {tgt}: {nn:,} non-NaN ({nn/len(df)*100:.1f}%)")
        else:
            log.warning(f"  MISSING TARGET: {tgt}")

    return ok


# ── Pool size sanity distribution ──────────────────────────────────────────────
def pool_size_distribution(univ_log: pd.DataFrame) -> dict:
    """Return distribution of per-month pool sizes: {bucket: count_of_months}."""
    all_months = pd.to_datetime(univ_log["date"]).dt.to_period("M").unique()
    total_months = len(all_months)

    active = univ_log[~univ_log["flat_month_flag"]].copy()
    active["date"] = pd.to_datetime(active["date"])
    per_month = active.groupby(active["date"].dt.to_period("M"))["ticker"].count()

    flat_univ = univ_log[univ_log["flat_month_flag"]].copy()
    flat_univ["date"] = pd.to_datetime(flat_univ["date"])
    flat_months = flat_univ["date"].dt.to_period("M").unique()
    n_flat = len(flat_months)

    # Buckets
    dist = {
        "flat (0)":  n_flat,
        "1-5":       int((per_month[(per_month >= 1) & (per_month <= 5)]).count()),
        "6-8":       int((per_month[(per_month >= 6) & (per_month <= 8)]).count()),
        "9-12":      int((per_month[(per_month >= 9) & (per_month <= 12)]).count()),
    }
    return dist, total_months, n_flat


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("Stage P2-4: KOSPI Large-Cap Universe and Track A Features")
    log.info(f"Window: {WINDOW_START} to {WINDOW_END}")
    log.info(f"ADV threshold: ${ADV_THRESH_USD:,.0f} USD, Top-N: {TOP_N}, Min pool: {MIN_POOL}")

    # ── Load .KS tickers ──
    ks_tickers = load_ks_tickers()
    if not ks_tickers:
        stop("No .KS tickers found in crypto_candidates_kr.csv")

    # ── Load price histories ──
    log.info("\nLoading .KS price histories via yfinance (with cache)...")
    price_map, load_log = load_ks_prices(ks_tickers)

    if not price_map:
        stop("No .KS ticker price data could be loaded")

    # ── Load BTC daily returns ──
    btc_ret = load_btc_daily()

    # ── Load rebalance dates ──
    rebal_dates = get_rebalance_dates()

    # ── Compute universe ──
    log.info("\nComputing KOSPI large-cap universe...")
    univ_log = compute_kospi_largecap_universe(price_map, btc_ret, rebal_dates)

    # ── Save universe log ──
    univ_log_path = OUT / "kospi_largecap_universe_log.csv"
    univ_log_out = univ_log.copy()
    univ_log_out["date"] = pd.to_datetime(univ_log_out["date"]).dt.strftime("%Y-%m-%d")
    univ_log_out.to_csv(univ_log_path, index=False)
    log.info(f"Written: {univ_log_path} ({len(univ_log_out)} rows)")

    # ── Validate universe log ──
    log.info("\nValidating universe log...")
    univ_ok = validate_universe_log(univ_log)

    # ── BLOCK: no .KS tickers survive in any month ──
    active_rows = univ_log[~univ_log["flat_month_flag"]]
    if len(active_rows) == 0:
        stop("BLOCK: No .KS tickers survive ADV filter in any month")

    # ── BLOCK check: >10% flat months ──
    dist, total_months, n_flat = pool_size_distribution(univ_log)
    flat_pct = n_flat / total_months if total_months > 0 else 0
    log.info(f"\nPool size distribution (by month count):")
    for bucket, count in dist.items():
        log.info(f"  {bucket}: {count} months")
    log.info(f"Total months: {total_months}, Flat months: {n_flat} ({flat_pct:.1%})")

    if flat_pct > 0.10:
        stop(f"BLOCK: {flat_pct:.1%} of OOS months are flat (>{0.10:.0%} threshold). "
             f"Variant is inconclusive per brief risk #3.")

    # ── Load Binance + FRED ──
    log.info("\nLoading Binance klines...")
    spot_klines   = load_spot_klines()
    perp_klines   = load_perp_klines()
    funding_rates = load_funding_rates()

    log.info("\nLoading FRED macro...")
    fred = load_fred()

    # ── Build overnight windows ──
    log.info("\nBuilding overnight windows...")
    windows = build_overnight_windows()

    # ── Crypto features ──
    log.info("\nComputing crypto features...")
    crypto_feat = compute_crypto_features(windows, spot_klines, perp_klines, funding_rates)

    # ── Macro features ──
    log.info("\nAttaching macro features...")
    macro_feat = attach_macro_features(windows, fred)

    # ── Build full feature table ──
    log.info("\nBuilding feature table...")
    feat_df = build_features(windows, crypto_feat, macro_feat, price_map, univ_log)

    # Drop helper columns
    for col in ["window_date"]:
        if col in feat_df.columns:
            feat_df.drop(columns=[col], inplace=True)

    # ── Validate features ──
    feat_ok = validate_features(feat_df)

    # BLOCK: >20% NaN in any non-perp feature column
    all_feat_cols = CRYPTO_FEAT_COLS + MACRO_FEAT_COLS + STOCK_FEAT_COLS
    for col in all_feat_cols:
        if col in feat_df.columns and col not in PERP_COLS:
            nan_pct = feat_df[col].isna().mean() * 100
            if nan_pct > 20:
                stop(f"BLOCK: feature {col} has {nan_pct:.1f}% NaN (>20% threshold)")

    # ── Save feature parquet ──
    feat_path = OUT / "features_track_a_kospi_largecap.parquet"
    feat_df.to_parquet(feat_path, index=False)
    log.info(f"\nSaved: {feat_path} ({feat_df.shape})")

    # ── Final summary ──
    n_ks_in_csv = len(ks_tickers)
    active_univ = univ_log[~univ_log["flat_month_flag"]]
    n_active_months = active_univ["date"].nunique()
    pool_sizes = active_univ.groupby("date")["ticker"].count() if len(active_univ) > 0 else pd.Series(dtype=int)

    log.info("\n" + "=" * 70)
    log.info("STAGE P2-4 FINAL SUMMARY")
    log.info("=" * 70)
    log.info(f".KS tickers from crypto_candidates_kr.csv: {n_ks_in_csv}")
    log.info(f"  Loaded: {len(load_log['found'])}, Missing: {len(load_log['missing'])}")
    log.info(f"Rebalance months total:  {total_months}")
    log.info(f"Active months (>=9):     {n_active_months}")
    log.info(f"Flat months (<9):        {n_flat} ({flat_pct:.1%} — threshold 10%)")
    if len(pool_sizes) > 0:
        log.info(f"Typical pool size:       median={pool_sizes.median():.0f}, range=[{pool_sizes.min()}, {pool_sizes.max()}]")
    log.info(f"Features parquet:        {feat_df.shape[0]:,} rows x {feat_df.shape[1]} columns")
    log.info(f"Feature count (21 expected): {sum(1 for c in feat_df.columns if c in all_feat_cols)}")
    log.info(f"Validation: universe={'OK' if univ_ok else 'FAIL'}, features={'OK' if feat_ok else 'FAIL'}")

    if not feat_ok or not univ_ok:
        log.error("BLOCK: validation failures — see above")
    else:
        log.info("Stage P2-4 complete. No BLOCK conditions triggered.")


if __name__ == "__main__":
    main()
