# ABOUTME: Stage P2-3 — control universe construction and Track A feature engineering.
# ABOUTME: Inputs: data/stock_picks/crypto_control_{hk,kr}.csv, data/stooq/, data/pykrx/ (via yfinance for KR control),
#           data/binance/, data/fred/, output/universe_log.csv.
#           Outputs: output/control_universe_log.csv, output/features_track_a_control_{hk,kr}.parquet,
#           logs/stage_p2-3_control.log.
# Run: source .venv/bin/activate && python3 scripts/stage_p2-3_control_universe.py

import os
import sys
import logging
import warnings
import time
import numpy as np
import pandas as pd
import exchange_calendars as xcals
import yfinance as yf
from pathlib import Path
from datetime import timezone

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity")
DATA = ROOT / "data"
OUT  = ROOT / "output"
LOGS = ROOT / "logs"
OUT.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

LOG_FILE = LOGS / "stage_p2-3_control.log"

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
WINDOW_START = "2019-01-01"
WINDOW_END   = "2026-04-15"
ADV_WINDOW   = 20          # trailing trading days
ADV_THRESH_USD = 500_000
TOP_N_MAX    = 30
HKD_USD      = 1 / 7.8    # constant HKD→USD
KRW_USD      = 1 / 1330   # constant KRW→USD (per P2-3 brief)
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
STOCK_FEAT_COLS = ["stock_rv_20d", "stock_ret_20d", "stock_prior_day_return"]
TARGET_COLS     = ["tgt_gap", "tgt_intraday", "tgt_cc"]
PERP_COLS       = ["btc_funding_rate_latest", "btc_funding_rate_delta"]


def stop(msg: str) -> None:
    log.error(f"[BLOCK] {msg}")
    sys.exit(1)


# ── Load Binance 1m spot klines ────────────────────────────────────────────────
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


# ── Load HK Stooq files ────────────────────────────────────────────────────────
def stooq_filename(ticker_raw: str):
    ticker_raw = ticker_raw.strip()
    if ticker_raw.upper().endswith(".HK"):
        numeric_part = ticker_raw[:-3]
        stripped = numeric_part.lstrip("0") or numeric_part
        return stripped + ".hk.txt"
    return None


def load_hk_control_prices(tickers: list) -> tuple:
    """Load Stooq HK daily prices for control tickers. Returns (price_map dict, log dict)."""
    hk_dir = DATA / "stooq" / "hk_daily"
    price_map = {}
    load_log = {"found": [], "missing": []}

    for ticker_raw in tickers:
        fname = stooq_filename(ticker_raw)
        if fname is None:
            load_log["missing"].append(f"{ticker_raw}: non-HK suffix")
            continue
        fpath = hk_dir / fname
        if not fpath.exists():
            load_log["missing"].append(f"{ticker_raw}: file {fname} not found")
            continue
        try:
            df = pd.read_csv(
                fpath, header=0,
                names=["ticker_col", "per", "date", "time", "open", "high", "low", "close", "vol", "openint"],
            )
            df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
            df = df.sort_values("date").set_index("date")
            df = df[["open", "high", "low", "close", "vol"]].astype(float)
            df = df.rename(columns={"vol": "volume"})
            df["adv_usd_raw"] = df["close"] * df["volume"] * HKD_USD
            df["ret"] = df["close"].pct_change()
            price_map[ticker_raw] = df
            load_log["found"].append(ticker_raw)
        except Exception as e:
            load_log["missing"].append(f"{ticker_raw}: parse error — {e}")

    log.info(f"HK control: {len(load_log['found'])} loaded, {len(load_log['missing'])} missing")
    return price_map, load_log


# ── Load KR control prices via yfinance ───────────────────────────────────────
def load_kr_control_prices_yfinance(tickers: list) -> tuple:
    """
    Pull daily OHLCV for KR control tickers via yfinance.
    KR tickers are in format 259960.KS or 041140.KQ — yfinance uses this format directly.
    Returns (price_map dict, load_log dict).
    """
    price_map = {}
    load_log = {"found": [], "missing": []}

    # Check if we have pre-cached data from a previous run
    cache_dir = DATA / "yfinance" / "kr_control"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        cache_file = cache_dir / f"{ticker.replace('.', '_')}.parquet"

        # Use cache if available and recent
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if len(df) > 0:
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    price_map[ticker] = df
                    load_log["found"].append(ticker)
                    log.info(f"KR control {ticker}: loaded from cache ({len(df)} rows)")
                    continue
            except Exception:
                pass

        # Pull from yfinance
        log.info(f"KR control {ticker}: pulling from yfinance...")
        try:
            t = yf.Ticker(ticker)
            raw = t.history(start=WINDOW_START, end="2026-04-18", interval="1d", auto_adjust=True)
            time.sleep(0.3)  # rate limit courtesy

            if raw is None or len(raw) == 0:
                load_log["missing"].append(f"{ticker}: yfinance returned empty")
                log.warning(f"KR control {ticker}: empty response")
                continue

            raw = raw.reset_index()
            raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
            if "datetime" in raw.columns:
                raw = raw.rename(columns={"datetime": "date"})
            raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None).dt.normalize()
            raw = raw.rename(columns={"stock_splits": "splits"} if "stock_splits" in raw.columns else {})
            raw = raw[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
            raw = raw.set_index("date").sort_index()
            raw = raw.astype(float)
            raw["adv_usd_raw"] = raw["close"] * raw["volume"] * KRW_USD
            raw["ret"] = raw["close"].pct_change()

            raw.to_parquet(cache_file)
            price_map[ticker] = raw
            load_log["found"].append(ticker)
            log.info(f"KR control {ticker}: {len(raw)} rows ({raw.index.min().date()} to {raw.index.max().date()})")
        except Exception as e:
            load_log["missing"].append(f"{ticker}: yfinance error — {e}")
            log.warning(f"KR control {ticker}: {e}")

    log.info(f"KR control: {len(load_log['found'])} loaded, {len(load_log['missing'])} missing")
    return price_map, load_log


# ── Rebalance dates from universe_log ─────────────────────────────────────────
def get_rebalance_dates(market: str) -> pd.DatetimeIndex:
    """Read rebalance dates from output/universe_log.csv (same grid as main universe)."""
    univ = pd.read_csv(OUT / "universe_log.csv")
    univ["date"] = pd.to_datetime(univ["date"])
    dates = univ[univ["market"] == market]["date"].sort_values().unique()
    return pd.DatetimeIndex(dates)


# ── ADV-only universe construction (no BTC corr filter) ───────────────────────
def compute_control_universe(price_map: dict, rebal_dates: pd.DatetimeIndex,
                              market_name: str) -> pd.DataFrame:
    """
    For each rebalance date, compute trailing 20-day ADV, apply $500K filter,
    rank by ADV, keep top 30. NO BTC correlation filter (by design).
    Returns DataFrame: date, market, ticker, adv_usd, rank.
    """
    results = []

    for reb in rebal_dates:
        ticker_rows = []
        for ticker, df in price_map.items():
            # Filter to data strictly before rebalance date
            hist = df[df.index < reb]
            if len(hist) < 5:
                continue
            # Trailing 20-day ADV
            adv_series = hist["adv_usd_raw"].iloc[-ADV_WINDOW:]
            adv_valid = adv_series.dropna()
            if len(adv_valid) < 5:
                continue
            adv_usd = adv_valid.mean()
            if adv_usd < ADV_THRESH_USD:
                continue
            ticker_rows.append({"ticker": ticker, "adv_usd": adv_usd})

        if not ticker_rows:
            continue

        month_df = pd.DataFrame(ticker_rows)
        month_df = month_df.sort_values("adv_usd", ascending=False).reset_index(drop=True)
        # Cap at top 30
        month_df = month_df.head(TOP_N_MAX)
        month_df["rank"] = month_df.index + 1
        month_df.insert(0, "date", reb)
        month_df.insert(1, "market", market_name)
        results.append(month_df)

    if not results:
        return pd.DataFrame(columns=["date", "market", "ticker", "adv_usd", "rank"])

    return pd.concat(results, ignore_index=True)


# ── Overnight windows (same logic as stage3_features.py) ─────────────────────
def build_overnight_windows(market: str) -> pd.DataFrame:
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
            "date":             sess.date(),
            "next_date":        next_sess.date(),
            "window_start_utc": w_start,
            "window_end_utc":   w_end,
            "is_weekend_gap":   gap_days > 3,
            "gap_calendar_days": gap_days,
        })
    df = pd.DataFrame(rows)
    log.info(f"{market}: {len(df)} overnight windows")
    return df


# ── Crypto feature computation (identical to stage3_features.py) ──────────────
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

    btc_spot = spot_klines["BTCUSDT"]
    eth_spot = spot_klines["ETHUSDT"]
    sol_spot = spot_klines["SOLUSDT"]
    bnb_spot = spot_klines["BNBUSDT"]
    xrp_spot = spot_klines["XRPUSDT"]
    btc_fund = funding_rates["BTCUSDT"]

    # Temp storage for vol surge computation
    btc_vols = np.full(n, np.nan)

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
            results["sol_ov_log_return_tmp"] = results.get("sol_ov_log_return_tmp", np.full(n, np.nan))
            results["sol_ov_log_return_tmp"][i] = log_return_from_endpoints(sol_slice)

        bnb_slice = slice_klines(bnb_spot, ws, we)
        if len(bnb_slice) > 0:
            results["bnb_ov_log_return_tmp"] = results.get("bnb_ov_log_return_tmp", np.full(n, np.nan))
            results["bnb_ov_log_return_tmp"][i] = log_return_from_endpoints(bnb_slice)

        xrp_slice = slice_klines(xrp_spot, ws, we)
        if len(xrp_slice) > 0:
            results["xrp_ov_log_return_tmp"] = results.get("xrp_ov_log_return_tmp", np.full(n, np.nan))
            results["xrp_ov_log_return_tmp"][i] = log_return_from_endpoints(xrp_slice)

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

    df = pd.DataFrame({k: v for k, v in results.items() if not k.endswith("_tmp")},
                      index=windows.index)

    # Cross-pair dispersion and BTC-ETH spread
    sol_tmp = results.get("sol_ov_log_return_tmp", np.full(n, np.nan))
    bnb_tmp = results.get("bnb_ov_log_return_tmp", np.full(n, np.nan))
    xrp_tmp = results.get("xrp_ov_log_return_tmp", np.full(n, np.nan))

    return_mat = np.column_stack([
        df["btc_ov_log_return"].values,
        df["eth_ov_log_return"].values,
        sol_tmp, bnb_tmp, xrp_tmp,
    ])
    df["crosspair_dispersion"] = np.nanstd(return_mat, axis=1)
    df["btc_eth_spread"]       = df["btc_ov_log_return"] - df["eth_ov_log_return"]

    # Volume surge: btc_ov_volume_usd / trailing 7-day mean (shifted 1)
    windows_reset = windows.reset_index(drop=True)
    dates = pd.to_datetime(windows_reset["date"])
    vol_series = pd.Series(btc_vols, index=dates).sort_index()
    rolling_mean = vol_series.rolling(7, min_periods=4).mean().shift(1)
    df["btc_ov_volume_surge"] = btc_vols / rolling_mean.values

    return df


# ── Macro feature join (identical to stage3_features.py) ─────────────────────
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


# ── Stock features (identical to stage3_features.py) ─────────────────────────
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
        "stock_rv_20d":          float(trail.std() * np.sqrt(252)),
        "stock_ret_20d":         float(trail.sum()),
        "stock_prior_day_return": prior_day_ret,
    }


# ── Targets (identical to stage3_features.py) ─────────────────────────────────
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


# ── Control universe map: date -> tickers ─────────────────────────────────────
def build_control_universe_map(market: str, ctrl_univ_log: pd.DataFrame) -> dict:
    sub = ctrl_univ_log[ctrl_univ_log["market"] == market].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    rebal_dates = sorted(sub["date"].unique())
    rebal_map = {}
    for rd in rebal_dates:
        tickers = sub[sub["date"] == rd]["ticker"].tolist()
        rebal_map[pd.Timestamp(rd).date()] = tickers
    return rebal_map


def get_tickers_for_date(date: pd.Timestamp, rebal_map: dict) -> list:
    d = date.date() if hasattr(date, "date") else date
    eligible = [rd for rd in rebal_map.keys() if rd <= d]
    if not eligible:
        return []
    return rebal_map[max(eligible)]


# ── Full feature table for a market ──────────────────────────────────────────
def build_features_for_market(market: str, windows: pd.DataFrame,
                               crypto_feat: pd.DataFrame, macro_feat: pd.DataFrame,
                               price_map: dict, ctrl_univ_log: pd.DataFrame) -> pd.DataFrame:
    rebal_map = build_control_universe_map(market, ctrl_univ_log)
    rows = []
    n_windows = len(windows)
    log.info(f"{market}: expanding windows to ticker rows...")

    for i, (_, wrow) in enumerate(windows.iterrows()):
        date     = pd.Timestamp(wrow["date"])
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
                "date":            next_date,
                "window_date":     date,
                "ticker":          ticker,
                "market":          market,
                "is_weekend_gap":  wrow["is_weekend_gap"],
                **cf, **mf, **sf, **tgt,
            }
            rows.append(row)

        if i % 100 == 0:
            log.info(f"  {market}: {i}/{n_windows} windows expanded")

    df = pd.DataFrame(rows)
    log.info(f"{market}: {len(df):,} rows")
    return df


# ── Validation ─────────────────────────────────────────────────────────────────
def validate_universe_log(ctrl_log: pd.DataFrame) -> bool:
    ok = True
    ctrl_log = ctrl_log.copy()
    ctrl_log["date"] = pd.to_datetime(ctrl_log["date"])

    # No duplicate (date, ticker) pairs
    dupes = ctrl_log.duplicated(["date", "ticker"]).sum()
    if dupes > 0:
        log.error(f"VALIDATION FAIL: {dupes} duplicate (date, ticker) pairs in universe log")
        ok = False
    else:
        log.info("Universe log: no duplicate (date, ticker) pairs")

    # Monotonic dates
    for mkt in ctrl_log["market"].unique():
        dates = ctrl_log[ctrl_log["market"] == mkt]["date"].sort_values()
        if not dates.is_monotonic_increasing:
            log.error(f"VALIDATION FAIL: {mkt} dates not monotonic")
            ok = False

    # Per-month ticker counts 0-30
    per_month = ctrl_log.groupby(["market", "date"])["ticker"].count()
    over_30 = (per_month > 30).sum()
    if over_30 > 0:
        log.error(f"VALIDATION FAIL: {over_30} months with > 30 tickers")
        ok = False
    log.info(f"Universe log: per-month counts min={per_month.min()}, max={per_month.max()}")

    # BLOCK: < 5 tickers on > 30% of months
    for mkt in ctrl_log["market"].unique():
        mkt_counts = ctrl_log[ctrl_log["market"] == mkt].groupby("date")["ticker"].count()
        thin_pct = (mkt_counts < 5).mean()
        if thin_pct > 0.30:
            stop(f"BLOCK: {mkt} control set too thin — {thin_pct:.1%} of rebalance months have < 5 tickers")

    return ok


def validate_features(df: pd.DataFrame, market: str, ctrl_log: pd.DataFrame) -> bool:
    ok = True
    log.info(f"\n{'='*60}")
    log.info(f"VALIDATION: {market} features")
    log.info(f"{'='*60}")
    log.info(f"Total rows: {len(df):,}")

    # No duplicate (date, ticker) pairs
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
            f"  {col:35s}: mean={s.mean():+.4f}  std={s.std():.4f}  "
            f"nan={nan_pct:.1f}%"
            + (" [perp]" if is_perp else "")
        )
        # BLOCK: > 20% NaN in any feature column
        if nan_pct > 20 and not is_perp:
            log.error(f"  BLOCK threshold check: {col} has {nan_pct:.1f}% NaN (>20% threshold)")
            ok = False

        # Sigma outlier check
        if s.notna().sum() > 10:
            mu, sigma = s.mean(), s.std()
            if sigma > 0:
                outliers = (np.abs(s - mu) > 10 * sigma).sum()
                if outliers > 0:
                    log.warning(f"  {col}: {outliers} values > 10 sigma from mean")

    log.info("\nTarget availability:")
    for tgt in TARGET_COLS:
        if tgt in df.columns:
            nn = df[tgt].notna().sum()
            log.info(f"  {tgt}: {nn:,} non-NaN ({nn/len(df)*100:.1f}%)")
        else:
            log.warning(f"  MISSING TARGET: {tgt}")

    # Check ticker-date in universe log vs feature parquet
    ctrl_sub = ctrl_log[ctrl_log["market"] == market].copy()
    ctrl_sub["date"] = pd.to_datetime(ctrl_sub["date"])
    feat_keys = set(zip(df["date"].dt.normalize(), df["ticker"]))
    # Get universe dates aligned to next_date (feature date = next_date after rebal)
    # The universe log has rebal dates; features have next trading day. Approximate check:
    univ_tickers = set(ctrl_sub["ticker"].unique())
    feat_tickers = set(df["ticker"].unique())
    missing_from_feat = univ_tickers - feat_tickers
    if missing_from_feat:
        log.warning(f"  Tickers in universe log but never in features: {missing_from_feat}")

    return ok


# ── Sanity table ───────────────────────────────────────────────────────────────
def print_sanity_table(df: pd.DataFrame, market: str):
    log.info(f"\n--- Sanity table: {market} per-ticker row counts ---")
    per_ticker = df.groupby("ticker").agg(
        rows=("date", "count"),
        tgt_gap_mean=("tgt_gap", "mean"),
        tgt_gap_std=("tgt_gap", "std"),
        stock_rv_mean=("stock_rv_20d", "mean"),
    ).sort_values("rows", ascending=False)
    log.info("\n" + per_ticker.to_string())

    log.info(f"\n--- Feature means/stds ({market}) ---")
    all_feat = CRYPTO_FEAT_COLS + MACRO_FEAT_COLS + STOCK_FEAT_COLS
    for col in all_feat:
        if col in df.columns:
            log.info(f"  {col:35s}: mean={df[col].mean():+.6f}  std={df[col].std():.6f}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("Stage P2-3: Control Universe Construction and Feature Engineering")
    log.info(f"Window: {WINDOW_START} to {WINDOW_END}")

    # ── Load control CSVs ──
    ctrl_hk_df = pd.read_csv(DATA / "stock_picks" / "crypto_control_hk.csv")
    ctrl_kr_df = pd.read_csv(DATA / "stock_picks" / "crypto_control_kr.csv")
    ctrl_hk_tickers = ctrl_hk_df["ticker"].str.strip().tolist()
    ctrl_kr_tickers = ctrl_kr_df["ticker"].str.strip().tolist()
    log.info(f"Loaded HK control tickers: {len(ctrl_hk_tickers)}")
    log.info(f"Loaded KR control tickers: {len(ctrl_kr_tickers)}")
    log.info(f"HK tickers: {ctrl_hk_tickers}")
    log.info(f"KR tickers: {ctrl_kr_tickers}")

    # ── Load price histories ──
    log.info("\nLoading HK control price histories...")
    hk_prices, hk_load_log = load_hk_control_prices(ctrl_hk_tickers)

    log.info("\nLoading KR control price histories via yfinance...")
    kr_prices, kr_load_log = load_kr_control_prices_yfinance(ctrl_kr_tickers)

    # ── Load rebalance dates from main universe_log ──
    log.info("\nReading rebalance dates from output/universe_log.csv...")
    hk_rebal = get_rebalance_dates("HK")
    kr_rebal = get_rebalance_dates("KR")
    log.info(f"HK rebalance dates: {len(hk_rebal)} ({hk_rebal[0].date()} to {hk_rebal[-1].date()})")
    log.info(f"KR rebalance dates: {len(kr_rebal)} ({kr_rebal[0].date()} to {kr_rebal[-1].date()})")

    # ── Compute control universes ──
    log.info("\nComputing HK control universe...")
    hk_univ = compute_control_universe(hk_prices, hk_rebal, "HK")

    log.info("\nComputing KR control universe...")
    kr_univ = compute_control_universe(kr_prices, kr_rebal, "KR")

    # ── Save control universe log ──
    ctrl_log = pd.concat([hk_univ, kr_univ], ignore_index=True)
    ctrl_log = ctrl_log.sort_values(["market", "date", "rank"]).reset_index(drop=True)
    ctrl_log["date"] = ctrl_log["date"].dt.strftime("%Y-%m-%d")
    ctrl_log_path = OUT / "control_universe_log.csv"
    ctrl_log.to_csv(ctrl_log_path, index=False)
    log.info(f"Written: {ctrl_log_path} ({len(ctrl_log)} rows)")

    # ── Validate universe log ──
    log.info("\nValidating universe log...")
    validate_universe_log(ctrl_log)

    # ── Load Binance + FRED (expensive, once) ──
    log.info("\nLoading Binance klines...")
    spot_klines   = load_spot_klines()
    perp_klines   = load_perp_klines()
    funding_rates = load_funding_rates()

    log.info("\nLoading FRED macro...")
    fred = load_fred()

    # ── Build overnight windows ──
    log.info("\nBuilding overnight windows...")
    hk_windows = build_overnight_windows("HK")
    kr_windows = build_overnight_windows("KR")

    # ── Compute crypto features (shared for both markets) ──
    log.info("\nComputing crypto features for HK windows...")
    hk_crypto = compute_crypto_features(hk_windows, spot_klines, perp_klines, funding_rates)

    log.info("\nComputing crypto features for KR windows...")
    kr_crypto = compute_crypto_features(kr_windows, spot_klines, perp_klines, funding_rates)

    # ── Macro features ──
    log.info("\nAttaching macro features...")
    hk_macro = attach_macro_features(hk_windows, fred)
    kr_macro = attach_macro_features(kr_windows, fred)

    # ── ctrl_log back as datetime for map building ──
    ctrl_log_dt = ctrl_log.copy()
    ctrl_log_dt["date"] = pd.to_datetime(ctrl_log_dt["date"])

    # ── Build feature tables ──
    log.info("\nBuilding HK control feature table...")
    hk_df = build_features_for_market("HK", hk_windows, hk_crypto, hk_macro,
                                       hk_prices, ctrl_log_dt)

    log.info("\nBuilding KR control feature table...")
    kr_df = build_features_for_market("KR", kr_windows, kr_crypto, kr_macro,
                                       kr_prices, ctrl_log_dt)

    # Drop helper columns
    for df in [hk_df, kr_df]:
        for col in ["window_date", "sol_ov_log_return", "bnb_ov_log_return", "xrp_ov_log_return"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    # ── Validate feature parquets ──
    hk_ok = validate_features(hk_df, "HK", ctrl_log_dt)
    kr_ok = validate_features(kr_df, "KR", ctrl_log_dt)

    # BLOCK: > 20% NaN in any feature column
    all_feat_cols = CRYPTO_FEAT_COLS + MACRO_FEAT_COLS + STOCK_FEAT_COLS
    for mkt, df in [("HK", hk_df), ("KR", kr_df)]:
        for col in all_feat_cols:
            if col in df.columns and col not in PERP_COLS:
                nan_pct = df[col].isna().mean() * 100
                if nan_pct > 20:
                    stop(f"BLOCK: {mkt} feature {col} has {nan_pct:.1f}% NaN (>20% threshold)")

    # ── Sanity tables ──
    if len(hk_df) > 0:
        print_sanity_table(hk_df, "HK")
    if len(kr_df) > 0:
        print_sanity_table(kr_df, "KR")

    # ── Save parquets (do NOT overwrite main universe parquets) ──
    hk_path = OUT / "features_track_a_control_hk.parquet"
    kr_path = OUT / "features_track_a_control_kr.parquet"
    hk_df.to_parquet(hk_path, index=False)
    kr_df.to_parquet(kr_path, index=False)
    log.info(f"\nSaved: {hk_path} ({hk_df.shape})")
    log.info(f"Saved: {kr_path} ({kr_df.shape})")

    # ── Summary log ──
    n_feat = len([c for c in hk_df.columns if c in all_feat_cols])

    log.info("\n" + "=" * 70)
    log.info("STAGE P2-3 SUMMARY")
    log.info("=" * 70)
    log.info(f"HK control: {len(ctrl_hk_tickers)} raw tickers from CSV")
    log.info(f"  Found in Stooq: {len(hk_load_log['found'])}")
    log.info(f"  Missing:        {len(hk_load_log['missing'])} — {hk_load_log['missing']}")
    log.info(f"KR control: {len(ctrl_kr_tickers)} raw tickers from CSV")
    log.info(f"  Found via yfinance: {len(kr_load_log['found'])}")
    log.info(f"  Missing:            {len(kr_load_log['missing'])} — {kr_load_log['missing']}")

    hk_cts = ctrl_log_dt[ctrl_log_dt["market"] == "HK"].groupby("date")["ticker"].count()
    kr_cts = ctrl_log_dt[ctrl_log_dt["market"] == "KR"].groupby("date")["ticker"].count()

    log.info(f"HK rebalance months: {len(hk_cts)}, typical pool size: median={hk_cts.median():.0f} (min={hk_cts.min()}, max={hk_cts.max()})")
    log.info(f"KR rebalance months: {len(kr_cts)}, typical pool size: median={kr_cts.median():.0f} (min={kr_cts.min()}, max={kr_cts.max()})")
    log.info(f"HK feature parquet:  {hk_df.shape[0]:,} rows x {hk_df.shape[1]} columns")
    log.info(f"KR feature parquet:  {kr_df.shape[0]:,} rows x {kr_df.shape[1]} columns")
    log.info(f"Feature count (non-meta): {n_feat} (expected 21: 12 crypto + 6 macro + 3 stock)")
    log.info(f"Sanity checks: {'PASS' if hk_ok and kr_ok else 'FAIL — see above'}")
    log.info(f"File sizes: {ctrl_log_path.stat().st_size//1024}KB, {hk_path.stat().st_size//1024}KB, {kr_path.stat().st_size//1024}KB")
    log.info("Stage P2-3 complete.")


if __name__ == "__main__":
    main()
