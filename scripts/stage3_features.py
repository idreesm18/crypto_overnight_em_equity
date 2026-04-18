# ABOUTME: Stage 3 Track A feature engineering for crypto overnight EM equity project.
# ABOUTME: Inputs: Binance 1m klines, FRED macro, Stooq HK, pykrx KR, universe_log.csv
# ABOUTME: Outputs: output/features_track_a_{hk,kr}.parquet, output/overnight_window_log.csv
# ABOUTME: Run: source .venv/bin/activate && python3 scripts/stage3_features.py

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import exchange_calendars as xcals
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

LOG_FILE = LOGS / "stage3_features.log"

# ── Logging setup ──────────────────────────────────────────────────────────────
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
WINDOW_END   = "2026-04-15"
MIN_PER_YEAR = 525_600  # minutes in a year for vol scaling

SPOT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
PERP_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


# ── Load Binance 1m spot klines ────────────────────────────────────────────────
def load_spot_klines() -> dict[str, pd.DataFrame]:
    """Load all spot 1m klines, indexed by open_time (UTC)."""
    klines = {}
    for sym in SPOT_SYMBOLS:
        p = DATA / "binance" / "spot_klines" / f"{sym}_1m.parquet"
        df = pd.read_parquet(p)
        df = df.set_index("open_time").sort_index()
        klines[sym] = df
        log.info(f"Spot {sym}: {len(df):,} rows, {df.index.min()} to {df.index.max()}")
    return klines


def load_perp_klines() -> dict[str, pd.DataFrame]:
    """Load BTC/ETH perp 1m klines, indexed by open_time (UTC)."""
    klines = {}
    for sym in PERP_SYMBOLS:
        p = DATA / "binance" / "perp_klines" / f"{sym}_1m.parquet"
        df = pd.read_parquet(p)
        df = df.set_index("open_time").sort_index()
        klines[sym] = df
        log.info(f"Perp {sym}: {len(df):,} rows, {df.index.min()} to {df.index.max()}")
    return klines


def load_funding_rates() -> dict[str, pd.DataFrame]:
    """Load BTC/ETH 8-hourly funding rates."""
    rates = {}
    for sym in PERP_SYMBOLS:
        p = DATA / "binance" / "funding_rates" / f"{sym}_funding.parquet"
        df = pd.read_parquet(p)
        df = df.sort_values("funding_time").reset_index(drop=True)
        df["funding_time"] = pd.to_datetime(df["funding_time"], utc=True)
        rates[sym] = df
        log.info(f"Funding {sym}: {len(df):,} rows, {df['funding_time'].min()} to {df['funding_time'].max()}")
    return rates


# ── Load macro data ────────────────────────────────────────────────────────────
def load_fred() -> pd.DataFrame:
    """Load all FRED series, merge on date, forward-fill holidays/weekends."""
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
    # Forward-fill up to 5 business days (holidays/weekends in FRED)
    fred = fred.ffill(limit=7)

    # Add derived columns
    fred["yield_curve_slope"] = fred["dgs10"] - fred["dgs2"]
    fred["vix_5d_change"]     = fred["vix"] - fred["vix"].shift(5)
    fred["dxy_5d_change"]     = fred["dxy"] - fred["dxy"].shift(5)

    log.info(f"FRED merged: {len(fred)} rows, {fred.index.min()} to {fred.index.max()}")
    return fred


# ── Overnight window construction ──────────────────────────────────────────────
def build_overnight_windows(market: str) -> pd.DataFrame:
    """
    For each trading session T, compute:
      - window_start: close of session T (UTC)
      - window_end: open of next session T+1 (UTC)
      - is_weekend_gap: True if gap spans a weekend (calendar gap > 3 days)
    Returns DataFrame with columns: date, window_start_utc, window_end_utc, is_weekend_gap
    """
    cal_name = "XHKG" if market == "HK" else "XKRX"
    cal = xcals.get_calendar(cal_name)

    sessions = cal.sessions_in_range(WINDOW_START, WINDOW_END)
    rows = []

    for i, sess in enumerate(sessions[:-1]):
        # Current session close
        try:
            w_start = cal.session_close(sess)
        except Exception:
            continue

        # Next session open
        next_sess = sessions[i + 1]
        try:
            w_end = cal.session_open(next_sess)
        except Exception:
            continue

        # Weekend gap: more than 3 calendar days between sessions
        gap_days = (next_sess - sess).days
        is_weekend_gap = gap_days > 3

        rows.append({
            "date":            sess.date(),       # trading date T (the session that closes)
            "next_date":       next_sess.date(),  # trading date T+1 (the session that opens)
            "window_start_utc": w_start,
            "window_end_utc":   w_end,
            "is_weekend_gap":  is_weekend_gap,
            "gap_calendar_days": gap_days,
        })

    df = pd.DataFrame(rows)
    log.info(f"{market}: {len(df)} overnight windows built, {df['is_weekend_gap'].sum()} weekend gaps")
    return df


# ── Crypto feature computation ─────────────────────────────────────────────────
def slice_klines(klines_df: pd.DataFrame, w_start, w_end) -> pd.DataFrame:
    """Extract 1m bars within [w_start, w_end) UTC."""
    mask = (klines_df.index >= w_start) & (klines_df.index < w_end)
    return klines_df.loc[mask]


def log_return_from_endpoints(df: pd.DataFrame) -> float:
    """log(last_close / first_open) over the window."""
    if len(df) == 0:
        return np.nan
    return np.log(df["close"].iloc[-1] / df["open"].iloc[0])


def realized_vol(df: pd.DataFrame) -> float:
    """Annualized stdev of 1-min log returns within window."""
    if len(df) < 2:
        return np.nan
    r = np.log(df["close"] / df["close"].shift(1)).dropna()
    if len(r) == 0:
        return np.nan
    return float(r.std() * np.sqrt(MIN_PER_YEAR))


def max_drawdown(df: pd.DataFrame) -> float:
    """(min after peak / peak) - 1 using high/close prices."""
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
    """(buy_quote - sell_quote) / total_quote aggregated over window."""
    if len(df) == 0:
        return np.nan
    total = df["quote_volume"].sum()
    if total == 0:
        return np.nan
    buy = df["taker_buy_quote_volume"].sum()
    sell = total - buy
    return (buy - sell) / total


def compute_crypto_features(
    windows: pd.DataFrame,
    spot_klines: dict,
    perp_klines: dict,
    funding_rates: dict,
) -> pd.DataFrame:
    """
    Compute all 12 crypto overnight features per window row.
    Returns DataFrame aligned with windows index.
    """
    n = len(windows)
    results = {
        "btc_ov_log_return":     np.full(n, np.nan),
        "eth_ov_log_return":     np.full(n, np.nan),
        "sol_ov_log_return":     np.full(n, np.nan),
        "bnb_ov_log_return":     np.full(n, np.nan),
        "xrp_ov_log_return":     np.full(n, np.nan),
        "btc_ov_realized_vol":   np.full(n, np.nan),
        "eth_ov_realized_vol":   np.full(n, np.nan),
        "btc_ov_max_drawdown":   np.full(n, np.nan),
        "btc_ov_volume_usd":     np.full(n, np.nan),
        "btc_ov_volume_surge":   np.full(n, np.nan),
        "btc_ov_taker_imbalance": np.full(n, np.nan),
        "crosspair_dispersion":  np.full(n, np.nan),
        "btc_eth_spread":        np.full(n, np.nan),
        "btc_funding_rate_latest": np.full(n, np.nan),
        "btc_funding_rate_delta":  np.full(n, np.nan),
    }

    # Precompute rolling 7-day BTC volume (indexed by window date)
    # We'll fill btc_ov_volume_usd first, then compute surge
    btc_spot = spot_klines["BTCUSDT"]
    eth_spot = spot_klines["ETHUSDT"]
    sol_spot = spot_klines["SOLUSDT"]
    bnb_spot = spot_klines["BNBUSDT"]
    xrp_spot = spot_klines["XRPUSDT"]
    btc_fund = funding_rates["BTCUSDT"]

    log.info("Computing per-window crypto features (this may take a few minutes)...")

    for i, row in enumerate(windows.itertuples(index=False)):
        ws = row.window_start_utc
        we = row.window_end_utc

        # BTC
        btc_slice = slice_klines(btc_spot, ws, we)
        if len(btc_slice) > 0:
            results["btc_ov_log_return"][i]   = log_return_from_endpoints(btc_slice)
            results["btc_ov_realized_vol"][i] = realized_vol(btc_slice)
            results["btc_ov_max_drawdown"][i] = max_drawdown(btc_slice)
            results["btc_ov_volume_usd"][i]   = btc_slice["quote_volume"].sum()
            results["btc_ov_taker_imbalance"][i] = taker_imbalance(btc_slice)

        # ETH
        eth_slice = slice_klines(eth_spot, ws, we)
        if len(eth_slice) > 0:
            results["eth_ov_log_return"][i]   = log_return_from_endpoints(eth_slice)
            results["eth_ov_realized_vol"][i] = realized_vol(eth_slice)

        # SOL (NaN pre-2020-08 — handled naturally)
        sol_slice = slice_klines(sol_spot, ws, we)
        if len(sol_slice) > 0:
            results["sol_ov_log_return"][i] = log_return_from_endpoints(sol_slice)

        # BNB
        bnb_slice = slice_klines(bnb_spot, ws, we)
        if len(bnb_slice) > 0:
            results["bnb_ov_log_return"][i] = log_return_from_endpoints(bnb_slice)

        # XRP
        xrp_slice = slice_klines(xrp_spot, ws, we)
        if len(xrp_slice) > 0:
            results["xrp_ov_log_return"][i] = log_return_from_endpoints(xrp_slice)

        # BTC funding rate (most recent reset BEFORE window end)
        fund_before = btc_fund[btc_fund["funding_time"] < we]
        if len(fund_before) >= 2:
            results["btc_funding_rate_latest"][i] = fund_before["funding_rate"].iloc[-1]
            results["btc_funding_rate_delta"][i]  = (
                fund_before["funding_rate"].iloc[-1] - fund_before["funding_rate"].iloc[-2]
            )
        elif len(fund_before) == 1:
            results["btc_funding_rate_latest"][i] = fund_before["funding_rate"].iloc[-1]

        if i % 200 == 0:
            log.info(f"  Processed {i}/{n} windows...")

    df = pd.DataFrame(results, index=windows.index)

    # Cross-pair dispersion (stdev of available returns)
    return_cols = [
        "btc_ov_log_return", "eth_ov_log_return",
        "sol_ov_log_return", "bnb_ov_log_return", "xrp_ov_log_return",
    ]
    df["crosspair_dispersion"] = df[return_cols].std(axis=1, skipna=True)
    df["btc_eth_spread"]       = df["btc_ov_log_return"] - df["eth_ov_log_return"]

    # Volume surge: btc_ov_volume_usd / trailing 7-day mean (excluding current day)
    windows_reset = windows.reset_index(drop=True)
    dates = pd.to_datetime(windows_reset["date"])
    vol_series = pd.Series(df["btc_ov_volume_usd"].values, index=dates)
    vol_series = vol_series.sort_index()
    # 7-day rolling mean shifted by 1 (excludes current day)
    rolling_mean = vol_series.rolling(7, min_periods=4).mean().shift(1)
    df["btc_ov_volume_surge"] = (
        df["btc_ov_volume_usd"].values / rolling_mean.values
    )

    return df


# ── Macro feature join ─────────────────────────────────────────────────────────
def attach_macro_features(windows: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    """Join 1-day-lagged macro features to the windows DataFrame."""
    # windows has a 'date' column (trading day T)
    # we want FRED values at T-1 (1-day lag)
    fred_df = fred.reset_index()
    fred_df.columns = ["fred_date"] + [c for c in fred_df.columns if c != "fred_date"][: len(fred_df.columns) - 1]
    # Re-read to be safe
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

    # Build a lookup: for each date T, get FRED row at T-1
    window_dates = pd.to_datetime(windows["date"])
    macro_rows = []
    for d in window_dates:
        # find latest FRED date <= d-1
        lag_date = d - pd.Timedelta(days=1)
        avail = fred_df[fred_df.index <= lag_date]
        if len(avail) == 0:
            macro_rows.append({v: np.nan for v in macro_cols.values()})
        else:
            last = avail.iloc[-1]
            macro_rows.append({v: last[k] for k, v in macro_cols.items()})

    macro_df = pd.DataFrame(macro_rows, index=windows.index)
    return macro_df


# ── Stock feature helpers ──────────────────────────────────────────────────────
def load_hk_price_history() -> dict[str, pd.DataFrame]:
    """
    Load Stooq HK daily OHLCV for each universe ticker.
    Returns dict keyed by ticker (e.g. '0285.HK').
    """
    hk_dir = DATA / "stooq" / "hk_daily"
    univ = pd.read_csv(ROOT / "output" / "universe_log.csv")
    hk_tickers = univ[univ["market"] == "HK"]["ticker"].unique()

    price_map = {}
    for ticker in hk_tickers:
        num = ticker.replace(".HK", "").lstrip("0")
        fpath = hk_dir / f"{num}.hk.txt"
        if not fpath.exists():
            log.warning(f"HK ticker {ticker}: file {fpath} not found, skipping")
            continue
        df = pd.read_csv(fpath, header=0)
        df.columns = [c.strip("<>").lower() for c in df.columns]
        df = df.rename(columns={"date": "date", "open": "open", "high": "high",
                                  "low": "low", "close": "close", "vol": "volume"})
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        df = df.sort_values("date").set_index("date")
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        price_map[ticker] = df

    log.info(f"HK price history loaded: {len(price_map)} tickers")
    return price_map


def load_kr_price_history() -> dict[str, pd.DataFrame]:
    """Load pykrx KR daily OHLCV per ticker."""
    df = pd.read_parquet(DATA / "pykrx" / "kr_daily" / "kr_ohlcv_mcap.parquet")
    df["date"] = pd.to_datetime(df["date"])
    price_map = {}
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").set_index("date")
        price_map[ticker] = grp[["open", "high", "low", "close", "volume"]].astype(float)
    log.info(f"KR price history loaded: {len(price_map)} tickers")
    return price_map


def compute_stock_features(date: pd.Timestamp, ticker: str, price_map: dict) -> dict:
    """
    Compute 3 stock-level features for a given ticker on date T.
    - stock_rv_20d: trailing 20-day annualized realized vol of daily log returns (ending T-1)
    - stock_ret_20d: trailing 20-day cumulative log return (ending T-1)
    - stock_prior_day_return: log return on T-1

    All use data BEFORE date T (no lookahead).
    """
    empty = {"stock_rv_20d": np.nan, "stock_ret_20d": np.nan, "stock_prior_day_return": np.nan}
    if ticker not in price_map:
        return empty

    hist = price_map[ticker]
    # Filter to dates strictly before T
    hist_before = hist[hist.index < date]

    if len(hist_before) < 2:
        return empty

    log_rets = np.log(hist_before["close"] / hist_before["close"].shift(1)).dropna()

    if len(log_rets) == 0:
        return empty

    # Prior-day return: last available log return before T
    prior_day_ret = float(log_rets.iloc[-1])

    # 20-day trailing: last 20 observations
    trail = log_rets.iloc[-20:]
    if len(trail) < 5:  # need at least 5 to compute vol
        rv_20d = np.nan
        ret_20d = np.nan
    else:
        rv_20d  = float(trail.std() * np.sqrt(252))  # daily → annual
        ret_20d = float(trail.sum())                   # cumulative log return

    return {
        "stock_rv_20d":          rv_20d,
        "stock_ret_20d":         ret_20d,
        "stock_prior_day_return": prior_day_ret,
    }


# ── Target computation ─────────────────────────────────────────────────────────
def compute_targets(date: pd.Timestamp, ticker: str, price_map: dict) -> dict:
    """
    Compute 3 targets using prices on date T.
    - tgt_gap:      log(open_T / close_{T-1})
    - tgt_intraday: log(close_T / open_T)
    - tgt_cc:       log(close_T / close_{T-1})

    Both T and T-1 data come from the equity price history.
    """
    empty = {"tgt_gap": np.nan, "tgt_intraday": np.nan, "tgt_cc": np.nan}
    if ticker not in price_map:
        return empty

    hist = price_map[ticker]
    # Need row for date T
    if date not in hist.index:
        return empty

    today = hist.loc[date]
    # Get prior trading day (latest row before date T)
    hist_before = hist[hist.index < date]
    if len(hist_before) == 0:
        return empty

    prev = hist_before.iloc[-1]

    if prev["close"] <= 0 or today["open"] <= 0 or today["close"] <= 0:
        return empty

    return {
        "tgt_gap":      float(np.log(today["open"]  / prev["close"])),
        "tgt_intraday": float(np.log(today["close"] / today["open"])),
        "tgt_cc":       float(np.log(today["close"] / prev["close"])),
    }


# ── Universe expansion: per-day tickers ────────────────────────────────────────
def build_universe_map(market: str) -> dict:
    """
    Build a mapping: date -> list of tickers in universe for that date.
    Apply the most recent monthly rebalance universe to each trading day.
    """
    univ = pd.read_csv(ROOT / "output" / "universe_log.csv")
    univ = univ[univ["market"] == market].copy()
    univ["date"] = pd.to_datetime(univ["date"])

    # Get sorted rebalance dates
    rebal_dates = sorted(univ["date"].unique())

    # Build dict: rebal_date -> ticker list
    rebal_map = {}
    for rd in rebal_dates:
        tickers = univ[univ["date"] == rd]["ticker"].tolist()
        rebal_map[pd.Timestamp(rd).date()] = tickers

    return rebal_map


def get_tickers_for_date(date: pd.Timestamp, rebal_map: dict) -> list:
    """Return universe tickers for date, using most recent rebalance."""
    d = date.date() if hasattr(date, "date") else date
    # Find the latest rebalance date <= d
    eligible = [rd for rd in rebal_map.keys() if rd <= d]
    if not eligible:
        return []
    latest = max(eligible)
    return rebal_map[latest]


# ── Main assembly ──────────────────────────────────────────────────────────────
def build_features_for_market(
    market: str,
    windows: pd.DataFrame,
    crypto_feat: pd.DataFrame,
    macro_feat: pd.DataFrame,
    price_map: dict,
) -> pd.DataFrame:
    """
    Expand windows to (date, ticker) rows and attach all features + targets.
    """
    rebal_map = build_universe_map(market)

    rows = []
    n_windows = len(windows)
    log.info(f"{market}: expanding windows to ticker rows...")

    for i, (_, wrow) in enumerate(windows.iterrows()):
        date = pd.Timestamp(wrow["date"])
        next_date = pd.Timestamp(wrow["next_date"])

        tickers = get_tickers_for_date(date, rebal_map)
        if not tickers:
            continue

        # Crypto features (same for all tickers on this date)
        cf = crypto_feat.iloc[i].to_dict()
        # Macro features
        mf = macro_feat.iloc[i].to_dict()

        for ticker in tickers:
            # Stock features (uses price history up to date T-1)
            sf = compute_stock_features(next_date, ticker, price_map)

            # Targets (uses next_date = T+1 open/close relative to T close)
            # Actually: the "date" column in output represents the trading day
            # whose OPEN gap we're predicting. The equity session that opens is next_date.
            # tgt_gap = log(open_{next_date} / close_{date}) -- but in Stooq we
            # have daily bars per session. So:
            # - close_{T} = last close before the overnight window ends = price on 'date'
            # - open_{T+1} = first price of next_date session
            # We compute targets using next_date's open/close and date's close.
            tgt = compute_targets_crossday(date, next_date, ticker, price_map)

            row = {
                "date":       next_date,  # prediction date = the equity open day
                "window_date": date,      # the day whose overnight window was used
                "ticker":     ticker,
                "market":     market,
                "is_weekend_gap": wrow["is_weekend_gap"],
                **cf,
                **mf,
                **sf,
                **tgt,
            }
            rows.append(row)

        if i % 100 == 0:
            log.info(f"  {market}: {i}/{n_windows} windows expanded")

    df = pd.DataFrame(rows)
    log.info(f"{market}: final rows = {len(df):,}")
    return df


def compute_targets_crossday(
    prev_date: pd.Timestamp,
    curr_date: pd.Timestamp,
    ticker: str,
    price_map: dict,
) -> dict:
    """
    tgt_gap:      log(open_{curr_date} / close_{prev_date})
    tgt_intraday: log(close_{curr_date} / open_{curr_date})
    tgt_cc:       log(close_{curr_date} / close_{prev_date})
    """
    empty = {"tgt_gap": np.nan, "tgt_intraday": np.nan, "tgt_cc": np.nan}
    if ticker not in price_map:
        return empty

    hist = price_map[ticker]

    if curr_date not in hist.index:
        return empty
    if prev_date not in hist.index:
        # Fall back to latest date strictly before curr_date
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


# ── Validation ─────────────────────────────────────────────────────────────────
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
NON_PERP_CRYPTO = [c for c in CRYPTO_FEAT_COLS if c not in PERP_COLS]


def validate(df: pd.DataFrame, market: str) -> bool:
    """Run all validation checks. Returns False if STOP condition met."""
    log.info(f"\n{'='*60}")
    log.info(f"VALIDATION: {market}")
    log.info(f"{'='*60}")
    log.info(f"Total rows: {len(df):,}")

    ok = True

    # Feature distribution
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
            f"min={s.min():+.4f}  max={s.max():+.4f}  NaN={nan_pct:.1f}%"
            + (" [perp, NaN expected pre-2020]" if is_perp else "")
        )
        # STOP: >50% NaN for non-perp features
        if not is_perp and nan_pct > 50:
            log.error(f"  STOP: {col} has {nan_pct:.1f}% NaN (>50% threshold)")
            ok = False
        # STOP: zero variance
        if s.std() == 0 and not s.isna().all():
            log.error(f"  STOP: {col} has zero variance")
            ok = False

    # Target availability
    log.info("\nTarget availability:")
    for tgt in TARGET_COLS:
        if tgt not in df.columns:
            log.warning(f"  MISSING TARGET: {tgt}")
            continue
        non_nan = df[tgt].notna().sum()
        log.info(f"  {tgt}: {non_nan:,} non-NaN rows ({non_nan/len(df)*100:.1f}%)")

    # tgt_gap NaN check
    tgt_gap_nan = df["tgt_gap"].isna().sum()
    log.info(f"\ntgt_gap NaN rows: {tgt_gap_nan} "
             f"({tgt_gap_nan/len(df)*100:.1f}% — expected ~0% for traded dates)")

    # No-lookahead check: crypto features use window ending BEFORE equity open
    # The feature's window_end_utc (stored in windows) < equity open.
    # We've designed the windows to end at equity open time; this is structural.
    log.info("\nLookahead check: all windows end at/before equity open (structural guarantee via exchange_calendars)")

    # Prior-day stock return check: confirm stock_prior_day_return uses T-1
    log.info("Prior-day return check: stock_prior_day_return computed from hist[hist.index < date] (structural guarantee)")

    return ok


# ── Append to feature_decisions.log ───────────────────────────────────────────
def append_usdt_drop_note():
    fd_path = LOGS / "feature_decisions.log"
    note = (
        "\n=== USDT PEG FEATURE DROPPED ===\n"
        "Date: 2026-04-16\n"
        "Trigger: yfinance BTC-USD is daily resolution only; "
        "intraday BTC/USD minute-level data unavailable for free.\n"
        "Could not compute clean intraday USDT/USD peg deviation within the overnight window.\n"
        "Decision: DROP this feature in Stage 3. "
        "Note in WRITEUP.md Data & Limitations as Pass 2 candidate (paid source: Kaiko, CCData).\n"
    )
    already_noted = False
    if fd_path.exists():
        with open(fd_path, "r") as f:
            if "USDT PEG FEATURE DROPPED" in f.read():
                already_noted = True
    if not already_noted:
        with open(fd_path, "a") as f:
            f.write(note)
        log.info("Appended USDT peg drop note to feature_decisions.log")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    log.info("Stage 3 Feature Engineering — Track A")
    log.info(f"Window: {WINDOW_START} to {WINDOW_END}")

    append_usdt_drop_note()

    # Load data
    log.info("\nLoading Binance data...")
    spot_klines  = load_spot_klines()
    perp_klines  = load_perp_klines()
    funding_rates = load_funding_rates()

    log.info("\nLoading FRED macro data...")
    fred = load_fred()

    log.info("\nLoading equity price histories...")
    hk_prices = load_hk_price_history()
    kr_prices = load_kr_price_history()

    # Build overnight windows for each market
    log.info("\nBuilding overnight windows...")
    hk_windows = build_overnight_windows("HK")
    kr_windows = build_overnight_windows("KR")

    # Save overnight window log (combined)
    hk_windows["market"] = "HK"
    kr_windows["market"] = "KR"
    window_log = pd.concat([hk_windows, kr_windows], ignore_index=True)
    window_log_path = OUT / "overnight_window_log.csv"
    window_log.to_csv(window_log_path, index=False)
    log.info(f"Overnight window log saved: {window_log_path}")

    # Compute crypto features (expensive: iterates over windows)
    log.info("\nComputing crypto features for HK windows...")
    hk_crypto = compute_crypto_features(hk_windows, spot_klines, perp_klines, funding_rates)

    log.info("\nComputing crypto features for KR windows...")
    kr_crypto = compute_crypto_features(kr_windows, spot_klines, perp_klines, funding_rates)

    # Compute macro features
    log.info("\nAttaching macro features...")
    hk_macro = attach_macro_features(hk_windows, fred)
    kr_macro = attach_macro_features(kr_windows, fred)

    # Build full feature tables
    log.info("\nBuilding HK full feature table...")
    hk_df = build_features_for_market("HK", hk_windows, hk_crypto, hk_macro, hk_prices)

    log.info("\nBuilding KR full feature table...")
    kr_df = build_features_for_market("KR", kr_windows, kr_crypto, kr_macro, kr_prices)

    # Drop internal helper columns before saving
    for df in [hk_df, kr_df]:
        if "window_date" in df.columns:
            df.drop(columns=["window_date"], inplace=True)
        # Remove internal return columns not in the final spec
        for col in ["sol_ov_log_return", "bnb_ov_log_return", "xrp_ov_log_return"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    # Validate
    log.info("\n" + "="*60)
    hk_ok = validate(hk_df, "HK")
    kr_ok = validate(kr_df, "KR")

    if not hk_ok or not kr_ok:
        log.error("STOP CONDITION MET. Halting before saving output.")
        sys.exit(1)

    # Save
    hk_path = OUT / "features_track_a_hk.parquet"
    kr_path = OUT / "features_track_a_kr.parquet"
    hk_df.to_parquet(hk_path, index=False)
    kr_df.to_parquet(kr_path, index=False)
    log.info(f"\nSaved: {hk_path}")
    log.info(f"Saved: {kr_path}")

    log.info("\n" + "="*60)
    log.info("Stage 3 complete.")
    log.info(f"HK rows: {len(hk_df):,}")
    log.info(f"KR rows: {len(kr_df):,}")


if __name__ == "__main__":
    main()
