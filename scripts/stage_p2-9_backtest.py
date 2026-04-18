# ABOUTME: Stage P2-9 — full backtest suite with borrow cost, regime gate, long-only, and sensitivity sweeps.
# ABOUTME: Inputs: output/predictions_*.csv, output/features_track_a_*.parquet, data/derived/*.parquet.
#           Output: output/backtest_*.csv, output/backtest_summary_pass2.csv, output/cost_sensitivity_pass2.csv,
#                   output/borrow_sensitivity_pass2.csv, output/breakeven_analysis_pass2.csv.
#           Run: source .venv/bin/activate && python3 scripts/stage_p2-9_backtest.py

import pandas as pd
import numpy as np
import warnings
import sys
import os
import time

warnings.filterwarnings("ignore")

PROJECT = "/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity_p2"
OUTPUT = os.path.join(PROJECT, "output")
LOGS = os.path.join(PROJECT, "logs")
DATA = os.path.join(PROJECT, "data")

ANN = 252.0
MIN_UNIVERSE_STOCK = 9   # go flat if fewer stocks in universe
IMPACT_CONST = 0.1
TRADE_SIZE_USD = 100_000.0

# ─── cost defaults (per brief) ────────────────────────────────────────────────
# Half-spread per side (decimal)
HALF_SPREAD = {
    "main_hk":    15e-4,
    "main_kr":    10e-4,
    "control_hk": 10e-4,
    "control_kr":  5e-4,
    "index_hk":    2e-4,
    "index_kr":    2e-4,
}

# Annual borrow rate (decimal); applied to short notional only
ANNUAL_BORROW = {
    "main_hk":    500e-4,
    "main_kr":    400e-4,
    "control_hk": 300e-4,
    "control_kr": 250e-4,
    "index_hk":    50e-4,
    "index_kr":    50e-4,
}
DAILY_BORROW = {k: v / ANN for k, v in ANNUAL_BORROW.items()}

# Index threshold half-sigma multiplier for entry/exit
INDEX_THRESHOLD_MULT = 0.5

# Regime gate warmup days
REGIME_WARMUP = 282   # 252 median + 30 ratio

# Sensitivity sweep multipliers
SPREAD_MULTS  = [0.5, 1.0, 1.5, 2.0]
BORROW_MULTS  = [0.0, 1.0, 1.5, 2.0]


# ─── universe → prediction file mapping ─────────────────────────────────────

def pred_file(model: str, universe: str, target: str) -> str:
    """Return path to prediction CSV for a given (model, universe, target) combo."""
    # TCN uses "main_hk" / "main_kr" naming in filenames
    if model == "lgbm":
        # universe is one of: main_hk, main_kr, control_hk, control_kr, index_hk, index_kr
        # but predictions_lgbm_{hk,kr}_{target}.csv for main; others have full name
        if universe in ("main_hk", "main_kr"):
            market = universe.split("_")[1]  # hk or kr
            fname = f"predictions_lgbm_{market}_{target}.csv"
        else:
            fname = f"predictions_lgbm_{universe}_{target}.csv"
    elif model == "tcn":
        fname = f"predictions_tcn_{universe}_{target}.csv"
    else:
        raise ValueError(f"Unknown model: {model}")
    return os.path.join(OUTPUT, fname)


# ─── cost data loading ────────────────────────────────────────────────────────

def load_stock_cost_data(universe: str) -> pd.DataFrame:
    """Load (date, ticker) indexed DataFrame with adv_usd and stock_rv_20d."""
    # Determine universe log and features parquet
    if universe in ("main_hk", "main_kr"):
        market = universe.split("_")[1].upper()
        ul = pd.read_csv(os.path.join(OUTPUT, "universe_log.csv"), parse_dates=["date"])
        ul = ul[ul["market"] == market][["date", "ticker", "adv_usd"]].copy()
        feat_file = f"features_track_a_{universe.split('_')[1]}.parquet"
    elif universe in ("control_hk", "control_kr"):
        market_suffix = universe.split("_")[1]
        market = market_suffix.upper()
        ul = pd.read_csv(os.path.join(OUTPUT, "control_universe_log.csv"), parse_dates=["date"])
        ul = ul[ul["market"] == market][["date", "ticker", "adv_usd"]].copy()
        feat_file = f"features_track_a_{universe}.parquet"
    else:
        return None  # index — no stock-level cost data needed

    feat = pd.read_parquet(
        os.path.join(OUTPUT, feat_file),
        columns=["date", "ticker", "stock_rv_20d"]
    )
    feat["date"] = pd.to_datetime(feat["date"])

    # Forward-fill monthly ADV to daily
    all_dates = feat[["date"]].drop_duplicates().sort_values("date")
    tickers = ul["ticker"].unique()
    grid = pd.MultiIndex.from_product([all_dates["date"], tickers], names=["date", "ticker"])
    ul_ff = ul.set_index(["date", "ticker"]).reindex(grid).reset_index()
    ul_ff = ul_ff.sort_values(["ticker", "date"])
    ul_ff["adv_usd"] = ul_ff.groupby("ticker")["adv_usd"].ffill().bfill()

    feat = feat.sort_values(["ticker", "date"])
    cost_df = ul_ff.merge(feat, on=["date", "ticker"], how="left")
    cost_df["stock_rv_20d"] = cost_df.groupby("ticker")["stock_rv_20d"].ffill().bfill()
    return cost_df.set_index(["date", "ticker"])


def compute_impact(adv_usd: float, daily_vol_ann: float) -> float:
    """Market impact per side (decimal). daily_vol_ann is annualized rv."""
    if (adv_usd is None or pd.isna(adv_usd) or adv_usd <= 0
            or daily_vol_ann is None or pd.isna(daily_vol_ann)):
        return 0.0
    daily_vol = daily_vol_ann / np.sqrt(ANN)
    return IMPACT_CONST * np.sqrt(TRADE_SIZE_USD / adv_usd) * daily_vol


# ─── regime gate ─────────────────────────────────────────────────────────────

def build_regime_series() -> pd.Series:
    """
    Compute daily regime gate: 1 if btc_mcap_30d / sp500_mcap_30d > 1y trailing median.
    Returns Series indexed by date (pandas Timestamp).
    First REGIME_WARMUP days default to 1.
    """
    # BTC close from 1-minute klines — resample to daily close
    btc_klines = pd.read_parquet(
        os.path.join(DATA, "binance", "spot_klines", "BTCUSDT_1m.parquet"),
        columns=["open_time", "close"]
    )
    btc_klines["date"] = btc_klines["open_time"].dt.normalize().dt.tz_localize(None)
    btc_daily = btc_klines.groupby("date")["close"].last().reset_index()
    btc_daily.columns = ["date", "btc_close"]

    btc_supply = pd.read_parquet(os.path.join(DATA, "derived", "btc_supply.parquet"))
    btc_supply["date"] = pd.to_datetime(btc_supply["date"]).dt.tz_localize(None)

    sp500 = pd.read_parquet(os.path.join(DATA, "derived", "sp500_mcap_proxy.parquet"))
    sp500["date"] = pd.to_datetime(sp500["date"]).dt.tz_localize(None)

    # Merge to daily panel
    regime = btc_daily.merge(btc_supply, on="date", how="inner")
    regime = regime.merge(sp500[["date", "sp500_mcap_proxy"]], on="date", how="inner")
    regime = regime.sort_values("date").reset_index(drop=True)

    # BTC mcap = price * supply
    regime["btc_mcap"] = regime["btc_close"] * regime["btc_supply"]

    # 30-day rolling means
    regime["btc_mcap_30d"]  = regime["btc_mcap"].rolling(30, min_periods=1).mean()
    regime["sp500_mcap_30d"] = regime["sp500_mcap_proxy"].rolling(30, min_periods=1).mean()
    regime["ratio"] = regime["btc_mcap_30d"] / regime["sp500_mcap_30d"]

    # 1-year (252-day) trailing median of ratio, excluding day T
    regime["ratio_1y_median"] = regime["ratio"].shift(1).rolling(252, min_periods=1).median()

    # Gate = 1 if ratio > 1y median, else 0
    regime["gate"] = (regime["ratio"] > regime["ratio_1y_median"]).astype(int)

    # Apply warmup default = 1
    regime.loc[regime.index < REGIME_WARMUP, "gate"] = 1

    regime["date"] = pd.to_datetime(regime["date"])
    return regime.set_index("date")["gate"]


# ─── stock tercile backtest ───────────────────────────────────────────────────

def run_stock_backtest(
    universe: str,
    target: str,
    model: str,
    cost_data,
    spread_mult: float = 1.0,
    borrow_mult: float = 1.0,
    strategy: str = "long_short",   # "long_short" or "long_only"
    gate_series: pd.Series = None,  # None = gate off; Series = gate on
) -> pd.DataFrame:
    """
    Daily tercile backtest for stock universes.
    Returns per-day DataFrame with returns and cost decomposition.
    """
    fpath = pred_file(model, universe, target)
    pred = pd.read_csv(fpath, parse_dates=["date"])
    pred = pred.sort_values(["date", "ticker"]).reset_index(drop=True)

    half_spread = HALF_SPREAD[universe] * spread_mult
    daily_borrow = DAILY_BORROW[universe] * borrow_mult

    records = []
    prev_long = set()
    prev_short = set()

    for date, day_df in pred.groupby("date"):
        # Regime gate check
        gate = 1
        if gate_series is not None:
            dt = pd.Timestamp(date)
            if dt in gate_series.index:
                gate = int(gate_series.loc[dt])
            # dates before regime series start default to 1 (warmup)

        n = len(day_df)
        if n < MIN_UNIVERSE_STOCK or gate == 0:
            records.append({
                "date": date,
                "long_leg_return": 0.0,
                "short_leg_return": 0.0,
                "gross_return": 0.0,
                "turnover": 0.0,
                "spread_cost": 0.0,
                "impact_cost": 0.0,
                "borrow_cost": 0.0,
                "net_return": 0.0,
                "gate": gate,
                "flat": True,
            })
            prev_long, prev_short = set(), set()
            continue

        day_df = day_df.sort_values("y_pred").reset_index(drop=True)
        tercile = n // 3
        short_set = set(day_df.iloc[:tercile]["ticker"])
        long_set  = set(day_df.iloc[n - tercile:]["ticker"])

        # Returns
        long_ret  = day_df[day_df["ticker"].isin(long_set)]["y_actual"].mean()
        short_ret = day_df[day_df["ticker"].isin(short_set)]["y_actual"].mean()

        if strategy == "long_only":
            # Long top-third vs equal-weight benchmark (full universe)
            bm_ret = day_df["y_actual"].mean()
            gross_ret = long_ret - bm_ret
        else:
            gross_ret = long_ret - short_ret

        # Turnover
        long_overlap  = len(long_set & prev_long)  / max(len(long_set),  1)
        short_overlap = len(short_set & prev_short) / max(len(short_set), 1)
        turnover = 1.0 - (long_overlap + short_overlap) / 2.0

        # Changed names — entry or exit
        long_entries  = long_set  - prev_long
        long_exits    = prev_long - long_set
        short_entries = short_set  - prev_short
        short_exits   = prev_short - short_set
        changed = long_entries | long_exits | short_entries | short_exits

        # Spread + impact cost
        spread_cost_total = 0.0
        impact_cost_total = 0.0
        total_legs = max(len(long_set), 1) + (max(len(short_set), 1) if strategy == "long_short" else 0)

        for tkr in changed:
            key = (date, tkr)
            if cost_data is not None and key in cost_data.index:
                row = cost_data.loc[key]
                adv = float(row["adv_usd"])
                rv  = float(row["stock_rv_20d"])
            else:
                adv, rv = np.nan, np.nan

            impact = compute_impact(adv, rv) * spread_mult  # impact scales with spread_mult
            # Per-stock round-trip cost weighted by portfolio weight 1/tercile per leg
            weight = 1.0 / (tercile * (2 if strategy == "long_short" else 1))
            spread_cost_total += half_spread * 2 * weight
            impact_cost_total += impact * 2 * weight

        # Borrow cost: applied daily to short notional; no borrow in long-only
        if strategy == "long_only":
            borrow_cost = 0.0
        else:
            # Short leg = 1.0 notional (equal weight across tercile stocks)
            borrow_cost = daily_borrow

        cost_total = spread_cost_total + impact_cost_total + borrow_cost
        net_ret = gross_ret - cost_total

        records.append({
            "date": date,
            "long_leg_return": long_ret,
            "short_leg_return": short_ret,
            "gross_return": gross_ret,
            "turnover": turnover,
            "spread_cost": spread_cost_total,
            "impact_cost": impact_cost_total,
            "borrow_cost": borrow_cost,
            "net_return": net_ret,
            "gate": gate,
            "flat": False,
        })
        prev_long, prev_short = long_set, short_set

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["cumulative_net"] = df["net_return"].cumsum()
    return df


# ─── index threshold backtest ─────────────────────────────────────────────────

def run_index_backtest(
    universe: str,
    target: str,
    model: str,
    spread_mult: float = 1.0,
    borrow_mult: float = 1.0,
    gate_series: pd.Series = None,
) -> pd.DataFrame:
    """
    Threshold long/short/flat strategy for index universes.
    Threshold = ±0.5 * stdev(y_pred) over the training window (fold rolling).
    """
    fpath = pred_file(model, universe, target)
    pred = pd.read_csv(fpath, parse_dates=["date"])
    pred = pred.sort_values("date").reset_index(drop=True)

    half_spread = HALF_SPREAD[universe] * spread_mult
    daily_borrow = DAILY_BORROW[universe] * borrow_mult

    # Compute threshold per fold using training window stdev
    # For each fold, threshold = 0.5 * stdev(y_pred across all earlier folds)
    # We compute it as a rolling expanding stdev up to the current fold start
    folds = pred["fold_id"].unique()
    # Use expanding stdev: at fold f, stdev of all y_pred before fold f
    fold_pred_stdev = {}
    for f in sorted(folds):
        prior = pred[pred["fold_id"] < f]["y_pred"]
        if len(prior) >= 10:
            fold_pred_stdev[f] = prior.std()
        else:
            # fallback: use stdev of current fold's y_pred
            fold_pred_stdev[f] = pred[pred["fold_id"] == f]["y_pred"].std()
    # Broadcast stdev back to each row
    pred["ypred_stdev"] = pred["fold_id"].map(fold_pred_stdev)
    pred["threshold"] = INDEX_THRESHOLD_MULT * pred["ypred_stdev"]

    records = []
    prev_pos = 0  # -1=short, 0=flat, 1=long

    for _, row in pred.iterrows():
        date = row["date"]

        # Regime gate
        gate = 1
        if gate_series is not None:
            dt = pd.Timestamp(date)
            if dt in gate_series.index:
                gate = int(gate_series.loc[dt])

        y_pred   = row["y_pred"]
        y_actual = row["y_actual"]
        threshold = row["threshold"]

        if gate == 0 or pd.isna(threshold) or threshold == 0:
            # Gate closed or no threshold — go flat
            records.append({
                "date": date,
                "y_pred": y_pred,
                "y_actual": y_actual,
                "position": 0,
                "gross_return": 0.0,
                "spread_cost": 0.0,
                "borrow_cost": 0.0,
                "net_return": 0.0,
                "gate": gate,
                "flat": True,
            })
            prev_pos = 0
            continue

        # Determine position
        if y_pred > threshold:
            pos = 1
        elif y_pred < -threshold:
            pos = -1
        else:
            pos = 0

        # Gross return
        gross_ret = pos * y_actual

        # Turnover-based spread cost (only on position changes)
        if pos != prev_pos:
            # Round-trip per side = half_spread * 2
            spread_cost = half_spread * 2
        else:
            spread_cost = 0.0

        # Borrow cost for short position
        borrow_cost = daily_borrow if pos == -1 else 0.0

        net_ret = gross_ret - spread_cost - borrow_cost

        records.append({
            "date": date,
            "y_pred": y_pred,
            "y_actual": y_actual,
            "position": pos,
            "gross_return": gross_ret,
            "spread_cost": spread_cost,
            "borrow_cost": borrow_cost,
            "net_return": net_ret,
            "gate": gate,
            "flat": (pos == 0),
        })
        prev_pos = pos

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["impact_cost"] = 0.0   # no impact model for index
    df["turnover"] = (df["position"].diff().abs() > 0).astype(float)
    df["cumulative_net"] = df["net_return"].cumsum()
    return df


# ─── metrics ─────────────────────────────────────────────────────────────────

def metrics(df: pd.DataFrame) -> dict:
    """Compute performance metrics from a daily backtest DataFrame."""
    ret_g = df["gross_return"]
    ret_n = df["net_return"]
    pct_inv = (~df["flat"]).sum() / max(len(df), 1) * 100

    ann_ret_g = ret_g.sum() * ANN / max(len(df), 1)
    ann_ret_n = ret_n.sum() * ANN / max(len(df), 1)
    ann_vol   = ret_n.std() * np.sqrt(ANN)

    gross_sharpe = ann_ret_g / (ret_g.std() * np.sqrt(ANN)) if ret_g.std() > 0 else np.nan
    net_sharpe   = ann_ret_n / ann_vol if ann_vol > 0 else np.nan

    cum      = df["cumulative_net"]
    roll_max = cum.cummax()
    dd       = cum - roll_max
    max_dd   = dd.min()
    calmar   = ann_ret_n / abs(max_dd) if max_dd < 0 else np.nan

    avg_to   = df["turnover"].mean()

    spread_total  = df["spread_cost"].sum()
    impact_total  = df.get("impact_cost", pd.Series(0.0, index=df.index)).sum()
    borrow_total  = df["borrow_cost"].sum()
    gross_pnl     = ret_g.sum()
    net_pnl       = ret_n.sum()
    residual      = net_pnl - gross_pnl + spread_total + impact_total + borrow_total

    return {
        "gross_sharpe":     gross_sharpe,
        "net_sharpe":       net_sharpe,
        "ann_return":       ann_ret_n,
        "max_drawdown":     max_dd,
        "calmar":           calmar,
        "daily_turnover":   avg_to,
        "pct_days_invested": pct_inv,
        "gross_pnl":        gross_pnl,
        "spread_cost":      spread_total,
        "impact_cost":      impact_total,
        "borrow_cost":      borrow_total,
        "residual":         residual,
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=== Stage P2-9: Backtest Suite (borrow + regime gate) ===\n")

    # ── Build regime gate ────────────────────────────────────────────────────
    log("Building regime gate series...")
    gate_series = build_regime_series()
    gate_on_frac = gate_series.mean()
    log(f"  Gate=1 fraction: {gate_on_frac:.3f}  "
        f"dates: {gate_series.index.min().date()} to {gate_series.index.max().date()}\n")

    # ── Load cost data ────────────────────────────────────────────────────────
    log("Loading cost data for stock universes...")
    cost_data = {}
    stock_universes = ["main_hk", "main_kr", "control_hk", "control_kr"]
    for u in stock_universes:
        cost_data[u] = load_stock_cost_data(u)
        log(f"  {u}: loaded {len(cost_data[u]):,} rows")

    log("")

    # ── Define full configuration grid ───────────────────────────────────────
    # Models × universes mapping
    CONFIGS = [
        # (model, universe, is_index)
        ("lgbm", "main_hk",    False),
        ("lgbm", "main_kr",    False),
        ("lgbm", "control_hk", False),
        ("lgbm", "control_kr", False),
        ("lgbm", "index_hk",   True),
        ("lgbm", "index_kr",   True),
        ("tcn",  "main_hk",    False),
        ("tcn",  "main_kr",    False),
        ("tcn",  "index_hk",   True),
        ("tcn",  "index_kr",   True),
    ]

    TARGETS = ["gap", "intraday", "cc"]
    STRATEGIES_STOCK = ["long_short", "long_only"]
    GATE_VARIANTS = ["gate_on", "gate_off"]

    # ── Primary backtest: 1x spread, 1x borrow ────────────────────────────────
    summary_rows = []
    log("Running primary backtests (1x spread, 1x borrow)...\n")

    for model, universe, is_index in CONFIGS:
        cd = cost_data.get(universe, None)

        for target in TARGETS:
            # Gate-on and gate-off
            for gate_label, gate_arg in [("gate_on", gate_series), ("gate_off", None)]:
                if is_index:
                    strategies = [("index_threshold", None)]
                else:
                    strategies = [
                        ("tercile_ls", "long_short"),
                        ("tercile_lo", "long_only"),
                    ]

                for strat_label, strat_arg in strategies:
                    cfg_str = f"{model}/{universe}/{target}/{strat_label}/{gate_label}"

                    if is_index:
                        df = run_index_backtest(
                            universe, target, model,
                            spread_mult=1.0, borrow_mult=1.0,
                            gate_series=gate_arg,
                        )
                    else:
                        df = run_stock_backtest(
                            universe, target, model, cd,
                            spread_mult=1.0, borrow_mult=1.0,
                            strategy=strat_arg,
                            gate_series=gate_arg,
                        )

                    m = metrics(df)

                    # Validation
                    if np.isnan(m["net_sharpe"]):
                        log(f"BLOCK: NaN net_sharpe for {cfg_str}")
                        sys.exit(1)

                    if m["net_sharpe"] != 0:
                        tol = 1e-6
                        expected_net_pnl = m["gross_pnl"] - m["spread_cost"] - m["impact_cost"] - m["borrow_cost"]
                        actual_net_pnl = df["net_return"].sum()
                        assert abs(actual_net_pnl - expected_net_pnl) < tol * (abs(actual_net_pnl) + 1), \
                            f"Net PnL validation failed for {cfg_str}"

                    if strat_label == "tercile_lo":
                        assert df["borrow_cost"].sum() == 0.0, \
                            f"Long-only should have zero borrow for {cfg_str}"

                    # Save per-day file
                    save_cols = ["date", "gross_return", "turnover",
                                 "spread_cost", "borrow_cost", "net_return", "cumulative_net"]
                    if "impact_cost" in df.columns:
                        save_cols.insert(4, "impact_cost")
                    if "gate" in df.columns:
                        save_cols.append("gate")

                    out_fname = f"backtest_{model}_{universe}_{target}_{strat_label}_{gate_label}.csv"
                    df[[c for c in save_cols if c in df.columns]].to_csv(
                        os.path.join(OUTPUT, out_fname), index=False
                    )

                    row = {
                        "model":            model,
                        "universe":         universe,
                        "target":           target,
                        "strategy":         strat_label,
                        "gate":             gate_label,
                        "gross_sharpe":     m["gross_sharpe"],
                        "net_sharpe":       m["net_sharpe"],
                        "ann_return":       m["ann_return"],
                        "max_drawdown":     m["max_drawdown"],
                        "daily_turnover":   m["daily_turnover"],
                        "calmar":           m["calmar"],
                        "pct_days_invested": m["pct_days_invested"],
                        "gross_pnl":        m["gross_pnl"],
                        "spread_cost":      m["spread_cost"],
                        "impact_cost":      m["impact_cost"],
                        "borrow_cost":      m["borrow_cost"],
                        "residual":         m["residual"],
                    }
                    summary_rows.append(row)

                    log(f"  {cfg_str}  "
                        f"net_SR={m['net_sharpe']:.3f}  "
                        f"ann_ret={m['ann_return']:.3f}  "
                        f"DD={m['max_drawdown']:.3f}  "
                        f"TO={m['daily_turnover']:.3f}")

    # Save primary summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT, "backtest_summary_pass2.csv"), index=False)
    log(f"\nPrimary summary: {len(summary_df)} rows written.")

    # ── Spread sensitivity sweep ──────────────────────────────────────────────
    log("\nRunning spread sensitivity sweep (borrow=1x)...\n")
    # Primary configs: main + control (stock, long_short) + index (index_threshold), gate_on only
    primary_configs = [
        (model, universe, is_index)
        for model, universe, is_index in CONFIGS
        if universe in ("main_hk", "main_kr", "index_hk", "index_kr")
    ]

    spread_rows = []
    for model, universe, is_index in primary_configs:
        cd = cost_data.get(universe, None)
        for target in TARGETS:
            for sm in SPREAD_MULTS:
                if is_index:
                    df = run_index_backtest(
                        universe, target, model,
                        spread_mult=sm, borrow_mult=1.0,
                        gate_series=gate_series,
                    )
                    strat = "index_threshold"
                else:
                    df = run_stock_backtest(
                        universe, target, model, cd,
                        spread_mult=sm, borrow_mult=1.0,
                        strategy="long_short",
                        gate_series=gate_series,
                    )
                    strat = "tercile_ls"
                m = metrics(df)
                spread_rows.append({
                    "model": model, "universe": universe, "target": target,
                    "strategy": strat, "spread_mult": sm, "borrow_mult": 1.0,
                    "net_sharpe": m["net_sharpe"],
                    "ann_return": m["ann_return"],
                    "max_drawdown": m["max_drawdown"],
                    "spread_cost": m["spread_cost"],
                    "borrow_cost": m["borrow_cost"],
                })

    spread_df = pd.DataFrame(spread_rows)
    spread_df.to_csv(os.path.join(OUTPUT, "cost_sensitivity_pass2.csv"), index=False)
    log(f"Spread sensitivity: {len(spread_df)} rows written.")

    # ── Borrow sensitivity sweep ──────────────────────────────────────────────
    log("\nRunning borrow sensitivity sweep (spread=1x)...\n")
    borrow_rows = []
    for model, universe, is_index in primary_configs:
        cd = cost_data.get(universe, None)
        for target in TARGETS:
            for bm in BORROW_MULTS:
                if is_index:
                    df = run_index_backtest(
                        universe, target, model,
                        spread_mult=1.0, borrow_mult=bm,
                        gate_series=gate_series,
                    )
                    strat = "index_threshold"
                else:
                    df = run_stock_backtest(
                        universe, target, model, cd,
                        spread_mult=1.0, borrow_mult=bm,
                        strategy="long_short",
                        gate_series=gate_series,
                    )
                    strat = "tercile_ls"
                m = metrics(df)
                borrow_rows.append({
                    "model": model, "universe": universe, "target": target,
                    "strategy": strat, "spread_mult": 1.0, "borrow_mult": bm,
                    "net_sharpe": m["net_sharpe"],
                    "ann_return": m["ann_return"],
                    "max_drawdown": m["max_drawdown"],
                    "spread_cost": m["spread_cost"],
                    "borrow_cost": m["borrow_cost"],
                })

    borrow_df = pd.DataFrame(borrow_rows)
    borrow_df.to_csv(os.path.join(OUTPUT, "borrow_sensitivity_pass2.csv"), index=False)
    log(f"Borrow sensitivity: {len(borrow_df)} rows written.")

    # ── Breakeven analysis ────────────────────────────────────────────────────
    log("\nComputing breakeven (spread_mult, borrow_mult) per primary config...\n")
    be_rows = []

    def find_breakeven_1d(vals, sharpes, axis_name):
        """Interpolate x at which sharpe crosses zero."""
        for i in range(len(sharpes) - 1):
            if sharpes[i] >= 0 and sharpes[i + 1] < 0:
                # linear interpolation
                x0, x1 = vals[i], vals[i + 1]
                s0, s1 = sharpes[i], sharpes[i + 1]
                return x0 + (x1 - x0) * s0 / (s0 - s1)
        if sharpes[-1] >= 0:
            return np.inf   # does not break even in sweep range
        return np.nan

    for model, universe, is_index in primary_configs:
        cd = cost_data.get(universe, None)
        for target in TARGETS:
            strat = "index_threshold" if is_index else "tercile_ls"

            # Spread breakeven
            spread_sharpes = []
            for sm in SPREAD_MULTS:
                if is_index:
                    df = run_index_backtest(universe, target, model,
                                            spread_mult=sm, borrow_mult=1.0,
                                            gate_series=gate_series)
                else:
                    df = run_stock_backtest(universe, target, model, cd,
                                            spread_mult=sm, borrow_mult=1.0,
                                            strategy="long_short",
                                            gate_series=gate_series)
                spread_sharpes.append(metrics(df)["net_sharpe"])
            be_spread = find_breakeven_1d(SPREAD_MULTS, spread_sharpes, "spread_mult")

            # Borrow breakeven
            borrow_sharpes = []
            for bm in BORROW_MULTS:
                if is_index:
                    df = run_index_backtest(universe, target, model,
                                            spread_mult=1.0, borrow_mult=bm,
                                            gate_series=gate_series)
                else:
                    df = run_stock_backtest(universe, target, model, cd,
                                            spread_mult=1.0, borrow_mult=bm,
                                            strategy="long_short",
                                            gate_series=gate_series)
                borrow_sharpes.append(metrics(df)["net_sharpe"])
            be_borrow = find_breakeven_1d(BORROW_MULTS, borrow_sharpes, "borrow_mult")

            be_rows.append({
                "model": model, "universe": universe, "target": target,
                "strategy": strat,
                "breakeven_spread_mult": be_spread,
                "breakeven_borrow_mult": be_borrow,
                "net_sharpe_base": spread_sharpes[SPREAD_MULTS.index(1.0)],
            })
            log(f"  {model}/{universe}/{target}: be_spread={be_spread:.2f}x  be_borrow={be_borrow:.2f}x")

    be_df = pd.DataFrame(be_rows)
    be_df.to_csv(os.path.join(OUTPUT, "breakeven_analysis_pass2.csv"), index=False)
    log(f"Breakeven analysis: {len(be_df)} rows written.")

    # ── Validation checks ─────────────────────────────────────────────────────
    log("\n=== Validation ===")
    # 1. No NaN net_sharpe
    nan_sharpe = summary_df["net_sharpe"].isna().sum()
    log(f"  NaN net_sharpe rows: {nan_sharpe}")
    if nan_sharpe > 0:
        log(f"BLOCK: {nan_sharpe} NaN net_sharpe rows in backtest_summary_pass2.csv")
        sys.exit(1)

    # 2. Row count >= 200 (across all four output CSVs combined)
    total_rows = len(summary_df) + len(spread_df) + len(borrow_df) + len(be_df)
    log(f"  Total summary rows: {len(summary_df)}  (combined across all CSVs: {total_rows})")
    if total_rows < 200:
        log(f"BLOCK: Only {total_rows} combined rows — expected >= 200")
        sys.exit(1)

    # 3. Runtime
    elapsed = time.time() - t0
    log(f"  Elapsed: {elapsed/60:.1f} min")
    if elapsed > 3600:
        log(f"BLOCK: Runtime {elapsed/60:.1f} min exceeds 60 min")
        sys.exit(1)

    # 4. Turnover plausible for tercile (0–2 per day)
    stock_rows = summary_df[summary_df["strategy"].str.startswith("tercile")]
    to_max = stock_rows["daily_turnover"].max()
    log(f"  Max daily turnover (tercile): {to_max:.3f}")
    if to_max > 2.1:
        log(f"  WARNING: Turnover {to_max:.3f} exceeds 2.0 for tercile strategy")

    # 5. Long-only borrow = 0 (absolute sum, not per-row float precision issue)
    lo_borrow = summary_df[summary_df["strategy"] == "tercile_lo"]["borrow_cost"].abs().max()
    log(f"  Max |borrow_cost| for long-only: {lo_borrow:.6f}")
    assert lo_borrow < 1e-6, "Long-only rows must have zero borrow cost"

    # ── Final summary table ───────────────────────────────────────────────────
    log("\n=== FINAL SUMMARY ===")
    log(f"Total backtest configurations: {len(summary_df)}")

    top5 = summary_df.nlargest(5, "net_sharpe")[
        ["model", "universe", "target", "strategy", "gate",
         "net_sharpe", "ann_return", "max_drawdown", "calmar"]
    ]
    log("\nTop 5 by net_sharpe:")
    log(top5.to_string(index=False))

    best_net_sr = summary_df["net_sharpe"].max()
    clears_half = (summary_df["net_sharpe"] >= 0.5).sum()
    log(f"\nBest net_sharpe: {best_net_sr:.3f}")
    log(f"Configs clearing +0.5 net Sharpe: {clears_half}")

    # Gate comparison
    gate_on_med  = summary_df[summary_df["gate"] == "gate_on"]["net_sharpe"].median()
    gate_off_med = summary_df[summary_df["gate"] == "gate_off"]["net_sharpe"].median()
    log(f"\nGate-on median net_sharpe:  {gate_on_med:.3f}")
    log(f"Gate-off median net_sharpe: {gate_off_med:.3f}")
    gate_delta = gate_on_med - gate_off_med
    log(f"Gate-on advantage (median): {gate_delta:+.3f}")

    log(f"\nSpread sensitivity rows: {len(spread_df)}")
    log(f"Borrow sensitivity rows:  {len(borrow_df)}")
    log(f"Breakeven analysis rows:  {len(be_df)}")
    log(f"\nRuntime: {elapsed/60:.1f} min")
    log("\n=== Done. All outputs written. ===")

    # Save log
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, "stage_p2-9_backtest.log"), "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
