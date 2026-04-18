# ABOUTME: Stage P2-10 — Expanded feature ablation (tier 1-2 category/subcategory + tier 3 per-feature LOO).
# ABOUTME: Inputs: output/features_track_a_{hk,kr}.parquet. Outputs: output/feature_ablation_pass2.csv, output/feature_ablation_per_feature.csv.
# ABOUTME: Run: source .venv/bin/activate && python3 scripts/stage_p2-10_ablation.py

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb

warnings.filterwarnings("ignore")

PROJECT_ROOT = "/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity_p2"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

WALL_CLOCK_LIMIT = 4 * 3600  # 4 hours hard stop
SCRIPT_START = time.time()

# ── baseline Pass 1 gap results ─────────────────────────────────────────────
# from backtest_summary.csv Pass1 and per-date IC from predictions
BASELINE = {
    "hk": {"gross_sharpe": 3.037845, "net_sharpe": -1.565692, "mean_oos_ic": 0.060994},
    "kr": {"gross_sharpe": 3.768042, "net_sharpe": -0.281931, "mean_oos_ic": 0.060820},
}

# ── fixed best hyperparams from Pass 1 (from training_log_lgbm_{hk,kr}_gap.csv last fold) ─
BEST_PARAMS = {
    "hk": {
        "n_estimators": 100, "max_depth": 5, "learning_rate": 0.01,
        "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 1.0, "reg_lambda": 0.1,
    },
    "kr": {
        "n_estimators": 100, "max_depth": 5, "learning_rate": 0.01,
        "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 1.0, "reg_lambda": 0.1,
    },
}

# ── feature group definitions ────────────────────────────────────────────────
# Full feature set (21 features, derived from parquet columns minus meta/target cols)
META_COLS = {"date", "ticker", "market", "is_weekend_gap", "tgt_gap", "tgt_intraday", "tgt_cc"}

# Per brief P2-10 feature groupings
CRYPTO_OVERNIGHT_FEATURES = [
    "btc_ov_log_return", "eth_ov_log_return",
    "btc_ov_realized_vol", "eth_ov_realized_vol",
    "btc_ov_max_drawdown", "btc_ov_volume_usd",
    "btc_ov_volume_surge", "btc_ov_taker_imbalance",
    "crosspair_dispersion", "btc_eth_spread",
    "btc_funding_rate_latest", "btc_funding_rate_delta",
]
MACRO_FEATURES = [
    "vix_level", "dxy_level", "yield_curve_slope",
    "vix_5d_change", "dxy_5d_change", "breakeven_5y",
]
STOCK_LEVEL_FEATURES = [
    "stock_rv_20d", "stock_ret_20d", "stock_prior_day_return",
]

OVERNIGHT_RETURN_FEATURES = ["btc_ov_log_return", "eth_ov_log_return"]
OVERNIGHT_VOL_FEATURES = ["btc_ov_realized_vol", "eth_ov_realized_vol"]
FUNDING_RATE_FEATURES = ["btc_funding_rate_latest", "btc_funding_rate_delta"]

# Tier 1-2 ablations: (ablation_name, list of features to drop)
TIER12_ABLATIONS = [
    ("drop_crypto_overnight_all", CRYPTO_OVERNIGHT_FEATURES),
    ("drop_macro_all", MACRO_FEATURES),
    ("drop_stock_level_all", STOCK_LEVEL_FEATURES),
    ("drop_ov_return", OVERNIGHT_RETURN_FEATURES),
    ("drop_ov_vol", OVERNIGHT_VOL_FEATURES),
    ("drop_funding_rate", FUNDING_RATE_FEATURES),
]

MARKETS = ["hk", "kr"]
TARGET_COL = "tgt_gap"
MIN_DAILY_TICKERS_FOR_IC = 5
PURGE_GAP_DAYS = 5
ANN = 252.0
HALF_SPREAD = {"hk": 15e-4, "kr": 10e-4}
IMPACT_CONST = 0.1
TRADE_SIZE_USD = 100_000.0
MIN_UNIVERSE = 9


# ── helpers ──────────────────────────────────────────────────────────────────

def check_wall_clock(label=""):
    elapsed = time.time() - SCRIPT_START
    if elapsed > WALL_CLOCK_LIMIT:
        raise RuntimeError(f"BLOCK: Wall clock exceeded 4h at [{label}] ({elapsed/3600:.2f}h)")
    return elapsed


def spearman_ic(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return np.nan
    r, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(r)


def compute_mean_daily_ic(pred_df):
    """Per-date cross-sectional Spearman IC, then mean across dates."""
    ics = []
    for d, g in pred_df.groupby("date"):
        g = g.dropna(subset=["y_pred", "y_actual"])
        if len(g) < MIN_DAILY_TICKERS_FOR_IC:
            continue
        ic = spearman_ic(g["y_actual"].values, g["y_pred"].values)
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else np.nan


def load_cost_data(market):
    """Load universe_log and features for cost computation."""
    ul = pd.read_csv(os.path.join(OUTPUT_DIR, "universe_log.csv"), parse_dates=["date"])
    ul = ul[ul["market"] == market.upper()][["date", "ticker", "adv_usd"]].copy()

    feat = pd.read_parquet(
        os.path.join(OUTPUT_DIR, f"features_track_a_{market}.parquet"),
        columns=["date", "ticker", "stock_rv_20d"]
    )
    feat["date"] = pd.to_datetime(feat["date"])

    all_dates = feat[["date"]].drop_duplicates().sort_values("date")
    tickers = ul["ticker"].unique()
    grid = pd.DataFrame(
        [(d, t) for d in all_dates["date"] for t in tickers],
        columns=["date", "ticker"]
    )
    ul_ff = grid.merge(ul, on=["date", "ticker"], how="left")
    ul_ff = ul_ff.sort_values(["ticker", "date"])
    ul_ff["adv_usd"] = ul_ff.groupby("ticker")["adv_usd"].ffill().bfill()

    cost_df = ul_ff.merge(feat, on=["date", "ticker"], how="left")
    cost_df["stock_rv_20d"] = cost_df.groupby("ticker")["stock_rv_20d"].ffill().bfill()
    return cost_df.set_index(["date", "ticker"])


def compute_impact(adv_usd, daily_vol):
    if adv_usd <= 0 or np.isnan(adv_usd) or np.isnan(daily_vol):
        return 0.0
    daily_vol_dec = daily_vol / np.sqrt(ANN)
    return IMPACT_CONST * np.sqrt(TRADE_SIZE_USD / adv_usd) * daily_vol_dec


def run_backtest(pred_df, cost_data, market):
    """Tercile long-short backtest, gate off. Returns daily DataFrame."""
    pred_df = pred_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    half_spread = HALF_SPREAD[market]
    records = []
    prev_long = set()
    prev_short = set()

    for date, day_df in pred_df.groupby("date"):
        n = len(day_df)
        if n < MIN_UNIVERSE:
            records.append({"date": date, "gross_return": 0.0, "net_return": 0.0})
            prev_long, prev_short = set(), set()
            continue

        day_df = day_df.sort_values("y_pred").reset_index(drop=True)
        tercile = n // 3
        short_set = set(day_df.iloc[:tercile]["ticker"])
        long_set = set(day_df.iloc[n - tercile:]["ticker"])

        long_ret = day_df[day_df["ticker"].isin(long_set)]["y_actual"].mean()
        short_ret = day_df[day_df["ticker"].isin(short_set)]["y_actual"].mean()
        gross_ret = long_ret - short_ret

        long_entries = long_set - prev_long
        long_exits = prev_long - long_set
        short_entries = short_set - prev_short
        short_exits = prev_short - short_set
        changed = long_entries | long_exits | short_entries | short_exits

        cost_dec = 0.0
        for tkr in changed:
            key = (date, tkr)
            if key in cost_data.index:
                row = cost_data.loc[key]
                adv = row["adv_usd"]
                rv = row["stock_rv_20d"]
            else:
                adv, rv = np.nan, np.nan
            impact = compute_impact(adv, rv)
            cost_dec += (half_spread + impact) * 2 / (tercile * 2)

        net_ret = gross_ret - cost_dec
        records.append({"date": date, "gross_return": gross_ret, "net_return": net_ret})
        prev_long, prev_short = long_set, short_set

    df = pd.DataFrame(records)
    return df


def compute_sharpes(bt_df):
    ret_g = bt_df["gross_return"]
    ret_n = bt_df["net_return"]
    if len(bt_df) == 0:
        return np.nan, np.nan
    ann_ret_g = ret_g.sum() * ANN / len(bt_df)
    ann_ret_n = ret_n.sum() * ANN / len(bt_df)
    gross_sharpe = ann_ret_g / (ret_g.std() * np.sqrt(ANN)) if ret_g.std() > 0 else np.nan
    net_sharpe = ann_ret_n / (ret_n.std() * np.sqrt(ANN)) if ret_n.std() > 0 else np.nan
    return gross_sharpe, net_sharpe


# ── walk-forward training ─────────────────────────────────────────────────────

def run_walk_forward(df_all, feature_cols, target_col, params):
    """
    Walk-forward OOS. Uses fixed hyperparams (no search). Returns prediction DataFrame.
    """
    df_all = df_all.sort_values(["date", "ticker"]).reset_index(drop=True)
    all_dates = sorted(df_all["date"].unique())
    all_months = sorted(set(pd.Timestamp(d).strftime("%Y-%m") for d in all_dates))

    oos_months = []
    for m in all_months:
        month_start = pd.Timestamp(m + "-01")
        train_days = [d for d in all_dates if d < month_start]
        if len(train_days) >= 252:
            oos_months.append(m)

    if not oos_months:
        raise RuntimeError("No OOS months (need >= 252 training days)")

    pred_rows = []
    for month_str in oos_months:
        month_start = pd.Timestamp(month_str + "-01")
        next_month = month_start + pd.offsets.MonthBegin(1)

        train_mask = df_all["date"] < month_start
        test_mask = (df_all["date"] >= month_start) & (df_all["date"] < next_month)

        train_df = df_all[train_mask].dropna(subset=[target_col])
        test_df = df_all[test_mask].dropna(subset=[target_col])

        if len(train_df) < 10 or len(test_df) == 0:
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values

        model = lgb.LGBMRegressor(verbose=-1, n_jobs=-1, random_state=42, **params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        for i in range(len(test_df)):
            pred_rows.append({
                "date": test_df["date"].iloc[i],
                "ticker": test_df["ticker"].iloc[i],
                "y_pred": y_pred[i],
                "y_actual": y_test[i],
            })

    return pd.DataFrame(pred_rows)


def run_one_ablation(df, feature_cols_all, drop_features, params, cost_data, market, label=""):
    """
    Train with drop_features removed, run backtest, return (gross_sharpe, net_sharpe, mean_ic).
    """
    check_wall_clock(label)
    feature_cols = [f for f in feature_cols_all if f not in drop_features]
    if len(feature_cols) == 0:
        raise ValueError(f"All features dropped in ablation: {label}")

    pred_df = run_walk_forward(df, feature_cols, TARGET_COL, params)
    if pred_df.empty:
        return np.nan, np.nan, np.nan

    mean_ic = compute_mean_daily_ic(pred_df)
    bt_df = run_backtest(pred_df, cost_data, market)
    gross_sharpe, net_sharpe = compute_sharpes(bt_df)
    return gross_sharpe, net_sharpe, mean_ic


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"=== Stage P2-10 Expanded Ablation ===")
    log(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── skip-if-exists guard ─────────────────────────────────────────────────
    tier12_path = os.path.join(OUTPUT_DIR, "feature_ablation_pass2.csv")
    pf_path = os.path.join(OUTPUT_DIR, "feature_ablation_per_feature.csv")
    if os.path.exists(tier12_path) and os.path.exists(pf_path):
        t12 = pd.read_csv(tier12_path)
        pf = pd.read_csv(pf_path)
        log(f"Both output files already exist ({len(t12)} pass2 rows, {len(pf)} per-feature rows). Skipping.")
        return

    # Load data
    data = {}
    cost_data_all = {}
    feature_cols_all = {}
    for mkt in MARKETS:
        df = pd.read_parquet(os.path.join(OUTPUT_DIR, f"features_track_a_{mkt}.parquet"))
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        data[mkt] = df
        fcols = [c for c in df.columns if c not in META_COLS]
        feature_cols_all[mkt] = fcols
        log(f"[{mkt.upper()}] loaded {len(df)} rows, {len(fcols)} features: {fcols}")
        cost_data_all[mkt] = load_cost_data(mkt)
        log(f"[{mkt.upper()}] cost_data loaded, {len(cost_data_all[mkt])} index entries")

    # ── TIER 1-2 ──────────────────────────────────────────────────────────────
    log("\n--- TIER 1-2: Category and Subcategory Ablations ---")
    tier12_rows = []
    tier12_ckpt_path = os.path.join(OUTPUT_DIR, "feature_ablation_pass2_ckpt.csv")

    # Resume from checkpoint if partial progress exists
    completed_tier12 = set()
    if os.path.exists(tier12_ckpt_path):
        ckpt_df = pd.read_csv(tier12_ckpt_path)
        tier12_rows = ckpt_df.to_dict("records")
        completed_tier12 = set(zip(ckpt_df["ablation_name"], ckpt_df["market"]))
        log(f"  Resuming tier 1-2 from checkpoint: {len(tier12_rows)} rows already done")

    for ablation_name, drop_features in TIER12_ABLATIONS:
        for mkt in MARKETS:
            key = (ablation_name, mkt.upper())
            if key in completed_tier12:
                log(f"  Skipping (checkpoint): {ablation_name} | {mkt.upper()}")
                continue

            check_wall_clock(f"tier12 {ablation_name} {mkt}")
            t0 = time.time()
            log(f"  Running: {ablation_name} | {mkt.upper()}")

            # Verify features exist in dataset
            actual_drop = [f for f in drop_features if f in feature_cols_all[mkt]]
            missing = [f for f in drop_features if f not in feature_cols_all[mkt]]
            if missing:
                log(f"    WARNING: features not in dataset: {missing}")

            try:
                gs, ns, ic = run_one_ablation(
                    data[mkt], feature_cols_all[mkt], actual_drop,
                    BEST_PARAMS[mkt], cost_data_all[mkt], mkt,
                    label=f"{ablation_name}_{mkt}"
                )
                base = BASELINE[mkt]
                delta_gs = gs - base["gross_sharpe"]
                delta_ns = ns - base["net_sharpe"]
                delta_ic = ic - base["mean_oos_ic"]

                elapsed = time.time() - t0
                log(f"    gs={gs:.4f} ns={ns:.4f} ic={ic:.6f} | dgs={delta_gs:.4f} dns={delta_ns:.4f} dic={delta_ic:.6f} | {elapsed:.0f}s")

                row = {
                    "ablation_name": ablation_name,
                    "market": mkt.upper(),
                    "target": "gap",
                    "delta_gross_sharpe": round(delta_gs, 6),
                    "delta_net_sharpe": round(delta_ns, 6),
                    "delta_ic": round(delta_ic, 8),
                    "ablated_gross_sharpe": round(gs, 6),
                    "ablated_net_sharpe": round(ns, 6),
                    "ablated_mean_ic": round(ic, 8),
                }
            except RuntimeError as e:
                if "BLOCK" in str(e):
                    log(f"BLOCK: {e}")
                    raise
                log(f"    ERROR: {e}")
                row = {
                    "ablation_name": ablation_name, "market": mkt.upper(), "target": "gap",
                    "delta_gross_sharpe": np.nan, "delta_net_sharpe": np.nan, "delta_ic": np.nan,
                    "ablated_gross_sharpe": np.nan, "ablated_net_sharpe": np.nan, "ablated_mean_ic": np.nan,
                }

            tier12_rows.append(row)
            # Intermediate checkpoint after each ablation
            pd.DataFrame(tier12_rows).to_csv(tier12_ckpt_path, index=False)
            log(f"    Checkpoint saved: {len(tier12_rows)} rows")

    tier12_df = pd.DataFrame(tier12_rows)
    tier12_df.to_csv(tier12_path, index=False)
    log(f"\nSaved tier 1-2: {tier12_path} ({len(tier12_df)} rows)")

    # Validate no duplicate (ablation_name, market) keys
    dups = tier12_df.duplicated(subset=["ablation_name", "market"]).sum()
    if dups > 0:
        log(f"WARNING: {dups} duplicate (ablation_name, market) in tier12 output")

    # ── TIER 3: PER-FEATURE LOO ───────────────────────────────────────────────
    log("\n--- TIER 3: Per-Feature Leave-One-Out ---")

    # Project time for tier 3: ~1-2 min per model, 21 features x 2 markets = 42 models
    # Check if we're at risk
    elapsed_so_far = time.time() - SCRIPT_START
    log(f"Elapsed after tier 1-2: {elapsed_so_far/60:.1f} min")
    projected_tier3 = elapsed_so_far + 42 * 120  # 42 models x 2 min each (optimistic)
    use_kr_only = projected_tier3 > 3 * 3600
    if use_kr_only:
        tier3_markets = ["kr"]
        log(f"  FALLBACK: Projecting {projected_tier3/3600:.1f}h > 3h; running KR-only for tier 3.")
    else:
        tier3_markets = MARKETS
        log(f"  Running both markets for tier 3 (projected {projected_tier3/3600:.1f}h)")

    per_feature_rows = []
    pf_ckpt_path = os.path.join(OUTPUT_DIR, "feature_ablation_per_feature_ckpt.csv")

    # Resume from checkpoint if partial progress exists
    completed_pf = set()
    if os.path.exists(pf_ckpt_path):
        pf_ckpt_df = pd.read_csv(pf_ckpt_path)
        per_feature_rows = pf_ckpt_df.to_dict("records")
        completed_pf = set(zip(pf_ckpt_df["feature_name"], pf_ckpt_df["market"]))
        log(f"  Resuming tier 3 from checkpoint: {len(per_feature_rows)} rows already done")

    for mkt in tier3_markets:
        fcols = feature_cols_all[mkt]
        log(f"\n  [{mkt.upper()}] Per-feature LOO on {len(fcols)} features")
        for feat in fcols:
            key = (feat, mkt.upper())
            if key in completed_pf:
                log(f"    Skipping (checkpoint): {feat} | {mkt.upper()}")
                continue

            check_wall_clock(f"tier3 {feat} {mkt}")
            t0 = time.time()
            try:
                gs, ns, ic = run_one_ablation(
                    data[mkt], fcols, [feat],
                    BEST_PARAMS[mkt], cost_data_all[mkt], mkt,
                    label=f"loo_{feat}_{mkt}"
                )
                base = BASELINE[mkt]
                delta_gs = gs - base["gross_sharpe"]
                delta_ns = ns - base["net_sharpe"]
                delta_ic = ic - base["mean_oos_ic"]
                elapsed = time.time() - t0
                log(f"    drop[{feat}] gs={gs:.4f} ns={ns:.4f} ic={ic:.6f} | dgs={delta_gs:.4f} dns={delta_ns:.4f} | {elapsed:.0f}s")

                pf_row = {
                    "feature_name": feat,
                    "market": mkt.upper(),
                    "target": "gap",
                    "delta_gross_sharpe": round(delta_gs, 6),
                    "delta_net_sharpe": round(delta_ns, 6),
                    "delta_ic": round(delta_ic, 8),
                    "ablated_gross_sharpe": round(gs, 6),
                    "ablated_net_sharpe": round(ns, 6),
                    "ablated_mean_ic": round(ic, 8),
                }
            except RuntimeError as e:
                if "BLOCK" in str(e):
                    log(f"BLOCK: {e}")
                    raise
                log(f"    ERROR dropping {feat}: {e}")
                pf_row = {
                    "feature_name": feat, "market": mkt.upper(), "target": "gap",
                    "delta_gross_sharpe": np.nan, "delta_net_sharpe": np.nan, "delta_ic": np.nan,
                    "ablated_gross_sharpe": np.nan, "ablated_net_sharpe": np.nan, "ablated_mean_ic": np.nan,
                }

            per_feature_rows.append(pf_row)
            # Intermediate checkpoint after each feature
            pd.DataFrame(per_feature_rows).to_csv(pf_ckpt_path, index=False)
            log(f"    Per-feature checkpoint: {len(per_feature_rows)} rows")

    pf_df = pd.DataFrame(per_feature_rows)
    pf_df.to_csv(pf_path, index=False)
    log(f"\nSaved per-feature: {pf_path} ({len(pf_df)} rows)")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    total_wall = time.time() - SCRIPT_START
    log(f"\n{'='*60}")
    log(f"FINAL SUMMARY")
    log(f"{'='*60}")
    log(f"Rows in feature_ablation_pass2.csv (tier 1-2): {len(tier12_df)}")
    fallback_note = " [KR-only fallback applied]" if use_kr_only else ""
    log(f"Rows in feature_ablation_per_feature.csv (tier 3): {len(pf_df)}{fallback_note}")

    for mkt in MARKETS:
        mkt_up = mkt.upper()
        t12_mkt = tier12_df[tier12_df["market"] == mkt_up]
        if not t12_mkt.empty:
            max_drop = t12_mkt.loc[t12_mkt["delta_net_sharpe"].idxmin()]
            min_drop = t12_mkt.loc[t12_mkt["delta_net_sharpe"].idxmax()]
            log(f"\n  {mkt_up} (tier 1-2):")
            log(f"    Largest delta_net_sharpe drop: {max_drop['ablation_name']} (delta={max_drop['delta_net_sharpe']:.4f})")
            log(f"    Smallest delta_net_sharpe drop: {min_drop['ablation_name']} (delta={min_drop['delta_net_sharpe']:.4f})")

        pf_mkt = pf_df[pf_df["market"] == mkt_up] if not pf_df.empty else pd.DataFrame()
        if not pf_mkt.empty:
            pf_valid = pf_mkt.dropna(subset=["delta_net_sharpe"])
            if not pf_valid.empty:
                max_pf = pf_valid.loc[pf_valid["delta_net_sharpe"].idxmin()]
                min_pf = pf_valid.loc[pf_valid["delta_net_sharpe"].idxmax()]
                log(f"  {mkt_up} (tier 3):")
                log(f"    Largest delta_net_sharpe drop: {max_pf['feature_name']} (delta={max_pf['delta_net_sharpe']:.4f})")
                log(f"    Smallest delta_net_sharpe drop: {min_pf['feature_name']} (delta={min_pf['delta_net_sharpe']:.4f})")

    log(f"\nTotal wall clock: {total_wall/60:.1f} min ({total_wall/3600:.2f}h)")
    log(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Write log file
    log_path = os.path.join(LOG_DIR, "stage_p2-10_ablation.log")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nLog written: {log_path}")


if __name__ == "__main__":
    main()
