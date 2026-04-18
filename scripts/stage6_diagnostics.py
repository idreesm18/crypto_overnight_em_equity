# ABOUTME: Stage 6 diagnostics: feature ablation, return decomposition, regime analysis, SHAP stability, weekend effect.
# ABOUTME: Inputs: output/features_track_a_{hk,kr}.parquet, output/predictions_lgbm_*.csv, output/shap_per_fold_*.parquet,
#          output/training_log_lgbm_*.csv, output/backtest_lgbm_*.csv, output/backtest_summary.csv,
#          data/fred/fred_all.parquet, data/yfinance/btcusd_daily.parquet
# ABOUTME: Outputs: output/feature_ablation.csv, output/return_decomposition.csv, output/regime_analysis.csv,
#          output/shap_stability.csv, output/shap_fold_overlap.csv, output/weekend_effect.csv, output/diagnostics_summary.txt
# ABOUTME: Run: source .venv/bin/activate && python3 scripts/stage6_diagnostics.py

import ast
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb

warnings.filterwarnings("ignore")

PROJECT_ROOT = "/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity"
OUTPUT_DIR = f"{PROJECT_ROOT}/output"
LOG_DIR = f"{PROJECT_ROOT}/logs"

MARKETS = ["hk", "kr"]
TARGETS = ["gap", "intraday", "cc"]

# Feature groups as defined in brief
CRYPTO_OV_FEATURES = [
    "btc_ov_log_return", "eth_ov_log_return", "btc_ov_realized_vol",
    "eth_ov_realized_vol", "btc_ov_max_drawdown", "btc_ov_volume_usd",
    "btc_ov_volume_surge", "btc_ov_taker_imbalance", "crosspair_dispersion",
    "btc_eth_spread", "btc_funding_rate_latest", "btc_funding_rate_delta",
]
MACRO_FEATURES = [
    "vix_level", "vix_5d_change", "yield_curve_slope",
    "dxy_level", "dxy_5d_change", "breakeven_5y",
]
STOCK_FEATURES = ["stock_rv_20d", "stock_ret_20d", "stock_prior_day_return"]

FEATURE_GROUPS = {
    "crypto": CRYPTO_OV_FEATURES,
    "macro": MACRO_FEATURES,
    "stock_level": STOCK_FEATURES,
}

EXCLUDE_COLS = {"date", "ticker", "market", "is_weekend_gap", "tgt_gap", "tgt_intraday", "tgt_cc"}
TARGET_COLS = {"gap": "tgt_gap", "intraday": "tgt_intraday", "cc": "tgt_cc"}

DEFAULT_PARAMS = {
    "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
    "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.1,
}
MIN_DAILY_TICKERS_FOR_IC = 5
HK_COST_BPS = 15
KR_COST_BPS = 10


# ─── Utility functions ────────────────────────────────────────────────────────

def spearman_ic(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return np.nan
    r, _ = spearmanr(y_true[mask], y_pred[mask])
    return r


def compute_fold_ic(y_true, y_pred, dates):
    """Per-day IC averaged over trading days."""
    df = pd.DataFrame({"date": dates, "y_true": y_true, "y_pred": y_pred})
    ics = []
    for d, grp in df.groupby("date"):
        valid = grp.dropna(subset=["y_true", "y_pred"])
        if len(valid) < MIN_DAILY_TICKERS_FOR_IC:
            continue
        ic = spearman_ic(valid["y_true"].values, valid["y_pred"].values)
        if not np.isnan(ic):
            ics.append(ic)
    return np.mean(ics) if ics else np.nan


def dir_acc(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return (np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean()


def tercile_backtest(pred_df, cost_bps_per_side):
    """
    Tercile long/short portfolio from predictions.
    pred_df must have: date, y_pred, y_actual
    Returns (gross_sharpe, net_sharpe).
    """
    daily_returns = []
    for d, grp in pred_df.groupby("date"):
        grp = grp.dropna(subset=["y_pred", "y_actual"])
        n = len(grp)
        if n < 9:
            continue
        tercile_size = n // 3
        sorted_grp = grp.sort_values("y_pred")
        short_leg = sorted_grp.iloc[:tercile_size]
        long_leg = sorted_grp.iloc[-tercile_size:]
        gross_ret = long_leg["y_actual"].mean() - short_leg["y_actual"].mean()
        # cost: enter + exit, both legs
        cost = 4 * cost_bps_per_side / 10000
        net_ret = gross_ret - cost
        daily_returns.append({"date": d, "gross": gross_ret, "net": net_ret})

    if not daily_returns:
        return np.nan, np.nan

    ret_df = pd.DataFrame(daily_returns)
    ann = 252
    gross_sharpe = (ret_df["gross"].mean() / ret_df["gross"].std()) * np.sqrt(ann) if ret_df["gross"].std() > 0 else np.nan
    net_sharpe = (ret_df["net"].mean() / ret_df["net"].std()) * np.sqrt(ann) if ret_df["net"].std() > 0 else np.nan
    return gross_sharpe, net_sharpe


def bootstrap_pvalue(ic_series, n_boot=1000, seed=42):
    """Two-sided bootstrap p-value for mean IC != 0."""
    ic_arr = np.array(ic_series.dropna())
    if len(ic_arr) < 5:
        return np.nan
    obs_mean = ic_arr.mean()
    rng = np.random.default_rng(seed)
    boot_means = [rng.choice(ic_arr, size=len(ic_arr), replace=True).mean() for _ in range(n_boot)]
    # Null: center bootstrap distribution at 0
    centered = np.array(boot_means) - obs_mean
    p = (np.abs(centered) >= np.abs(obs_mean)).mean()
    return p


def get_best_params(mkt, tgt):
    """Read final-row best_params from training log. Fall back to defaults."""
    try:
        log = pd.read_csv(f"{OUTPUT_DIR}/training_log_lgbm_{mkt}_{tgt}.csv")
        last_params_str = log.iloc[-1]["best_params"]
        params = ast.literal_eval(last_params_str)
        return params
    except Exception as e:
        print(f"  WARNING: could not parse best_params for {mkt}_{tgt}: {e}. Using defaults.")
        return DEFAULT_PARAMS.copy()


def run_ablation_fold(df_all, feature_cols, target_col, oos_months, params):
    """
    Walk-forward LightGBM with fixed params across all folds.
    Returns pred_df with date, y_pred, y_actual.
    STOP if empty predictions returned.
    """
    pred_rows = []
    for month_str in oos_months:
        month_start = pd.Timestamp(month_str + "-01")
        next_month = month_start + pd.offsets.MonthBegin(1)

        train_mask = df_all["date"] < month_start
        test_mask = (df_all["date"] >= month_start) & (df_all["date"] < next_month)

        train_df = df_all[train_mask].dropna(subset=[target_col])
        test_df = df_all[test_mask].dropna(subset=[target_col])

        if len(train_df) == 0 or len(test_df) == 0:
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


# ─── 6A. Feature Ablation ─────────────────────────────────────────────────────

def run_ablation(baseline_summary):
    """
    Run 18 ablation experiments (2 markets x 3 targets x 3 dropped groups).
    Uses final-fold best_params from training log (no hyperparameter search).
    Returns DataFrame with ablation results.
    """
    print("\n" + "=" * 60)
    print("6A. FEATURE ABLATION")
    print("=" * 60)

    rows = []
    for mkt in MARKETS:
        df = pd.read_parquet(f"{OUTPUT_DIR}/features_track_a_{mkt}.parquet")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        all_feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
        all_dates = sorted(df["date"].unique())
        all_months = sorted(set(pd.Timestamp(d).strftime("%Y-%m") for d in all_dates))
        oos_months = [m for m in all_months
                      if len([d for d in all_dates if d < pd.Timestamp(m + "-01")]) >= 252]

        cost_bps = HK_COST_BPS if mkt == "hk" else KR_COST_BPS

        for tgt in TARGETS:
            target_col = TARGET_COLS[tgt]
            params = get_best_params(mkt, tgt)

            # Read baseline gross/net sharpe
            b_row = baseline_summary[(baseline_summary["market"] == mkt.upper()) &
                                     (baseline_summary["target"] == tgt)]
            if len(b_row) == 0:
                print(f"  WARNING: no baseline for {mkt}_{tgt}")
                continue
            baseline_gross = b_row.iloc[0]["gross_sharpe"]

            # Baseline IC from training log
            log_df = pd.read_csv(f"{OUTPUT_DIR}/training_log_lgbm_{mkt}_{tgt}.csv")
            baseline_ic = log_df["ic_test"].mean()

            for group_name, group_feats in FEATURE_GROUPS.items():
                # Ablated feature set
                ablated_cols = [c for c in all_feature_cols if c not in group_feats]
                if len(ablated_cols) == 0:
                    print(f"  SKIP {mkt}_{tgt} drop={group_name}: no features left")
                    continue

                print(f"  [{mkt}_{tgt}] drop={group_name}: {len(all_feature_cols)} -> {len(ablated_cols)} features")
                pred_df = run_ablation_fold(df, ablated_cols, target_col, oos_months, params)

                if pred_df.empty:
                    print(f"  STOP: empty predictions for {mkt}_{tgt} drop={group_name}")
                    raise RuntimeError(f"Empty predictions: {mkt}_{tgt} drop={group_name}")

                # Compute metrics
                mean_ic = compute_fold_ic(
                    pred_df["y_actual"].values,
                    pred_df["y_pred"].values,
                    pred_df["date"].values,
                )
                da = dir_acc(pred_df["y_actual"].values, pred_df["y_pred"].values)
                gross_sh, net_sh = tercile_backtest(pred_df, cost_bps)

                delta_ic = mean_ic - baseline_ic
                delta_gross = gross_sh - baseline_gross
                # net baseline from backtest summary
                b_net = b_row.iloc[0]["net_sharpe"]
                delta_net = (net_sh - b_net) if not np.isnan(net_sh) else np.nan

                flag = "FLAG_LARGE_DROP" if (not np.isnan(delta_gross) and delta_gross < -0.15) else ""

                rows.append({
                    "market": mkt.upper(),
                    "target": tgt,
                    "dropped_group": group_name,
                    "mean_ic": round(mean_ic, 6) if not np.isnan(mean_ic) else np.nan,
                    "dir_acc": round(da, 4) if not np.isnan(da) else np.nan,
                    "gross_sharpe": round(gross_sh, 4) if not np.isnan(gross_sh) else np.nan,
                    "net_sharpe": round(net_sh, 4) if not np.isnan(net_sh) else np.nan,
                    "delta_ic_vs_baseline": round(delta_ic, 6) if not np.isnan(delta_ic) else np.nan,
                    "delta_gross_sharpe_vs_baseline": round(delta_gross, 4) if not np.isnan(delta_gross) else np.nan,
                    "delta_net_sharpe_vs_baseline": round(delta_net, 4) if not np.isnan(delta_net) else np.nan,
                    "meets_criterion_3": flag,
                })

    abl_df = pd.DataFrame(rows)
    abl_df.to_csv(f"{OUTPUT_DIR}/feature_ablation.csv", index=False)
    print(f"\n  Saved: output/feature_ablation.csv ({len(abl_df)} rows)")
    return abl_df


# ─── 6B. Return Component Analysis ───────────────────────────────────────────

def run_return_decomposition():
    print("\n" + "=" * 60)
    print("6B. RETURN COMPONENT ANALYSIS")
    print("=" * 60)
    rows = []
    for mkt in MARKETS:
        for tgt in TARGETS:
            log_df = pd.read_csv(f"{OUTPUT_DIR}/training_log_lgbm_{mkt}_{tgt}.csv")
            mean_ic = log_df["ic_test"].mean()
            median_ic = log_df["ic_test"].median()
            mean_dir_acc = log_df["dir_acc_test"].mean()
            n_folds = len(log_df)

            # Bootstrap p-value on per-fold ICs
            p = bootstrap_pvalue(log_df["ic_test"])

            rows.append({
                "market": mkt.upper(),
                "target": tgt,
                "mean_ic_test": round(mean_ic, 6),
                "median_ic_test": round(median_ic, 6),
                "dir_acc_mean": round(mean_dir_acc, 4),
                "n_folds": n_folds,
                "bootstrap_p_value": round(p, 4) if not np.isnan(p) else np.nan,
            })

    decomp_df = pd.DataFrame(rows)
    decomp_df.to_csv(f"{OUTPUT_DIR}/return_decomposition.csv", index=False)
    print(f"  Saved: output/return_decomposition.csv")

    # Print IC differences gap vs intraday
    for mkt in MARKETS:
        gap_ic = decomp_df[(decomp_df["market"] == mkt.upper()) & (decomp_df["target"] == "gap")]["mean_ic_test"].values[0]
        intra_ic = decomp_df[(decomp_df["market"] == mkt.upper()) & (decomp_df["target"] == "intraday")]["mean_ic_test"].values[0]
        print(f"  {mkt.upper()}: gap_ic={gap_ic:.4f}, intraday_ic={intra_ic:.4f}, diff={gap_ic - intra_ic:.4f}")

    return decomp_df


# ─── 6C. Regime Analysis ─────────────────────────────────────────────────────

def build_regime_flags(pred_df, fred_df, btc_df):
    """Attach regime labels to prediction dates."""
    # Forward-fill FRED and BTC to daily
    vix = fred_df[fred_df["series"] == "VIXCLS"][["date", "value"]].rename(columns={"value": "vix"})
    vix = vix.sort_values("date").drop_duplicates("date")

    btc = btc_df[["date", "close"]].sort_values("date").drop_duplicates("date")
    btc["date"] = pd.to_datetime(btc["date"])

    # Unique prediction dates
    pred_dates = pred_df[["date"]].drop_duplicates().sort_values("date")
    pred_dates["date"] = pd.to_datetime(pred_dates["date"])

    vix["date"] = pd.to_datetime(vix["date"])
    merged = pred_dates.merge(vix.rename(columns={"vix": "vix_level"}), on="date", how="left")
    merged = merged.sort_values("date")
    merged["vix_level"] = merged["vix_level"].ffill()

    # BTC 30d trailing return at T-1
    btc_base = btc.set_index("date")["close"]
    btc_ret_30d = btc_base.pct_change(30)  # T-day: uses close(T) / close(T-30) - 1
    # Use T-1 value for prediction date T
    btc_ret_30d_lag = btc_ret_30d.shift(1)

    # BTC 30d realized vol: std of daily returns over trailing 30 days, at T-1
    btc_daily_ret = btc_base.pct_change()
    btc_rvol_30d = btc_daily_ret.rolling(30).std() * np.sqrt(252)
    btc_rvol_30d_lag = btc_rvol_30d.shift(1)

    # Attach to merged
    btc_ret_df = btc_ret_30d_lag.reset_index()
    btc_ret_df.columns = ["date", "btc_ret_30d"]
    btc_rvol_df = btc_rvol_30d_lag.reset_index()
    btc_rvol_df.columns = ["date", "btc_rvol_30d"]

    merged = merged.merge(btc_ret_df, on="date", how="left")
    merged = merged.merge(btc_rvol_df, on="date", how="left")
    merged["btc_ret_30d"] = merged["btc_ret_30d"].ffill()
    merged["btc_rvol_30d"] = merged["btc_rvol_30d"].ffill()

    # Rolling median of BTC vol (over full available history up to T-1)
    merged = merged.sort_values("date")
    merged["btc_rvol_median"] = merged["btc_rvol_30d"].expanding().median()

    # Regime flags
    merged["vix_regime"] = np.where(merged["vix_level"] > 25, "high_vix", "low_vix")
    merged["btc_trend"] = np.where(merged["btc_ret_30d"] > 0, "btc_up", "btc_down")
    merged["crypto_vol"] = np.where(
        merged["btc_rvol_30d"] > merged["btc_rvol_median"], "high_cvol", "low_cvol"
    )

    return merged[["date", "vix_regime", "btc_trend", "crypto_vol"]]


def run_regime_analysis():
    print("\n" + "=" * 60)
    print("6C. REGIME ANALYSIS")
    print("=" * 60)

    fred_df = pd.read_parquet(f"{PROJECT_ROOT}/data/fred/fred_all.parquet")
    fred_df["date"] = pd.to_datetime(fred_df["date"])
    btc_df = pd.read_parquet(f"{PROJECT_ROOT}/data/yfinance/btcusd_daily.parquet")
    btc_df["date"] = pd.to_datetime(btc_df["date"])

    rows = []
    for mkt in MARKETS:
        cost_bps = HK_COST_BPS if mkt == "hk" else KR_COST_BPS
        for tgt in TARGETS:
            pred_df = pd.read_csv(f"{OUTPUT_DIR}/predictions_lgbm_{mkt}_{tgt}.csv")
            pred_df["date"] = pd.to_datetime(pred_df["date"])

            regime_flags = build_regime_flags(pred_df, fred_df, btc_df)
            pred_merged = pred_df.merge(regime_flags, on="date", how="left")

            # regime_type x bucket
            for regime_type, bucket_col in [
                ("vix", "vix_regime"),
                ("btc_trend", "btc_trend"),
                ("crypto_vol", "crypto_vol"),
            ]:
                for bucket, subset in pred_merged.groupby(bucket_col):
                    if subset.empty:
                        continue
                    n_days = subset["date"].nunique()
                    mean_ic = compute_fold_ic(
                        subset["y_actual"].values,
                        subset["y_pred"].values,
                        subset["date"].values,
                    )
                    gross_sh, _ = tercile_backtest(subset, cost_bps)
                    rows.append({
                        "market": mkt.upper(),
                        "target": tgt,
                        "regime_type": regime_type,
                        "regime_bucket": bucket,
                        "n_days": n_days,
                        "mean_ic": round(mean_ic, 6) if not np.isnan(mean_ic) else np.nan,
                        "gross_sharpe": round(gross_sh, 4) if not np.isnan(gross_sh) else np.nan,
                    })

    regime_df = pd.DataFrame(rows)

    # Validate: no NaN mean_ic allowed
    nan_ic = regime_df["mean_ic"].isna().sum()
    if nan_ic > 0:
        print(f"  WARNING: {nan_ic} NaN mean_ic values in regime_analysis. Inspect output.")

    regime_df.to_csv(f"{OUTPUT_DIR}/regime_analysis.csv", index=False)
    print(f"  Saved: output/regime_analysis.csv ({len(regime_df)} rows, {nan_ic} NaN mean_ic)")
    return regime_df


# ─── 6D. SHAP Stability ───────────────────────────────────────────────────────

def run_shap_stability():
    print("\n" + "=" * 60)
    print("6D. SHAP STABILITY")
    print("=" * 60)

    stability_rows = []
    overlap_rows = []

    for mkt in MARKETS:
        for tgt in TARGETS:
            shap_df = pd.read_parquet(f"{OUTPUT_DIR}/shap_per_fold_{mkt}_{tgt}.parquet")

            # Per-fold mean absolute SHAP, ranked
            fold_rankings = {}  # fold_id -> list of (feature, abs_shap) sorted desc
            for fold_id, fold_data in shap_df.groupby("fold_id"):
                feat_shap = fold_data.groupby("feature")["shap_value"].apply(
                    lambda x: np.abs(x).mean()
                ).sort_values(ascending=False)
                fold_rankings[fold_id] = feat_shap

            all_folds = sorted(fold_rankings.keys())
            n_folds = len(all_folds)

            # Top-10 per fold
            top10_per_fold = {f: set(fold_rankings[f].head(10).index) for f in all_folds}

            # % overlap between consecutive folds
            for i in range(len(all_folds) - 1):
                f1, f2 = all_folds[i], all_folds[i + 1]
                overlap = len(top10_per_fold[f1] & top10_per_fold[f2]) / 10.0
                # Spearman rank correlation on all features
                s1 = fold_rankings[f1]
                s2 = fold_rankings[f2]
                common = s1.index.intersection(s2.index)
                if len(common) >= 5:
                    rank_corr, _ = spearmanr(
                        s1.loc[common].rank(ascending=False),
                        s2.loc[common].rank(ascending=False),
                    )
                else:
                    rank_corr = np.nan
                overlap_rows.append({
                    "market": mkt.upper(),
                    "target": tgt,
                    "fold_pair_label": f"{f1}_vs_{f2}",
                    "top10_overlap_pct": round(overlap * 100, 1),
                    "rank_corr": round(rank_corr, 4) if not np.isnan(rank_corr) else np.nan,
                })

            # Overall mean absolute SHAP and rank
            overall_shap = shap_df.groupby("feature")["shap_value"].apply(
                lambda x: np.abs(x).mean()
            ).sort_values(ascending=False)
            overall_rank = {feat: rank + 1 for rank, feat in enumerate(overall_shap.index)}

            # % folds in top-10
            n_folds_top10 = {feat: 0 for feat in overall_shap.index}
            for f in all_folds:
                for feat in top10_per_fold[f]:
                    if feat in n_folds_top10:
                        n_folds_top10[feat] += 1

            for feat in overall_shap.index:
                stability_rows.append({
                    "market": mkt.upper(),
                    "target": tgt,
                    "feature": feat,
                    "mean_abs_shap_overall": round(overall_shap[feat], 8),
                    "overall_rank": overall_rank[feat],
                    "pct_folds_in_top10": round(100 * n_folds_top10[feat] / max(n_folds, 1), 1),
                })

    stab_df = pd.DataFrame(stability_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    stab_df.to_csv(f"{OUTPUT_DIR}/shap_stability.csv", index=False)
    overlap_df.to_csv(f"{OUTPUT_DIR}/shap_fold_overlap.csv", index=False)
    print(f"  Saved: output/shap_stability.csv ({len(stab_df)} rows)")
    print(f"  Saved: output/shap_fold_overlap.csv ({len(overlap_df)} rows)")
    return stab_df, overlap_df


# ─── 6E. Weekend Effect ───────────────────────────────────────────────────────

def run_weekend_effect():
    print("\n" + "=" * 60)
    print("6E. WEEKEND EFFECT")
    print("=" * 60)

    rows = []
    for mkt in MARKETS:
        cost_bps = HK_COST_BPS if mkt == "hk" else KR_COST_BPS
        for tgt in TARGETS:
            pred_df = pd.read_csv(f"{OUTPUT_DIR}/predictions_lgbm_{mkt}_{tgt}.csv")
            pred_df["date"] = pd.to_datetime(pred_df["date"])
            pred_df["day_of_week"] = pred_df["date"].dt.dayofweek  # 0=Mon, 4=Fri

            for bucket_label, bucket_mask in [
                ("monday", pred_df["day_of_week"] == 0),
                ("tue_fri", pred_df["day_of_week"] > 0),
            ]:
                subset = pred_df[bucket_mask]
                if subset.empty:
                    continue
                n_days = subset["date"].nunique()
                mean_ic = compute_fold_ic(
                    subset["y_actual"].values,
                    subset["y_pred"].values,
                    subset["date"].values,
                )
                gross_sh, _ = tercile_backtest(subset, cost_bps)
                # Annualized return
                ret_df = []
                for d, grp in subset.groupby("date"):
                    grp2 = grp.dropna(subset=["y_pred", "y_actual"])
                    n = len(grp2)
                    if n < 9:
                        continue
                    tercile_size = n // 3
                    sorted_grp = grp2.sort_values("y_pred")
                    gross_ret = (
                        sorted_grp.iloc[-tercile_size:]["y_actual"].mean()
                        - sorted_grp.iloc[:tercile_size]["y_actual"].mean()
                    )
                    ret_df.append(gross_ret)
                ann_return = np.mean(ret_df) * 252 if ret_df else np.nan

                rows.append({
                    "market": mkt.upper(),
                    "target": tgt,
                    "day_bucket": bucket_label,
                    "n_days": n_days,
                    "mean_ic": round(mean_ic, 6) if not np.isnan(mean_ic) else np.nan,
                    "gross_sharpe": round(gross_sh, 4) if not np.isnan(gross_sh) else np.nan,
                    "ann_return": round(ann_return, 4) if not np.isnan(ann_return) else np.nan,
                })

    we_df = pd.DataFrame(rows)
    we_df.to_csv(f"{OUTPUT_DIR}/weekend_effect.csv", index=False)
    print(f"  Saved: output/weekend_effect.csv ({len(we_df)} rows)")
    return we_df


# ─── 6F. Summary ─────────────────────────────────────────────────────────────

def write_summary(abl_df, decomp_df, regime_df, stab_df, overlap_df, we_df):
    print("\n" + "=" * 60)
    print("6F. WRITING DIAGNOSTICS SUMMARY")
    print("=" * 60)

    lines = []
    lines.append("STAGE 6 DIAGNOSTICS SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    # --- Ablation ---
    lines.append("6A. FEATURE ABLATION")
    lines.append("-" * 40)
    flagged = abl_df[abl_df["meets_criterion_3"] == "FLAG_LARGE_DROP"]
    if len(flagged) > 0:
        lines.append(f"  CRITERION 3 MET: {len(flagged)} (market, target, group) combos show delta_gross_sharpe < -0.15:")
        for _, row in flagged.iterrows():
            lines.append(f"    {row['market']} / {row['target']} / drop={row['dropped_group']}: "
                         f"delta_gross_sharpe={row['delta_gross_sharpe_vs_baseline']:.3f}")
    else:
        lines.append("  CRITERION 3 NOT MET: No (market, target, group) shows delta_gross_sharpe < -0.15.")

    lines.append("")
    lines.append("  Most load-bearing group per (market, target):")
    for (mkt, tgt), grp in abl_df.groupby(["market", "target"]):
        worst = grp.sort_values("delta_gross_sharpe_vs_baseline").iloc[0]
        lines.append(f"    {mkt}/{tgt}: dropping '{worst['dropped_group']}' causes largest gross Sharpe drop: "
                     f"{worst['delta_gross_sharpe_vs_baseline']:.3f}")

    lines.append("")

    # --- Return Decomposition ---
    lines.append("6B. RETURN COMPONENT ANALYSIS")
    lines.append("-" * 40)
    for mkt in ["HK", "KR"]:
        sub = decomp_df[decomp_df["market"] == mkt]
        for _, row in sub.iterrows():
            lines.append(f"  {mkt}/{row['target']}: mean_ic={row['mean_ic_test']:.4f}, "
                         f"dir_acc={row['dir_acc_mean']:.3f}, p={row['bootstrap_p_value']:.3f}")
        gap_ic = sub[sub["target"] == "gap"]["mean_ic_test"].values[0]
        intra_ic = sub[sub["target"] == "intraday"]["mean_ic_test"].values[0]
        diff = gap_ic - intra_ic
        criterion2 = "CRITERION 2 MET" if diff > 0.02 else "criterion 2 not met"
        lines.append(f"  {mkt}: gap_ic - intraday_ic = {diff:.4f} => {criterion2}")
        lines.append("")

    # --- Regime Analysis ---
    lines.append("6C. REGIME ANALYSIS")
    lines.append("-" * 40)
    for mkt in ["HK", "KR"]:
        for tgt in ["gap"]:
            sub = regime_df[(regime_df["market"] == mkt) & (regime_df["target"] == tgt)]
            for regime_type in ["vix", "btc_trend", "crypto_vol"]:
                r = sub[sub["regime_type"] == regime_type]
                if r.empty:
                    continue
                pieces = []
                for _, row in r.iterrows():
                    pieces.append(f"{row['regime_bucket']}: IC={row['mean_ic']:.4f} Sharpe={row['gross_sharpe']:.2f} (n={row['n_days']})")
                lines.append(f"  {mkt}/{tgt}/{regime_type}: " + " | ".join(pieces))
    lines.append("")

    # --- SHAP Stability ---
    lines.append("6D. SHAP STABILITY")
    lines.append("-" * 40)
    # Mean overlap and rank corr
    mean_overlap = overlap_df["top10_overlap_pct"].mean()
    mean_rcorr = overlap_df["rank_corr"].mean()
    lines.append(f"  Avg top-10 overlap consecutive folds: {mean_overlap:.1f}%")
    lines.append(f"  Avg Spearman rank correlation across folds: {mean_rcorr:.3f}")

    lines.append("  Top-3 load-bearing features (gap target, by pct_folds_in_top10):")
    for mkt in ["HK", "KR"]:
        sub = stab_df[(stab_df["market"] == mkt) & (stab_df["target"] == "gap")]
        top3 = sub.sort_values("pct_folds_in_top10", ascending=False).head(3)
        for _, row in top3.iterrows():
            lines.append(f"    {mkt}/gap: {row['feature']} — {row['pct_folds_in_top10']:.0f}% folds in top10, "
                         f"mean_abs_shap={row['mean_abs_shap_overall']:.2e}")

    lines.append("")

    # --- Weekend Effect ---
    lines.append("6E. WEEKEND EFFECT")
    lines.append("-" * 40)
    for mkt in ["HK", "KR"]:
        for tgt in ["gap"]:
            sub = we_df[(we_df["market"] == mkt) & (we_df["target"] == tgt)]
            mon = sub[sub["day_bucket"] == "monday"]
            tue = sub[sub["day_bucket"] == "tue_fri"]
            if not mon.empty and not tue.empty:
                lines.append(
                    f"  {mkt}/{tgt}: Monday IC={mon.iloc[0]['mean_ic']:.4f} "
                    f"Sharpe={mon.iloc[0]['gross_sharpe']:.2f} (n={mon.iloc[0]['n_days']}) | "
                    f"Tue-Fri IC={tue.iloc[0]['mean_ic']:.4f} "
                    f"Sharpe={tue.iloc[0]['gross_sharpe']:.2f} (n={tue.iloc[0]['n_days']})"
                )
    lines.append("")
    lines.append("  Hypothesis: Monday (Friday close -> Monday open, ~65-hr window) should show")
    lines.append("  stronger signal if overnight crypto information persists through the weekend.")
    mon_sub = we_df[(we_df["target"] == "gap") & (we_df["day_bucket"] == "monday")]
    tue_sub = we_df[(we_df["target"] == "gap") & (we_df["day_bucket"] == "tue_fri")]
    if not mon_sub.empty and not tue_sub.empty:
        avg_mon_ic = mon_sub["mean_ic"].mean()
        avg_tue_ic = tue_sub["mean_ic"].mean()
        if avg_mon_ic > avg_tue_ic:
            lines.append(f"  SUPPORTS hypothesis: Monday IC ({avg_mon_ic:.4f}) > Tue-Fri IC ({avg_tue_ic:.4f})")
        else:
            lines.append(f"  CONTRADICTS hypothesis: Monday IC ({avg_mon_ic:.4f}) <= Tue-Fri IC ({avg_tue_ic:.4f})")

    lines.append("")
    lines.append("Output files:")
    for f in ["feature_ablation.csv", "return_decomposition.csv", "regime_analysis.csv",
              "shap_stability.csv", "shap_fold_overlap.csv", "weekend_effect.csv"]:
        lines.append(f"  output/{f}")

    summary_text = "\n".join(lines)
    with open(f"{OUTPUT_DIR}/diagnostics_summary.txt", "w") as fh:
        fh.write(summary_text)
    print(f"  Saved: output/diagnostics_summary.txt")
    print()
    print(summary_text)


# ─── Validation ───────────────────────────────────────────────────────────────

def validate(abl_df, regime_df):
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # 18 ablation rows
    n_rows = len(abl_df)
    expected = 18
    status = "PASS" if n_rows == expected else f"FAIL: expected {expected}"
    print(f"  Ablation rows: {n_rows} [{status}]")

    # Regime NaN check
    nan_ic = regime_df["mean_ic"].isna().sum()
    status2 = "PASS" if nan_ic == 0 else f"FAIL: {nan_ic} NaN"
    print(f"  Regime NaN mean_ic: {nan_ic} [{status2}]")

    # Compact ablation table
    print("\n  ABLATION COMPACT TABLE (market, target, dropped_group, delta_gross_sharpe):")
    print(f"  {'market':<6} {'target':<9} {'dropped_group':<14} {'delta_gross_sharpe':>18} {'flag':<20}")
    for _, row in abl_df.sort_values(["market", "target", "dropped_group"]).iterrows():
        flag = row["meets_criterion_3"] if row["meets_criterion_3"] else ""
        val = f"{row['delta_gross_sharpe_vs_baseline']:.4f}" if not np.isnan(row["delta_gross_sharpe_vs_baseline"]) else "NaN"
        print(f"  {row['market']:<6} {row['target']:<9} {row['dropped_group']:<14} {val:>18} {flag:<20}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.time()
    print("Stage 6 Diagnostics starting...")

    # Load baseline backtest summary
    baseline_summary = pd.read_csv(f"{OUTPUT_DIR}/backtest_summary.csv")
    # Normalize market column
    baseline_summary["market"] = baseline_summary["market"].str.upper()

    abl_df = run_ablation(baseline_summary)
    decomp_df = run_return_decomposition()
    regime_df = run_regime_analysis()
    stab_df, overlap_df = run_shap_stability()
    we_df = run_weekend_effect()
    write_summary(abl_df, decomp_df, regime_df, stab_df, overlap_df, we_df)
    validate(abl_df, regime_df)

    elapsed = (time.time() - t0) / 60
    print(f"\nStage 6 complete in {elapsed:.1f} min.")


if __name__ == "__main__":
    main()
