# ABOUTME: Stage P2-7: Walk-forward LightGBM training for 12 configurations (4 variants x 3 targets).
# ABOUTME: Inputs: output/features_track_a_control_{hk,kr}.parquet, output/features_track_a_index.parquet
# ABOUTME: Outputs: predictions, SHAP, training log CSVs/parquets per run; summary in logs/stage_p2-7_lightgbm.log
# ABOUTME: Run: source .venv/bin/activate && python3 scripts/stage_p2-7_lightgbm.py

import os
import time
import warnings
import random
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")

PROJECT_ROOT = "/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity_p2"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Hyperparameter grids (verbatim from Pass 1) ---
PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_samples": [10, 20, 50],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [0, 0.1, 1.0],
}

FALLBACK_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [5],
    "learning_rate": [0.05],
    "min_child_samples": [20],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [0.1],
}

N_ITER_SEARCH = 10
N_CV_FOLDS = 3
PURGE_GAP_DAYS = 5
MIN_DAILY_TICKERS_FOR_IC = 5
OVERFIT_THRESHOLD = 0.20
MIN_TRAIN_DAYS = 252

# Control variants: target cols use "tgt_" prefix
CONTROL_TARGET_COLS = {"gap": "tgt_gap", "intraday": "tgt_intraday", "cc": "tgt_cc"}
# Index variants: target cols are named directly
INDEX_TARGET_COLS = {"gap": "gap_return", "intraday": "intraday_return", "cc": "cc_return"}

# For index, MIN_DAILY_TICKERS_FOR_IC = 1 since only 1 ticker per model
INDEX_MIN_TICKERS = 1

# Configurations: (variant, market, parquet_path, ticker_filter, target_col_map, min_tickers_ic)
CONFIGS = [
    # Control HK
    {
        "variant": "control", "market": "hk",
        "parquet": os.path.join(OUTPUT_DIR, "features_track_a_control_hk.parquet"),
        "ticker_filter": None,
        "target_col_map": CONTROL_TARGET_COLS,
        "min_tickers_ic": MIN_DAILY_TICKERS_FOR_IC,
    },
    # Control KR
    {
        "variant": "control", "market": "kr",
        "parquet": os.path.join(OUTPUT_DIR, "features_track_a_control_kr.parquet"),
        "ticker_filter": None,
        "target_col_map": CONTROL_TARGET_COLS,
        "min_tickers_ic": MIN_DAILY_TICKERS_FOR_IC,
    },
    # Index HSI
    {
        "variant": "index", "market": "hk",
        "parquet": os.path.join(OUTPUT_DIR, "features_track_a_index.parquet"),
        "ticker_filter": "^HSI",
        "target_col_map": INDEX_TARGET_COLS,
        "min_tickers_ic": INDEX_MIN_TICKERS,
    },
    # Index KOSPI
    {
        "variant": "index", "market": "kr",
        "parquet": os.path.join(OUTPUT_DIR, "features_track_a_index.parquet"),
        "ticker_filter": "^KS11",
        "target_col_map": INDEX_TARGET_COLS,
        "min_tickers_ic": INDEX_MIN_TICKERS,
    },
]

TARGETS = ["gap", "intraday", "cc"]


# --- Core helper functions (identical to Pass 1) ---

def spearman_ic(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return np.nan
    r, _ = spearmanr(y_true[mask], y_pred[mask])
    return r


def compute_fold_ic(y_true, y_pred, dates, min_tickers=MIN_DAILY_TICKERS_FOR_IC):
    df = pd.DataFrame({"date": dates, "y_true": y_true, "y_pred": y_pred})
    ics = []
    skipped = 0
    for d, grp in df.groupby("date"):
        valid = grp.dropna(subset=["y_true", "y_pred"])
        if len(valid) < min_tickers:
            skipped += 1
            continue
        ic = spearman_ic(valid["y_true"].values, valid["y_pred"].values)
        if not np.isnan(ic):
            ics.append(ic)
    mean_ic = np.mean(ics) if ics else np.nan
    return mean_ic, skipped


def sample_params(grid, n_iter, seed=42):
    rng = random.Random(seed)
    keys = list(grid.keys())
    seen = set()
    samples = []
    attempts = 0
    max_attempts = n_iter * 20
    while len(samples) < n_iter and attempts < max_attempts:
        attempts += 1
        combo = tuple(rng.choice(grid[k]) for k in keys)
        if combo not in seen:
            seen.add(combo)
            samples.append(dict(zip(keys, combo)))
    return samples


def purged_cv_indices(dates_array, n_splits, purge_days):
    n = len(dates_array)
    fold_size = n // (n_splits + 1)
    folds = []
    for i in range(n_splits):
        val_start = (i + 1) * fold_size
        val_end = min(val_start + fold_size, n)
        purge_start = max(0, val_start - purge_days)
        train_idx = list(range(0, purge_start))
        val_idx = list(range(val_start, val_end))
        if len(train_idx) < 10 or len(val_idx) < 5:
            continue
        folds.append((train_idx, val_idx))
    return folds


def fit_and_score_params(params, X_train_rows, y_train_rows, date_ordinals, n_splits, purge_days):
    folds = purged_cv_indices(date_ordinals, n_splits, purge_days)
    if not folds:
        return np.nan
    fold_ics = []
    for train_idx, val_idx in folds:
        X_tr = X_train_rows[train_idx]
        y_tr = y_train_rows[train_idx]
        X_val = X_train_rows[val_idx]
        y_val = y_train_rows[val_idx]
        valid_tr = ~np.isnan(y_tr)
        valid_val = ~np.isnan(y_val)
        if valid_tr.sum() < 10 or valid_val.sum() < 2:
            continue
        model = lgb.LGBMRegressor(verbose=-1, n_jobs=-1, random_state=42, **params)
        model.fit(X_tr[valid_tr], y_tr[valid_tr])
        preds = model.predict(X_val[valid_val])
        ic = spearman_ic(y_val[valid_val], preds)
        if not np.isnan(ic):
            fold_ics.append(ic)
    return np.mean(fold_ics) if fold_ics else np.nan


def run_hyperparameter_search(X_train, y_train, dates_train, fallback_log, seed=42):
    unique_dates = sorted(np.unique(dates_train))
    date_to_ord = {d: i for i, d in enumerate(unique_dates)}
    date_ordinals = np.array([date_to_ord[d] for d in dates_train])

    X_arr = X_train.values if hasattr(X_train, "values") else np.array(X_train)
    y_arr = np.array(y_train, dtype=float)

    candidates = sample_params(PARAM_GRID, N_ITER_SEARCH, seed=seed)
    t0 = time.time()
    first_ic = fit_and_score_params(
        candidates[0], X_arr, y_arr, date_ordinals, N_CV_FOLDS, PURGE_GAP_DAYS
    )
    t1 = time.time()
    elapsed_first = t1 - t0

    use_fallback = False
    if elapsed_first > 600:
        use_fallback = True
        fallback_log.append("FALLBACK: first candidate took {:.1f}s; switching to fallback grid.".format(elapsed_first))
        print(f"  [FALLBACK] First candidate took {elapsed_first:.1f}s — switching to fallback grid.")
        candidates = list(itertools.product(*FALLBACK_GRID.values()))
        candidates = [dict(zip(FALLBACK_GRID.keys(), c)) for c in candidates]
        fallback_log.append(f"Fallback grid has {len(candidates)} combinations.")
    else:
        candidates = [candidates[0]] + candidates[1:]

    results = [(candidates[0], first_ic)]
    for i, params in enumerate(candidates[1:], start=1):
        ic = fit_and_score_params(
            params, X_arr, y_arr, date_ordinals, N_CV_FOLDS, PURGE_GAP_DAYS
        )
        results.append((params, ic))

    results_valid = [(p, ic) for p, ic in results if not np.isnan(ic)]
    if not results_valid:
        best_params = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
                       "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
                       "reg_alpha": 0.1, "reg_lambda": 0.1}
        fallback_log.append("WARNING: all CV folds returned NaN IC; using hardcoded default params.")
    else:
        best_params, best_ic = max(results_valid, key=lambda x: x[1])
        print(f"  Best CV IC: {best_ic:.4f} | params: {best_params}")

    return best_params, use_fallback


def run_target(df_all, feature_cols, target_name, target_col, all_dates, oos_months,
               min_tickers_ic=MIN_DAILY_TICKERS_FOR_IC):
    """Full walk-forward for one target. Returns (pred_rows, shap_rows, log_rows, fallback_notes, wall_time)."""
    print(f"\n{'='*60}")
    print(f"  TARGET: {target_name.upper()}")
    print(f"{'='*60}")
    t_start = time.time()

    pred_rows = []
    shap_rows = []
    log_rows = []
    fallback_notes = []
    best_params = None
    used_fallback = False
    last_search_year = None

    for fold_id, month_str in enumerate(oos_months):
        month_start = pd.Timestamp(month_str + "-01")
        next_month = month_start + pd.offsets.MonthBegin(1)

        train_mask = df_all["date"] < month_start
        test_mask = (df_all["date"] >= month_start) & (df_all["date"] < next_month)

        train_df = df_all[train_mask].copy().dropna(subset=[target_col])
        test_df = df_all[test_mask].copy().dropna(subset=[target_col])

        if len(train_df) == 0 or len(test_df) == 0:
            print(f"  Fold {fold_id:02d} ({month_str}): SKIP — empty train or test after NaN drop.")
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols]
        y_test = test_df[target_col].values
        dates_train = train_df["date"].values

        current_year = month_start.year
        is_year_boundary = (month_start.month == 1)
        is_first_fold = (fold_id == 0)
        do_search = is_first_fold or (is_year_boundary and current_year != last_search_year)

        if do_search:
            print(f"  Fold {fold_id:02d} ({month_str}): HYPERPARAMETER SEARCH (n_train={len(train_df)})")
            t_srch = time.time()
            best_params, uf = run_hyperparameter_search(
                X_train, y_train, dates_train, fallback_notes, seed=42 + fold_id
            )
            used_fallback = used_fallback or uf
            last_search_year = current_year
            print(f"  Search done in {time.time() - t_srch:.1f}s.")
        else:
            print(f"  Fold {fold_id:02d} ({month_str}): reusing params (n_train={len(train_df)}, n_test={len(test_df)})")

        # Fit final model
        model = lgb.LGBMRegressor(verbose=-1, n_jobs=-1, random_state=42, **best_params)
        X_fit = train_df[feature_cols]
        y_fit = train_df[target_col].values
        dates_fit = train_df["date"].values
        model.fit(X_fit, y_fit)

        y_pred_test = model.predict(X_test)

        ic_test, n_skipped = compute_fold_ic(y_test, y_pred_test, test_df["date"].values, min_tickers=min_tickers_ic)
        mse_test = np.mean((y_test - y_pred_test) ** 2)
        mae_test = np.mean(np.abs(y_test - y_pred_test))
        sign_match = (np.sign(y_pred_test) == np.sign(y_test)) & (y_test != 0)
        dir_acc_test = sign_match.mean() if len(sign_match) > 0 else np.nan

        y_pred_train = model.predict(X_fit)
        ic_train, _ = compute_fold_ic(y_fit, y_pred_train, dates_fit, min_tickers=min_tickers_ic)
        overfit_flag = ""
        if not np.isnan(ic_train) and not np.isnan(ic_test):
            if (ic_train - ic_test) > OVERFIT_THRESHOLD:
                overfit_flag = "OVERFIT_FLAG"
                print(f"  *** OVERFIT FLAG: ic_train={ic_train:.4f}, ic_test={ic_test:.4f}, diff={ic_train - ic_test:.4f}")

        print(f"  Fold {fold_id:02d} ({month_str}): n_test={len(test_df)}, IC_test={ic_test:.4f}, dir_acc={dir_acc_test:.3f} {overfit_flag}")

        for i in range(len(test_df)):
            pred_rows.append({
                "date": test_df["date"].iloc[i],
                "ticker": test_df["ticker"].iloc[i],
                "y_pred": y_pred_test[i],
                "y_actual": y_test[i],
                "fold_id": fold_id,
            })

        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test)
            for i in range(len(test_df)):
                for j, feat in enumerate(feature_cols):
                    shap_rows.append({
                        "date": test_df["date"].iloc[i],
                        "ticker": test_df["ticker"].iloc[i],
                        "fold_id": fold_id,
                        "feature": feat,
                        "shap_value": shap_vals[i, j],
                    })
        except Exception as e:
            print(f"  SHAP error fold {fold_id}: {e}")

        log_rows.append({
            "fold_id": fold_id,
            "train_start": str(train_df["date"].min().date()),
            "train_end": str(train_df["date"].max().date()),
            "test_start": str(test_df["date"].min().date()),
            "test_end": str(test_df["date"].max().date()),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "ic_train": round(ic_train, 6) if not np.isnan(ic_train) else np.nan,
            "ic_test": round(ic_test, 6) if not np.isnan(ic_test) else np.nan,
            "mse_test": round(mse_test, 8),
            "mae_test": round(mae_test, 6),
            "dir_acc_test": round(dir_acc_test, 4) if not np.isnan(dir_acc_test) else np.nan,
            "best_params": str(best_params),
            "overfit_flag": overfit_flag,
            "n_ic_days_skipped": n_skipped,
        })

    wall_time = time.time() - t_start
    print(f"\n  Target '{target_name}' done in {wall_time/60:.1f} min.")
    return pred_rows, shap_rows, log_rows, fallback_notes, wall_time


def run_config(cfg, global_log_lines, global_start):
    """Run all 3 targets for a given config. Returns list of summary dicts."""
    variant = cfg["variant"]
    market = cfg["market"]
    ticker_filter = cfg["ticker_filter"]
    target_col_map = cfg["target_col_map"]
    min_tickers_ic = cfg["min_tickers_ic"]
    label = f"{variant}_{market}"

    print(f"\n{'#'*70}")
    print(f"CONFIG: {label.upper()}  ticker_filter={ticker_filter}")
    print(f"{'#'*70}")

    df = pd.read_parquet(cfg["parquet"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if ticker_filter is not None:
        df = df[df["ticker"] == ticker_filter].copy()
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Exclude columns: non-feature cols
    # Control: has market, is_weekend_gap, tgt_* cols
    # Index: has is_weekend_gap, *_return target cols
    base_exclude = {"date", "ticker", "market", "is_weekend_gap"}
    target_cols_all = set(target_col_map.values())
    # Also exclude any other return/target cols that may appear
    all_return_cols = {c for c in df.columns if c.endswith("_return") or c.startswith("tgt_")}
    exclude_cols = base_exclude | target_cols_all | all_return_cols
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    # Compute OOS months
    all_dates = sorted(df["date"].unique())
    all_months = sorted(set(pd.Timestamp(d).strftime("%Y-%m") for d in all_dates))
    oos_months = []
    for m in all_months:
        month_start = pd.Timestamp(m + "-01")
        train_days = [d for d in all_dates if d < month_start]
        if len(train_days) >= MIN_TRAIN_DAYS:
            oos_months.append(m)
    print(f"  OOS months: {len(oos_months)}  ({oos_months[0]} to {oos_months[-1]})")

    run_summaries = []

    for target_name in TARGETS:
        target_col = target_col_map[target_name]
        run_label = f"{label}_{target_name}"
        t_run_start = time.time()

        # Skip if predictions CSV already exists and is non-empty
        pred_path_check = os.path.join(OUTPUT_DIR, f"predictions_lgbm_{run_label}.csv")
        if os.path.exists(pred_path_check) and os.path.getsize(pred_path_check) > 0:
            print(f"  SKIP {run_label}: predictions CSV already exists ({pred_path_check})")
            global_log_lines.append(f"SKIP {run_label}: predictions CSV already exists.")
            # Read existing file to build summary
            existing_pred = pd.read_csv(pred_path_check)
            existing_log_path = os.path.join(OUTPUT_DIR, f"training_log_lgbm_{run_label}.csv")
            if os.path.exists(existing_log_path):
                existing_log = pd.read_csv(existing_log_path)
                mean_ic = existing_log["ic_test"].mean()
                n_folds = len(existing_log)
            else:
                mean_ic = np.nan
                n_folds = 0
            run_summaries.append({
                "run": run_label,
                "status": "SKIP",
                "n_rows": len(existing_pred),
                "n_folds": n_folds,
                "oos_mean_ic": round(mean_ic, 4) if not np.isnan(mean_ic) else np.nan,
                "n_overfit_flags": 0,
                "fallback": False,
                "elapsed_s": 0,
            })
            continue

        pred_rows, shap_rows, log_rows, fallback_notes, wall_time = run_target(
            df, feature_cols, target_name, target_col, all_dates, oos_months,
            min_tickers_ic=min_tickers_ic,
        )

        if not pred_rows:
            msg = f"BLOCK: {run_label} — predictions empty after all folds."
            print(f"  {msg}")
            global_log_lines.append(msg)
            run_summaries.append({
                "run": run_label, "status": "BLOCK", "n_rows": 0,
                "n_folds": 0, "oos_mean_ic": np.nan, "fallback": False,
                "elapsed_s": int(time.time() - t_run_start),
            })
            continue

        pred_df = pd.DataFrame(pred_rows)
        shap_df = pd.DataFrame(shap_rows)
        log_df = pd.DataFrame(log_rows)

        # Check for NaN in y_pred
        nan_preds = pred_df["y_pred"].isna().sum()
        if nan_preds > 0:
            msg = f"BLOCK: {run_label} — {nan_preds} NaN predictions found."
            print(f"  {msg}")
            global_log_lines.append(msg)

        # Output paths
        pred_path = os.path.join(OUTPUT_DIR, f"predictions_lgbm_{run_label}.csv")
        shap_path = os.path.join(OUTPUT_DIR, f"shap_per_fold_{run_label}.parquet")
        log_path = os.path.join(OUTPUT_DIR, f"training_log_lgbm_{run_label}.csv")

        pred_df.to_csv(pred_path, index=False)
        shap_df.to_parquet(shap_path, index=False)
        log_df.to_csv(log_path, index=False)

        print(f"  Saved: {pred_path}")

        mean_ic = log_df["ic_test"].mean()
        n_folds = len(log_df)
        n_overfit = (log_df["overfit_flag"] == "OVERFIT_FLAG").sum()
        elapsed = int(time.time() - t_run_start)

        log_line = (
            f"{run_label}: rows={len(pred_df)}, folds={n_folds}, "
            f"oos_mean_ic={mean_ic:.4f}, overfit_flags={n_overfit}, "
            f"fallback={'YES' if fallback_notes else 'NO'}, elapsed={elapsed}s"
        )
        global_log_lines.append(log_line)
        print(f"  LOG: {log_line}")

        if fallback_notes:
            for fn in fallback_notes:
                global_log_lines.append(f"  {run_label} FALLBACK: {fn}")

        run_summaries.append({
            "run": run_label,
            "status": "OK",
            "n_rows": len(pred_df),
            "n_folds": n_folds,
            "oos_mean_ic": round(mean_ic, 4) if not np.isnan(mean_ic) else np.nan,
            "n_overfit_flags": n_overfit,
            "fallback": len(fallback_notes) > 0,
            "elapsed_s": elapsed,
        })

    return run_summaries


def main():
    overall_start = time.time()
    global_log_lines = [
        "=== STAGE P2-7 LIGHTGBM LOG ===",
        f"Start: {pd.Timestamp.now()}",
        "",
    ]

    all_summaries = []

    for cfg in CONFIGS:
        summaries = run_config(cfg, global_log_lines, overall_start)
        all_summaries.extend(summaries)

    total_wall = time.time() - overall_start

    # --- Final summary print ---
    print(f"\n{'='*70}")
    print("FINAL SUMMARY — STAGE P2-7")
    print(f"{'='*70}")
    print(f"{'Run':<35} {'Status':<8} {'Rows':>8} {'Folds':>6} {'OOS IC':>8} {'Fallback':<10} {'Time(s)':>8}")
    print("-" * 90)
    for s in all_summaries:
        print(
            f"{s['run']:<35} {s['status']:<8} {s.get('n_rows', 0):>8} "
            f"{s.get('n_folds', 0):>6} {s.get('oos_mean_ic', float('nan')):>8.4f} "
            f"{'YES' if s.get('fallback') else 'NO':<10} {s.get('elapsed_s', 0):>8}"
        )
    print(f"\nTotal wall clock: {total_wall/60:.1f} min")

    blocks = [s for s in all_summaries if s["status"] == "BLOCK"]
    if blocks:
        print(f"\nBLOCK RUNS: {[b['run'] for b in blocks]}")

    completed = [s for s in all_summaries if s["status"] in ("OK", "SKIP")]
    print(f"\nCompleted: {len(completed)}/12 runs")

    # --- Write log file ---
    global_log_lines.append("")
    global_log_lines.append("=== PER-RUN SUMMARY TABLE ===")
    global_log_lines.append(f"{'Run':<35} {'Status':<8} {'Rows':>8} {'Folds':>6} {'OOS IC':>8} {'Fallback':<10}")
    for s in all_summaries:
        global_log_lines.append(
            f"{s['run']:<35} {s['status']:<8} {s.get('n_rows', 0):>8} "
            f"{s.get('n_folds', 0):>6} {s.get('oos_mean_ic', float('nan')):>8.4f} "
            f"{'YES' if s.get('fallback') else 'NO':<10}"
        )
    global_log_lines.append("")
    global_log_lines.append(f"Total wall clock: {total_wall/60:.1f} min")
    global_log_lines.append(f"Completed runs (OK+SKIP): {len(completed)}/12")
    if blocks:
        global_log_lines.append(f"BLOCK runs: {[b['run'] for b in blocks]}")

    log_path = os.path.join(LOG_DIR, "stage_p2-7_lightgbm.log")
    with open(log_path, "w") as f:
        f.write("\n".join(global_log_lines) + "\n")
    print(f"\nLog written: {log_path}")


if __name__ == "__main__":
    main()
