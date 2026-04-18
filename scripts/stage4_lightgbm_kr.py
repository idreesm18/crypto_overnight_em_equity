# ABOUTME: Walk-forward LightGBM training for KR market, 3 targets: gap (PRIMARY), intraday, close-to-close.
# ABOUTME: Inputs: output/features_track_a_kr.parquet. Outputs: predictions, SHAP, training log CSVs/parquets per target.
# ABOUTME: Run: source .venv/bin/activate && python3 scripts/stage4_lightgbm_kr.py

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

PROJECT_ROOT = "/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity"
INPUT_PATH = os.path.join(PROJECT_ROOT, "output", "features_track_a_kr.parquet")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TARGETS = ["gap", "intraday", "cc"]
TARGET_COLS = {"gap": "tgt_gap", "intraday": "tgt_intraday", "cc": "tgt_cc"}

EXCLUDE_COLS = {"date", "ticker", "market", "is_weekend_gap", "tgt_gap", "tgt_intraday", "tgt_cc"}

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


def spearman_ic(y_true, y_pred):
    """Compute Spearman IC, returning NaN if insufficient data."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return np.nan
    r, _ = spearmanr(y_true[mask], y_pred[mask])
    return r


def compute_fold_ic(y_true, y_pred, dates):
    """Per-day IC averaged over days. Skip days with < MIN_DAILY_TICKERS_FOR_IC tickers."""
    df = pd.DataFrame({"date": dates, "y_true": y_true, "y_pred": y_pred})
    ics = []
    skipped = 0
    for d, grp in df.groupby("date"):
        valid = grp.dropna(subset=["y_true", "y_pred"])
        if len(valid) < MIN_DAILY_TICKERS_FOR_IC:
            skipped += 1
            continue
        ic = spearman_ic(valid["y_true"].values, valid["y_pred"].values)
        if not np.isnan(ic):
            ics.append(ic)
    mean_ic = np.mean(ics) if ics else np.nan
    return mean_ic, skipped


def sample_params(grid, n_iter, seed=42):
    """Random sample of parameter combinations from grid."""
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
    """
    Purged time-series CV. Returns list of (train_idx, val_idx) tuples.
    Uses last n_splits contiguous blocks as val, with purge before each.
    dates_array: sorted numpy array of trading day indices (0..N-1 after dedup-date ordering).
    """
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
    """Run CV for one param set, return mean IC across folds."""
    folds = purged_cv_indices(date_ordinals, n_splits, purge_days)
    if not folds:
        return np.nan
    fold_ics = []
    for train_idx, val_idx in folds:
        X_tr = X_train_rows[train_idx]
        y_tr = y_train_rows[train_idx]
        X_val = X_train_rows[val_idx]
        y_val = y_train_rows[val_idx]
        # drop NaN target rows
        valid_tr = ~np.isnan(y_tr)
        valid_val = ~np.isnan(y_val)
        if valid_tr.sum() < 10 or valid_val.sum() < 2:
            continue
        model = lgb.LGBMRegressor(
            verbose=-1,
            n_jobs=-1,
            random_state=42,
            **params,
        )
        model.fit(X_tr[valid_tr], y_tr[valid_tr])
        preds = model.predict(X_val[valid_val])
        ic = spearman_ic(y_val[valid_val], preds)
        if not np.isnan(ic):
            fold_ics.append(ic)
    return np.mean(fold_ics) if fold_ics else np.nan


def run_hyperparameter_search(X_train, y_train, dates_train, fallback_log, seed=42):
    """
    Run randomized search with timing fallback.
    Returns (best_params, used_fallback).
    """
    # Prepare date ordinals for purged CV (map each row to sorted date index)
    unique_dates = sorted(np.unique(dates_train))
    date_to_ord = {d: i for i, d in enumerate(unique_dates)}
    date_ordinals = np.array([date_to_ord[d] for d in dates_train])

    X_arr = X_train.values if hasattr(X_train, "values") else np.array(X_train)
    y_arr = np.array(y_train, dtype=float)

    # Time the first candidate to check if we need fallback
    candidates = sample_params(PARAM_GRID, N_ITER_SEARCH, seed=seed)
    t0 = time.time()
    first_ic = fit_and_score_params(
        candidates[0], X_arr, y_arr, date_ordinals, N_CV_FOLDS, PURGE_GAP_DAYS
    )
    t1 = time.time()
    elapsed_first = t1 - t0

    use_fallback = False
    if elapsed_first > 600:  # 10 min for one candidate means full search >> 10 min
        use_fallback = True
        fallback_log.append("FALLBACK: first candidate took {:.1f}s; switching to fallback grid.".format(elapsed_first))
        print(f"  [FALLBACK] First candidate took {elapsed_first:.1f}s — switching to fallback grid.")
        candidates = list(itertools.product(*FALLBACK_GRID.values()))
        candidates = [dict(zip(FALLBACK_GRID.keys(), c)) for c in candidates]
        fallback_log.append(f"Fallback grid has {len(candidates)} combinations.")
    else:
        # Continue with remaining random candidates
        candidates = [candidates[0]] + candidates[1:]  # include already-evaluated first

    results = [(candidates[0], first_ic)]
    for i, params in enumerate(candidates[1:], start=1):
        ic = fit_and_score_params(
            params, X_arr, y_arr, date_ordinals, N_CV_FOLDS, PURGE_GAP_DAYS
        )
        results.append((params, ic))

    # Pick best by IC
    results_valid = [(p, ic) for p, ic in results if not np.isnan(ic)]
    if not results_valid:
        # all NaN — return a reasonable default
        best_params = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
                       "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
                       "reg_alpha": 0.1, "reg_lambda": 0.1}
        fallback_log.append("WARNING: all CV folds returned NaN IC; using hardcoded default params.")
    else:
        best_params, best_ic = max(results_valid, key=lambda x: x[1])
        print(f"  Best CV IC: {best_ic:.4f} | params: {best_params}")

    return best_params, use_fallback


def run_target(df_all, feature_cols, target_name, target_col, all_dates, oos_months):
    """
    Full walk-forward for one target. Returns (pred_df, shap_rows, log_rows, fallback_notes).
    """
    print(f"\n{'='*60}")
    print(f"TARGET: {target_name.upper()}")
    print(f"{'='*60}")
    t_start_target = time.time()

    pred_rows = []
    shap_rows = []
    log_rows = []
    fallback_notes = []
    best_params = None
    used_fallback = False
    last_search_year = None

    for fold_id, month_str in enumerate(oos_months):
        month_start = pd.Timestamp(month_str + "-01")
        # Determine month end (first day of next month)
        next_month = month_start + pd.offsets.MonthBegin(1)

        # Train: all rows with date < month_start
        train_mask = df_all["date"] < month_start
        test_mask = (df_all["date"] >= month_start) & (df_all["date"] < next_month)

        train_df = df_all[train_mask].copy()
        test_df = df_all[test_mask].copy()

        # Drop NaN target rows
        train_df = train_df.dropna(subset=[target_col])
        test_df = test_df.dropna(subset=[target_col])

        if len(train_df) == 0 or len(test_df) == 0:
            print(f"  Fold {fold_id:02d} ({month_str}): SKIP — empty train or test after NaN drop.")
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols]
        y_test = test_df[target_col].values
        dates_train = train_df["date"].values

        # Determine if we should search this fold
        current_year = month_start.year
        is_year_boundary = (month_start.month == 1)
        is_first_fold = (fold_id == 0)

        do_search = is_first_fold or (is_year_boundary and current_year != last_search_year)

        if do_search:
            print(f"  Fold {fold_id:02d} ({month_str}): HYPERPARAMETER SEARCH (n_train={len(train_df)})")
            t_srch = time.time()
            best_params, uf = run_hyperparameter_search(
                X_train, y_train, dates_train,
                fallback_notes,
                seed=42 + fold_id
            )
            used_fallback = used_fallback or uf
            last_search_year = current_year
            elapsed_srch = time.time() - t_srch
            print(f"  Search done in {elapsed_srch:.1f}s.")
        else:
            print(f"  Fold {fold_id:02d} ({month_str}): reusing params (n_train={len(train_df)}, n_test={len(test_df)})")

        # Fit final model
        model = lgb.LGBMRegressor(verbose=-1, n_jobs=-1, random_state=42, **best_params)
        train_df_fit = train_df.dropna(subset=[target_col])
        X_fit = train_df_fit[feature_cols]
        y_fit = train_df_fit[target_col].values
        dates_fit = train_df_fit["date"].values
        model.fit(X_fit, y_fit)

        # Predict on test
        y_pred_test = model.predict(X_test)

        # Metrics on test
        ic_test, n_skipped = compute_fold_ic(y_test, y_pred_test, test_df["date"].values)
        mse_test = np.mean((y_test - y_pred_test) ** 2)
        mae_test = np.mean(np.abs(y_test - y_pred_test))
        # directional accuracy
        sign_match = (np.sign(y_pred_test) == np.sign(y_test)) & (y_test != 0)
        dir_acc_test = sign_match.mean() if len(sign_match) > 0 else np.nan

        # IC on training set (for overfit detection only)
        y_pred_train = model.predict(X_fit)
        ic_train, _ = compute_fold_ic(y_fit, y_pred_train, dates_fit)
        overfit_flag = ""
        if not np.isnan(ic_train) and not np.isnan(ic_test):
            if (ic_train - ic_test) > OVERFIT_THRESHOLD:
                overfit_flag = "OVERFIT_FLAG"
                print(f"  *** OVERFIT FLAG: ic_train={ic_train:.4f}, ic_test={ic_test:.4f}, diff={ic_train - ic_test:.4f}")

        print(f"  Fold {fold_id:02d} ({month_str}): n_test={len(test_df)}, IC_test={ic_test:.4f}, dir_acc={dir_acc_test:.3f} {overfit_flag}")

        # Store predictions
        for i in range(len(test_df)):
            pred_rows.append({
                "date": test_df["date"].iloc[i],
                "ticker": test_df["ticker"].iloc[i],
                "y_pred": y_pred_test[i],
                "y_actual": y_test[i],
                "fold_id": fold_id,
            })

        # SHAP on test set
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test)  # shape: (n_test, n_features)
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

        # Log row
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

    wall_time = time.time() - t_start_target
    print(f"\n  Target '{target_name}' done in {wall_time/60:.1f} min.")
    return pred_rows, shap_rows, log_rows, fallback_notes, wall_time


def validate_outputs(pred_df, target_name):
    """Inline validation of predictions dataframe."""
    print(f"\n--- VALIDATION: {target_name} ---")
    print(f"  n_predictions: {len(pred_df)}")
    dups = pred_df.duplicated(subset=["date", "ticker"]).sum()
    print(f"  Duplicate (date, ticker) rows: {dups}")
    if dups > 0:
        print("  WARNING: duplicates found!")
    # Confirm all prediction dates are after their training window
    # (For expanding window: test dates are always in month M, train dates are all < M. Already guaranteed by construction.)
    print(f"  Mean IC_test (from log): computed per target in summary below.")
    print(f"  date range: {pred_df['date'].min()} to {pred_df['date'].max()}")
    print(f"  tickers: {pred_df['ticker'].nunique()}")


def main():
    overall_start = time.time()
    print("Loading features parquet...")
    df = pd.read_parquet(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # Compute OOS months: months where training window >= 252 trading days
    all_dates = sorted(df["date"].unique())
    all_months = sorted(set(pd.Timestamp(d).strftime("%Y-%m") for d in all_dates))
    oos_months = []
    for m in all_months:
        month_start = pd.Timestamp(m + "-01")
        train_days = [d for d in all_dates if d < month_start]
        if len(train_days) >= 252:
            oos_months.append(m)
    print(f"OOS months: {len(oos_months)} ({oos_months[0]} to {oos_months[-1]})")

    summary = []
    pred_dfs = {}

    for target_name in TARGETS:
        target_col = TARGET_COLS[target_name]
        t0 = time.time()

        pred_rows, shap_rows, log_rows, fallback_notes, wall_time = run_target(
            df, feature_cols, target_name, target_col, all_dates, oos_months
        )

        if not pred_rows:
            print(f"STOP: {target_name} predictions empty. Halting.")
            raise RuntimeError(f"Empty predictions for target {target_name}")

        # Build dataframes
        pred_df = pd.DataFrame(pred_rows)
        shap_df = pd.DataFrame(shap_rows)
        log_df = pd.DataFrame(log_rows)

        pred_dfs[target_name] = pred_df

        # Save outputs
        pred_path = os.path.join(OUTPUT_DIR, f"predictions_lgbm_kr_{target_name}.csv")
        shap_path = os.path.join(OUTPUT_DIR, f"shap_per_fold_kr_{target_name}.parquet")
        log_path = os.path.join(OUTPUT_DIR, f"training_log_lgbm_kr_{target_name}.csv")

        pred_df.to_csv(pred_path, index=False)
        print(f"  Saved predictions: {pred_path}")

        shap_df.to_parquet(shap_path, index=False)
        print(f"  Saved SHAP: {shap_path}")

        log_df.to_csv(log_path, index=False)
        print(f"  Saved training log: {log_path}")

        # Validate
        validate_outputs(pred_df, target_name)

        # Compute summary stats
        mean_ic = log_df["ic_test"].mean()
        median_ic = log_df["ic_test"].median()
        mean_dir_acc = log_df["dir_acc_test"].mean()
        n_folds = len(log_df)
        n_overfit = (log_df["overfit_flag"] == "OVERFIT_FLAG").sum()

        summary.append({
            "target": target_name,
            "n_predictions": len(pred_df),
            "mean_ic_test": round(mean_ic, 4),
            "median_ic_test": round(median_ic, 4),
            "mean_dir_acc": round(mean_dir_acc, 4),
            "n_folds": n_folds,
            "n_overfit_flags": n_overfit,
            "wall_time_min": round(wall_time / 60, 1),
            "fallback_applied": len(fallback_notes) > 0,
        })

        if fallback_notes:
            print(f"  Fallback notes: {fallback_notes}")

    total_wall = time.time() - overall_start
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for s in summary:
        print(f"\nTarget: {s['target'].upper()}")
        print(f"  n_predictions:  {s['n_predictions']}")
        print(f"  mean IC_test:   {s['mean_ic_test']}")
        print(f"  median IC_test: {s['median_ic_test']}")
        print(f"  dir_acc:        {s['mean_dir_acc']}")
        print(f"  n_folds:        {s['n_folds']}")
        print(f"  overfit_flags:  {s['n_overfit_flags']}")
        print(f"  fallback:       {s['fallback_applied']}")
        print(f"  wall time:      {s['wall_time_min']} min")

    print(f"\nTotal wall time: {total_wall/60:.1f} min")

    # Cross-target row count validation
    print(f"\n{'='*60}")
    print("CROSS-TARGET ROW COUNT VALIDATION")
    print(f"{'='*60}")
    counts = {t: len(pred_dfs[t]) for t in TARGETS}
    for t, c in counts.items():
        print(f"  predictions_lgbm_kr_{t}.csv: {c} rows")
    if len(set(counts.values())) == 1:
        print("  PASS: all targets have equal row counts.")
    else:
        print("  WARNING: row counts differ across targets!")

    # Output file paths
    print(f"\nOutput files:")
    for t in TARGETS:
        print(f"  output/predictions_lgbm_kr_{t}.csv")
        print(f"  output/shap_per_fold_kr_{t}.parquet")
        print(f"  output/training_log_lgbm_kr_{t}.csv")


if __name__ == "__main__":
    main()
