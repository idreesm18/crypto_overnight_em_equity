# ABOUTME: Stage P2-11 — diagnostics and horse race for crypto overnight EM equity project.
# ABOUTME: Inputs: prediction CSVs, backtest CSVs, feature parquets. Outputs: 7 diagnostic files in output/.
# Run: source .venv/bin/activate && python3 scripts/stage_p2-11_diagnostics.py

import os
import sys
import glob
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE = Path("/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity_p2")
OUT = BASE / "output"
LOG = BASE / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG / "stage_p2-11_diagnostics.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

BLOCK = 20
N_BOOT = 1000
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def cs_ic(df: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman IC per date."""
    def _ic(g):
        if len(g) < 3:
            return np.nan
        r, _ = stats.spearmanr(g["y_pred"], g["y_actual"])
        return r
    return df.groupby("date").apply(_ic)


def ts_ic(df: pd.DataFrame) -> pd.Series:
    """Time-series Spearman IC (single ticker per date).

    For index universes there is exactly one observation per date, so
    cross-sectional IC is undefined. Instead we use a 30-day rolling
    Spearman correlation between y_pred and y_actual over the sorted
    time series, producing one IC estimate per day (window >= 10 obs).
    This gives a real distribution for block-bootstrapping.
    Log note: rolling window = 30 days; requires >= 10 non-NaN obs per window.
    """
    df = df.sort_values("date").reset_index(drop=True)
    pred = df["y_pred"].values.astype(float)
    actual = df["y_actual"].values.astype(float)
    dates = df["date"].values
    window = 30
    min_obs = 10
    ics = np.full(len(df), np.nan)
    for i in range(len(df)):
        start = max(0, i - window + 1)
        p_w = pred[start : i + 1]
        a_w = actual[start : i + 1]
        valid = ~(np.isnan(p_w) | np.isnan(a_w))
        if valid.sum() >= min_obs:
            r, _ = stats.spearmanr(p_w[valid], a_w[valid])
            ics[i] = r
    return pd.Series(ics, index=dates)


def ic_series(df: pd.DataFrame, is_index: bool = False) -> pd.Series:
    if is_index:
        return ts_ic(df)
    return cs_ic(df)


def block_bootstrap_mean(series: np.ndarray, n_boot: int, block: int, rng) -> np.ndarray:
    """Block-bootstrap the mean of `series`. Returns array of boot means."""
    n = len(series)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        boot_means[i] = series[idx].mean()
    return boot_means


def bootstrap_p_value(observed_mean: float, boot_means: np.ndarray) -> float:
    """Two-sided p-value (null-centred block bootstrap).

    Re-centre the bootstrap distribution around 0 (subtract the bootstrap
    mean), then count how often the null distribution is at least as extreme
    as the observed statistic. This is the standard null-centred bootstrap
    p-value (Efron & Tibshirani 1993, Chapter 16).
    """
    centred = boot_means - boot_means.mean()
    return float(np.mean(np.abs(centred) >= np.abs(observed_mean)))


def sharpe(rets: np.ndarray, ann: float = 252.0) -> float:
    if len(rets) == 0 or rets.std() == 0:
        return np.nan
    return float(rets.mean() / rets.std() * np.sqrt(ann))


def directional_accuracy(df: pd.DataFrame) -> float:
    ok = (np.sign(df["y_pred"]) == np.sign(df["y_actual"])) & (df["y_actual"] != 0)
    return float(ok.mean())


def rolling_ic(ic_series: pd.Series, window: int = 63) -> tuple:
    rolled = ic_series.dropna().rolling(window).mean()
    return float(rolled.mean()), float(rolled.std())


# ---------------------------------------------------------------------------
# 0. Load prediction files
# ---------------------------------------------------------------------------

log.info("Loading prediction files...")

def load_pred(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


# LGBM: 6 universes (hk, kr, control_hk, control_kr, index_hk, index_kr)
# Note: predictions_lgbm_hk_* = main_hk, predictions_lgbm_kr_* = main_kr
LGBM_UNIVERSES = {
    "main_hk":    "predictions_lgbm_hk_{tgt}.csv",
    "main_kr":    "predictions_lgbm_kr_{tgt}.csv",
    "control_hk": "predictions_lgbm_control_hk_{tgt}.csv",
    "control_kr": "predictions_lgbm_control_kr_{tgt}.csv",
    "index_hk":   "predictions_lgbm_index_hk_{tgt}.csv",
    "index_kr":   "predictions_lgbm_index_kr_{tgt}.csv",
}

# TCN: 4 universes (main_hk, main_kr, index_hk, index_kr)
TCN_UNIVERSES = {
    "main_hk":  "predictions_tcn_main_hk_{tgt}.csv",
    "main_kr":  "predictions_tcn_main_kr_{tgt}.csv",
    "index_hk": "predictions_tcn_index_hk_{tgt}.csv",
    "index_kr": "predictions_tcn_index_kr_{tgt}.csv",
}

TARGETS = ["gap", "intraday", "cc"]
INDEX_UNIVS = {"index_hk", "index_kr"}

preds = {}  # (model, universe, target) -> DataFrame

for univ, tmpl in LGBM_UNIVERSES.items():
    for tgt in TARGETS:
        path = OUT / tmpl.format(tgt=tgt)
        if path.exists():
            preds[("lgbm", univ, tgt)] = load_pred(str(path))
        else:
            log.warning(f"Missing: {path}")

for univ, tmpl in TCN_UNIVERSES.items():
    for tgt in TARGETS:
        path = OUT / tmpl.format(tgt=tgt)
        if path.exists():
            preds[("tcn", univ, tgt)] = load_pred(str(path))
        else:
            log.warning(f"Missing: {path}")

log.info(f"Loaded {len(preds)} prediction files.")

# Load backtest summary
bsum = pd.read_csv(OUT / "backtest_summary_pass2.csv")

# Load pass-1 per-day backtests
p1_hk_gap = pd.read_csv(OUT / "backtest_lgbm_hk_gap.csv", parse_dates=["date"])
p1_kr_gap = pd.read_csv(OUT / "backtest_lgbm_kr_gap.csv", parse_dates=["date"])

# Load feature parquets for regime analysis
feat_hk = pd.read_parquet(OUT / "features_track_a_hk.parquet")
feat_kr = pd.read_parquet(OUT / "features_track_a_kr.parquet")
feat_hk["date"] = pd.to_datetime(feat_hk["date"])
feat_kr["date"] = pd.to_datetime(feat_kr["date"])

# ---------------------------------------------------------------------------
# 1. horse_race.csv
# ---------------------------------------------------------------------------

log.info("Computing horse_race.csv...")

horse_rows = []

def compute_row(model, univ, tgt, df):
    is_idx = univ in INDEX_UNIVS
    ics = ic_series(df, is_idx)
    ics_clean = ics.dropna()
    if len(ics_clean) == 0:
        return None
    mean_ic = float(ics_clean.mean())
    dir_acc = directional_accuracy(df)
    # Gross sharpe: use cross-section mean return per day
    daily_gross = df.groupby("date")["y_actual"].mean()
    gross_sh = sharpe(daily_gross.values)
    # Net sharpe 1x1x: assume 50bps round-trip cost, 1x leverage
    # Use backtest_summary_pass2 net_sharpe where available
    net_sh = np.nan
    subset = bsum[
        (bsum["model"] == model) & (bsum["universe"] == univ) & (bsum["target"] == tgt)
    ]
    if len(subset) > 0:
        # prefer tercile_ls gate_off for stock universes, index_threshold gate_off for index
        strat = "index_threshold" if is_idx else "tercile_ls"
        row = subset[(subset["strategy"] == strat) & (subset["gate"] == "gate_off")]
        if len(row) > 0:
            net_sh = float(row["net_sharpe"].iloc[0])
    roll_mean, roll_std = rolling_ic(ics)
    return {
        "universe": univ,
        "target": tgt,
        "model": model.upper(),
        "oos_mean_ic": round(mean_ic, 6),
        "directional_accuracy": round(dir_acc, 4),
        "gross_sharpe": round(gross_sh, 4) if not np.isnan(gross_sh) else np.nan,
        "net_sharpe_1x1x": round(net_sh, 4) if not np.isnan(net_sh) else np.nan,
        "rolling_3mo_ic_mean": round(roll_mean, 6) if not np.isnan(roll_mean) else np.nan,
        "rolling_3mo_ic_std": round(roll_std, 6) if not np.isnan(roll_std) else np.nan,
    }

# LGBM rows
for univ in LGBM_UNIVERSES:
    for tgt in TARGETS:
        key = ("lgbm", univ, tgt)
        if key in preds:
            r = compute_row("lgbm", univ, tgt, preds[key])
            if r:
                horse_rows.append(r)

# TCN rows
for univ in TCN_UNIVERSES:
    for tgt in TARGETS:
        key = ("tcn", univ, tgt)
        if key in preds:
            r = compute_row("tcn", univ, tgt, preds[key])
            if r:
                horse_rows.append(r)

# ENSEMBLE rows (main_hk, main_kr, index_hk, index_kr)
ENSEMBLE_UNIVS = list(TCN_UNIVERSES.keys())
for univ in ENSEMBLE_UNIVS:
    for tgt in TARGETS:
        lgbm_key = ("lgbm", univ, tgt)
        tcn_key = ("tcn", univ, tgt)
        if lgbm_key not in preds or tcn_key not in preds:
            log.warning(f"Missing pred for ensemble {univ} {tgt}")
            continue
        df_l = preds[lgbm_key].copy()
        df_t = preds[tcn_key].copy()
        merged = pd.merge(
            df_l[["date", "ticker", "y_pred", "y_actual"]],
            df_t[["date", "ticker", "y_pred"]],
            on=["date", "ticker"],
            suffixes=("_lgbm", "_tcn"),
        )
        if len(merged) == 0:
            continue
        merged["y_pred"] = 0.5 * merged["y_pred_lgbm"] + 0.5 * merged["y_pred_tcn"]
        r = compute_row("ensemble", univ, tgt, merged[["date", "ticker", "y_pred", "y_actual"]])
        if r:
            horse_rows.append(r)

horse_df = pd.DataFrame(horse_rows)
log.info(f"horse_race rows: {len(horse_df)}")

# Check for duplicates
dupes = horse_df.duplicated(["universe", "target", "model"])
if dupes.any():
    log.error(f"Duplicate keys in horse_race.csv: {horse_df[dupes]}")
    sys.exit("BLOCK: duplicate keys in horse_race.csv")

# Paired block-bootstrap: TCN vs LGBM gap IC diff per main universe
log.info("Running horse_race_bootstrap block-bootstrap...")
boot_rows = []
for univ in ["main_hk", "main_kr"]:
    for tgt in TARGETS:
        lgbm_key = ("lgbm", univ, tgt)
        tcn_key = ("tcn", univ, tgt)
        if lgbm_key not in preds or tcn_key not in preds:
            continue
        df_l = preds[lgbm_key]
        df_t = preds[tcn_key]
        ics_l = cs_ic(df_l)
        ics_t = cs_ic(df_t)
        dates = ics_l.index.intersection(ics_t.index)
        diff = (ics_t.loc[dates] - ics_l.loc[dates]).dropna().values
        if len(diff) < BLOCK * 2:
            boot_rows.append({"universe": univ, "target": tgt, "ic_diff": np.nan, "p_value": np.nan})
            continue
        obs = diff.mean()
        boot_means = block_bootstrap_mean(diff, N_BOOT, BLOCK, RNG)
        p = bootstrap_p_value(obs, boot_means)
        boot_rows.append({"universe": univ, "target": tgt, "ic_diff": round(float(obs), 6), "p_value": round(p, 4)})

boot_df = pd.DataFrame(boot_rows)

horse_df.to_csv(OUT / "horse_race.csv", index=False)
boot_df.to_csv(OUT / "horse_race_bootstrap.csv", index=False)
log.info(f"horse_race.csv written ({len(horse_df)} rows). horse_race_bootstrap.csv ({len(boot_df)} rows).")

# Validate bootstrap p-values
if boot_df["p_value"].isna().all() and len(boot_df) > 0:
    sys.exit("BLOCK: bootstrap p-values are NaN for all main configurations")

# ---------------------------------------------------------------------------
# 2. control_vs_main_comparison.csv
# ---------------------------------------------------------------------------

log.info("Computing control_vs_main_comparison.csv...")

cmp_rows = []
for market, lgbm_main_univ, lgbm_ctrl_univ in [
    ("HK", "main_hk", "control_hk"),
    ("KR", "main_kr", "control_kr"),
]:
    def _mean_ic(univ, tgt):
        k = ("lgbm", univ, tgt)
        if k not in preds:
            return np.nan
        df = preds[k]
        return float(cs_ic(df).dropna().mean())

    main_gap_ic = _mean_ic(lgbm_main_univ, "gap")
    ctrl_gap_ic = _mean_ic(lgbm_ctrl_univ, "gap")
    main_intra_ic = _mean_ic(lgbm_main_univ, "intraday")
    ctrl_intra_ic = _mean_ic(lgbm_ctrl_univ, "intraday")

    main_ratio = main_gap_ic / main_intra_ic if main_intra_ic != 0 else np.nan
    ctrl_ratio = ctrl_gap_ic / ctrl_intra_ic if ctrl_intra_ic != 0 else np.nan

    # Bootstrap on daily IC difference (main gap IC - control gap IC)
    k_main = ("lgbm", lgbm_main_univ, "gap")
    k_ctrl = ("lgbm", lgbm_ctrl_univ, "gap")
    p_val = np.nan
    if k_main in preds and k_ctrl in preds:
        ics_main = cs_ic(preds[k_main])
        ics_ctrl = cs_ic(preds[k_ctrl])
        dates = ics_main.index.intersection(ics_ctrl.index)
        diff = (ics_main.loc[dates] - ics_ctrl.loc[dates]).dropna().values
        if len(diff) >= BLOCK * 2:
            obs = diff.mean()
            bm = block_bootstrap_mean(diff, N_BOOT, BLOCK, RNG)
            p_val = bootstrap_p_value(obs, bm)

    cmp_rows.append({
        "market": market,
        "main_gap_ic": round(main_gap_ic, 6),
        "control_gap_ic": round(ctrl_gap_ic, 6),
        "main_intraday_ic": round(main_intra_ic, 6),
        "control_intraday_ic": round(ctrl_intra_ic, 6),
        "main_ratio": round(float(main_ratio), 4) if not np.isnan(main_ratio) else np.nan,
        "control_ratio": round(float(ctrl_ratio), 4) if not np.isnan(ctrl_ratio) else np.nan,
        "main_minus_control_gap_ic_diff": round(float(main_gap_ic - ctrl_gap_ic), 6),
        "bootstrap_p_value": round(float(p_val), 4) if not np.isnan(p_val) else np.nan,
    })

cmp_df = pd.DataFrame(cmp_rows)
if len(cmp_df) != 2:
    sys.exit(f"BLOCK: control_vs_main_comparison.csv has {len(cmp_df)} rows, expected 2")

cmp_df.to_csv(OUT / "control_vs_main_comparison.csv", index=False)
log.info(f"control_vs_main_comparison.csv written ({len(cmp_df)} rows).")

# ---------------------------------------------------------------------------
# 3. regime_gate_comparison.csv
# ---------------------------------------------------------------------------

log.info("Computing regime_gate_comparison.csv...")

gate_rows = []
for _, r in bsum.iterrows():
    gate_rows.append({
        "universe": r["universe"],
        "model": r["model"],
        "target": r["target"],
        "strategy": r["strategy"],
        "gate": r["gate"],
        "net_sharpe": r["net_sharpe"],
        "ann_return": r["ann_return"],
        "pct_days_invested": r["pct_days_invested"],
    })

gate_df = pd.DataFrame(gate_rows)

# Pivot so we have gate_on and gate_off side by side
gate_on = gate_df[gate_df["gate"] == "gate_on"].rename(
    columns={"net_sharpe": "gate_on_sharpe", "ann_return": "gate_on_ann_return", "pct_days_invested": "days_active_gate_on"}
)[["universe", "model", "target", "strategy", "gate_on_sharpe", "gate_on_ann_return", "days_active_gate_on"]]

gate_off = gate_df[gate_df["gate"] == "gate_off"].rename(
    columns={"net_sharpe": "gate_off_sharpe", "ann_return": "gate_off_ann_return", "pct_days_invested": "days_total"}
)[["universe", "model", "target", "strategy", "gate_off_sharpe", "gate_off_ann_return", "days_total"]]

rgc = pd.merge(gate_on, gate_off, on=["universe", "model", "target", "strategy"], how="outer")
# days_active_gate_on is pct_days_invested of gate_on strategy; days_total = 100 for gate_off
rgc["cost_of_inactivity"] = rgc["gate_off_ann_return"] - rgc["gate_on_ann_return"]
rgc = rgc.drop(columns=["gate_on_ann_return", "gate_off_ann_return"])

rgc.to_csv(OUT / "regime_gate_comparison.csv", index=False)
log.info(f"regime_gate_comparison.csv written ({len(rgc)} rows).")

# ---------------------------------------------------------------------------
# 4. long_short_decomposition.csv
# ---------------------------------------------------------------------------

log.info("Computing long_short_decomposition.csv...")

DECOMP_UNIVS = ["main_hk", "main_kr", "control_hk", "control_kr"]
ls_rows = []

for univ in DECOMP_UNIVS:
    for model in ["lgbm", "tcn"]:
        if model == "tcn" and univ in ["control_hk", "control_kr"]:
            continue  # No TCN for control universes
        for tgt in TARGETS:
            # Long-short gate_off
            ls_file = OUT / f"backtest_{model}_{univ}_{tgt}_tercile_ls_gate_off.csv"
            lo_file = OUT / f"backtest_{model}_{univ}_{tgt}_tercile_lo_gate_off.csv"

            if not ls_file.exists() and not lo_file.exists():
                continue

            row = {"universe": univ, "model": model, "target": tgt}

            # Long-short: only combined available in pass-2 files
            if ls_file.exists():
                df_ls = pd.read_csv(str(ls_file), parse_dates=["date"])
                row["combined_return"] = round(float(df_ls["gross_return"].mean() * 252), 4)
                row["long_sharpe"] = round(sharpe(df_ls["gross_return"].values), 4)
                row["note_ls"] = "combined only; no leg breakdown in pass-2 files"
                row["long_leg_return"] = np.nan
                row["short_leg_return"] = np.nan
                row["short_sharpe"] = np.nan
            else:
                row["combined_return"] = np.nan
                row["long_sharpe"] = np.nan
                row["long_leg_return"] = np.nan
                row["short_leg_return"] = np.nan
                row["short_sharpe"] = np.nan
                row["note_ls"] = "missing"

            # Long-only (for IC by leg)
            if lo_file.exists():
                df_lo = pd.read_csv(str(lo_file), parse_dates=["date"])
                row["long_only_return"] = round(float(df_lo["gross_return"].mean() * 252), 4)
                row["long_only_sharpe"] = round(sharpe(df_lo["gross_return"].values), 4)
            else:
                row["long_only_return"] = np.nan
                row["long_only_sharpe"] = np.nan

            # IC split (long vs short terciles from predictions)
            lgbm_key = ("lgbm" if model == "lgbm" else "tcn", univ, tgt)
            if lgbm_key in preds:
                df_pred = preds[lgbm_key]
                # Long: top tercile, Short: bottom tercile
                def _tercile_ic(df, which="long"):
                    def _g(g):
                        if len(g) < 3:
                            return np.nan
                        t = pd.qcut(g["y_pred"], 3, labels=False, duplicates="drop")
                        if which == "long":
                            sub = g[t == 2]
                        else:
                            sub = g[t == 0]
                        if len(sub) < 2:
                            return np.nan
                        r, _ = stats.spearmanr(sub["y_pred"], sub["y_actual"])
                        return r
                    return df.groupby("date").apply(_g).dropna().mean()
                row["long_ic"] = round(float(_tercile_ic(df_pred, "long")), 6)
                row["short_ic"] = round(float(_tercile_ic(df_pred, "short")), 6)
            else:
                row["long_ic"] = np.nan
                row["short_ic"] = np.nan

            ls_rows.append(row)

ls_df = pd.DataFrame(ls_rows)
cols_order = ["universe", "model", "target", "long_leg_return", "short_leg_return",
              "combined_return", "long_ic", "short_ic", "long_sharpe", "short_sharpe"]
for c in cols_order:
    if c not in ls_df.columns:
        ls_df[c] = np.nan
ls_df = ls_df[cols_order + [c for c in ls_df.columns if c not in cols_order]]

ls_df.to_csv(OUT / "long_short_decomposition.csv", index=False)
log.info(f"long_short_decomposition.csv written ({len(ls_df)} rows). Note: pass-2 per-day files lack leg-level returns; long_leg_return and short_leg_return set to NaN.")

# ---------------------------------------------------------------------------
# 5. return_decomposition_pass2.csv
# ---------------------------------------------------------------------------

log.info("Computing return_decomposition_pass2.csv...")

# Build from backtest_summary_pass2
decomp_rows = []
for _, r in bsum.iterrows():
    decomp_rows.append({
        "universe": r["universe"],
        "model": r["model"],
        "target": r["target"],
        "strategy": r["strategy"],
        "gate": r["gate"],
        "gross_pnl": r.get("gross_pnl", np.nan),
        "spread_cost": r.get("spread_cost", np.nan),
        "impact_cost": r.get("impact_cost", np.nan),
        "borrow_cost": r.get("borrow_cost", np.nan),
        "residual": r.get("residual", np.nan),
        "net_pnl": r.get("gross_pnl", np.nan) - r.get("spread_cost", 0) - r.get("impact_cost", 0) - r.get("borrow_cost", 0),
    })

decomp_df = pd.DataFrame(decomp_rows)

# Check if pass-1 return_decomposition.csv exists and merge
p1_decomp_path = OUT / "return_decomposition.csv"
if p1_decomp_path.exists():
    p1_decomp = pd.read_csv(str(p1_decomp_path))
    log.info(f"Pass-1 return_decomposition.csv found ({len(p1_decomp)} rows) — appending pass-2 rows.")
    # pass-1 has different schema; add source column to distinguish
    p1_decomp["source"] = "pass1"
    decomp_df["source"] = "pass2"
    # Only write pass2 rows to return_decomposition_pass2.csv (separate file)
else:
    log.info("No pass-1 return_decomposition.csv found; creating fresh pass-2 file.")

decomp_df.to_csv(OUT / "return_decomposition_pass2.csv", index=False)
log.info(f"return_decomposition_pass2.csv written ({len(decomp_df)} rows).")

# ---------------------------------------------------------------------------
# 6. regime_analysis_pass2.csv
# ---------------------------------------------------------------------------

log.info("Computing regime_analysis_pass2.csv...")

# Build daily BTC price from 1-minute klines → daily close
btc_klines = pd.read_parquet(
    BASE / "data/binance/spot_klines/BTCUSDT_1m.parquet"
)
btc_klines["date"] = btc_klines["open_time"].dt.tz_localize(None).dt.normalize()
btc_daily = btc_klines.groupby("date")["close"].last().reset_index()
btc_daily.columns = ["date", "btc_close"]
btc_daily["date"] = pd.to_datetime(btc_daily["date"])
btc_daily = btc_daily.sort_values("date")
btc_daily["btc_ret_60d"] = btc_daily["btc_close"].pct_change(60)
btc_daily["btc_realized_vol_30d"] = (
    np.log(btc_daily["btc_close"] / btc_daily["btc_close"].shift(1))
    .rolling(30)
    .std()
    * np.sqrt(252)
)
btc_vol_median = btc_daily["btc_realized_vol_30d"].median()

# Build a date-level regime table from HK features (VIX available there)
feat_daily = (
    feat_hk.groupby("date")["vix_level"].first().reset_index()
)
feat_daily = feat_daily.merge(btc_daily[["date", "btc_ret_60d", "btc_realized_vol_30d"]], on="date", how="left")

REGIME_TYPES = [
    ("VIX_high", lambda row: row["vix_level"] > 25),
    ("VIX_low",  lambda row: row["vix_level"] <= 25),
    ("BTC_up_trend",    lambda row: row["btc_ret_60d"] > 0),
    ("BTC_down_trend",  lambda row: row["btc_ret_60d"] <= 0),
    ("crypto_vol_high", lambda row: row["btc_realized_vol_30d"] > btc_vol_median),
    ("crypto_vol_low",  lambda row: row["btc_realized_vol_30d"] <= btc_vol_median),
]

regime_rows = []

for univ in ["main_hk", "main_kr", "index_hk", "index_kr"]:
    for tgt in TARGETS:
        # Use LGBM predictions (most complete)
        k = ("lgbm", univ, tgt)
        if k not in preds:
            continue
        df = preds[k]
        is_idx = univ in INDEX_UNIVS
        ics = ic_series(df, is_idx)
        ics_df = ics.reset_index()
        ics_df.columns = ["date", "ic"]
        ics_df["date"] = pd.to_datetime(ics_df["date"])
        ics_df = ics_df.merge(feat_daily, on="date", how="left")

        # Also get daily backtest returns for Sharpe
        strat = "index_threshold" if is_idx else "tercile_ls"
        bt_file = OUT / f"backtest_lgbm_{univ}_{tgt}_{strat}_gate_off.csv"
        if bt_file.exists():
            bt_df = pd.read_csv(str(bt_file), parse_dates=["date"])
        else:
            bt_df = None

        for regime_type, flag_fn in REGIME_TYPES:
            mask = ics_df.apply(flag_fn, axis=1)
            sub = ics_df[mask]
            n_days = int(mask.sum())
            mean_ic = float(sub["ic"].mean()) if n_days > 0 else np.nan
            # Gross Sharpe in regime
            gross_sh = np.nan
            if bt_df is not None and n_days > 0:
                bt_sub = bt_df[bt_df["date"].isin(sub["date"])]
                if len(bt_sub) > 1:
                    gross_sh = sharpe(bt_sub["gross_return"].values)
            regime_rows.append({
                "universe": univ,
                "target": tgt,
                "regime_type": regime_type,
                "mean_ic": round(mean_ic, 6) if not np.isnan(mean_ic) else np.nan,
                "gross_sharpe": round(gross_sh, 4) if not np.isnan(gross_sh) else np.nan,
                "n_days": n_days,
            })

regime_df = pd.DataFrame(regime_rows)
regime_df.to_csv(OUT / "regime_analysis_pass2.csv", index=False)
log.info(f"regime_analysis_pass2.csv written ({len(regime_df)} rows).")

# ---------------------------------------------------------------------------
# 7. diagnostics_summary_pass2.txt
# ---------------------------------------------------------------------------

log.info("Writing diagnostics_summary_pass2.txt...")

lines = []
lines.append("=" * 70)
lines.append("DIAGNOSTICS SUMMARY — PASS 2")
lines.append("=" * 70)
lines.append("")

# ---- Acceptance criteria ----
lines.append("ACCEPTANCE CRITERIA EVALUATION")
lines.append("-" * 50)

# (1) Index gap IC > 0.03 & p < 0.05 bootstrap
lines.append("")
lines.append("(1) Index gap IC > 0.03 & p < 0.05 bootstrap")
for univ in ["index_hk", "index_kr"]:
    k = ("lgbm", univ, "gap")
    if k not in preds:
        lines.append(f"  {univ}: MISSING predictions")
        continue
    df = preds[k]
    ics = ts_ic(df)
    mean_ic = float(ics.dropna().mean())
    # Bootstrap p-value on ts IC series
    arr = ics.dropna().values
    if len(arr) >= BLOCK * 2:
        bm = block_bootstrap_mean(arr, N_BOOT, BLOCK, RNG)
        p = bootstrap_p_value(mean_ic, bm)
    else:
        p = np.nan
    verdict = "PASS" if (mean_ic > 0.03 and not np.isnan(p) and p < 0.05) else "FAIL"
    lines.append(f"  {univ}: IC={mean_ic:.4f}, p={p:.4f} => {verdict}")

# (2) KR KOSPI large-cap net Sharpe > 0.5
lines.append("")
lines.append("(2) KR KOSPI large-cap net Sharpe > 0.5")
lines.append("  SKIPPED: KOSPI large-cap was evaluated in P2-4 but no dedicated backtest")
lines.append("  universe was carried through P2-9. The 'main_kr' universe covers HSI")
lines.append("  large-cap equivalents for KR; no separate KOSPI large-cap backtest exists.")

# (3) TCN > LGBM gap IC by > 0.01 & p < 0.05
lines.append("")
lines.append("(3) TCN > LGBM gap IC by > 0.01 & p < 0.05 (paired block-bootstrap)")
for _, br in boot_df[boot_df["target"] == "gap"].iterrows():
    ic_diff = br["ic_diff"]
    p = br["p_value"]
    verdict = "PASS" if (not np.isnan(ic_diff) and ic_diff > 0.01 and not np.isnan(p) and p < 0.05) else "FAIL"
    lines.append(f"  {br['universe']} gap: IC_diff={ic_diff:.4f}, p={p:.4f} => {verdict}")

# (4) Control gap:intraday < 1.5 AND main gap:intraday >= 2.5
lines.append("")
lines.append("(4) Control gap:intraday < 1.5 AND main gap:intraday >= 2.5")
for _, cr in cmp_df.iterrows():
    main_r = cr["main_ratio"]
    ctrl_r = cr["control_ratio"]
    v1 = "PASS" if not np.isnan(ctrl_r) and ctrl_r < 1.5 else "FAIL"
    v2 = "PASS" if not np.isnan(main_r) and main_r >= 2.5 else "FAIL"
    lines.append(f"  {cr['market']}: main_ratio={main_r:.4f} [{v2}], control_ratio={ctrl_r:.4f} [{v1}]")

# Framing rule per brief (decision rule based on gap ratios)
lines.append("")
lines.append("FRAMING RULE (per brief decision rule)")
lines.append("-" * 50)
for _, cr in cmp_df.iterrows():
    m = cr["market"]
    main_r = cr["main_ratio"]
    ctrl_r = cr["control_ratio"]
    gap_diff = cr["main_minus_control_gap_ic_diff"]
    # Rule A: main_ratio >= 2.5 AND ctrl_ratio < 1.5 AND gap_diff > 0  => signal is real
    # Rule B: main_ratio >= 2.5 AND ctrl_ratio >= 1.5                  => unclear framing
    # Rule C: main_ratio < 2.5 AND ctrl_ratio < 1.5                    => signal weak but control cleaner
    # Rule D: main_ratio < 2.5 AND ctrl_ratio >= 1.5                   => both weak, no clear separation
    if not np.isnan(main_r) and not np.isnan(ctrl_r):
        if main_r >= 2.5 and ctrl_r < 1.5 and gap_diff > 0:
            rule = "A — crypto-specific overnight signal is genuine; main clearly separates from control"
        elif main_r >= 2.5 and ctrl_r >= 1.5:
            rule = "B — main shows gap dominance but control also elevated; framing uncertain"
        elif main_r < 2.5 and ctrl_r < 1.5:
            rule = "C — gap:intraday modest for both; signal exists but crypto-specificity unclear"
        else:
            rule = "D — neither main nor control shows clear gap:intraday separation; null framing"
    else:
        rule = "UNKNOWN — missing ratio data"
    lines.append(f"  {m}: gap_ratio_main={main_r:.4f}, gap_ratio_control={ctrl_r:.4f} => Rule {rule}")

lines.append("")

# ---- Top 10 by net_sharpe ----
lines.append("TOP 10 BACKTEST CONFIGS BY NET SHARPE (from backtest_summary_pass2.csv)")
lines.append("-" * 50)
top10 = bsum.nlargest(10, "net_sharpe")[
    ["model", "universe", "target", "strategy", "gate", "net_sharpe", "ann_return"]
]
lines.append(top10.to_string(index=False))
lines.append("")

# ---- Control vs main headline ----
lines.append("CONTROL VS MAIN GAP IC COMPARISON")
lines.append("-" * 50)
lines.append(cmp_df[["market", "main_gap_ic", "control_gap_ic", "main_ratio", "control_ratio",
                       "main_minus_control_gap_ic_diff", "bootstrap_p_value"]].to_string(index=False))
lines.append("")

# ---- Regime-gate headline ----
lines.append("REGIME-GATE COMPARISON (top configs by |cost_of_inactivity|)")
lines.append("-" * 50)
rgc_show = rgc.dropna(subset=["cost_of_inactivity"]).nlargest(10, "cost_of_inactivity")
lines.append(rgc_show[["universe", "model", "target", "strategy", "gate_on_sharpe", "gate_off_sharpe", "cost_of_inactivity"]].to_string(index=False))
lines.append("")

# ---- Notable patterns ----
lines.append("NOTABLE PATTERNS")
lines.append("-" * 50)
# Horse race summary
lgbm_gap_ics = horse_df[(horse_df["target"] == "gap") & (horse_df["model"] == "LGBM")][["universe", "oos_mean_ic"]]
lines.append("LGBM OOS gap IC by universe:")
lines.append(lgbm_gap_ics.to_string(index=False))
lines.append("")
tcn_gap = horse_df[(horse_df["target"] == "gap") & (horse_df["model"] == "TCN")][["universe", "oos_mean_ic"]]
if len(tcn_gap) > 0:
    lines.append("TCN OOS gap IC by universe:")
    lines.append(tcn_gap.to_string(index=False))
    lines.append("")
ens_gap = horse_df[(horse_df["target"] == "gap") & (horse_df["model"] == "ENSEMBLE")][["universe", "oos_mean_ic"]]
if len(ens_gap) > 0:
    lines.append("ENSEMBLE OOS gap IC by universe:")
    lines.append(ens_gap.to_string(index=False))
    lines.append("")

lines.append("=" * 70)
summary_text = "\n".join(lines)

with open(OUT / "diagnostics_summary_pass2.txt", "w") as f:
    f.write(summary_text)

log.info("diagnostics_summary_pass2.txt written.")

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

log.info("Running validation checks...")
outputs = [
    "horse_race.csv",
    "horse_race_bootstrap.csv",
    "control_vs_main_comparison.csv",
    "regime_gate_comparison.csv",
    "long_short_decomposition.csv",
    "return_decomposition_pass2.csv",
    "regime_analysis_pass2.csv",
    "diagnostics_summary_pass2.txt",
]

all_ok = True
for fname in outputs:
    fp = OUT / fname
    if not fp.exists():
        log.error(f"BLOCK: Missing output file: {fname}")
        all_ok = False
    else:
        log.info(f"  OK: {fname} ({fp.stat().st_size} bytes)")

# Check control_vs_main has 2 rows
if len(cmp_df) != 2:
    log.error(f"BLOCK: control_vs_main_comparison.csv has {len(cmp_df)} rows (expected 2)")
    all_ok = False

# Check bootstrap p-values in [0,1]
for pv in boot_df["p_value"].dropna():
    if not (0.0 <= pv <= 1.0):
        log.error(f"BLOCK: Bootstrap p-value out of range: {pv}")
        all_ok = False

if all_ok:
    log.info("All validation checks passed.")
else:
    sys.exit("BLOCK: Validation failed — see log.")

# ---------------------------------------------------------------------------
# Final summary log
# ---------------------------------------------------------------------------

log.info("")
log.info("=" * 60)
log.info("FINAL SUMMARY")
log.info("=" * 60)

output_counts = {
    "horse_race.csv": len(horse_df),
    "horse_race_bootstrap.csv": len(boot_df),
    "control_vs_main_comparison.csv": len(cmp_df),
    "regime_gate_comparison.csv": len(rgc),
    "long_short_decomposition.csv": len(ls_df),
    "return_decomposition_pass2.csv": len(decomp_df),
    "regime_analysis_pass2.csv": len(regime_df),
    "diagnostics_summary_pass2.txt": 1,
}

log.info("7 output files (+ horse_race_bootstrap.csv as side table):")
for fname, count in output_counts.items():
    log.info(f"  {fname}: {count} rows/entries")

log.info("")
log.info("ACCEPTANCE CRITERIA:")
# Criterion 1
for univ in ["index_hk", "index_kr"]:
    k = ("lgbm", univ, "gap")
    if k in preds:
        df = preds[k]
        ics = ts_ic(df)
        mic = float(ics.dropna().mean())
        hr_row = horse_df[(horse_df["universe"] == univ) & (horse_df["model"] == "LGBM") & (horse_df["target"] == "gap")]
        log.info(f"  #1 Index gap IC [{univ}]: IC={mic:.4f}")

# Criterion 3 (TCN vs LGBM)
log.info("  #3 TCN vs LGBM gap IC (block-bootstrap):")
for _, br in boot_df[boot_df["target"] == "gap"].iterrows():
    log.info(f"    {br['universe']}: IC_diff={br['ic_diff']:.4f}, p={br['p_value']:.4f}")

# Criterion 4 (ratios)
log.info("  #4 gap:intraday ratios:")
for _, cr in cmp_df.iterrows():
    log.info(f"    {cr['market']}: main={cr['main_ratio']:.4f}, control={cr['control_ratio']:.4f}")

# Framing rule
log.info("  Framing rules:")
for _, cr in cmp_df.iterrows():
    main_r = cr["main_ratio"]
    ctrl_r = cr["control_ratio"]
    gap_diff = cr["main_minus_control_gap_ic_diff"]
    if not np.isnan(main_r) and not np.isnan(ctrl_r):
        if main_r >= 2.5 and ctrl_r < 1.5 and gap_diff > 0:
            rule = "A"
        elif main_r >= 2.5 and ctrl_r >= 1.5:
            rule = "B"
        elif main_r < 2.5 and ctrl_r < 1.5:
            rule = "C"
        else:
            rule = "D"
    else:
        rule = "?"
    log.info(f"    {cr['market']}: Rule {rule}")

log.info("Stage P2-11 complete.")
