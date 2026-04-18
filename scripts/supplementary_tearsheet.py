# ABOUTME: Supplementary tearsheet — year-by-year index Sharpe and extended cost sensitivity.
# ABOUTME: Inputs: output/backtest_lgbm_index_*.csv, output/predictions_lgbm_index_*.csv, output/cost_sensitivity_pass2.csv
#           Outputs: output/index_yearly_tearsheet.csv, output/cost_sensitivity_extended.csv, logs/supplementary_tearsheet.log
#           Run: source .venv/bin/activate && python3 scripts/supplementary_tearsheet.py

import sys
import os
import time
import datetime
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT = "/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity_p2"
OUTPUT  = os.path.join(PROJECT, "output")
LOGS    = os.path.join(PROJECT, "logs")

# ── import stage_p2-9 functions via sys.path insert ──────────────────────────
SCRIPTS = os.path.join(PROJECT, "scripts")
sys.path.insert(0, SCRIPTS)

# The stage_p2-9 script uses a hyphen in the filename, so we must use importlib
import importlib.util
spec = importlib.util.spec_from_file_location(
    "stage_p2_9_backtest",
    os.path.join(SCRIPTS, "stage_p2-9_backtest.py")
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

run_index_backtest   = _mod.run_index_backtest
build_regime_series  = _mod.build_regime_series
metrics              = _mod.metrics
pred_file            = _mod.pred_file


ANN = 252.0

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def yearly_max_drawdown(net_returns: pd.Series) -> float:
    """Max drawdown computed on intra-year cumulative sum (not global cumulative)."""
    cum = net_returns.cumsum()
    roll_max = cum.cummax()
    dd = cum - roll_max
    return dd.min()


def compute_yearly_tearsheet(
    bt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    config_label: str,
) -> pd.DataFrame:
    """
    Given a per-day backtest df and a predictions df, compute yearly metrics.
    """
    bt_df = bt_df.copy()
    bt_df["date"] = pd.to_datetime(bt_df["date"])
    bt_df["year"] = bt_df["date"].dt.year

    pred_df = pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    pred_df["year"] = pred_df["date"].dt.year

    rows = []
    for year, grp in bt_df.groupby("year"):
        n_days = len(grp)
        if n_days == 0:
            continue

        # pct_invested: fraction of rows where strategy is active
        # Use net_return != 0 as proxy (no position column in saved backtest)
        pct_invested = (grp["net_return"] != 0).mean()

        # Annual Sharpe from daily net_return
        mu  = grp["net_return"].mean()
        std = grp["net_return"].std()
        if std > 0:
            annual_sharpe = mu / std * np.sqrt(n_days)
        else:
            annual_sharpe = np.nan

        # Annualised return
        ann_return = grp["net_return"].sum() * 252.0 / n_days

        # Max drawdown (intra-year)
        mdd = yearly_max_drawdown(grp["net_return"])

        # IC: Spearman correlation between y_pred and y_actual in that year
        yr_pred = pred_df[pred_df["year"] == year]
        if len(yr_pred) >= 5:
            corr, _ = stats.spearmanr(yr_pred["y_pred"], yr_pred["y_actual"])
            mean_ic = float(corr) if not np.isnan(corr) else np.nan
        else:
            mean_ic = np.nan

        rows.append({
            "config":        config_label,
            "year":          year,
            "n_days":        n_days,
            "pct_invested":  round(pct_invested, 4),
            "annual_sharpe": round(annual_sharpe, 4) if not np.isnan(annual_sharpe) else np.nan,
            "ann_return":    round(ann_return, 4),
            "max_drawdown":  round(mdd, 4),
            "mean_ic":       round(mean_ic, 4) if not np.isnan(mean_ic) else np.nan,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# DELIVERABLE 1 — Year-by-year index Sharpe tearsheet
# ─────────────────────────────────────────────────────────────────────────────

def deliverable1(log_lines: list) -> pd.DataFrame:
    log_lines.append("\n=== DELIVERABLE 1: Year-by-year index Sharpe tearsheet ===\n")

    CONFIGS = [
        ("lgbm_index_kr_gap_index_threshold_gate_off", "predictions_lgbm_index_kr_gap.csv"),
        ("lgbm_index_kr_gap_index_threshold_gate_on",  "predictions_lgbm_index_kr_gap.csv"),
        ("lgbm_index_hk_gap_index_threshold_gate_off", "predictions_lgbm_index_hk_gap.csv"),
        ("lgbm_index_hk_gap_index_threshold_gate_on",  "predictions_lgbm_index_hk_gap.csv"),
    ]

    all_rows = []

    for cfg, pred_fname in CONFIGS:
        bt_path   = os.path.join(OUTPUT, f"backtest_{cfg}.csv")
        pred_path = os.path.join(OUTPUT, pred_fname)

        # Confirm files exist and can be read — BLOCK condition
        try:
            bt_df   = pd.read_csv(bt_path)
            pred_df = pd.read_csv(pred_path)
        except Exception as e:
            print(f"BLOCK: Cannot read {bt_path} or {pred_path}: {e}", flush=True)
            sys.exit(1)

        log_lines.append(f"  {cfg}: backtest shape={bt_df.shape}, pred shape={pred_df.shape}")

        # Validate columns
        required_bt_cols = ["date", "net_return"]
        for c in required_bt_cols:
            if c not in bt_df.columns:
                print(f"BLOCK: Missing column '{c}' in {bt_path}", flush=True)
                sys.exit(1)

        tsheet = compute_yearly_tearsheet(bt_df, pred_df, cfg)
        all_rows.append(tsheet)

    tearsheet_df = pd.concat(all_rows, ignore_index=True)

    # Validation
    assert (tearsheet_df["n_days"] > 0).all(), "BLOCK: Some years have n_days == 0"

    years = tearsheet_df["year"].unique()
    if 2020 in years:
        yr2020 = tearsheet_df[tearsheet_df["year"] == 2020]["n_days"].min()
        assert yr2020 < 252, f"2020 should be partial year (n_days={yr2020})"
    if 2026 in years:
        yr2026 = tearsheet_df[tearsheet_df["year"] == 2026]["n_days"].min()
        assert yr2026 < 252, f"2026 should be partial year (n_days={yr2026})"

    # Weighted-average Sharpe sanity vs aggregate
    summary_df = pd.read_csv(os.path.join(OUTPUT, "backtest_summary_pass2.csv"))

    for cfg, _ in CONFIGS:
        # parse config label to match summary
        parts = cfg.split("_")
        # format: lgbm_index_kr_gap_index_threshold_gate_off
        # model=lgbm, universe=index_kr, target=gap, strategy=index_threshold, gate=gate_off
        model    = parts[0]
        universe = parts[1] + "_" + parts[2]
        target   = parts[3]
        strategy = parts[4] + "_" + parts[5]
        gate     = parts[6] + "_" + parts[7]

        agg_row = summary_df[
            (summary_df["model"]    == model)    &
            (summary_df["universe"] == universe) &
            (summary_df["target"]   == target)   &
            (summary_df["strategy"] == strategy) &
            (summary_df["gate"]     == gate)
        ]

        sub = tearsheet_df[tearsheet_df["config"] == cfg]
        weighted_sharpe = np.average(
            sub["annual_sharpe"].fillna(0),
            weights=sub["n_days"]
        )
        agg_sharpe = float(agg_row["net_sharpe"].values[0]) if len(agg_row) > 0 else np.nan
        same_sign = (weighted_sharpe * agg_sharpe) >= 0 if not np.isnan(agg_sharpe) else True
        log_lines.append(
            f"  {cfg}: weighted_yr_sharpe={weighted_sharpe:.3f}  "
            f"agg_net_sharpe={agg_sharpe:.3f}  same_sign={same_sign}"
        )

    # Save
    out_path = os.path.join(OUTPUT, "index_yearly_tearsheet.csv")
    tearsheet_df.to_csv(out_path, index=False)
    log_lines.append(f"\n  Saved {len(tearsheet_df)} rows -> {out_path}")

    # Print to stdout
    print("\n=== DELIVERABLE 1: Year-by-year index Sharpe tearsheet ===")
    print(tearsheet_df.to_string(index=False))

    return tearsheet_df


# ─────────────────────────────────────────────────────────────────────────────
# DELIVERABLE 2 — Extended cost sensitivity 5x / 7x / 10x / 14x
# ─────────────────────────────────────────────────────────────────────────────

def deliverable2(gate_series: pd.Series, log_lines: list) -> pd.DataFrame:
    log_lines.append("\n=== DELIVERABLE 2: Extended cost sensitivity ===\n")

    EXTENDED_MULTS = [5.0, 7.0, 10.0, 14.0]
    INDEX_CONFIGS = [
        ("lgbm", "index_kr", "gap"),
        ("lgbm", "index_hk", "gap"),
    ]
    GATE_VARIANTS = [
        ("gate_off", None),
        ("gate_on",  gate_series),
    ]

    rows = []
    for model, universe, target in INDEX_CONFIGS:
        for gate_label, gate_arg in GATE_VARIANTS:
            for sm in EXTENDED_MULTS:
                df = run_index_backtest(
                    universe, target, model,
                    spread_mult=sm, borrow_mult=1.0,
                    gate_series=gate_arg,
                )
                m = metrics(df)

                # BLOCK: NaN at 14x
                if sm == 14.0 and np.isnan(m["net_sharpe"]):
                    print(f"BLOCK: net_sharpe is NaN at 14x for {model}/{universe}/{target}/{gate_label}", flush=True)
                    sys.exit(1)

                rows.append({
                    "model":        model,
                    "universe":     universe,
                    "target":       target,
                    "strategy":     "index_threshold",
                    "gate":         gate_label,
                    "spread_mult":  sm,
                    "borrow_mult":  1.0,
                    "net_sharpe":   m["net_sharpe"],
                    "ann_return":   m["ann_return"],
                    "max_drawdown": m["max_drawdown"],
                    "spread_cost":  m["spread_cost"],
                    "borrow_cost":  m["borrow_cost"],
                })
                log_lines.append(
                    f"  {model}/{universe}/{target}/{gate_label} "
                    f"sm={sm}x  net_SR={m['net_sharpe']:.4f}"
                )

    ext_df = pd.DataFrame(rows)

    # Spot-check: run one config at 1x and compare to cost_sensitivity_pass2
    chk_df = run_index_backtest("index_kr", "gap", "lgbm",
                                 spread_mult=1.0, borrow_mult=1.0,
                                 gate_series=gate_series)
    chk_m = metrics(chk_df)
    ref_df = pd.read_csv(os.path.join(OUTPUT, "cost_sensitivity_pass2.csv"))
    ref_row = ref_df[
        (ref_df["model"]       == "lgbm")       &
        (ref_df["universe"]    == "index_kr")   &
        (ref_df["target"]      == "gap")         &
        (ref_df["spread_mult"] == 1.0)
    ]
    ref_sharpe = float(ref_row["net_sharpe"].values[0]) if len(ref_row) > 0 else np.nan
    diff = abs(chk_m["net_sharpe"] - ref_sharpe)
    print(f"\nSpot-check (index_kr/gap/lgbm @ 1x gate_on): "
          f"new_script={chk_m['net_sharpe']:.6f}  "
          f"cost_sensitivity_pass2={ref_sharpe:.6f}  "
          f"diff={diff:.2e}  {'OK' if diff < 1e-4 else 'WARN'}")
    log_lines.append(
        f"\n  Spot-check 1x: new={chk_m['net_sharpe']:.6f}  ref={ref_sharpe:.6f}  diff={diff:.2e}"
    )

    # Save extended sensitivity (do NOT overwrite cost_sensitivity_pass2.csv)
    out_path = os.path.join(OUTPUT, "cost_sensitivity_extended.csv")
    ext_df.to_csv(out_path, index=False)
    log_lines.append(f"  Saved {len(ext_df)} rows -> {out_path}")

    # ── Comparison table: 1x / 2x from pass2, 5x/7x/10x/14x from new run ────
    pass2_df = pd.read_csv(os.path.join(OUTPUT, "cost_sensitivity_pass2.csv"))

    print("\n=== DELIVERABLE 2: Net Sharpe comparison table ===")
    print(f"{'Config':<45} {'1x':>8} {'2x':>8} {'5x':>8} {'7x':>8} {'10x':>8} {'14x':>8}")
    print("-" * 101)

    log_lines.append("\n  Comparison table (net_sharpe by spread_mult):")
    log_lines.append(f"  {'Config':<45} {'1x':>8} {'2x':>8} {'5x':>8} {'7x':>8} {'10x':>8} {'14x':>8}")

    for model, universe, target in INDEX_CONFIGS:
        for gate_label, _ in GATE_VARIANTS:
            label = f"{model}/{universe}/{target}/{gate_label}"

            # 1x and 2x from pass2 (gate_on only in pass2)
            def get_pass2(sm):
                r = pass2_df[
                    (pass2_df["model"]       == model)    &
                    (pass2_df["universe"]    == universe) &
                    (pass2_df["target"]      == target)   &
                    (pass2_df["spread_mult"] == sm)
                ]
                return float(r["net_sharpe"].values[0]) if len(r) > 0 else np.nan

            s1  = get_pass2(1.0)
            s2  = get_pass2(2.0)

            def get_ext(sm):
                r = ext_df[
                    (ext_df["model"]       == model)    &
                    (ext_df["universe"]    == universe) &
                    (ext_df["target"]      == target)   &
                    (ext_df["gate"]        == gate_label) &
                    (ext_df["spread_mult"] == sm)
                ]
                return float(r["net_sharpe"].values[0]) if len(r) > 0 else np.nan

            s5  = get_ext(5.0)
            s7  = get_ext(7.0)
            s10 = get_ext(10.0)
            s14 = get_ext(14.0)

            line = (f"{label:<45} {s1:>8.3f} {s2:>8.3f} "
                    f"{s5:>8.3f} {s7:>8.3f} {s10:>8.3f} {s14:>8.3f}")
            print(line)
            log_lines.append("  " + line)

            # Monotonicity check (1x < 2x < 5x < ... is descending)
            sharpes = [s1, s2, s5, s7, s10, s14]
            mults   = [1, 2, 5, 7, 10, 14]
            for i in range(len(sharpes) - 1):
                if sharpes[i+1] > sharpes[i] + 0.05:
                    print(f"  WARN: non-monotonic at {mults[i]}x -> {mults[i+1]}x: "
                          f"{sharpes[i]:.3f} -> {sharpes[i+1]:.3f} for {label}")

    print("")

    return ext_df


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def final_summary(tearsheet_df: pd.DataFrame, ext_df: pd.DataFrame, log_lines: list):
    log_lines.append("\n=== FINAL SUMMARY ===\n")
    print("\n=== FINAL SUMMARY ===")

    # ── Deliverable 1 ──────────────────────────────────────────────────────────
    kr_off = tearsheet_df[
        tearsheet_df["config"] == "lgbm_index_kr_gap_index_threshold_gate_off"
    ].copy()

    # Lowest Sharpe year
    lowest_row  = kr_off.loc[kr_off["annual_sharpe"].idxmin()]
    highest_row = kr_off.loc[kr_off["annual_sharpe"].idxmax()]

    msg1 = (f"Lowest annual_sharpe for index_kr gap gate_off: "
            f"year={int(lowest_row['year'])}  sharpe={lowest_row['annual_sharpe']:.4f}")
    msg2 = (f"Highest annual_sharpe for index_kr gap gate_off: "
            f"year={int(highest_row['year'])}  sharpe={highest_row['annual_sharpe']:.4f}")

    # 2023 attenuation
    row_2023 = kr_off[kr_off["year"] == 2023]
    if len(row_2023) > 0:
        sharpe_2023 = float(row_2023["annual_sharpe"].values[0])
        ic_2023     = float(row_2023["mean_ic"].values[0]) if not np.isnan(row_2023["mean_ic"].values[0]) else np.nan
        attenuation = (sharpe_2023 < 0.5) or (not np.isnan(ic_2023) and ic_2023 < 0.05)
        msg3 = (f"2023 attenuation at index level: "
                f"annual_sharpe={sharpe_2023:.4f}  mean_ic={ic_2023:.4f}  "
                f"ATTENUATED={'YES' if attenuation else 'NO'}")
    else:
        msg3 = "2023 not present in tearsheet (check date range)"

    for m in [msg1, msg2, msg3]:
        print(m)
        log_lines.append("  " + m)

    # ── Deliverable 2 ──────────────────────────────────────────────────────────
    def get_ext_sharpe(gate_label: str, sm: float) -> float:
        r = ext_df[
            (ext_df["universe"]    == "index_kr") &
            (ext_df["target"]      == "gap")      &
            (ext_df["gate"]        == gate_label) &
            (ext_df["spread_mult"] == sm)
        ]
        return float(r["net_sharpe"].values[0]) if len(r) > 0 else np.nan

    s7_off  = get_ext_sharpe("gate_off", 7.0)
    s10_off = get_ext_sharpe("gate_off", 10.0)

    msg4 = (f"index_kr gap gate_off net_sharpe @ 7x spread: {s7_off:.4f}")
    msg5 = (f"index_kr gap gate_off net_sharpe @ 10x spread: {s10_off:.4f}")

    # WRITEUP verdict: "net Sharpe 1.5-2.0 at realistic costs"
    # 7x is aggressive/realistic; 10x is extreme
    if s7_off >= 1.5:
        verdict = "CONFIRMED"
    elif s7_off >= 1.0:
        verdict = "TOO OPTIMISTIC"
    else:
        verdict = "TOO OPTIMISTIC"

    msg6 = (f"WRITEUP claim 'net Sharpe 1.5-2.0 at realistic costs': {verdict}  "
            f"(7x={s7_off:.3f}, 10x={s10_off:.3f})")

    for m in [msg4, msg5, msg6]:
        print(m)
        log_lines.append("  " + m)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log_lines = []

    date_header = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines.append(f"=== supplementary_tearsheet.py  {date_header} ===")

    # ── Build regime gate once (shared by both deliverables) ─────────────────
    print("Building regime gate series...", flush=True)
    try:
        gate_series = build_regime_series()
    except Exception as e:
        print(f"BLOCK: build_regime_series() failed: {e}", flush=True)
        sys.exit(1)
    print(f"  Gate built: {gate_series.index.min().date()} to {gate_series.index.max().date()}", flush=True)
    log_lines.append(f"\nRegime gate: {gate_series.index.min().date()} to {gate_series.index.max().date()}")

    # ── DELIVERABLE 1 ─────────────────────────────────────────────────────────
    tearsheet_df = deliverable1(log_lines)

    # ── DELIVERABLE 2 ─────────────────────────────────────────────────────────
    ext_df = deliverable2(gate_series, log_lines)

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    final_summary(tearsheet_df, ext_df, log_lines)

    # ── Print kr_gap_gate_off year table ─────────────────────────────────────
    print("\n--- index_kr gap gate_off year-by-year ---")
    kr_off = tearsheet_df[
        tearsheet_df["config"] == "lgbm_index_kr_gap_index_threshold_gate_off"
    ][["year", "n_days", "pct_invested", "annual_sharpe", "ann_return", "max_drawdown", "mean_ic"]]
    print(kr_off.to_string(index=False))

    # ── Runtime check ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    elapsed_msg = f"\nTotal runtime: {elapsed/60:.1f} min"
    print(elapsed_msg)
    log_lines.append(elapsed_msg)
    if elapsed > 1800:
        print("BLOCK: Runtime exceeded 30 minutes", flush=True)
        sys.exit(1)

    # ── Write log ─────────────────────────────────────────────────────────────
    os.makedirs(LOGS, exist_ok=True)
    log_path = os.path.join(LOGS, "supplementary_tearsheet.log")
    with open(log_path, "a") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\nLog appended to {log_path}")
    print("=== Done. Both outputs written. ===")


if __name__ == "__main__":
    main()
