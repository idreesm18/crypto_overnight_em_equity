# ABOUTME: Stage 5 — portfolio construction and backtest for the crypto overnight EM equity signal.
# ABOUTME: Inputs: output/predictions_lgbm_{hk,kr}_{gap,intraday,cc}.csv, output/features_track_a_{hk,kr}.parquet,
#          output/universe_log.csv. Run: source .venv/bin/activate && python3 scripts/stage5_backtest.py.

import pandas as pd
import numpy as np
import warnings
import sys
import os

warnings.filterwarnings("ignore")

PROJECT = "/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity"
OUTPUT = os.path.join(PROJECT, "output")
LOGS = os.path.join(PROJECT, "logs")

# ─── cost parameters ────────────────────────────────────────────────────────
HALF_SPREAD = {"HK": 15e-4, "KR": 10e-4}   # 15 / 10 bps per side in decimal
IMPACT_CONST = 0.1
TRADE_SIZE_USD = 100_000.0
COST_MULTS = [0.5, 1.0, 1.5, 2.0]
MIN_UNIVERSE = 9          # go flat if fewer stocks
BLOCK_SIZE = 20           # trading days for block bootstrap
N_BOOT = 1000
ANN = 252.0

# ─── helpers ────────────────────────────────────────────────────────────────

def load_cost_data(market: str) -> pd.DataFrame:
    """Return daily (date, ticker, adv_usd, stock_rv_20d) panel.

    ADV comes from universe_log (monthly, forward-filled).
    stock_rv_20d comes from features parquet (daily).
    """
    ul = pd.read_csv(os.path.join(OUTPUT, "universe_log.csv"), parse_dates=["date"])
    ul = ul[ul["market"] == market.upper()][["date", "ticker", "adv_usd"]].copy()
    ul = ul.sort_values(["ticker", "date"])

    feat = pd.read_parquet(os.path.join(OUTPUT, f"features_track_a_{market.lower()}.parquet"),
                           columns=["date", "ticker", "stock_rv_20d"])
    feat["date"] = pd.to_datetime(feat["date"])
    feat = feat.sort_values(["ticker", "date"])

    # Forward-fill monthly ADV to daily by merging on ticker then reindex
    # Build full date grid from features
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


def compute_impact(adv_usd: float, daily_vol: float) -> float:
    """Market impact per side (decimal return units)."""
    if adv_usd <= 0 or np.isnan(adv_usd) or np.isnan(daily_vol):
        return 0.0
    # daily_vol is annualized decimal → convert to daily decimal
    daily_vol_dec = daily_vol / np.sqrt(ANN)
    return IMPACT_CONST * np.sqrt(TRADE_SIZE_USD / adv_usd) * daily_vol_dec


def compute_portfolio_cost(leg_tickers, cost_data, market: str,
                           date, cost_mult: float = 1.0) -> float:
    """Cost per portfolio (decimal) for one leg (entry + exit = 2 sides each)."""
    half_spread = HALF_SPREAD[market.upper()] * cost_mult
    total_cost = 0.0
    n = len(leg_tickers)
    if n == 0:
        return 0.0
    for tkr in leg_tickers:
        key = (date, tkr)
        if key in cost_data.index:
            row = cost_data.loc[key]
            adv = row["adv_usd"]
            rv = row["stock_rv_20d"]
        else:
            adv, rv = np.nan, np.nan
        impact = compute_impact(adv, rv) * cost_mult
        # entry + exit, equally weighted (weight = 1/n)
        cost_per_stock = (half_spread + impact) * 2  # 2 sides
        total_cost += cost_per_stock / n
    return total_cost


def run_backtest(market: str, target: str,
                 cost_data: pd.DataFrame,
                 cost_mult: float = 1.0) -> pd.DataFrame:
    """Run one daily rebalance backtest. Returns daily DataFrame."""
    fname = os.path.join(OUTPUT, f"predictions_lgbm_{market.lower()}_{target}.csv")
    pred = pd.read_csv(fname, parse_dates=["date"])
    pred = pred.sort_values(["date", "ticker"]).reset_index(drop=True)

    records = []
    prev_long = set()
    prev_short = set()

    for date, day_df in pred.groupby("date"):
        n = len(day_df)
        if n < MIN_UNIVERSE:
            records.append({
                "date": date, "long_leg_return": 0.0, "short_leg_return": 0.0,
                "gross_return": 0.0, "turnover": 0.0, "cost_bps": 0.0,
                "net_return": 0.0, "flat": True
            })
            prev_long, prev_short = set(), set()
            continue

        day_df = day_df.sort_values("y_pred").reset_index(drop=True)
        tercile = n // 3
        short_set = set(day_df.iloc[:tercile]["ticker"])
        long_set = set(day_df.iloc[n - tercile:]["ticker"])

        long_ret = day_df[day_df["ticker"].isin(long_set)]["y_actual"].mean()
        short_ret = day_df[day_df["ticker"].isin(short_set)]["y_actual"].mean()
        gross_ret = long_ret - short_ret

        # Turnover: fraction of long + short legs that changed
        long_overlap = len(long_set & prev_long) / max(len(long_set), 1)
        short_overlap = len(short_set & prev_short) / max(len(short_set), 1)
        turnover = 1.0 - (long_overlap + short_overlap) / 2.0

        # Cost: only for stocks that changed (entered/exited)
        long_entries = long_set - prev_long
        long_exits = prev_long - long_set
        short_entries = short_set - prev_short
        short_exits = prev_short - short_set

        # Entry + exit cost for changed names
        changed = long_entries | long_exits | short_entries | short_exits
        cost_dec = 0.0
        half_spread = HALF_SPREAD[market.upper()] * cost_mult
        total_legs = max(len(long_set), 1) + max(len(short_set), 1)
        for tkr in changed:
            key = (date, tkr)
            if key in cost_data.index:
                row = cost_data.loc[key]
                adv = row["adv_usd"]
                rv = row["stock_rv_20d"]
            else:
                adv, rv = np.nan, np.nan
            impact = compute_impact(adv, rv) * cost_mult
            # weight in portfolio = 1/tercile; cost is (spread+impact)*2 sides per round trip
            cost_dec += (half_spread + impact) * 2 / (tercile * 2)

        assert cost_dec >= 0, f"Negative cost at {date}"

        net_ret = gross_ret - cost_dec
        cost_bps = cost_dec * 1e4

        records.append({
            "date": date,
            "long_leg_return": long_ret,
            "short_leg_return": short_ret,
            "gross_return": gross_ret,
            "turnover": turnover,
            "cost_bps": cost_bps,
            "net_return": net_ret,
            "flat": False
        })
        prev_long, prev_short = long_set, short_set

    df = pd.DataFrame(records)
    df["cumulative_net"] = df["net_return"].cumsum()
    return df


def metrics(df: pd.DataFrame) -> dict:
    """Compute performance metrics from daily backtest DataFrame."""
    ret_g = df["gross_return"]
    ret_n = df["net_return"]
    if "flat" in df.columns:
        pct_inv = (~df["flat"]).sum() / len(df) * 100
    else:
        pct_inv = (df["gross_return"] != 0).sum() / len(df) * 100

    ann_ret_g = ret_g.sum() * ANN / len(df)  # scale by actual days
    ann_ret_n = ret_n.sum() * ANN / len(df)
    ann_vol = ret_n.std() * np.sqrt(ANN)
    gross_sharpe = ann_ret_g / (ret_g.std() * np.sqrt(ANN)) if ret_g.std() > 0 else np.nan
    net_sharpe = ann_ret_n / ann_vol if ann_vol > 0 else np.nan

    cum = df["cumulative_net"]
    roll_max = cum.cummax()
    dd = cum - roll_max
    max_dd = dd.min()
    calmar = ann_ret_n / abs(max_dd) if max_dd < 0 else np.nan

    avg_to = df["turnover"].mean()
    avg_cost_bps = df["cost_bps"].mean()

    return {
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "ann_return_gross": ann_ret_g,
        "ann_return_net": ann_ret_n,
        "ann_vol": ann_vol,
        "max_drawdown_net": max_dd,
        "calmar_net": calmar,
        "avg_turnover": avg_to,
        "pct_days_invested": pct_inv,
        "avg_cost_bps": avg_cost_bps,
    }


def breakeven_multiplier(market: str, target: str,
                         cost_data: pd.DataFrame) -> float:
    """Grid search for cost multiplier that drives net Sharpe to zero."""
    lo, hi = 0.0, 20.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        df_mid = run_backtest(market, target, cost_data, cost_mult=mid)
        m = metrics(df_mid)
        ns = m["net_sharpe"]
        if np.isnan(ns) or ns > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.01:
            break
    return (lo + hi) / 2.0


def block_bootstrap_ic_pvalue(market: str, target: str,
                              block_size: int = BLOCK_SIZE,
                              n_boot: int = N_BOOT) -> float:
    """Block bootstrap p-value for OOS IC (Spearman corr(y_pred, y_actual))."""
    fname = os.path.join(OUTPUT, f"predictions_lgbm_{market.lower()}_{target}.csv")
    pred = pd.read_csv(fname, parse_dates=["date"])
    pred = pred.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Observed IC: per-date rank correlation, then mean
    dates = pred["date"].unique()
    per_date_ic = []
    for d in dates:
        day = pred[pred["date"] == d]
        if len(day) < 3:
            continue
        ic = day["y_pred"].rank().corr(day["y_actual"].rank())
        per_date_ic.append(ic)
    per_date_ic = np.array(per_date_ic)
    obs_ic = np.nanmean(per_date_ic)

    # Block bootstrap over dates
    n_dates = len(per_date_ic)
    n_blocks = int(np.ceil(n_dates / block_size))
    rng = np.random.default_rng(42)
    boot_ics = []
    for _ in range(n_boot):
        starts = rng.integers(0, n_dates, size=n_blocks)
        idx = []
        for s in starts:
            idx.extend(range(s, min(s + block_size, n_dates)))
        idx = idx[:n_dates]
        boot_ics.append(np.nanmean(per_date_ic[idx]))

    boot_ics = np.array(boot_ics)
    # Two-sided p-value: fraction of bootstrap ICs >= observed under H0 IC=0
    # Under H0, center bootstrap distribution at 0
    boot_centered = boot_ics - np.mean(boot_ics)
    p_val = np.mean(np.abs(boot_centered) >= np.abs(obs_ic))
    return float(p_val), float(obs_ic)


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=== Stage 5: Portfolio Construction & Backtest ===\n")

    markets = ["HK", "KR"]
    targets = ["gap", "intraday", "cc"]

    # Load cost data once per market
    cost_data = {}
    for mkt in markets:
        log(f"Loading cost data for {mkt}...")
        cost_data[mkt] = load_cost_data(mkt)

    # ── Run 6 backtests at 1x cost ──────────────────────────────────────────
    summary_rows = []
    daily_files = {}

    log("\nRunning backtests (1x cost)...\n")
    for mkt in markets:
        for tgt in targets:
            log(f"  {mkt} × {tgt}")
            df = run_backtest(mkt, tgt, cost_data[mkt], cost_mult=1.0)

            # Validation: net_return ≈ gross_return - cost_drag
            df["_computed_net"] = df["gross_return"] - df["cost_bps"] / 1e4
            diff = (df["net_return"] - df["_computed_net"]).abs().max()
            assert diff < 1e-8, f"Net return validation failed at {mkt} {tgt}: max diff={diff}"
            df.drop(columns=["_computed_net"], inplace=True)

            # Save daily file (drop internal 'flat' column)
            save_cols = ["date", "long_leg_return", "short_leg_return",
                         "gross_return", "turnover", "cost_bps", "net_return", "cumulative_net"]
            outname = os.path.join(OUTPUT, f"backtest_lgbm_{mkt.lower()}_{tgt}.csv")
            df[save_cols].to_csv(outname, index=False)
            daily_files[(mkt, tgt)] = df

            m = metrics(df)
            if np.isnan(m["net_sharpe"]):
                log(f"  WARNING: NaN Sharpe for {mkt} {tgt} — investigating")
                invested_days = (~df["flat"].fillna(False)).sum() if "flat" in df.columns else len(df[df["gross_return"] != 0])
                log(f"  Non-zero gross return days: {invested_days}")
                sys.exit(f"STOP: NaN Sharpe for {mkt} {tgt}")

            if m["avg_cost_bps"] < 0:
                sys.exit(f"STOP: Negative costs for {mkt} {tgt}")

            if m["pct_days_invested"] < (20 / len(df) * 100 * 10):
                log(f"  FLAG: Degenerate — fewer than 20 days invested for {mkt} {tgt}")

            summary_rows.append({"market": mkt, "target": tgt, **m})
            log(f"    gross_sharpe={m['gross_sharpe']:.3f}  net_sharpe={m['net_sharpe']:.3f}"
                f"  ann_ret_net={m['ann_return_net']:.3f}  turnover={m['avg_turnover']:.3f}"
                f"  %invested={m['pct_days_invested']:.1f}%")

    # ── Bootstrap IC p-values ────────────────────────────────────────────────
    log("\nBlock-bootstrap IC p-values (1000 iters, 20-day blocks)...\n")
    boot_results = {}
    for mkt in markets:
        for tgt in ["gap", "cc"]:
            p, ic = block_bootstrap_ic_pvalue(mkt, tgt)
            boot_results[(mkt, tgt)] = (p, ic)
            log(f"  {mkt} {tgt}: IC={ic:.4f}  p={p:.4f}")

    # ── Breakeven cost multipliers ───────────────────────────────────────────
    log("\nBreakeven cost multipliers (bisection)...\n")
    be_mults = {}
    for mkt in markets:
        for tgt in targets:
            be = breakeven_multiplier(mkt, tgt, cost_data[mkt])
            be_mults[(mkt, tgt)] = be
            log(f"  {mkt} {tgt}: breakeven_mult={be:.2f}x")

    # Add breakeven to summary
    for row in summary_rows:
        row["breakeven_cost_multiplier"] = be_mults[(row["market"], row["target"])]
        for tgt in ["gap", "cc"]:
            if (row["market"], tgt) in boot_results:
                p, ic = boot_results[(row["market"], tgt)]
                if row["target"] == tgt:
                    row["bootstrap_ic_pvalue"] = p
                    row["obs_ic"] = ic
        if "bootstrap_ic_pvalue" not in row:
            row["bootstrap_ic_pvalue"] = np.nan
            row["obs_ic"] = np.nan

    # ── Cost sensitivity sweep ───────────────────────────────────────────────
    log("\nCost sensitivity sweep...\n")
    sens_rows = []
    for mkt in markets:
        for tgt in targets:
            for cm in COST_MULTS:
                df_s = run_backtest(mkt, tgt, cost_data[mkt], cost_mult=cm)
                m_s = metrics(df_s)
                sens_rows.append({
                    "market": mkt, "target": tgt, "cost_multiplier": cm,
                    "net_sharpe": m_s["net_sharpe"],
                    "ann_return_net": m_s["ann_return_net"],
                    "max_drawdown_net": m_s["max_drawdown_net"],
                })
                log(f"  {mkt} {tgt} {cm:.1f}x → net_sharpe={m_s['net_sharpe']:.3f}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    col_order = ["market", "target", "gross_sharpe", "net_sharpe",
                 "ann_return_gross", "ann_return_net", "ann_vol",
                 "max_drawdown_net", "calmar_net", "avg_turnover",
                 "pct_days_invested", "breakeven_cost_multiplier",
                 "bootstrap_ic_pvalue", "obs_ic", "avg_cost_bps"]
    summary_df = summary_df[col_order]
    summary_df.to_csv(os.path.join(OUTPUT, "backtest_summary.csv"), index=False)

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(os.path.join(OUTPUT, "cost_sensitivity.csv"), index=False)

    # ── Print final summary table ─────────────────────────────────────────────
    log("\n=== SUMMARY TABLE ===")
    log(f"{'Market':<6} {'Target':<10} {'GrossSR':>8} {'NetSR':>8} "
        f"{'AnnRetNet':>10} {'Turnover':>9} {'%Inv':>6} {'BreakevenX':>11}")
    for row in summary_rows:
        log(f"{row['market']:<6} {row['target']:<10} {row['gross_sharpe']:>8.3f} "
            f"{row['net_sharpe']:>8.3f} {row['ann_return_net']:>10.3f} "
            f"{row['avg_turnover']:>9.3f} {row['pct_days_invested']:>6.1f}% "
            f"{row['breakeven_cost_multiplier']:>10.2f}x")

    log("\n=== BOOTSTRAP IC P-VALUES ===")
    for mkt in markets:
        for tgt in ["gap", "cc"]:
            p, ic = boot_results[(mkt, tgt)]
            flag = " ** p<0.05" if p < 0.05 else ""
            log(f"  {mkt} {tgt}: IC={ic:.4f}  p={p:.4f}{flag}")

    log("\n=== COST SENSITIVITY @ 2x ===")
    for mkt in markets:
        for tgt in targets:
            row_2x = next(r for r in sens_rows if r["market"] == mkt and r["target"] == tgt and r["cost_multiplier"] == 2.0)
            log(f"  {mkt} {tgt} 2x: net_sharpe={row_2x['net_sharpe']:.3f}")

    # Save log
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, "stage5_backtest.log"), "w") as f:
        f.write("\n".join(log_lines))

    log("\n=== Done. Outputs written. ===")


if __name__ == "__main__":
    main()
