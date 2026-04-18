# ABOUTME: Stage 2 — monthly-rebalanced universe construction for HK and KR crypto-exposed equities.
# ABOUTME: Inputs: data/stooq/hk_daily/*.txt, data/pykrx/kr_daily/kr_ohlcv_mcap.parquet, data/yfinance/btcusd_daily.parquet,
#           data/crypto_candidates_hk.csv, data/crypto_candidates_kr.csv.
#           Outputs: output/universe_log.csv (date, market, ticker, btc_corr, adv_usd, rank),
#           output/universe_summary.txt.
# Run: source .venv/bin/activate && python3 scripts/stage2_universe.py

import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
HK_CAND_CSV   = ROOT / "data" / "crypto_candidates_hk.csv"
KR_CAND_CSV   = ROOT / "data" / "crypto_candidates_kr.csv"
HK_DAILY_DIR  = ROOT / "data" / "stooq" / "hk_daily"
KR_PARQUET    = ROOT / "data" / "pykrx" / "kr_daily" / "kr_ohlcv_mcap.parquet"
BTC_PARQUET   = ROOT / "data" / "yfinance" / "btcusd_daily.parquet"
OUTPUT_DIR    = ROOT / "output"
OUTPUT_LOG    = OUTPUT_DIR / "universe_log.csv"
OUTPUT_SUM    = OUTPUT_DIR / "universe_summary.txt"

# ── parameters ─────────────────────────────────────────────────────────────────
ANALYSIS_START = pd.Timestamp("2019-01-01")
ANALYSIS_END   = pd.Timestamp("2026-04-15")
CORR_WINDOW    = 60   # trailing trading days for BTC correlation
ADV_WINDOW     = 20   # trailing trading days for ADV
ADV_THRESH_USD = 500_000
TOP_N_MIN      = 20
TOP_N_MAX      = 30
HKD_USD        = 1 / 7.8   # constant HKD→USD; HK$ pegged within 7.75–7.85
KRW_USD        = 1 / 1300  # constant KRW→USD; Pass 1 approximation

# ── stop-condition helper ───────────────────────────────────────────────────────
def stop(msg: str) -> None:
    print(f"\n[STOP] {msg}", file=sys.stderr)
    sys.exit(1)

# ── input validation ────────────────────────────────────────────────────────────
for f in [HK_CAND_CSV, KR_CAND_CSV, HK_DAILY_DIR, KR_PARQUET, BTC_PARQUET]:
    if not Path(f).exists():
        stop(f"Required input missing: {f}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load candidates ─────────────────────────────────────────────────────────────
hk_cands = pd.read_csv(HK_CAND_CSV)
kr_cands = pd.read_csv(KR_CAND_CSV)

# ── load BTC ────────────────────────────────────────────────────────────────────
btc = pd.read_parquet(BTC_PARQUET)[["date", "close"]].copy()
btc["date"] = pd.to_datetime(btc["date"])
btc = btc.sort_values("date").set_index("date")
btc["ret"] = btc["close"].pct_change()
btc = btc[["ret"]].rename(columns={"ret": "btc_ret"})

# ── load KR OHLCV ───────────────────────────────────────────────────────────────
kr = pd.read_parquet(KR_PARQUET).copy()
kr["date"] = pd.to_datetime(kr["date"])
kr = kr.sort_values(["ticker", "date"])
kr["adv_usd_raw"] = kr["close"] * kr["volume"] * KRW_USD
# compute daily return per ticker
kr["ret"] = kr.groupby("ticker")["close"].pct_change()

# ── load HK Stooq files ──────────────────────────────────────────────────────────
def stooq_filename(raw_ticker: str):
    """Convert HK candidate ticker to expected Stooq filename stem."""
    raw_ticker = raw_ticker.strip()
    if raw_ticker.upper().endswith(".HK"):
        numeric_part = raw_ticker[:-3]
        # try stripping leading zeros
        stripped = numeric_part.lstrip("0") or numeric_part
        if stripped.isdigit() or numeric_part.isdigit():
            return stripped + ".hk.txt"
        else:
            # non-numeric like HSK, SORA → literal lowercase
            return numeric_part.lower() + ".hk.txt"
    # non-HK suffixed (MSW.US, IMG.US) → no HK file
    return None

print("Loading HK Stooq files...")
hk_frames = []
hk_warnings = []
hk_matched = []
hk_unmatched = []

for _, row in hk_cands.iterrows():
    ticker_raw = row["ticker"].strip()
    suffix = ticker_raw.upper().split(".")[-1]
    if suffix != "HK":
        hk_unmatched.append(ticker_raw)
        hk_warnings.append(f"SKIP (non-HK suffix): {ticker_raw} — no Stooq file expected for .{suffix} listings")
        continue
    fname = stooq_filename(ticker_raw)
    fpath = HK_DAILY_DIR / fname if fname else None
    if fpath is None or not fpath.exists():
        hk_unmatched.append(ticker_raw)
        hk_warnings.append(f"SKIP (file not found): {ticker_raw} → expected {fname}")
        continue
    try:
        df = pd.read_csv(
            fpath,
            header=0,
            names=["ticker_col", "per", "date", "time", "open", "high", "low", "close", "vol", "openint"],
        )
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        df = df.sort_values("date")
        df["ticker"] = ticker_raw
        df["adv_usd_raw"] = df["close"] * df["vol"] * HKD_USD
        df["ret"] = df["close"].pct_change()
        df = df[["date", "ticker", "close", "ret", "adv_usd_raw"]]
        hk_frames.append(df)
        hk_matched.append(ticker_raw)
    except Exception as e:
        hk_unmatched.append(ticker_raw)
        hk_warnings.append(f"SKIP (parse error): {ticker_raw} — {e}")

hk = pd.concat(hk_frames, ignore_index=True) if hk_frames else pd.DataFrame()

# ── generate rebalance dates ─────────────────────────────────────────────────────
def first_trading_days(price_df: pd.DataFrame, date_col: str = "date") -> pd.DatetimeIndex:
    """Return the first trading day of each month in the analysis window, per the data."""
    d = pd.to_datetime(price_df[date_col])
    d = d[(d >= ANALYSIS_START) & (d <= ANALYSIS_END)]
    d = d.sort_values().drop_duplicates()
    df_tmp = pd.DataFrame({"date": d})
    df_tmp["ym"] = df_tmp["date"].dt.to_period("M")
    first_days = df_tmp.groupby("ym")["date"].min()
    return pd.DatetimeIndex(first_days.values)

hk_rebal_dates = first_trading_days(hk, "date") if not hk.empty else pd.DatetimeIndex([])
kr_rebal_dates = first_trading_days(kr, "date")

print(f"HK rebalance dates: {len(hk_rebal_dates)}")
print(f"KR rebalance dates: {len(kr_rebal_dates)}")

# ── compute universe at each rebalance date ──────────────────────────────────────
def compute_universe(price_df: pd.DataFrame, rebal_dates: pd.DatetimeIndex,
                     market_name: str, fx_note: str) -> pd.DataFrame:
    """
    For each rebalance date, compute trailing BTC correlation and ADV,
    apply filter, rank, select top N.

    price_df must have columns: date, ticker, ret, adv_usd_raw
    All computation uses data strictly before the rebalance date.
    """
    results = []
    all_dates = pd.DatetimeIndex(sorted(price_df["date"].unique()))
    btc_dates = pd.DatetimeIndex(sorted(btc.index))

    for reb in rebal_dates:
        # trading days strictly before rebalance date (in stock data)
        prior_stock_days = all_dates[all_dates < reb]
        if len(prior_stock_days) < max(CORR_WINDOW, ADV_WINDOW):
            continue  # insufficient history

        corr_window_days = prior_stock_days[-CORR_WINDOW:]
        adv_window_days  = prior_stock_days[-ADV_WINDOW:]

        # BTC returns over same corr window (align on date)
        btc_window = btc.loc[btc.index.isin(corr_window_days), "btc_ret"].dropna()
        if len(btc_window) < 30:
            continue  # too few BTC obs for correlation

        ticker_rows = []
        for tkr, grp in price_df.groupby("ticker"):
            grp = grp.set_index("date").sort_index()

            # correlation
            stock_ret = grp["ret"].reindex(corr_window_days).dropna()
            common_idx = stock_ret.index.intersection(btc_window.index)
            if len(common_idx) < 30:
                btc_corr = np.nan
            else:
                btc_corr = stock_ret.loc[common_idx].corr(btc_window.loc[common_idx])

            # ADV
            adv_series = grp["adv_usd_raw"].reindex(adv_window_days).dropna()
            adv_usd = adv_series.mean() if len(adv_series) >= 5 else np.nan

            if pd.isna(adv_usd) or adv_usd < ADV_THRESH_USD:
                continue
            if pd.isna(btc_corr):
                continue

            ticker_rows.append({"ticker": tkr, "btc_corr": btc_corr, "adv_usd": adv_usd})

        if not ticker_rows:
            continue

        month_df = pd.DataFrame(ticker_rows)
        month_df = month_df.sort_values("btc_corr", ascending=False).reset_index(drop=True)
        month_df["rank"] = month_df.index + 1

        # select top N
        n_select = min(TOP_N_MAX, max(len(month_df), 0))
        month_df = month_df.head(n_select)

        month_df.insert(0, "date", reb)
        month_df.insert(1, "market", market_name)
        results.append(month_df)

    if not results:
        return pd.DataFrame(columns=["date", "market", "ticker", "btc_corr", "adv_usd", "rank"])

    return pd.concat(results, ignore_index=True)

print("Computing HK universe...")
hk_univ = compute_universe(hk, hk_rebal_dates, "HK", f"HKD/USD constant {1/HKD_USD:.1f}")

print("Computing KR universe...")
# KR: exclude ME2ON.KQ (already absent from parquet, but skip silently)
kr_excl = kr[kr["ticker"] != "ME2ON.KQ"].copy()
kr_univ = compute_universe(kr_excl, kr_rebal_dates, "KR", f"KRW/USD constant {1/KRW_USD:.0f}")

# ── combine and write ────────────────────────────────────────────────────────────
universe_log = pd.concat([hk_univ, kr_univ], ignore_index=True)
universe_log = universe_log.sort_values(["market", "date", "rank"]).reset_index(drop=True)
universe_log["date"] = universe_log["date"].dt.strftime("%Y-%m-%d")
universe_log.to_csv(OUTPUT_LOG, index=False)
print(f"Written: {OUTPUT_LOG}")

# ── validation stats ─────────────────────────────────────────────────────────────
def market_stats(df: pd.DataFrame, market: str):
    sub = df[df["market"] == market]
    if sub.empty:
        return {"n_rebal": 0, "median_size": 0, "min_size": 0, "max_size": 0,
                "months_under_10": 0, "avg_btc_corr": np.nan}
    sizes = sub.groupby("date")["ticker"].count()
    return {
        "n_rebal": len(sizes),
        "median_size": int(sizes.median()),
        "min_size": int(sizes.min()),
        "max_size": int(sizes.max()),
        "months_under_10": int((sizes < 10).sum()),
        "avg_btc_corr": round(sub["btc_corr"].mean(), 4),
    }

hk_st = market_stats(universe_log, "HK")
kr_st = market_stats(universe_log, "KR")

lines = []
lines.append("=" * 70)
lines.append("STAGE 2 UNIVERSE SUMMARY")
lines.append("=" * 70)
lines.append("")
lines.append("FX ASSUMPTIONS (Pass 1 constants):")
lines.append(f"  HK: {1/HKD_USD:.1f} HKD/USD (constant). HKD pegged within 7.75-7.85.")
lines.append(f"  KR: {1/KRW_USD:.0f} KRW/USD (constant). Simple approximation; Pass 2")
lines.append(f"      should replace with daily FRED DTWEXBGS-derived or BOK rate.")
lines.append("")
lines.append("UNIVERSE PARAMETERS:")
lines.append(f"  Analysis window   : {ANALYSIS_START.date()} to {ANALYSIS_END.date()}")
lines.append(f"  BTC corr window   : {CORR_WINDOW} trailing trading days")
lines.append(f"  ADV window        : {ADV_WINDOW} trailing trading days")
lines.append(f"  ADV threshold     : ${ADV_THRESH_USD:,.0f} USD")
lines.append(f"  Universe cap      : top {TOP_N_MAX} by BTC correlation; floor = all that pass ADV")
lines.append("")
lines.append("HK MARKET:")
lines.append(f"  Rebalance dates   : {hk_st['n_rebal']}")
lines.append(f"  Universe size     : median={hk_st['median_size']}, min={hk_st['min_size']}, max={hk_st['max_size']}")
lines.append(f"  Months < 10 tickers : {hk_st['months_under_10']}")
lines.append(f"  Avg BTC correlation : {hk_st['avg_btc_corr']}")
lines.append(f"  Matched HK tickers  : {len(hk_matched)}")
lines.append(f"  Unmatched/skipped   : {len(hk_unmatched)}")
lines.append("")
lines.append("KR MARKET:")
lines.append(f"  Rebalance dates   : {kr_st['n_rebal']}")
lines.append(f"  Universe size     : median={kr_st['median_size']}, min={kr_st['min_size']}, max={kr_st['max_size']}")
lines.append(f"  Months < 10 tickers : {kr_st['months_under_10']}")
lines.append(f"  Avg BTC correlation : {kr_st['avg_btc_corr']}")
lines.append("")
lines.append("UNMATCHED / SKIPPED HK TICKERS:")
for w in hk_warnings:
    lines.append(f"  {w}")
if not hk_warnings:
    lines.append("  None")
lines.append("")
lines.append("ME2ON.KQ (KR): excluded from pykrx pull (non-standard ticker). Skipped silently.")
lines.append("")
lines.append("=" * 70)

summary_text = "\n".join(lines)
print("\n" + summary_text)

with open(OUTPUT_SUM, "w") as f:
    f.write(summary_text + "\n")
print(f"\nWritten: {OUTPUT_SUM}")

# ── stop conditions ──────────────────────────────────────────────────────────────
total_months = max(hk_st["n_rebal"], kr_st["n_rebal"])
for mkt, st in [("HK", hk_st), ("KR", kr_st)]:
    if st["n_rebal"] > 0 and st["months_under_10"] > st["n_rebal"] / 2:
        stop(
            f"{mkt}: more than half of rebalance months have universe size < 10 "
            f"({st['months_under_10']} of {st['n_rebal']}). "
            "Crypto-exposed candidate pool too thin for tercile analysis. Surface to orchestrator."
        )

print("\n[OK] Stage 2 complete. No stop conditions triggered.")
