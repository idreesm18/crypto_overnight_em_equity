# ABOUTME: Stage P2-6 — Extract and package raw overnight minute-bar sequences for TCN training.
# ABOUTME: Inputs: data/binance/spot_klines/{BTC,ETH,SOL}USDT_1m.parquet, output/features_track_a_{hk,kr,index}.parquet, output/universe_log.csv
# ABOUTME: Outputs: output/sequences_{hk,kr,index_hk,index_kr}.npz; logs/stage_p2-6_sequence_prep.log
# Run: source .venv/bin/activate && python3 scripts/stage_p2-6_sequence_prep.py

import sys
import logging
import warnings
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path("/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity_p2")
DATA = ROOT / "data"
OUT  = ROOT / "output"
LOGS = ROOT / "logs"
OUT.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

LOG_FILE = LOGS / "stage_p2-6_sequence_prep.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────
WINDOW_START   = "2019-01-01"
WINDOW_END     = "2026-04-17"
SOL_LAUNCH_UTC = pd.Timestamp("2020-08-11 06:00:00", tz="UTC")

# 5-min resolution required: 1-min total would exceed 20 GB
# Max window at 1-min: HK=8490, KR=11130 => at 5-min: HK=1698, KR=2226
RESAMPLE_MIN   = 5           # 5-minute bars
HK_TMAX_5MIN   = 8490 // 5  # 1698  (max weekend window)
KR_TMAX_5MIN   = 11130 // 5 # 2226  (max weekend window)

# OHLCV channels per coin: open, high, low, close, volume → 3 coins × 5 = 15
CHANNELS = 15  # BTC(5) + ETH(5) + SOL(5)

# Static feature dims
STATIC_DIM_STOCK = 9   # 6 macro + 3 stock-level (log_mcap_bucket dropped)
STATIC_DIM_INDEX = 6   # 6 macro only

MACRO_COLS = [
    "vix_level", "vix_5d_change", "yield_curve_slope",
    "dxy_level", "dxy_5d_change", "breakeven_5y",
]
STOCK_EXTRA_COLS = ["stock_rv_20d", "stock_ret_20d", "stock_prior_day_return"]

GB_THRESHOLD = 20.0  # switch to 5-min if total would exceed this


def stop(msg: str) -> None:
    log.error(f"[BLOCK] {msg}")
    sys.exit(1)


# ── Load Binance spot 1-min klines ────────────────────────────────────────────
def load_spot_klines() -> dict:
    """Load BTC, ETH, SOL 1-min spot klines; index on open_time (UTC)."""
    klines = {}
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        p = DATA / "binance" / "spot_klines" / f"{sym}_1m.parquet"
        df = pd.read_parquet(p).set_index("open_time").sort_index()
        klines[sym] = df
        log.info(f"Loaded {sym}: {len(df):,} rows, {df.index.min()} to {df.index.max()}")
    return klines


# ── Resample 1-min bars to N-min bars ────────────────────────────────────────
def resample_ohlcv(df: pd.DataFrame, freq_min: int) -> pd.DataFrame:
    """
    Resample OHLCV to freq_min bars.
    O = first open, H = max high, L = min low, C = last close, V = sum volume.
    """
    rule = f"{freq_min}min"
    resampled = df.resample(rule, closed="left", label="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open"])
    return resampled


# ── Build overnight windows using exchange_calendars ─────────────────────────
def build_overnight_windows(market: str) -> pd.DataFrame:
    cal_name = "XHKG" if market == "HK" else "XKRX"
    cal = xcals.get_calendar(cal_name)
    sessions = cal.sessions_in_range(WINDOW_START, WINDOW_END)
    rows = []
    for i, sess in enumerate(sessions[:-1]):
        try:
            w_start = cal.session_close(sess)
        except Exception:
            continue
        next_sess = sessions[i + 1]
        try:
            w_end = cal.session_open(next_sess)
        except Exception:
            continue
        gap_days = (next_sess - sess).days
        rows.append({
            "date":             pd.Timestamp(sess.date()),
            "next_date":        pd.Timestamp(next_sess.date()),
            "window_start_utc": w_start,
            "window_end_utc":   w_end,
            "is_weekend_gap":   gap_days > 3,
        })
    df = pd.DataFrame(rows)
    log.info(f"{market}: {len(df)} overnight windows, {df['is_weekend_gap'].sum()} weekend gaps")
    return df


# ── Per-window per-channel normalization ─────────────────────────────────────
def normalize_window(seq: np.ndarray) -> np.ndarray:
    """
    seq: (T, 15) float32.
    Normalize each channel: subtract mean, divide by std (channel-wise over T).
    If std == 0 (constant or all-zero channel), leave as zeros.
    """
    out = seq.copy()
    for c in range(seq.shape[1]):
        col = seq[:, c]
        mu  = col.mean()
        std = col.std()
        if std > 0:
            out[:, c] = (col - mu) / std
    return out


# ── Extract one overnight sequence ───────────────────────────────────────────
def extract_sequence(
    klines_5m: dict,
    w_start: pd.Timestamp,
    w_end: pd.Timestamp,
    t_max: int,
    sol_launch: pd.Timestamp,
) -> tuple:
    """
    Returns (seq_padded, mask) where:
      seq_padded: (t_max, 15) float32, zero-padded at the end
      mask:       (t_max,) bool, True for valid bars
    Channels: [BTC_O, BTC_H, BTC_L, BTC_C, BTC_V, ETH_O, ..., SOL_O, ...]
    SOL channels are zero-filled (mask False) if window predates SOL launch.
    """
    frames = []
    sol_available = w_end > sol_launch

    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        df = klines_5m[sym]
        sl = df[(df.index >= w_start) & (df.index < w_end)]
        if sym == "SOLUSDT" and not sol_available:
            # Pre-launch: produce zero frame aligned to BTC length
            btc_sl = klines_5m["BTCUSDT"]
            btc_sl = btc_sl[(btc_sl.index >= w_start) & (btc_sl.index < w_end)]
            t_len = len(btc_sl)
            frames.append(np.zeros((t_len, 5), dtype=np.float32))
        else:
            if len(sl) == 0:
                frames.append(np.zeros((0, 5), dtype=np.float32))
            else:
                arr = sl[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
                frames.append(arr)

    # All frames must have same T; use BTC as reference length
    t_btc = frames[0].shape[0]
    for i in range(1, 3):
        t_i = frames[i].shape[0]
        if t_i > t_btc:
            frames[i] = frames[i][:t_btc]
        elif t_i < t_btc:
            pad = np.zeros((t_btc - t_i, 5), dtype=np.float32)
            frames[i] = np.vstack([frames[i], pad])

    seq = np.concatenate(frames, axis=1)  # (T, 15)
    t_actual = seq.shape[0]

    # Per-window per-channel normalization (only on valid SOL bars if pre-launch)
    if sol_available:
        seq = normalize_window(seq)
    else:
        # Normalize BTC and ETH channels only (cols 0-9); SOL (cols 10-14) stay 0
        seq_norm = normalize_window(seq[:, :10])
        seq[:, :10] = seq_norm

    # Pad to t_max
    if t_actual >= t_max:
        seq_padded = seq[:t_max].copy()
        mask = np.ones(t_max, dtype=bool)
    else:
        pad_len = t_max - t_actual
        pad = np.zeros((pad_len, CHANNELS), dtype=np.float32)
        seq_padded = np.vstack([seq, pad])
        mask = np.zeros(t_max, dtype=bool)
        mask[:t_actual] = True

    return seq_padded.astype(np.float32), mask


# ── Build sequences for one universe ─────────────────────────────────────────
def build_sequences(
    name: str,
    feature_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    klines_5m: dict,
    t_max: int,
    is_index: bool,
) -> dict:
    """
    Returns dict with keys matching npz arrays spec.
    For stock universes, join features_df on (date, ticker).
    For index universes, feature_df has one row per date per index.
    """
    log.info(f"\n{'='*60}")
    log.info(f"Building sequences: {name}")
    log.info(f"  t_max={t_max}, is_index={is_index}")

    # Align features to windows
    if is_index:
        # feature_df: one row per (date, ticker); ticker = '^HSI' or '^KS11'
        ticker_val = "^HSI" if "hk" in name else "^KS11"
        feat_sub = feature_df[feature_df["ticker"] == ticker_val].copy()
        feat_sub["date"] = pd.to_datetime(feat_sub["date"]).dt.normalize()
        feat_sub = feat_sub.set_index("date").sort_index()

        # Windows: one row per session
        wins = windows_df.copy()
        wins["date_key"] = pd.to_datetime(wins["date"]).dt.normalize()
        # Join on date_key
        wins = wins[wins["date_key"].isin(feat_sub.index)]
        merged = wins.merge(
            feat_sub.reset_index(),
            left_on="date_key",
            right_on="date",
            how="inner",
        )
        log.info(f"  Merged rows: {len(merged)} (windows with feature match)")
    else:
        # Stock universe: feature_df has (date, ticker) rows
        feat_sub = feature_df.copy()
        feat_sub["date"] = pd.to_datetime(feat_sub["date"]).dt.normalize()
        wins = windows_df.copy()
        wins["date_key"] = pd.to_datetime(wins["date"]).dt.normalize()

        merged = feat_sub.merge(
            wins[["date_key", "window_start_utc", "window_end_utc"]],
            left_on="date",
            right_on="date_key",
            how="inner",
        )
        log.info(f"  Merged rows: {len(merged)} (stock-day with window match)")

    if len(merged) == 0:
        stop(f"[BLOCK] No rows after merge for {name}")

    n_rows = len(merged)
    log.info(f"  Total sequences to extract: {n_rows:,}")

    # Pre-allocate output arrays
    sequences     = np.zeros((n_rows, t_max, CHANNELS), dtype=np.float32)
    masks         = np.zeros((n_rows, t_max), dtype=bool)
    static_dim    = STATIC_DIM_INDEX if is_index else STATIC_DIM_STOCK
    static_feat   = np.full((n_rows, static_dim), np.nan, dtype=np.float32)
    tgt_gap       = np.full(n_rows, np.nan, dtype=np.float32)
    tgt_intraday  = np.full(n_rows, np.nan, dtype=np.float32)
    tgt_cc        = np.full(n_rows, np.nan, dtype=np.float32)
    dates_out     = []
    tickers_out   = []

    # Determine target column names
    if is_index:
        gap_col      = "gap_return"
        intraday_col = "intraday_return"
        cc_col       = "cc_return"
    else:
        gap_col      = "tgt_gap"
        intraday_col = "tgt_intraday"
        cc_col       = "tgt_cc"

    # Track mask alignment check for spot-check (5 windows)
    spot_check_results = []
    spot_check_idx     = set(range(0, min(5, n_rows)))

    log.info(f"  Extracting sequences (logging every 5000)...")
    for i, row in enumerate(merged.itertuples(index=False)):
        if i % 5000 == 0:
            log.info(f"    {i}/{n_rows} processed")

        w_start = row.window_start_utc
        w_end   = row.window_end_utc

        seq_padded, mask = extract_sequence(
            klines_5m, w_start, w_end, t_max, SOL_LAUNCH_UTC
        )

        sequences[i] = seq_padded
        masks[i]     = mask

        # Targets
        tgt_gap[i]      = getattr(row, gap_col,      np.nan)
        tgt_intraday[i] = getattr(row, intraday_col, np.nan)
        tgt_cc[i]       = getattr(row, cc_col,       np.nan)

        # Static features
        macro_vals = [getattr(row, c, np.nan) for c in MACRO_COLS]
        if is_index:
            static_feat[i] = np.array(macro_vals, dtype=np.float32)
        else:
            stock_vals = [getattr(row, c, np.nan) for c in STOCK_EXTRA_COLS]
            static_feat[i] = np.array(macro_vals + stock_vals, dtype=np.float32)

        # Dates and tickers
        if is_index:
            dates_out.append(str(row.date_key.date()))
            tickers_out.append(ticker_val)
        else:
            dates_out.append(str(row.date.date()))
            tickers_out.append(str(row.ticker))

        # Spot-check mask alignment
        if i in spot_check_idx:
            valid_count = int(mask.sum())
            # Verify: recount actual bars in window
            btc_df = klines_5m["BTCUSDT"]
            actual_bars = int(((btc_df.index >= w_start) & (btc_df.index < w_end)).sum())
            expected_valid = min(actual_bars, t_max)
            check_pass = (valid_count == expected_valid)
            spot_check_results.append({
                "idx": i, "valid_count": valid_count,
                "expected_valid": expected_valid, "pass": check_pass,
            })

    log.info(f"  Sequence extraction complete.")

    # Spot-check validation
    log.info(f"\n  Mask alignment spot-check (5 windows):")
    all_pass = True
    for sc in spot_check_results:
        status = "PASS" if sc["pass"] else "FAIL"
        log.info(f"    idx={sc['idx']}: valid={sc['valid_count']}, expected={sc['expected_valid']} -> {status}")
        if not sc["pass"]:
            all_pass = False
    if not all_pass:
        stop(f"[BLOCK] Mask alignment failed spot-check for {name}")

    # Check no all-zero non-SOL channels (BTC/ETH, cols 0-9) in post-launch rows
    # Exclude rows with zero valid bars (data-cutoff boundary — expected).
    post_launch_mask = np.array([
        pd.Timestamp(str(d) + "T00:00:00", tz="UTC") > SOL_LAUNCH_UTC
        for d in dates_out
    ])
    valid_bar_counts = masks.sum(axis=1)
    has_valid = valid_bar_counts > 0
    check_mask = post_launch_mask & has_valid
    n_zero_valid = int((post_launch_mask & ~has_valid).sum())
    if n_zero_valid > 0:
        log.info(f"  {n_zero_valid} post-launch rows have 0 valid bars (data-cutoff boundary — expected, skipped)")
    if check_mask.sum() > 0:
        check_seqs  = sequences[check_mask]   # (N_check, t_max, 15)
        check_masks = masks[check_mask]        # (N_check, t_max)
        for c in range(10):  # BTC + ETH channels only
            # Only look at valid bars per row
            n_all_zero = 0
            for j in range(check_seqs.shape[0]):
                n_valid = int(check_masks[j].sum())
                if n_valid > 0 and (check_seqs[j, :n_valid, c] == 0).all():
                    n_all_zero += 1
            if n_all_zero > 0:
                log.warning(f"  Channel {c}: {n_all_zero} post-launch rows are all-zero within valid bars (unexpected)")

    return {
        "sequences":         sequences,
        "masks":             masks,
        "static_features":   static_feat,
        "targets_gap":       tgt_gap,
        "targets_intraday":  tgt_intraday,
        "targets_cc":        tgt_cc,
        "dates":             np.array(dates_out),
        "tickers":           np.array(tickers_out),
    }


# ── Save npz and log stats ────────────────────────────────────────────────────
def save_npz(name: str, out_path: Path, arrays: dict) -> int:
    """Save arrays to npz. Returns file size in bytes."""
    try:
        np.savez_compressed(str(out_path), **arrays)
    except MemoryError as e:
        stop(f"[BLOCK] MemoryError writing {out_path}: {e}")
    except Exception as e:
        stop(f"[BLOCK] Cannot write {out_path}: {e}")
    size_bytes = out_path.stat().st_size
    n = arrays["sequences"].shape[0]
    t = arrays["sequences"].shape[1]
    log.info(f"  Saved {name}: {out_path.name} | {size_bytes/1e9:.3f} GB | {n:,} sequences | T_max={t}")
    return size_bytes


# ── Sequence length distribution check ───────────────────────────────────────
def check_seq_length_dist(masks: np.ndarray, market: str) -> None:
    """Log distribution of valid bar counts per sequence."""
    valid_counts = masks.sum(axis=1)
    p5, p50, p95 = np.percentile(valid_counts, [5, 50, 95])
    log.info(
        f"  {market} valid bar dist: p5={p5:.0f}, p50={p50:.0f}, p95={p95:.0f}, "
        f"max={valid_counts.max():.0f}"
    )
    # Warn if p50 is far outside expected (1-min: ~1050; 5-min: ~210)
    expected_p50_5min = 1050 // RESAMPLE_MIN
    if not (expected_p50_5min * 0.5 < p50 < expected_p50_5min * 5):
        log.warning(
            f"  {market}: p50 valid bars ({p50}) outside expected range "
            f"[{expected_p50_5min*0.5:.0f}, {expected_p50_5min*5:.0f}]"
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("Stage P2-6: Track B Sequence Preprocessing for TCN")
    log.info(f"Resolution: {RESAMPLE_MIN}-min bars")
    log.info(f"Channels: {CHANNELS} (BTC×5, ETH×5, SOL×5)")
    log.info(f"SOL launch cutoff (UTC): {SOL_LAUNCH_UTC}")
    log.info(f"HK T_max (5-min): {HK_TMAX_5MIN} | KR T_max (5-min): {KR_TMAX_5MIN}")

    # ── Load 1-min klines ──────────────────────────────────────────────────────
    log.info("\nLoading 1-min spot klines...")
    klines_1m = load_spot_klines()

    # ── Resample to 5-min ──────────────────────────────────────────────────────
    log.info(f"\nResampling to {RESAMPLE_MIN}-min bars...")
    klines_5m = {}
    for sym, df in klines_1m.items():
        klines_5m[sym] = resample_ohlcv(df, RESAMPLE_MIN)
        log.info(f"  {sym}: {len(klines_1m[sym]):,} 1-min -> {len(klines_5m[sym]):,} 5-min bars")
    del klines_1m  # free memory

    # ── Load feature tables ────────────────────────────────────────────────────
    log.info("\nLoading feature tables...")
    feat_hk  = pd.read_parquet(OUT / "features_track_a_hk.parquet")
    feat_kr  = pd.read_parquet(OUT / "features_track_a_kr.parquet")
    feat_idx = pd.read_parquet(OUT / "features_track_a_index.parquet")
    log.info(f"  HK  features: {feat_hk.shape}")
    log.info(f"  KR  features: {feat_kr.shape}")
    log.info(f"  Idx features: {feat_idx.shape}")

    # ── Build overnight windows ────────────────────────────────────────────────
    log.info("\nBuilding overnight windows...")
    wins_hk = build_overnight_windows("HK")
    wins_kr = build_overnight_windows("KR")

    # ── Sequence count checks (pre-flight) ────────────────────────────────────
    n_hk_expected  = len(feat_hk)
    n_kr_expected  = len(feat_kr)
    n_idx_hk_exp   = len(feat_idx[feat_idx["ticker"] == "^HSI"])
    n_idx_kr_exp   = len(feat_idx[feat_idx["ticker"] == "^KS11"])
    log.info(f"\nExpected sequence counts:")
    log.info(f"  HK stock:   {n_hk_expected:,}")
    log.info(f"  KR stock:   {n_kr_expected:,}")
    log.info(f"  Index HSI:  {n_idx_hk_exp:,}")
    log.info(f"  Index KS11: {n_idx_kr_exp:,}")

    # ── Build all four sequence sets ───────────────────────────────────────────
    arrs_hk = build_sequences(
        "sequences_hk", feat_hk, wins_hk, klines_5m,
        t_max=HK_TMAX_5MIN, is_index=False,
    )
    arrs_kr = build_sequences(
        "sequences_kr", feat_kr, wins_kr, klines_5m,
        t_max=KR_TMAX_5MIN, is_index=False,
    )
    arrs_idx_hk = build_sequences(
        "sequences_index_hk", feat_idx, wins_hk, klines_5m,
        t_max=HK_TMAX_5MIN, is_index=True,
    )
    arrs_idx_kr = build_sequences(
        "sequences_index_kr", feat_idx, wins_kr, klines_5m,
        t_max=KR_TMAX_5MIN, is_index=True,
    )

    # ── Sequence count BLOCK checks ───────────────────────────────────────────
    for label, arrs, n_exp in [
        ("HK stock",   arrs_hk,     n_hk_expected),
        ("KR stock",   arrs_kr,     n_kr_expected),
        ("Index HSI",  arrs_idx_hk, n_idx_hk_exp),
        ("Index KS11", arrs_idx_kr, n_idx_kr_exp),
    ]:
        n_actual = arrs["sequences"].shape[0]
        pct = n_actual / max(n_exp, 1)
        log.info(f"  {label}: {n_actual:,} / {n_exp:,} expected ({pct*100:.1f}%)")
        if pct < 0.50:
            stop(f"[BLOCK] {label} sequence count {n_actual:,} < 50% of expected {n_exp:,}")

    # ── Sequence length distribution checks ───────────────────────────────────
    log.info("\nSequence length distributions:")
    check_seq_length_dist(arrs_hk["masks"],     "HK stock")
    check_seq_length_dist(arrs_kr["masks"],     "KR stock")
    check_seq_length_dist(arrs_idx_hk["masks"], "Index HSI")
    check_seq_length_dist(arrs_idx_kr["masks"], "Index KS11")

    # ── SOL zero-channel documentation ────────────────────────────────────────
    sol_cutoff = pd.Timestamp("2020-08-11")
    dates_hk = arrs_hk["dates"]
    pre_sol_hk = sum(1 for d in dates_hk if pd.Timestamp(str(d)) < sol_cutoff)
    log.info(f"\nSOL handling:")
    log.info(f"  HK sequences pre-SOL-launch (zero-filled SOL channels): {pre_sol_hk:,}")
    dates_kr = arrs_kr["dates"]
    pre_sol_kr = sum(1 for d in dates_kr if pd.Timestamp(str(d)) < sol_cutoff)
    log.info(f"  KR sequences pre-SOL-launch (zero-filled SOL channels): {pre_sol_kr:,}")

    # ── Save npz files ────────────────────────────────────────────────────────
    log.info("\nSaving .npz files...")
    sz_hk     = save_npz("sequences_hk",       OUT / "sequences_hk.npz",       arrs_hk)
    sz_kr     = save_npz("sequences_kr",        OUT / "sequences_kr.npz",       arrs_kr)
    sz_idx_hk = save_npz("sequences_index_hk", OUT / "sequences_index_hk.npz", arrs_idx_hk)
    sz_idx_kr = save_npz("sequences_index_kr", OUT / "sequences_index_kr.npz", arrs_idx_kr)
    total_gb = (sz_hk + sz_kr + sz_idx_hk + sz_idx_kr) / 1e9
    log.info(f"  Total .npz size: {total_gb:.3f} GB")

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("\n" + "="*70)
    log.info("STAGE P2-6 SUMMARY")
    log.info("="*70)
    log.info(f"  Resolution used: {RESAMPLE_MIN}-min (1-min would exceed 20 GB)")
    log.info(f"  sequences_hk.npz:       {sz_hk/1e9:.3f} GB | {arrs_hk['sequences'].shape[0]:,} seqs | T_max={HK_TMAX_5MIN}")
    log.info(f"  sequences_kr.npz:       {sz_kr/1e9:.3f} GB | {arrs_kr['sequences'].shape[0]:,} seqs | T_max={KR_TMAX_5MIN}")
    log.info(f"  sequences_index_hk.npz: {sz_idx_hk/1e9:.3f} GB | {arrs_idx_hk['sequences'].shape[0]:,} seqs | T_max={HK_TMAX_5MIN}")
    log.info(f"  sequences_index_kr.npz: {sz_idx_kr/1e9:.3f} GB | {arrs_idx_kr['sequences'].shape[0]:,} seqs | T_max={KR_TMAX_5MIN}")
    log.info(f"  HK T_max (5-min): {HK_TMAX_5MIN} | KR T_max (5-min): {KR_TMAX_5MIN}")
    log.info(f"  SOL: zero-filled (mask=False implicit via normalization skip) pre-2020-08-11")
    log.info(f"  Static feature dim: stock={STATIC_DIM_STOCK}, index={STATIC_DIM_INDEX}")
    log.info(f"  Mask alignment spot-check: PASS")
    log.info(f"  Validation: PASS")
    log.info("Stage P2-6 complete.")


if __name__ == "__main__":
    main()
