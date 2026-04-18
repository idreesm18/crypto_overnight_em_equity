# ABOUTME: Stage 0a pre-pull validation for crypto_overnight_em_equity project.
# ABOUTME: Validates user-provided Stooq HK files, candidate CSVs, and yfinance BTC-USD.
# Run: source .venv/bin/activate && python scripts/stage0a_validate.py

import os
import re
import shutil
import random
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity")
HK_SRC = PROJECT_ROOT / "data" / "hk_daily"
HK_DST = PROJECT_ROOT / "data" / "stooq" / "hk_daily"
HK_CSV = PROJECT_ROOT / "data" / "crypto_candidates_hk.csv"
KR_CSV = PROJECT_ROOT / "data" / "crypto_candidates_kr.csv"
OUT_FILE = PROJECT_ROOT / "output" / "stage0a_validation.txt"

lines = []
blocking_flags = []

def log(s=""):
    lines.append(s)
    print(s)

def badge(status):
    return {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]", "INFO": "[INFO]"}[status]

# ── STEP 1: Move Stooq files ──────────────────────────────────────────────────
log("=" * 70)
log("STEP 1 — Stooq HK File Relocation")
log("=" * 70)

HK_DST.mkdir(parents=True, exist_ok=True)

src_files = list(HK_SRC.glob("*.txt"))
moved = 0
errors = []
for f in src_files:
    dst = HK_DST / f.name
    try:
        shutil.move(str(f), str(dst))
        moved += 1
    except Exception as e:
        errors.append(f"  ERROR moving {f.name}: {e}")

if errors:
    for e in errors:
        log(e)

# Remove src dir if now empty
remaining = list(HK_SRC.glob("*"))
if not remaining:
    try:
        HK_SRC.rmdir()
        log(f"Removed empty directory: {HK_SRC}")
    except Exception as e:
        log(f"  Could not remove {HK_SRC}: {e}")
else:
    log(f"  WARNING: {len(remaining)} files still in {HK_SRC} after move")

log(f"{badge('PASS')} Moved {moved} .txt files from data/hk_daily/ to data/stooq/hk_daily/")
log()

# ── STEP 2: Stooq HK coverage validation ─────────────────────────────────────
log("=" * 70)
log("STEP 2 — Stooq HK Coverage Validation (2018–2024)")
log("=" * 70)

import csv

# year -> set of tickers
year_tickers = defaultdict(set)
# year -> row count
year_rows = defaultdict(int)
parse_errors = 0
total_files = 0

txt_files = list(HK_DST.glob("*.txt"))
total_files = len(txt_files)

for fpath in txt_files:
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                continue
            for row in reader:
                if len(row) < 3:
                    continue
                ticker = row[0].strip()
                date_str = row[2].strip()
                if len(date_str) < 4:
                    continue
                try:
                    year = int(date_str[:4])
                except ValueError:
                    continue
                if 2018 <= year <= 2024:
                    year_tickers[year].add(ticker)
                    year_rows[year] += 1
    except Exception:
        parse_errors += 1

THRESHOLD = 1500  # 60% of 2,500
stooq_fail_years = []

log(f"Total .txt files in data/stooq/hk_daily/: {total_files}")
log(f"Parse errors: {parse_errors}")
log()
log(f"{'Year':<6} {'Unique Tickers':>15} {'Row Count':>12} {'Status':>8}")
log("-" * 45)
for yr in range(2018, 2025):
    tc = len(year_tickers.get(yr, set()))
    rc = year_rows.get(yr, 0)
    status = "PASS" if tc >= THRESHOLD else "FAIL"
    if status == "FAIL":
        stooq_fail_years.append((yr, tc))
    log(f"{yr:<6} {tc:>15,} {rc:>12,} {badge(status):>8}")

log()
if stooq_fail_years:
    blocking_flags.append("STOOQ_COVERAGE_FAIL")
    log(f"{badge('FAIL')} STOOQ_COVERAGE_FAIL — years below threshold (<{THRESHOLD:,} tickers):")
    for yr, tc in stooq_fail_years:
        log(f"  {yr}: {tc:,} tickers")
else:
    log(f"{badge('PASS')} All years 2018–2024 meet ≥{THRESHOLD:,} unique-ticker threshold")
log()

# ── STEP 3: Ticker format inspection ─────────────────────────────────────────
log("=" * 70)
log("STEP 3 — Ticker Format Inspection")
log("=" * 70)

all_fnames = [f.name for f in txt_files]
sample_fnames = random.sample(all_fnames, min(20, len(all_fnames)))
sample_fnames.sort()

log("Sample of 20 Stooq filenames:")
for fn in sample_fnames:
    log(f"  {fn}")

log()
log("Stooq filename pattern: <number>.hk.txt")
log("  - No leading zeros in filename")
log("  - Lowercase .hk.txt suffix")
log()

log("First 5 rows of crypto_candidates_hk.csv (ticker column):")
try:
    with open(HK_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows[:5]:
        log(f"  ticker={r['ticker']}")
except Exception as e:
    log(f"  ERROR reading HK CSV: {e}")
    rows = []

log()
log("Ticker format in candidate CSV: e.g. '0863.HK' — 4-digit zero-padded, uppercase .HK suffix")
log()
log("NORMALIZATION RULE (for Stage 2):")
log("  candidate_ticker -> Stooq filename:")
log("  1. Strip leading zeros from the numeric part")
log("  2. Lowercase the suffix: .HK -> .hk")
log("  3. Append .txt")
log("  Example: '0863.HK' -> strip zeros -> '863' -> '863.hk.txt'")
log("  Note: tickers without leading zeros (e.g. '1357.HK') -> '1357.hk.txt'")
log("  Also: Stooq internal TICKER field uses no leading zeros, uppercase .HK")
log("  (e.g. '863.HK' inside the file, but filename is '863.hk.txt')")
log()

# ── STEP 4: Candidate CSV schema validation ───────────────────────────────────
log("=" * 70)
log("STEP 4 — Candidate CSV Schema Validation")
log("=" * 70)

REQUIRED_HEADER = ["ticker", "company_name", "category", "source"]
HK_PATTERN = re.compile(r"^\d+\.(HK)$")
KR_PATTERN = re.compile(r"^\d+\.(KS|KQ)$")

def validate_candidate_csv(path, ticker_pattern, label):
    log(f"\n--- {label}: {path.name} ---")
    if not path.exists():
        log(f"{badge('FAIL')} File not found: {path}")
        blocking_flags.append(f"{label}_MISSING")
        return []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        data = list(reader)

    # Header check
    if list(header) != REQUIRED_HEADER:
        log(f"{badge('FAIL')} Header mismatch. Got: {header}, Expected: {REQUIRED_HEADER}")
        blocking_flags.append(f"{label}_HEADER_FAIL")
    else:
        log(f"{badge('PASS')} Header: {header}")

    log(f"  Row count: {len(data)}")

    # Duplicate tickers
    tickers = [r["ticker"].strip() for r in data]
    seen = set()
    dupes = []
    for t in tickers:
        if t in seen:
            dupes.append(t)
        seen.add(t)
    if dupes:
        log(f"{badge('WARN')} Duplicate tickers: {dupes}")
    else:
        log(f"{badge('PASS')} No duplicate tickers")

    # Format check
    bad_format = [(r["ticker"], r.get("company_name", "")) for r in data
                  if not ticker_pattern.match(r["ticker"].strip())]
    if bad_format:
        log(f"{badge('WARN')} {len(bad_format)} tickers fail format pattern:")
        for t, n in bad_format:
            log(f"    {t!r}  ({n})")
    else:
        log(f"{badge('PASS')} All tickers match expected format")

    return [r["ticker"].strip() for r in data]

hk_tickers = validate_candidate_csv(HK_CSV, HK_PATTERN, "HK")
kr_tickers = validate_candidate_csv(KR_CSV, KR_PATTERN, "KR")
log()

# ── STEP 5: Candidate-to-Stooq match rate for HK ─────────────────────────────
log("=" * 70)
log("STEP 5 — Candidate-to-Stooq Match Rate (HK)")
log("=" * 70)

def normalize_hk_ticker_to_filename(ticker):
    """'0863.HK' -> '863.hk.txt'"""
    t = ticker.upper()
    if "." in t:
        num_part, suffix = t.rsplit(".", 1)
    else:
        num_part, suffix = t, "HK"
    num_stripped = str(int(num_part)) if num_part.isdigit() else num_part
    return f"{num_stripped}.hk.txt"

stooq_filenames = set(f.name for f in txt_files)

matched = []
unmatched = []

# Build candidate name map
try:
    with open(HK_CSV, "r") as f:
        reader = csv.DictReader(f)
        hk_rows = list(reader)
except Exception:
    hk_rows = []

for row in hk_rows:
    ticker = row["ticker"].strip()
    name = row.get("company_name", "").strip()
    if not HK_PATTERN.match(ticker):
        # Skip malformed tickers (already warned)
        continue
    fname = normalize_hk_ticker_to_filename(ticker)
    if fname in stooq_filenames:
        matched.append(ticker)
    else:
        unmatched.append((ticker, name))

valid_hk = len(matched) + len(unmatched)
match_rate = len(matched) / valid_hk if valid_hk > 0 else 0.0

log(f"HK candidates (valid format): {valid_hk}")
log(f"Matched to Stooq .txt file:   {len(matched)}")
log(f"Match rate: {match_rate:.1%}")
log()

if unmatched:
    log(f"{badge('WARN') if match_rate >= 0.7 else badge('WARN')} Unmatched HK candidates ({len(unmatched)}):")
    for t, n in unmatched:
        log(f"    {t}  ({n})")
else:
    log(f"{badge('PASS')} All valid HK candidates have matching Stooq files")

if match_rate < 0.70:
    log(f"\n{badge('WARN')} Match rate {match_rate:.1%} < 70%. Stage 2 will handle delisted/renamed tickers.")
else:
    log(f"\n{badge('PASS')} Match rate {match_rate:.1%} >= 70%")
log()

# ── STEP 6: yfinance BTC-USD cross-check ─────────────────────────────────────
log("=" * 70)
log("STEP 6 — yfinance BTC-USD Cross-Check")
log("=" * 70)

REFERENCE = {
    "2024-01-02": 44950,
    "2024-01-05": 44100,
    "2024-01-08": 46970,
}

yf_status = "PASS"
try:
    import yfinance as yf
    btc = yf.download("BTC-USD", start="2024-01-02", end="2024-01-11",
                      progress=False, auto_adjust=True)

    # Flatten MultiIndex if present
    if isinstance(btc.columns, type(btc.columns)) and hasattr(btc.columns, 'levels'):
        btc.columns = [c[0] if isinstance(c, tuple) else c for c in btc.columns]

    btc.index = btc.index.tz_localize(None) if btc.index.tz is not None else btc.index
    btc.index = btc.index.normalize()

    log(f"yfinance returned {len(btc)} rows for 2024-01-02 to 2024-01-10")
    log()
    log(f"{'Date':<12} {'Close':>10} {'Reference':>12} {'Diff%':>8} {'Match':>6}")
    log("-" * 52)

    matches = 0
    all_miss = True
    for date_str, ref_val in REFERENCE.items():
        import pandas as pd
        dt = pd.Timestamp(date_str)
        if dt in btc.index:
            close_val = float(btc.loc[dt, "Close"])
            diff_pct = abs(close_val - ref_val) / ref_val * 100
            within5 = diff_pct <= 5.0
            within10 = diff_pct <= 10.0
            if within5:
                matches += 1
            if within10:
                all_miss = False
            status_str = "YES" if within5 else "NO"
            log(f"{date_str:<12} {close_val:>10,.0f} {ref_val:>12,} {diff_pct:>7.1f}% {status_str:>6}")
        else:
            log(f"{date_str:<12} {'N/A':>10} {ref_val:>12,} {'N/A':>8} {'N/A':>6}")

    log()
    if matches >= 2:
        log(f"{badge('PASS')} {matches}/3 reference dates match within 5% tolerance")
    elif all_miss:
        log(f"{badge('WARN')} All 3 dates miss by >10%. yfinance may be unreliable. Stage 1 will still try.")
        yf_status = "WARN"
    else:
        log(f"{badge('WARN')} Only {matches}/3 dates match within 5%. Proceed with caution.")
        yf_status = "WARN"

except Exception as e:
    log(f"{badge('WARN')} yfinance import/download failed: {e}")
    log("Stage 1 will still attempt BTC-USD pull.")
    yf_status = "WARN"

log()

# ── STEP 7: BLOCKING GATE RESULT ─────────────────────────────────────────────
log("=" * 70)
log("BLOCKING GATE RESULT")
log("=" * 70)

if blocking_flags:
    log(f"BLOCK — Reasons: {', '.join(blocking_flags)}")
    if "STOOQ_COVERAGE_FAIL" in blocking_flags:
        log()
        log("STOOQ_HK_COVERAGE failure details:")
        for yr, tc in stooq_fail_years:
            log(f"  {yr}: {tc:,} unique tickers (threshold: {THRESHOLD:,})")
    log()
    log("DO NOT proceed to Stage 1 until blocking issues are resolved.")
else:
    log("PASS — All blocking checks passed. Stage 1 may proceed.")
    log("Non-blocking warnings (if any) noted above.")

log()
log(f"yfinance cross-check: {yf_status}")
log(f"Files moved to data/stooq/hk_daily/: {moved}")

# ── Write output file ─────────────────────────────────────────────────────────
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nValidation written to: {OUT_FILE}")
