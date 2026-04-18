# Orchestrator Prompt -- Crypto Overnight Signal for Asian Equities (Pass 1)
# Project: crypto_overnight_signal
# Pass: 1 of 2
# Start Claude Code with: claude --model claude-opus-4-6
# Then paste this prompt

---

Read CLAUDE.md for your role and operating instructions.
Read briefs/crypto_overnight_signal_pass1.md for the research question
and full pipeline specification.

You are the Opus orchestrator. Work autonomously through all stages
below. Delegate implementation to Sonnet subagents. Convene the
adversarial review panel after results are produced. Stop only on the
explicit stopping conditions in CLAUDE.md or Pass 1 blocking gates.

Project directory: /Users/idrees/Desktop/Claude/projects/crypto_overnight_signal/

User-provided items already in place:
- data/stooq/hk_daily/ with .txt files (manual download)
- data/crypto_candidates_hk.csv
- data/crypto_candidates_kr.csv
- .env with FRED_API_KEY

Do not re-download Stooq HK. Do not modify candidate CSVs. If any
user-provided item is missing at Stage -2 verification, STOP and
surface to user.

Pass 1 scope:
- Markets: HK + KR (both)
- Model: LightGBM only (no TCN, no Modal, no GPU)
- Universe: crypto-exposed stocks per market (20-30 after filters)
- Targets: gap (PRIMARY), intraday, close-to-close
- Review: 2 rounds (position + 1 rebuttal)
- Total LightGBM runs: 6 (1 model x 3 targets x 2 markets)

Pass 2 is a separate future run and adds TCN, index-level prediction,
and a third review round. Do not attempt Pass 2 work in this run.

---

SUBAGENT PROMPT FORMAT

Every Task() call must begin with the following identical preamble
before any role-specific instructions. The wording must be identical
across all calls for prompt caching to apply.

  OPERATING STANDARDS: Walk-forward OOS evaluation only. Never report
  in-sample performance as a finding. Null results are valid findings.
  Every claim must be supported by output files. Do not hedge empirical
  results; hedge interpretations only. Flag any result with train IC
  exceeding OOS IC by more than 0.20 as potential overfitting.
  Activate .venv before running any script: source .venv/bin/activate
  && python scripts/<name>.py. Do not install packages without logging
  the version to logs/stage_-1_env.log.
  CONTEXT FILES: briefs/crypto_overnight_signal_pass1.md, output/*.csv,
  output/*.parquet

Scope each subagent to the minimum file set for its stage. Do not hand
a subagent the entire output/ directory when only two files are needed.

---

CONCURRENCY

Spawn subagents sequentially. Do not spawn parallel Task() calls in Stages 1,
3, 4, or 6. The review panel in Stage 7 spawns sequentially (4 position
papers, then 4 rebuttals, then synthesis).

---

WRITING VOICE DIRECTIVES (enforce in Stage 8 documentation)

Hedged junior-researcher voice. State findings clearly; hedge
interpretations and causal language. Do not write with an edgy
tone. Use "in this sample" for scope-limited results.

Anti-AI writing patterns to avoid:
- Triple-parallel sentence structures
- Self-announcing topic sentences ("This section will discuss...")
- Uniform sentence rhythm
- Hype words: "robust", "powerful", "leverages", "crucial"
- Em-dashes (use colons, commas, or sentence breaks instead)

---
## Operational Preconditions

For any pipeline run that touches KRX data (Stages 1, 2, 3):

1. The Oracle Cloud Seoul VM must be running. User is responsible for
   keeping it up. If the VM is down, scripts/krx_tunnel.py will fail
   during tunnel open and the Stage 1 subagent must surface this to
   the user with the message: "Oracle Seoul VM unreachable: user must
   verify VM is running at cloud.oracle.com."

2. All pykrx-touching scripts MUST use the `krx_tunnel()` context
   manager from scripts/krx_tunnel.py. Do not import pykrx at module
   level. Do not strip the context manager as apparent boilerplate.
   It is load-bearing.

---

PIPELINE EXECUTION

STAGE -2 -- PROJECT FILE STRUCTURE

You create the directory tree yourself. No subagent needed.
Follow the tree in the brief. Verify user-provided items exist.
Block if any are missing.

STAGE -1 -- ENVIRONMENT SETUP

Spawn one Sonnet subagent. Task: create .venv with Python 3.11, install
pinned dependencies from requirements.txt, verify .env, run pykrx smoke
test on ticker 005930 for a 1-week range. Write logs/stage_-1_env.log.
STOP if Python 3.11 unavailable, pykrx smoke test fails, or FRED_API_KEY
missing.

STAGE 0 -- DATA VALIDATION (blocking gate)

Spawn one Sonnet subagent. Task: validate user-provided Stooq HK
coverage, pykrx KR coverage, Binance minute kline coverage (after
Stage 1 completes its first download batch), candidate CSV ticker
matching with ticker format normalization, yfinance BTC-USD cross-check.
Write output/stage0_validation.txt.

Important ordering: Stage 0 runs partially before Stage 1 (validate
user-provided items: Stooq HK, candidate CSVs) and partially after
Stage 1 (validate pulled data: Binance, pykrx, FRED, yfinance). Split
into Stage 0a (pre-Stage-1) and Stage 0b (post-Stage-1).

STOP if any validation fails. If Stooq HK fails the 60% threshold,
surface to user with the option to drop HK and continue KR-only.

STAGE 1 -- DATA PULL

Spawn one Sonnet subagent. Task: pull Binance spot + perp klines,
funding rates, liquidations for BTC/ETH/SOL/BNB/XRP. Pull pykrx
KR daily OHLCV + market cap. Pull FRED macro series. Pull yfinance
BTC-USD daily. Save as parquet in directories per brief. Do NOT
touch data/stooq/hk_daily/.

Haiku validation: schema checks, row counts, date coverage. Flag
any file with < 90% expected row count.

STAGE 2 -- UNIVERSE CONSTRUCTION

Spawn one Sonnet subagent. Task: read candidate CSVs, compute trailing
60-day BTC correlation per candidate at each monthly rebalance date,
apply $500K ADV filter, select top 20-30 per market. Write
output/universe_log.csv and output/universe_summary.txt. No lookahead.

Haiku validation: verify universe_log.csv has monotonic rebalance
dates, no duplicate ticker-date rows, universe size 20-30 per market.

STAGE 3 -- FEATURE ENGINEERING (Track A only)

Spawn one Sonnet subagent. Task: define overnight window per day per
market (UTC, per brief spec), extract Binance minute klines within
window, compute ~22 overnight crypto features, ~6 macro features
(1-day lag), ~4 stock-level features. Output features_track_a_hk.parquet
and features_track_a_kr.parquet.

Haiku validation: feature distributions per market (no extreme
values, no all-zero columns), row counts match expected trading days,
no lookahead (all timestamps in feature row precede market open).

STAGE 4 -- LIGHTGBM TRAINING AND PREDICTION

Spawn one Sonnet subagent. Task: walk-forward LightGBM on each
market x target (6 runs). Hyperparameter tuning per brief spec with
50-iteration randomized search, 3-fold purged time-series CV.
Apply runtime adjustment if > 10 min/fold. Save SHAP per fold.
Output predictions_lgbm_{market}_{target}.csv and
shap_per_fold_{market}_{target}.parquet and
training_log_lgbm_{market}_{target}.csv.

Haiku validation: prediction file schema (date, ticker, y_pred,
y_actual, fold_id), no duplicate rows, OOS predictions only (no
training dates).

STAGE 5 -- PORTFOLIO CONSTRUCTION AND BACKTEST

Spawn one Sonnet subagent. Task: apply tercile strategy to OOS
predictions, compute gross and net daily returns with cost model per
brief, P&L decomposition, cost sensitivity sweep (0.5x/1x/1.5x/2x),
breakeven spread analysis. Output backtest_lgbm_{market}_{target}.csv,
backtest_summary.csv, cost_sensitivity.csv.

Haiku validation: Sharpe ratios finite, turnover in plausible range,
net returns = gross returns - cost drag within rounding error.

STAGE 6 -- DIAGNOSTICS

Spawn one Sonnet subagent. Task: 3 category-level feature ablation
runs, return component analysis, regime analysis (VIX/BTC trend/crypto
vol), SHAP stability, weekend effect. Output per brief spec.

STAGE 7 -- ADVERSARIAL REVIEW PANEL (2 rounds)

ROUND 1: Spawn 4 Sonnet subagents sequentially. Each receives file
paths to all Stage 5 and Stage 6 outputs plus their adversarial stance
(Skeptic, Believer, Literature Reviewer, Practitioner per brief).
Each writes a 1-2 page position paper to reviews/{role}_position.md.

ROUND 2: Spawn 4 Sonnet subagents sequentially. Each receives the
four Round 1 position papers and writes a rebuttal to
reviews/{role}_rebuttal.md. The rebuttal must engage with the
strongest counterargument from each of the other three panelists.

SYNTHESIS: You (Opus) read all 8 documents. Write reviews/synthesis.md
addressing: (a) whether any finding survives the panel's scrutiny,
(b) the strongest unresolved objection, (c) what the next test or
Pass 2 extension should resolve.

STAGE 8 -- DOCUMENTATION

You (Opus) write README.md and WRITEUP.md after reading
reviews/synthesis.md, output/backtest_summary.csv, and
output/diagnostics_summary.txt.

README.md: project overview, data sources (note Stooq HK is
user-provided and why), methodology summary, key results table,
reproduction instructions.

WRITEUP.md: full research document. Hypothesis, data, methodology,
results, diagnostics, limitations, conclusion. Apply writing voice
directives above. Include explicit section: "What Pass 2 Will Add"
with the TCN horse race, index-level prediction, and expanded
ablation scope.

Resume bullet: 1-2 lines mapping to the resume gap keywords in the
brief. Place in README.md under a "Resume Framing" section.

---

STOPPING CONDITIONS (pause and surface to user)

- Any Stage -2, -1, or 0 blocking gate fails
- Stooq HK coverage < 60% (surface option to drop HK)
- pykrx scraper fails mid-run
- Any stage produces an output file with < 50% expected row count
- Any subagent reports an error it cannot resolve in 2 retries

---

DELIVERABLES CHECKLIST (verify at end of run)

- [ ] All Stage 4 prediction CSVs present (6 files)
- [ ] output/backtest_summary.csv with 6 rows (one per market x target)
- [ ] output/feature_ablation.csv with 3 ablation runs per market x target
- [ ] reviews/ contains 8 .md files + synthesis.md
- [ ] README.md and WRITEUP.md written
- [ ] Resume bullet in README.md
- [ ] logs/ contains per-stage logs
- [ ] results_ledger.md updated with Pass 1 entry

Begin with Stage -2.