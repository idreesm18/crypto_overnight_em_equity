# Research Brief: Crypto Overnight Signal for Asian Equity Markets (Pass 1)
# Project name: crypto_overnight_signal
# Pass: 1 of 2

---

## Hypothesis

Crypto markets trade 24/7; Asian equity markets do not. Between HKEX and
KRX equity close and next-day open, ~17 hours of crypto trading occur.
Information flowing through crypto overnight (risk appetite, dollar demand,
liquidation stress) may not be fully priced into crypto-exposed equities
at the open, creating a predictable component in next-day returns.

This pass tests the hypothesis using LightGBM on hand-engineered overnight
crypto features, applied to crypto-exposed stock universes in Hong Kong
and South Korea. Pass 2 adds a TCN deep learning model, index-level
prediction, and expanded diagnostics.

---

## Resume Gaps Closed

- ML for return prediction: LightGBM with walk-forward CV, SHAP, feature ablation
- Systematic strategy design: explicit long/short rules from model output, daily rebalance
- Cross-asset signal: crypto features predicting equity returns across two markets
- Portfolio construction: long/short portfolio with transaction cost modeling

---

## Prediction Targets

Three targets per stock per day. Same feature set; only the dependent
variable changes.

  1. Close-to-close return: R_cc(t) = close(t) / close(t-1) - 1
  2. Overnight gap return (PRIMARY): R_gap(t) = open(t) / close(t-1) - 1
  3. Intraday return: R_intra(t) = close(t) / open(t) - 1

The decomposition tests where overnight crypto information gets
incorporated. Gap predictability without intraday predictability
confirms pricing at the open.

---

## Universe Construction

Two-layer selection, applied at each monthly rebalance (first trading
day of each month):

  Layer 1 (static, pre-pipeline): User-provided candidate CSVs of 40-60
  stocks per market with fundamental crypto exposure (exchanges, miners,
  hardware, blockchain infra, fintech with crypto custody/trading,
  companies holding BTC on balance sheet).

  Files: data/crypto_candidates_hk.csv, data/crypto_candidates_kr.csv
  Columns: ticker, company_name, category, source
  BLOCKING: Pipeline cannot start without these files.

  Layer 2 (dynamic, per-rebalance):
  1. Compute trailing 60-day Pearson correlation of each candidate's
     daily return with BTC daily return.
  2. Liquidity filter: trailing 20-day ADV > $500K USD equivalent.
  3. Rank by BTC correlation within each market. Select top 20-30.
  4. Log to output/universe_log.csv: date, market, ticker, btc_corr,
     adv, rank.

  No lookahead: correlation and ADV use only data available as of the
  rebalance date.

  Acknowledged circularity: selecting on BTC correlation then predicting
  with BTC-derived features is mildly tautological. Standard practice in
  cross-asset signal research. State transparently in writeup.

---

## Data Sources

### Source 1: Binance Bulk Downloads (data.binance.vision)
  Content: BTC, ETH, SOL, BNB, XRP spot 1-min klines; BTC/ETH perp
  1-min klines; 8-hourly funding rates; event-level liquidations.
  Period: Spot from 2017; perps from 2020.
  Format: CSV zipped by day.
  Auth: None.
  Quirks: Occasional missing minutes during maintenance. Forward-fill
  gaps <= 5 min; flag and exclude days with gaps > 30 min.

### Source 2: Stooq HK Data (USER-PROVIDED, already in data/)
  Content: HKEX daily OHLCV.
  Location: data/stooq/hk_daily/ (user has placed files here manually
  before pipeline start). If files are elsewhere in data/, Stage -1
  or Stage 0 must locate them and either move to data/stooq/hk_daily/
  or update downstream path references.
  Format: .txt files (treat as CSV).
  Auth: N/A (pre-downloaded).
  Stage 1 does NOT re-download Stooq HK. Stage 0 still validates
  coverage against known HKEX listing counts (~2,500 as of 2024).
  If coverage < 60% in any year, STOP and surface to user.
  The tickers might be different, but that would be because
  Stooq adds a suffix. Clean them to match the actual tickers
  if that is the case.

### Source 3: pykrx (Python package, scrapes KRX website)

  Content: KOSPI + KOSDAQ daily OHLCV + market cap.
  Frequency: Daily.
  Period: 2015-present (reliable).
  Auth: None required, but KRX geo-blocks US IPs. Pipeline opens an
        SSH SOCKS5 tunnel through an Oracle Cloud Seoul VM
        automatically via scripts/krx_tunnel.py (see Operational
        Setup below).
  Size: ~250 MB in parquet.
  Known quirks:
    - Scraper may break if KRX changes its website.
    - Rate-limit requests to 1 per second (independent of transport).
    - Some tickers have missing days: forward-fill only if gap <= 5
      trading days; longer gaps indicate a trading halt or data issue
      and should be left as NaN.
    - Columns return in Korean. Rename in code:
        시가 -> open, 고가 -> high, 저가 -> low, 종가 -> close,
        거래량 -> volume, 등락률 -> pct_change
  FRAGILE. Stage 0 must validate row counts and date coverage against
  known benchmarks.
  Status: NOT YET DOWNLOADED. Stage 1 must pull this.

#### Operational Setup: KRX Geo-Block Workaround

  User prerequisites (done once, not managed by pipeline):
    - Oracle Cloud Always Free VM provisioned in Chuncheon, South Korea
    - SSH key pair at ~/.ssh/oracle_seoul
    - SSH config entry in ~/.ssh/config:

        Host oracle-seoul
            HostName <vps-public-ip>
            User ubuntu
            IdentityFile ~/.ssh/oracle_seoul
            ServerAliveInterval 60
            ServerAliveCountMax 3

    - Dependencies: pip install "requests[socks]" pykrx

  Pipeline responsibilities (automated via scripts/krx_tunnel.py):
    - Open SSH SOCKS5 tunnel on localhost:1080 before any pykrx call
    - Verify tunnel exits through KR before proceeding (fail loudly if not)
    - Set HTTP_PROXY and HTTPS_PROXY env vars for the duration of pykrx use
    - Tear down tunnel and unset env vars when pykrx work completes

  Required usage pattern in any script that imports pykrx:

    from krx_tunnel import krx_tunnel

    with krx_tunnel():
        from pykrx import stock  # MUST be inside the with block
        # ... pykrx calls here

  Import order note: pykrx reads proxy env vars at import time, so
  `from pykrx import stock` must happen inside the `with krx_tunnel():`
  block. Module-level pykrx imports will not pick up the proxy config.

  The socks5h:// scheme (not socks5://) routes DNS through the tunnel
  too. KRX domain resolution itself appears to be geo-sensitive.

### Source 4: FRED API (fred.stlouisfed.org)
  Series: VIXCLS, DTWEXBGS, DGS10, DGS2, DFF, T5YIE, T10YIE.
  Auth: Free API key (FRED_API_KEY in .env).
  Quirks: Missing values on holidays/weekends. Forward-fill.

### Source 5: yfinance
  Content: BTC-USD daily (for universe correlation filter).
  Auth: None.
  Quirks: Unofficial; cross-check against known values on 3-5 dates.

---

## Overnight Window Definition (UTC)

  Hong Kong:
    Close: 08:00 UTC (16:00 HKT)
    Next open: 01:30 UTC+1 day (09:30 HKT)
    Window: 08:00 UTC day T to 01:30 UTC day T+1 (~17.5 hours)

  South Korea:
    Close: 06:30 UTC (15:30 KST)
    Next open: 00:00 UTC+1 day (09:00 KST)
    Window: 06:30 UTC day T to 00:00 UTC day T+1 (~17.5 hours)

  All Binance timestamps are UTC. Feature engineering must convert
  exchange-local times to UTC and extract the correct minute bars.

  Edge cases:
  - Exchange holidays: skip. Use exchange_calendars package.
  - HKEX half-days (close 12:00 HKT = 04:00 UTC): use actual close
    if calendar available; otherwise flag as limitation.
  - Weekends: Friday close to Monday open (~65 hours). Include but
    flag (column: is_weekend_gap).

  Log ambiguous/excluded dates to output/overnight_window_log.csv.

---

## Feature Construction (Track A only)

All features computed from Binance minute klines within the overnight
window. No feature uses data from after equity market opens.

  Overnight crypto features (~22):
  - BTC overnight return (log)
  - ETH overnight return (log)
  - BTC overnight realized vol (annualized, from 1-min returns)
  - ETH overnight realized vol
  - BTC overnight max drawdown (peak-to-trough within window)
  - BTC overnight volume (USD notional)
  - BTC volume surge ratio: overnight vol / trailing 7-day avg overnight vol
  - BTC taker buy/sell imbalance: (buy_vol - sell_vol) / total_vol
    (requires taker buy base asset volume from klines)
  - Cross-pair momentum dispersion: stdev of overnight returns across
    BTC, ETH, SOL, BNB, XRP
  - BTC-ETH overnight return spread
  - BTC perp funding rate (most recent 8-hour reset before equity open)
  - BTC perp funding rate change (current minus prior reset)
  - BTC perp liquidation intensity: count and USD notional within window
  - USDT peg deviation: USDT/USD deviation from $1.00 at equity open.
    Source: Binance USDTUSD or implied from BTC/USDT vs BTC/USD.
    If unavailable, drop and note absence.

  Macro conditioning features (daily, 1-day lag, ~6):
  - VIX level
  - VIX 5-day change
  - Yield curve slope (DGS10 - DGS2)
  - DXY level
  - DXY 5-day change
  - 5Y breakeven inflation (T5YIE)

  Stock-level features (~4):
  - Trailing 20-day realized vol
  - Trailing 20-day return (momentum)
  - Log market cap bucket (small/mid/large tercile)
  - Prior-day return

  Total: ~32 features.

  Restricted fallback set (7 features, run alongside full set):
  BTC overnight return, BTC overnight vol, funding rate level,
  liquidation intensity, VIX level, yield curve slope, prior-day
  stock return.

---

## Model Specification: LightGBM

  Input: Track A features (flat table, one row per stock-day).
  Target: Three separate models per market (gap, intraday, close-to-close).
  Walk-forward: Expanding window. 252-day minimum training period.
  Monthly rebalance. At each fold, train on all data up to rebalance
  date, predict the next month OOS.
  Total model runs: 1 model x 3 targets x 2 markets = 6.

  Hyperparameter search:
    n_estimators: [100, 300, 500]
    max_depth: [3, 5, 7]
    learning_rate: [0.01, 0.05, 0.1]
    min_child_samples: [10, 20, 50]
    subsample: [0.7, 0.8, 1.0]
    colsample_bytree: [0.7, 0.8, 1.0]
    reg_alpha: [0, 0.1, 1.0]
    reg_lambda: [0, 0.1, 1.0]

  Tuning: Randomized search, 50 iterations, 3-fold time-series CV on
  training set (purged, no lookahead).

  RUNTIME ADJUSTMENT: If > 10 min/fold, reduce to 20 iterations. If
  still slow, fix learning_rate=0.05, max_depth=5, search only
  n_estimators and regularization.

  SHAP: Computed on each fold's training set. Save to
  output/shap_per_fold_{market}_{target}.parquet.

  Evaluation metrics per fold:
  - Directional accuracy
  - Information coefficient (Spearman rank correlation, cross-sectional)
  - MSE and MAE

---

## Strategy Layer

  For each market (HK, KR) separately, at each daily rebalance:
  1. Generate predictions for all stocks in current universe.
  2. Sort by predicted close-to-close return.
  3. Long top tercile, short bottom tercile, equal-weight within legs.
  4. Hold one day.
  5. If N < 9 stocks on a given day, go flat.

### Transaction Cost Model

  Half-spread per trade:
  - HK stocks: 15 bps/side (small-cap, lower liquidity)
  - KR stocks: 10 bps/side

  Market impact (stocks only):
    Impact = 0.1 * sqrt(trade_size / ADV) * daily_vol
    trade_size = $100K per position; ADV = trailing 20-day avg daily
    volume (USD); daily_vol = trailing 20-day realized daily vol.

  Total cost per trade = spread + impact. Apply to entry and exit.

  Cost sensitivity sweep: 0.5x, 1x, 1.5x, 2x default spread.
  Breakeven spread analysis: at what spread does net Sharpe = 0?

### Performance Metrics
  Per market, per target: gross Sharpe, net Sharpe, annualized return,
  max drawdown, avg daily turnover, Calmar ratio, % days invested.

---

## Pipeline Stages

### Stage -2: Project File Structure (pre-flight, idempotent)

The orchestrator creates the following directory structure at the
project root before any other work. Create directories only if they
do not exist; do not overwrite anything.
crypto_overnight_signal/
├── .venv/                            (created in Stage -1)
├── .env                              (USER-PROVIDED: FRED_API_KEY)
├── requirements.txt                  (created in Stage -1)
├── data/
│   ├── binance/
│   │   ├── spot_klines/
│   │   ├── perp_klines/
│   │   ├── funding_rates/
│   │   └── liquidations/
│   ├── stooq/hk_daily/               (USER-PROVIDED: .txt files already here)
│   ├── pykrx/kr_daily/
│   ├── fred/
│   ├── yfinance/
│   ├── crypto_candidates_hk.csv      (USER-PROVIDED)
│   └── crypto_candidates_kr.csv      (USER-PROVIDED)
├── scripts/                          (pipeline scripts written per stage)
├── output/                           (all CSV/parquet outputs)
├── reviews/                          (review panel documents)
├── logs/                             (per-stage logs)
├── README.md                         (written in Stage 8)
└── WRITEUP.md                        (written in Stage 8)
Verification after creation:

data/stooq/hk_daily/ is non-empty (contains .txt files).
data/crypto_candidates_hk.csv and data/crypto_candidates_kr.csv
exist and have header: ticker, company_name, category, source.
.env exists at project root.

If any USER-PROVIDED item is missing, STOP and surface to user before
proceeding to Stage -1.

### Stage -1: Environment Setup (first agent action)

Before any data pull, the orchestrator creates an isolated Python
environment at the project root.

  1. Verify Python 3.11 is available:
       python3.11 --version
     If not found, STOP and surface to user.
  2. Create venv at .venv using Python 3.11 explicitly:
       python3.11 -m venv .venv
  3. Activate and upgrade pip:
       source .venv/bin/activate
       pip install --upgrade pip
  4. Verify interpreter is 3.11 inside the venv:
       python --version
     Must report 3.11.x. If not, STOP and surface to user.
  5. Install pinned dependencies from requirements.txt:
       lightgbm>=4.0
       numpy>=1.24
       pandas>=2.0
       scipy>=1.10
       shap>=0.44
       scikit-learn>=1.3
       pykrx>=1.0.45
       yfinance>=0.2.40
       fredapi>=0.5
       requests>=2.31
       pyarrow>=14.0
       exchange_calendars>=4.5
       python-dotenv>=1.0
  6. Verify .env file exists at project root with FRED_API_KEY set.
     If missing, STOP and surface to user.
  7. Smoke test pykrx: run a minimal query for a known ticker (e.g.,
     005930 Samsung Electronics, 1 week of data). If the scraper
     returns empty or errors, STOP and surface to user. This is the
     earliest possible detection point for KRX website changes that
     break pykrx.
  8. Write env log to logs/stage_-1_env.log listing Python version
     and installed package versions.

  All subsequent stages must run inside the .venv. Subagents invoke
  scripts via: source .venv/bin/activate && python scripts/...

### Stage 0: Data Validation (blocking gate)
  1. Validate user-provided Stooq HK data at data/stooq/hk_daily/:
     directory exists and contains .txt files; count unique tickers
     per year vs ~2,500 expected. If < 60% coverage in any year, STOP.
     Report ticker format (with/without leading zeros, suffix convention)
     to inform candidate CSV matching in step 4.
  2. Validate pykrx KR: unique tickers/year vs ~2,500 expected.
     Date range 2018-present, < 5% missing trading days. If fail, STOP.
  3. Validate Binance minute klines: date coverage, missing minutes/day,
     flag days > 30 min missing.
  4. Validate crypto candidate CSVs exist and tickers match price data.
     Normalize ticker format between candidate CSV and Stooq HK files
     if conventions differ. Report unmatched tickers.
  5. Cross-check yfinance BTC-USD on 3-5 reference dates.
  OUT: output/stage0_validation.txt

### Stage 1: Data Pull
  Download and cache all raw data to parquet:
    data/binance/spot_klines/
    data/binance/perp_klines/
    data/binance/funding_rates/
    data/binance/liquidations/
    data/pykrx/kr_daily/
    data/fred/
    data/yfinance/
  Stooq HK data is USER-PROVIDED at data/stooq/hk_daily/ and is NOT
  downloaded by this stage. Stage 1 must verify the directory exists
  and contains .txt files before proceeding; otherwise STOP.
  OUT: data/**/*.parquet, logs/stage1_pull.log

### Stage 2: Universe Construction
  Monthly rebalance. Correlation + liquidity filter on candidate CSVs.
  OUT: output/universe_log.csv, output/universe_summary.txt

### Stage 3: Feature Engineering (Track A only)
  1. Define overnight window per day per market (UTC).
  2. Extract Binance minute klines within window.
  3. Compute all Track A features.
  4. Merge with lagged macro features (1-day lag).
  5. Merge with stock-level features.
  OUT: output/features_track_a_hk.parquet,
       output/features_track_a_kr.parquet

### Stage 4: Model Training and Prediction
  LightGBM walk-forward on each market x target (6 runs).
  Hyperparameter tuning per fold. SHAP per fold. OOS predictions saved.
  OUT: output/predictions_lgbm_{market}_{target}.csv
       output/shap_per_fold_{market}_{target}.parquet
       output/training_log_lgbm_{market}_{target}.csv
  Columns: date, ticker, y_pred, y_actual, fold_id

### Stage 5: Portfolio Construction and Backtest
  1. Apply tercile strategy rules to OOS predictions.
  2. Daily portfolio returns, gross and net of costs.
  3. P&L decomposition: alpha return, cost drag, turnover.
  4. Cost sensitivity sweep (0.5x, 1x, 1.5x, 2x).
  5. Breakeven spread analysis.
  OUT: output/backtest_lgbm_{market}_{target}.csv
       output/backtest_summary.csv
       output/cost_sensitivity.csv

### Stage 6: Diagnostics

  6A. Feature Ablation (3 category-level runs per market x target):
    (1) Drop all crypto overnight features.
    (2) Drop all macro features.
    (3) Drop all stock-level features.
    Compare OOS Sharpe across runs.
    OUT: output/feature_ablation.csv

  6B. Return Component Analysis:
    Compare model performance across gap, intraday, close-to-close.
    IC and directional accuracy for each.
    OUT: output/return_decomposition.csv

  6C. Regime Analysis:
    Split by: VIX high/low (threshold 25), BTC trending up/down
    (trailing 30-day return sign), crypto vol regime (trailing 30-day
    BTC vol above/below median).
    OUT: output/regime_analysis.csv

  6D. SHAP Stability:
    Top-10 feature rankings by fold. Stable = signal; rotating = noise.
    OUT: output/shap_stability.csv

  6E. Weekend Effect:
    Monday (long overnight) vs Tuesday-Friday performance comparison.
    OUT: output/weekend_effect.csv

  OUT: output/diagnostics_summary.txt

### Stage 7: Adversarial Review Panel (2 rounds)

  Panel (4 agents, each receives file paths to output CSVs):

    The Skeptic: overfitting, data snooping, universe selection
    circularity, multiple testing across 6 model runs, Bonferroni.

    The Believer: economic mechanism (timezone gap), cross-market
    consistency, signal strength in predicted subsamples.

    The Literature Reviewer: crypto spillover literature, overnight
    return predictability, cross-asset information transmission.

    The Practitioner: capacity, execution (short availability in
    HK/KR, borrow costs), signal decay, net-of-cost viability.

  ROUND 1: Position papers (1-2 pages each).
  ROUND 2: Each reads all 4 papers, writes rebuttal.

  SYNTHESIS: Opus reads all 8 documents. Writes synthesis.md addressing:
  (a) whether any finding survives scrutiny, (b) strongest unresolved
  objection, (c) what Pass 2 or next test should resolve.

  OUT: reviews/{skeptic,believer,literature,practitioner}_position.md
       reviews/{skeptic,believer,literature,practitioner}_rebuttal.md
       reviews/synthesis.md

### Stage 8: Documentation

  Opus writes after synthesis:
  - README.md: overview, data sources, methodology, key results, repro.
  - WRITEUP.md: full research document. Hedged junior-researcher voice.
    Avoid triple-parallel structures, self-announcing topic sentences,
    uniform sentence rhythm.
  - Resume bullet: 1-2 lines mapping to resume gap keywords.
  - Explicit section noting what Pass 2 will add (TCN, index, expanded
    ablation).

---

## Acceptance Criteria

The project produces a finding regardless of signal existence. Three
conditions, any one constituting a positive result:

  1. LightGBM produces OOS IC > 0.03 (p < 0.05 via bootstrap) on
     close-to-close returns in at least one market.

  2. Return component analysis shows significantly higher predictability
     for gap vs intraday (IC difference > 0.02).

  3. Feature ablation identifies a feature group whose removal degrades
     OOS Sharpe by > 0.15, isolating the information channel.

If none hold: well-constructed null. The strategy infrastructure, feature
ablation, and return decomposition are deliverables.

---

## Dependencies

  Handled by Stage -1 (environment setup). Full list pinned in
  requirements.txt and installed into .venv. Core packages:

  lightgbm, numpy, pandas, scipy, shap, scikit-learn, pykrx,
  yfinance, fredapi, requests, pyarrow, exchange_calendars,
  python-dotenv

  Environment: .env at project root with FRED_API_KEY.

  No Modal. No torch. No GPU.

---

## Known Risks

  1. Signal may not exist. Mitigated by: infrastructure and ablation
     are deliverables regardless.
  2. pykrx fragility. Mitigated by: Stage 0 gate.
  3. Stooq HK coverage. User has manually placed Stooq HK data at
     data/stooq/hk_daily/. Stage 0 still validates coverage against
     60% threshold. If fails, HK analysis drops and brief becomes KR-only.
     Additional risk: ticker format mismatch between user-provided Stooq
     files and user-provided candidate CSV. Stage 0 step 4 normalizes.
  4. Overfitting: 32 features on thin universe. Mitigated by: walk-forward,
     restricted feature set fallback, ablation, review panel.
  5. Thin universe (20-30 stocks/market). Tercile sorts on 20 are noisy.
  6. Short availability in HK/KR. Practitioner panelist flags this.
  7. USDT peg data may be unavailable. Drop feature and note if so.
  8. User must provide crypto candidate CSVs before pipeline starts.
     This is manual research work outside the pipeline scope.