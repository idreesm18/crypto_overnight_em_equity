# Crypto Overnight Signal for Asian Equities

## Overview

Crypto markets trade 24/7; Asian equity markets do not. Between the HKEX or KRX close and the next day's open, roughly 17 hours of crypto trading occur. The project tests whether overnight crypto activity carries information that is not yet priced into crypto-exposed Asian equities at the open.

Scope: LightGBM and a TCN horse race on hand-engineered features and raw minute-bar sequences, four universes in parallel (main crypto-filtered HK and KR, non-crypto-filtered control HK and KR, KR KOSPI large-cap, index HSI and KOSPI), three targets per observation (overnight gap, intraday, close-to-close), walk-forward OOS evaluation 2020-02 through 2026-04 (75 monthly folds per market-target), with borrow cost model, regime gate, long-only variant, expanded feature ablation, and a three-round adversarial review panel.

A narrower initial version (main universe only, LightGBM, no control universe, no TCN, no regime gate) lives on the `p1` branch as a reference.

## Data sources

| Source | Content | Usage | Notes |
|---|---|---|---|
| Binance bulk archive | Spot and perp 1m klines for BTC/ETH/SOL/BNB/XRP; 8-hourly funding | Crypto overnight features | No auth. Liquidation snapshots unavailable; that feature stays dropped. |
| Stooq HK | Daily OHLCV for HKEX | HK equity returns and targets | User-provided. 2,869 .txt files at `data/stooq/hk_daily/`. |
| pykrx (v1.0.51) | KRX daily OHLCV | KR equity returns and targets | SSH SOCKS5 tunnel to Oracle Cloud Seoul VM. Batch endpoint login-gated; per-ticker endpoint used. KRX mcap endpoint authenticated-only; mcap feature stays dropped. |
| FRED | VIX, DXY, 2Y/10Y, 5Y breakeven, DFF | Macro features with 1-day lag | Free API key. |
| yfinance indices | ^HSI, ^KS11 daily OHLCV | Index-level prediction targets | No auth. |
| yfinance SP500 | ^GSPC daily close | Market-cap proxy for regime gate | No auth. |
| Derived BTC supply | Deterministic issuance schedule | Regime gate | Calculated, not pulled. |
| Crypto candidates (user-provided) | 40 HK + 39 KR tickers with fundamental crypto exposure | Main universe layer 1 | CSVs at `data/stock_picks/crypto_candidates_{hk,kr}.csv`. |
| Crypto controls (user-provided) | 42 HK + 39 KR tickers, non-crypto-exposed, matched on size and liquidity | Control universe layer 1 | CSVs at `data/stock_picks/crypto_control_{hk,kr}.csv`. |

User-provided data is the four candidate CSVs, Stooq HK .txt files, and a FRED_API_KEY. The user maintains an Oracle Cloud Seoul VM for the KRX tunnel. Modal credentials for the TCN stage are read from `~/.modal.toml`.

## Methodology summary

1. Universes. Four universes run in parallel. Main (crypto-filtered HK and KR, 88 monthly rebalances, median 24 HK and 30 KR tickers). Control (same ADV filter, no BTC correlation filter, median pool 30). KR KOSPI large-cap ($50M ADV on .KS tickers, flagged inconclusive at 94 percent flat months per brief risk #3). Index (^HSI and ^KS11, 1,793 and 1,790 trading days).

2. Features. Track A, 21 hand-engineered daily-summary features for stock universes (12 crypto overnight + 6 macro + 3 stock-level) and 18 for indices. Track B, raw minute-bar sequences for the TCN, 5-minute resampling, per-window per-channel normalization, boolean mask for padding.

3. Models. LightGBM walk-forward (expanding window, 252-day minimum training, monthly rebalance, 3-fold purged time-series CV, 5-day purge buffer, 10-iteration randomized hyperparameter search, 18 total runs across universes and targets). TCN on Modal T4 with per-fold checkpointing, 12 runs across four universes and three targets. Horse race with paired block-bootstrap.

4. Strategies. Stock tercile long-short (main, control) and binary split (KOSPI large-cap, scoped but inconclusive). Long-only variant for every stock universe. Index threshold long-short-flat using 0.5 times in-sample stdev of y_pred.

5. Costs. Per-ticker per-day half-spreads from the Corwin–Schultz (2012) high-low estimator (20-day trailing mean; market-median fallback; fixed-assumption fallback during the initial warmup), Kyle-style impact for stocks, annualized borrow on short notional, and a regime gate toggle. Spread and borrow sensitivity sweeps (now applied to the CS estimate as an estimator-accuracy stress), breakeven analysis. The original fixed-spread backtest is retained alongside the CS-spread backtest for comparison (`output/backtest_summary_pass2.csv` vs `output/backtest_summary_pass2_cs.csv`).

6. Regime gate. BTC market cap divided by S&P 500 market-cap proxy, each 30-day mean, compared to 1-year trailing median. Warmup 282 days defaults to active. Every backtest runs gate-on and gate-off.

7. Diagnostics. Horse race, control vs main comparison (the central control-universe result), regime-gate comparison, long-only vs long-short decomposition, regime splits (VIX, BTC trend, crypto vol), return decomposition, expanded ablation (category, subcategory, per-feature leave-one-out).

8. Review. Four-panelist adversarial review, three rounds (position, first rebuttal, second rebuttal), orchestrator synthesis. Twelve review files plus `reviews/synthesis_pass2.md`.

## Key results

Walk-forward OOS, 2020-02 through 2026-04, 75 monthly folds per market-target.

### Main universe headline

| Market | Target | Mean IC | Bootstrap p | Gross SR | Net SR (1x cost) | Breakeven cost multiplier |
|---|---|---|---|---|---|---|
| HK | gap | 0.061 | < 0.001 | 3.04 | -1.57 | 0.66x |
| HK | intraday | 0.020 | < 0.001 | 0.35 | -2.99 | 0.10x |
| HK | cc | 0.014 | 0.036 | 0.98 | -1.25 | 0.43x |
| KR | gap | 0.061 | < 0.001 | 3.77 | -0.28 | 0.93x |
| KR | intraday | 0.023 | 0.019 | 1.00 | -1.46 | 0.41x |
| KR | cc | 0.008 | 0.255 | 0.59 | -1.58 | 0.27x |

### Control versus main comparison (central control-universe result)

| Market | Main gap IC | Control gap IC | Diff | Bootstrap p | Main ratio | Control ratio |
|---|---|---|---|---|---|---|
| HK | 0.061 | 0.051 | +0.010 | 0.93 | 3.14 | 4.80 |
| KR | 0.061 | 0.037 | +0.024 | 0.008 | 2.15 | 0.88 |

HK: gap dominance is not crypto-specific in this sample; the control ratio exceeds the main ratio. KR: an incremental crypto channel is identified, statistically significant, magnitude ~40 percent of raw main IC. Applying the brief's decision rule, Rule D (mixed evidence) is the best-fitting framing for both markets.

### Backtest headlines

The cost model uses **CS spreads for stock universes** (where the estimator is above its noise floor) and **exchange-tick-floor realistic spreads for index ETFs** (where CS is at the noise floor and overstates the true spread per the peer-reviewed literature). `output/backtest_summary_pass2_cs.csv` is the primary backtest summary; the Pass 2 fixed-spread file (`output/backtest_summary_pass2.csv`) remains on disk for audit only.

Realistic index spreads applied in the backtest:
- **2800.HK**: 15 bps round-trip pre-2020-06-01, 8 bps round-trip post-reform, per HKEX's June 2020 ETP spread-table reform [HKEX-ETP, HKEX-Min] and continuous-SMM obligations. 8 bps is one ETP tick (HKD 0.02) at 2800.HK's HKD 26 price level.
- **069500.KS**: 5 bps round-trip uniform across 2019-2026, per KRX's flat 5-KRW ETF tick at KODEX 200's ~33,500 KRW price level plus LP spread-obligation rules.

**Stock universes: not deployable.** Every stock-tercile strategy (main or control, long-short or long-only, gate-on or gate-off, LightGBM or TCN) is net-negative under CS spreads at 1x. No point in the 0.5x-2x spread sweep rescues a stock universe. All four stock universes are empirically CS-covered: Stage P2-18 pulled daily OHLCV for all 38 control_kr tickers via pykrx (67,603 rows, 0 skips). The Rule-D IC findings (control vs main) remain valid research contributions but do not translate into a tradable single-stock strategy.

**Index universes: seven deployable configurations.** Under tick-floor realistic spreads at 1x spread and 1x borrow:

| Configuration | net Sharpe |
|---|---:|
| LightGBM index_kr gap gate_off | 3.64 |
| LightGBM index_kr gap gate_on  | 2.60 |
| LightGBM index_hk gap gate_off | 1.96 |
| LightGBM index_hk gap gate_on  | 1.19 |
| LightGBM index_kr cc  gate_off | 1.10 |
| LightGBM index_kr cc  gate_on  | 0.85 |
| LightGBM index_hk cc  gate_off | 0.55 |

Two TCN configurations are positive but sub-threshold (index_hk cc gate_on 0.44, gate_off 0.16). The index fixed-vs-CS comparison in `output/cs_vs_fixed_comparison.csv` covers all 24 index configurations.

**Why the revision from the earlier CS noise-floor result.** Applying CS to index ETFs (HSI_proxy and 069500.KS) produced figures of 25-30 bps round-trip. Those sit at the estimator's noise floor (raw-negative fractions 53% and 52%) and overstate true effective spreads by 3-5× per Corwin-Schultz (2012, Table 6: 18% correlation with true spread for large-caps) and Tremacoldi-Rossi & Irwin (2022 JFQA: high raw-negative fraction is a direct noise-floor indicator with downward-truncation bias on the floored 20d mean). Tick-structure-derived spreads replace the CS values for indexes only; stock CS stays.

### Horse race

LightGBM significantly outperforms TCN on main-universe gap IC: HK IC diff -0.021 (p=0.006), KR IC diff -0.028 (p<0.001). Every TCN fold shows an overfit flag. The gap signal lives in daily summary features, not minute-bar temporal structure.

### Acceptance criteria

Criterion 1 (index gap IC > 0.03, p < 0.05): PASS (HK rolling time-series IC 0.185, KR 0.277, both p < 0.001).
Criterion 2 (KR KOSPI large-cap net Sharpe > 0.5): SKIPPED per brief risk #3.
Criterion 3 (TCN > LightGBM by 0.01 IC): FAIL (LightGBM wins).
Criterion 4 (control < 1.5 AND main >= 2.5 ratio in same market): PARTIAL (KR direction correct, magnitude below threshold; HK control higher than main).

See `reviews/synthesis_pass2.md` for the synthesis of all twelve panel documents.

## Reproduction instructions

Prerequisites:
- Python 3.11 on the host
- `~/.ssh/oracle_seoul` key and SSH config `oracle-seoul` pointing to a KR-exit VM (required for pykrx)
- A FRED API key
- Modal credentials at `~/.modal.toml` (required for the TCN stage)
- User-provided control CSVs at `data/stock_picks/crypto_control_{hk,kr}.csv`

```
cd /path/to/crypto_overnight_em_equity
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install "requests[socks]"

# Place Stooq HK .txt files under data/stooq/hk_daily/
# Place user-provided candidate CSVs at data/stock_picks/
# Create .env with FRED_API_KEY=...

# Pipeline scripts (run in order):
python scripts/stage0a_validate.py
python scripts/stage1_pull_fred.py
python scripts/stage1_pull_yfinance.py
python scripts/stage1_pull_binance.py
python scripts/stage1_pull_pykrx.py
python scripts/stage2_universe.py
python scripts/stage3_features.py
python scripts/stage4_lightgbm_hk.py
python scripts/stage4_lightgbm_kr.py
python scripts/stage5_backtest.py
python scripts/stage6_diagnostics.py

# Extended pipeline stages:
python scripts/stage_p2-2_pull.py
python scripts/stage_p2-3_control_universe.py
python scripts/stage_p2-4_kospi_largecap.py      # Produces inconclusive universe log; downstream skips large-cap.
python scripts/stage_p2-5_index_features.py
python scripts/stage_p2-6_sequence_prep.py
python scripts/stage_p2-7_lightgbm.py            # 12 LightGBM runs (control + index)
modal run --detach scripts/stage_p2-8_tcn_modal.py  # 12 TCN runs on Modal T4
python scripts/stage_p2-9_backtest.py
python scripts/stage_p2-10_ablation.py
python scripts/stage_p2-11_diagnostics.py

# CS-spread cost-model update (writes *_cs.csv outputs alongside the Pass 2 baseline):
python scripts/stage_p2-17_pull_kodex.py         # Pull 069500.KS (KODEX 200) real ETF OHLC via pykrx
python scripts/stage_p2-18_pull_control_kr_ohlc.py  # Pull OHLCV for all 38 control_kr tickers via pykrx
python scripts/stage_p2-15_cs_spread.py          # Corwin–Schultz per-ticker per-day spreads (unions main+control KR)
python scripts/stage_p2-16_backtest_cs.py        # Re-runs the backtest matrix using CS spreads

# Notebooks (executed cleanly):
for i in 01 02 03 04 05 06 07; do
  jupyter nbconvert --to notebook --execute --inplace notebooks/${i}_pass2_*.ipynb
done
```

Review papers (twelve panel files plus `reviews/synthesis_pass2.md`) are authored content, not generated by a script.

## Output map

- `output/backtest_summary_pass2.csv`, `output/cost_sensitivity_pass2.csv`, `output/borrow_sensitivity_pass2.csv`, `output/breakeven_analysis_pass2.csv` (fixed-spread cost model, Pass 2 baseline)
- `output/backtest_summary_pass2_cs.csv`, `output/cost_sensitivity_pass2_cs.csv`, `output/borrow_sensitivity_pass2_cs.csv`, `output/breakeven_analysis_pass2_cs.csv` (CS-spread cost model, primary for headline verdicts; Stage P2-16 rerun with real ETF OHLC)
- `output/cs_spread.parquet`, `output/cs_spread_diagnostics.csv`, `output/cs_spread_summary.txt` (Corwin–Schultz estimator outputs; KR ETF uses real 069500.KS data)
- `data/pykrx/etfs/069500_KS_daily.parquet` (KODEX 200 daily OHLCV from pykrx, 1,796 rows, 2019-01-02 to 2026-04-24, Stage P2-17)
- `data/pykrx/kr_daily/kr_control_ohlcv.parquet` (OHLCV for 38 control_kr tickers from pykrx, 67,603 rows, 2019-01-02 to 2026-04-24, 0 skips, Stage P2-18)
- `logs/stage_p2-17_env.log` (pykrx install + pull log, Stage P2-17)
- `output/cs_vs_fixed_comparison.csv` (side-by-side CS vs fixed-spread backtest net Sharpe)
- `output/control_vs_main_comparison.csv`
- `output/horse_race.csv`, `output/horse_race_bootstrap.csv`
- `output/regime_gate_comparison.csv`, `output/long_short_decomposition.csv`
- `output/control_universe_log.csv`, `output/kospi_largecap_universe_log.csv`
- `output/features_track_a_{control_hk,control_kr,index}.parquet`
- `output/sequences_{hk,kr,index_hk,index_kr}.npz`
- `output/predictions_lgbm_{control_hk,control_kr,index_hk,index_kr}_{gap,intraday,cc}.csv`
- `output/predictions_tcn_{main_hk,main_kr,index_hk,index_kr}_{gap,intraday,cc}.csv`, `output/training_log_tcn_*.csv`
- `output/feature_ablation_pass2.csv`, `output/feature_ablation_per_feature.csv`
- `output/return_decomposition_pass2.csv`, `output/regime_analysis_pass2.csv`
- `output/diagnostics_summary_pass2.txt`
- Main-universe Pass 1 artifacts: `output/backtest_summary.csv`, `output/predictions_lgbm_{hk,kr}_*.csv`, `output/features_track_a_{hk,kr}.parquet`, `output/shap_per_fold_*.parquet`, etc.

Review documents:
- `reviews/{skeptic,believer,literature,practitioner}_position.md` and `_rebuttal.md` (initial review)
- `reviews/{skeptic,believer,literature,practitioner}_p2.md`, `_rebuttal_p2.md`, `_rebuttal2_p2.md` (three-round panel)
- `reviews/synthesis.md`, `reviews/synthesis_pass2.md`, `reviews/synthesis_cs_update.md` (CS-spread cost-model addendum)

Notebooks: `notebooks/01_pass2_data_additions.ipynb` through `notebooks/07_pass2_diagnostics.ipynb`.

- Index-ETF cost model uses tick-floor-derived realistic spreads (8 bps 2800.HK post-2020-reform, 15 bps pre-reform; 5 bps 069500.KS uniform) rather than CS, because the CS estimator is at its noise floor for both index instruments (raw-negative fractions 53% and 52% — the 50%+ threshold flagged by Tremacoldi-Rossi & Irwin 2022 as indicating estimator failure). A live paper trade with executed-spread tracking would convert the tick-derived bound into a measured point estimate.
- All 38 control_kr tickers now have empirical CS coverage (Stage P2-18, 67,603 rows, median 80.5 bps round-trip). Under real CS, control_kr is net-negative across all 12 configurations. The prior control_kr coverage gap is resolved.
- Raw-negative fraction for stock markets is ~41%, above the 35% heuristic in the CS brief but consistent with liquid large-cap regimes where the estimator hits its signal-to-noise floor. The overnight adjustment is correctly ordered (the floor would be ~51% without it). Stock CS verdicts are robust; the caveat lives in interpretation.
- A 30-day live paper trade with executed-spread tracking on KR index futures is the right next step to settle whether the 1.08 net Sharpe on LightGBM index_kr gap gate_off under CS (real ETF OHLC) is representative of live execution.
- Regime gate reduces Sharpe in this sample rather than improving it. The BTC/S&P 500 market-cap ratio versus 1-year median is reported as a sensitivity, not a validated ingredient.
- Year-by-year index tearsheet (`output/index_yearly_tearsheet.csv`) shows the KR 2023 attenuation reproduces at the index level (annual Sharpe 0.36, IC 0.09); parallels the stock-level 2023 attenuation. Decay versus transient regime cannot be discriminated with one occurrence.
- KR short-sale ban November 2023 to March 2024 is not gated in the backtest. For futures-based implementations this is immaterial.
- KOSPI large-cap variant is inconclusive (94 percent flat months at $50M ADV threshold). Acceptance criterion #2 was not testable.
- TCN significantly underperforms LightGBM and shows pervasive overfit flags. The signal in this sample lives in daily-summary features, not minute-bar temporal structure.
- Three features (log mcap bucket, BTC liquidation intensity, USDT peg deviation) remain dropped; enabling credentials were not available in this run.
- Seven years OOS covers one partial crypto cycle. Strategy untested in a sustained bear from post-ETF-era highs.
- Control universe is manually curated; the HK null (p=0.93) survives this concern, the KR result (p=0.008) is more vulnerable to it.

See `logs/feature_decisions.log` and `logs/feature_restoration_decisions.log` for the complete record of each feature-level deviation from the brief.
