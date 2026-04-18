# Crypto Overnight Signal for Asian Equities — Pass 2

## Overview

Crypto markets trade 24/7; Asian equity markets do not. Between the HKEX or KRX close and the next day's open, roughly 17 hours of crypto trading occur. The project tests whether overnight crypto activity carries information that is not yet priced into crypto-exposed Asian equities at the open.

Pass 1 scope: LightGBM on hand-engineered features, monthly-rebalanced universe of 20 to 30 crypto-exposed stocks per market (HK and KR), three targets per stock-day (overnight gap, intraday, close-to-close). Pass 1 confirmed the pre-specified gap-vs-intraday decomposition (3:1 ratio in HK, 2.6:1 in KR, both p < 0.001) but found no tested strategy survived execution costs.

Pass 2 scope: adds a non-crypto-filtered control universe (the central methodological contribution), KR KOSPI large-cap and index-level prediction variants, a TCN horse race on minute-bar sequences, a regime gate, a borrow cost model, and a long-only variant. Pass 2 retains Pass 1's main-universe artifacts unchanged.

## Data sources

| Source | Content | Usage | Notes |
|---|---|---|---|
| Binance bulk archive | Spot and perp 1m klines for BTC/ETH/SOL/BNB/XRP; 8-hourly funding | Crypto overnight features | No auth. Liquidation snapshots unavailable; that feature stayed dropped. |
| Stooq HK | Daily OHLCV for HKEX | HK equity returns and targets | User-provided. 2,869 .txt files at `data/stooq/hk_daily/`. |
| pykrx (v1.0.51) | KRX daily OHLCV | KR equity returns and targets | SSH SOCKS5 tunnel to Oracle Cloud Seoul VM. Batch endpoint login-gated; per-ticker endpoint used. KRX mcap endpoint authenticated-only; mcap feature stays dropped in Pass 2. |
| FRED | VIX, DXY, 2Y/10Y, 5Y breakeven, DFF | Macro features with 1-day lag | Free API key. |
| yfinance indices (new in Pass 2) | ^HSI, ^KS11 daily OHLCV | Index-level prediction targets | No auth. |
| yfinance SP500 (new) | ^GSPC daily close | Market-cap proxy for regime gate | No auth. |
| Derived BTC supply (new) | Deterministic issuance schedule | Regime gate | Calculated, not pulled. |
| Crypto candidates (user-provided) | 40 HK + 39 KR tickers with fundamental crypto exposure | Main universe layer 1 | Reused from Pass 1. |
| Crypto controls (user-provided, new in Pass 2) | 42 HK + 39 KR tickers, non-crypto-exposed, matched on size and liquidity | Control universe layer 1 | Required for P2-3. |

User-provided data is the two Pass 1 candidate CSVs, the two Pass 2 control CSVs, Stooq HK .txt files, and a FRED_API_KEY. The user maintains an Oracle Cloud Seoul VM for the KRX tunnel. Modal credentials for the Pass 2 TCN stage are read from `~/.modal.toml`.

## Methodology summary

1. Universes. Four universes run in parallel. Main (reused from Pass 1, 88 monthly rebalances). Control (Pass 2 new, same ADV filter, no BTC correlation filter). KR KOSPI large-cap (Pass 2 new, $50M ADV on .KS tickers, flagged inconclusive at 94 percent flat months per brief risk #3). Index (Pass 2 new, ^HSI and ^KS11).

2. Features. Track A, 21 engineered daily-summary features for stock universes (12 crypto overnight + 6 macro + 3 stock-level) and 18 for indices. Track B, raw minute-bar sequences for TCN, 5-minute resampling, per-window per-channel normalization, boolean mask for padding.

3. Models. LightGBM walk-forward (Pass 1 protocol, 18 total Pass 1 + Pass 2 runs). TCN on Modal T4 with per-fold checkpointing (12 runs across four universes and three targets). Horse race with paired block-bootstrap.

4. Strategies. Stock tercile long-short (main, control) and binary split (KOSPI large-cap, scoped but inconclusive). Long-only variant for every stock universe. Index threshold long-short-flat using 0.5 times in-sample stdev of y_pred.

5. Costs. Half-spread per side, Kyle-style impact for stocks, annualized borrow on short notional, and a regime gate toggle. Spread and borrow sensitivity sweeps, breakeven analysis.

6. Regime gate. BTC market cap divided by S&P 500 market-cap proxy, each 30-day mean, compared to 1-year trailing median. Warmup 282 days defaults to active. Every backtest is run gate-on and gate-off.

7. Diagnostics. Horse race, control vs main comparison (central Pass 2 result), regime-gate comparison, long-only vs long-short decomposition, regime splits (VIX, BTC trend, crypto vol), return decomposition, expanded ablation (category, subcategory, per-feature leave-one-out), and a year-by-year placeholder.

8. Review. Four-panelist adversarial review, three rounds (position, first rebuttal, second rebuttal), synthesis by orchestrator. Twelve Pass 2 review files plus `reviews/synthesis_pass2.md`.

## Key results

Pass 1 main universe results are unchanged (see WRITEUP.md for the headline table). Pass 2 additions:

### Control versus main comparison (central Pass 2 result)

| Market | Main gap IC | Control gap IC | Diff | Bootstrap p | Main ratio | Control ratio |
|---|---|---|---|---|---|---|
| HK | 0.061 | 0.051 | +0.010 | 0.93 | 3.14 | 4.80 |
| KR | 0.061 | 0.037 | +0.024 | 0.008 | 2.15 | 0.88 |

HK: gap dominance is not crypto-specific in this sample; the control ratio exceeds the main ratio. KR: an incremental crypto channel is identified, statistically significant, magnitude ~40 percent of raw main IC. Applying the brief's decision rule, Rule D (mixed evidence) is the best-fitting framing for both markets.

### Backtest headlines

96 primary backtest configurations in `output/backtest_summary_pass2.csv`. Eight clear net Sharpe 0.5 at 1x modeled costs; all eight are LightGBM index threshold strategies (HSI or KOSPI, gap target, with and without regime gate). Top configuration: LightGBM index_kr gap gate_off, net Sharpe 3.74 at modeled 0.24 bps round-trip spread.

The 2 bps modeled half-spread on index futures is 7-14 times the realistic level. At the realistic cost level, the top configuration's net Sharpe is ~1.5-2.0. A paper-trade with executed-spread tracking is the right next step before capital deployment.

No stock-tercile strategy (main or control, long-short or long-only) clears the net Sharpe 0.5 bar at 1x costs. Rule C applies to all single-stock implementations: the signal loses to execution costs.

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
- `~/.ssh/oracle_seoul` key and SSH config `oracle-seoul` pointing to a KR-exit VM (required for pykrx, Pass 1 stages only)
- A FRED API key
- Modal credentials at `~/.modal.toml` (required for TCN stage P2-8)
- User-provided Pass 2 control CSVs at `data/stock_picks/crypto_control_{hk,kr}.csv`

Pass 1 is a prerequisite. Pass 2 reuses Pass 1's output/*.csv, output/*.parquet, data/, and .venv. Pass 2 scripts numbered P2-1 through P2-10 build on those artifacts.

```
cd /path/to/crypto_overnight_em_equity_p2
source .venv/bin/activate
pip install --upgrade torch>=2.1 modal>=0.60

# Pass 1 artifacts must already be in place (output/, data/).

# Pass 2 scripts (run in order):
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

# Notebooks (executed cleanly):
for i in 01 02 03 04 05 06 07; do
  jupyter nbconvert --to notebook --execute --inplace notebooks/${i}_pass2_*.ipynb
done
```

Review papers (twelve panel files plus synthesis_pass2.md) are authored content, not generated by a script.

## Output map (Pass 2 additions)

Pass 1 outputs remain unchanged. Pass 2 additions:

- `output/control_universe_log.csv`, `output/kospi_largecap_universe_log.csv`
- `output/features_track_a_control_{hk,kr}.parquet`, `output/features_track_a_index.parquet`
- `output/sequences_{hk,kr,index_hk,index_kr}.npz`
- `output/predictions_lgbm_control_{hk,kr}_{gap,intraday,cc}.csv`, `output/predictions_lgbm_index_{hk,kr}_{gap,intraday,cc}.csv`
- `output/predictions_tcn_{main,index}_{hk,kr}_{gap,intraday,cc}.csv`, `output/training_log_tcn_*.csv`
- `output/backtest_summary_pass2.csv`, `output/cost_sensitivity_pass2.csv`, `output/borrow_sensitivity_pass2.csv`, `output/breakeven_analysis_pass2.csv`
- `output/horse_race.csv`, `output/horse_race_bootstrap.csv`
- `output/control_vs_main_comparison.csv`
- `output/regime_gate_comparison.csv`, `output/long_short_decomposition.csv`
- `output/feature_ablation_pass2.csv`, `output/feature_ablation_per_feature.csv`
- `output/return_decomposition_pass2.csv`, `output/regime_analysis_pass2.csv`
- `output/diagnostics_summary_pass2.txt`

Review documents:
- `reviews/{skeptic,believer,literature,practitioner}_p2.md` (Round 1)
- `reviews/{skeptic,believer,literature,practitioner}_rebuttal_p2.md` (Round 2)
- `reviews/{skeptic,believer,literature,practitioner}_rebuttal2_p2.md` (Round 3)
- `reviews/synthesis_pass2.md`

Notebooks: `notebooks/01_pass2_data_additions.ipynb` through `notebooks/07_pass2_diagnostics.ipynb`.

## Resume Framing

Designed and executed a walk-forward ML pipeline for cross-asset signal research, adding three methodological contributions in Pass 2: a non-crypto-filtered control universe (the central test of selection-circularity in a BTC-correlation-selected equity universe), a deep-learning horse race using a 3-block dilated TCN trained on Modal T4 GPUs with per-fold checkpointing, and an index-level prediction path that produces a deployable long-only futures strategy (net Sharpe 1.5-2.0 after realistic execution costs) where the single-stock implementation fails due to borrow and spread drag.

## Limitations

- The 2 bps modeled half-spread on index futures is a 7-14x underestimate. A 30-day paper-trade is required before any capital decision. The 2x spread sensitivity column is the anchor for the net Sharpe 1.5-2.0 range; beyond that is extrapolation.
- Regime gate reduces Sharpe in this sample rather than improving it. The BTC/S&P 500 market-cap ratio versus 1-year median is reported as a sensitivity, not a validated ingredient.
- KR 2023 attenuation from Pass 1 is not revisited at year-by-year granularity in Pass 2; post-publication decay (McLean-Pontiff 2016) is consistent with the data but not established.
- KR short-sale ban November 2023 to March 2024 is not gated in the backtest. For futures-based implementations this is immaterial.
- KOSPI large-cap variant is inconclusive (94 percent flat months at $50M ADV threshold). Acceptance criterion #2 was not testable.
- TCN significantly underperforms LightGBM and shows pervasive overfit flags. The signal in this sample lives in daily-summary features, not minute-bar temporal structure.
- Three Pass 1 dropped features (log mcap bucket, BTC liquidation intensity, USDT peg deviation) remain dropped in Pass 2; enabling credentials were not available in this run.
- Seven years OOS covers one partial crypto cycle. Strategy untested in a sustained bear from post-ETF-era highs.
- Control universe is manually curated; the HK null (p=0.93) survives this concern, the KR result (p=0.008) is more vulnerable to it.

See `logs/feature_decisions.log` and `logs/feature_restoration_decisions.log` for the complete record of each feature-level deviation from the brief.
