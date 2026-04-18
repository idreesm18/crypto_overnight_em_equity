# Crypto Overnight Signal for Asian Equities

## Hypothesis

Crypto markets trade 24/7. Asian equity markets do not. Between the HKEX or KRX cash-equity close and the next day's open, roughly 17 hours of crypto trading occur. If overnight crypto activity carries information about risk appetite, dollar demand, or liquidation stress that is not yet priced into crypto-exposed Asian equities at the next open, the overnight gap return should be more predictable than the intraday return.

The research tests four questions. First, does the overnight crypto channel show up as a gap-to-intraday IC differential in cross-sectional tests on crypto-exposed HK and KR equities? Second, is that predictability actually a crypto channel, or a short-term reversal effect inflated by the BTC-correlation-filtered universe? A parallel pipeline on a control universe with matched size and liquidity but no crypto-correlation filter isolates this. Third, does the signal survive transaction costs in any implementable form, once a borrow cost model and a regime gate are added? Fourth, does a temporal model on minute bars extract information the 21 hand-engineered features miss?

The answers in this sample: the gap-to-intraday asymmetry holds cross-market (3:1 HK, 2.6:1 KR, both p < 0.001); the crypto channel is partially identified in KR and not identified in HK; no single-stock strategy survives costs but two index-futures strategies do; the TCN is worse than LightGBM on gap IC.

## Data

All data covers 2018-09-01 through 2026-04-17.

Binance bulk archive provides spot and perpetual 1-minute klines for BTC, ETH, SOL, BNB, XRP and 8-hourly funding rates for BTC and ETH. No authentication. Binance stopped publishing per-day liquidation snapshots in its bulk archive, so the originally-specified liquidation intensity feature was dropped.

Stooq HK daily .txt files are user-provided (2,869 files). Filenames use a lower-case suffix with no leading zeros.

KR daily OHLCV come from pykrx v1.0.51. KRX geo-blocks US IPs and has a login requirement on the batch-by-date endpoint. Access is through an SSH SOCKS5 tunnel to a Korean-exit Oracle Cloud VM, with a per-ticker endpoint loop. The market-cap endpoint is authenticated-only even with the tunnel; the log mcap bucket feature stays dropped.

FRED provides VIXCLS, DTWEXBGS (DXY proxy), DGS10, DGS2, T5YIE, T10YIE, and DFF. yfinance provides BTC-USD daily closes, index OHLCV for ^HSI and ^KS11, and S&P 500 daily close (^GSPC). BTC circulating supply is computed deterministically from the issuance schedule, anchored at 19.6M BTC on 2024-01-01, advancing at 900 BTC/day pre-halving and 450 BTC/day post the 2024-04-19 halving.

The S&P 500 market-cap proxy is computed as index level times a scaling factor calibrated to total S&P 500 market cap on 2024-12-31 (approximately $52.2T against a close of 5881.63, giving factor 8.875e9). The BTC-to-S&P 500 market-cap ratio, with its trailing 1-year median, is the regime gate variable.

Two user-provided candidate CSVs list 40 HK and 39 KR crypto-exposed tickers for the main universe. Two user-provided control CSVs at `data/stock_picks/crypto_control_{hk,kr}.csv` contain 42 HK and 39 KR tickers drawn from liquid sectors with minimal crypto exposure (traditional banks, brokers, industrial conglomerates), matched on size and liquidity to the main universe.

Three features originally specified in the brief are dropped for data-availability reasons: log mcap bucket (KRX authenticated endpoint), BTC perp liquidation intensity (paid data source required), and USDT peg deviation (minute-level BTC/USD required). The enabling credentials were not available in this run. The restoration decision is logged at `logs/feature_restoration_decisions.log`.

## Methodology

### Universes

The pipeline runs four universes in parallel.

*Main universe* applies a trailing 60-day BTC correlation rank and a $500K trailing 20-day ADV filter at each monthly rebalance, selecting the top 20 to 30 names per market. 88 rebalance months from 2019-01-02 through 2026-04-01, median 24 HK and 30 KR tickers per month.

*Control universe* applies the same $500K ADV filter, no BTC correlation filter, capped at 30 per market. 88 rebalance months HK, 87 KR, median pool 30 in each. Two tickers fell out of data (0273.HK Stooq file missing, 041140.KQ delisted). Logged at `output/control_universe_log.csv`, features at `output/features_track_a_control_{hk,kr}.parquet`.

*KR KOSPI large-cap* restricts the KR main candidate list to .KS tickers with $50M trailing-20-day ADV, ranked by BTC correlation, top 12. Only 14 of 39 KR candidates are .KS, and at the $50M threshold 83 of 88 rebalance months cannot form a pool of 9+ names. The variant is flagged inconclusive and downstream backtest configurations for large-cap are skipped. Detail at `logs/stage_p2-4_kospi_largecap.log`.

*Index* is a two-ticker universe (^HSI, ^KS11) with 1,793 and 1,790 trading days. Features are 18: 12 crypto overnight plus 6 macro, no stock-level features.

### Features

Track A, the hand-engineered daily feature set used by LightGBM, contains 21 features for stock universes: 12 crypto overnight (BTC and ETH log returns over the window, realized vol, BTC max drawdown, USD notional volume, volume surge, taker buy/sell imbalance, cross-pair dispersion, BTC-ETH return spread, BTC perp funding level and change), 6 macro (VIX level, VIX 5-day change, DGS10-DGS2 slope, DXY level, DXY 5-day change, 5-year breakeven), and 3 stock-level (trailing 20-day realized vol, trailing 20-day log return, prior-day log return). Macro features are lagged 1 day. The index universe drops the three stock-level features for 18 total. No lookahead.

The HK overnight window runs 08:00 UTC day T to 01:30 UTC day T+1 (17.5 hours); KR runs 06:30 UTC day T to 00:00 UTC day T+1 (17.5 hours). Non-trading-day boundaries are handled via exchange_calendars; weekend gaps are flagged in an `is_weekend_gap` column.

Track B is the raw-sequence track for the TCN. Per overnight window, minute bars for BTC, ETH, SOL are extracted and normalized per-channel per-window, then padded to the per-market maximum length with a boolean mask. 5-minute resampling keeps the file size tractable (~1.7 GB total); 1-minute would have required ~57 GB. SOL pre-launch days (before 2020-08-11) are zero-filled for its five channels with the mask reflecting missing data. Static features (macro plus stock-level, or macro only for index) concatenate at the post-encoder stage. Files: `output/sequences_{hk,kr,index_hk,index_kr}.npz`.

### LightGBM

Walk-forward expanding window, 252 trading-day minimum training, monthly rebalance cadence. At each monthly boundary the model trains on all data with date before the rebalance and predicts OOS rows in the coming month. Hyperparameter search uses 10-iteration randomized search with 3-fold purged time-series CV inside the training set, 5-day purge buffer. Search runs at year boundaries and at the first fold; hyperparameters are reused between searches. 18 LightGBM models total across four universes and three targets. SHAP is computed per fold where feasible.

An overfit flag fires if training IC exceeds test IC by more than 0.20 in a fold. On the gap target for the main universe the flag is rare (1 HK, 0 KR). On intraday it fires more often, reflecting the model's tendency to memorize in-sample noise when OOS signal is weak.

### TCN

Three dilated causal convolutional blocks, dilation 1 / 4 / 16, 64 filters, kernel size 7, BatchNorm and ReLU with Dropout 0.2, residual connections. Global average pooling, concatenation with static features, FC head 32 then 1. AdamW optimizer with weight_decay 1e-4, LR 1e-3 cosine-annealed to 1e-5, batch size 64, max 30 epochs with patience 5.

Training runs on Modal T4 GPUs. The runtime-adjustment ladder moved to Step 1 from the start: 5-minute sequences, 2 blocks (dropping dilation 16), batch size 128, max epochs 15. Modal's default 2-hour per-function timeout was exceeded on the main-universe configs during the first attempt; the longer main sequence counts (40,808 HK rows, 51,876 KR rows) push walk-forward to ~75 folds averaging 2.5 hours. The retry added per-fold checkpointing to the Modal volume, increased timeout to 4 hours, and used `starmap(return_exceptions=True)` so a single-container failure does not cascade-cancel siblings. The final run wrote 12 TCN prediction CSVs to local disk.

### Strategies

Stock universes use a tercile rule: long top third, short bottom third, equal weight within legs, one-day hold, flat if universe < 9 on a given day. A long-only variant uses the top tercile benchmarked against an equal-weight portfolio of that day's full universe, no short leg, no borrow cost.

Indices use a threshold strategy: long if y_pred exceeds 0.5 times the in-sample standard deviation of predictions, short if below negative that threshold, flat otherwise. Recomputed at each monthly rebalance.

### Costs

Half-spread per side: 15 bps HK main, 10 bps control HK and KR main, 5 bps control KR, 2 bps index (the last is an underestimate; see Limitations). Kyle-style impact `0.1 * sqrt(trade_size / ADV) * daily_vol` with $100K per position, applied to stock strategies only. Borrow cost annualized, applied daily to short notional: 500 bps HK main shorts, 300 bps control HK, 400 bps KR main, 250 bps control KR, 50 bps index. Long-only variants pay no borrow.

Spread sensitivity and borrow sensitivity sweeps each cover 0.5x, 1x, 1.5x, 2x at 1x of the other dimension, plus breakeven analysis per primary configuration.

### Regime gate

At each day T, the variable is the 30-day mean of BTC market cap divided by the 30-day mean of the S&P 500 market-cap proxy. Compared to the 252-day trailing median of that ratio (excluding day T). If the ratio exceeds the median, gate = 1 (active); otherwise gate = 0 (flat). The 282-day warmup (252 median plus 30 ratio) defaults to gate = 1. Every backtest is run gate-on and gate-off.

## Results

### Lead result: control versus main

This is the central finding. The main universe's gap IC could reflect either a crypto channel or a short-term reversal pattern amplified by the BTC-correlation filter. The control universe tests whether the pattern survives the removal of that filter.

| Market | Main gap IC | Control gap IC | Diff | Bootstrap p | Main gap:intraday | Control gap:intraday |
|---|---|---|---|---|---|---|
| HK | 0.061 | 0.051 | +0.010 | 0.93 | 3.14 | 4.80 |
| KR | 0.061 | 0.037 | +0.024 | 0.008 | 2.15 | 0.88 |

In HK, the control universe gap IC is statistically indistinguishable from main (p=0.93), and the control gap-to-intraday ratio (4.80) exceeds the main ratio (3.14). Both facts point the same way: the HK gap-dominance pattern is not crypto-specific in this sample. The HK result reads as a Lo-MacKinlay selection-bias null.

In KR, the control gap IC is 0.024 lower than main with p=0.008, and the control gap-to-intraday ratio is 0.88 versus main's 2.15. The control ratio clears the brief's upper bound of 1.5 but the main ratio misses the lower bound of 2.5. The direction is consistent with an incremental crypto channel that explains roughly 40 percent of the raw main gap IC, but magnitudes fall short of the strong-version acceptance threshold.

Applying the brief's decision rule: neither market satisfies Rule A (cleanly crypto-specific), neither satisfies Rule B (ratios comparable within 30 percent), and Rule D (mixed evidence) is the best-fitting framing for both.

### Main universe headline

The cross-sectional gap-vs-intraday decomposition is the pre-specified directional finding. Both markets show gap IC near 0.061 with p < 0.001, intraday IC near 0.02. Gap IC exceeds intraday IC by 0.041 in HK and 0.038 in KR. Close-to-close predictability is weak; HK cc mean IC of 0.014 is below the 0.03 threshold though its bootstrap p-value (0.036) is significant, and KR cc does not achieve statistical significance.

The Monday gap (65-hour Friday-to-Monday overnight) carries more information than the mid-week 17-hour gap: Monday IC 0.082 in HK and 0.065 in KR, versus Tuesday-Friday 0.056 and 0.060. Both markets show Monday > mid-week; the cross-market consistency is consistent with a common mechanism rather than a market-specific artifact. SHAP stability across folds is 75 percent top-10 overlap with three features (eth_ov_log_return, btc_ov_log_return, vix_level) in the top ten in at least 87 percent of folds for the gap target.

### Acceptance criteria

| Criterion | Result | Verdict |
|---|---|---|
| 1. Index gap IC > 0.03 with bootstrap p < 0.05, in at least one market | HK 0.185 and KR 0.277 (rolling 30-day time-series Spearman), p < 0.001 | PASS |
| 2. KR KOSPI large-cap net Sharpe > 0.5 | Universe too thin (5 of 88 months with pool >= 9) | SKIPPED per brief risk #3 |
| 3. TCN gap IC exceeds LightGBM by > 0.01, paired bootstrap p < 0.05 | HK -0.021 (p=0.006); KR -0.028 (p<0.001); LightGBM wins | FAIL |
| 4. Control gap:intraday < 1.5 AND main gap:intraday >= 2.5 in same market | HK main 3.14 pass, control 4.80 fail. KR main 2.15 fail, control 0.88 pass | PARTIAL |

Criterion 1 passes on the index targets. Criterion 4 passes directionally in KR but not on magnitude.

### Horse race: LightGBM versus TCN

On the matched main universes, LightGBM significantly outperforms TCN on gap IC. The paired block-bootstrap IC differences are -0.021 in HK (p=0.006) and -0.028 in KR (p<0.001). Every TCN fold shows the overfit flag (train IC 0.33 to 0.38 versus OOS IC 0.03 to 0.09). A 50-50 ensemble is slightly worse than LightGBM alone.

Two observations sharpen the interpretation. First, the signal in this sample lives in daily-summary features rather than in minute-bar temporal structure, consistent with the tabular-data-benchmark literature (Grinsztajn et al. 2022, Gu-Kelly-Xiu 2020). Second, the index TCN predictions are the only way to generate a model comparison on index targets because cross-sectional IC is undefined for a single-ticker-per-date universe. Index TCN contributes to the backtest and the horse race, not to a claim of temporal-model superiority.

### Backtests

Of 96 primary configurations in `output/backtest_summary_pass2.csv`, eight clear net Sharpe 0.5 at 1x modeled costs. All eight are LightGBM index threshold strategies on HSI or KOSPI, gap target, both with gate-on and gate-off. Top row: LightGBM index_kr gap gate_off at net Sharpe 3.74, annualized return 47.2 percent, max drawdown -13.0 percent.

The modeled 2 bps half-spread on index futures is an underestimate. Realistic KOSPI 200 and HSI futures half-spread is 1 to 4 bps, and including slippage at size, the effective cost is 7 to 14 times the modeled level. At the realistic multiplier, the top-config net Sharpe falls to roughly 1.5 to 2.0. The 2x spread sensitivity column shows Sharpe ~2.3 at 2x modeled, which is within the same order of magnitude as the realistic floor. Extrapolation beyond the 2x column is not a validated number; a live paper-trade with executed-spread tracking is the correct next step.

No stock-tercile strategy (main or control, long-short or long-only, gate-on or gate-off) clears net Sharpe 0.5 at 1x costs. Rule C from the brief applies to all single-stock implementations: the signal loses to execution costs. The cross-sectional IC exists (0.06 for main, 0.04-0.05 for control KR and HK), but HK main half-spread 15 bps plus impact plus 500 bps annualized short-leg borrow, and KR main 10 bps plus 400 bps borrow, consume the gross P&L. Long-only variants reduce the cost drag but fall short of the 0.5 Sharpe bar.

### Regime gate

Gate-on consistently reduces Sharpe relative to gate-off across the top configurations. For LGBM index_kr gap, gate-off is 3.74 net Sharpe and gate-on is 2.69; for LGBM index_hk gap, 2.38 versus 1.55. Days active under gate-on are 1,101 of 1,522 for KR and similar for HK; the inactive ~30 percent of days contain positive returns that the gate excludes. The constructed regime variable is not identifying low-IC periods in this sample. The gate is documented as a reported sensitivity, not a validated ingredient.

### Feature ablation

Tier 1 at the category level, on main universe gap:

| Ablation | HK delta net Sharpe | KR delta net Sharpe |
|---|---|---|
| Drop crypto overnight (12 features) | -1.155 | -0.692 |
| Drop macro (6 features) | near zero | +0.02 |
| Drop stock-level (3 features) | large but delta_ic NaN | large but delta_ic NaN |

Subcategory tier: drop overnight returns has smaller effect than drop crypto-overnight-all, consistent with multiple crypto features carrying incremental information. Per-feature leave-one-out on 21 features across both markets: `stock_rv_20d` is the single most important feature (HK -0.92, KR -2.38 on net Sharpe). The stock-level block is load-bearing first; crypto overnight features are individually second-tier but jointly meaningful.

### Regime splits and long-only decomposition

Regime analysis on `output/regime_analysis_pass2.csv`: for index_kr gap, IC floors above 0.237 across VIX high/low, BTC up/down, and crypto-vol high/low cells; no regime cell inverts or collapses. For main universes, the KR year-by-year fold IC shows a 2023 attenuation (mean IC 0.001, 5 of 12 folds positive) against 0.060 to 0.091 in 2020 through 2022 and a recovery in 2024 onward. `output/long_short_decomposition.csv` shows alpha concentration in the long leg for all four stock universes; the short leg adds gross signal but its cost profile (spread plus borrow) consumes the added contribution.

## Limitations

The 2 bps modeled half-spread on index futures is a 7 to 14 times underestimate of realistic transaction cost. The sensitivity table tops out at 2x; beyond that the net Sharpe range (1.5 to 2.0 at realistic costs) is an out-of-file extrapolation. A 30-day live paper-trade with executed-spread tracking is the right next step before any capital decision.

The index-level IC metric (rolling 30-day time-series Spearman) is not cross-sectional IC. It has a defensible lineage (Jegadeesh-Titman 1993 rolling momentum IC; block bootstrap handles autocorrelation) but the null benchmark question matters. A random-walk-prediction benchmark comparison is not included in this run.

Regime gate reduces Sharpe rather than improving it. The gate variable (BTC / S&P 500 market-cap ratio versus 1-year trailing median) is a reasonable first attempt but does not actually identify low-IC periods in the 2019-2026 sample. A year-by-year Sharpe breakdown for index strategies would be required to discriminate McLean-Pontiff post-publication decay from regime-conditional behavior.

The KR 2023 attenuation is not revisited at per-year granularity for the index strategies. If the attenuation reflects post-ETF crypto-to-equity information-speed compression, it is candidate evidence for decay. Seven years OOS covers one partial crypto cycle; a second bear from elevated levels has not been tested.

The KR short-sale ban window (November 2023 to March 2024) is not gated in the backtest. For index futures this is immaterial; for stock L/S, the backtest reports counterfactual short-leg P&L during that window.

The KOSPI large-cap variant is inconclusive, not negative. The $50M ADV threshold eliminated 83 of 88 rebalance months. Acceptance criterion #2 was not testable. Whether a reduced $25M threshold or a different candidate list would produce a valid test is not resolved here.

The TCN significantly underperforms LightGBM on gap IC and is overfit flagged on every fold. This argues against using TCN as the primary model, not against the underlying signal. The LightGBM outperformance is consistent with tabular-benchmark literature but does not rule out the possibility that LightGBM itself captures some amount of sample-specific structure.

Three features originally specified in the brief remain dropped: log mcap bucket, BTC perp liquidation intensity, and USDT peg deviation. None of the enabling credentials were available in this run. The feature restoration framework is in place for a future run.

The control universe is manually curated. While it carries no BTC-correlation filter, its selection by design captures the matched-sector, matched-size criterion, which is itself a signature. The HK null (p=0.93) survives this concern; the KR significant difference (p=0.008) is more vulnerable to it.

## Conclusion

The finding that survives adversarial review is twofold. One, index-level gap prediction for HSI and KOSPI produces a tradable signal at the futures-implementation level, net Sharpe in the 1.5 to 2.0 range after realistic execution costs. Two, the universe-selection circularity objection is resolved asymmetrically: HK's gap dominance does not distinguish from a universe-wide overnight-reversal pattern, while KR retains an incremental crypto component of roughly 40 percent of the raw main IC, statistically significant at p=0.008.

The single-stock strategy loses to execution costs. Main HK, main KR, control HK, control KR, and the KR large-cap (inconclusive) all fall below the net Sharpe 0.5 bar at 1x modeled costs and at realistic cost multiples. Long-only variants narrow the gap but do not clear the bar. Rule C from the brief's decision framework applies cleanly to the single-stock implementation.

The TCN horse race is informative as a negative result. Minute-bar temporal structure, at least as extracted by a 2-block dilated TCN, does not add information beyond the 21 engineered daily-summary features. LightGBM remains the primary model.

What remains unresolved. The index net Sharpe range (1.5 to 2.0 at realistic cost) is an extrapolation from a 2x spread sensitivity column, not a measurement. A 30-day paper trade with executed-spread tracking would upgrade this to a measurement. The KR 2023 attenuation and the BTC-to-S&P 500 ratio regime gate, taken together, present a candidate decay story that neither supports nor rules out post-publication decay per McLean-Pontiff (2016). A year-by-year Sharpe tearsheet for the index strategies, plus a cross-market replication attempt on Taiwan or SGX, would address this more directly than further manipulation of the HK and KR pipelines.

Four panelists converged on the practitioner's paper-trade gate as the correct next operational step. The research pipeline is deployable in a form narrower than initially hypothesized: long-only index futures on KOSPI and HSI, LightGBM, gate-off (default), with a pre-specified 6-month rolling-Sharpe kill switch. All other configurations remain research artifacts.

## Appendix — Files

Key outputs:
- `output/backtest_summary_pass2.csv`, `output/cost_sensitivity_pass2.csv`, `output/borrow_sensitivity_pass2.csv`, `output/breakeven_analysis_pass2.csv`
- `output/control_vs_main_comparison.csv` (the central control-universe result)
- `output/horse_race.csv`, `output/horse_race_bootstrap.csv`
- `output/regime_gate_comparison.csv`, `output/long_short_decomposition.csv`
- `output/regime_analysis_pass2.csv`, `output/return_decomposition_pass2.csv`
- `output/feature_ablation_pass2.csv`, `output/feature_ablation_per_feature.csv`
- `output/diagnostics_summary_pass2.txt`
- `output/predictions_lgbm_{control,index}_{hk,kr}_{gap,intraday,cc}.csv` (12 files)
- `output/predictions_tcn_{main,index}_{hk,kr}_{gap,intraday,cc}.csv` (12 files)
- `output/features_track_a_{control_hk,control_kr,index}.parquet`
- `output/sequences_{hk,kr,index_hk,index_kr}.npz`

Review documents (twelve panelist papers plus synthesis):
- `reviews/{skeptic,believer,literature,practitioner}_p2.md`
- `reviews/{skeptic,believer,literature,practitioner}_rebuttal_p2.md`
- `reviews/{skeptic,believer,literature,practitioner}_rebuttal2_p2.md`
- `reviews/synthesis_pass2.md`

Notebooks (one per pipeline stage):
- `notebooks/01_pass2_data_additions.ipynb`
- `notebooks/02_pass2_universes.ipynb`
- `notebooks/03_pass2_features.ipynb`
- `notebooks/04_pass2_model_training.ipynb`
- `notebooks/05_pass2_backtest_and_costs.ipynb`
- `notebooks/06_pass2_ablation.ipynb`
- `notebooks/07_pass2_diagnostics.ipynb`

Decision logs (`logs/feature_decisions.log`, `logs/feature_restoration_decisions.log`) document each feature-level deviation from the brief. A narrower initial version of the pipeline (main universe, LightGBM only, no control universe, no TCN, no regime gate) lives on the `p1` branch as a reference.
