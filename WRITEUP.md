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

Pass 2 used fixed per-market half-spread assumptions (15 bps HK main, 10 bps control HK and KR main, 5 bps control KR, 2 bps index). The CS-spread update replaces those with per-ticker per-day empirical estimates from the Corwin–Schultz (2012) high-low estimator, built from daily OHLC already in the pipeline. The method: for each consecutive day pair per ticker, apply the overnight adjustment to H and L, then compute β (sum of squared log H/L over two days), γ (squared log of two-day-window H/L), α = (√(2β) − √β)/(3 − 2√2) − √(γ/(3 − 2√2)), and S = 2(e^α − 1)/(1 + e^α); floor S at zero when α is negative. Assign S to day t+1. Take the trailing 20-trading-day mean (minimum 12 valid observations) as the ticker's daily round-trip spread. Fall back to that day's cross-sectional market median when fewer than 12 observations exist and at least five tickers in the market have valid values; during the initial warmup, fall back to the old fixed assumption until the market median becomes available. Half-spread per side is cs_spread / 2.

Market impact (Kyle-style `0.1 * sqrt(trade_size / ADV) * daily_vol`, $100K per position, stock strategies only) and borrow cost (annualized, applied daily to short notional: 500 bps HK main, 300 bps control HK, 400 bps KR main, 250 bps control KR, 50 bps index) are unchanged from Pass 2. Long-only variants pay no borrow.

Spread sensitivity and borrow sensitivity sweeps each cover 0.5x, 1x, 1.5x, 2x at 1x of the other dimension. Under CS spreads the multiplier is applied to the estimator output, so the sweep now answers "what if the CS estimator is biased by factor Y?" rather than "what if the true spread is X bps?".

Realized CS spreads (round-trip, bps, medians from `output/cs_spread_summary.txt`):

| Market | CS median 20d |
|---|---:|
| HK stocks | 98.4 |
| KR stocks | 84.8 |
| HSI_proxy (stand-in for 2800.HK) | 30.4 (floored) |
| 069500.KS (KODEX 200, pykrx) | 25.5 (floored) |

**CS is used for stock universes. For index ETFs, CS is replaced by a tick-floor-derived realistic spread.** Both index CS figures (HSI_proxy 30.4 bps, 069500.KS 25.5 bps) show raw-negative fractions above 50% (53.2% and 52.4% respectively), which places the estimator at its signal-to-noise floor [CS-2012, TR-2022]. Corwin & Schultz (2012, Table 6) themselves report that the cross-sectional correlation between CS estimates and true TAQ effective spreads falls from ~70% for illiquid stocks to ~18% for liquid large-cap names; for index-tracking liquid ETFs this correlation is effectively noise. Tremacoldi-Rossi & Irwin (2022, JFQA) formalize that CS bias increases with the volatility-to-spread ratio and that high raw-negative fractions are a direct diagnostic of noise-floor violation, with zero-flooring introducing a downward-truncation bias on the 20-day mean. Abdi & Ranaldo (2017, RFS) show CS underperforms competing estimators specifically on the most-liquid quintile.

For index ETFs the realistic round-trip effective spread is derived from exchange tick structure and market-making rules, not from the CS estimator:

**2800.HK Tracker Fund of Hong Kong.** HKEX introduced a separate tighter ETP spread table on 1 June 2020, cutting minimum ticks for high-liquidity ETPs by 50-90% across price bands [HKEX-ETP, HKEX-Min]. At 2800.HK's current HKD 26 price level the ETP minimum tick is HKD 0.02, giving a one-tick quoted round-trip of 0.02/26 ≈ **8 bps**. HKEX mandates at least one Securities Market Maker per ETP counter with continuous quoting obligations; for 2800.HK (AUM ~HKD 142B, multiple competing SMMs) the realized spread typically sits at the one-tick floor during regular hours. The August 2025 Phase 1 equity tick reform explicitly excluded ETPs on the grounds that "minimum spreads for ETPs were reduced in 2020 and are trading efficiently" [HKEX-Dec24]. Pre-reform (before 2020-06-01) the applicable equity tick was HKD 0.05 at the HKD 20-30 band, giving a conservative ~15 bps round-trip. The backtest applies 15 bps round-trip pre-reform and 8 bps round-trip post-reform as a date-indexed step function.

**069500.KS (KODEX 200).** KRX applies a flat 5-KRW tick to KOSPI 200 ETFs across the full analysis window; the 2023 KRX tick reform affected single-stock ticks, not ETF ticks. At KODEX 200's ~33,500 KRW price level, 5 KRW is ≈1.5 bps per side, implying a ~3 bps round-trip tick-floor. KRX Liquidity Providers on popular KOSPI 200 ETFs have spread-obligation rules that keep the realized spread tight. The backtest applies a uniform **5 bps** round-trip for the full 2019-2026 window as a conservative best-estimate anchored to the tick floor plus a small LP-quote margin.

**Citation keys** (full refs in WRITEUP appendix / README):

- [HKEX-ETP] HKEX News Release, 18 May 2020, "HKEX to Introduce New Initiatives to Enhance Liquidity of ETPs."
- [HKEX-Min] HKEX, "Reduction of Minimum Spreads" (official spread table).
- [HKEX-Dec24] HKEX News Release, December 2024, "HKEX to Reduce Minimum Spreads in Hong Kong Securities Market" (confirms ETP exclusion from Phase 1).
- [CS-2012] Corwin, S.A. and Schultz, P. (2012), *Journal of Finance* 67(2):719–760.
- [TR-2022] Tremacoldi-Rossi, P. and Irwin, S.H. (2022), *Journal of Financial and Quantitative Analysis*, SSRN 4216953.
- [AR-2017] Abdi, F. and Ranaldo, A. (2017), *Review of Financial Studies* 30(12):4437–4480.

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

CS spreads are the cost model throughout this section. The Pass 2 fixed-spread backtest remains on disk (`output/backtest_summary_pass2.csv`) for audit but is not carried forward as a deployable verdict for any configuration.

**Stock universes.** Every stock-tercile strategy (main or control, long-short or long-only, gate-on or gate-off, LightGBM or TCN) is net-negative under CS spreads at 1x, across all three targets. No configuration in the 0.5x-2x spread sensitivity sweep lifts a stock universe above zero. The Pass 2 synthesis had already applied Rule C to the single-stock implementation; the CS update confirms it with a realistic-cost empirical basis (HK stocks 98 bps round-trip, KR main 85 bps, each ~3-4x the Pass 2 fixed assumption). The stock strategies are not pursued further here; the Rule-D IC findings (control-vs-main) remain valid research contributions but do not translate into a tradable single-stock strategy. All four stock universes are now empirically CS-covered: Stage P2-18 pulled daily OHLCV for all 38 control_kr tickers via pykrx (67,603 rows, 0 skips; median CS spread 80.5 bps round-trip). Under real CS, control_kr is also net-negative across all 12 configurations, with all three previously marginally positive configurations (gap/tercile_ls and intraday/tercile_ls at fixed 5 bps/side) flipping to net-negative Sharpe of −5 to −8. The real CS spread for control_kr tickers (~80 bps round-trip) is approximately 8× the Pass 2 fixed assumption (10 bps round-trip), explaining the large downward shift.

**Index universes.** Under tick-floor realistic spreads (8 bps post-reform for 2800.HK, 5 bps for 069500.KS) at 1x spread and 1x borrow, seven configurations clear the 0.5 net-Sharpe threshold — all LightGBM on gap and cc targets:

| Configuration | fixed-baseline net SR | tick-floor net SR | delta |
|---|---:|---:|---:|
| LightGBM index_kr gap gate_off | 3.74 | **3.64** | −0.10 |
| LightGBM index_kr gap gate_on  | 2.69 | **2.60** | −0.10 |
| LightGBM index_hk gap gate_off | 2.38 | **1.96** | −0.42 |
| LightGBM index_hk gap gate_on  | 1.55 | **1.19** | −0.36 |
| LightGBM index_kr cc  gate_off | 1.16 | **1.10** | −0.06 |
| LightGBM index_kr cc  gate_on  | 0.90 | **0.85** | −0.05 |
| LightGBM index_hk cc  gate_off | 0.83 | **0.55** | −0.28 |

Two TCN configurations are positive but sub-threshold (tcn index_hk cc gate_on 0.44, gate_off 0.16). All remaining 15 index configurations are net-negative. See `output/cs_vs_fixed_comparison.csv` for the full 24-row side-by-side and `output/backtest_summary_pass2_cs.csv` for the primary summary.

The tick-floor index spreads are modestly higher than the Pass 2 fixed 2 bps/side (4 bps round-trip) baseline but materially lower than the CS noise-floor proxy figures of 25-30 bps round-trip. Compared to the fixed-baseline backtest, every index configuration moves in the same direction (spreads are higher, so net Sharpe drops), but the seven LGBM gap and cc survivors remain well above the 0.5 threshold. Compared to the earlier CS-noise-floor analysis, five configurations that had "flipped to negative" (index_hk gap gate-on/gate-off, index_hk cc gate-off, index_kr cc gate-on/gate-off) are restored as deployable: they were artifacts of estimator noise-floor bias rather than real cost-survival failures.

The old extended sensitivity sweep (`output/cost_sensitivity_extended.csv`, 5x-14x of 2 bps) is superseded. Under the tick-floor methodology, the relevant sensitivity now stress-tests estimator bias around the tick floor: 0.5x = spreads 50% tighter than tick floor (unlikely), 1x = tick-floor baseline, 1.5x = realized spread 50% wider than one tick (occasional), 2x = two-tick-wide spread (high-vol conditions).

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

All four stock universes are empirically CS-covered. The stock backtest verdict under CS is unambiguous and universal: no stock-tercile configuration is deployable at realistic transaction costs in this sample.

The index cost model uses tick-floor-derived realistic spreads rather than CS for index ETFs. The CS estimator is at its noise floor for both index instruments (raw-negative fractions 53.2% and 52.4%, above the 50% threshold documented in [TR-2022]), producing floored 20d medians (25-30 bps) that overstate the true effective spread by roughly 3× for 2800.HK and 5× for 069500.KS. Corwin & Schultz (2012, Table 6) themselves report 18% cross-sectional correlation with true effective spreads for liquid large-cap names, effectively noise for these purposes. The backtest therefore substitutes exchange-official tick-structure spreads (8 bps round-trip for 2800.HK post-2020-reform, 5 bps for 069500.KS) with citations to HKEX news releases, HKEX's current ETP spread table, and KRX tick documentation. A live paper trade with executed-spread tracking would confirm the tick-floor assumption; a direct 2800.HK and 069500.KS intraday-tick pull (Bloomberg/TAQ-equivalent) would produce a measured point estimate rather than a tick-derived bound.

The index-level IC metric (rolling 30-day time-series Spearman) is not cross-sectional IC. It has a defensible lineage (Jegadeesh-Titman 1993 rolling momentum IC; block bootstrap handles autocorrelation) but the null benchmark question matters. A random-walk-prediction benchmark comparison is not included in this run.

A 30-day live paper trade with executed-spread tracking for index futures would settle whether the realized net Sharpe matches the tick-floor projection.

Regime gate reduces Sharpe rather than improving it. The gate variable (BTC / S&P 500 market-cap ratio versus 1-year trailing median) is a reasonable first attempt but does not actually identify low-IC periods in the 2019-2026 sample.

The year-by-year tearsheet for the index strategies (`output/index_yearly_tearsheet.csv`) shows the KR 2023 attenuation reproduces at the index level: index_kr gap gate_off annual Sharpe is 0.36 in 2023 (mean IC 0.09, pct_invested 0.40) against 1.8 to 6.7 in every other year 2020 through 2026. The index attenuation parallels the stock-level 2023 attenuation, which is consistent with a shared macro-regime effect rather than a model-specific artifact. Whether this reflects post-ETF information-speed compression (McLean-Pontiff 2016) or a transient regime cannot be discriminated with one occurrence. Seven years OOS covers one partial crypto cycle; a second bear from elevated levels has not been tested.

The KR short-sale ban window (November 2023 to March 2024) is not gated in the backtest. For index futures this is immaterial; for stock L/S, the backtest reports counterfactual short-leg P&L during that window.

The KOSPI large-cap variant is inconclusive, not negative. The $50M ADV threshold eliminated 83 of 88 rebalance months. Acceptance criterion #2 was not testable. Whether a reduced $25M threshold or a different candidate list would produce a valid test is not resolved here.

The TCN significantly underperforms LightGBM on gap IC and is overfit flagged on every fold. This argues against using TCN as the primary model, not against the underlying signal. The LightGBM outperformance is consistent with tabular-benchmark literature but does not rule out the possibility that LightGBM itself captures some amount of sample-specific structure.

Three features originally specified in the brief remain dropped: log mcap bucket, BTC perp liquidation intensity, and USDT peg deviation. None of the enabling credentials were available in this run. The feature restoration framework is in place for a future run.

The control universe is manually curated. While it carries no BTC-correlation filter, its selection by design captures the matched-sector, matched-size criterion, which is itself a signature. The HK null (p=0.93) survives this concern; the KR significant difference (p=0.008) is more vulnerable to it.

## Conclusion

Under CS spreads for stocks and exchange-tick-floor spreads for index ETFs (8 bps post-reform for 2800.HK, 5 bps for 069500.KS), seven configurations clear the 0.5 net-Sharpe threshold at 1x spread and 1x borrow: LightGBM index_kr and index_hk on the gap and cc targets, gate-on and gate-off. Top row: LightGBM index_kr gap gate_off at net Sharpe 3.64. Two additional TCN configurations (index_hk cc gate-on and gate-off) are positive but sub-threshold. The deployable form of the research is index-futures-level, across both HSI and KOSPI-200, with gap as the strongest single-target signal.

The universe-selection circularity finding from Pass 2 stands as a research result, not as a trading strategy. HK main's gap-dominance pattern does not distinguish from a universe-wide overnight-reversal pattern (control gap IC p=0.93), while KR retains an incremental crypto component at roughly 40 percent of the raw main IC (p=0.008). Under realistic CS costs neither translates into a tradable single-stock strategy in this sample.

The TCN horse race remains a negative result under realistic costs. Minute-bar temporal structure, as extracted by the two-block dilated TCN, does not add information beyond the engineered daily features in a way that survives transaction costs.

What remains unresolved. The index cost model now rests on exchange-tick-floor arithmetic with exchange-official and peer-reviewed citations; the ~8 bps 2800.HK and ~5 bps 069500.KS figures are consistent with HKEX ETP spread rules, KRX LP obligations, and the observed post-2020 regime, but a direct TAQ-equivalent intraday-tick pull would convert these from tick-derived bounds into measured point estimates. The KR 2023 attenuation is a single occurrence: decay versus transient regime cannot be discriminated with this sample. A 30-day live paper trade with executed-spread tracking on HSI and KOSPI-200 futures is the right next operational step and would confirm that the realized net Sharpe tracks the tick-floor projection (~2-3.6 on the gap configurations).

The research pipeline is deployable under realistic costs as a dual-market index-futures strategy: LightGBM on the gap target for both HSI and KOSPI 200, gate-off (default), with a pre-specified rolling-Sharpe kill switch. Secondary configurations (index cc, index gap gate-on) provide additional diversification at lower per-line Sharpe.

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
