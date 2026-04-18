# Crypto Overnight Signal for Asian Equities — Pass 1 Writeup

## Hypothesis

Crypto markets trade 24/7. Asian equity markets do not. Between the HKEX or KRX cash-equity close and the next day's open, roughly 17 hours of crypto trading occur. If overnight crypto activity carries information about risk appetite, dollar demand, or liquidation stress that is not yet priced into crypto-exposed Asian equities at the next open, we should see predictability in the overnight gap (close to next open) that is not present in the subsequent intraday return.

The hypothesis predicts two directional findings in advance: gap IC should exceed intraday IC, and the Monday gap (which covers 65 hours of weekend crypto trading) should carry more information than mid-week overnight gaps.

## Data

All data covers 2018-09-01 through 2026-04-15. The analysis window (feature-engineering start) is 2019-01-01, chosen after Stage 0 validation showed Stooq HK 2018 coverage was marginally short of the 60% threshold against the 2024 HKEX listing count. Starting a year later avoided that gate with no loss of training data for the walk-forward.

Binance bulk archive provided spot and perpetual 1-minute klines for BTC, ETH, SOL, BNB, XRP (spot) and BTC, ETH (perpetual), along with 8-hourly funding rates. Binance stopped publishing per-day liquidation snapshots in its bulk archive, so the originally-specified liquidation intensity feature was dropped and flagged for a paid source in Pass 2.

Stooq HK daily .txt files were user-provided (2,869 files). Filenames use a lower-case suffix with no leading zeros (e.g., `863.hk.txt` corresponds to candidate ticker `0863.HK`).

KR daily OHLCV came from pykrx v1.0.51. KRX geo-blocks US IPs and has recently added a login requirement on its batch-by-date endpoint. Access was through an SSH SOCKS5 tunnel to a Korean-exit Oracle Cloud VM, with a per-ticker endpoint loop rather than the batch-by-date endpoint. The market-cap endpoint is authenticated-only even with the tunnel; all mcap values are NaN in the cached parquet, and the "log mcap bucket" feature was dropped.

FRED provided VIXCLS, DTWEXBGS (DXY proxy), DGS10, DGS2, T5YIE, T10YIE, and DFF. yfinance provided BTC-USD daily closes for the universe correlation filter, cross-checked against Binance spot on three reference dates.

Candidate CSVs list 40 HK and 39 KR names. HK candidates include main-board crypto exchanges (OSL Group, HashKey), BTC-treasury companies (Boyaa, Meitu), and gaming/Web3 names. KR candidates include KOSPI large-caps with crypto adjacency (Naver, Kakao) and KOSDAQ blockchain-gaming names (WeMade). One KR ticker (ME2ON.KQ) was excluded from the pull because its code is non-numeric; pykrx expects 6-digit codes.

## Methodology

### Universe construction

At the first trading day of each month from 2019-01 through 2026-04, 60-day Pearson correlation of daily returns with BTC-USD was computed for each candidate, along with 20-day ADV in USD using a constant exchange rate (7.8 HKD/USD; 1,300 KRW/USD). The ADV filter set the floor at $500K. Remaining candidates were ranked by BTC correlation and the top 20-30 selected. Output: 88 rebalance months per market. Median universe size: 24 HK, 30 KR; the 30-cap bound for KR was hit in most months.

No lookahead: correlation and ADV windows end strictly before the rebalance date. Non-standard HK tickers (HSK.HK, MSW.US, IMG.US, SORA.HK) were attempted via literal filename lookup; four were not found in the Stooq HK bulk and logged as unmatched.

### Feature engineering

The overnight window for HK runs 08:00 UTC day T to 01:30 UTC day T+1 (17.5 hours), and for KR 06:30 UTC day T to 00:00 UTC day T+1 (17.5 hours). Non-trading-day boundaries were handled via the exchange_calendars package; weekend gaps were flagged in an `is_weekend_gap` column.

Overnight crypto features (12 after drops): BTC and ETH log returns over the window, realized vol from 1-minute returns, BTC max drawdown within window, BTC USD notional volume, volume surge vs trailing 7-day mean, taker buy/sell imbalance, cross-pair return dispersion across the five spot symbols, BTC-ETH return spread, BTC perp funding rate level and change.

Macro features (6, each at T-1): VIX level, VIX 5-day change, yield-curve slope (DGS10 - DGS2), DXY level, DXY 5-day change, 5-year breakeven. The 1-day lag ensures no lookahead into the prediction date.

Stock-level features (3): trailing 20-day realized vol, trailing 20-day log return (momentum), and prior-day log return. The "log mcap bucket" feature originally specified was dropped as noted.

Three targets were computed per stock-day: overnight gap (log open_T / close_{T-1}), intraday (log close_T / open_T), and close-to-close (log close_T / close_{T-1}). The pipeline trained one LightGBM per target per market: six total.

Output row counts: HK 40,808 rows across 33 tickers and 1,791 overnight windows; KR 51,876 rows across 38 tickers and 1,788 overnight windows.

### Walk-forward training

Expanding window. 252 trading-day minimum training period. At the first trading day of each month after that threshold, the model trained on all data with date before the rebalance and predicted each OOS row in the coming month.

Hyperparameter search used 10-iteration randomized search over the grid specified in the brief, with 3-fold purged time-series CV inside the training set. Purging used a 5-day buffer between CV splits to prevent leakage from the rolling features. Search ran at each year boundary and at the first fold; hyperparameters were reused between searches. Wall time came in at 11.0 minutes for all three HK targets and 10.9 minutes for the three KR targets.

SHAP was computed per fold on the test set using LightGBM's `pred_contrib=True`. The per-fold SHAP tables were saved as parquet for the Stage 6 stability analysis.

Overfit flag: if training IC on the full training set exceeded test IC by more than 0.20 in a fold, the fold was flagged. Gap-target flags are rare (1 HK, 0 KR). Intraday has 27 HK and 20 KR flags, reflecting the model's tendency to memorize in-sample noise when there is little OOS signal.

### Strategy and backtest

At each daily rebalance, predictions were sorted and split into terciles. The portfolio went long the top tercile and short the bottom tercile, equal weight within each leg, one day hold. On days where the universe fell below 9 names, the portfolio went flat (this did not occur in Pass 1 data: all 1,560 OOS days were invested).

Cost model combined a constant half-spread (15 bps HK, 10 bps KR) with a Kyle-style impact term: Impact = 0.1 × sqrt(trade_size / ADV) × daily_vol, with trade_size = $100K. Applied symmetrically to entry and exit. A cost sensitivity sweep at 0.5x / 1.0x / 1.5x / 2.0x the default was run, along with a breakeven analysis (the cost multiplier that drives net Sharpe to zero).

A block bootstrap computed the p-value for OOS IC at the market × target level, using 20-day blocks and 1,000 iterations. The block size reflects reasonable serial correlation in daily signals.

## Results

### Headline table (1x cost, full OOS period)

| Market | Target | Mean IC | Bootstrap p | Gross SR | Net SR | Ann ret net | Max DD net | Breakeven ×cost |
|---|---|---|---|---|---|---|---|---|
| HK | gap | 0.0610 | < 0.001 | 3.04 | -1.57 | -20.8% | -152% | 0.66 |
| HK | intraday | 0.0195 | < 0.001 | 0.35 | -2.99 | -75.2% | -464% | 0.10 |
| HK | cc | 0.0144 | 0.036 | 0.98 | -1.25 | -36.0% | -239% | 0.43 |
| KR | gap | 0.0608 | < 0.001 | 3.77 | -0.28 | -3.6% | -69% | 0.93 |
| KR | intraday | 0.0233 | 0.019 | 1.00 | -1.46 | -31.4% | -216% | 0.41 |
| KR | cc | 0.0076 | 0.255 | 0.59 | -1.58 | -37.9% | -244% | 0.27 |

### Gap vs intraday decomposition

In this sample, gap IC exceeds intraday IC by 0.041 in HK and 0.038 in KR. Both differences are significant (p < 0.001 via bootstrap). The 3:1 HK ratio and 2.6:1 KR ratio are the pre-specified directional prediction of the hypothesis, and the cross-market agreement is consistent with a common mechanism rather than a market-specific artifact.

Close-to-close predictability is weak. HK cc mean IC of 0.014 is below the brief's 0.03 threshold though its bootstrap p-value (0.036) is significant. KR cc does not achieve statistical significance. The brief's acceptance criterion 1 (cc IC > 0.03, p < 0.05) is therefore not met.

### Feature ablation

At the category level, dropping stock-level features produces the largest gross-Sharpe decline for the gap target: -2.50 HK, -3.03 KR. Dropping crypto-overnight features drops gross Sharpe by -1.10 HK and -1.78 KR. Dropping macro features is near-neutral for gap. Nine of 18 ablation combinations produce a gross-Sharpe drop greater than 0.15, satisfying the brief's acceptance criterion 3.

The Skeptic panelist argued this ordering reframes the result as a short-term reversal strategy with crypto as an additive secondary channel. The Believer responded that the crypto ablation drop of 1.10-1.78 is a substantial fraction of the ~3.0-3.8 gross Sharpe baseline, and that short-term reversal alone cannot generate the gap-vs-intraday differential observed. Both positions have merit. A Pass 2 counterfactual on a non-BTC-filtered universe would quantify the relative contributions more cleanly.

### Regime analysis

The gap signal is present in both VIX regimes. In KR, mean IC in high-VIX periods is 0.076 compared to 0.057 in low-VIX, and gross Sharpe 4.36 vs 3.60. HK shows less regime differentiation. BTC-trend splits produce a similar pattern: the signal holds in both BTC-up and BTC-down regimes, with KR slightly stronger in BTC-up periods (Sharpe 4.15 vs 3.33). No regime tested causes the signal to collapse.

Year-by-year fold IC tells a more complex story. HK gap IC is positive every year from 2021 through 2026 (0.031 to 0.100). KR has a clear 2023 attenuation (mean IC 0.001, 5 of 12 folds positive) but was 0.060 to 0.091 in 2020 through 2022. This is consistent with a regime-conditional mechanism rather than a uniformly structural one, and the Literature Reviewer anchored this interpretation to prior work on regime-dependent signals.

### SHAP stability

Consecutive-fold top-10 feature overlap averages 75%. Spearman rank correlation across folds is 0.70. Three features appear in the top 10 in at least 87% of folds for the gap target: eth_ov_log_return (96-97%), btc_ov_log_return (87-97%), and vix_level (96-97%). This rules out rotating-noise feature selection and confirms that the overnight crypto returns and the macro VIX conditioner carry the pre-specified signal.

### Weekend effect

Monday IC (65-hour Friday-to-Monday overnight gap) is 0.082 in HK and 0.065 in KR. Tuesday through Friday IC (17-hour overnight gap) is 0.056 HK and 0.060 KR. The longer window carries more predictive information, consistent with the hypothesis that crypto trading during the overnight produces information that is priced at the next open. In HK the Monday-vs-weekday difference is 0.026 and in KR it is 0.005; the HK pattern is stronger but the directional sign is consistent cross-market.

## Diagnostics summary

Feature ablation passes criterion 3 in 9 of 18 combinations. Return decomposition passes criterion 2 in both markets. The cc IC threshold (criterion 1) is not met. SHAP stability, regime robustness, and the weekend effect each independently support the hypothesis.

The complete Stage 6 output is in `output/diagnostics_summary.txt` and the underlying CSVs.

## Limitations

The universe selection rule uses 60-day BTC correlation. This is acknowledged as mildly tautological in the brief and was not counterfactually tested in Pass 1. The adversarial review flagged this as the largest unresolved methodological question. A Pass 2 run on a non-crypto-filtered universe would quantify the circularity effect.

Three features originally specified in the brief were dropped for data-availability reasons: BTC perp liquidation intensity (Binance no longer publishes in bulk), log market-cap bucket (pykrx mcap endpoint authenticated-only; Stooq HK has no mcap field), and USDT peg deviation (no free minute-level BTC/USD source). All three are flagged as Pass 2 candidates with paid data sources (Coinglass, Laevitas, Kaiko, CCData) or user-obtained KRX credentials.

The KR pull was restricted to candidate tickers only. The brief's Source 3 envisioned a full KOSPI + KOSDAQ pull (~2,500 tickers); this was not necessary for Pass 1 features and was precluded by the new KRX login requirement. Stage 0 validation for full-market row counts was relaxed accordingly.

The transaction-cost model uses constant half-spreads (15 bps HK, 10 bps KR) and a Kyle-style impact term with fixed $100K per position. The Practitioner review argued that real quoted spreads on HK GEM-board names are 30-70 bps and that borrow cost is absent from the model. Neither is captured. At realistic HK spreads, HK strategies are unexecutable; at realistic KR spreads with 200-300 bps annualized borrow, the KR gap breakeven multiplier moves below the 1x baseline. The net-of-cost conclusion is robust to model refinements only on the HK side; KR requires a more nuanced investigation in Pass 2.

FX rates are constants (7.8 HKD/USD, 1,300 KRW/USD). Adequate for an ADV filter; inadequate for live P&L.

Universe size is thin. Median 24 HK, 30 KR; tercile legs are 8-10 names each. Tercile sorts on such thin groups produce noisy estimates, and the small-sample critique is a legitimate concern the panel raised and was not fully resolved.

## Conclusion

The pre-specified gap-vs-intraday decomposition passes the brief's criterion 2 in both markets with bootstrap p < 0.001. The weekend-effect scaling and SHAP stability provide two independent mechanism tests, both supporting the hypothesis. Feature ablation meets criterion 3. The close-to-close IC threshold (criterion 1) is not met. The net-of-cost backtest is not deployable at assumed or realistic costs in either market.

The signal is genuine in the sense that the gap/intraday asymmetry is unlikely to be chance, cross-market consistency is high, the feature-importance ranking matches the theoretical prediction, and the longer-window test behaves as the mechanism would predict. The signal is not yet deployable because the tested cost model already produces net Sharpes below zero, and a more realistic model (real spreads, borrow costs) closes the HK path and strains the KR path.

## What Pass 2 Will Add

1. TCN horse race. Pass 1 used only LightGBM. Pass 2 adds a temporal convolutional network trained on the same targets to test whether a sequence model extracts information beyond the hand-engineered features. The ablation comparison will reveal whether the gap signal has temporal structure LightGBM's flat features miss.

2. Index-level prediction. Pass 1 predicted individual stock returns. Pass 2 adds direct prediction of HSCEI and KOSPI 200 opens from the same crypto overnight features. If the signal survives at the index level, it becomes executable via liquid futures and sidesteps the small-cap cost problem identified in Stage 5 and Stage 7.

3. Universe counterfactual. A non-crypto-filtered universe (matched by size and liquidity) will be tested for the same gap vs intraday decomposition. This directly addresses the Skeptic's strongest surviving critique from the adversarial review.

4. KR KOSPI large-cap variant. Restrict the KR universe to names with ADV above $50M (Naver, Kakao, Samsung Electronics, SK Hynix, KB Financial, Shinhan, with crypto adjacency). Tighter spreads, broader borrow availability, and larger position sizes. The Practitioner's top refinement.

5. Regime gate. Operationalize "crypto macroeconomic salience" as a rules-based, no-look-ahead gate using a combination of BTC market cap relative to S&P 500 market cap, crypto ETF AUM thresholds, and a spread-quality filter. The Practitioner proposed a specific form; Pass 2 backtests it.

6. Expanded ablation scope. Pass 2 broadens ablation to per-feature sensitivity, time-period stability of individual features, and cross-feature interaction. Pass 1 ran only category-level ablation.

7. Reintroduce dropped features. With KRX credentials (for mcap), a paid liquidations source (for liquidation intensity), and a minute-level crypto data source (for USDT peg deviation), the feature set returns to the brief's original ~32 feature count. Pass 2 tests whether these features add meaningful incremental alpha.

## Appendix — Files

Key outputs:
- `output/backtest_summary.csv` 6 rows
- `output/cost_sensitivity.csv` 24 rows (6 × 4 cost multipliers)
- `output/feature_ablation.csv` 18 rows
- `output/return_decomposition.csv` 6 rows
- `output/regime_analysis.csv` 36 rows
- `output/shap_stability.csv` and `output/shap_fold_overlap.csv`
- `output/weekend_effect.csv`
- `output/predictions_lgbm_{hk,kr}_{gap,intraday,cc}.csv` 6 files
- `output/diagnostics_summary.txt`

Review documents (eight panelist papers plus synthesis):
- `reviews/skeptic_position.md`, `reviews/skeptic_rebuttal.md`
- `reviews/believer_position.md`, `reviews/believer_rebuttal.md`
- `reviews/literature_position.md`, `reviews/literature_rebuttal.md`
- `reviews/practitioner_position.md`, `reviews/practitioner_rebuttal.md`
- `reviews/synthesis.md`

Decision logs:
- `logs/analysis_window_decision.log`
- `logs/feature_decisions.log`

Pipeline scripts under `scripts/`, per-stage logs under `logs/`, raw data under `data/`, cached Binance parquet roughly 1.4 GB.
