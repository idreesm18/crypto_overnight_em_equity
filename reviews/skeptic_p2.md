# Skeptic Position Paper — Pass 2

**Role:** The Skeptic
**Round:** P2-12, Round 1

---

## 1. The Central Control Result: HK Offers No Crypto-Specific Finding

The Pass 2 brief positioned the control universe test as the definitive answer to the circularity critique raised in Pass 1. The result for HK decisively fails that test in the wrong direction.

From `output/control_vs_main_comparison.csv`:

| Market | Main gap IC | Control gap IC | Diff | p-value |
|--------|------------|---------------|------|---------|
| HK | 0.0610 | 0.0514 | 0.0096 | 0.927 |
| KR | 0.0608 | 0.0371 | 0.0237 | 0.008 |

In HK, the control universe gap IC of 0.051 is not significantly different from the main gap IC of 0.061 (bootstrap p = 0.927). The null of equal IC cannot be rejected. More damaging: the control gap:intraday ratio is 4.80, compared to 4 only 3.14 for the main universe. The non-crypto-correlated stocks exhibit stronger gap dominance than the crypto-correlated stocks. If the gap IC pattern were driven by overnight crypto information, the main universe should have substantially higher gap IC and a higher ratio. The opposite pattern holds in HK. The HK finding is a short-term cross-sectional reversal effect, not a crypto information effect.

The diagnostics file classifies this under "Rule B — framing uncertain." That is diplomatic. The data are clear: the HK result provides no evidence of a crypto-specific channel. It provides evidence of a gap-dominated reversal pattern that exists equally in stocks with no crypto exposure.

---

## 2. KR Control Difference: Sampling Variance and Multiple Testing

KR shows a statistically significant control difference (p = 0.008, diff = 0.024). Before treating this as confirmation of a crypto-specific channel, apply appropriate corrections.

**Multiple testing across the expanded test family.** Count the tests conducted in Pass 2: 6 primary IC comparisons (main_hk/kr, control_hk/kr, index_hk/kr across targets); 2 gap:intraday ratio comparisons; 14 feature ablation group tests; 40+ per-feature ablation tests; 6 regime cells per universe (6 universes); 3 model comparisons per universe; horse race entries across all configs. A conservative lower bound is 100+ tests. Bonferroni at alpha=0.05 requires p < 0.0005. The KR control p-value of 0.008 fails this correction. A Holm-Bonferroni step-down procedure applied to the full test family would likely reach the same conclusion. Reporting the KR result as significant without Bonferroni adjustment is a multiple-testing error across an expanded search space.

**Residual selection in the control CSV.** The control universe is described as non-crypto-correlated, but the selection procedure itself is data-driven. If control membership is determined using the same sample period as the backtest (or with any look-ahead in the correlation threshold), the control IC is not a clean baseline. The 0.037 control gap IC in KR is not zero -- it is elevated, consistent with a gap reversal pattern common to volatile emerging market equities regardless of crypto exposure. The difference between 0.061 and 0.037 may reflect different levels of a common factor (gap reversal), not a crypto-specific increment.

---

## 3. Index IC Metric Is Not Defensible as OOS

Index-level results pass acceptance criterion 1 with rolling time-series IC of 0.185 (HK) and 0.277 (KR), both p < 0.001. The skeptic challenges the metric itself.

**Rolling 30-day time-series IC on a 2-ticker universe is not cross-sectional IC.** For a single ticker per market, "cross-sectional IC" is undefined. The reported metric is an autocorrelation of the model's prediction with the next-day return over rolling 30-day windows. A 30-day rolling window smooths noise mechanically: if predictions have even modest serial correlation, the rolling IC will appear stable and significant even when the underlying daily hit rate is near 50%.

**Autocorrelation inflates rolling IC significance.** The model is re-trained with a growing window. Training on recent data will produce predictions that are themselves autocorrelated (momentum in predictions even when underlying returns are not), and the Newey-West or bootstrap correction must account for autocorrelation in the rolling IC series, not just the raw return series. If the reported p-values use the raw IC series without HAC correction for autocorrelation in the IC itself, the p-values are understated.

**The strategy uses 2 tickers.** The index threshold strategy on index_hk and index_kr trades a single index position per day when the threshold fires. From `output/backtest_summary_pass2.csv`, gate-off pct_days_invested is 39% (HK) and 59% (KR). This is a binary long/short market timing strategy on two assets, not a cross-sectional signal family. An IC of 0.277 on a 2-asset daily time series over 6 years (approximately 1,500 observations) is a different claim than an IC of 0.06 over a 30-stock cross-section. The realized significance depends entirely on whether the daily directional bets are independent and whether the OOS evaluation is genuinely hold-out.

---

## 4. TCN Failure Undermines Temporal Structure Claims

From `output/horse_race_bootstrap.csv`, TCN significantly underperforms LGBM on main universe gap IC:

| Market | IC diff (TCN - LGBM) | p-value |
|--------|---------------------|---------|
| HK | -0.0206 | 0.006 |
| KR | -0.0281 | 0.000 |

Criterion 3 fails in both markets. The brief treats this as a failure of the TCN variant. I treat it as evidence against the signal itself.

TCN is designed to exploit temporal structure in minute bars. If overnight crypto information creates a predictable component in gap returns through a mechanism with temporal texture (return patterns within the overnight window, volume dynamics, funding rate evolution), a well-specified TCN operating on minute-level data should capture it at least as well as 21 hand-engineered features. That it captures significantly less is evidence that the signal, to the extent it exists, is concentrated in the summary statistics already in the feature set, not in genuine temporal structure. The gap IC is a function of a few aggregated statistics, not a persistent temporal pattern.

The TCN overfitting flags reinforce this: train IC 0.33-0.38 vs OOS 0.03-0.09 for TCN on main universe gap (from `output/diagnostics_summary_pass2.txt`). The model memorizes training folds and finds no generalizable temporal pattern. The differential between LGBM and TCN OOS IC is 0.021-0.028, both highly significant. Any claim about temporal structure in overnight crypto information should explain why the model designed to exploit that structure fails so completely.

---

## 5. Backtest Results: Index Strategies Are Not Evidence of a Signal Family

Eight index configurations clear +0.5 net Sharpe; zero stock-tercile configurations do. This asymmetry is presented as positive. It is concerning.

**Two assets is not a signal family.** All 8 clearing configurations are index_hk or index_kr with LGBM. The index threshold strategy executes one long or short position per day when a gap threshold fires. There is no diversification across names, no cross-sectional spread, and no evidence that the IC generalizes beyond the market-level direction call. A single strategy on two assets that happened to have Sharpe above 0.5 in a 6-year backtest is consistent with chance at the 5% level without any adjustment for the number of configurations tested.

**Cost assumptions for index are not stress-tested.** The spread cost for index strategies is 2 bps per trade (from `output/backtest_summary_pass2.csv`, spread_cost column shows 0.26-0.30 per year for gate-off configs). For a strategy that is invested 39-59% of days, this is approximately 0.4-0.6 bps effective daily cost. This may be achievable for a liquid futures index, but: (a) no implementation vehicle is specified; (b) the strategy fires on overnight gap signals, meaning executions at the open may face wider effective spreads than the 2 bps assumption; (c) market impact for a position that size-adjusts based on a binary signal is zero in the model but nonzero in practice.

**Gate reduces Sharpe substantially and costs are counted as inactivity.** From `output/regime_gate_comparison.csv` (diagnostics file), the gate-on Sharpe is 2.69 vs gate-off 3.74 for index_kr gap. The cost_of_inactivity is 0.235, meaning the gate destroys 6.3% of the gate-off Sharpe. A regime gate that reduces strategy Sharpe by this amount while being presented as a risk management improvement is not an improvement. It is a signal that the gate does not identify regimes where the model is more accurate -- it simply reduces exposure and thereby reduces both return and risk proportionally, landing at a lower Sharpe.

---

## Bottom Line

HK provides no evidence of a crypto-specific information channel: the control gap IC is not significantly different from the main gap IC, and the control gap:intraday ratio exceeds the main ratio. This finding survives no interpretation as crypto-specific. KR shows a statistically significant control difference (p = 0.008), but this result fails Bonferroni correction over the expanded Pass 2 test family (>100 tests, required threshold p < 0.0005), and the magnitude of the difference (0.024) is plausibly attributable to different gap-reversal baseline levels across two non-comparable universes, not crypto information. The index strategies are two-asset market timing, not a generalizable cross-sectional signal. The TCN's complete failure to add incremental value over hand-engineered features is evidence against temporal structure in the signal. What survives: a gap-dominated reversal pattern in crypto-correlated EM equities that is real, present in both markets, and not tradeable after realistic costs in either market. What does not survive: the claim that this pattern is crypto-specific rather than a cross-sectional reversal artifact on volatile equity universes.
