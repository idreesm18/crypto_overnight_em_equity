# Skeptic Second Rebuttal — Pass 2, Round 3

**Role:** The Skeptic
**Round:** P2-12, Round 3

---

## What This Debate Has Settled Against Me

Three concessions are warranted after reading all eight documents.

The KR control gap:intraday ratio contrast (main 2.15 vs control 0.88) is a genuine structural distinction. This is not the same as saying p=0.008 is unambiguous, but the ratio contrast across HK and KR goes in the predicted direction: KR shows the crypto-channel pattern cleanly; HK does not. I overstated the symmetry of those two failures in Round 1.

Index futures as the implementation vehicle is the right frame. The Practitioner independently arrived at a realistic cost range of 1.6-4 bps per round-trip (versus the model's implied 0.24 bps) and concluded the strategy survives at something in the 1.5-2.5 net Sharpe range. That is a first-principles engineering judgment, not a backtest result. I accept it.

The long-only versus long-short decomposition -- long-only KR gap gross Sharpe 3.38 versus combined 3.77, with short leg net-negative after costs -- is load-bearing evidence. The alpha concentrates in the long leg. Beta exposure during strong crypto regimes is a valid concern, but the regime splits show IC floors above 0.237 even in BTC uptrend and VIX-low environments where beta provides the most tailwind. The regime-beta confound is weakened, not eliminated, by this evidence.

---

## The Cost Error Is Larger Than Believer Acknowledges

The debate has produced agreement that the model's implied cost is wrong. The Practitioner pegs realistic KOSPI 200 futures at 1.6-3.2 bps half-spread, versus the backtest's implied 0.24 bps per round-trip. That is a 7-14x understatement, not a 2-5x correction. At 7-14x, the cost sensitivity table in `output/cost_sensitivity_pass2.csv` tops out at 2x multiplier and does not answer the question. The Believer writes that "15-20x cost inflation would be required to kill the trade" -- but 15x is inside the plausible range for gap-open execution on a threshold-triggered strategy. On days when the signal fires, crypto overnight returns have already moved sharply, and the equity open is precisely when institutional order flow is most concentrated. Spreads on signal days are not the same as average spreads. The backtest does not differentiate.

At 7x realistic cost (the low end of the Practitioner's range), index_kr gap gate_off net Sharpe falls from 3.74 to approximately 1.5-2.0 by linear extrapolation. At the high end (14x), it approaches 1.0. Those are not "still surviving" numbers with comfortable margins -- they are minimum-threshold numbers where operational frictions (margin funding cost, fill quality at the open, exchange latency) could push realization below 1.0. The Practitioner proposes a 30-day paper trade as the correct gate, which is reasonable -- but that gate has not been passed. Treating "would trade pending paper trade" as validation is premature.

---

## LGBM Overfit Is Unresolved by TCN Failure

The Believer and Literature jointly argue that TCN's failure is informative about mechanism (daily summary statistic, not temporal sequence) rather than about signal integrity. This framing is coherent. What it does not address is the TCN OVERFIT_FLAG itself: train IC 0.33-0.38 versus OOS 0.03-0.09 for TCN on main gap universe (`diagnostics_summary_pass2.txt`). That is a 4-10x fold memorization ratio. TCN memorizes training data and finds no generalizable pattern.

The question this raises for LGBM is not answered by TCN's failure. LGBM OOS gap IC is 0.061 (main, both markets) with no OVERFIT_FLAG, but the gap between LGBM and TCN OOS IC is 0.028-0.029 (both markets, p=0.000). If TCN cannot generalize the overnight window pattern, and TCN is architecturally designed to find exactly that kind of temporal dependence, then LGBM's advantage is in its 21 hand-engineered features -- features that were selected with knowledge of what drives crypto-to-equity spillovers. That selection process is not independent of the outcome. The feature engineering process itself encodes prior beliefs about which statistics matter. LGBM succeeds by fitting those encoded beliefs against training data. Whether those beliefs survive in a world where the channel is already partially arbitraged (2023 attenuation, ETF adoption) is not answered by beating a mis-specified temporal model.

The 2023 attenuation is the concrete evidence. The Literature Reviewer accepts McLean-Pontiff decay as a real risk but does not assess its magnitude. The Practitioner's kill switch (rolling 6-month Sharpe below 0 in two consecutive 3-month windows) is a reasonable operational response, but it is a loss-limiting device, not evidence that the decay is not already underway. If index_kr gap Sharpe in 2023 was materially below the full-period 3.74, that year-by-year breakdown is missing from the output. The Practitioner explicitly flags this as a required gap before any investor pitch (`practitioner_p2.md`: "I do not have year-by-year index_kr Sharpe breakdown in the Pass 2 files, which is a gap before pitching").

---

## Regime Gate Anomaly Is Not Explained

The regime gate reduces index_kr gap Sharpe from 3.74 (gate_off) to 2.69 (gate_on), a 28% reduction, while cutting invested days from 58.8% to 38.5%. The Believer and Practitioner note this but treat it as a conservative choice, not a diagnostic. A regime gate that degrades Sharpe in the full-period backtest should improve OOS performance if it correctly identifies low-signal regimes. Its failure to do so -- gate-on performs worse than gate-off even in-sample -- is evidence that the regime indicator is not identifying genuine signal states. An effective gate should improve Sharpe by filtering out low-IC periods. This one does the opposite, suggesting the regime splits that show IC floor above 0.237 are not exploitable for timing. The regime stability result and the gate degradation result are in tension and no panelist resolved this tension.

---

## Remaining Skeptical Claims After Three Rounds

- **The cost error is not bounded by the sensitivity table.** The model implies 0.24 bps per round-trip; realistic KOSPI 200 execution is 1.6-4 bps. At the realistic range (7-14x model cost), net Sharpe falls to 1.0-2.0. The paper trade gate is the right test but has not been passed. Treating a 30-day paper trading intention as equivalent to demonstrated net performance is not justified.

- **LGBM OOS IC of 0.061 may be partly feature-selection overfit.** TCN -- the model that cannot overfit feature engineering -- achieves OOS IC of only 0.032-0.034 on the same data. The gap of 0.028 IC points is statistically significant (p=0.000) and is not explained by "signal is tabular not temporal." LGBM's advantage is partly structural (tabular data suitability) and partly the product of 21 engineered features designed to capture the hypothesized channel.

- **2023 attenuation plus regime gate degradation together suggest the signal may be decaying.** The year-by-year Sharpe breakdown for index_kr is absent from output files. Without it, the 3.74 full-period Sharpe cannot be verified as distributed across the full OOS window rather than concentrated in 2020-2022 crypto-bull and 2024-2026 post-ETF-adoption periods. McLean-Pontiff decay combined with ETF-driven institutional monitoring is a plausible mechanism for a structurally shrinking edge.

- **Seven years is one crypto cycle, and the strategy has not been tested through a full bear-to-bear sequence.** The sample covers the 2020-2021 bull, the 2022 bear, and the 2023-2026 partial recovery and ETF adoption. A strategy that earns most of its Sharpe in the 2020-2021 and 2024-2026 bull segments and attenuates in 2022-2023 is a crypto-beta expression in a novel index vehicle, not a structural information-speed anomaly. That claim cannot be resolved without the year-by-year tearsheet the Practitioner correctly identified as missing.
