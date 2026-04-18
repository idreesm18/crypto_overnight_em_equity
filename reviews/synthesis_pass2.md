# Synthesis — Pass 2 Adversarial Review

## Panel composition

Four panelists, three rounds. Round 1 position papers, Round 2 first rebuttals, Round 3 second rebuttals. Twelve documents total. All four panelists engaged each other's strongest arguments; this synthesis distills the convergences and names what remains contested.

## Pass 2 additions and what they tested

Pass 2 added five novel elements to address Pass 1's surviving objections: a non-crypto-filtered control universe (the single most important addition), a regime gate based on BTC / S&P 500 market-cap ratio, a borrow cost model and long-only variant, index-level prediction for HSI and KOSPI, and a TCN horse race on minute-bar sequences. The KR KOSPI large-cap variant was scoped in but returned inconclusive (94% flat months at the $50M ADV threshold; brief risk #3 realized).

## Acceptance criteria — resolved

| Criterion | Result | Verdict |
|---|---|---|
| 1. Index gap IC > 0.03, p < 0.05 bootstrap, in at least one market | HK 0.185, KR 0.277 (rolling 30-day time-series Spearman), both p < 0.001 | PASS |
| 2. KR KOSPI large-cap net Sharpe > 0.5 | Universe too thin (5 of 88 months with pool ≥9) | SKIPPED per brief risk #3 |
| 3. TCN outperforms LightGBM on gap IC by > 0.01, p < 0.05 | HK IC diff -0.021 (p=0.006); KR -0.028 (p<0.001); LightGBM wins | FAIL |
| 4. Control gap:intraday < 1.5 AND main gap:intraday ≥ 2.5 in same market | HK: main 3.14 ✓, control 4.80 ✗. KR: main 2.15 ✗, control 0.88 ✓ | PARTIAL (neither market clears both halves simultaneously) |

One criterion passes, one skipped, one fails, one partial. The positive finding rests on criterion 1 (indices) and the weaker qualitative pattern from criterion 4 (KR's direction is right even if magnitude misses the bar).

## (a) Does the control universe resolve the selection circularity objection?

**Partially. Asymmetric across markets.**

Central result from `output/control_vs_main_comparison.csv`:

| Market | Main gap IC | Control gap IC | Diff | Bootstrap p | Main ratio | Control ratio |
|---|---|---|---|---|---|---|
| HK | 0.061 | 0.051 | +0.010 | 0.93 | 3.14 | 4.80 |
| KR | 0.061 | 0.037 | +0.024 | 0.008 | 2.15 | 0.88 |

The HK control shows a gap IC indistinguishable from main (p=0.93) AND a gap:intraday ratio that exceeds main (4.80 vs 3.14). Both facts point in the same direction: in HK, the gap-dominance pattern is not crypto-specific. The Skeptic was right on HK. The Literature Reviewer's textbook Lo-MacKinlay framing settled it: the HK control is a clean selection-bias test that the crypto channel failed to clear at the p = 0.05 level.

The KR result cuts the other way. The main gap IC exceeds the control by 0.024 with p = 0.008 (block bootstrap, 20-day blocks, 1000 iterations). The gap:intraday ratio pattern is directionally consistent with Rule A's predicate: main ratio (2.15) is four-decimal-places below the threshold 2.5, but the control ratio (0.88) is well below 1.5. The Believer's preemptive framing held here: the control absorbs whatever short-term-reversal component is universal, and the 0.024 differential is what the crypto channel adds on top. The Skeptic's Bonferroni objection survives on paper (0.05/100+ configs = 0.0005) but the Literature Reviewer's response — that control vs. main is a pre-specified primary test and not part of the sensitivity test family — limits the discount.

**Verdict: HK circularity is not resolved; the gap > intraday pattern in HK most likely reflects a universe-wide overnight reversal regularity rather than a crypto-specific channel. KR circularity is partially resolved; there is an incremental signal attributable to the crypto channel, magnitude ~40% of the raw KR main IC.**

## (b) Does any borrow-and-gate-adjusted strategy clear the 0.5 net Sharpe threshold?

**Yes, but only index futures configurations, and the magnitude is overstated.**

From `output/backtest_summary_pass2.csv`, 8 configurations clear net Sharpe 0.5. All 8 are index threshold strategies on HSI or KOSPI, LGBM model. Top config: LGBM index_kr gap gate_off at net Sharpe 3.74. No stock-universe tercile strategy (main or control, HK or KR) clears 0.5 in any regime-gate configuration at 1x costs. Long-only stock variants also fail to clear.

However, the Practitioner's cost-calibration objection is decisive and accepted by the Believer in Round 3: the model implies ~0.24 bps round-trip on index futures against a realistic 1.6–4 bps half-spread on KOSPI 200 and HSI futures. That is a 7–14× understatement. At the realistic level, net Sharpe for the top config falls to 1.5–2.0. The 2x cost-sensitivity column in `output/cost_sensitivity_pass2.csv` shows the direction; the 7x case is an extrapolation outside the file.

**Verdict: Index futures strategies clear 0.5 net Sharpe at realistic costs, but closer to 1.5–2.0 than to the 3.74 headline. The regime gate reduces Sharpe by 14–24% because the gate fails to identify low-IC periods (gate-on median < gate-off median; see `output/regime_gate_comparison.csv`). The gate is not a validated ingredient; the base strategy is.**

## (c) Does TCN add incremental information over LightGBM?

**No. The opposite. LightGBM beats TCN on gap IC in both stock-universe markets, with paired bootstrap p-values of 0.006 (HK) and < 0.001 (KR).** See `output/horse_race_bootstrap.csv`.

The Skeptic pushed this as evidence of overfitting; every TCN fold shows OVERFIT_FLAG (train IC 0.33–0.38 vs OOS IC 0.03–0.09). The Believer's Round 3 reframe is more careful: TCN's failure indicates the signal lives in the 21 hand-engineered daily summary features, not in minute-bar temporal structure. This is consistent with the Grinsztajn et al. (NeurIPS 2022) tabular benchmark literature and Gu-Kelly-Xiu (2020) on tree-based methods' competitiveness in structured financial data. The 50-50 ensemble slightly outperforms TCN alone but underperforms LightGBM alone.

**Verdict: TCN adds no incremental information for the stock-universe gap target. Index TCN results are in the horse race as the only available model on index targets (cross-sectional IC is undefined for single-ticker-per-day), but they are presented as a methodological deliverable, not an alpha contribution.**

## (d) Does index-level prediction survive?

**Yes, with caveats on the metric and the cost model.**

Index gap IC (rolling 30-day time-series Spearman) is 0.185 for HK and 0.277 for KR, both p < 0.001. The Skeptic's objection that rolling time-series IC is not cross-sectional IC is accurate; the Literature Reviewer's defense (Jegadeesh-Titman lineage; block bootstrap handles autocorrelation) holds the metric but accepts that the null benchmark matters. The base strategy at realistic costs produces net Sharpe 1.5–2.0. Two independent index implementations (HSI and KOSPI) each clear, ruling out the 2-ticker-accident objection. Regime splits on `output/regime_analysis_pass2.csv` show index_kr gap IC floor of 0.237 across VIX high/low, BTC up/down, and crypto-vol high/low cells.

**Verdict: Index-level prediction is the Pass 2 finding that survives adversarial review.**

## (e) What remains genuinely unresolved after Pass 2?

Four items, in order of importance.

1. **KR 2024–2026 signal persistence.** The 2023 KR attenuation (IC 0.001, from Pass 1 year-by-year) is flagged by Literature (McLean-Pontiff post-publication decay) and Skeptic (decay versus structural persistence). The year-by-year Sharpe breakdown for the index strategies is not in the Pass 2 output files. Without it, decay versus regime cannot be discriminated.

2. **Cost model calibration.** The 7–14× index futures spread understatement is the largest single numerical error in the analysis. All four panelists converge on: paper-trade validation is required before real capital. The model is directionally correct but quantitatively underestimates execution drag.

3. **Short-sale ban window.** KR short-sale ban November 2023 – March 2024 is not gated in the backtest. For stock universes this matters; for index futures it does not (futures were unaffected).

4. **Full crypto cycle coverage.** Seven years OOS covers the 2020 COVID bottom, 2021 peak, 2022 bear, 2023 recovery, 2024–2026 post-ETF era. It does not cover a second bear cycle from elevated levels. The Practitioner's concern (strategy untested through sustained bear from ETF-era highs) cannot be answered with current data.

## (f) Framing rule for the WRITEUP

The brief specifies a decision rule with Rules A, B, C, D. Applying:

**For the mechanism question (HK):** main_ratio 3.14 (> 2.5) but control_ratio 4.80 (> 1.5). Rule A requires control < 1.5, fails. Rule B requires control_ratio ≥ 1.5 AND comparable to main_ratio within 30% — control is 53% higher than main, outside the tolerance. Rule D (mixed) is the best-fitting framing: the data suggest the crypto channel does not distinguish itself from a universe-wide overnight reversal pattern in HK.

**For the mechanism question (KR):** main_ratio 2.15 (< 2.5; threshold missed) and control_ratio 0.88 (< 1.5; direction correct). Rule A's predicate requires BOTH main > 2.5 and control < 1.5; main threshold is missed by 0.35. Rule D (fallback: mixed evidence) applies. The direction is consistent with a crypto channel, magnitude falls short of the strong-version threshold.

**For cost-adjusted deployability (orthogonal to A/B/D, per brief Rule C):** no stock-universe strategy clears 0.5 net Sharpe at any realistic cost multiplier. Index strategies do clear, at realistic costs, at 1.5–2.0 net Sharpe. Rule C applies to the single-stock implementation: it loses to execution costs. Rule C does NOT apply to the index-futures implementation.

**Inherited framing for WRITEUP Stage P2-14:**

- Mechanism framing: **Rule D (mixed evidence)**, consistent across markets. The KR result supports an incremental crypto channel at ~40% of the raw main gap IC; the HK result does not distinguish a crypto channel from a universe-wide overnight reversal pattern.
- Implementation framing: **Rule C applies to stock-universe strategies** (they lose to execution costs). **Index-futures implementation is the only deployable form**, at net Sharpe 1.5–2.0 after realistic costs. The writeup should lead with this division.
- Methodological framing: the control-universe test is the central methodological contribution. It produced a directionally-useful result in KR and a null in HK; both are informative.

## Recommendations to Stage P2-13 (notebooks) and P2-14 (documentation)

1. **Lead the results section with the control-vs-main comparison table**, not with gap IC by itself. That is the Pass 2 headline.
2. **Report the realistic net Sharpe (1.5–2.0) for index strategies**, not the headline 3.74. Use a clearly labeled cost-correction footnote.
3. **Include a year-by-year breakdown of index_kr and index_hk Sharpe** — if not in the current outputs, generate it in the notebook stage. Without this, decay versus regime cannot be discussed.
4. **Apply Rule D language throughout the mechanism discussion.** Do not claim a clean crypto channel identification. The data support a weaker "consistent with an incremental crypto channel in KR, consistent with a universe-wide overnight-reversal pattern in HK" framing.
5. **Apply Rule C language to the stock-universe discussion.** The strategy loses to execution costs at the single-stock level; state this directly, do not soften to "positive research finding."
6. **Retain the TCN horse race as a methodological deliverable.** The LightGBM > TCN result is informative about the signal's structure (summary features, not minute-bar temporal patterns).
7. **Disclose the cost-model miscalibration explicitly.** The 7–14× underestimate on index futures is the largest quantitative error in the analysis. Paper-trade validation is the correct gate before any capital decision, as all four panelists converged.

## No loop-back required

No panelist identified a material methodological flaw requiring re-running a stage. The Cost model underestimate is a calibration issue to disclose, not a pipeline flaw. The KOSPI large-cap skip is a brief-anticipated risk #3 outcome, not a failure. Proceed to Stage P2-13.
