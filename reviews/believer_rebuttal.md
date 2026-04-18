# Believer Rebuttal — Round 2

**Role:** The Believer  
**Date:** 2026-04-16

---

## Response to Skeptic

The Skeptic's strongest claim is that stock-level features (20d vol, 20d momentum, prior-day return)
dominate crypto features in the ablation, therefore this is a momentum/reversal model in disguise.
This argument conflates absolute importance with incremental attribution, and it fails the mechanism
test.

**Short-term reversal alone cannot produce a gap-vs-intraday asymmetry.** If the model were purely
capturing cross-sectional momentum/reversal, removing crypto features would reduce the gap model and
intraday model symmetrically — reversal has no reason to concentrate at the open. The data shows
gap IC is 3.1x intraday IC in HK and 2.6x in KR. The ablation Sharpe drops (−1.1 HK, −1.8 KR)
are the *extra* loss from removing a feature group that mechanistically should affect the gap
window specifically. That asymmetry cannot be explained by reversal. The Skeptic's narrative
requires explaining why reversal-only strategies produce a pre-specified gap-vs-intraday IC split
that exactly matches the overnight crypto channel hypothesis. It cannot.

**The circularity critique is testable but not tested.** The Skeptic argues that selecting on
BTC correlation then predicting with BTC overnight returns produces IC mechanically. The falsifying
test is simple: does a univariate BTC-beta model (predict gap return = BTC overnight return × beta)
achieve the same 0.06 IC? If so, the LightGBM with 21 features adds nothing and the circularity
critique is fatal. If not, the multivariate model's outperformance *is* the added value of the full
crypto feature set. The SHAP evidence is directional but not conclusive here; the Skeptic correctly
identifies this gap. However, the burden is symmetric: the Skeptic cannot assert the IC *is*
mechanically inflated without demonstrating the univariate benchmark.

**Bonferroni survival:** The Skeptic claims multiple testing inflates the results. The primary test
is gap IC in both markets simultaneously. With 12 primary tests and Bonferroni threshold p < 0.0042,
the gap IC bootstrap p-values of 0.000 in both markets survive cleanly. The HK close-to-close IC
(p = 0.036) does not, and I concede that secondary finding. The regime IC differentials lack
reported p-values and cannot be used as confirmatory evidence under multiple testing discipline.
I acknowledge that secondary narrative. The core finding — pre-specified gap-vs-intraday split,
cross-market, p < 0.001 — is not touched by the multiple testing critique.

---

## Response to Literature Reviewer

The Literature Reviewer raises two points: (1) LPS warns that overnight anomalies often fail in
emerging markets; (2) Harvey-Liu-Zhu multiple testing adjustment is warranted.

**HK and KR are not the markets LPS warns about.** The LPS emerging-market caveat applies to
thin-price-discovery markets: LatAm frontier names, MENA exchanges with restricted foreign access,
sub-$1B market cap universes without institutional participation. Hong Kong's equity market
aggregates hundreds of billions in daily turnover; it hosts one of the world's largest derivatives
markets; it functions as the primary offshore pricing venue for China-adjacent risk. Korea's KOSPI
and KOSDAQ together trade $15–30B daily. These are not the informationally impoverished markets
where LPS warnings apply.

More importantly, the cross-market convergence directly contradicts the "fragile emerging market
anomaly" hypothesis. Fragile anomalies are idiosyncratic — they appear in one market, fail in
another, vary by period. The HK and KR gap ICs are 0.0607 and 0.0610. If the LPS warning applied
here, we would expect market-specific dispersion: one market works, one does not, IC differences
of 0.02–0.04. Instead, we get convergence to the fourth decimal across two structurally different
markets. This is what a genuine mechanism produces, not what a fragile emerging-market artifact
produces.

**Harvey-Liu-Zhu:** Agreed in principle. The six model IC tests (2 markets × 3 targets) require
a corrected threshold of approximately p < 0.008 (Bonferroni at α = 0.05 / 6). The gap IC
p-values of 0.000 in both markets survive this. The Literature Reviewer has correctly flagged
the secondary tests — regime cells, ablation comparisons — as requiring explicit correction.
Those should be treated as exploratory, not confirmatory. I accept this framing.

---

## Response to Practitioner

The Practitioner makes three distinct points: (a) gross signal is real but net strategy is not
viable; (b) borrow costs and spread reality kill the economics; (c) performance is regime-concentrated
in 2025–2026 non-recurring events. I will take these in order.

**(a) Net viability.** Conceded for HK, substantially conceded for KR. The Believer position has
never argued that the current parameterization is deployable. The argument is that the information
channel is real. Grossman-Stiglitz equilibrium permits real signals that cannot be arbitraged at
scale due to transaction costs in thin universes. HK and KR small-cap crypto-exposed equities are
exactly the universe where this equilibrium should hold: high costs, low liquidity, persistent
mispricing that any reasonable desk would find prohibitively expensive to capture. The Practitioner's
cost critique validates the mechanism, it does not refute it.

**(b) Borrow and spread reality.** The Practitioner is correct and I have no rebuttal on the
specifics. The HK GEM-adjacent names at 30–70 bps realized spread versus 15 bps assumed, combined
with unavailable or costly borrow, make HK structurally unexecutable. The KR case with realistic
20 bps spreads and 200–300 bps borrow is also net-negative. A KOSPI large-cap sub-strategy
(Naver, Kakao, Samsung, SK Hynix) with realistic cost assumptions is the only viable forward
path, and that requires a separate backtest. These points score.

**(c) Regime concentration — the central empirical question.** The Practitioner claimed performance
is concentrated in 2025–2026. I can now address this with fold-level data from the training logs.

**HK gap model — fold-level IC by year (ic_test):**

| Year | Mean OOS IC | N folds | N positive |
|------|-------------|---------|------------|
| 2020 | 0.033       | 1       | 1          |
| 2021 | 0.031       | 12      | 10/12      |
| 2022 | 0.053       | 12      | 11/12      |
| 2023 | 0.053       | 12      | 10/12      |
| 2024 | 0.080       | 12      | 10/12      |
| 2025 | 0.076       | 12      | 10/12      |
| 2026 | 0.100       | 4       | 4/4        |

**KR gap model — fold-level IC by year (ic_test):**

| Year | Mean OOS IC | N folds | N positive |
|------|-------------|---------|------------|
| 2020 | 0.060       | 11      | 11/11      |
| 2021 | 0.070       | 12      | 11/12      |
| 2022 | 0.091       | 12      | 11/12      |
| 2023 | 0.001       | 12      | 5/12       |
| 2024 | 0.038       | 12      | 11/12      |
| 2025 | 0.095       | 12      | 12/12      |
| 2026 | 0.098       | 4       | 3/4        |

This data is both supportive and honestly damaging for different markets.

**HK is not regime-concentrated.** The IC is positive and meaningful in every year from 2021
onward (0.031, 0.053, 0.053, 0.080, 0.076, 0.100). The 10–11/12 fold-positive rate is consistent
across all years. The signal strengthens post-2024 but is not concentrated there — 2022 and 2023
both show 0.053 mean IC. The Practitioner's regime-concentration critique does not hold for HK.

**KR 2023 is a genuine failure that must be acknowledged.** KR shows mean IC of 0.001 in 2023,
with only 5 of 12 folds positive. This is noise by any standard. It coincides with the KOSDAQ
short-sale ban (November 2023–March 2024) and a period of broad crypto stagnation before the
ETF approvals. The KR signal did not work in 2023. This is a real finding, and the Practitioner
is correct that it is masked by the full-period aggregate.

However, the Practitioner's framing of 2025–2026 as "non-recurring event-driven" performance
that carries the result is only half-true. For KR, the signal also worked clearly in 2020–2022
(mean IC 0.060–0.091). 2023 was the anomaly, not the rule. The structural interpretation is that
KR signal operates when crypto markets have macro salience for Asian equity investors — 2020–2022
(DeFi boom, crypto mainstreaming) and 2025–2026 (spot ETF, institutional integration) — and
attenuates in 2023 when crypto became a sideshow after the bear market and before institutional
legitimization. That is consistent with the proposed mechanism, not evidence against it.

The Practitioner deserves credit for identifying the 2023 KR gap. I incorporate it.

---

## Refined Position

The core finding stands: a pre-specified, cross-market, overnight crypto information channel
into Asian equity gap returns is real, statistically robust (p < 0.001, both markets), and
mechanistically coherent (3:1 gap-vs-intraday asymmetry, 65-hour Monday enhancement, SHAP
stability across 75 folds). These properties cannot be produced by pure reversal/momentum noise
or universe selection circularity alone.

What the Practitioner and fold-level data force me to concede: the signal is not uniformly
distributed across time. KR 2023 is a genuine gap year (5/12 folds positive, near-zero mean IC)
that the full-period aggregate obscures. The mechanism appears to require active crypto salience
in equity investor attention — when crypto is a macro factor (2020–2022, 2025–2026), the channel
operates; when crypto was in a reputational trough (2023), it attenuates in KR specifically.

The refined claim is therefore: **the overnight crypto-to-equity-gap information channel is real
but episodic, not uniformly structural.** It is strong when crypto carries macro risk factor status
for Asian institutional investors, and weak when it does not. This is a more nuanced finding than
"perpetual alpha source" and a more accurate one. The WRITEUP should frame the result as
evidence for a conditional mechanism tied to crypto's macroeconomic salience, with explicit
acknowledgment of the 2023 KR attenuation, and should recommend regime-gating on crypto market
attention proxies (e.g., BTC trending positive, spot ETF AUM growth) as a condition for live
deployment. The HK evidence is cleaner and more consistent year-over-year; the KR evidence is
stronger in peak years but requires a regime gate.

The mechanism is real. The strategy, in its current form, is not deployable. These are compatible
findings.
