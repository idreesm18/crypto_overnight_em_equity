# Literature Reviewer Rebuttal — Round 2
## Crypto Overnight EM Equity Signal (Pass 1)

**Role:** Literature Reviewer  
**Date:** 2026-04-16

---

## Literature on Circularity

The Skeptic's claim that "universe circularity makes the 0.06 IC mechanical" is the strongest methodological critique in this panel and deserves precise literature grounding. Lopez de Prado (2018, *Advances in Financial Machine Learning*, Chapter 8) identifies "selection-on-the-dependent-variable" as among the most pernicious forms of backtest inflation, and the construction here — filtering stocks by trailing BTC correlation, then predicting their returns using BTC overnight returns — is a textbook instance of the pattern he flags. Harvey, Liu, and Zhu (2016, *Review of Financial Studies*, "...and the Cross-Section of Expected Returns") require that any IC reported from an actively selected universe be judged against the full battery of implicit tests, not just the headline t-statistic; the Believer's Bonferroni calculation covers explicit model runs but does not address the universe-design degrees of freedom.

However, the literature is also clear that circularity is a question of *magnitude*, not a binary invalidation. The Skeptic's claim that the IC "partly measures the selection rule working as designed" is correct in direction but unquantified in magnitude. The falsifying test — compare gap IC on a non-BTC-filtered universe of same-country small-caps — was not run, and without it the contamination cannot be bounded. This is not a critique the Believer's SHAP stability argument resolves: SHAP stability confirms that the model learns the same feature ranking across folds, but that ranking may be stable precisely *because* the universe is persistently filtered to stocks for which BTC overnight returns are relevant. Stability of feature importance is consistent with both "genuine mechanism" and "persistent tautology."

The literature cannot resolve this without the counterfactual. What it can say is that a gap-to-intraday IC ratio of 3:1, concentrated in the overnight window, is harder to attribute purely to circularity than a uniform IC elevation across all return windows. Circularity would inflate gap and intraday IC equally (both are calculated on the same universe); the directional asymmetry retains some mechanistic content even in the presence of circularity. This is a partial defense of the LPS anchor — not a full one.

---

## Scope of Lou-Polk-Skouras Mechanism

The Skeptic's rebuttal raises the most technically correct objection in this panel: LPS (2019) is a population-level finding about the aggregate cross-section of US equities, not a prediction about themed sub-universes of 20–30 stocks pre-filtered by crypto co-movement. This distinction matters because the LPS mechanism depends on institutional market-on-open (MOO) order flow aggregating dispersed information across a broad cross-section; the information aggregation function is coherent at the population level and need not hold in concentrated themed portfolios.

To my knowledge, no published work has directly tested whether the gap-vs-intraday asymmetry identified by LPS extends to thematic equity sub-universes defined by cross-asset correlation. The closest analogues — thematic ETFs (ARK Innovation ARKK, Amplify Transformational Data Sharing BLOK, Grayscale-adjacent crypto-equity proxies) — are held as pooled instruments and do not permit the individual-stock gap decomposition that Pass 1 performs. Academic literature on crypto ETF overnight behavior does not, to my knowledge, exist at the individual-constituent level. This is an honest gap: the LPS extension claimed in Round 1 is plausible, the mechanism is directionally consistent with the gap-vs-intraday finding, but LPS cannot be called a theoretical anchor for what is empirically a novel sub-universe application.

The appropriate characterization is: Pass 1 *tests* whether the LPS mechanism extends to themed sub-universes with a cross-asset conditioning variable. The result (3:1 gap-to-intraday IC in both markets) is consistent with extension, but the architecture of the test — universe-filtered on the same variable that drives the signal — prevents a clean interpretation. The contribution is the test itself and its directional result, not a confirmed extension.

---

## Emerging vs. Developed Market Classification Issue

The Believer's Round 2 rebuttal argues that HK and KR are not the "thin, price-discovery-impoverished" markets where LPS warns about overnight anomaly decay. This is partially correct and partially wrong in ways the literature can adjudicate.

Hong Kong is unambiguously MSCI Developed Markets. The LPS warning about overnight anomaly decay in less developed markets does not apply to HK's aggregate market. However, Pass 1's HK universe is not the aggregate HK market — it is GEM-board micro-caps and main-board names with ADV at or near the $500K floor. For this specific sub-universe, the institutional MOO order flow that LPS's mechanism requires is sparse. The Practitioner documents this directly: 0863.HK (OSL Group) at HKD 1.50 implies a minimum quoted spread of 67 bps, 4.5x the model assumption. These names do not attract the institutional participation that makes the LPS aggregation mechanism work.

Korea is more complex. MSCI classifies KR as Emerging Markets (explicitly citing restricted foreign access and settlement infrastructure); FTSE Russell classifies it as Developed. The LPS caveat about emerging markets is not mechanistically about the MSCI label — it is about the quality of price formation at the open auction, which in KR's case is genuinely mixed: KOSPI blue-chips have deep institutional participation, but KOSDAQ names (the ones driving the crypto signal) have thinner order books. The KR 2023 attenuation — mean OOS IC of 0.001 across 12 folds, only 5/12 positive, coinciding with the KOSDAQ short-sale ban — is empirical evidence consistent with the LPS decay concern in thinner market conditions. The short-sale ban itself is not a price formation issue, but its correlation with the attenuation period suggests the signal depends on market-structure integrity that was temporarily suspended. This is not evidence that the mechanism is fragile in general; it is evidence that it requires a minimum market-functioning threshold.

The clean formulation: HK aggregate market is developed (LPS warning does not apply to aggregate); HK universe for this strategy is micro-cap and trades like an emerging market (LPS warning applies to the specific sub-universe). KR aggregate is institutionally developed by turnover but MSCI-classified as emerging; the KR universe is mixed (KOSDAQ micro-caps where LPS warning applies, KOSPI large-caps where it does not).

---

## Regime-Dependent Signals in Literature

The Believer's most important concession — that KR 2023 is a genuine failure year (IC 0.001, 5/12 folds positive) while 2020–2022 and 2025–2026 worked (mean IC 0.060–0.095) — raises a foundational question the field has grappled with: when a signal is positive in some regimes and near-zero in others, is it a finding or a hypothesis needing more data?

The relevant literature addresses this directly. Campbell and Cochrane (1999, *Journal of Political Economy*, habit formation) establish that asset pricing anomalies can be conditionally strong (in high marginal utility / risk-off states) and attenuated in low-surplus environments. Muir (2017, *Journal of Finance*, "Financial Crises and Risk Premia") shows that risk premia are concentrated in periods of financial stress, not distributed uniformly across regimes. More directly, Avramov and Chordia (2006, *Review of Financial Studies*) demonstrate that anomaly returns are conditionally time-varying and tied to macroeconomic states. None of this literature implies that regime-dependent signals are invalid — it implies they require regime-conditioning for robust out-of-sample performance.

The Believer's reframing — "the channel operates when crypto carries macro salience for Asian institutional investors" — is structurally consistent with the habit-formation / regime-conditional framework. KR 2023 attenuation coinciding with the KOSDAQ short-sale ban and the post-crash crypto bear market is interpretable as a regime where the mechanism's preconditions (institutional attention, freely functioning markets, crypto as a live macro factor) were absent. HK, by contrast, shows no comparable attenuation in 2023 (mean IC 0.053, 10/12 folds positive) — a cross-market difference that is itself informative: HK's signal persisted through the same crypto bear market that attenuated KR's, suggesting different structural sensitivities.

The field would classify this finding as follows: a signal that is positive in ~60–80% of monthly folds across 6 years (HK: 10/12 in most years; KR: mixed in 2023) but near-zero in identifiable regimes with interpretable macroeconomic causes is a *conditional* finding, not a null result. The literature precedent (Avramov-Chordia, Muir) supports framing it as a regime-conditional mechanism requiring explicit regime-gating in any forward application, not as evidence of spuriousness.

---

## Contribution Assessment

Having read all eight documents in this review panel, the literature placement of Pass 1 can be stated with reasonable precision.

Pass 1 is best characterized as **Option B** — an extension of overnight-transmission literature to crypto as a cross-asset signal source — with important qualifications. It is not a clean replication of LPS (2019), which requires a population-level cross-section and institutional MOO order flow that the universe here lacks at scale. It is not purely novel in the sense of establishing a new mechanism; the Hamao-Masulis-Ng (1990) overnight international transmission framework, the Lou-Polk-Skouras gap-vs-intraday decomposition, and the Bouri/Corbet crypto-to-equity spillover literature all provide antecedents for individual components of the result.

What is genuinely new: the application of the gap-vs-intraday decomposition to a cross-asset setting where the overnight information source is crypto markets and the receiving universe is curated by crypto co-movement. The 3:1 gap-to-intraday IC ratio confirmed in both HK and KR, pre-specified, is a clean empirical observation that the prior literature has not produced in this specific configuration. The fold-level data provides a concrete anchor: HK shows mean OOS IC of 0.053 in both 2022 and 2023 (11–10/12 folds positive), demonstrating that the signal was not purely 2025–2026 event-driven in that market, even as KR faltered in the same period with IC of 0.001 and only 5/12 folds positive. This cross-market divergence in the same calendar year is informative and unreported in prior literature.

The circularity problem documented by Lopez de Prado (2018) and the multiple-testing discipline required by Harvey-Liu-Zhu (2016) mean that the IC magnitude (0.06) should be treated as an upper-bound estimate pending a counterfactual test on a non-filtered universe. The primary finding — gap > intraday, cross-market, pre-specified, p < 0.001 — is robust to the corrections both panels required. Secondary findings (regime IC differentials, weekend effect in KR) are exploratory and should be labeled as such.

The WRITEUP should position Pass 1 as: *an empirical test of whether the Lou-Polk-Skouras overnight mechanism extends to cross-asset, cross-market settings with crypto as the information source, conducted on curated crypto-exposed equity universes in HK and KR. The directional result is confirmed (gap > intraday, 3:1 ratio, both markets); the magnitude is upward-biased by universe circularity; the mechanism is conditional on crypto carrying institutional macro-factor status (strongest 2020–2022, 2025–2026; attenuated in KR 2023). The result is a hypothesis-confirming empirical observation requiring further validation on non-filtered universes and large-cap sub-sets before treatment as a deployable signal.*

That is where Pass 1 sits: confirmed direction, uncertain magnitude, conditional regime structure, genuine methodological novelty in the specific cross-asset overnight decomposition, and unresolved circularity that the literature requires be tested before the IC can be taken at face value.
