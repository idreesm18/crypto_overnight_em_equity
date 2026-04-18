# Literature Rebuttal — Pass 2
**Role:** Literature Reviewer (Round 2 Rebuttal)
**Date:** 2026-04-17

---

## Introduction

The Round 1 papers share broad agreement: overnight gap IC is real, concentrated at the open, and consistent with prior work on information-speed differentials. Disagreements are methodological and implementational. This rebuttal corrects the Skeptic's over-conservative multiple-testing choice, flags the Believer's omission of McLean-Pontiff decay risk, and aligns with the Practitioner on EM short constraints while defending the index futures cost conclusion on Novy-Marx-Velikov grounds.

---

## vs. Skeptic: Bonferroni Is a Choice, Not the Standard

The Skeptic invokes Bonferroni correction over 100+ tests to dismiss the KR control difference (p = 0.008). Harvey, Liu, and Zhu (2016) -- which the Skeptic cites -- explicitly calibrate t-statistic thresholds to the correlation structure of tests, not raw test counts. Bonferroni assumes test independence; it is maximally conservative when tests are positively correlated, as they are here. Benjamini and Hochberg (1995) false discovery rate control and Romano and Wolf (2005) stepdown procedures would be substantially less punishing on this correlated family. Bonferroni is a conservative upper bound, not a consensus standard.

On rolling IC: autocorrelation inflating significance is a legitimate concern in principle, but the block bootstrap preserves temporal dependence and accounts for autocorrelation in the IC series. The correct null is a random-walk-prediction baseline, not zero -- as Jegadeesh and Titman (1993) established for rolling momentum evaluation. If the Skeptic believes the block length is mis-specified, that is a concrete and testable claim, not a general confound.

---

## vs. Believer: The Channel May Already Be Decaying

The Believer correctly identifies the KR control differential (p = 0.008) as the crypto-channel identification and the index IC ratios as the clearest signal -- agreed. But the Believer frames the 2023 IC attenuation as a regime artifact without engaging with the publication-decay literature.

McLean and Pontiff (Journal of Finance, 2016) document that anomaly returns decline after the signal becomes known, with the largest declines where arbitrage costs are lowest. The crypto-to-equity channel is not a published anomaly in the McLean-Pontiff sense, but institutional crypto desks and crypto ETF products proliferated through 2022-2024, lowering the information-speed differential that Lou et al. (2019) require for the channel to persist. If informed participants began monitoring the channel earlier than the backtest assumes, 2023 attenuation is the expected trajectory under McLean-Pontiff, not an anomaly.

On SHAP stability: 75% top-10 fold overlap for LGBM is literature-consistent with a real signal per Gu, Kelly, and Xiu (2020). But the TCN OVERFIT_FLAG -- train IC 0.33-0.38 versus OOS IC 0.03-0.09 -- means TCN memorizes fold-specific microstructure. LGBM stability and TCN instability are compatible but should not be conflated into a single claim about signal robustness.

---

## vs. Practitioner: The EM Short Literature Agrees

The Practitioner argues that KOSDAQ micro-cap borrow is optimistic at the default 400 bps and that long-only is the correct deployment form. The EM short-sale constraint literature supports this directly. Nagel (2005) and Jones and Lamont (2002) document that constrained shorting leads to overpricing persistence -- meaning short legs in these markets are structurally impaired, not merely expensive. The 2023-2024 KR short-sale ban is a concrete example. The Pass 2 long-short decomposition confirms the prediction: long-only KR gap Sharpe 3.38 versus a short leg that nets negative after costs. Long-only or index-futures-long is the literature-preferred form.

On index futures spreads: the Practitioner flags a potential 10x undercount (2-4 bps half-spread actual versus the model's implied 0.24 bps). Novy-Marx and Velikov (2016) find that anomalies in liquid asset classes survive realistic cost scaling far better than single-name anomalies. KOSPI 200 futures are among the most liquid futures globally -- the regime where Novy-Marx-Velikov predict survival. Cost sensitivity shows index_kr gap at Sharpe 2.31 at 2x spread. The spread assumption needs correction in any forward characterization, but the framework supports survival at realistic futures costs.

---

## Where the Literature Leaves Us

The literature's verdict is mixed in a structured way. The gap-dominance result is consistent with three independent mechanisms (Lou et al. 2019; Hendershott et al. 2020; Berkman et al. 2012). The crypto channel shows incremental identification in KR but not HK, exactly as the selection-bias literature predicts (Lo-MacKinlay 1990; Harvey-Liu-Zhu 2016). The EM short-constraint literature supports long-only or index-futures-long as the implementable form. What the literature cannot resolve is whether the 2023 IC attenuation marks McLean-Pontiff-style decay or a transient regime effect -- that requires additional OOS years. Document the result; do not frame it as a stable persistent anomaly pending that resolution.
