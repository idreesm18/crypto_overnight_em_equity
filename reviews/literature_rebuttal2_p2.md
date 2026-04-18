# Literature Second Rebuttal -- Pass 2, Round 3

**Role:** Literature Reviewer (Closing Perspective)
**Round:** P2-12, Round 3
**Date:** 2026-04-17

---

## What Three Rounds Have Established

The gap-dominance finding is durable. All panelists accepted that overnight gap IC substantially exceeds intraday IC in KR (ratio 2.15) and HK (3.14), aligning with Lou, Polk, and Skouras (2019, JFE) and Hendershott, Livdan, and Rosch (2020, JFE): predictability concentrating at the open is the expected signature of an information source that resolves before equity markets open.

The HK null is methodologically clean. When the control gap:intraday ratio (4.80) exceeds the main (3.14), the Lo and MacKinlay (1990, RFS) test fails precisely as the selection-bias literature predicts.

The KR identification is the more consequential result. A pre-specified main-minus-control gap IC of 0.024 (p = 0.008), combined with a control gap:intraday ratio of 0.88, is the structural fingerprint the crypto channel predicts. Benjamini and Hochberg (1995) and Romano and Wolf (2005) are substantially less punishing on this correlated, pre-specified family than Bonferroni. Honest characterization: suggestive but not definitive on one market.

Feature ablation -- dropping crypto overnight features reduces gap IC by 0.021 in HK and 0.033 in KR versus at most 0.005 for macro features -- is consistent with Liu and Tsyvinski (2021, RFS) and is cost-model-independent.

Long-only index futures as the deployment form is a four-panelist consensus, consistent with Novy-Marx and Velikov (2016, RFS) and the EM short-constraint literature (Nagel 2005; Jones and Lamont 2002).

---

## What Remains Unresolved

Three issues survive all three rounds without resolution.

KR signal stability in 2024-2026. The 2023 attenuation is candidate McLean and Pontiff (2016, JF) decay -- crypto ETF proliferation and institutional desk growth reduced the information-speed differential Lou et al. require. The year-by-year index_kr Sharpe breakdown is absent from output files; the decay hypothesis and the transient-regime hypothesis are observationally equivalent without it.

LGBM vs. TCN. Grinsztajn, Oyallon, and Varoquaux (NeurIPS 2022) predict tree-based outperformance on tabular data, explaining part of the gap. But TCN's OVERFIT_FLAG (train IC 0.33-0.38 versus OOS 0.03-0.09) makes it a poorly calibrated benchmark; the 0.028 IC difference may partly measure model specification, not signal content.

Cost model calibration. Realistic KOSPI 200 futures execution is 1.6-4 bps per round-trip versus the model's implied 0.24 bps. Ratcliffe, Miao, and Harvey (2018) and Novy-Marx and Velikov (2016) find liquid-asset-class strategies survive realistic cost scaling, but both papers calibrated from actual execution data. The paper-trade gate the Practitioner proposes is the right instrument; it has not been passed.

---

## Recommended WRITEUP Framing

The debate supports framing the result as: "Evidence consistent with a real cross-asset information-speed channel between crypto and Asian equity gap returns, with long-only index-futures implementation being the only deployment form that survives realistic execution costs. The KR attenuation in 2023 and the novel nature of the channel warrant ongoing monitoring and pre-specified kill switches."

---

## Three Next-Step Research Questions

1. Persistent monitoring of gap IC in KR through 2025-2026 -- is the 2023 attenuation permanent decay or transient? A rolling annual IC series against the crypto ETF adoption timeline would allow a structured McLean-Pontiff test.

2. Cross-market replication: does the same pattern appear in Taiwan, India, or SGX? If the information-speed channel is structural, it should generalize to other crypto-active EM markets with similar overnight gap mechanics.

3. Event study around major crypto events (ETF approvals, major exchange failures, regulatory announcements): does gap IC spike in the 48-hour windows around large directional overnight crypto moves? A Fama-Fisher-Jensen-Roll design on the main-minus-control gap return difference would provide event-based identification independent of the rolling regression.

---

## Literature Verdict After Three Rounds

The overnight-gap-dominance finding is supported by three independent prior mechanisms (Lou-Polk-Skouras 2019; Hendershott-Livdan-Rosch 2020; Berkman-Koch-Tuttle-Zhang 2012). The KR crypto-channel identification has survived a Lo-MacKinlay selection-bias test and is consistent with Liu-Tsyvinski factor incrementality, but rests on a single-market pre-specified test that does not clear the most conservative correction. The long-only index-futures conclusion converges across Novy-Marx-Velikov empirics and EM short-constraint theory. The 2023 attenuation is a standing McLean-Pontiff warning the literature cannot resolve without more OOS years. Scholarly verdict: a credible, novel first demonstration of a crypto-to-equity gap channel -- the most important next step is live paper-trade validation with realistic cost logging and annual IC monitoring through 2026.
