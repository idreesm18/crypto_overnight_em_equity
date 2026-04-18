# Believer Second Rebuttal -- Pass 2, Round 3

**Role:** Believer
**Round:** P2-12, Round 3

---

## Concession 1: Realistic Net Sharpe Is 1.5-2.0, Not 3.74

The Practitioner established this in Round 1. The Skeptic correctly presses it in Round 3. The model-implied 0.24 bps per round-trip for index_kr gap gate_off is wrong by 7-14x against realistic KOSPI 200 futures execution (1.6-4 bps half-spread per the Practitioner's direct market knowledge, `practitioner_p2.md`). At 7x (low end), net Sharpe falls to approximately 1.5-2.0 by linear extrapolation from `output/cost_sensitivity_pass2.csv`. The Round 1 Believer position that the "strategy nets Sharpe 3.74 after costs" is not the right number to carry forward.

The corrected claim: the index_kr gap signal produces a gross Sharpe of approximately 4.1 and, under realistic futures execution costs, a net Sharpe of 1.5-2.0. That is tradeable. It is not spectacular. The Believer concedes the Round 1 framing overstated the realized edge.

---

## Concession 2: McLean-Pontiff Decay Is a Plausible Mechanism for 2023 Attenuation

The Literature Reviewer raised McLean-Pontiff (2016) as the correct framework for the 2023 KR fold-level IC attenuation (`literature_rebuttal_p2.md`). The mechanism is concrete: crypto ETF proliferation and institutional crypto desk growth through 2022-2024 lowered the information-speed differential that the overnight channel exploits. This is not a refutation -- the signal may still be live -- but it is a plausible causal path from structural persistence to gradual erosion.

The correct response is monitoring, not exit. The Practitioner's kill switch (rolling 6-month Sharpe below 0 in two consecutive 3-month windows, `practitioner_rebuttal_p2.md`) is the right operational instrument. The decay mechanism, if active, will trigger the kill switch before a material drawdown accumulates. Exit on evidence of decay, not on the theoretical risk of it. McLean-Pontiff decay is a live hypothesis; it is not a confirmed verdict on the basis of one calendar year's attenuation.

---

## Concession 3: Seven Years Is One Partial Crypto Cycle

The Skeptic is correct that the sample covers one bull-bear-recovery sequence (2020-2021 bull, 2022 bear, 2023-2026 partial recovery and ETF adoption, `skeptic_rebuttal2_p2.md`). The finding should be characterized as: consistent with a real signal over 7 OOS years, with the caveat that a full bear-to-bear cycle has not been observed and the year-by-year index_kr Sharpe breakdown is absent from the output files. The Practitioner explicitly flags this as a required gap before any investor pitch (`practitioner_p2.md`). The Believer agrees. The full-period Sharpe is informative; the temporal distribution of that Sharpe is not yet established from the output files.

---

## Hold: The KR Gap:Intraday Ratio Asymmetry Is Structural Evidence

The Skeptic conceded this in Round 3: "the ratio contrast across HK and KR goes in the predicted direction" (`skeptic_rebuttal2_p2.md`). KR main gap:intraday ratio is 2.15; KR control ratio is 0.88 (`diagnostics_summary_pass2.txt`, `control_vs_main_comparison.csv`). The control non-crypto-filtered universe shows no gap dominance in KR; the crypto-filtered main universe shows strong gap dominance. This contrast cannot be explained by selection bias (which would predict the opposite, as in HK), by a beta confound (which operates at the return level, not the IC ratio), by Bonferroni correction (which applies to p-values, not ratios), or by cost model errors (which affect net Sharpe, not IC structure). Three full rounds of adversarial debate have not produced a mechanism that explains the KR control ratio contrast other than the crypto channel.

---

## Hold: Feature Ablation Evidence Is Cost-Model-Independent

Dropping all crypto overnight features reduces net Sharpe by 1.16 units in HK and 0.69 units in KR (`output/feature_ablation_pass2.csv`). The Skeptic's LGBM overfit concern applies to absolute IC levels; it does not explain why a specific category of features -- and the same category across both markets -- produces the largest ablation penalty. Feature ablation operates on relative importance within the LGBM model. If LGBM were simply overfitting hand-engineered features as the Skeptic suggests, removing any category should produce proportional degradation. Instead, dropping crypto overnight features produces a disproportionate penalty relative to their share of the feature set. This is an internally consistent signal that the category carries load, independent of whether the absolute IC level is inflated by feature selection.

---

## Hold: Two Independent Index Futures Implementations Both Clear

HSI and KOSPI 200 are separate strategies with separate signal paths, separate regime splits, and separate control universe tests. Both clear the IC acceptance criterion with p=0.000 block bootstrap (`output/backtest_summary_pass2.csv`). The "2-ticker accident" argument would require both to be independent accidents. The Skeptic did not advance that framing in Round 3, and the Practitioner independently allocated desk capital to both with separate size limits (`practitioner_rebuttal_p2.md`). Two implementations clearing independently is corroborating evidence, not a statistical test -- but it is not noise.

---

## Hold: Long-Only Index Futures Is the Agreed Deployment Form

The Literature Reviewer confirmed EM short structural impairment via Nagel (2005) and Jones and Lamont (2002). The Practitioner confirmed it from first-principles market knowledge. The Believer confirmed it from the decomposition in `long_short_decomposition.csv`. All four panelists now agree that index-futures-long -- or long-only equity with index future as the short hedge -- is the correct implementation. The single-stock long-short backtest is not the finding; it is the background that explains why the finding lives at the index level.

---

## What the Final Believer Position Is

The finding, stated with appropriate precision: the crypto overnight channel predicts next-day gap returns in KR index futures with OOS IC of 0.277 (p=0.000, block bootstrap), supported by a pre-specified control universe test showing crypto-specific incremental IC (gap 0.024, p=0.008 in KR, consistent with the predicted null in HK), a structural gap:intraday ratio contrast that survives all methodological challenges, and a feature ablation profile that isolates the crypto overnight category as load-bearing across both markets. After realistic futures execution costs (7-14x the model assumption), net Sharpe is approximately 1.5-2.0, which remains above the minimum deployment threshold. The McLean-Pontiff decay mechanism is a live risk requiring active monitoring; the 2023 attenuation is candidate evidence, not confirmation. Seven OOS years is one partial cycle; the year-by-year Sharpe breakdown is a required disclosure before any capital allocation. The strategy is a legitimate research finding that warrants live paper trading validation -- not an immediate full-size allocation and not a null result.
