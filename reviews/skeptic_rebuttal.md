# Skeptic Rebuttal — Round 2
## Crypto Overnight EM Equity Signal (Pass 1)

**Role:** The Skeptic  
**Date:** 2026-04-16

---

## Response to Believer

The Believer's strongest claim is that cross-market IC convergence to 0.0607 HK / 0.0610 KR is "the gold standard anti-overfit test" — that two independent markets producing the same number proves a reproducible mechanism.

This argument fails on a shared-input problem. HK and KR are not information-independent samples. Both models take BTCUSDT overnight returns as their dominant input (eth_ov_log_return and btc_ov_log_return rank top-2 in 87–97% of folds in both markets per `shap_stability.csv`). Both universes are selected using the same BTC correlation filter. The "two markets" are essentially two portfolios of stocks filtered by the same signal (BTC co-movement) and predicted by the same features (BTC/ETH overnight return). IC convergence to the fourth decimal is precisely what you would expect when both outputs share the same dominant input channel — it confirms feature engineering consistency, not mechanistic independence.

More damaging is the fold-level IC evidence from `training_log_lgbm_hk_gap.csv`. The Believer presents the mean OOS IC of 0.061 as a stable, pre-specified finding confirmed across 75 folds. The fold data tells a different story:

- **Early 2020 folds (0–10): ic_train is blank (NaN)** — the IC computation was suppressed, indicating insufficient cross-sectional variance. These folds contribute zero to the mean while consuming a full month of OOS test data.
- **2021 negative/near-zero folds:** fold 18 (Aug 2021) = −0.079; fold 19 (Sep 2021) = 0.004; fold 22 (Dec 2021) = −0.009; fold 14 (Apr 2021) = 0.002.
- **2022–2024 volatile folds:** fold 34 (Dec 2022) = −0.037; fold 38 (Apr 2023) = −0.072; fold 44 (Oct 2023) = −0.003; fold 47 (Jan 2024) = −0.016; fold 54 (Aug 2024) = −0.007; fold 55 (Sep 2024) = 0.006.
- **2025–2026 folds dominate the positive mean:** fold 51 = 0.181, fold 57 = 0.151, fold 62 = 0.132, fold 63 = 0.168, fold 64 = 0.134, fold 65 = 0.126, fold 73 = 0.125, fold 74 = 0.117.

The mean IC of 0.061 is pulled substantially by a cluster of high-IC folds concentrated in 2025–2026 — the same period the Practitioner identifies as driven by HK crypto ETF listings, Naver-Dunamu merger activity, and won-stablecoin speculation. At least 11 of 75 folds produce negative or near-zero OOS IC (< 0.01). The distribution is not "smoothly positive across 75 folds." It is right-skewed by a handful of recent folds that the Believer frames as "structural" but the Practitioner correctly identifies as event-driven and non-recurring.

The Believer's "pre-registered directional test" also overstates what pre-registration provides here. The gap > intraday hypothesis was stated in advance, but the *magnitude* of the ratio was not. A 3:1 ratio sounds impressive; a result driven by the shared-input mechanism described above would also produce gap > intraday (gap returns are mechanically more responsive to the overnight crypto input than intraday returns, purely by construction).

---

## Response to Literature Reviewer

The Literature Reviewer anchors Pass 1 in Lou, Polk, and Skouras (2019) as a "direct anchor" predicting gap > intraday IC. This framing is overreaching in two specific ways.

**First, LPS operates on a broad, diversified cross-section — not a theme-filtered universe.** LPS's decomposition holds for the *aggregate* equity market, where overnight loading of anomaly returns reflects how institutional MOO orders price in publicly available signals across thousands of stocks. The LPS mechanism assumes that the overnight component captures broad beta and cross-sectional mispricing corrections. Pass 1's universe is not the aggregate market — it is a 20–30 name portfolio pre-selected for BTC correlation. For such a concentrated, thematically filtered universe, the gap/intraday decomposition is expected by construction: stocks in the universe were chosen because they co-move with BTC overnight; of course the overnight gap is the primary window where BTC information gets priced in. LPS cannot meaningfully be called a theoretical anchor for what is, mechanically, a tautology.

**Second, LPS actually cautions that overnight-loading anomalies weaken in less developed or higher-cost markets.** HK GEM board names and KOSDAQ micro-caps are precisely the thin, high-cost environments where LPS's mechanism is least expected to operate cleanly: institutional MOO order flow is sparse, price formation at the open auction is dominated by retail and semi-informed participants, and the information aggregation function is noisier. The Literature Reviewer acknowledges (Section 5) that "0.06 IC is plausibly inflated by the filter" but dismisses this with three qualitative observations. None of those observations (SHAP stability, gap > intraday concentration, regime positivity) controls for the shared-input problem. LPS is a population-level finding; applying it to a 20-name filtered portfolio as a theoretical "anchor" is a category error.

---

## Response to Practitioner

The Practitioner concludes "signal is real; strategy is not deployable." I agree on the second point but sharpen the implication of the first.

The Practitioner's own year-by-year evidence is the most important finding in the entire review panel, and it directly contradicts the Believer's "robust across regimes" narrative. From the Practitioner's Section 5d: 2021 HK net −57%, 2022 HK net −34%, 2022 KR net −10%, with the full-period Sharpe "carried by 2025–2026 crypto events" (HK crypto ETF, Naver-Dunamu, won-stablecoin speculation). The Believer's regime analysis shows IC positive "in every regime cell" — but those regime cells are defined by VIX levels and BTC direction, not by whether a structural market event occurred. The regime splits do not isolate 2025–2026 event-driven folds from structural signal folds. The positive mean is not regime-robust; it is event-driven on specific non-recurring catalysts.

Now engage the ablation argument directly. The Believer claims: crypto features drop gross Sharpe by 1.1 (HK) and 1.8 (KR), therefore the crypto channel is "load-bearing." But the ablation file (`feature_ablation.csv`) contains the net-Sharpe figures the Believer did not present:

| Market | Condition | Gross Sharpe | Net Sharpe |
|--------|-----------|-------------|------------|
| HK gap | Full model | 3.038 | −1.563 |
| HK gap | Crypto dropped | 1.937 | −9.363 |
| KR gap | Full model | 3.768 | −0.284 |
| KR gap | Crypto dropped | 1.985 | −5.586 |

Both the full model and the crypto-stripped model are net-negative. The crypto channel adds 1.1–1.8 gross Sharpe points but saves approximately 7.8 (HK) and 5.3 (KR) net Sharpe points relative to the stripped model — almost entirely through turnover reduction, not better signal. When crypto features are stripped, the model increases trading activity to compensate, amplifying the cost drag. The net Sharpe improvement from crypto is a model mechanics artifact. In both cases the strategy nets to a loss. The Believer's "load-bearing" framing is correct in gross terms and irrelevant in net terms.

---

## Position Sharpened

The single sharpest attack, which survives all rebuttals: **the mean OOS IC of 0.061 is dominated by a concentrated cluster of 2025–2026 folds that the Practitioner identifies as event-driven and non-recurring, not by a smoothly positive signal across the full period.**

Of 75 HK gap folds, at least 11 produce negative or near-zero OOS IC (< 0.01). The high-IC folds (0.12–0.18) are concentrated in 2025–2026 — precisely the period of HK crypto ETF listings, Naver-Dunamu merger speculation, and won-stablecoin events. The full-period mean obscures a distribution where the median period (2020–2024) is substantially closer to zero. Neither the Believer's cross-market convergence argument, the Literature Reviewer's LPS anchor, nor the Practitioner's "signal is real" concession engages with the specific fold-level distribution showing that the headline number is driven by recent, non-structural events.

A genuine structural signal would exhibit positive IC in 2020–2024 folds at comparable magnitude to 2025–2026. It does not. The burden of proof — that the mechanism is structural rather than event-driven — has not been met.
