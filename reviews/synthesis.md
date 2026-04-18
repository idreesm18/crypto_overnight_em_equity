# Synthesis — Pass 1 Adversarial Review

## Panel composition

Four panelists, two rounds. Round 1 position papers: Skeptic, Believer, Literature Reviewer, Practitioner. Round 2 rebuttals, each engaging the other three.

The debate produced tight agreement on three points, productive disagreement on one, and a clean operational consensus on deployability.

## Acceptance criteria — resolved

The brief specified three criteria, any one of which would constitute a positive finding.

| Criterion | Result | Verdict |
|---|---|---|
| 1. OOS IC > 0.03 (p < 0.05 bootstrap) on close-to-close in at least one market | HK cc IC = 0.014, p = 0.036; KR cc IC = 0.008, p = 0.25 | FAIL on IC magnitude; HK p-value significant but IC below threshold |
| 2. Gap vs intraday IC differential > 0.02 | HK +0.041; KR +0.038; both p < 0.001 | PASS |
| 3. Feature ablation degrades gross Sharpe by > 0.15 | 9 of 18 ablation runs qualify | PASS |

Two of three criteria passed cleanly. Criterion 2 — the pre-specified decomposition result — passed in both markets with a 3:1 IC ratio that the Believer correctly labeled the hardest-to-reverse finding.

## What survives scrutiny

Three findings hold after adversarial review, each in attenuated form.

**Cross-market gap-vs-intraday asymmetry.** HK gap IC 0.0607, KR gap IC 0.0610, both p < 0.001 via 1000-iteration block bootstrap with 20-day blocks. Gap:intraday ratio 3:1 in HK, 2.6:1 in KR. The Skeptic attacked the "identical to four decimals" framing as evidence of shared input dependence rather than independent confirmation. The Literature Reviewer accepted this partial validity while noting that a pure tautology would inflate gap and intraday IC equally, which is not what we observe. The asymmetry itself survives Bonferroni correction at 0.05/12 = 0.0042 in both markets.

**Weekend-window scaling.** Monday IC (65-hour Fri→Mon overnight) exceeds Tue-Fri IC (17-hour overnight) in both markets: HK 0.082 vs 0.056, KR 0.065 vs 0.060. This is the cleanest mechanism test — Nature varies window length for us, and IC scales with it. No panelist disputed the direction; the Skeptic raised the sample-size question (~300 Mondays) which is valid but does not overturn the sign.

**SHAP stability and feature attribution.** 75% top-10 overlap between consecutive folds; Spearman rank correlation 0.70. Top three for gap: eth_ov_log_return (96-97% of folds), btc_ov_log_return (87-97%), vix_level (96-97%). This rules out rotating-noise feature selection and confirms the crypto-overnight features carry the pre-specified signal channel.

## Points of productive disagreement

**Is the crypto channel load-bearing, or is this a stock-level reversal strategy with crypto noise regularization?**

The Skeptic's strongest attack: removing stock-level features (20d vol, 20d momentum, prior-day return) drops gross Sharpe 2.50 HK / 3.03 KR. Removing crypto features drops Sharpe 1.10 HK / 1.78 KR. Stock-level dominates.

The Believer's rebuttal: both drops are large. Short-term reversal being strongest is expected and well-known. The crypto contribution on top (+1.10 to +1.78 Sharpe) is the incremental alpha. Critically, short-term reversal alone does not produce gap > intraday IC differential; crypto-overnight information specifically does.

Unresolved: a counterfactual on a non-BTC-filtered universe would quantify the circularity effect. Neither panel nor pipeline tested this in Pass 1. Pass 2 must add it.

**Regime concentration.**

The Skeptic's rebuttal attacked year-by-year consistency: IC clusters in 2025-2026 coinciding with HK crypto ETF listings and Naver-Dunamu merger.

The Believer examined actual fold-level IC and responded with data: HK gap IC is positive every year 2021-2026 (0.031-0.100). KR has a real 2023 attenuation (mean IC 0.001, 5/12 folds positive) but 0.060-0.091 in 2020-2022. The Believer conceded the KR 2023 point and refined: "mechanism conditional on crypto macroeconomic salience".

The Literature Reviewer concurred with Avramov-Chordia and Muir (2017) on regime-dependent signals: this is regime-conditional, not uniformly structural, and the writeup must frame it as such.

## Strongest unresolved objection

**The universe-selection circularity critique has no counterfactual.** The universe is selected on trailing 60-day BTC correlation, then predicted with BTC/ETH overnight returns as the top-two SHAP features. The Skeptic's claim that the 0.06 IC partially reflects the selection rule is valid but unquantified. No run in Pass 1 tests a non-crypto-filtered universe.

This does not invalidate the gap-vs-intraday asymmetry (which is a within-universe decomposition, immune to the filter) but it does cap the claim about the absolute magnitude of the crypto overnight information channel.

## Operational consensus

All four panelists agree: no tested strategy is deployable at assumed or realistic costs.

The Practitioner's Round 2 analysis closed the HK path: minimum quoted spread on the actual trading names (30-70 bps) exceeds breakeven (9.9 bps) by 3-7×. Borrow is unavailable for the GEM-board tickers in the HK universe. No refinement resolves this.

KR has one survivable path: KOSPI large-caps only, gap-only, with a market-functioning regime gate. This has not been backtested. The Practitioner's threshold for continuing: KR KOSPI-only net Sharpe above 0.5. Below that, it becomes a documented null result.

## Pass 2 priorities

1. **Universe circularity counterfactual.** Run gap-vs-intraday decomposition on a non-crypto-filtered universe (matched size-cap and liquidity). If the asymmetry persists at comparable magnitude, the Skeptic's circularity critique is quantitatively dismissed. If it collapses, the result is primarily a universe-selection artifact.

2. **KR KOSPI large-cap variant.** Restrict KR universe to KOSPI names with ADV > $50M (Naver, Kakao, Samsung, SK Hynix, KB Financial, Shinhan, etc., with crypto adjacency). Tighter spreads, borrow availability, and larger position sizes. The Practitioner's top refinement.

3. **Regime gate design.** Operationalize "crypto macroeconomic salience" into a rules-based, no-look-ahead gate. Candidate variables: 30-day BTC market cap / S&P 500 market cap ratio; crypto ETF AUM level; a 5-day observed spread vs 6-month mean.

4. **TCN addition and index-level prediction** as specified in the Pass 2 brief scope.

5. **Event-identification overlay.** Year-by-year fold IC shows non-trivial concentration in specific periods. Tag individual OOS months with known crypto events (ETF launches, exchange bankruptcies, mainstream integration announcements) and test whether signal strength scales with event intensity or is orthogonal.

## Framing recommendation for README and WRITEUP

Lead with the gap-vs-intraday asymmetry. This is the pre-specified finding, and it is cross-market consistent, Bonferroni-surviving, and mechanism-supporting (weekend-effect scaling reinforces it).

Frame the net-of-cost result honestly: a signal that fails on realistic execution friction is a research finding, not a deployable strategy. The Practitioner's "closed on HK, conditional on KR" verdict should appear in the limitations section.

Document the universe circularity as the largest unresolved question and the primary motivator for a Pass 2 control test.

Avoid "robust", "powerful", "leverages", "crucial". Use "in this sample" for scope-limited claims. No em-dashes.
