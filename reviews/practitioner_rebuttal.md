# Practitioner Rebuttal — Round 2
## Crypto Overnight EM Equity Signal (Pass 1)

**Role:** The Practitioner  
**Date:** 2026-04-16

---

## Response to Skeptic-Believer Exchange on Regime Concentration

The Skeptic's Round 2 sharpened claim — that the mean OOS IC of 0.061 is "dominated by a concentrated cluster of 2025–2026 folds" driven by non-recurring events — is directionally correct for KR but overstated for HK.

The Believer's fold-level table changes what a trader should conclude:

- **HK:** IC positive in 10–11 of 12 folds every year from 2021–2025, with mean IC 0.031–0.053 across 2021–2023 before the 2025–2026 uptick. This is not regime-concentrated. A year-over-year Sharpe-grade consistency test on HK — is mean annual IC > 0 in 5 of 6 years? Yes — passes.
- **KR:** 2023 is a genuine hole: mean IC 0.001, 5/12 folds positive. The KR result is regime-dependent in a way HK's is not.

The operational implication is not symmetric. For capital allocation purposes, HK's consistency profile is closer to Sharpe-grade than KR's. But HK also has the structurally insoluble cost problem (breakeven at 9.9 bps vs. realistic 30–70 bps minimum quoted spread on the names that drive the signal). Year-over-year IC consistency is irrelevant when the strategy nets below −5 Sharpe at realistic spreads.

KR is the only market where a path to deployment exists, and KR is also the market where regime dependency is most pronounced. That is the core tension. The Believer frames 2023 KR attenuation as an interpretable regime gap consistent with the mechanism; the Skeptic frames it as evidence the signal is not structural. From a trading desk's perspective, the question is simpler: if you ran this strategy live in 2023 you would have lost money in KR, and there was no ex-ante signal that 2023 would be different from 2022. A risk manager requiring forward-looking regime identification — not backward-looking narrative — would not have gotten it.

The Believer's "macro salience" framing is intellectually honest but operationally insufficient. A regime gate that is only visible in hindsight is not a regime gate.

---

## Response to Literature Reviewer

The Literature Reviewer's contribution to this panel is genuine and the WRITEUP framing it provides (Option B: extension of LPS overnight mechanism, conditional on crypto macro salience, direction confirmed, magnitude upward-biased) is correct and appropriately hedged.

As a practitioner, I do not place weight on the literature pedigree when deciding whether to allocate capital. "Extension of prior literature" is important for academic publication; it is a secondary consideration for a desk that cares only about deployability. So my response is brief:

The Literature Reviewer correctly identifies that circularity inflates the IC magnitude without making the directional result (gap > intraday) disappear. The directional result is the empirical contribution. The IC magnitude — 0.06 — should not be used as an input to capital allocation sizing models because it is upward-biased by universe construction. This is an important operational point: if a quant analyst hands a PM a Sharpe derived from a 0.06 IC and the true IC is 0.04 (post-circularity correction), the PM will over-size the position and discover the shortfall in live trading.

The Literature Reviewer's characterization of KR 2023 attenuation (short-sale ban coinciding with signal failure) is the single most actionable framing in the panel: the signal requires a minimum market-functioning threshold. That is a concrete, forward-facing condition a trading desk can monitor. The Skeptic's fold-level attack and the Literature Reviewer's market-structure framing combine to the same operational conclusion: strategy requires a regime gate, and market structure integrity (short-sales permitted, normal spread environment) is a more reliable gate than "crypto macro salience."

---

## Refinement Opportunities for Pass 2

Given everything in this panel, two refinements have the highest expected impact on deployability:

**Refinement 1: KR KOSPI large-cap only (Naver, Kakao, Samsung, SK Hynix, KB Financial, Shinhan)**

The KOSDAQ names that drive the crypto signal in the current backtest (Bitplanet, WeMade, Vidente) have realistic half-spreads of 15–30 bps versus the modeled 10 bps, and face sporadic short-sale bans that shut the strategy down at exactly the wrong moments. Restricting to KOSPI large-caps with $50M+ ADV allows modeling realistic 3–8 bps spreads, eliminates borrow friction, and enables position sizes of $500K–$2M per name without meaningful impact. The expected gross Sharpe on KOSPI large-caps will fall — these names have lower crypto beta and faster information incorporation — but the net Sharpe may be positive for the first time. If KR gap IC on KOSPI-only names falls below 0.03, the signal is too attenuated to support a strategy; above 0.03 with a net Sharpe of 0.5+, there is a case for a pilot allocation.

**Refinement 2: Explicit market-functioning regime gate (short-sale permitted + spread environment normal)**

The 2023 KR attenuation is not purely a crypto-salience story; it coincides directly with the November 2023–March 2024 KOSDAQ short-sale ban. A rules-based gate: require (a) short-selling on the target names is currently permitted under FSC rules, and (b) observed 5-day average spread on the universe is below 1.5× its trailing 6-month mean. Both conditions can be monitored daily from exchange data without look-ahead. This gate would have correctly turned off the strategy during the 2023–2024 ban period in KR and during elevated-spread periods in HK. It addresses the Skeptic's "no ex-ante regime identification" objection with a concrete, rule-based, non-look-ahead criterion — not a "trade when it works" heuristic. Expected effect: reduces trading days by 15–25%, eliminates the worst drawdown periods, improves net Sharpe by reducing the tail of extreme cost drag.

**Refinement 3 (secondary): Reduce to gap-only, drop cc and intraday targets from live strategy spec**

The intraday IC (0.019–0.023) is structurally weaker and adds turnover without proportionate signal contribution. Running a gap-only strategy halves the return target complexity and focuses cost budget on the single window where the mechanism demonstrably operates. In live execution, this means orders placed at the pre-open auction only, with positions closed at intraday VWAP or next-day open rather than actively managed intraday. Estimated impact: reduces daily turnover by 30–40%, modestly improves net Sharpe by reducing cost drag, simplifies infrastructure requirements.

---

## Operational Verdict

The panel has converged on a result the Believer correctly frames: the information channel is real, the current strategy is not deployable. My Round 1 verdict stands and is strengthened by the fold-level data.

**HK is closed.** No refinement resolves the structural cost problem. The minimum quoted spread on the actual trading names (30–70 bps) exceeds the breakeven spread (9.9 bps) by 3–7×. Borrow is unavailable or prohibitively expensive. Year-over-year IC consistency is real but irrelevant — consistency of a signal that cannot be captured at any realistic cost is a research finding, not a trading opportunity. HK goes in the WRITEUP as evidence for the mechanism, not as a candidate for a live strategy.

**KR has exactly one path.** KOSPI large-cap names only, market-functioning regime gate active, gap-only signal, positions sized at 5% ADV maximum across 6–8 names. This sub-strategy has not been backtested; everything above is hypothesized direction, not measured outcome. Before allocating any research budget to live infrastructure, Pass 2 must produce a backtest on this specific configuration. If KR gap IC on KOSPI-only names with realistic spreads produces a net Sharpe above 0.5, the case for a 6-month paper-trade pilot is legitimate. Below 0.5 or negative: null result, documented and published, no further capital allocated.

The PM framing: the research budget question is not "does the signal exist?" — it does. It is "can we capture it net of costs at any realistic scale?" The current answer for HK is definitively no. For KR it is "unknown, pending Pass 2 on the KOSPI-only sub-universe." Allocating budget to improve the cost side of the current parameterization (better execution on KOSDAQ names, cheaper borrow) is lower expected value than running the KOSPI-only Pass 2 and getting a clean answer. Move to Pass 2.
