# Skeptic Rebuttal — Pass 2, Round 2

**Role:** The Skeptic
**Round:** P2-12, Round 2

I engage the strongest argument from each panelist and state what my Round 1 position concedes versus what it holds.

---

## Engaging the Believer: "KR p=0.008 is the identification; control doing well IS the point"

The Believer argues the control's elevated gap IC (0.037 KR) is expected -- a reduced-crypto, not zero-crypto, baseline -- and the residual 0.024 differential (p=0.008) is the clean crypto-specific increment. The KR control gap:intraday ratio of 0.88 versus the main ratio of 2.15 (`diagnostics_summary_pass2.txt`) is a genuine structural distinction; I concede it.

What I hold: p=0.008 is a single test selected after observing the HK null. The Pass 2 test family encompasses 100+ comparisons across IC, regime, ablation, and horse race dimensions. Bonferroni-corrected alpha is p < 0.0005; the KR control p-value fails by an order of magnitude with no correction offered. On TCN: ruling out one mechanism does not validate another. If the signal were real at the claimed magnitude, TCN on the same overnight window should not underperform by 0.028 IC points (p=0.000, `horse_race_bootstrap.csv`).

---

## Engaging the Literature Reviewer: "KR clears the Lo-MacKinlay bar; Grossman-Stiglitz explains why it persists"

The Reviewer's strongest argument: when the HK control gap:intraday ratio (4.80) exceeds the main ratio (3.14), the Lo-MacKinlay bar is not cleared, and the HK result is a selection artifact. I accept this -- it sharpens my Round 1 position.

Where I push back: Grossman-Stiglitz. The Reviewer argues net-of-cost unviability is consistent with signal real but arbitrage costly. It is equally consistent with overfit noise. Both produce high gross IC and negative single-stock net Sharpe -- Grossman-Stiglitz cannot distinguish them. The diagnostic that can is OOS stability in a model that does not benefit from feature engineering. TCN OOS gap IC is 0.032-0.034 (`diagnostics_summary_pass2.txt`) versus LGBM 0.061. If the channel were structurally real at that magnitude, TCN on the same overnight window should not underperform by 0.028 IC points (p=0.000). Grossman-Stiglitz does not explain the TCN-LGBM gap; overfitting does. The Lou-Polk-Skouras mechanism is valid for gap concentration generally but does not identify the source as crypto-specific.

---

## Engaging the Practitioner: "Long-only dominates; index at 10x modeled spread still clears"

**On long-only dominance.** The Practitioner reads `long_short_decomposition.csv` accurately: main_kr LGBM gap long-only gross Sharpe 3.38, combined 3.77. But long-only outperformance in high-beta crypto-adjacent EM equities across 2020-2021 and 2024 is beta, not alpha. `regime_analysis_pass2.csv` confirms: index_kr gap IC is weakest in BTC_up_trend (0.237) and VIX_low (0.259) -- exactly when long-only accrues the most return. If the long leg reflected genuine alpha, IC should be highest when beta provides least tailwind. It is lowest.

**On "index at 10x modeled spread still clears."** Realistic KOSPI 200 futures spreads are 4+ bps per round-trip; the model implies 0.24 bps -- a 17x, not 10x, understatement. `cost_sensitivity_pass2.csv` tops out at 2x multiplier (index_kr gap net Sharpe 2.31). At 17x the table has no answer. Extrapolating from a "gentle gradient" ignores that gap signal days are precisely when spreads widen nonlinearly. A 17x cost error should trigger revaluation of every other assumption: market impact, open-auction slippage, margin funding. "Would trade at $50M pending paper trading" is intent, not evidence.

---

## Surviving Skeptic Claim

I revise one position: the HK null has a cleaner location in Lo-MacKinlay selection bias, and the KR gap:intraday contrast (2.15 vs 0.88) is a genuine structural distinction I did not adequately separate in Round 1. HK and KR are not symmetric failures.

Four positions hold: (1) KR control p=0.008 fails Bonferroni over 100+ realized tests and requires independent replication. (2) TCN OVERFIT_FLAG on every fold -- OOS IC 0.032-0.069 versus train IC 0.33-0.38 (`diagnostics_summary_pass2.txt`) -- is unexplained and consistent with feature-engineering overfit, not genuine temporal structure. (3) Long-only alpha is undecomposed from beta; IC is weakest in the regimes where long-only accrues the most return. (4) The index cost model is understated approximately 17x; the sensitivity table tops out at 2x and "survives" is extrapolation, not file evidence.

What can be stated: KR index gap IC of 0.277 (p=0.000) is real, the control gap:intraday distinction in KR is consistent with a crypto channel, and index futures is the correct deployment frame. What remains open: whether the result survives multiple-testing correction, whether LGBM's IC is signal or feature overfit, and whether realistic execution leaves positive net performance.
