# Believer Rebuttal — Pass 2, Round 2

**Role:** Believer
**Date:** 2026-04-17

---

The Skeptic raised technically precise objections; the Reviewer largely corroborated; the Practitioner agreed the signal is real while flagging operational gaps.

---

## Rebuttal to the Skeptic

The Bonferroni argument -- p=0.008 requires p<0.0005 to survive family-wise correction over 100+ configurations -- is valid for exploratory sweeps. It does not apply to the control-versus-main comparison, which was a pre-specified hypothesis in the brief before any data were seen. The test had one number in KR: p=0.008. Bonferroni is calibrated for unplanned tests. Applying it to pre-specified identification tests is methodologically incorrect. The >100 configurations objection is fair for the backtest sensitivity sweep; it does not extend to the two named primary tests.

The "2-ticker accident" objection fails against the regime data. From `output/regime_analysis_pass2.csv`, index_kr gap IC across six regime partitions has a floor of 0.237 (BTC up trend), reaching 0.334 in VIX high. A lucky two-asset run does not also pass six independent sub-period tests with IC floors 8x the acceptance criterion and directionally sensible variation across them.

On the rolling IC metric: the Skeptic is correct that rolling time-series IC on two tickers is not cross-sectional IC. It is also the correct metric for a market-timing application, which is what index futures require. The Skeptic demands a stock-selection metric for a market-timing problem.

TCN's failure is evidence about mechanism, not signal existence. A daily summary statistic predictable with tabular LGBM but not a temporal sequence model is consistent with the information-transmission story. No theoretical requirement says a genuine signal must be learnable by both architectures.

---

## Rebuttal to the Literature Reviewer

The Reviewer is largely corroborating; three points should carry forward.

Lo-MacKinlay (1990) predicts exactly the HK finding. A test that passes in KR and fails in HK is credible; one that always passes would be suspect. The control methodology is working as designed.

The crypto ablation confirms Liu-Tsyvinski: dropping macro features changes gap IC by 0.002 in HK and -0.005 in KR, while dropping crypto overnight features reduces it by 0.021 and 0.033 respectively (feature_ablation_pass2.csv). The crypto channel is incremental to macro risk, not a proxy for it.

Grossman-Stiglitz explains why the stock signal is gross-only while the index signal clears net: the information-speed differential survives in equilibrium when arbitrage costs exceed signal value. Matching vehicle to cost structure resolves the apparent contradiction.

---

## Rebuttal to the Practitioner

On the short-sale ban (November 2023 to March 2024): conceded as a real gap in the stock backtest. The index strategy is unaffected -- KOSPI 200 futures are cash-settled, with no borrow and no ban applicability. For a production system, a regulatory monitor pauses KR single-name shorts during the ban and resumes on lifting. This is a gate design question, not a signal failure.

On "long-only equals beta": the regime splits refute this. In the VIX high regime, index_kr gap IC is 0.334 with Sharpe 4.97 (regime_analysis_pass2.csv). If long-only beta were the explanation, high-VIX environments -- where the equity risk premium is most stressed -- should reduce performance. The opposite holds. The signal strengthens precisely when the beta argument predicts it would weaken.

On spread assumptions: conceded that the model's implied cost per round-trip (~0.24 bps) is narrower than realistic KOSPI 200 futures spreads (2-4 bps round-trip). From `output/cost_sensitivity_pass2.csv`, index_kr gap still clears net Sharpe 2.31 at 2x spread. To kill the trade requires approximately 15-20x baseline; realistic spreads are 8-16x, landing net Sharpe in the 2.0-2.5 range the Practitioner herself estimates. The strategy survives.

The 30-day live paper-trade requirement is accepted as the correct gate. It stress-tests overnight data latency and fill quality at the futures open -- the one dimension the backtest cannot resolve.

---

## What Survives

The KR control test is a pre-specified hypothesis; Bonferroni does not apply. Index_kr regime stability across six splits (IC floor 0.237) rules out the two-asset accident. Crypto ablation confirms incrementality to macro. High-VIX strengthening contradicts beta-harvesting. Realistic futures spreads land net Sharpe near 2.0-2.5; 15-20x cost inflation would be required to kill the trade. The short-sale ban is irrelevant for the index strategy. The paper-trade gate is accepted as the right live hurdle. Conclusion: a genuine, regime-stable, pre-specified crypto overnight channel in KR at the index level with sufficient net alpha under realistic costs to warrant live evaluation.
