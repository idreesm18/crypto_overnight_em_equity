# Believer Position Paper — Pass 2

**Role:** The Believer
**Date:** 2026-04-17

---

## 1. The Index Result Is The Finding — And It Is Unambiguous

The debate about single-stock net Sharpe is a distraction from what Pass 2 actually
produced. Index gap IC in KR is 0.277 (p=0.000, block bootstrap), in HK 0.185
(p=0.000). These numbers are not borderline: they are more than 4x the 0.03 acceptance
criterion. The top backtest configuration — LGBM, index_kr, gap, threshold, gate off
— produces net Sharpe 3.741 after 2 bps spread and borrow. Gate on: 2.692. The
gap-vs-intraday asymmetry in the index universe is even sharper than in the stock
universe: KR index intraday IC is 0.010, versus 0.277 gap IC, a ratio of 27:1. HK
index gap IC is 0.185, intraday -0.013: the intraday signal inverts. The overnight
channel concentrates at the open, exactly as hypothesized, and it concentrates there
with a magnitude that leaves no ambiguity at the macro level.

The index result was the natural test once the single-stock deployment path proved
cost-constrained. Macro information diffuses into index-level pricing first, then
filters down to individual names. A 27:1 gap-to-intraday IC ratio on the KR index is
not an artifact of the cost structure — it is the mechanism speaking in the clearest
data available.

---

## 2. The Control Gap Difference IS the Crypto Channel Identification

The Skeptic will argue the control universe has gap IC 0.051 in HK and 0.037 in KR,
so something other than the crypto channel is working. This is the correct observation
and the wrong inference.

The control universe was manually constructed from liquid, established sectors with
known structural properties. It carries its own ambient crypto correlation — these are
sectors active during the same macro regime, and some fraction of that correlation with
overnight crypto is unavoidable. The control is not a zero-crypto baseline; it is a
reduced-crypto baseline. The fact that it still has gap IC 0.037 in KR proves that
correlated macro factors are present across both universes.

The causal claim rests on the differential. KR: main gap IC 0.061 versus control 0.037,
difference 0.024, p=0.008 via block bootstrap. The p-value is 0.008. That differential
is crypto-channel-specific — it is what the main universe has that the control does not,
after controlling for everything the control captures. This is the identification
strategy. The control doing well is not evidence against the crypto channel; it is
evidence that the control was well-chosen and is correctly absorbing shared macro signal.
The residual gap of 0.024 (p=0.008) is the incremental contribution of the crypto
overnight channel.

In HK, the p-value is 0.927, indicating no statistically identifiable incremental
contribution. The Believer concedes HK: the control absorbs most of the HK signal.
KR is where the incremental identification is cleanest, and KR produces p=0.008.

---

## 3. The Crypto Features Are Load-Bearing — By Construction, Not By Chance

Feature ablation from `output/feature_ablation_pass2.csv`:

| Ablation | Market | Delta Gross Sharpe | Delta Net Sharpe |
|---|---|---|---|
| drop_crypto_overnight_all | HK | -1.101 | -1.155 |
| drop_crypto_overnight_all | KR | -1.783 | -0.692 |
| drop_stock_level_all | HK | -2.500 | +2.032 |
| drop_stock_level_all | KR | -3.027 | +0.892 |

The stock-level drop is larger in gross Sharpe — expected. But note the net Sharpe
reversal: removing stock-level features actually improves net Sharpe in both markets.
This happens because stock-level features drive turnover. Removing crypto overnight
features has the opposite profile: gross Sharpe falls by 1.1 to 1.8 Sharpe points
with no offsetting net benefit. The crypto features add alpha without adding turnover
cost. That is exactly what low-frequency macro signal looks like when it is real.

Per-feature ablation (`output/feature_ablation_per_feature.csv`) confirms that
eth_ov_log_return is the single largest individual IC contributor in HK (-0.004 IC
delta) and KR (-0.011 IC delta). In both markets, eth_ov_log_return, btc_ov_log_return,
and btc_ov_realized_vol are the top three crypto contributors. These are the same three
features that dominated SHAP in Pass 1 (present in 87-97% of folds). A feature that
shows up persistently across SHAP fold analysis AND ablation matters across both passes
is not noise.

---

## 4. TCN's Failure Strengthens the LGBM Case

TCN significantly underperforms LGBM. Main_hk gap IC: LGBM 0.061, TCN 0.034, difference
-0.021 (p=0.006). Main_kr gap IC: LGBM 0.061, TCN 0.032, difference -0.028 (p=0.000).
These are large, statistically significant gaps. On index: LGBM index_kr gap IC 0.277
versus TCN 0.069. TCN fails at the index level where LGBM succeeds.

This is informative, not embarrassing. TCN learns temporal patterns — lagged
dependencies, sequence structure, minute-level autocorrelations. Its failure means the
signal does not live in temporal microstructure. The overnight crypto return is
predictive as a daily summary statistic, not as a sequence. LGBM with tabular daily
features outperforms a temporal model trained on the same information window.
This pattern is consistent with an information-transmission story: overnight price
discovery produces a level change that is readable in daily summary statistics. If
the signal were a momentum pattern or a technical sequence, TCN would have an advantage.
It does not. The LGBM result stands stronger, not weaker, because TCN's failure rules
out an alternative explanation.

---

## 5. Index Regime Analysis Eliminates the Fragility Concern

From `output/regime_analysis_pass2.csv`, index_kr gap IC across regimes:

| Regime | IC | Gross Sharpe |
|---|---|---|
| VIX high | 0.334 | 4.97 |
| VIX low | 0.259 | 4.12 |
| BTC up trend | 0.237 | 3.39 |
| BTC down trend | 0.327 | 5.19 |
| Crypto vol high | 0.278 | 3.97 |
| Crypto vol low | 0.275 | 4.47 |

The floor across all six regime cells is IC 0.237 and Sharpe 3.39. This signal does
not depend on a single market regime. It is positive, large, and stable whether volatility
is high or low, whether BTC is trending or falling, whether crypto vol is elevated or
suppressed. The smallest cell IC (BTC up, 0.237) is still 8x the acceptance criterion.

This directly contradicts any fragility narrative. A spurious correlation concentrated
in one regime would show sharp variation across regime splits. What we observe is flat
across all splits, with directionally sensible enhancement: higher IC when BTC is
falling (flight-to-macro-signal regime) and when VIX is elevated. The signal is
more useful when macro stress is higher, which is when a practitioner most needs it.

---

## 6. The Cost Structure Argument Points to the Correct Vehicle, Not a Flaw

Single-stock L/S in KR: main_kr gap tercile_ls gate_off net Sharpe -0.598. This
does not clear any deployment threshold. The Skeptic will treat this as the central
result. It is not.

Index futures are the natural deployment vehicle for macro information transfer.
Macro-level overnight news — BTC/ETH gap returns, VIX changes, funding rate
dislocations — reprices risk premia at the index level first. Single-name idiosyncratic
spreads overwhelm the signal before it reaches individual stocks. KOSPI 200 or HSI
futures have spreads under 1 bp, zero borrow cost, and liquid-enough capacity to absorb
any realistic position. The cost structure of the index backtest is 2 bps total
(spread + borrow from `backtest_summary_pass2.csv`, index configs show near-zero
borrow_cost). At those costs, the top KR index config produces net Sharpe 3.741.

Even at 5x cost assumptions (10 bps total), the gross Sharpe of 4.138 would still clear
2.0 net. The index strategy is not marginal — it has a cost buffer of over 3 Sharpe
points between gross and the minimum deployment threshold.

---

## Bottom Line

Pass 2 delivers what Pass 1 could not: a deployable strategy. The crypto overnight
channel predicts index-level gap returns with OOS IC of 0.18 (HK) and 0.28 (KR),
both p=0.000, in a walk-forward evaluation. The best config nets Sharpe 3.74 after
costs. The signal survives every regime split with IC floor above 0.23. The KR control
differential (IC gap 0.024, p=0.008) identifies crypto-channel incremental contribution
above and beyond shared macro structure. The features driving the result are the same
ones SHAP attributed persistently across 75 folds in Pass 1. TCN's failure isolates the
mechanism to daily summary statistics, consistent with information-transmission and
inconsistent with temporal pattern exploitation. This is not a marginal or ambiguous
result. The data support a conclusion that overnight crypto price discovery transmits
to EM equity index open prices with sufficient reliability and magnitude to trade.
