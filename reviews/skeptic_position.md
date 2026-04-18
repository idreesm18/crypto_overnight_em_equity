# Skeptic Position Paper: Crypto Overnight EM Equity Signal (Pass 1)

**Role:** The Skeptic  
**Date:** 2026-04-16  

---

## 1. Stock-Level Features Are Doing the Work — Not Crypto

The ablation numbers make this claim falsifiable. From `output/feature_ablation.csv`:

| Market | Drop group | Delta gross Sharpe |
|--------|-----------|-------------------|
| HK | stock_level | −2.500 |
| HK | crypto | −1.101 |
| KR | stock_level | −3.027 |
| KR | crypto | −1.783 |

In both markets, removing the **three stock-level features** (20d realized vol, 20d momentum, prior-day return) is more destructive than removing all crypto overnight features. These stock-level features are short-term reversal and momentum — not a hypothesis about crypto spillover. The model is predominantly a short-term cross-sectional momentum/reversal model on a BTC-correlated equity universe. The crypto features add a real but secondary increment of roughly 1.1–1.8 Sharpe points.

Stated more directly: if this were run on *any* volatile small-cap universe — not one selected for BTC correlation — stock-level features alone would likely produce a comparable base Sharpe. The crypto channel is the garnish, not the meal.

---

## 2. Universe Selection Circularity Is Not Benign

The brief acknowledges the universe is selected on trailing 60-day BTC correlation, then predicted using BTC and ETH overnight returns as the top SHAP features. From `output/shap_stability.csv`, eth_ov_log_return and btc_ov_log_return rank #1 and #2 in both HK and KR gap models (97%+ fold frequency in top-10). The brief calls this "mildly tautological" — but that is a political framing. Consider what this actually does mechanically:

- Stocks are selected precisely because they co-move with BTC over the prior 60 days.
- The same BTC overnight return is then the strongest predictor of the next-day gap return for those stocks.
- When BTC has a large overnight move, the selected stocks (whose recent returns are mechanically correlated with BTC) will also gap, on average, in the same direction.

The IC of ~0.06 partly measures the *selection rule working as designed*, not an independent information channel. No analysis exists comparing gap IC on a non-crypto-filtered universe (e.g., KOSPI 200 by market cap). Without that counterfactual, the 0.06 IC cannot be separated from the circularity. The brief acknowledges the issue and moves on; that is insufficient.

---

## 3. Multiple Testing: The Headline Result Has Not Been Corrected

Count the hypothesis tests embedded in the reported results:

- 6 market×target IC tests (primary)  
- 18 ablation comparisons (3 groups × 2 markets × 3 targets)  
- 18 regime cells (3 splits × 2 buckets × 3 targets, both markets)  
- 2 weekend effect comparisons  
- SHAP stability (not a test, but used as confirmatory)

A conservative lower bound is ~44 tests. Bonferroni at α=0.05 requires p < 0.0011 per test. The headline gap IC p-values are reported as 0.000 (block bootstrap) in `output/return_decomposition.csv`, which survive. But several secondary findings do not. The HK close-to-close IC of 0.014 has p=0.036 (`return_decomposition.csv`); under Bonferroni this fails at the corrected threshold. The regime IC differentials (IC in high-VIX vs low-VIX: 0.068 vs 0.060 in HK, from `output/regime_analysis.csv`) have not been tested for significance at all — they are presented as narrative support without p-values. Under Bonferroni, the regime claims are unsubstantiated.

The weekend effect in KR is 0.065 vs 0.060 (difference 0.005) from `output/weekend_effect.csv`. With ~296 Monday days spread across ~30 stocks ≈ ~8,880 stock-day observations on Mondays — and the IC difference being 0.005 — this is almost certainly not significant. No p-value is reported for the IC difference itself. The HK difference (0.082 vs 0.056) is larger and may be real, but it too lacks a reported test statistic. Using uncorrected narrative framing of untested differences is data snooping.

---

## 4. Systematic Train-OOS IC Gap in KR — Not Random Noise

From `output/training_log_lgbm_kr_gap.csv`, early folds show severe overfitting: fold 0 ic_train=0.158 vs ic_test=0.009; fold 1 ic_train=0.142 vs ic_test=0.094; fold 2 ic_train=0.139 vs ic_test=0.031. This pattern continues throughout: KR train IC averages ~0.09–0.14 in folds 0–22 while test IC averages ~0.05. The differential compresses over time as the training set grows (the growing-window design dilutes early overfitting), but it never disappears.

The HK model is cleaner — only 1 OVERFIT_FLAG in 75 folds — but still exhibits a structural gap: ic_train in later folds (e.g., fold 47: 0.112, fold 51: 0.110, fold 73: 0.096) consistently exceeds ic_test (ranging from −0.016 to 0.182, high variance). The mean OOS IC of 0.061 is real, but it is achieved with train ICs 0.06–0.11 across the same period, meaning the model is still substantially in-sample at every fold.

That said, per the operating standards, the flag threshold is train IC − OOS IC > 0.20. HK gap: 1 flag, KR gap: 0 flags by that metric. This is the strongest defense of the result — and I acknowledge it. My argument is more structural: the train-test IC gap *direction* is systematic throughout, indicating the model consistently extracts more from the training data than it delivers OOS.

---

## 5. Transaction Costs Are Not Survivable in HK — and Barely in KR

From `output/backtest_summary.csv`:

| Market | Gross Sharpe | Net Sharpe | Breakeven multiplier |
|--------|-------------|-----------|---------------------|
| HK | 3.04 | −1.57 | 0.659 |
| KR | 3.77 | −0.28 | 0.933 |

HK requires costs to be 34% of their modeled value to break even. The modeled half-spread is 15 bps/side. Even at 10 bps — aggressive for the small-cap crypto-exposed HK names on the GEM board — the strategy is net-negative. These are not deep-value names with tight spreads; they are thinly traded, volatile small-caps that tend to have elevated spreads precisely during crypto stress events (the regime when the signal is strongest, per `output/regime_analysis.csv`: high-VIX IC=0.068 vs low-VIX 0.060). The assumption that costs are constant and uncorrelated with the signal is itself optimistic.

KR at 0.93x is closest to viable — but that means any realism adjustment kills it. Borrow cost for short positions (standard in KOSDAQ: 50–150 bps annualized for small-caps) is not in the model. Short availability constraints — particularly during high-crypto-vol events when the short leg would need to be reset — are not modeled. The KR gross Sharpe of 3.77 on 8–10 name terciles over 6 years is a small-portfolio artifact, not a deployable strategy. Strategy capacity at $100K per position across 20 names is roughly $2M gross. Any institutional desk needs to scale this 50–100x, at which point market impact swamps the signal.

---

## Strongest Argument Against

**The stock-level feature dominance is not rebuttable.** The ablation output is unambiguous: removing 20d realized vol, 20d momentum, and prior-day return destroys more Sharpe than removing all crypto features. The stated hypothesis — overnight crypto information creates a predictable component in gap returns of crypto-exposed equities — is not the primary mechanism detected by the model. The model has found a short-term cross-sectional reversal/momentum pattern on a volatile equity universe, augmented by a real but secondary crypto channel. The crypto channel adds 1.1–1.8 Sharpe points gross, which sounds meaningful, but nets to a loss in both markets. The headline result is not a false positive — the OOS IC of 0.06 is likely real — but it is misattributed. This is a momentum/reversal model, not a crypto spillover model.
