# Literature Review (Pass 2): Crypto Overnight Signal for Asian Equity Markets

**Role:** Literature Reviewer
**Date:** 2026-04-17

---

## 1. Overnight vs. Intraday IC Ratio in Context

The most direct literary precedent for the gap-to-intraday IC asymmetry is Lou, Polk, and Skouras (2019, "A Tug of War: Overnight Versus Intraday Expected Returns," *Journal of Financial Economics*). Their central finding is that expected-return anomalies in US equities are disproportionately priced at the overnight gap rather than during the trading session. The interpretation is institutional: informed traders and market-on-open order submitters incorporate information before the market opens, while intraday price formation is dominated by noise and liquidity-driven reversals.

Pass 2 reproduces this pattern in a cross-asset setting. The main-universe gap-to-intraday IC ratios are 3.1 in HK and 2.1 in KR (diagnostics_summary_pass2.txt, control_vs_main_comparison.csv). These are consistent with, though do not replicate, the Lou et al. claim: the ratios here involve a cross-asset crypto predictor applied to thin Asian equity markets, not US factor anomalies. The directional alignment is strong; the mechanism the two studies posit is the same (information consolidated at the open), applied to a different information source.

Hendershott, Livdan, and Rosch (2020, "Asset Pricing: A Tale of Night and Day," *Journal of Financial Economics*) extend the overnight theme by showing that overnight and daytime betas differ in their cross-sectional pricing implications. The index-level results in Pass 2 (index_hk gap IC = 0.185, index_kr gap IC = 0.277; both p = 0.000) relative to near-zero intraday IC (index_hk intraday IC = -0.013, index_kr intraday IC = 0.010) are an extreme version of the Hendershott et al. pattern: overnight variation in crypto carries essentially all the predictive information at the index level, while intraday does not.

Berkman, Koch, Tuttle, and Zhang (2012, "Paying Attention: Overnight Returns and the Hidden Cost of Buying at the Open," *Journal of Financial and Quantitative Analysis*) attribute overnight return concentration to attention effects. Retail and institutional attention is highest at the open; stocks that attracted attention during the prior trading session (or overnight news cycle) experience concentrated price adjustment at the gap. This framework accommodates the crypto overnight channel naturally: crypto price moves during Asian market closure are a salient, attention-triggering signal for crypto-exposed equity investors who review overnight developments before the open.

The 3:1 and 2.1:1 IC ratios in Pass 2 are thus CONSISTENT with three distinct but compatible mechanisms from prior work: institutional order flow at the open (Lou et al.), differential overnight beta pricing (Hendershott et al.), and attention-driven concentration at the open (Berkman et al.). No prior paper has applied this decomposition to a cross-asset, cross-timezone information channel of this kind.

---

## 2. Cross-Asset Information Transmission

The traditional lead-lag literature provides the structural foundation. Hong, Torous, and Valkanov (2007, "Do Industries Lead Stock Markets?" *Journal of Financial Economics*) show that industry returns lead broad market returns by up to one month, attributing this to information diffusion speeds. The mechanism they identify applies here: continuously-traded crypto markets process global information around the clock, while gap-constrained Asian equity markets can only respond at the open. The information-speed differential is structural, not informational asymmetry in the classical sense.

For cross-asset crypto-to-equity transmission, the literature is sparse. Corbet, Meegan, Larkin, Lucey, and Yarovaya (2018, "Exploring the Dynamic Relationships between Cryptocurrencies and Other Financial Assets," *Economics Letters*) document asymmetric volatility transmission from crypto to equities, with crypto more often the net transmitter. Pass 2 takes this transmission direction as a given and asks a more refined question: in which return component (gap vs. intraday) does the transmission appear? The answer -- concentrated at the gap, near-zero intraday -- is not directly addressed in Corbet et al. or subsequent daily-frequency spillover work.

Liu and Tsyvinski (2021, "Risks and Returns of Cryptocurrency," *Review of Financial Studies*) frame crypto as a distinct risk factor with partial but not full overlap with equity risk appetite proxies. The Pass 2 SHAP results are consistent with this: eth_ov_log_return and btc_ov_log_return together account for the majority of gap IC, while vix_level also contributes but less than either crypto feature (feature_ablation_pass2.csv: drop_crypto_overnight_all reduces gap IC by 0.021 in HK and 0.033 in KR, compared to drop_macro_all reducing it by 0.002 in HK and increasing it by 0.005 in KR). The crypto channel is incremental to the macro channel, consistent with Liu-Tsyvinski.

The notion of a "Forecasting Ethereum with Machine Learning" pipeline (Liew and Mayster and related work) is a precursor to the cross-asset prediction framework here, though that literature targets crypto prices from financial inputs, not equity returns from crypto inputs. Pass 2 reverses the prediction direction and applies it to a curated selection-filtered universe. Academic precedent for that specific configuration is thin; Pass 2 is a relatively novel data point.

---

## 3. The Control Universe Result and Selection Bias Literature

The control universe test is the clearest methodological contribution of Pass 2. The brief established that selecting on trailing BTC correlation then predicting with BTC/ETH features is "mildly tautological," and required a counterfactual universe to quantify this.

The result is market-dependent. In KR, the main-minus-control gap IC difference is 0.024 with bootstrap p = 0.008, indicating that the crypto-filtered universe captures meaningfully more gap predictability than a matched non-crypto universe. In HK, the difference is only 0.010 with p = 0.927: statistically indistinguishable from zero.

Lo and MacKinlay (1990, "Data-Snooping Biases in Tests of Financial Asset Pricing Models," *Review of Financial Studies*) established the general principle that universe construction criteria correlated with the target variable inflate apparent performance. Harvey, Liu, and Zhu (2016, "... and the Cross-Section of Expected Returns," *Review of Financial Studies*) extend this to the multiple-testing problem in factor research. Both frameworks predict exactly the HK finding: when the control gap IC ratio (4.80) exceeds the main gap IC ratio (3.14), the selection filter is not adding gap-specific predictability -- it is selecting stocks that are gap-predictable for reasons that have nothing to do with crypto. The HK signal does not clear the circularity threshold.

Kadan and Liu (2014, "Performance Evaluation with High Moments and Disaster Risk," *Journal of Financial Economics*) provide a more general framework: measures of performance that do not account for the selection mechanism that generated the evaluation universe can overstate genuine alpha. The HK control result is a concrete demonstration of this: the "crypto overnight signal" in HK is not specific to crypto-exposed stocks.

The KR result is more favorable. A statistically significant main-minus-control gap IC difference (p = 0.008) indicates that the selection rule does add something: the KR crypto-filtered universe is genuinely more gap-predictable from crypto features than a matched control. This is the one market where the circularity critique is partially, quantitatively addressed. The KR control gap:intraday ratio of 0.88 (below 1.0) further reinforces this: non-crypto-filtered KR stocks show no gap dominance at all, whereas the main KR universe shows a 2.15 gap:intraday ratio. The contrast is mechanistically interpretable.

---

## 4. Model Selection: LGBM vs. TCN

The horse race result -- LGBM significantly outperforming TCN on gap IC in both markets (horse_race_bootstrap.csv: HK IC difference = -0.021, p = 0.006; KR IC difference = -0.028, p = 0.000) -- is CONSISTENT with the tabular machine learning benchmark literature.

Gu, Kelly, and Xiu (2020, "Empirical Asset Pricing via Machine Learning," *Review of Financial Studies*) find that tree-based methods and neural networks both outperform classical OLS in US equity return prediction, but their comparison does not sharply distinguish gradient boosting from deep learning on structured financial data. The broader tabular ML benchmark literature (Grinsztajn, Oyallon, and Varoquaux 2022, "Why Tree-Based Models Still Outperform Deep Learning on Tabular Data," *NeurIPS 2022*) documents a systematic advantage for tree-based models on structured/tabular data with mixed feature types, irregular distributions, and small-to-medium sample sizes -- precisely the conditions here (short panel, heterogeneous feature types, limited observations per fold). LGBM's superiority over TCN on gap prediction is therefore not surprising and is consistent with first principles for the data regime.

---

## 5. Regime Conditioning

The regime analysis (regime_analysis_pass2.csv) shows that gap IC is elevated in high-VIX periods in both markets (HK VIX-high: 0.084 vs. VIX-low: 0.057; KR VIX-high: 0.079 vs. VIX-low: 0.054). This is consistent with Cooper, Gutierrez, and Hameed (2004, "Market States and Momentum," *Journal of Finance*), who show that strategy performance depends on prevailing market conditions, and with Asness, Moskowitz, and Pedersen (2013, "Value and Momentum Everywhere," *Journal of Finance*), who document that cross-asset strategies exhibit regime-conditional behavior. The VIX conditioning here aligns with Bouri et al.'s finding that crypto-equity co-movement increases during stress: the gap-predictability channel is stronger precisely when global risk appetite moves are largest, amplifying the crypto signal's relevance.

---

## 6. Market Efficiency Implications

If the crypto overnight gap signal is genuine in KR and partly circularity-driven in HK, the efficiency interpretation differs by market. For KR, the result is consistent with Grossman and Stiglitz (1980, "On the Impossibility of Informationally Efficient Markets," *American Economic Review*): a persistent information-speed differential between 24-hour crypto markets and gap-constrained equity markets can survive in equilibrium if the costs of arbitraging it exceed the signal value. The net-of-cost results confirm this: the KR index-level strategy has net Sharpe 3.74 (index only, low cost), but individual stock strategies fail on realistic transaction costs.

Chordia, Roll, and Subrahmanyam (2008, "Liquidity and Market Efficiency," *Journal of Financial Economics*) show that thin markets with high transaction costs retain more return predictability. Both the HK small-cap and KR universes fit this description, and the signal surviving on gross terms while failing net is the canonical Chordia et al. pattern.

For the index-level result (index gap IC 0.185 HK, 0.277 KR), an information-speed rather than risk-premium interpretation is more plausible: crypto ETF or index products reflect crypto price information continuously, while the equity index gaps at the open. This is not a risk premium but a microstructural lag.

---

## Bottom Line

Pass 2 situates cleanly in the overnight return anomaly literature (Lou et al. 2019; Hendershott et al. 2020; Berkman et al. 2012): gap-concentrated IC from a cross-asset crypto predictor is exactly what those frameworks predict for an information source that resolves before the equity open. The cross-asset crypto-to-equity channel is under-studied at this granularity; Pass 2 adds a novel market and time-window dimension.

The control universe finding divides the finding by market in a way that is directly predicted by the selection bias literature (Lo-MacKinlay 1990; Harvey-Liu-Zhu 2016): HK shows no crypto-specificity after the circularity test; KR retains a statistically significant increment. The LGBM-over-TCN result is consistent with established tabular ML benchmarks. Taken together, the Pass 2 findings partially confirm and partially qualify the Pass 1 signal: genuine in KR at the index level and with crypto-specificity surviving the counterfactual, less clearly so in HK individual stocks where the selection mechanism cannot be distinguished from the signal mechanism.
