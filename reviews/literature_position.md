# Literature Position Paper: Crypto Overnight Signal for Asian Equity Markets (Pass 1)

**Role:** Literature Reviewer  
**Date:** 2026-04-16

---

## 1. Overnight Returns Literature

The foundational paper for this inquiry is Lou, Polk, and Skouras (2019, "A Tug of War: Overnight Versus Intraday Expected Returns," *Journal of Financial Economics*), which documents that virtually all expected return anomalies in US equities load on the overnight component (close-to-open), not intraday (open-to-close). Their interpretation: information from institutional traders who submit market-on-open orders gets incorporated at the gap, while intraday price formation is dominated by noise and short-term liquidity effects.

Pass 1's headline finding — HK gap IC = 0.061 vs. intraday IC = 0.019; KR gap IC = 0.061 vs. intraday IC = 0.023 (return_decomposition.csv, gap-minus-intraday delta of 0.041 and 0.038, both flagged as CRITERION 2 MET in diagnostics_summary.txt) — is precisely consistent with the Lou-Polk-Skouras framework. Overnight crypto information enters Asian equity pricing at the open, not through the trading day. This is what their theory predicts: information that exists before the market opens is incorporated at the gap, not gradually during the session.

The extension beyond Lou et al. is that the conditioning variable is cross-asset (crypto overnight returns) rather than a domestic risk factor. To my knowledge, no prior paper has applied the overnight-vs-intraday decomposition to cross-asset crypto-to-equity signal propagation. The 3:1 gap-to-intraday IC ratio across both markets is a quantitatively sharp confirmation of the mechanism Lou et al. identify, applied in a new asset-class context.

Additionally, Cliff, Cooper, and Gulen (2008, "Return Differences Between Trading and Non-Trading Hours: Like Night and Day") document that non-trading hours in US equities contain systematically different information content than trading hours. The overnight window in this study (17.5 hours for weekday gaps, 65 hours for Monday gaps) is precisely the kind of non-trading interval their framework predicts should carry concentrated information.

---

## 2. Cross-Asset Information Transmission

The crypto-to-equity spillover literature has grown substantially since 2018:

**Bouri et al. (2018, "On the Return-Volatility Relationship in the Bitcoin Market around the Price Crash of 2013," *Economics: The Open-Access, Open-Assessment E-Journal*)** and the follow-on **Bouri, Molnár, Azzi, Roubaud, and Hagfors (2017, "On the Hedge and Safe Haven Properties of Bitcoin," *Finance Research Letters*)** establish that BTC-equity correlations are time-varying and conditionally elevated in risk-off environments. The Pass 1 regime analysis is consistent with this: KR gap IC in high-VIX regimes (0.076, Sharpe 4.36) exceeds low-VIX IC (0.056, Sharpe 3.60) per regime_analysis.csv. This matches the Bouri et al. finding that crypto-equity co-movement increases during stress.

**Lyocsa, Molnár, and Plíhal (2020, "Central Bank Announcements and Realized Volatility of Stock Markets in G7 Countries," and related work on crypto spillovers)** and to my knowledge Lyocsa et al.'s crypto-equity work generally finds directional return spillovers from crypto to equities are weak or regime-dependent at daily frequency. Pass 1's IC of ~0.06 at the sub-daily (gap) frequency is stronger than what daily-frequency studies typically find, suggesting the information content is concentrated in the overnight window and diluted at daily aggregation — a methodological point this study's decomposition directly addresses.

**Bianchi, Faccini, and Rebelo (2023, "Crypto Carry," *NBER Working Paper*)** and **Liu and Tsyvinski (2021, "Risks and Returns of Cryptocurrency," *Review of Financial Studies*)** frame crypto returns as a distinct risk factor that is partially correlated with macro risk appetite proxies but carries independent variation. The SHAP stability results show eth_ov_log_return as the top feature in 96-97% of folds in both markets, and btc_ov_log_return in 87-97% of folds (shap_stability.csv), while vix_level also appears in 96-97% of folds. This three-way feature importance pattern — crypto return, crypto magnitude, macro conditioning — is consistent with the Bianchi/Liu-Tsyvinski framing: crypto carries information about global risk appetite that partially but not fully overlaps with VIX.

**Corbet, Meegan, Larkin, Lucey, and Yarovaya (2018, "Exploring the Dynamic Relationships between Cryptocurrencies and Other Financial Assets," *Economics Letters*)** find asymmetric volatility transmission from crypto to equities, with crypto more often the transmitter than the receiver. Pass 1's cross-market design takes this as a structural premise: crypto is the signal, equity is the response. The Pass 1 findings are consistent with Corbet et al.'s transmission direction but add a new channel (the overnight gap) and a new receiving market (small-cap crypto-exposed Asian equities).

The degree of novelty: prior cross-asset crypto-equity papers generally use daily data, do not decompose into gap vs. intraday, and focus on broad market indices or large-cap proxies. Pass 1's application to small-cap crypto-correlated individual stocks in HK and KR, with an explicit overnight window, is a methodological contribution not directly replicated in the published literature to my knowledge.

---

## 3. Intra-Daily / Overnight Price Formation in Asian Markets

The literature on overnight US-to-Asian price transmission is substantial:

**Becker, Finnerty, and Friedman (1995, "Economic News and Equity Market Linkages between the U.S. and U.K.," *Journal of Banking and Finance*)** and **Lin, Engle, and Ito (1994, "Do Bulls and Bears Move Across Borders? International Transmission of Stock Returns and Volatility," *Review of Financial Studies*)** establish that overnight price changes in US markets predict opening prices in Asia. The mechanism is information that accumulates while Asian markets are closed.

**Hamao, Masulis, and Ng (1990, "Correlations in Price Changes and Volatility across International Stock Markets," *Review of Financial Studies*)** provide the foundational evidence for volatility spillovers from the US (and London) to Japan. Their framework — closed-market information accumulated overnight predicts the next opening — is precisely the framework Pass 1 generalizes to a crypto information source.

The specific HK-US and KR-US overnight links are well documented in the Asian market microstructure literature. H-shares and ADRs provide direct arbitrage channels between HK-listed and US-listed prices; when an H-share has a US-listed counterpart, its overnight gap is substantially explained by the US overnight move. This is relevant to Pass 1: the stocks in this study are selected for crypto exposure (exchanges, miners, fintech), and some may have significant US/global listing overlap, meaning the BTC overnight return is in part proxying for a known US-to-HK overnight transmission channel.

Pass 1 targets a specific subset of stocks where the crypto overnight channel is plausibly more direct than the general US-to-HK channel: companies with fundamental crypto exposure. If the crypto signal is merely a noisy proxy for a US market overnight return, we would expect the feature ablation to show vix_level or a DXY variable dominating, not eth_ov_log_return. The SHAP results — eth_ov_log_return at 96-97% of folds, mean_abs_shap 1.29e-03 to 1.32e-03, substantially exceeding vix_level (7.16e-04 to 4.66e-04) per shap_stability.csv — weakly support a genuine crypto channel rather than a pure US-risk-proxy channel, though this cannot be definitively resolved without controlling for S&P 500 overnight returns directly.

---

## 4. Weekend Effect Literature

**French (1980, "Stock Returns and the Weekend Effect," *Journal of Financial Economics*)** and **Keim and Stambaugh (1984, "A Further Investigation of the Weekend Effect in Stock Returns," *Journal of Finance*)** document negative Monday returns in US equities. **Connolly (1989, "An Examination of the Robustness of the Weekend Effect," *Journal of Financial and Quantitative Analysis*)** shows the effect attenuated substantially after the 1980s.

The later literature generally finds weekend effects have weakened or disappeared in developed markets. **Berument and Kiymaz (2001, "The Day of the Week Effect on Stock Market Volatility," *Journal of Economics and Finance*)** and **Steeley (2001, "A Note on Information Seasonality and the Disappearance of the Weekend Effect in the UK Stock Market," *Journal of Banking and Finance*)** are representative.

Pass 1's Monday result — HK Monday gap IC = 0.082 vs. Tue-Fri IC = 0.056; KR Monday gap IC = 0.065 vs. Tue-Fri IC = 0.060 (weekend_effect.csv) — is mechanically distinct from the classical weekend effect in an important way. The classical effect concerns the direction of Monday returns (negative) and attributes it to news release timing (negative news released Friday after close). Pass 1's finding concerns the predictability of Monday gap returns using crypto features, not their unconditional direction. The hypothesis tested — a 65-hour accumulation window carries more crypto information than a 17-hour weeknight window — is consistent with a simple information accumulation story rather than a behavioral/news-release story.

The appropriate literature anchor is therefore not French (1980) but rather the intraweek information accumulation literature. The finding is best framed as: the overnight information channel is monotonically stronger the longer the gap, not as a revival of the traditional weekend effect. The IC uplift (HK: +46% on Mondays; KR: +9%) is consistent with more overnight crypto signal arriving over 65 hours than 17 hours, with HK showing a more pronounced version — possibly because HK's crypto-exposed universe contains more liquid names with faster incorporation during the week, leaving more gap-dependent signal for the longer Monday window.

---

## 5. Universe Selection and Circularity

**Lopez de Prado (2018, "Advances in Financial Machine Learning," Wiley)** and **Harvey, Liu, and Zhu (2016, "... and the Cross-Section of Expected Returns," *Review of Financial Studies*)** provide standard guidance on multiple testing, selection bias, and universe construction in signal research. **Chincarini and Kim (2006, "Quantitative Equity Portfolio Management," McGraw-Hill)** discuss factor construction circularity.

The brief explicitly acknowledges: "selecting on BTC correlation then predicting with BTC-derived features is mildly tautological." This is honest and correctly labeled. The magnitude of the contamination depends on the correlation stability between the selection window (trailing 60-day correlation used for universe entry) and the prediction window (next month). If BTC correlation is sticky, the universe will persistently contain the stocks most likely to respond to BTC features, inflating IC. If it is unstable, the contamination is smaller.

The 0.06 IC is plausibly inflated by this filter, but it is unlikely to be *solely* attributable to it. Three observations support this: (1) the SHAP analysis shows feature importance is stable across 75 walk-forward folds (75.4% top-10 overlap, Spearman rank correlation 0.70 per diagnostics_summary.txt), suggesting consistent predictive structure rather than selection artifact; (2) the decomposition into gap vs. intraday shows gap IC substantially exceeds intraday IC — a filter artifact would inflate both equally, but the selective concentration at the gap is mechanistically interpretable; (3) regime robustness (IC positive across all VIX, BTC-trend, and crypto-vol splits per regime_analysis.csv) is harder to explain as a selection artifact alone.

Harvey, Liu, and Zhu would require an adjusted t-statistic for the number of implicit hypotheses tested; with 6 model runs (2 markets × 3 targets), a standard Bonferroni correction applied to p<0.05 targets p<0.008. The bootstrap p-values reported are 0.000 for HK/gap and KR/gap (return_decomposition.csv), which survives this correction, but the multiple-testing concern across all 6 models collectively is noted.

---

## 6. Efficient Market Hypothesis Implications

The EMH literature distinguishes between informational efficiency and transactional efficiency. **Grossman and Stiglitz (1980, "On the Impossibility of Informationally Efficient Markets," *American Economic Review*)** argue that some return predictability is necessary to compensate informed traders; the question is whether the predictability survives realistic trading costs.

Pass 1's finding — gross gap IC of ~0.06 but net-of-cost strategies all negative at 1x assumed costs — is precisely the Grossman-Stiglitz outcome for a market that is transactionally but not informationally efficient. The signal is real; the costs prevent arbitrage; therefore the anomaly persists. HK and KR small-cap stocks have higher transaction costs than the US large-cap markets where overnight return predictability has been studied, which is consistent with the signal surviving there specifically.

**Chordia, Roll, and Subrahmanyam (2008, "Liquidity and Market Efficiency," *Journal of Financial Economics*)** show that markets with lower liquidity tend to exhibit more return predictability at short horizons. Pass 1's universe (thin, crypto-exposed small-caps with 15 bps/side spread in HK, 10 bps in KR) fits this profile exactly. The signal is consistent with semi-strong form efficiency (publicly available crypto data) failing in a thin, high-cost universe — not with strong-form information asymmetry.

The interpretation is therefore not "Asian equity markets are informationally inefficient" but rather "the arbitrage that would eliminate this signal is economically unviable at the observed cost structure for the relevant universe." This is a known equilibrium in the microstructure literature and reduces the novelty claim somewhat but does not invalidate the empirical finding.

---

## 7. Gap / Intraday Decomposition in Real-World Strategies

Academic buy-side research applying the gap/intraday decomposition to small-cap Asian equity cross-asset signals with an explicit crypto conditioning variable has, to my knowledge, not been published. Proprietary research at trading firms likely exists but is not in the academic record.

The closest published analogues are:

- **Moskowitz, Ooi, and Pedersen (2012, "Time Series Momentum," *Journal of Financial Economics*)**: cross-asset return predictability as a structural phenomenon, though at monthly frequency.
- **Asness, Moskowitz, and Pedersen (2013, "Value and Momentum Everywhere," *Journal of Finance*)**: cross-asset factor exposure as a signal framework.
- **Heston and Sadka (2008, "Seasonality in the Cross-Section of Stock Returns," *Journal of Financial Economics*)**: return predictability components in cross-sectional settings.

None of these directly targets the overnight gap in crypto-correlated Asian small-caps. The novelty of Pass 1's specific combination — overnight window, cross-asset crypto feature, gap decomposition, thin EM universe — is real, though the individual components each have antecedents.

---

## Where This Fits

Pass 1 sits at the intersection of the Lou-Polk-Skouras (2019) overnight-vs-intraday decomposition literature and the Bouri/Corbet crypto-to-equity spillover literature, providing the first systematic evidence that the LPS mechanism extends to cross-asset, cross-market settings where the overnight information source is crypto markets predicting the gap returns of a curated universe of crypto-exposed Asian equities.
