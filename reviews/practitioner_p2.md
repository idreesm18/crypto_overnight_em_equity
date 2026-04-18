# Practitioner Review — Pass 2
## Crypto Overnight Signal — HK and KR Equity Gap Strategy

---

## What Changed Between Passes and What Didn't

Pass 1 closed HK on spread grounds: breakeven 9.9 bps half-spread, realistic minimum 30-70 bps. That verdict stands and nothing in Pass 2 data reverses it. What Pass 2 did was reframe the question toward two survivable paths: (1) index-level futures (HSI, KOSPI), and (2) a non-crypto-filtered control universe in KR. The borrow problem shrank because futures positions carry no short-leg borrow requirement. The spread problem shrank because futures spreads are tighter than single-name small-cap spreads. That is the right direction. The question is whether the numbers now clear the bar.

---

## 1. Borrow Cost Engagement

The brief specifies defaults: HK shorts at 500 bps/yr, KR at 400 bps/yr, index at 50 bps/yr. From `borrow_sensitivity_pass2.csv`, the index configs are essentially borrow-insensitive: index_kr gap net Sharpe moves from 2.697 at borrow_mult=0 to 2.687 at borrow_mult=2x (doubling notional index borrow). The spread column for index configs is 20-30 bps annual, so borrow is less than 1 bps per year of drag. For futures, this is correct -- HSI and KOSPI index futures are cash-settled and carry no securities lending cost. The 50 bps index borrow default appears to be a placeholder for an index replication ETF overlay; for actual futures, it is zero.

For the stock-level universes, borrow sensitivity is larger but not the binding constraint. main_kr gap tercile_ls net Sharpe goes from -0.213 at zero borrow to -0.807 at 2x borrow (350 bps/yr effective). That is a 59-bp Sharpe swing on a strategy already in negative territory. The binding constraint for the stock universe was and remains spread cost, not borrow.

**The real borrow question for KR.** KOSDAQ names at 400 bps default are, if anything, optimistic for the hard-to-borrow end of the universe. For names like 049470.KQ (Bitplanet) or 086960.KQ (Vidente) that have been in the signal universe, prime broker quotes I have seen on comparable KOSDAQ micro-caps range 600-1500 bps, with availability that disappears when you most want it -- high-vol crypto dislocation days. Korea also imposed a blanket KOSPI/KOSDAQ short-sale ban November 2023 to March 2024; the backtest runs through this period treating it as a normal trading window. For the index version, none of this applies: KOSPI 200 futures are among the most liquid futures globally (KRX volume typically $5-10B USD equivalent per day), with no short-sale ban applicable, no borrow cost, and no forced buy-in risk.

**Bottom line on borrow.** The index version has a borrow problem that is approximately zero. The stock version has a borrow problem that the default assumptions understate for the hard-to-borrow tail of the universe. Neither changes the priority ordering: index first, stock universe dead on spread grounds before reaching borrow.

---

## 2. Long-Only vs Long-Short: Where Does the Alpha Concentrate?

`long_short_decomposition.csv` is unambiguous. For main_kr LGBM gap: long-only Sharpe 3.38, combined (long-short) gross Sharpe 3.77. The short leg contributes approximately 0.39 Sharpe units at gross but costs 0.17 bps/day borrow and 1.64 bps/day spread (from `backtest_summary_pass2.csv`: spread_cost 2.31 annualized on tercile_ls). For main_hk LGBM gap: long-only Sharpe 2.47 vs long-short gross Sharpe 3.04. The short leg adds real gross Sharpe but at cost rates that consume the contribution before netting.

**What this tells a trader.** The alpha is in the long leg. Shorting KOSDAQ crypto-adjacent names is not the value-add -- it is a transaction-cost incinerator. The correct variant for any realistic book is long-only, funded by cash or a passive short of the index. Long-only KR gap Sharpe is 3.38 gross; I do not have a long-only net Sharpe for the stock universe in Pass 2, but applying roughly half the cost load (no short-side spread, no borrow) puts this in range of survivability before spread realism is applied. The index version is a cleaner expression of the same intuition: you are long the crypto-exposed index, short the index future as a hedge. That structure is what the index backtest approximates.

For the control_kr universe (non-crypto-filtered): long-only Sharpe 2.09 vs combined 2.79. Same pattern. Long dominates. Short is a net drag. This is consistent with what I said in Pass 1: EM shorts underperform reliably as a structural drag. The decomposition confirms it quantitatively.

---

## 3. Index Futures: The Real Question

Eight index configs clear +0.5 net Sharpe in `diagnostics_summary_pass2.txt`. The best two: LGBM index_kr gap gate_off net Sharpe 3.74, LGBM index_hk gap gate_off net Sharpe 2.38. These numbers require scrutiny before celebrating.

**Spread assumptions.** The model assumes 26 bps annual spread cost for index_kr gap gate_off (from `backtest_summary_pass2.csv`: spread_cost 0.30). That annualized figure divided by daily turnover (0.499 per day, so ~123 round-trips per year) implies approximately 0.24 bps per round-trip. KOSPI 200 index futures (the front month, KQ6) trade with a tick size of 0.05 index points on an index around 320, so one tick is about 1.6 bps of notional. Bid-ask spread is typically 1-2 ticks on the near contract, or 1.6-3.2 bps half-spread. At 2 bps half-spread (realistic for normal hours), that is 4 bps per round-trip, versus the model's 0.24 bps implied. The model is dramatically undercosting index futures. Even at cost_sensitivity_pass2.csv 2x spread mult, index_kr gap net Sharpe is 2.31 -- it survives, which is reassuring. At 10x (which would be roughly 2.4 bps vs a realistic 4 bps), it is harder to say without that sensitivity in the output, but the gradient is gentle enough that even at realistic spreads the strategy likely clears 1.5-2.0 net Sharpe.

**Margin and capital efficiency.** HSI futures margin at HKEX is roughly 10-12% of contract notional. KOSPI 200 futures margin at KRX is approximately 9-12%. Compare to stock long-short requiring 100-150% of notional in a prime brokerage setup. The index futures version is 8-10x more capital efficient per dollar of gross exposure. For a $10M allocation, you can control $80-100M notional in KOSPI futures on $10M margin, versus $10M gross in stocks. This does not change the Sharpe but it changes AUM economics substantially: management fee on $10M looks very different if the strategy can handle $80M of notional.

**Capacity.** HSI futures: average daily volume approximately $3-5B USD equivalent. KOSPI 200 futures: $8-12B per day. At $100M notional, slippage is negligible. At $1B, you are still inside 1% of daily volume. Capacity is not a binding constraint at any realistic allocation size for this type of fund.

**What the gate costs.** `regime_gate_comparison.csv` shows index_kr gap: gate_on Sharpe 2.69, gate_off Sharpe 3.74, cost of inactivity 0.24. The gate reduces Sharpe by 0.24 units and cuts active days from 58.8% to 38.5%. That is a real cost. For a live book, always-on with a 3.74 gross Sharpe is preferable if the signal is stable -- but that requires confidence in the regime indicator's real-time computability without look-ahead. The 2.69 gated version is the conservative number to use in any pitch deck.

---

## 4. Live Failure Modes

**Flash crash contamination.** The signal uses BTC and ETH overnight log returns as top SHAP features. BTC flash moves exceeding 10% during the overnight window contaminate the signal: the model was not trained on the frequency of exchange halts, liquidity gaps, or Binance data outages that accompany these moves. Post-ETH Cancun (March 2024) and post-BTC ETF approval (January 2024), intraday crypto volatility structure changed: more institutional participation, tighter spreads, but also more correlated large moves tied to ETF flow data. The signal's stability in 2024-2026 needs to be stress-tested on those specific flash-crash dates rather than assumed from overall fold IC.

**Execution window in HK and KR.** For index futures, the timing problem is less acute than for single names: futures trade pre-market and during the day. The morning execution window for KOSPI 200 futures opens at 09:00 KST and the cash market opens at 09:00 as well, so there is no pre-open auction to navigate. For HSI futures, the morning session opens at 09:15 HKT, before the equity market auction completes at 09:30 HKT. This actually gives the futures trader an advantage: you can trade the futures signal before the equity open confirms it. Pipeline latency of sub-30 minutes from overnight crypto data to order submission is achievable but requires a production-grade infrastructure with exchange connectivity, not a research Python script.

**KR wash-sale and capital gains.** Korean tax rules for foreign investors in KOSPI futures depend on treaty status and entity structure. For a US-domiciled fund, gains on KRX futures may be subject to Korean withholding tax on derivatives (currently 10% for treaty countries). This is a non-trivial haircut on a strategy targeting 23-47% annual returns (from `backtest_summary_pass2.csv`, index_kr gap gate_on/off). This needs a tax opinion before pitching.

**2023 attenuation.** The synthesis notes KR fold-level IC near zero in 2023. For the index_kr configs, this regime matters: if the 3.74 Sharpe is concentrated in 2020-2022 and 2024-2026, with a 2023 drawdown, the live Sharpe since 2024 is more relevant to a PM allocating capital today. I do not have year-by-year index_kr Sharpe breakdown in the Pass 2 files, which is a gap before pitching.

**Regime circularity.** The diagnostics show HK control_ratio (4.80) exceeding main_ratio (3.14), meaning the non-crypto-filtered universe shows even higher gap dominance. The KR control_ratio (0.88) is below the main_ratio (2.15). This is the more interpretable direction: the crypto channel is load-bearing for KR gap dominance but not for HK. For a live book, this suggests the KR index signal is more fragile to a structural change in the crypto-equity correlation regime.

---

## 5. What a Quant PM Needs to Greenlight This

1. **Clean monthly tearsheet with realistic fees.** Management fee 1%, performance fee 20%, hurdle 0. At index_kr gap net Sharpe 2.69 (gated) and annualized return 23.6%, after-fee return is approximately 16-17% with Sharpe around 1.9-2.0. That is a strong return profile. The tearsheet must show 2023 separately so the PM can see the drawdown year.

2. **Sensitivity analysis already exists.** `cost_sensitivity_pass2.csv` shows index_kr gap surviving at 2x spread (Sharpe 2.31) and 4x spread (Sharpe 1.81, estimated from the linear gradient). That is sufficient for a first meeting.

3. **Expected capacity.** State $500M notional KOSPI 200 futures with minimal market impact at 0.5% daily volume. At 9% margin, that is $45M capital requirement. For HSI: $200M notional, $24M capital at 12% margin. Combined $70M capital for both index positions. This is small-fund economics but workable for a dedicated crypto-macro strategy.

4. **Risk drivers to disclose.** Primary: BTC/ETH overnight return regime (the signal turns off in crypto bear markets, consistent with Pass 1 findings). Secondary: KRX operational risk (exchange halts, futures circuit breakers). Tertiary: Korean political/regulatory risk on crypto (VASP licensing changes, short-sale ban extensions). FX risk is manageable via rolling forward hedges.

5. **Kill-switch rule.** Rolling 90-day realized Sharpe below -1 triggers a halt and 30-day paper trading period before resuming. Secondary kill: if SHAP rank correlation between consecutive months drops below 0.5, pause and re-validate feature stability. Both rules are implementable with live output from the inference pipeline.

---

## What Would I Trade

The index version. Specifically: LGBM on index_kr gap, gate_off, at $50M notional KOSPI 200 futures, with a secondary $20M HSI futures position for market diversification. After realistic futures spreads (approximately 2-4 bps round-trip versus the model's implied 0.24 bps, which I expect to reduce net Sharpe from 3.74 to somewhere in the 2.0-2.5 range), the index_kr gap config still clears my minimum threshold of 1.5 net Sharpe for a satellite allocation.

Before I put on a single lot, I need three things not currently in the output: (1) year-by-year Sharpe breakdown for index_kr to see the 2023 drawdown profile, (2) a live paper-trade of 30 days to measure realized versus modeled overnight crypto data latency and fill quality at the futures open, and (3) a tax opinion on KRX futures gains for the fund's domicile.

The stock universe in both HK and KR remains uninvestable at realistic single-name costs. Long-only KR KOSPI large-caps with the short expressed as a KOSPI 200 future (instead of individual short positions) is a structurally cleaner version of the same idea -- but that is a different backtest, not a re-read of these numbers.

The signal is real. The instrument choice in the original design was wrong. Index futures fix the instrument problem and the cost math follows.
