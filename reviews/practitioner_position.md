# Practitioner Position Paper
## Crypto Overnight Signal — HK & KR Equity Gap Strategy
### Pass 1 Adversarial Review Panel

---

## Summary of Results Under Review

- HK gap: gross Sharpe 3.04, net Sharpe -1.57, breakeven 0.66x spread
- KR gap: gross Sharpe 3.77, net Sharpe -0.28, breakeven 0.93x spread
- Spread assumption: 15 bps/side HK, 10 bps/side KR
- Position size: $100K per name, 8-10 names per leg
- Turnover: 42-50% daily (effectively daily rebalance)

The cost model does not include borrow. This is the first of several places where the backtest lives in a world that does not exist.

---

## 1. Short Availability and Borrow Cost

**HK.** Five of the HK candidate names — 8267.HK (Linekong), 8540.HK (Victory Securities), 8005.HK (Yuxing InfoTech), 8521.HK (WebX), SORA.HK — are GEM board micro-caps. Four of these were skipped entirely by Stage 2 because Stooq had no price data for them (`universe_summary.txt`). The ones that did make it through — 0863.HK (OSL Group), 1726.HK (HKE Holdings), 1022.HK (Lingxi Interactive) — are main board but thinly traded, with observed ADV entries at or just above the $500K floor.

Short-selling on the HKEX requires borrowing shares through a securities lending arrangement. For main board names below $5M ADV, borrow is sporadic and expensive: typical rates run 200-500 bps annualized for thin names, and availability can disappear at exactly the moments you want to be short (high-vol crypto dislocations). GEM board names are often not borrowable at all through prime brokers; naked short-selling is not permitted under HKEX rules.

The cost model includes zero borrow cost. At 200 bps/yr on the short leg ($800K notional, 8 names × $100K):

- Annual borrow drag: $16,000 (2.0% of $800K)
- As annualized return on $1.6M gross: approximately 100 bps/yr
- Sharpe unit impact: ~-0.08 (vol 13.3%)

At 300 bps borrow, impact is ~-0.12 Sharpe. KR gap net Sharpe is already -0.28; adding 300 bps borrow pushes it to approximately -0.40. This is before addressing spread realism.

**KR.** Korea restricts short-selling on KOSDAQ names below certain market cap thresholds. KOSDAQ names like 049470.KQ (Bitplanet) — observed at $500K ADV in July 2025 — are precisely the names likely to face temporary short-sale bans during high-stress periods. Korea imposed a blanket KOSPI/KOSDAQ short-sale ban from November 2023 to March 2024. The backtest treats this as a normal operating period.

**Revised KR breakeven with borrow:** From 0.93x (model) to approximately 0.81x at 300 bps borrow. The strategy is net-negative at any realistic borrow rate before touching spread realism.

---

## 2. Spread Reality

**HK.** The model assumes 15 bps half-spread for HK. This might be defensible for 0700.HK (Tencent, $800M+ ADV) or 0388.HK (HKEX, $100M+ ADV). It is not defensible for the names that actually drive the signal.

0863.HK (OSL Group) trades in the HKD 1-3 range historically. HKEX minimum tick is HKD 0.01. At HKD 1.50, the minimum quoted spread is 0.01/1.50 = 67 bps. This is 4.5x the model assumption. Universe data confirms 0863.HK appeared at $502K ADV in October 2022 — a period when the model was most active in crypto-adjacent names.

Other HK names: 1726.HK (HKE Holdings) at $530K ADV, 1022.HK (Lingxi Interactive) at $516K ADV. For sub-$1M ADV HK names, realistic half-spreads are 30-80 bps.

**Model breakeven at real HK spreads:** Breakeven is 0.66x of 15 bps = 9.9 bps. If actual spread is 45 bps (3x), the model is operating at 4.5x breakeven. Net Sharpe at 2x spread multiplier: -5.9. This is not a rounding error; the HK version is structurally unexecutable.

**KR.** 10 bps half-spread for KOSDAQ names is generous. KOSPI blue-chips (Samsung, Hynix) trade at 2-5 bps, but these don't drive the crypto signal. KOSDAQ names in the universe — 049470.KQ (Bitplanet), 112040.KQ (WeMade), 086960.KQ (Vidente) — realistically trade at 15-30 bps half-spread. At 20 bps (2x model): KR gap net Sharpe moves to approximately -2.3 (from `cost_sensitivity.csv`: 2x gives -4.26, but that also doubles impact; spread-only doubling is less severe but still deeply negative).

---

## 3. Market Impact Model Critique

The model uses: `0.1 × sqrt(trade_size / ADV) × daily_vol`.

For $100K into a $502K ADV name (0863.HK, Oct 2022): `0.1 × sqrt(0.199) × vol`. At vol = 3% daily, impact = 0.1 × 0.447 × 0.03 = 0.13% = 13 bps. This is not unreasonable as a starting point.

However, the calibration constant 0.1 comes from liquid US equities with continuous dealer markets. For HK/KR small caps with intermittent order flow and no dedicated market-makers in many GEM/KOSDAQ names, the realized impact constant is 2-3x higher — call it 0.25-0.30. At 0.25:

- 0863.HK impact estimate: 33 bps per trade (vs 13 bps modeled)
- This additional 20 bps × 2 sides × 42.75% daily turnover = 17 bps/day extra
- Annual: 4,300 bps = 43 percentage points of additional drag

The model also assumes $100K trade size is achievable. At $502K ADV, a $100K order is 20% of one day's volume. Any institutional desk would spread this over 3-5 days minimum or accept significant market impact. The 1-day hold period means there is no opportunity to spread the order.

---

## 4. Capacity

At face value: HK = 16 positions × $100K = $1.6M gross, KR = 20 positions × $100K = $2M gross. Already sub-institutional scale.

But the real constraint is worse. For the 3.6% of HK universe entries below $1M ADV (`universe_log.csv` analysis), a 5% ADV position limit (standard institutional constraint) implies:

- $500K ADV name: maximum position $25K, not $100K
- $800K ADV name: maximum position $40K

Scaling to realistic position sizes reduces gross P&L proportionally. If 4 of 8 short-leg names average $600K ADV, those positions must be cut by 60%, reducing gross by roughly 30% on the short leg. The gross Sharpe number survives only at the artificial $100K fixed size.

**Realistic capacity:** HK: $500K-1M gross. KR: $1M-2M gross. Below the threshold where most systematic desks would allocate infrastructure. Management fee economics on a $2M book don't work.

---

## 5. Live Failure Modes

**a. Execution timing.** The signal uses features available at equity close + overnight crypto data, then aims to trade at next equity open. In HK, pre-open is 09:00-09:30 HKT (consolidated auction). In KR, 08:30-09:00 KST. The strategy must submit orders at or before open to capture the gap return. This requires:

1. Overnight Binance data pulled, cleaned, and features computed by ~08:50 HKT / 08:20 KST
2. Model inference run
3. Orders submitted to prime broker pre-open

This is achievable technically but requires a live data pipeline with < 30-minute latency from exchange open to order submission. Any infra failure means missing the signal window entirely. The gap return (`R_gap = open/close - 1`) is the target — you must trade at open to capture it, not at open + 10 minutes.

**b. Data dependencies.** Requires: (1) live Binance minute data (feasible via API), (2) FRED daily macro (1-day lag, low criticality), (3) reliable HK/KR pre-open price reference. HK pre-open auction prices are available via HKEX data feeds; KR pre-open prices via KRX or Bloomberg. Both require paid subscriptions. The backtest used Stooq static files and pykrx, neither of which provides real-time data.

**c. Concentration risk.** All 8 long-leg names are in the "crypto-exposed" theme. Year-by-year net returns tell the story: 2021 HK net -57%, 2022 HK net -34%, 2022 KR net -10% (`computed from backtest_lgbm_{hk,kr}_gap.csv`). Crypto winter (2022: BTC -64%) produced persistent losses net of costs across both markets. The regime analysis (`regime_analysis.csv`) shows HK gap gross Sharpe 3.44 in low-VIX and 1.92 in high-VIX — the strategy underperforms precisely when crypto stress peaks, which is when long-leg co-movement peaks. The reported full-period Sharpe masks this regime dependency.

**d. Recent data recency bias.** 2025-2026 YTD shows dramatically better net performance: KR +22.6% (2025), KR +38.5% (2026 partial), HK +26.5% (2026 partial). This period coincides with HK crypto ETF listings (Apr 2024), Naver-Dunamu merger (Nov 2025), won-stablecoin speculation (ME2ON +289%, KakaoP +143% in Jun 2025). These are event-driven, non-recurring catalysts — not evidence of structural signal persistence.

---

## 6. What a Desk Needs to See Before Running This

1. **Real borrow rates from prime broker.** Get a Goldman or Morgan Stanley PB to quote borrow availability and rates for the bottom 10 names by ADV in each universe. If more than 2 names are unavailable to borrow, the portfolio construction breaks. This is the single most important ask — it cannot be backtested.

2. **Walk-forward with real bid/ask spreads.** Source historical Level 1 data (Bloomberg or Refinitiv tick data) for the actual universe names and reconstruct costs using realized bid/ask at open auction, not a constant bps estimate. The gap between 15 bps assumed and 30-70 bps realistic is the entire P&L margin.

3. **Regime gate.** Show net Sharpe conditional on BTC trailing 30-day return > 0. The regime analysis shows HK gap gross Sharpe 3.31 in BTC-up vs 2.72 in BTC-down — the signal is directionally conditional. A desk would want to see the strategy turned off in BTC bear regimes, even if this means sitting out 45% of days.

4. **Larger-cap KR subset only.** Run KR strategy restricted to KOSPI-listed names (Samsung, SK Hynix, Naver, Kakao, KB Financial, Shinhan) with $50M+ ADV. Accept lower gross Sharpe in exchange for realistic cost assumptions. Model 5 bps half-spread on these names. If KR gap Sharpe survives at net positive with those constraints, the sub-strategy has a path.

5. **Live paper-trade for 3 months.** Submit orders at open every day, record actual fill prices vs modeled open prices, compute realized vs modeled slippage. The gap between open price and actual fill in a thin-book pre-auction environment is not captured by any model.

---

## 7. Operational Reality — Does This Strategy Have a Path to Live?

**HK: No.** The HK version fails before reaching any other consideration. Breakeven at 9.9 bps half-spread; realistic minimum spread for the actual trading names is 30-70 bps. Borrow on GEM-adjacent names is either unavailable or prohibitively expensive. Annual cost drag at realistic spreads exceeds 100% of gross P&L. The HK version is a research finding, not a strategy.

**KR: Conditional, with major surgery required.** The raw numbers are closer (breakeven 0.93x, net Sharpe -0.28), but the margin is thinner than it looks. Adding realistic borrow (200-300 bps) pushes effective breakeven to ~0.80x — already below where the model sits. At realistic 20 bps half-spread (2x model), the strategy is net-negative.

The sub-strategy that might work: **KR KOSPI-only, large-cap filter, 50%+ ADV reduction in position size, regime-gated.** Restrict to the 8-10 largest KR names by ADV (Naver, Kakao, Samsung, SK Hynix, KB Financial, Shinhan). Model realistic spreads (5-10 bps). Accept that gross Sharpe will fall below 2. Size positions at 5% ADV max. If the signal survives — meaning net Sharpe > 0.5 with realistic costs — it could be pitched as a low-capacity satellite allocation to a crypto-aware EM desk. That is a big "if" that requires a fresh backtest on that sub-universe, not a re-read of these numbers.

---

## VERDICT — Research Finding or Deployable Strategy?

**Research finding.** Neither market version survives contact with real execution costs. HK is terminal at current spread assumptions; KR is marginally negative and worsens materially with realistic borrow and spreads. The gross signal (Sharpe 3-4) is real and worth publishing as a cross-asset information spillover finding. The net strategy is not deployable in its current form at any reasonable scale.

The one productive path: a KR KOSPI large-cap sub-strategy with reconstituted cost assumptions, a strict regime gate, and a live data infrastructure audit. That is a Pass 2 or a separate project, not a tweak to this backtest.
