# Practitioner Rebuttal — Pass 2
**Role:** Practitioner (Round 2)
**Date:** 2026-04-17

---

## Intro

My Round 1 position -- index futures yes, stock universe no -- is unchanged. The debate sharpened three issues: the Skeptic's Bonferroni argument, the Believer's cost extrapolation, and Literature's McLean-Pontiff decay warning.

---

## vs. Skeptic

Bonferroni is a publication-threshold tool, not a position-sizing tool. On index_kr gap gate_off, net Sharpe 3.74 with a 2x spread sensitivity of 2.31 (`cost_sensitivity_pass2.csv`): a 3x haircut for cost misspecification still leaves Sharpe above 1.0. I take that bet. I am not submitting to a journal; I am allocating capital with a kill switch.

On the "2-ticker accident" claim: HSI and KOSPI are two separately constructed strategies with independent signal paths, independent regime splits (KR IC floor 0.237 across all six regime cells in `regime_analysis_pass2.csv`), and different crypto-channel identification strengths. That both clear independently is two votes in favor, not one accident.

The Skeptic's strongest point -- which I concede -- is that 2x cost sensitivity does not validate survival at 10x. The model implies 0.24 bps round-trip for index_kr gap gate_off versus a realistic 1.6-4 bps for KOSPI 200 front-month futures. I flagged this in Round 1. I handle it with 30-day paper trading, not a statistical argument.

---

## vs. Believer

The Believer argues index futures survive at 10x cost and that regime splits eliminate fragility. I am more constructive than the Skeptic, but the Believer is moving faster than the data on two counts.

First, extrapolating linearly from 2x to 10x cost is not validated. At 10x you are at 12-16 bps half-spread -- a deteriorated futures market, not just normal bid-ask, and one where market impact compounds nonlinearly as size scales. "10x still clears 2.0 net Sharpe" is a model output, not a demonstrated result. The proper test is a paper trade with actual executed spreads logged against the model assumption.

Second, the KR short-sale ban (November 2023 to March 2024) is a strategy redesign issue, not an operational footnote. KOSPI 200 futures were not banned, so the index backtest is clean. But any PM doing due diligence will ask about that period. The Believer should have drawn this distinction sharply; leaving it as a minor note is a credibility problem in a pitch.

---

## vs. Literature

I agree with Literature on EM short structural impairment and long-only as the preferred form. The `long_short_decomposition.csv` confirms it: the short leg in main_kr LGBM gap adds 0.39 gross Sharpe and costs more in spread and borrow than it contributes. Index futures are the correct translation -- long the index, exit on signal reversal, no individual name shorts.

On McLean-Pontiff decay: the 2023 IC attenuation is candidate evidence. Literature warns about decay but does not supply a live risk-management response. My answer: rolling 6-month Sharpe kill switch -- if trailing realized Sharpe drops below 0 in two consecutive 3-month windows, go flat and re-validate. That rule belongs in the strategy spec before the first trade.

The Novy-Marx-Velikov survival-at-liquid-spreads point is genuine supporting evidence for index futures. I accept it as corroboration, not validation of the 10x cost extrapolation.

---

## What I Would Commit Desk Capital To After This Debate

**INDEX_KR GAP, GATE_OFF:** $50M KOSPI 200 futures notional, 30-day paper-trade validation first with actual spread logs against the model's 0.24 bps implied cost. If realized cost is below 4 bps per round-trip and P&L tracks within 20% of model expectation, ramp to $150M.

**INDEX_HK GAP, GATE_OFF:** $30M HSI futures notional, run simultaneously as a diversifier. HK control result (p=0.927) means crypto-channel identification is weak; size accordingly.

**NO STOCK UNIVERSE:** zero allocation to long-short or long-only individual names in HK or KR. Negative net Sharpe across every stock-universe config in `backtest_summary_pass2.csv`, plus EM short structural issues and 2023 KR short-ban contamination, make this a wall.

**Kill switch:** 6-month rolling Sharpe below 0 in two consecutive 3-month windows means flat positions and re-investigation. **Max allocation: $200M combined notional**, sized for 10% VaR at 2.5x baseline volatility. Tax opinion on KRX futures gains and year-by-year Sharpe breakdown for index_kr (to show the 2023 drawdown) are required before any investor pitch.

The Skeptic would not trade this. The Believer would trade today at full size. I am between them: paper trade first, ramp on evidence, kill switch pre-specified.
