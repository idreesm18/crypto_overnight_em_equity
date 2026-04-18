# Practitioner Second Rebuttal -- Pass 2, Round 3

**Role:** Practitioner (Final Round)
**Date:** 2026-04-17

---

## Where the Debate Has Landed

Three rounds of review have moved the panel from positions to a convergent operational frame. The Skeptic conceded the KR structural differentiation (main gap:intraday ratio 2.15 vs. control 0.88) and accepted index futures as the correct vehicle. The Believer conceded that realistic net Sharpe is 1.5-2.0, not 3.74. Literature endorsed long-only EM index deployment and flagged McLean-Pontiff decay as a live monitoring requirement. All four panelists said paper trade before capital.

That is enough to make a trade decision.

---

## Final Positions: What Changes, What Holds

**Conceding to the Skeptic -- but narrowly.** The Skeptic's strongest remaining argument is the cost extrapolation: the model implies 0.24 bps per round-trip for index_kr gap gate_off; realistic KOSPI 200 front-month execution is 1.6-4 bps. At 7-14x model cost, net Sharpe falls to 1.0-2.0 (`cost_sensitivity_pass2.csv` caps at 2x and shows Sharpe 2.31, so the 7-14x range requires extrapolation). The Skeptic is correct that this extrapolation has not been validated. I do not dispute it. The paper trade IS the validation test -- not a placeholder for it.

The Skeptic's second concern -- 2023 attenuation plus missing year-by-year Sharpe breakdown -- is operationally valid. If the 3.74 full-period Sharpe is concentrated in 2020-2022 and 2024-2026 crypto-bull periods, the effective live Sharpe since 2024 may be materially lower. That is a required disclosure before any investor pitch. It does not block paper trading, but it blocks capital commitment beyond the paper trade gate.

**Holding against the Skeptic on two-ticker dismissal.** HSI and KOSPI are independently specified strategies with separate signal paths, separate control tests, and separate regime splits. KR IC floor is 0.237 across all six regime cells (`regime_analysis_pass2.csv`). That both clear is two corroborating votes, not one accident on two coins.

**Believer's net Sharpe concession is now my operating number.** I will not pitch 3.74. I will pitch 1.5-2.0 net Sharpe with a paper-trade-validated cost basis. The gross-to-net compression is acknowledged; the margin of safety above 1.0 is the argument.

**Literature's decay warning requires a pre-trade response.** The 2023 KR attenuation is candidate McLean-Pontiff decay. The kill switches below are designed to catch that before capital is impaired.

---

## Final Deployment Decision

### FUND -- Index Futures Only

**KOSPI 200 futures (KRX): $30M notional.** LGBM, index_kr, gap, gate_off configuration. Paper trade 30 trading days minimum, 60 preferred, before any real capital.

**HSI futures (HKEX): $20M notional.** HK control result (p=0.927) means the crypto channel identification is weak; size is half KR accordingly. Run simultaneously as a diversifier, not a primary conviction position.

**Combined starting allocation: $50M notional combined.** Cap at $200M combined if paper trade validates.

---

## Paper Trade Protocol

Duration: 30 trading days minimum; 60 preferred before ramp decision.

Metrics tracked daily:
- Realized P&L vs. modeled P&L (signal income component vs. total)
- Executed spread per round-trip logged against model's implied 0.24 bps
- 10th and 90th percentile slippage at the open
- Predicted IC vs. realized IC in each market

Ramp decision rules:
- Paper trade net Sharpe >= 50% of modeled 1.5-2.0 net Sharpe range: ramp to real capital over 2 weeks
- Paper trade net Sharpe < 30% of modeled range: do not deploy; investigate cost model and re-specify before any further step
- Realized spread consistently above 4 bps per round-trip: re-estimate net Sharpe at that cost level; ramp only if revised estimate clears 1.0

---

## Pre-Specified Kill Switches (Required Before First Trade)

These are non-negotiable conditions that must be documented and approved by the PM before Day 1.

1. Rolling 6-month Sharpe drops below 0 in two consecutive 3-month windows: cut both positions to zero, investigate, re-validate before resuming.
2. Monthly time-series IC (measured daily within each calendar month) drops below 0.03 in either KR or HK: halve the affected market position immediately.
3. Short-sale ban or material regulatory action on either venue (KRX, HKEX): immediate flatten of affected position. No exceptions -- the November 2023 KR short-sale ban shows this risk is real.
4. Daily loss exceeds 2x the expected 99th percentile VaR: halt trading for 5 business days, review execution logs and model output, resume only after sign-off.

These rules exist in writing before the first lot is traded or they do not trade at all.

---

## Do Not Fund

**Stock-universe long-short, HK or KR.** Every single-stock L/S config in `backtest_summary_pass2.csv` is negative net Sharpe. EM short structural issues, the 2023 KR short-sale ban contaminating the backtest, and borrow costs understated for KOSDAQ micro-caps (realistic 600-1500 bps vs. the 400 bps model default) combine to make this uninvestable. `borrow_sensitivity_pass2.csv` shows 1.5x borrow kills most stock configs. This is a wall, not a threshold.

**Stock-universe long-only, HK or KR.** Long-only gross Sharpe is positive (main_kr LGBM gap long-only 3.38, `long_short_decomposition.csv`). Net Sharpe is not. At 15 bps HK / 10 bps KR realistic single-name spread assumptions, long-only stock configs do not survive. The long-only gross case is real; the net case is not.

**TCN-based models.** LGBM beats TCN net of costs in every configuration (`horse_race_bootstrap.csv`: HK IC difference -0.021, p=0.006; KR IC difference -0.028, p=0.000). TCN OVERFIT_FLAG is unacceptable in a live model. No reason to add complexity or retain a model that memorizes training folds and fails OOS.

**Borrow-cost-sensitive configs of any kind.** `borrow_sensitivity_pass2.csv` confirms 1.5x borrow kills most stock configs. Any strategy where borrow sensitivity is a primary risk factor is off the table given EM securities lending market dynamics.

---

## Remaining Risks for PM Disclosure

- The 7-year OOS window covers one partial crypto cycle: 2020-2021 bull, 2022 bear, 2023-2026 partial recovery and ETF adoption. The strategy has not been tested through a sustained bear market. If index_kr Sharpe is concentrated in 2020-2022 and 2024-2026, the effective live edge may be significantly weaker than the full-period number implies. Year-by-year Sharpe breakdown for index_kr is required before any investor pitch.
- 2023 KR attenuation is a standing warning. If it repeats in 2026, cut the KR position in half immediately under Kill Switch 2 and investigate. Do not wait for the rolling 6-month trigger.
- Cost model needs live re-calibration. The paper trade IS the calibration test. Do not treat modeled net Sharpe as validated until the paper trade spread logs confirm the execution cost assumption.
- Data dependency: the model requires Binance minute bars, FRED VIXCLS, and yfinance KRX data on a daily cadence. Any data outage exceeding 1 calendar day means model is down and all positions go flat. Operational monitoring for data feed health is not optional.
- Tax treatment of KRX futures gains for the fund's domicile needs a legal opinion before pitching. Korean withholding tax on derivatives for treaty-country US funds is a non-trivial haircut on stated returns.

---

## Required Before Any Investor Pitch

- Year-by-year Sharpe breakdown for index_kr showing the 2023 drawdown profile explicitly
- Paper trade results with executed spread logs
- Tax opinion on KRX futures gains
- Pre-specified kill switch documentation signed by PM

---

Trade decision: FUND -- $30M KOSPI 200 futures + $20M HSI futures, paper trade only until ramp criteria met. NO-FUND for all single-stock configurations.
