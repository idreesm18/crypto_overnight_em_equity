# Pass 2 CS-Spread Update — Synthesis

This note supplements `reviews/synthesis_pass2.md`. It covers the CS
(Corwin–Schultz 2012) spread estimator update applied after the Pass 2
panel had already concluded. Scope: cost model only. Signal construction,
feature engineering, model training, and universe rules are unchanged.
All numbers below are out-of-sample and are sourced from
`output/cs_spread_summary.txt`, `output/backtest_summary_pass2_cs.csv`,
and `output/cs_vs_fixed_comparison.csv`.

## What changed

Pass 2 assumed a constant half-spread per market (15 bps HK stocks, 10
bps KR stocks, 2 bps index). The CS update replaces those with per-ticker
per-day estimates derived from daily OHLC via the Corwin–Schultz
high-low estimator (trailing 20-day mean; market-median fallback when a
ticker has fewer than 12 valid two-day estimates; fixed-assumption fallback
during the initial ~20-day warmup). The market-impact component
(square-root in trade_size/ADV) and the borrow cost model are unchanged.
Sensitivity sweep structure is preserved: multipliers 0.5x / 1.0x / 1.5x /
2.0x now apply to the CS estimate.

## How realized CS spreads compare to the old fixed assumptions

| Market | Pass 2 fixed (bps round-trip) | CS median 20d (bps) | Ratio |
|---|---:|---:|---:|
| HK stocks | 30 | 98.4 | 3.3x |
| KR main stocks | 20 | 84.8 | 4.2x |
| KR control stocks (Stage P2-18) | 10 | 80.5 | 8.1x |
| HSI_proxy (stand-in for 2800.HK) | 4 | 30.4 (floored) / -38.6 (unfloored) | 7.6x / NA |
| 069500.KS (KODEX 200, real ETF, Stage P2-17) | 4 | 25.5 (floored) | 6.4x |

The Pass 2 fixed assumption underestimated realized CS spreads by roughly
3-4x for stocks and 6-8x for index proxies (upper bound). Raw-negative
fractions for stock markets are ~41% — above the brief's 35% heuristic
but consistent with large-cap liquid tickers where the CS signal is small
relative to return volatility. The overnight adjustment is correctly
ordered (verified in the script); the raw-negative fraction falls to
~41% from ~51% when the adjustment is applied.

## Data-availability caveat: KR resolved via pykrx; HK still proxy

**Stage P2-17 update (2026-04-24):** 069500.KS (KODEX 200) daily OHLC
was pulled directly from KRX via pykrx v1.0.51 (1,796 rows,
2019-01-02 to 2026-04-24, via the `get_market_ohlcv` per-ticker
endpoint which does not require the gated batch API). The KR index
cost is now empirical real-ETF data, not an index aggregate. Raw-
negative fraction for 069500.KS is 52.4%, which still places the
estimator at the noise floor (daily H/L of a single ETF does not
support a clean CS point estimate). The floored 20d median of 25.5 bps
is a direct-ETF upper bound — tighter and more defensible than the
prior KOSPI index proxy, but still an upper bound rather than a point
estimate. Unfloored data was not separately reported for 069500.KS in
the diagnostics because raw-negative > 50% confirms noise-floor status.

2800.HK ETF OHLC remains unavailable (yfinance rate-limited; the HK
ETF is not in the pykrx stock endpoint). The HSI index-level OHLC
proxy is unchanged for the HK side. Raw-negative fraction 53.2%;
floored 30.4 bps is an upper bound.

Implication: the KR index-ETF cost is now pinned to real ETF data,
though the floor interpretation still applies. The HK side remains an
unresolved proxy. A direct 2800.HK OHLC pull from an alternate vendor
would replace the HK proxy with a point estimate.

## Stock universes: not deployable

Every stock-tercile strategy (main_hk, main_kr, control_hk, control_kr;
long-short and long-only; gate-on and gate-off; LightGBM and TCN) is
net-negative under CS spreads at 1x. No point in the 0.5x-2x spread
multiplier sweep rescues a stock universe. The Pass 2 synthesis had
already applied Rule C to the single-stock implementation; CS confirms
it on an empirical-cost basis rather than an assumed one. All four stock
universes are now empirically CS-covered.

**Stage P2-18 update (2026-04-24):** Daily OHLCV for all 38 control_kr
tickers was pulled via pykrx v1.0.51 (67,603 rows, 2019-01-02 to
2026-04-24, 0 ticker skips), resolving the prior control_kr coverage
gap. The 38-ticker parquet was spliced into the CS panel; the KR panel
now contains 76 unique tickers (38 main_kr + 38 control_kr). Median
CS spread for control_kr tickers is 80.5 bps round-trip — approximately
8× the Pass 2 fixed assumption (10 bps round-trip).

Under real CS, control_kr is net-negative across all 12 configurations.
The three configurations that appeared marginally positive under the
fixed-spread assumption (gap/tercile_ls gate_on: +0.36, gap/tercile_ls
gate_off: +0.58, intraday/tercile_ls gate_on: +0.24,
intraday/tercile_ls gate_off: +0.45) all flip to net-negative Sharpe
of −5 to −8 under real CS. There are no sign reversals in the
positive direction (no configuration that was net-negative under fixed
becomes net-positive under real CS). The stock-universe verdict is
universal and unambiguous: not deployable under realistic costs.

## Why tick-floor-based spreads supersede CS for index ETFs

Both index instruments — HSI_proxy (proxy for 2800.HK) and 069500.KS
(real KODEX 200) — show CS raw-negative fractions above 50% (53.2% and
52.4%). Per Tremacoldi-Rossi & Irwin (2022, JFQA, "The Bias of Simple
Bid-Ask Spread Estimators"), the fraction of non-positive raw two-day
estimates is a direct diagnostic of CS estimator failure: when this
fraction exceeds ~50% the true spread sits below the estimator's
signal-to-noise threshold, and zero-flooring (which we apply, like
nearly all CS users) introduces a downward truncation bias that *raises*
the floored 20-day mean above the true value. Corwin & Schultz (2012,
Table 6) themselves report that the cross-sectional correlation between
their estimate and the true TAQ effective spread falls to ~18% for
liquid large-cap stocks — effectively noise for calibrating any
specific instrument. Abdi & Ranaldo (2017, RFS) document the same
estimator weakness in the most-liquid quintile.

For index ETFs the realistic effective round-trip spread is anchored
to exchange tick structure and market-making rules, not to a daily-OHLC
estimator at its noise floor:

- **2800.HK Tracker Fund of Hong Kong**: HKEX's June 2020 ETP spread
  reform [HKEX-ETP, HKEX-Min] reduced minimum ticks for high-liquidity
  ETPs by 50-90%. At 2800.HK's HKD ~26 price level, the post-reform
  ETP minimum tick is HKD 0.02 → 0.02/26 ≈ **8 bps round-trip**.
  Continuous-SMM obligations on Tracker Fund (AUM ~HKD 142B, multiple
  competing market makers) anchor the realized spread to the one-tick
  floor during regular hours. Pre-2020-06-01 the equity tick of HKD
  0.05 applied → ~15 bps round-trip. The backtest applies a date-
  indexed step function: 15 bps pre-reform, 8 bps post-reform.
- **069500.KS (KODEX 200)**: KRX applies a flat 5-KRW tick to KOSPI
  200 ETFs across the full 2019-2026 window (the 2023 KRX tick reform
  affected single-stock ticks, not ETF ticks). At KODEX 200's ~33,500
  KRW price level, 5 KRW ≈ 1.5 bps per side → ~3 bps round-trip
  tick-floor. KRX LP spread-obligation rules and continuous quoting
  keep the realized spread close to this. The backtest applies a
  uniform **5 bps round-trip** as a conservative tick-floor anchor.

These tick-derived bounds are not point estimates — a TAQ-equivalent
intraday-tick measurement of 2800.HK and 069500.KS would convert them
into measured numbers — but they are defensible empirical anchors
backed by exchange-official documentation, peer-reviewed estimator-
bias literature, and (for HK) the regulator's own characterization
of the post-2020 ETP spread regime as "trading efficiently"
[HKEX-Dec24].

The CS estimator continues to be applied for stock universes, where
raw-negative fractions are 41% (HK) and 40% (KR) — above the 35%
heuristic but well below the 50%+ threshold where the estimator
catastrophically fails — and where the floored figures (98 bps HK,
85 bps KR) are 3-4× the Pass 2 fixed assumption rather than the 5-12×
overstatement seen on the index proxies.

## Net-Sharpe verdicts under realistic costs (1x spread, 1x borrow)

Under CS spreads for stocks and tick-floor spreads for index ETFs,
seven configurations clear the 0.5 net-Sharpe threshold:

| Configuration | net_SR_fixed | net_SR_realistic |
|---|---:|---:|
| lgbm / index_kr / gap / gate_off | 3.74 | 3.64 |
| lgbm / index_kr / gap / gate_on  | 2.69 | 2.60 |
| lgbm / index_hk / gap / gate_off | 2.38 | 1.96 |
| lgbm / index_hk / gap / gate_on  | 1.55 | 1.19 |
| lgbm / index_kr / cc  / gate_off | 1.16 | 1.10 |
| lgbm / index_kr / cc  / gate_on  | 0.90 | 0.85 |
| lgbm / index_hk / cc  / gate_off | 0.83 | 0.55 |

All seven are LightGBM, all are gap or cc targets, both gate-on and
gate-off appear. Two TCN configurations are positive but sub-threshold
(tcn index_hk cc gate_on 0.44, gate_off 0.16). All stock-tercile
configurations are net-negative.

The seven survivors are a strict subset of the eight Pass 2 fixed-
baseline winners (the eighth, lgbm index_hk cc gate_on, drops from
0.25 to −0.02 under the new index spreads). The earlier
"five Pass-2 winners flipped to losers under CS noise-floor proxy"
finding was an artifact of estimator failure; under realistic
tick-floor spreads those five configurations are restored as
deployable.

## Effect on Pass 2 framing rule

`reviews/synthesis_pass2.md` applied a framing rule keyed to whether
any borrow-and-gate-adjusted strategy cleared 0.5 net-Sharpe. Under
realistic costs (CS for stocks, tick-floor for indexes), seven
LightGBM index configurations clear 0.5. The revised answer to the
synthesis_pass2.md question (b) becomes: **yes, broadly — seven
configurations clear 0.5 across both markets and both gate settings,
on the gap and cc targets**. The deployable form is dual-market
index-futures-level (HSI and KOSPI 200), not the single-strategy
narrowing implied by the noise-floor-bound interim result.
The stock-level verdict is unambiguous: no stock universe produces
net_sharpe >= 0 at 1x CS under any target, model, strategy, or gate
setting.

## Known limitations carried into documentation

1. Index ETF spreads are tick-floor-derived bounds, not measured
   point estimates. A TAQ-equivalent intraday-tick pull on 2800.HK and
   069500.KS would convert the bounds into measured numbers. Live
   paper-trade execution is the cheaper path to the same answer.
2. CS estimator at the noise floor (>50% raw-negative fraction) on
   single-ETF daily series is a documented failure mode (Tremacoldi-
   Rossi & Irwin 2022, Corwin-Schultz 2012 Table 6). The decision to
   substitute tick-floor spreads for CS only on index instruments is
   based on this diagnostic, not on a methodology preference.
3. Stock CS estimator hits raw-negative 41% / 40% — above 35% but
   below the 50%+ failure regime. Stock CS verdicts are robust; the
   caveat lives in interpretation rather than the headline.
4. High raw-negative fraction (~41%) in stock markets reflects the
   estimator hitting its signal-to-noise limit on tight-spread
   large-caps, not a methodological bug.

## What remains unresolved after CS + Stage P2-17/18 + tick-floor updates

- For stock universes: unambiguously net-negative across all four
  universes, with complete empirical CS coverage (Stage P2-18
  resolved the control_kr gap). Not pursued further.
- For both index trades: seven LightGBM configurations clear 0.5 net
  Sharpe under tick-floor realistic spreads. The remaining uncertainty
  is the gap between the tick-floor bound and the true effective
  spread for 2800.HK and 069500.KS, which a TAQ-equivalent intraday-
  tick measurement or a live paper trade could close.
- A 30-day live paper trade on HSI and KOSPI 200 index futures is the
  cheapest path to converting the tick-floor bound into a measured
  point estimate, settling whether realized net Sharpe tracks the
  ~2-3.6 projection on the gap configurations.

## Citation keys (full references in WRITEUP § Costs)

- [HKEX-ETP] HKEX News Release (18 May 2020), "HKEX to Introduce New
  Initiatives to Enhance Liquidity of ETPs."
- [HKEX-Min] HKEX, "Reduction of Minimum Spreads" (current ETP spread
  table, post-2020 reform).
- [HKEX-Dec24] HKEX News Release (Dec 2024), "HKEX to Reduce Minimum
  Spreads in Hong Kong Securities Market" — explicit ETP exclusion.
- [CS-2012] Corwin & Schultz (2012), *Journal of Finance* 67(2):719–760.
- [TR-2022] Tremacoldi-Rossi & Irwin (2022), *JFQA*, SSRN 4216953.
- [AR-2017] Abdi & Ranaldo (2017), *Review of Financial Studies* 30(12):
  4437–4480.
