# Crypto Overnight Signal for Asian Equities — Pass 1

## Overview

Crypto markets trade 24/7; Asian equity markets do not. Between the HKEX or KRX close and the next day's open, roughly 17 hours of crypto trading occur. Pass 1 tests whether overnight crypto activity carries information that is not yet priced into crypto-exposed Asian equities at the open.

Pass 1 scope: LightGBM only, hand-engineered features, monthly-rebalanced universe of 20-30 crypto-exposed stocks per market (HK and KR), three targets per stock-day (overnight gap, intraday, close-to-close). Pass 2 adds a TCN, index-level prediction, and expanded ablation.

## Data sources

| Source | Content | Pass 1 usage | Notes |
|---|---|---|---|
| Binance bulk archive | Spot and perp 1m klines for BTC/ETH/SOL/BNB/XRP; 8-hourly funding rates | Overnight crypto features | No auth. Liquidation snapshots unavailable post-2024; that feature dropped. |
| Stooq HK | Daily OHLCV for HKEX | HK equity returns and targets | User-provided. 2,869 .txt files placed at `data/stooq/hk_daily/`. |
| pykrx (v1.0.51) | KRX daily OHLCV | KR equity returns and targets | Access via SSH SOCKS5 tunnel to Oracle Cloud Seoul VM; KRX geo-blocks US IPs. Batch endpoint now login-gated; per-ticker endpoint used. Market cap endpoint authenticated-only — mcap feature dropped. |
| FRED | VIX, DXY, 2Y/10Y Treasury, 5Y breakeven, DFF | Macro features with 1-day lag | Free API key. |
| yfinance | BTC-USD daily | Universe correlation filter | Cross-checked against Binance spot. |
| Crypto candidates (user-provided) | 40 HK + 39 KR tickers with fundamental crypto exposure | Universe construction layer 1 | One non-numeric KR ticker (ME2ON.KQ) excluded. |

User-provided data is Stooq HK daily .txt files, the two candidate CSVs, and a FRED_API_KEY. The user set up and maintains an Oracle Cloud Seoul VM for the KRX tunnel.

## Methodology summary

1. Universe (Stage 2). Layer 1 is the user-provided candidate CSV. Layer 2 filters monthly on trailing 60-day BTC correlation and trailing 20-day ADV > $500K USD, selecting the top 20-30 per market. 88 rebalance months per market, 2019-01-02 through 2026-04-01.

2. Features (Stage 3). Overnight window is 08:00 UTC day T to 01:30 UTC day T+1 for HK, 06:30 UTC day T to 00:00 UTC day T+1 for KR. Final feature set: ~12 overnight crypto features, 6 lagged macro features, 3 stock-level features. The "log mcap bucket" and "BTC perp liquidation intensity" features specified in the brief were dropped due to data access constraints. The "USDT peg deviation" feature was also dropped: computing it within the overnight window requires minute-level BTC/USD data, which is unavailable for free.

3. Model (Stage 4). LightGBM walk-forward, expanding window, 252 trading-day minimum. Monthly rebalance. Hyperparameter search (10 iterations × 3-fold purged time-series CV) at each year boundary; reuse within year. Three models per market, one per target. 75 OOS folds per market × target.

4. Strategy (Stage 5). Sort predictions each day; long top tercile, short bottom tercile, equal weight within each leg, one-day hold. Go flat if fewer than 9 stocks in universe. Cost model: HK 15 bps/side, KR 10 bps/side, plus Kyle-style impact with $100K notional per position.

5. Diagnostics (Stage 6). Feature ablation at the category level, return-component decomposition, VIX and BTC-trend regime splits, SHAP stability across folds, weekend-effect comparison.

6. Review (Stage 7). Four-panelist adversarial review with two rounds, synthesis at the end.

## Key results

Walk-forward OOS, 2020-02 through 2026-04, 75 monthly folds per market × target.

| Market | Target | Mean IC | Bootstrap p | Gross SR | Net SR (1x cost) | Breakeven cost multiplier |
|---|---|---|---|---|---|---|
| HK | gap | 0.0610 | < 0.001 | 3.04 | -1.57 | 0.66x |
| HK | intraday | 0.0195 | < 0.001 | 0.35 | -2.99 | 0.10x |
| HK | cc | 0.0144 | 0.036 | 0.98 | -1.25 | 0.43x |
| KR | gap | 0.0608 | < 0.001 | 3.77 | -0.28 | 0.93x |
| KR | intraday | 0.0233 | 0.019 | 1.00 | -1.46 | 0.41x |
| KR | cc | 0.0076 | 0.255 | 0.59 | -1.58 | 0.27x |

Headline findings (see `reviews/synthesis.md` for panel review):

1. The gap vs intraday IC difference is +0.041 in HK and +0.038 in KR, both p < 0.001 via bootstrap. This passes the brief's acceptance criterion 2 in both markets and is the pre-specified decomposition result.
2. Feature ablation at the category level satisfies criterion 3. Nine of 18 combinations produce a gross Sharpe drop greater than 0.15; stock-level features dominate, with crypto-overnight features second.
3. Weekend effect: Monday IC (65-hour overnight window) exceeds Tue-Fri IC in both markets, consistent with the hypothesis that longer overnight windows carry more crypto information.
4. Net of realistic costs, no strategy is viable. Every strategy's breakeven cost multiplier is below 1x. The Practitioner review closed the HK path and left KR KOSPI large-caps as the only conditionally-viable subset for Pass 2 investigation.

## Reproduction instructions

Prerequisites:
- Python 3.11 on the host
- `~/.ssh/oracle_seoul` key and SSH config entry `oracle-seoul` pointing to a KR-exit VM (required for pykrx)
- A FRED API key

Steps:
```
cd /path/to/crypto_overnight_em_equity
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install "requests[socks]"

# Place Stooq HK .txt files under data/stooq/hk_daily/
# Place user-provided candidate CSVs at data/crypto_candidates_hk.csv and data/crypto_candidates_kr.csv
# Create .env with FRED_API_KEY=...

# Pipeline scripts (run in order, sequentially)
python scripts/stage0a_validate.py
python scripts/stage1_pull_fred.py
python scripts/stage1_pull_yfinance.py
python scripts/stage1_pull_binance.py
python scripts/stage1_pull_pykrx.py        # uses the KR tunnel internally
python scripts/stage2_universe.py
python scripts/stage3_features.py
python scripts/stage4_lightgbm_hk.py
python scripts/stage4_lightgbm_kr.py
python scripts/stage5_backtest.py
python scripts/stage6_diagnostics.py
```

Review papers and synthesis in `reviews/` are authored content, not generated by a script.

## Output map

- `output/backtest_summary.csv` — 6-row headline summary
- `output/cost_sensitivity.csv` — net Sharpe at 0.5x / 1x / 1.5x / 2x costs
- `output/feature_ablation.csv` — 18 ablation runs with delta Sharpe
- `output/return_decomposition.csv` — gap vs intraday vs cc per market
- `output/regime_analysis.csv` — VIX high/low, BTC trend up/down, crypto vol regime
- `output/shap_stability.csv` and `output/shap_fold_overlap.csv`
- `output/weekend_effect.csv`
- `output/predictions_lgbm_{hk,kr}_{gap,intraday,cc}.csv` — OOS predictions with fold IDs
- `reviews/{skeptic,believer,literature,practitioner}_{position,rebuttal}.md`
- `reviews/synthesis.md`

## Resume Framing

Built a walk-forward ML pipeline for cross-asset signal research: LightGBM with purged time-series CV and per-fold SHAP, applied to overnight crypto features predicting next-day Asian equity returns across HK and KR, with long/short tercile backtest, Kyle-style cost modeling, feature ablation, and a four-panelist adversarial review producing a synthesized go/no-go on deployability.

## Limitations

- The "log mcap bucket" stock-level feature was dropped because pykrx's market-cap endpoint is authenticated-only and Stooq HK files carry no mcap field. A Pass 2 with KRX credentials or a paid source would restore it.
- Binance no longer publishes liquidation snapshots in the bulk archive. The "BTC perp liquidation intensity" feature is absent from Pass 1 and flagged for a paid source (e.g., Coinglass) in Pass 2.
- The USDT peg deviation feature requires minute-level BTC/USD data, which is not available free. Dropped in Pass 1; flagged for Kaiko or CCData in Pass 2.
- The universe-selection filter uses 60-day BTC correlation. A non-BTC-filtered control universe has not been tested. This is the largest unresolved methodological question and the primary motivation for a Pass 2 control test.
- FX conversions use constant rates (7.8 HKD/USD, 1,300 KRW/USD). Sufficient for a bps-level ADV filter; insufficient for a live strategy.
- The KR pull was restricted to candidate tickers only (not the full KOSPI + KOSDAQ universe), because the batch-by-date KRX endpoint now requires login.

See `logs/feature_decisions.log` for the complete record of each feature-level deviation from the brief.
