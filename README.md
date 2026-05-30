# GARCH-Jump Option Market-Making System

A full end-to-end option pricing and market-making pipeline for A-share equities, built on a **GARCH(1,1) + Compound Poisson Jump** model.

## Overview

The system estimates a stochastic volatility model from historical stock prices, runs a dynamic delta-hedging backtest to quantify model bias and hedge uncertainty, and produces **arbitrage-free Bid / Mid / Ask implied volatility surfaces** calibrated to empirical hedging error statistics.

```
Stock prices  →  GARCH+Jump MLE  →  MC backtest  →  HE statistics  →  Bid/Ask surface
   (Stage 0)       (Stage 1)         (Stage 2)        (Stage 3)          (Stage 4)
                                                                              ↓
                                                              Cross-stock analysis (Stage 5)
```

## Key Features

- **GARCH(1,1) + Compound Poisson Jump** fitted by maximum likelihood (SLSQP, stationarity-constrained)
- **Adaptive Monte Carlo** with eSSVI arbitrage-free smile calibration (Gatheral-Jacquier conditions)
- **Three-step quoting engine (Stage 4)**:
  - *S4-10*: Asymmetric jump-parameter adjustment → structural Bid/Ask spread
  - *S4-10b*: Mean-HE bias correction → de-biased mid volatility
  - *S4-11*: σ\_HE uncertainty overlay with OTM floor → final spread width
  - *S4-12*: Butterfly no-arbitrage fix
- **Cross-stock analysis**: HE bias heatmaps, Spearman rank correlation with market cap
- **Market-cap stratified stock selection** (20 stocks sampled across the A-share universe (Shanghai & Shenzhen), selected by dividing all eligible stocks into 20 equal-count market-cap bins and drawing one stock from each bin — ensuring broad coverage across the full market-cap spectrum.)

## Project Structure

```
project_updates/
├── Vol/backtest/             # Core pipeline (all runnable modules)
│   ├── config.py             # ★ Single configuration file — edit before running
│   ├── run_all.py            # Master pipeline: Stages 1–4 for all tickers
│   ├── stage1_fit.py         # Stage 1: GARCH+Jump MLE
│   ├── stage2_backtest.py    # Stage 2: MC simulation + delta-hedging backtest
│   ├── stage3_stats.py       # Stage 3: Hedging error statistics
│   ├── stage4_quote.py       # Stage 4: Bid/Mid/Ask surface generation
│   ├── stage5_analysis.py    # Stage 5: Cross-stock analysis (run after all tickers)
│   ├── gen_ppt_charts.py     # Chart generation (vol surface + bid-ask smile)
│   ├── garch.py              # GARCH+Jump filter and MLE
│   ├── simulation.py         # Vectorised Monte Carlo path generation
│   ├── vol_surface.py        # eSSVI calibration and IV surface
│   ├── pricing.py            # Black-Scholes pricing, IV inversion, Greeks
│   ├── stats.py              # Hedging error aggregation (HE/S and HE/premium)
│   ├── trading.py            # Strategy definitions (short_call / short_put)
│   ├── backtest.py           # Daily delta-hedging loop
│   ├── analysis.py           # Stage 5 analysis functions
│   └── data.py               # Data loading utilities
│
├── Data/
│   ├── mkt_cap_bins.py       # Market-cap stratification → 20-bin portfolio
│   ├── data_yf.py            # Download daily prices via yfinance
│   └── Data_downloading.py   # Alternative downloader
│
├── Documents/
│   └── vol_pipeline_step_map.csv   # Full step-by-step pipeline reference
│
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download price data
Edit the `tickers` list in `Data/data_yf.py` (Shanghai: `600909.SS`, Shenzhen: `002390.SZ`), then:
```bash
cd project_updates
python3 Data/data_yf.py
```
Output: `Data/latest_data/{TICKER}.csv` (columns: `date`, `close`, adjusted)

### 3. Configure
Edit `Vol/backtest/config.py` — the **only file you need to change**:
```python
TICKERS            = ["600909", "002390", "600128"]  # tickers to run
TRAIN_START        = "2021-01-01"
TRAIN_END          = "2026-03-09"
BACKTEST_START     = "2021-07-01"
PRINT_VOL_SURFACE_DATE = "2026-03-09"   # date for the output quoting surface
```

### 4. Run the full pipeline (Stages 1–4)
```bash
python3 Vol/backtest/run_all.py
```
Results are saved to `Results/` (auto-created).

### 5. Cross-stock analysis (after all tickers complete)
```bash
python3 Vol/backtest/stage5_analysis.py
```

### 6. Generate charts
```bash
python3 Vol/backtest/gen_ppt_charts.py
```
Charts saved to `Results/Charts/`.

## Key Outputs

| File | Description |
|------|-------------|
| `{TICKER}_short_call_price_surface.csv` | Bid / Mid / Ask implied vol surface for each (tenor, strike) node |
| `{TICKER}_short_call_he_stats.csv` | Hedging error statistics per (tenor, moneyness) — wide format |
| `{TICKER}_short_call_he_detail.csv` | Same, long format — used by Stage 5 |
| `{TICKER}_short_call_he_premium_stats.csv` | HE normalised by option premium (near-ATM reference) |
| `{TICKER}_short_call_iv_surface.csv` | eSSVI model implied vol surface |
| `{TICKER}_short_call_params.csv` | Fitted GARCH+Jump parameters |
| `cross_he_tstat_heatmap.csv` | Cross-stock t-stat heatmap (Stage 5) |
| `Results/Charts/*.png` | Vol surface (3D) and Bid-Ask smile charts |

## Pipeline Reference

See [`Documents/vol_pipeline_step_map.csv`](Documents/vol_pipeline_step_map.csv) for a complete step-by-step description of every stage, including inputs, outputs, and key functions.

## Dependencies

- Python 3.10+
- `numpy`, `pandas`, `scipy`, `matplotlib`
- `yfinance`, `curl_cffi` (data download)
- `python-docx` (documentation generation only)

## Notes

- **Data not included**: stock price CSVs and market-cap snapshots are not tracked in this repository. Download them using the provided scripts.
- **Results not included**: all output files in `Results/` are generated and excluded from version control.
- All pipeline parameters are centralised in `Vol/backtest/config.py` — no other file needs to be edited for a standard run.
