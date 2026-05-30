"""
run_all.py
==========
Master pipeline — runs Stages 1–4 for each ticker in config.TICKERS.

Set TICKERS in config.py to one or more tickers (e.g. ["603986"] or
["603986", "000058"]). Each ticker must have a <TICKER>.csv under DATA_DIR.

Stage overview
--------------
  Stage 1 │ GARCH+Jump Model Fitting          → model.json  h_series.csv
  Stage 2 │ Monte Carlo + Dynamic Backtest   → trades.csv  iv_surface.csv
  Stage 3 │ Hedging Error Statistics         → he_stats.csv
  Stage 4 │ Bid / Mid / Ask Vol Surface      → vol_quotes.csv  win_rate_test.csv

Run
---
    cd project_updates
    python3 Vol/backtest/run_all.py

To run a single stage for the first ticker in TICKERS:
    python3 Vol/backtest/stage1_fit.py
    python3 Vol/backtest/stage2_backtest.py
    python3 Vol/backtest/stage3_stats.py
    python3 Vol/backtest/stage4_quote.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

# ── package path ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG)
sys.path.insert(0, _PKG)
sys.path.insert(0, _ROOT)

import backtest.config as config
from backtest.config import (
    DATA_DIR, TRAIN_START, TRAIN_END,
    BACKTEST_START, BACKTEST_END,
    ENTRY_FREQ_DAYS, TENORS_DAYS, Q_GRID,
    PRINT_VOL_SURFACE_DATE,
    N_BATCH, N_MAX,
    R_ANNUAL, Q_ANNUAL, SEED, TRADING_DAYS,
)
from backtest.trading import STRATEGY_NAME
from backtest.data    import _load_price_df

from backtest.stage1_fit      import run_stage1, save_checkpoint as _save1
from backtest.stage1_fit      import _model_path, _h_series_path
from backtest.stage2_backtest import run_stage2, save_checkpoint as _save2
from backtest.stage2_backtest import _trades_path, _daily_log_path, _iv_surface_path
from backtest.stage3_stats    import run_stage3, save_checkpoint as _save3
from backtest.stage3_stats    import load_checkpoint as _load3, _he_stats_path
from backtest.stage4_quote    import run_stage4, _nearest_trading_date, _price_surface_path

_SEP = "═" * 65


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    t_total = time.time()
    results_dir = os.path.join(_ROOT, "Results")
    os.makedirs(results_dir, exist_ok=True)

    tickers = list(config.TICKERS)
    if not tickers:
        print("[ERROR] config.TICKERS is empty. Set at least one ticker in config.py.")
        sys.exit(1)

    step_times: dict[str, float] = {
        "stage1_fit": 0.0, "stage2_backtest": 0.0,
        "stage3_stats": 0.0, "stage4_quote": 0.0,
    }

    print(_SEP)
    print(f"  Full Pipeline  —  Tickers: {tickers}  Strategy: {STRATEGY_NAME}")
    print(f"  Training  : {TRAIN_START} → {TRAIN_END}")
    print(f"  Backtest  : {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Tenors    : {TENORS_DAYS}")
    print(f"  Q-grid    : {len(Q_GRID)} quantile points  "
          f"({Q_GRID[0]:.0%} – {Q_GRID[-1]:.0%})")
    print(f"  Outputs   : {results_dir}")
    print(_SEP)

    for ticker in tickers:
        config.TICKER = ticker
        _s = f"_{STRATEGY_NAME}"

        print(f"\n{_SEP}")
        print(f"  ▶ Ticker: {ticker}")
        print(_SEP)

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 1: GARCH+Jump Fitting
        # ══════════════════════════════════════════════════════════════════════
        _t = time.time()
        (garch_params, jump_params, fit_info,
         h_series, price_df_full, ann_rv, S0) = run_stage1(verbose=True)
        _save1(garch_params, jump_params, fit_info, h_series, ann_rv, S0, results_dir)

        omega, alpha, beta = (garch_params[k] for k in ("omega", "alpha", "beta"))
        lam, mu_j, sigma_j = (jump_params[k] for k in ("lam", "mu_j", "sigma_j"))
        phi = alpha + beta
        var_j = lam * (mu_j**2 + sigma_j**2)
        m_j2 = var_j + (lam * mu_j)**2
        h_bar = (omega + alpha * m_j2) / max(1 - phi, 1e-12)
        lr_vol = float(np.sqrt((h_bar + var_j) * TRADING_DAYS) * 100)
        pd.DataFrame([{
            "ticker": config.TICKER, "train_start": TRAIN_START, "train_end": TRAIN_END,
            "omega": omega, "alpha": alpha, "beta": beta, "persistence": phi,
            "lam": lam, "mu_j": mu_j, "sigma_j": sigma_j,
            "lr_vol_pct": lr_vol, "rv_ann_pct": ann_rv,
            "loglike": fit_info["loglike"], "n_obs": fit_info["n_obs"],
            "n_jumps_implied": fit_info["n_jumps_implied"],
        }]).to_csv(os.path.join(results_dir, f"{config.TICKER}{_s}_params.csv"), index=False)

        step_times["stage1_fit"] += time.time() - _t
        print(f"\n  ✓ Stage 1 done  ({time.time() - _t:.1f}s)")

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 2: Monte Carlo + Backtest
        # ══════════════════════════════════════════════════════════════════════
        _t = time.time()
        trades, daily_log, iv_surface_df, k_surface_df = run_stage2(
            garch_params, jump_params, h_series, price_df_full, verbose=True,
        )
        _save2(trades, daily_log, iv_surface_df, k_surface_df, results_dir)
        step_times["stage2_backtest"] += time.time() - _t
        print(f"\n  ✓ Stage 2 done  ({time.time() - _t:.1f}s)")

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 3: HE Statistics
        # ══════════════════════════════════════════════════════════════════════
        _t = time.time()
        he_stats = run_stage3(trades, iv_surface_df, verbose=True,
                              results_dir=results_dir)
        _save3(he_stats, results_dir)
        he_stats = _load3(results_dir)  # MultiIndex (T_days, q) for stage4
        step_times["stage3_stats"] += time.time() - _t
        print(f"\n  ✓ Stage 3 done  ({time.time() - _t:.1f}s)")

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 4: Bid / Mid / Ask Price Surface
        # ══════════════════════════════════════════════════════════════════════
        _t = time.time()
        if iv_surface_df is not None and k_surface_df is not None:
            price_df       = _load_price_df(
                os.path.join(_ROOT, DATA_DIR, f"{config.TICKER}.csv"))
            vol_date       = _nearest_trading_date(price_df, PRINT_VOL_SURFACE_DATE)
            vol_ts         = pd.Timestamp(vol_date)
            price_df_dates = pd.to_datetime(price_df["date"]).dt.normalize()
            date_mask      = price_df_dates == vol_ts
            if date_mask.any():
                S_vol = float(price_df.loc[date_mask, "close"].iloc[0])
            else:
                sorted_px = price_df.assign(_d=price_df_dates).sort_values("_d")
                S_vol = float(sorted_px[sorted_px["_d"] <= vol_ts]["close"].iloc[-1])

            price_table = run_stage4(
                iv_surface_df = iv_surface_df,
                k_surface_df  = k_surface_df,
                S_vol         = S_vol,
                price_df      = price_df,
                h_series      = h_series,
                garch_params  = garch_params,
                jump_params   = jump_params,
                pricing_date  = vol_date,
                he_detail     = he_stats,
                verbose       = True,
            )
            price_table.to_csv(_price_surface_path(results_dir))
            step_times["stage4_quote"] += time.time() - _t
            print(f"\n  ✓ Stage 4 done  ({time.time() - _t:.1f}s)")
        else:
            print("\n  [Stage 4 skipped] Set PRINT_VOL_SURFACE_DATE in config.py to enable.")

    step_times["TOTAL"] = time.time() - t_total
    print(f"\n{_SEP}")
    print("  Timing Summary")
    print(_SEP)
    for stage, elapsed in step_times.items():
        pct = elapsed / step_times["TOTAL"] * 100
        print(f"  {stage:<28s} {elapsed:7.1f}s  ({pct:.1f}%)")
    print(_SEP)
    print("\n  All outputs saved to:", results_dir)
    print(_SEP)
