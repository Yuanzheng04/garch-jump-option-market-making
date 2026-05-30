"""
stage1_fit.py
=============
Stage 1 — GARCH+Jump Model Fitting

Steps
-----
  Step 1 │ Load price data & compute log-returns
  Step 2 │ Initial variance filter  (h₀ = sample variance, GARCH recursion seed)
  Step 3 │ MLE estimation of GARCH+Jump parameters  (ω, α, β, λ, μ_J, σ_J)
  Step 4 │ Re-filter GARCH conditional variance with final parameters → h_series

Outputs  (Results/)
-------------------
  {TICKER}_model.json        ← machine-readable params  (read by stage2)
  {TICKER}_h_series.csv      ← daily conditional variance h_t  (read by stage2)
  {TICKER}_{STRAT}_params.csv← human-readable model summary

Run
---
    cd project_updates
    python3 Vol/backtest/stage1_fit.py

Note: stage1 output does NOT depend on the trading strategy (ACTIVE_STRATEGY).
The same model.json is shared across all strategies for the same ticker / period.
"""

from __future__ import annotations

import json
import os
import sys
import time

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
    DATA_DIR, TRAIN_START, TRAIN_END, BURN_IN_YEARS, BACKTEST_START, TRADING_DAYS,
)
from backtest.trading import STRATEGY_NAME
from backtest.data    import _load_price_df, _slice_df, _log_returns
from backtest.garch   import fit_garch_jump, _filter_garch_states

_SEP = "═" * 65


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _model_path(results_dir: str) -> str:
    return os.path.join(results_dir, f"{config.TICKER}_model.json")

def _h_series_path(results_dir: str) -> str:
    return os.path.join(results_dir, f"{config.TICKER}_h_series.csv")

def save_checkpoint(
        garch_params: dict,
        jump_params:  dict,
        fit_info:     dict,
        h_series:     pd.Series,
        ann_rv:       float,
        S0:           float,
        results_dir:  str,
) -> None:
    """Save model.json + h_series.csv so stage2 can be run independently."""
    os.makedirs(results_dir, exist_ok=True)

    # model.json  — all scalars (safe for JSON serialisation)
    model = {
        "ticker":      config.TICKER,
        "train_start": TRAIN_START,
        "train_end":   TRAIN_END,
        "ann_rv":      float(ann_rv),
        "S0":          float(S0),
        "garch": {k: float(v) for k, v in garch_params.items()},
        "jump":  {k: float(v) for k, v in jump_params.items()},
        "fit":   {k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                      int(v)  if isinstance(v, (int,  np.integer)) else
                      float(v) if isinstance(v, (float, np.floating)) else str(v))
                  for k, v in fit_info.items()},
    }
    with open(_model_path(results_dir), "w") as f:
        json.dump(model, f, indent=2)

    # h_series.csv
    h_df = h_series.to_frame("h_t")
    h_df.index.name = "date"
    h_df.to_csv(_h_series_path(results_dir))


def load_checkpoint(results_dir: str) -> tuple[dict, dict, dict, pd.Series, float, float]:
    """Load stage1 checkpoint from disk."""
    mp = _model_path(results_dir)
    hp = _h_series_path(results_dir)
    for p in (mp, hp):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found: {p}\n"
                "Run  python3 Vol/backtest/stage1_fit.py  first."
            )
    with open(mp) as f:
        model = json.load(f)

    garch_params = model["garch"]
    jump_params  = model["jump"]
    fit_info     = model["fit"]
    ann_rv       = float(model["ann_rv"])
    S0           = float(model["S0"])

    h_df     = pd.read_csv(hp, index_col=0, parse_dates=True)
    h_series = h_df["h_t"].rename("h_t")
    h_series.index = pd.DatetimeIndex(h_series.index)

    return garch_params, jump_params, fit_info, h_series, ann_rv, S0


# ── core logic ────────────────────────────────────────────────────────────────

def run_stage1(verbose: bool = True) -> tuple[
        dict, dict, dict, pd.Series, pd.DataFrame, float, float]:
    """
    Execute Stage 1 entirely in memory.

    Returns
    -------
    garch_params, jump_params, fit_info, h_series, price_df_full, ann_rv, S0
    """
    t0 = time.time()
    csv_path = os.path.join(_ROOT, DATA_DIR, f"{config.TICKER}.csv")

    # ── Step 1: Load price data ───────────────────────────────────────────────
    if verbose:
        print(f"\n{_SEP}")
        print("  STAGE 1 — GARCH+Jump Model Fitting")
        print(f"  Ticker : {config.TICKER}    Training : {TRAIN_START} → {TRAIN_END}")
        print(_SEP)
        print("\n── Step 1: Load price data ─────────────────────────────────────")

    price_df_raw = _load_price_df(csv_path)

    pre_rows      = price_df_raw[price_df_raw["date"] < pd.Timestamp(TRAIN_START)]
    extra_start   = (pre_rows.iloc[-1]["date"].strftime("%Y-%m-%d")
                     if not pre_rows.empty else TRAIN_START)
    price_df_full = _slice_df(price_df_raw, extra_start, TRAIN_END)

    train_df      = _slice_df(price_df_raw, TRAIN_START, TRAIN_END)
    prices_train  = train_df["close"].values.astype(float)
    r_train       = _log_returns(prices_train)
    S0            = float(train_df["close"].iloc[-1])
    ann_rv        = float(np.std(r_train) * np.sqrt(TRADING_DAYS) * 100)

    if verbose:
        print(f"  Training rows : {len(train_df):,}   (returns: {len(r_train):,})")
        print(f"  S₀ (last training close) : {S0:.4f}")
        print(f"  Realised vol (annualised) : {ann_rv:.2f}%")

    # ── Step 2: Initial variance filter  (inside fit_garch_jump) ─────────────
    if verbose:
        print("\n── Step 2: Initial GARCH Variance Filter ───────────────────────")
        var0 = float(np.var(r_train))
        h0   = np.sqrt(var0 * TRADING_DAYS) * 100
        print(f"  Sample variance of returns  : {var0:.2e}")
        print(f"  Implied initialisation vol  : {h0:.2f}% ann.")
        print("  Starting SLSQP optimisation (4 initial points) …")

    # ── Step 3: MLE parameter estimation ─────────────────────────────────────
    if verbose:
        print("\n── Step 3: MLE Estimation of GARCH+Jump Parameters ────────────")
    garch_params, jump_params, fit_info = fit_garch_jump(r_train, verbose=verbose)

    # ── Step 4: Re-filter GARCH conditional variance with final parameters ────
    if verbose:
        print("\n── Step 4: Re-filter GARCH Conditional Variance → h_series ────")
    h_series = _filter_garch_states(price_df_full, garch_params, jump_params)

    if verbose:
        h_ann = np.sqrt(h_series.values * TRADING_DAYS) * 100
        print(f"  h_series length : {len(h_series):,}  "
              f"(date range: {h_series.index[0].date()} → {h_series.index[-1].date()})")
        print(f"  h_t range (ann. vol): "
              f"{h_ann.min():.1f}% – {h_ann.max():.1f}%  "
              f"(mean: {h_ann.mean():.1f}%)")
        burn_in_end   = pd.Timestamp(BACKTEST_START) - pd.offsets.BDay(1)
        burn_in_mask  = h_series.index < pd.Timestamp(BACKTEST_START)
        n_burn        = int(burn_in_mask.sum())
        burn_label = (f"{int(BURN_IN_YEARS * 12)} months"
                      if BURN_IN_YEARS < 1 else f"{BURN_IN_YEARS} year(s)")
        print(f"\n  Burn-in : {burn_label}  "
              f"({h_series.index[0].date()} → {burn_in_end.date()},  {n_burn} trading days)")
        print(f"  Active  : BACKTEST_START = {BACKTEST_START}  "
              f"(h_t has had ≥ {n_burn} steps to converge from h̄)")
        print(f"\n  [Stage 1 total: {time.time()-t0:.1f}s]")

    return garch_params, jump_params, fit_info, h_series, price_df_full, ann_rv, S0


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    results_dir = os.path.join(_ROOT, "Results")
    _s = f"_{STRATEGY_NAME}"

    (garch_params, jump_params, fit_info,
     h_series, price_df_full, ann_rv, S0) = run_stage1(verbose=True)

    # ── Save checkpoints ──────────────────────────────────────────────────────
    print("\n── Saving Stage 1 Checkpoints ──────────────────────────────────")
    save_checkpoint(garch_params, jump_params, fit_info, h_series,
                    ann_rv, S0, results_dir)
    print(f"  [Saved] model params   → {_model_path(results_dir)}")
    print(f"  [Saved] h_series       → {_h_series_path(results_dir)}")

    # ── Human-readable params CSV ─────────────────────────────────────────────
    omega   = garch_params["omega"]
    alpha   = garch_params["alpha"]
    beta    = garch_params["beta"]
    lam     = jump_params["lam"]
    mu_j    = jump_params["mu_j"]
    sigma_j = jump_params["sigma_j"]
    phi     = alpha + beta
    var_j   = lam * (mu_j**2 + sigma_j**2)
    m_j2    = var_j + (lam * mu_j)**2
    h_bar   = (omega + alpha * m_j2) / max(1 - phi, 1e-12)
    lr_vol  = float(np.sqrt((h_bar + var_j) * TRADING_DAYS) * 100)

    params_df = pd.DataFrame([{
        "ticker":          config.TICKER,
        "train_start":     TRAIN_START,
        "train_end":       TRAIN_END,
        "omega":           omega,
        "alpha":           alpha,
        "beta":            beta,
        "persistence":     phi,
        "lam":             lam,
        "mu_j":            mu_j,
        "sigma_j":         sigma_j,
        "lr_vol_pct":      lr_vol,
        "rv_ann_pct":      ann_rv,
        "loglike":         fit_info["loglike"],
        "n_obs":           fit_info["n_obs"],
        "n_jumps_implied": fit_info["n_jumps_implied"],
    }])
    params_path = os.path.join(results_dir, f"{config.TICKER}{_s}_params.csv")
    params_df.to_csv(params_path, index=False)
    print(f"  [Saved] params summary → {params_path}")
    print(_SEP)
