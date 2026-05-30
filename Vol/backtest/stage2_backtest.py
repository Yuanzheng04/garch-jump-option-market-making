"""
stage2_backtest.py
==================
Stage 2 — Monte Carlo Simulation & Dynamic Backtest

Steps
-----
  Step 5 │ Adaptive N-calibration  (determine N_block for ≤ 50 bps MC error)
  Step 6 │ Generate random-path block  (Z + Poisson-jump matrix, shape T×N)
  Step 7 │ (Optional) eSSVI calibration & vol-surface snapshot for one date
  Step 8 │ Dynamic delta-hedging backtest  (trade every ENTRY_FREQ_DAYS)

Reads  (Results/)
-----------------
  {TICKER}_model.json        ← GARCH+Jump params  (from stage1)
  {TICKER}_h_series.csv      ← daily h_t          (from stage1)

Outputs  (Results/)
-------------------
  {TICKER}_{STRAT}_trades.csv       ← raw trade records  (read by stage3)
  {TICKER}_{STRAT}_iv_surface.csv   ← IV surface on PRINT_VOL_SURFACE_DATE
                                       (if set in config.py)

Run
---
    cd project_updates
    python3 Vol/backtest/stage2_backtest.py
"""

from __future__ import annotations

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
    DATA_DIR, TRAIN_START, TRAIN_END,
    BACKTEST_START, BACKTEST_END,
    ENTRY_FREQ_DAYS, TENORS_DAYS, Q_GRID,
    PRINT_VOL_SURFACE_DATE,
    N_BATCH, N_MAX,
    R_ANNUAL, Q_ANNUAL, SEED, TRADING_DAYS,
)
from backtest.trading    import STRATEGY_NAME
from backtest.data       import _load_price_df, _slice_df
from backtest.simulation import _make_rand_block
from backtest.vol_surface import _build_ref_mc_surface, plot_vol_surface
from backtest.backtest   import run_backtest

from backtest.stage1_fit import load_checkpoint as _load_stage1

_SEP = "═" * 65


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _trades_path(results_dir: str) -> str:
    return os.path.join(results_dir, f"{config.TICKER}_{STRATEGY_NAME}_trades.csv")

def _daily_log_path(results_dir: str) -> str:
    return os.path.join(results_dir, f"{config.TICKER}_{STRATEGY_NAME}_trade_daily_log.csv")

def _iv_surface_path(results_dir: str) -> str:
    return os.path.join(results_dir, f"{config.TICKER}_{STRATEGY_NAME}_iv_surface.csv")

def save_checkpoint(
        trades:        pd.DataFrame,
        daily_log:     pd.DataFrame,
        iv_surface_df: pd.DataFrame | None,
        k_surface_df:  pd.DataFrame | None,
        results_dir:   str,
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    trades.to_csv(_trades_path(results_dir), index=False)
    if not daily_log.empty:
        daily_log.to_csv(_daily_log_path(results_dir), index=False)
    if iv_surface_df is not None and k_surface_df is not None:
        _save_iv_csv(iv_surface_df, k_surface_df, results_dir)


def load_checkpoint(results_dir: str) -> tuple[pd.DataFrame,
                                               pd.DataFrame | None,
                                               pd.DataFrame | None]:
    tp = _trades_path(results_dir)
    if not os.path.exists(tp):
        raise FileNotFoundError(
            f"Stage 2 checkpoint not found: {tp}\n"
            "Run  python3 Vol/backtest/stage2_backtest.py  first."
        )
    trades = pd.read_csv(tp, parse_dates=["entry_date", "expiry_date"])
    trades["q"]      = trades["q"].astype(float)
    trades["T_days"] = trades["T_days"].astype(int)

    iv_surface_df, k_surface_df = None, None
    ip = _iv_surface_path(results_dir)
    if os.path.exists(ip):
        raw = pd.read_csv(ip, index_col=[0, 1])
        raw.index.names = ["tenor_days", "metric"]
        k_surface_df  = raw.xs("k",   level="metric").astype(float)
        iv_surface_df = raw.xs("IV%", level="metric").astype(float) / 100.0
        k_surface_df.index  = k_surface_df.index.astype(int)
        iv_surface_df.index = iv_surface_df.index.astype(int)

    return trades, iv_surface_df, k_surface_df


def _save_iv_csv(
        iv_surface_df: pd.DataFrame,
        k_surface_df:  pd.DataFrame,
        results_dir:   str,
) -> None:
    surf_labels  = list(iv_surface_df.columns)
    iv_rows_list = []
    for T_days in TENORS_DAYS:
        if T_days not in iv_surface_df.index:
            continue
        avail  = [lbl for lbl in surf_labels if lbl in k_surface_df.columns]
        k_vals = k_surface_df.loc[T_days, avail].values
        iv_val = (iv_surface_df.loc[T_days, avail].values * 100).round(2)
        iv_rows_list.append(
            pd.Series(k_vals.round(4), index=avail, name=(T_days, "k")))
        iv_rows_list.append(
            pd.Series(iv_val,          index=avail, name=(T_days, "IV%")))
    if iv_rows_list:
        iv_csv = pd.DataFrame(iv_rows_list)
        iv_csv.index = pd.MultiIndex.from_tuples(
            iv_csv.index, names=["tenor_days", "metric"])
        iv_csv.to_csv(_iv_surface_path(results_dir))


# ── core logic ────────────────────────────────────────────────────────────────

def run_stage2(
        garch_params:  dict,
        jump_params:   dict,
        h_series:      pd.Series,
        price_df_full: pd.DataFrame,
        verbose:       bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Execute Stage 2 entirely in memory.

    Returns
    -------
    trades, daily_log, iv_surface_df, k_surface_df
    """
    t0 = time.time()

    if verbose:
        print(f"\n{_SEP}")
        print("  STAGE 2 — Monte Carlo Simulation & Dynamic Backtest")
        print(f"  Strategy   : {STRATEGY_NAME}")
        print(f"  Backtest   : {BACKTEST_START} → {BACKTEST_END}")
        print(f"  Tenors     : {TENORS_DAYS}")
        print(_SEP)

    # ── Step 5: Adaptive N-calibration ───────────────────────────────────────
    if verbose:
        print("\n── Step 5: Adaptive N-Calibration (MC path count) ─────────────")

    bt_start_ts  = pd.Timestamp(BACKTEST_START)
    pre_bt_rows  = price_df_full[price_df_full["date"] < bt_start_ts]
    bt_day0_row  = price_df_full[price_df_full["date"] == bt_start_ts]
    S_day0       = float(bt_day0_row["close"].iloc[0]) if not bt_day0_row.empty else \
                   float(price_df_full["close"].iloc[-1])

    assert pd.Timestamp(BACKTEST_START) >= pd.Timestamp(TRAIN_START), \
        "BACKTEST_START must be ≥ TRAIN_START"
    assert pd.Timestamp(BACKTEST_END) <= pd.Timestamp(TRAIN_END), \
        "BACKTEST_END must be ≤ TRAIN_END  (in-sample backtest)"

    if not pre_bt_rows.empty and bt_start_ts in h_series.index:
        S_prev   = float(pre_bt_rows["close"].iloc[-1])
        r0       = np.log(S_day0 / S_prev)
        h_pre    = float(h_series[bt_start_ts])
        h0_post  = float(garch_params["omega"]
                         + garch_params["alpha"] * r0**2
                         + garch_params["beta"]  * h_pre)
    else:
        h0_post = garch_params["last_var"]

    if verbose:
        print(f"  S₀ on backtest day-0     : {S_day0:.4f}")
        print(f"  h₀ post-update (ann vol) : {np.sqrt(h0_post * TRADING_DAYS)*100:.2f}%")
        print(f"  Convergence tol : 50 bps   epoch : {N_BATCH:,}   n_max : {N_MAX:,}")

    _, _, n_required = _build_ref_mc_surface(
        S_day0, garch_params, jump_params, TENORS_DAYS, Q_GRID,
        h0=h0_post,
        epoch_size=N_BATCH, n_max=N_MAX, tol_bps=50.0,
        r=R_ANNUAL, q=Q_ANNUAL, seed=SEED, verbose=verbose,
    )
    N_block = ((n_required + N_BATCH - 1) // N_BATCH) * N_BATCH

    bt_dates = price_df_full[
        (price_df_full["date"] >= bt_start_ts) &
        (price_df_full["date"] <= pd.Timestamp(BACKTEST_END))
    ]
    n_bt    = len(bt_dates)
    T_max   = max(int(t) for t in TENORS_DAYS)
    T_total = n_bt + T_max
    mem_mb  = T_total * N_block * 2 * 8 / 1e6

    if verbose:
        print(f"\n  N_required={n_required:,}  →  N_block={N_block:,}")
        print(f"  Backtest trading days: {n_bt}   T_max: {T_max}   T_total: {T_total}")
        print(f"  Memory ≈ {mem_mb:.0f} MB  (Z + J matrices)")

    # ── Step 6: Generate random-path block ────────────────────────────────────
    if verbose:
        print("\n── Step 6: Generate Random-Path Block ──────────────────────────")
        print(f"  Generating {T_total:,} × {N_block:,} random matrices (seed={SEED}) …")

    t_sim = time.time()
    rand_block = _make_rand_block(
        T_total, N_block,
        jump_params["lam"], jump_params["mu_j"], jump_params["sigma_j"],
        seed=SEED,
    )
    if verbose:
        print(f"  [Step 6 time: {time.time()-t_sim:.1f}s]")

    # ── Step 7: (Optional) eSSVI vol surface for reference date ───────────────
    iv_surface_df: pd.DataFrame | None = None
    k_surface_df:  pd.DataFrame | None = None

    if PRINT_VOL_SURFACE_DATE is not None:
        if verbose:
            print(f"\n── Step 7: eSSVI Vol-Surface Snapshot  ({PRINT_VOL_SURFACE_DATE}) ─")
            print("  1) Simulate {N_block:,} paths from h_t on that date")
            print("  2) Compute quantile strikes → MC call prices")
            print("  3) Invert BS prices → implied vols")
            print("  4) Fit eSSVI smile  (θ, ρ, ψ per tenor)")
        t_vol = time.time()
        _charts_dir = os.path.join(_ROOT, "Results", "Charts")
        os.makedirs(_charts_dir, exist_ok=True)
        iv_surface_df, k_surface_df = plot_vol_surface(
            PRINT_VOL_SURFACE_DATE, price_df_full, h_series,
            garch_params, jump_params,
            TENORS_DAYS, Q_GRID,
            ticker=f"{config.TICKER}_{STRATEGY_NAME}",
            out_dir=_charts_dir,
        )
        if verbose:
            print(f"  [Step 7 time: {time.time()-t_vol:.1f}s]")
    else:
        if verbose:
            print("\n  [Step 7 skipped] Set PRINT_VOL_SURFACE_DATE in config.py to enable.")

    # ── Step 8: Dynamic delta-hedging backtest ────────────────────────────────
    if verbose:
        print(f"\n── Step 8: Dynamic Delta-Hedging Backtest ──────────────────────")
        print(f"  Strategy      : {STRATEGY_NAME}")
        print(f"  Entry every   : {ENTRY_FREQ_DAYS} trading days")
        print(f"  Tenors        : {TENORS_DAYS}")

    t_bt = time.time()
    trades, daily_log = run_backtest(
        price_df=price_df_full,
        h_series=h_series,
        garch_params=garch_params,
        jump_params=jump_params,
        rand_block=rand_block,
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
        tenors_days=TENORS_DAYS,
        q_grid=Q_GRID,
        entry_freq=ENTRY_FREQ_DAYS,
        r=R_ANNUAL,
        q=Q_ANNUAL,
        verbose=verbose,
    )

    if verbose:
        print(f"  [Step 8 time: {time.time()-t_bt:.1f}s]")
        print(f"\n  [Stage 2 total: {time.time()-t0:.1f}s]")

    return trades, daily_log, iv_surface_df, k_surface_df


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    results_dir = os.path.join(_ROOT, "Results")

    # Load stage1 checkpoint
    print(f"\n{_SEP}")
    print("  Loading Stage 1 checkpoint …")
    (garch_params, jump_params, fit_info,
     h_series, ann_rv, S0) = _load_stage1(results_dir)
    print(f"  Model loaded   : GARCH ω={garch_params['omega']:.2e}  "
          f"α={garch_params['alpha']:.4f}  β={garch_params['beta']:.4f}")
    print(f"  Jump           : λ={jump_params['lam']:.4f}  "
          f"μ_J={jump_params['mu_j']:.4f}  σ_J={jump_params['sigma_j']:.4f}")

    # Also need price_df_full — reload from raw data
    csv_path      = os.path.join(_ROOT, DATA_DIR, f"{config.TICKER}.csv")
    price_df_raw  = _load_price_df(csv_path)
    pre_rows      = price_df_raw[price_df_raw["date"] < pd.Timestamp(TRAIN_START)]
    extra_start   = (pre_rows.iloc[-1]["date"].strftime("%Y-%m-%d")
                     if not pre_rows.empty else TRAIN_START)
    price_df_full = _slice_df(price_df_raw, extra_start, TRAIN_END)

    # Run stage 2
    trades, daily_log, iv_surface_df, k_surface_df = run_stage2(
        garch_params, jump_params, h_series, price_df_full, verbose=True,
    )

    # Save checkpoints
    print("\n── Saving Stage 2 Checkpoints ──────────────────────────────────")
    save_checkpoint(trades, daily_log, iv_surface_df, k_surface_df, results_dir)
    print(f"  [Saved] trades       → {_trades_path(results_dir)}")
    if not daily_log.empty:
        print(f"  [Saved] daily_log    → {_daily_log_path(results_dir)}")
    if iv_surface_df is not None:
        print(f"  [Saved] iv_surface   → {_iv_surface_path(results_dir)}")
    print(_SEP)
