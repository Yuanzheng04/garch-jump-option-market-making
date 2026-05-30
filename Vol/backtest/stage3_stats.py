"""
stage3_stats.py
===============
Stage 3 — Hedging Error Statistics

Purpose
-------
Aggregate the raw per-trade hedging-error data produced by Stage 2 into
per-(tenor, quantile) summary statistics, and attach the model (fair) IV
from the fitted vol surface so that Stages 4/5 have everything they need
in a single file.

Steps
-----
  Step 9  │ Aggregate HE per (T_days, q) bucket via compute_he_stats:
           │   n, k_mean, premium_mean, HE_mean, HE_std
           │   HE_per_S_mean, HE_per_S_std   ← primary scale-free metric
           │
  Step 10 │ Build flat he_stats table in vol_quotes_integrated format:
           │   cols  = tenor_days | metric | q05 … ATM … q95
           │   rows per tenor:
           │     model_vol%       — fair IV from iv_surface (%)
           │     HE_per_S_mean_%  — mean  of HE/S (%)
           │     HE_per_S_std_%   — std   of HE/S (%)

Reads  (Results/)
-----------------
  {TICKER}_{STRAT}_trades.csv       ← raw trade records  (from Stage 2)
  {TICKER}_{STRAT}_iv_surface.csv   ← model IV and column ordering  (optional)

Outputs  (Results/)
-------------------
  {TICKER}_{STRAT}_he_stats.csv
      Wide format (tenor_days | metric | q05 … ATM … q95).
      Rows: model_vol%, HE_per_S_mean_%, HE_per_S_std_% for each tenor.

  {TICKER}_{STRAT}_he_detail.csv
      Long format indexed by (T_days, q).
      Columns: HE_per_S_mean, HE_per_S_std  (decimal fractions, not %).
      Used by Stage 5 cross-section analysis; avoids re-parsing he_stats.csv.

Run
---
    cd project_updates
    python3 Vol/backtest/stage3_stats.py
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
from backtest.config  import TENORS_DAYS, Q_GRID
from backtest.trading import STRATEGY_NAME
from backtest.stats   import compute_he_stats

from backtest.stage2_backtest import load_checkpoint as _load_stage2

_SEP = "═" * 65


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _he_stats_path(results_dir: str, ticker: str | None = None) -> str:
    t = ticker if ticker is not None else config.TICKER
    return os.path.join(results_dir, f"{t}_{STRATEGY_NAME}_he_stats.csv")


def _he_detail_path(results_dir: str, ticker: str | None = None) -> str:
    t = ticker if ticker is not None else config.TICKER
    return os.path.join(results_dir, f"{t}_{STRATEGY_NAME}_he_detail.csv")


def _he_premium_stats_path(results_dir: str, ticker: str | None = None) -> str:
    t = ticker if ticker is not None else config.TICKER
    return os.path.join(results_dir, f"{t}_{STRATEGY_NAME}_he_premium_stats.csv")


def _parse_he_stats_csv(path: str) -> pd.DataFrame:
    """
    Parse a wide-format he_stats CSV (tenor_days | metric | q05 … ATM … q95)
    and return a long-format DataFrame indexed by (T_days, q).

    Columns always present:
        HE_per_S_mean  — mean HE/S (decimal)
        HE_per_S_std   — std  HE/S (decimal)

    Columns added when available (newer output):
        HE_per_S_tstat — one-sample t-stat for H0: E[HE/S] = 0
        HE_per_S_pval  — two-sided p-value (< 0.05 = bias is significant)
    """
    flat = pd.read_csv(path)
    mon_cols = [c for c in flat.columns if c not in ("tenor_days", "metric")]

    def _col_to_q(col: str) -> float:
        return -1.0 if col == "ATM" else int(col[1:]) / 100.0

    records = []
    for T_int in flat["tenor_days"].unique():
        block   = flat[flat["tenor_days"] == T_int]
        mean_r  = block[block["metric"] == "HE_per_S_mean_%"]
        std_r   = block[block["metric"] == "HE_per_S_std_%"]
        tstat_r = block[block["metric"] == "HE_per_S_tstat"]
        pval_r  = block[block["metric"] == "HE_per_S_pval"]
        if mean_r.empty or std_r.empty:
            continue
        mean_row  = mean_r.iloc[0]
        std_row   = std_r.iloc[0]
        tstat_row = tstat_r.iloc[0] if not tstat_r.empty else None
        pval_row  = pval_r.iloc[0]  if not pval_r.empty  else None

        for col in mon_cols:
            q = _col_to_q(col)
            try:
                mean_val = float(mean_row[col]) / 100.0  # % → decimal
                std_val  = float(std_row[col])  / 100.0
            except (KeyError, ValueError):
                continue
            if not (np.isfinite(mean_val) and np.isfinite(std_val)):
                continue
            rec: dict = {
                "T_days":       int(T_int),
                "q":            q,
                "HE_per_S_mean": mean_val,
                "HE_per_S_std":  std_val,
            }
            # p-value and t-stat are stored as-is (not in %)
            if tstat_row is not None:
                try:
                    rec["HE_per_S_tstat"] = float(tstat_row[col])
                except (KeyError, ValueError):
                    rec["HE_per_S_tstat"] = float("nan")
            if pval_row is not None:
                try:
                    rec["HE_per_S_pval"] = float(pval_row[col])
                except (KeyError, ValueError):
                    rec["HE_per_S_pval"] = float("nan")
            records.append(rec)

    base_cols = ["HE_per_S_mean", "HE_per_S_std"]
    if not records:
        return pd.DataFrame(columns=base_cols)
    df = pd.DataFrame(records).set_index(["T_days", "q"])
    df.index = df.index.set_levels(
        [df.index.levels[0].astype(int),
         df.index.levels[1].astype(float)],
    )
    return df


def save_checkpoint(he_stats: pd.DataFrame, results_dir: str) -> None:
    """Save wide-format he_stats CSV and long-format he_detail CSV."""
    os.makedirs(results_dir, exist_ok=True)
    he_stats.to_csv(_he_stats_path(results_dir), index=False)
    detail = _parse_he_stats_csv(_he_stats_path(results_dir))
    detail.to_csv(_he_detail_path(results_dir))


def load_checkpoint(results_dir: str) -> pd.DataFrame:
    """
    Load he_stats for config.TICKER.
    Returns (T_days, q) MultiIndex DataFrame with columns
    HE_per_S_mean and HE_per_S_std (decimal fractions).
    """
    return load_checkpoint_for_ticker(results_dir, ticker=config.TICKER)


def load_checkpoint_for_ticker(results_dir: str, ticker: str) -> pd.DataFrame:
    """
    Load he_stats for any ticker.
    Returns (T_days, q) MultiIndex DataFrame with columns
    HE_per_S_mean and HE_per_S_std (decimal fractions).

    Priority:
      1. {ticker}_{strat}_he_detail.csv  (long format, direct load)
      2. {ticker}_{strat}_he_stats.csv   (wide format, parsed on the fly)
    """
    detail_path = _he_detail_path(results_dir, ticker)
    if os.path.exists(detail_path):
        df = pd.read_csv(detail_path, index_col=["T_days", "q"])
        df.index = df.index.set_levels(
            [df.index.levels[0].astype(int),
             df.index.levels[1].astype(float)],
        )
        return df

    stats_path = _he_stats_path(results_dir, ticker)
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Stage 3 checkpoint not found for ticker {ticker}:\n"
            f"  {stats_path}\n"
            "Run  python3 Vol/backtest/stage3_stats.py  first."
        )
    return _parse_he_stats_csv(stats_path)


# ── table builder ─────────────────────────────────────────────────────────────

def _build_he_stats(
        he_agg:        pd.DataFrame,
        iv_surface_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build flat he_stats in vol_quotes_integrated format:
        cols:  tenor_days | metric | q05 … ATM … q95
        rows per tenor:
            model_vol%       — annualised fair IV from iv_surface_df (%)
            HE_per_S_mean_%  — mean  of HE/S (%)
            HE_per_S_std_%   — std   of HE/S (%)
            HE_per_S_tstat   — one-sample t-stat for H0: E[HE/S] = 0
            HE_per_S_pval    — two-sided p-value (< 0.05 → bias is significant)
    ATM is inserted at its sorted-k position among the quantile columns.
    """
    q_labels = [f"q{int(q*100):02d}" for q in Q_GRID]

    if iv_surface_df is not None:
        mon_cols = list(iv_surface_df.columns)
    else:
        mon_cols = q_labels + ["ATM"]
        for T_days in TENORS_DAYS:
            if T_days not in he_agg.index.get_level_values("T_days"):
                continue
            sub = he_agg.xs(T_days, level="T_days")
            if -1.0 not in sub.index:
                continue
            k_atm = float(sub.loc[-1.0, "k_mean"])
            q_k   = [float(sub.loc[qv, "k_mean"]) if qv in sub.index else float("nan")
                     for qv in Q_GRID]
            ins   = sum(1 for k in q_k if not np.isnan(k) and k < k_atm)
            mon_cols = q_labels[:ins] + ["ATM"] + q_labels[ins:]
            break

    q_vals = [(-1.0 if lbl == "ATM" else int(lbl[1:]) / 100.0) for lbl in mon_cols]

    # check whether tstat/pval columns are available (compute_he_stats may be newer)
    has_tstat = "HE_per_S_tstat" in he_agg.columns
    has_pval  = "HE_per_S_pval"  in he_agg.columns

    rows = []
    for T_days in TENORS_DAYS:
        T_int = int(T_days)
        if T_int not in he_agg.index.get_level_values("T_days"):
            continue
        sub = he_agg.xs(T_int, level="T_days")

        if iv_surface_df is not None and T_int in iv_surface_df.index:
            model_vals = [round(float(iv_surface_df.loc[T_int, col]) * 100, 2)
                          if col in iv_surface_df.columns else float("nan")
                          for col in mon_cols]
        else:
            model_vals = [float("nan")] * len(mon_cols)

        mean_vals = [round(float(sub.loc[qv, "HE_per_S_mean"]) * 100.0, 4)
                     if qv in sub.index else float("nan")
                     for qv in q_vals]
        std_vals  = [round(float(sub.loc[qv, "HE_per_S_std"]) * 100.0, 4)
                     if qv in sub.index else float("nan")
                     for qv in q_vals]

        rows.append([T_int, "model_vol%"]      + model_vals)
        rows.append([T_int, "HE_per_S_mean_%"] + mean_vals)
        rows.append([T_int, "HE_per_S_std_%"]  + std_vals)

        if has_tstat:
            tstat_vals = [round(float(sub.loc[qv, "HE_per_S_tstat"]), 4)
                          if qv in sub.index else float("nan")
                          for qv in q_vals]
            rows.append([T_int, "HE_per_S_tstat"] + tstat_vals)

        if has_pval:
            pval_vals = [round(float(sub.loc[qv, "HE_per_S_pval"]), 4)
                         if qv in sub.index else float("nan")
                         for qv in q_vals]
            rows.append([T_int, "HE_per_S_pval"] + pval_vals)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["tenor_days", "metric"] + mon_cols)


def _build_he_premium_stats(
        he_agg:        pd.DataFrame,
        iv_surface_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build wide-format HE/premium stats table.
    Same column structure as he_stats.csv but normalised by option premium at entry.

    Rows per tenor:
        HE_per_prem_mean_%  — mean(HE / premium) × 100
        HE_per_prem_std_%   — std (HE / premium) × 100

    Caution: for deep OTM nodes, premium → 0, so this ratio can be very large
    and noisy. Near-ATM nodes are the most meaningful for this metric.
    """
    if "HE_per_premium_mean" not in he_agg.columns:
        return pd.DataFrame()

    q_labels = [f"q{int(q*100):02d}" for q in Q_GRID]

    if iv_surface_df is not None:
        mon_cols = list(iv_surface_df.columns)
    else:
        mon_cols = q_labels + ["ATM"]
        for T_days in TENORS_DAYS:
            if T_days not in he_agg.index.get_level_values("T_days"):
                continue
            sub = he_agg.xs(int(T_days), level="T_days")
            if -1.0 not in sub.index:
                continue
            k_atm = float(sub.loc[-1.0, "k_mean"])
            q_k   = [float(sub.loc[qv, "k_mean"]) if qv in sub.index else float("nan")
                     for qv in Q_GRID]
            ins   = sum(1 for k in q_k if not np.isnan(k) and k < k_atm)
            mon_cols = q_labels[:ins] + ["ATM"] + q_labels[ins:]
            break

    q_vals = [(-1.0 if lbl == "ATM" else int(lbl[1:]) / 100.0) for lbl in mon_cols]

    rows = []
    for T_days in TENORS_DAYS:
        T_int = int(T_days)
        if T_int not in he_agg.index.get_level_values("T_days"):
            continue
        sub = he_agg.xs(T_int, level="T_days")

        mean_vals = [round(float(sub.loc[qv, "HE_per_premium_mean"]) * 100.0, 4)
                     if qv in sub.index else float("nan")
                     for qv in q_vals]
        std_vals  = [round(float(sub.loc[qv, "HE_per_premium_std"]) * 100.0, 4)
                     if qv in sub.index else float("nan")
                     for qv in q_vals]

        rows.append([T_int, "HE_per_prem_mean_%"] + mean_vals)
        rows.append([T_int, "HE_per_prem_std_%"]  + std_vals)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["tenor_days", "metric"] + mon_cols)


# ── core logic ────────────────────────────────────────────────────────────────

def run_stage3(
        trades:        pd.DataFrame,
        iv_surface_df: pd.DataFrame | None = None,
        verbose:       bool = True,
        results_dir:   str | None = None,
) -> pd.DataFrame:
    """
    Execute Stage 3 entirely in memory.

    Returns
    -------
    he_stats : DataFrame
        Flat table (tenor_days, metric, q05 … ATM … q95) with rows
        model_vol%, HE_per_S_mean_%, HE_per_S_std_% per tenor.
        Same format as vol_quotes_integrated.csv.
    """
    t0 = time.time()

    if verbose:
        print(f"\n{_SEP}")
        print("  STAGE 3 — Hedging Error Statistics")
        print(f"  Strategy : {STRATEGY_NAME}")
        print(_SEP)

    # ── Step 9: Aggregate HE statistics ──────────────────────────────────────
    if verbose:
        print("\n── Step 9: Aggregate Hedging Error per (Tenor, Quantile) ───────")

    he_agg = compute_he_stats(trades, verbose=verbose)

    # ── Step 10: Build flat he_stats table ───────────────────────────────────
    if verbose:
        print("\n── Step 10: Build he_stats (model_vol% + HE_per_S) ─────────────")
        print("  Cols: tenor_days | metric | q05 … ATM … q95")

    he_stats = _build_he_stats(he_agg, iv_surface_df)

    # ── Premium stats ─────────────────────────────────────────────────────────
    he_premium_stats = _build_he_premium_stats(he_agg, iv_surface_df)

    if results_dir is not None and not he_premium_stats.empty:
        os.makedirs(results_dir, exist_ok=True)
        prem_path = _he_premium_stats_path(results_dir)
        he_premium_stats.to_csv(prem_path, index=False)
        if verbose:
            print("\n── HE / Premium Stats (saved separately) ───────────────────────")
            print("  Note: deep OTM values are amplified (premium → 0); "
                  "ATM nodes most meaningful.")
            print(he_premium_stats.to_string(index=False))
            print(f"\n  [Saved] he_premium_stats → {prem_path}")

    if verbose:
        if not he_stats.empty:
            print("\n── HE / S Stats ────────────────────────────────────────────────")
            print(he_stats.to_string(index=False))
        print(f"\n  [Stage 3 total: {time.time()-t0:.1f}s]")

    return he_stats


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    results_dir = os.path.join(_ROOT, "Results")

    print(f"\n{_SEP}")
    print("  Loading Stage 2 checkpoint …")
    trades, iv_surface_df, k_surface_df = _load_stage2(results_dir)
    print(f"  Trades loaded  : {len(trades):,} records")

    he_stats = run_stage3(trades, iv_surface_df, verbose=True,
                          results_dir=results_dir)

    print("\n── Saving Stage 3 Checkpoint ───────────────────────────────────")
    save_checkpoint(he_stats, results_dir)
    print(f"  [Saved] he_stats        → {_he_stats_path(results_dir)}")
    print(f"  [Saved] he_detail       → {_he_detail_path(results_dir)}")
    print(f"  [Saved] he_premium_stats→ {_he_premium_stats_path(results_dir)}")
    print(_SEP)
