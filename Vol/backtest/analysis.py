"""
analysis.py
===========
Statistical analysis of hedging-error std patterns.

Tests two theoretical predictions proved in the math write-up:

  Test A — std(HE/S) vs moneyness (quantile)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Prediction: std is HIGHEST at ATM and DECREASES monotonically as |k|
  increases toward wings (both high and low quantiles).
  Mechanism: Gamma ∝ N'(d1) is maximised at d1≈0 (ATM) and decays
  exponentially as |k| increases, so Var(HE) ∝ Gamma² does the same.

  Test B — std(HE/S) vs tenor
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Prediction: std INCREASES with T, driven by jump accumulation.
    - Diffusion component: roughly constant in T (∝ sqrt(Δt), not sqrt(T))
    - Jump component: ∝ sqrt(λT)   → combined std ≈ sqrt(C1 + C2·T)
  Fitting:  std(T)² = C1 + C2 · T  →  linear regression on std² vs T

Usage
-----
    python3 Vol/backtest/analysis.py          # runs on last backtest output
    from backtest.analysis import run_analysis
    run_analysis(he_stats, tenors_days, q_grid, out_dir)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG)
sys.path.insert(0, _PKG)
sys.path.insert(0, _ROOT)


# ── Test A: std vs quantile ───────────────────────────────────────────────────

def analyze_std_vs_quantile(
        he_stats:    pd.DataFrame,
        tenors_days: tuple | list,
        q_grid:      np.ndarray,
        verbose:     bool = True,
) -> pd.DataFrame:
    """
    For each tenor T, show std(HE/S) at each quantile label (sorted by k_mean).

    Theory check:
      - The column with the SMALLEST |k_mean| (closest to ATM) should have
        the LARGEST std.
      - std should decrease monotonically as |k_mean| increases.

    Returns DataFrame  rows=T_days, cols=labels, values=std(HE/S)×100 (in %)
    """
    q_labels  = [f"q{int(q*100):02d}" for q in q_grid]
    all_labels= q_labels + ["ATM"]   # order determined dynamically below
    q_vals    = {lbl: _label_to_q(lbl) for lbl in all_labels}

    rows = {}
    for T in tenors_days:
        if T not in he_stats.index.get_level_values("T_days"):
            continue
        sub  = he_stats.xs(T, level="T_days")
        # Sort labels by k_mean if available, else fall back to quantile ordering
        # (q05→ATM→q95 matches increasing log-moneyness k)
        k_map: dict[str, float] = {}
        has_k_mean = "k_mean" in sub.columns
        for lbl in all_labels:
            qv = q_vals[lbl]
            if qv not in sub.index:
                continue
            if has_k_mean:
                k_map[lbl] = float(sub.loc[qv, "k_mean"])
            else:
                # substitute ATM (-1.0) with 0.50 so it sorts between q45 and q55
                k_map[lbl] = 0.50 if qv < 0 else qv
        sorted_lbls = sorted(k_map, key=lambda l: k_map[l])

        row = {}
        for lbl in sorted_lbls:
            qv = q_vals[lbl]
            if qv in sub.index and "HE_per_S_std" in sub.columns:
                row[lbl] = round(float(sub.loc[qv, "HE_per_S_std"]) * 100, 4)
            else:
                row[lbl] = float("nan")
        rows[T] = row

    df = pd.DataFrame(rows).T
    df.index.name = "tenor_days"

    if verbose:
        print("\n══ Test A: std(HE/S) × 100  vs  moneyness  ═══════════════════")
        print("  Theory: highest at ATM, decreasing toward wings\n")
        print(df.to_string())
        print()
        _check_std_vs_quantile(df)

    return df


def _check_std_vs_quantile(std_df: pd.DataFrame) -> None:
    """
    For each tenor, find the label with max std and check if it is ATM
    or a neighbour of ATM.  Report monotonicity toward both wings.
    """
    print("  ── Monotonicity check (each tenor) ──────────────────────────")
    for T in std_df.index:
        row  = std_df.loc[T].dropna()
        if row.empty:
            continue
        cols     = list(row.index)
        vals     = list(row.values)
        peak_idx = int(np.argmax(vals))
        peak_lbl = cols[peak_idx]
        atm_idx  = cols.index("ATM") if "ATM" in cols else peak_idx

        # Check left wing (k < ATM): should be non-increasing toward left
        left_vals = vals[:atm_idx+1]
        left_ok   = all(left_vals[i] <= left_vals[i+1]
                        for i in range(len(left_vals)-1)) if len(left_vals)>1 else True

        # Check right wing (k > ATM): should be non-increasing toward right
        right_vals = vals[atm_idx:]
        right_ok   = all(right_vals[i] >= right_vals[i+1]
                         for i in range(len(right_vals)-1)) if len(right_vals)>1 else True

        status = "✓" if (left_ok and right_ok) else "⚠"
        print(f"    T={T:3d}d  peak at '{peak_lbl}' (idx {peak_idx})  "
              f"left-mono={'✓' if left_ok else '✗'}  "
              f"right-mono={'✓' if right_ok else '✗'}  {status}")


# ── Test B: std vs tenor ──────────────────────────────────────────────────────

def analyze_std_vs_tenor(
        he_stats:    pd.DataFrame,
        tenors_days: tuple | list,
        q_grid:      np.ndarray,
        trading_days: int = 252,
        verbose:     bool = True,
) -> dict[str, pd.DataFrame]:
    """
    For each quantile/ATM label, show how std(HE/S) evolves across tenors.

    Theory check:
      - std should increase with T (at least weakly).
      - std²  ≈ C1 + C2·T  →  fit a line to (T, std²) to estimate
        C1 (diffusion constant) and C2 (jump-intensity × crossing-prob factor).

    Returns dict with:
      "std_table"   : DataFrame  rows=labels, cols=T_days  values=std×100 (%)
      "fit_table"   : DataFrame  per label: C1, C2, R², monotone flag
    """
    q_labels  = [f"q{int(q*100):02d}" for q in q_grid]
    all_labels= q_labels + ["ATM"]

    # Build std_table[label, T]
    std_rows: dict[str, dict] = {lbl: {} for lbl in all_labels}
    for T in tenors_days:
        if T not in he_stats.index.get_level_values("T_days"):
            continue
        sub = he_stats.xs(T, level="T_days")
        for lbl in all_labels:
            qv = _label_to_q(lbl)
            if qv in sub.index and "HE_per_S_std" in sub.columns:
                std_rows[lbl][T] = float(sub.loc[qv, "HE_per_S_std"]) * 100

    std_df = pd.DataFrame(std_rows).T          # rows=labels, cols=T_days
    std_df.index.name  = "label"
    std_df.columns.name = "tenor_days"

    # Fit  std²(T) = C1 + C2·T  for each label
    fit_rows = []
    T_arr = np.array(sorted(tenors_days), dtype=float)
    for lbl in all_labels:
        vals_pct = np.array([std_rows[lbl].get(T, np.nan) for T in T_arr])
        ok       = np.isfinite(vals_pct) & (vals_pct > 0)
        if ok.sum() < 2:
            fit_rows.append({"label": lbl, "C1": np.nan, "C2": np.nan,
                             "R2": np.nan, "monotone": False})
            continue
        T_ok   = T_arr[ok]
        y_ok   = (vals_pct[ok] / 100) ** 2       # std² (in fraction² units)

        # OLS: y = C1 + C2·T
        X = np.column_stack([np.ones_like(T_ok), T_ok])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y_ok, rcond=None)
        except Exception:
            fit_rows.append({"label": lbl, "C1": np.nan, "C2": np.nan,
                             "R2": np.nan, "monotone": False})
            continue
        C1, C2 = float(beta[0]), float(beta[1])
        y_hat  = C1 + C2 * T_ok
        ss_tot = float(np.var(y_ok)) * len(y_ok)
        ss_res = float(np.sum((y_ok - y_hat)**2))
        R2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else float("nan")

        # Monotonicity: std values non-decreasing in T
        monotone = all(vals_pct[ok][i] <= vals_pct[ok][i+1] + 1e-6
                       for i in range(ok.sum()-1))
        fit_rows.append({
            "label":    lbl,
            "C1_pct2":  round(C1 * 1e4, 6),   # ×1e4 for readability (pct²)
            "C2_pct2_per_day": round(C2 * 1e4, 8),
            "R2":       round(R2, 4),
            "monotone": monotone,
        })

    fit_df = pd.DataFrame(fit_rows).set_index("label")

    if verbose:
        print("\n══ Test B: std(HE/S) × 100  vs  tenor  ══════════════════════")
        print("  Theory: std increases with T; std² ≈ C1 + C2·T\n")
        print(std_df.round(4).to_string())
        print("\n  ── Curve fit:  std²(T) = C1 + C2·T ─────────────────────────")
        print("  (C1_pct2, C2_pct2_per_day in units of (pct)²)\n")
        print(fit_df.to_string())
        print()
        _check_std_vs_tenor(std_df, fit_df)

    return {"std_table": std_df, "fit_table": fit_df}


def _check_std_vs_tenor(
        std_df: pd.DataFrame,
        fit_df: pd.DataFrame,
) -> None:
    print("  ── Summary ───────────────────────────────────────────────────")
    n_mono  = fit_df["monotone"].sum()
    n_total = len(fit_df)
    pos_C2  = (fit_df["C2_pct2_per_day"] > 0).sum()
    print(f"    Monotone increasing (std vs T) : {n_mono}/{n_total} labels")
    print(f"    Positive C2 (jump component)   : {pos_C2}/{n_total} labels")
    if "ATM" in fit_df.index:
        r = fit_df.loc["ATM"]
        print(f"    ATM fit: C1={r['C1_pct2']:.4f} pct²  "
              f"C2={r['C2_pct2_per_day']:.6f} pct²/day  "
              f"R²={r['R2']:.3f}  "
              f"{'monotone ✓' if r['monotone'] else 'non-monotone ⚠'}")


# ── Test A2: mean(HE/S) vs moneyness ─────────────────────────────────────────

def analyze_mean_vs_quantile(
        he_stats:    pd.DataFrame,
        tenors_days: tuple | list,
        q_grid:      np.ndarray,
        verbose:     bool = True,
) -> pd.DataFrame:
    """
    For each tenor T, show mean(HE/S) at each quantile label.

    HE = premium − payoff + hedge_pnl  (seller's perspective).
    mean > 0 → seller systematically profitable at that (T, k) node.
    mean < 0 → seller systematically losing at that node.

    Returns DataFrame  rows=T_days, cols=labels, values=mean(HE/S)×100 (in %)
    """
    q_labels   = [f"q{int(q*100):02d}" for q in q_grid]
    all_labels = q_labels + ["ATM"]
    q_vals     = {lbl: _label_to_q(lbl) for lbl in all_labels}

    rows: dict[int, dict[str, float]] = {}
    for T in tenors_days:
        if T not in he_stats.index.get_level_values("T_days"):
            continue
        sub = he_stats.xs(T, level="T_days")
        has_k_mean = "k_mean" in sub.columns
        k_map: dict[str, float] = {}
        for lbl in all_labels:
            qv = q_vals[lbl]
            if qv not in sub.index:
                continue
            k_map[lbl] = float(sub.loc[qv, "k_mean"]) if has_k_mean else (
                0.50 if qv < 0 else qv)
        sorted_lbls = sorted(k_map, key=lambda l: k_map[l])

        row: dict[str, float] = {}
        for lbl in sorted_lbls:
            qv = q_vals[lbl]
            if qv in sub.index and "HE_per_S_mean" in sub.columns:
                row[lbl] = round(float(sub.loc[qv, "HE_per_S_mean"]) * 100, 4)
            else:
                row[lbl] = float("nan")
        rows[int(T)] = row

    df = pd.DataFrame(rows).T
    df.index.name = "tenor_days"

    if verbose:
        print("\n══ Test A2: mean(HE/S) × 100  vs  moneyness  ════════════════")
        print("  HE = premium − payoff + hedge_pnl  (seller perspective)")
        print("  + = seller profitable,  − = seller losing\n")
        print(df.to_string())
        print()
        _check_mean_pattern(df)

    return df


def _check_mean_pattern(mean_df: pd.DataFrame) -> None:
    """Print sign pattern and where bias is most negative (OTM risk)."""
    print("  ── Bias sign by (tenor, moneyness) ──────────────────────────")
    for T in mean_df.index:
        row  = mean_df.loc[T].dropna()
        if row.empty:
            continue
        signs   = "".join("+" if v > 0 else ("−" if v < 0 else "0")
                          for v in row.values)
        neg_lbl = row.idxmin()
        pos_lbl = row.idxmax()
        print(f"    T={T:3d}d  [{signs}]  "
              f"most neg: {neg_lbl}={row[neg_lbl]:+.4f}%  "
              f"most pos: {pos_lbl}={row[pos_lbl]:+.4f}%")


# ── combined run ──────────────────────────────────────────────────────────────

def run_analysis(
        he_stats:     pd.DataFrame,
        tenors_days:  tuple | list,
        q_grid:       np.ndarray,
        out_dir:      str | None = None,
        trading_days: int = 252,
        verbose:      bool = True,
) -> dict:
    """
    Run Tests A, A2, B and optionally save results to CSVs.

    Returns dict with keys:
        'std_vs_quantile', 'mean_vs_quantile', 'std_vs_tenor', 'fit_table'
    """
    std_q  = analyze_std_vs_quantile(he_stats, tenors_days, q_grid,
                                     verbose=verbose)
    mean_q = analyze_mean_vs_quantile(he_stats, tenors_days, q_grid,
                                      verbose=verbose)
    res    = analyze_std_vs_tenor(he_stats, tenors_days, q_grid,
                                  trading_days=trading_days, verbose=verbose)
    std_t  = res["std_table"]
    fit_t  = res["fit_table"]

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        std_q.to_csv(os.path.join(out_dir, "std_vs_quantile.csv"))
        mean_q.to_csv(os.path.join(out_dir, "mean_vs_quantile.csv"))
        std_t.to_csv(os.path.join(out_dir, "std_vs_tenor.csv"))
        fit_t.to_csv(os.path.join(out_dir, "std_tenor_fit.csv"))
        if verbose:
            print(f"\n  [Saved] std/mean_vs_quantile / std_vs_tenor / fit → {out_dir}")

    return {
        "std_vs_quantile":  std_q,
        "mean_vs_quantile": mean_q,
        "std_vs_tenor":     std_t,
        "fit_table":        fit_t,
    }


# ── helper ───────────────────────────────────────────────────────────────────

def _label_to_q(label: str) -> float:
    if label == "ATM":
        return -1.0
    return int(label[1:]) / 100.0


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from backtest.config import TENORS_DAYS, Q_GRID, TRADING_DAYS
    from backtest.trading import STRATEGY_NAME

    results_dir = os.path.join(_ROOT, "Results")

    # Try to load he_stats from the most recent backtest output
    # (expects a single ticker run from run.py)
    import glob
    pattern = os.path.join(results_dir, f"*_{STRATEGY_NAME}_he_detail.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"No he_detail files found matching: {pattern}")
        print("Run run.py first to generate backtest results.")
        sys.exit(1)

    # Use the first matching file
    path   = files[0]
    ticker = os.path.basename(path).replace(f"_{STRATEGY_NAME}_he_detail.csv", "")
    print(f"Loading he_stats from: {path}  (ticker={ticker})")

    he_raw = pd.read_csv(path, index_col=["T_days", "q"])

    run_analysis(
        he_stats=he_raw,
        tenors_days=TENORS_DAYS,
        q_grid=Q_GRID,
        out_dir=results_dir,
        trading_days=TRADING_DAYS,
        verbose=True,
    )
