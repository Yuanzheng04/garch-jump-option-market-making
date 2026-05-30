"""
stats.py
========
Step 6 — Hedging error statistics.

Normalisation: HE / S_entry  (fraction of stock price at trade entry).
This is directly comparable across all strikes and tenors, unlike
HE / premium which is amplified for OTM options with tiny premiums.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats


def compute_he_stats(trades: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Aggregate hedging-error statistics per (T_days, q) bucket.

    Returns
    -------
    DataFrame  indexed by (T_days, q) with columns:
        n
        k_mean              — mean log(K / S_entry)  [log-moneyness at entry]
        premium_mean        — mean premium received   [for verification]
        HE_mean             — mean absolute HE        [for verification]
        HE_std              — std  absolute HE
        HE_per_S_mean       — mean HE / S_entry       [primary metric: scale-free]
        HE_per_S_std        — std  HE / S_entry
        HE_per_S_tstat      — one-sample t-stat testing H0: E[HE/S] = 0
        HE_per_S_pval       — two-sided p-value for the t-test above
                              (small p = systematic bias is statistically significant)

    HE_per_S normalisation
    ----------------------
    HE_per_S = HE / S_entry is the correct metric because:
      * S_entry is the same order of magnitude for every (K, T) node
      * Premium varies by orders of magnitude across strikes, making
        HE/premium incomparable across moneyness levels
      * In the bid-ask formula: P_ask = C_BS + (z_alpha*sigma - mu)*S,
        where mu = E[HE/S] and sigma = std(HE/S), so the normalisation is
        directly tied to the pricing formula.

    t-test interpretation
    ---------------------
    H0: the model has no systematic bias at this (T_days, q) node.
    p < 0.05  → bias is statistically significant at the 5% level.
    t > 0     → seller systematically overcharged (model overpriced).
    t < 0     → seller systematically undercharged (model underpriced).
    """
    if trades.empty:
        return pd.DataFrame()

    trades = trades.copy()
    trades["k"] = np.log(trades["K"] / trades["S_entry"])

    # HE_per_S may already be in trades (added in backtest.py).
    # Recompute defensively in case old trades files are passed in.
    if "HE_per_S" not in trades.columns:
        trades["HE_per_S"] = trades["HE"] / trades["S_entry"].clip(lower=1e-8)

    # HE_per_premium: clip at 1e-8 to prevent division by zero.
    # Note: this ratio explodes at deep OTM where premium → 0; interpret
    # near-ATM nodes; treat extreme-strike values with caution.
    trades["HE_per_premium"] = (
        trades["HE"] / trades["premium"].clip(lower=1e-8)
    )

    agg_spec: dict = {
        "n"                    : ("HE",             "count"),
        "k_mean"               : ("k",              "mean"),
        "premium_mean"         : ("premium",         "mean"),
        "HE_mean"              : ("HE",             "mean"),
        "HE_std"               : ("HE",             "std"),
        "HE_per_S_mean"        : ("HE_per_S",       "mean"),
        "HE_per_S_std"         : ("HE_per_S",       "std"),
        "HE_per_premium_mean"  : ("HE_per_premium", "mean"),
        "HE_per_premium_std"   : ("HE_per_premium", "std"),
    }

    grp   = trades.groupby(["T_days", "q"])
    stats = grp.agg(**agg_spec).round(6)

    # one-sample t-test: H0: E[HE/S] = 0  (two-sided)
    def _tstat(x: pd.Series) -> float:
        if len(x) < 2:
            return float("nan")
        return float(_scipy_stats.ttest_1samp(x, popmean=0.0).statistic)

    def _pval(x: pd.Series) -> float:
        if len(x) < 2:
            return float("nan")
        return float(_scipy_stats.ttest_1samp(x, popmean=0.0).pvalue)

    stats["HE_per_S_tstat"] = grp["HE_per_S"].apply(_tstat).round(4)
    stats["HE_per_S_pval"]  = grp["HE_per_S"].apply(_pval).round(4)

    if verbose:
        print("\n── STEP 6: Hedging Error Summary (per q × Tenor) ─────────────")
        print("  HE = premium − payoff + hedge_pnl   (seller perspective)")
        print("  HE/S = HE / S_entry  (scale-free, comparable across strikes)")
        print("  t-stat / p-val: one-sample t-test of H0: E[HE/S] = 0\n")
        print(stats[["n", "k_mean", "HE_per_S_mean", "HE_per_S_std",
                      "HE_per_S_tstat", "HE_per_S_pval"]].to_string())
        print("\n  ── Cross-section summary ──")
        print(f"  Overall mean HE / S_entry   : "
              f"{trades['HE_per_S'].mean()*100:+.3f}%")
        print(f"  Overall std  HE / S_entry   : "
              f"{trades['HE_per_S'].std()*100:.3f}%")
        print(f"  Overall mean HE / premium   : "
              f"{trades['HE_per_premium'].mean()*100:+.3f}%"
              f"  (ATM nodes only meaningful)")
        print(f"  Fraction of profitable trades (HE>0) : "
              f"{(trades['HE'] > 0).mean()*100:.1f}%  (seller made money)")

    return stats
