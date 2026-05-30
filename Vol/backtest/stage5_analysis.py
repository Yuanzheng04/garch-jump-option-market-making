"""
stage5_analysis.py
==================
Stage 5 — Hedging Error Pattern Analysis   [S5-13 … S5-18]

Purpose
-------
Test theoretical predictions about how the hedging error (HE) varies across
moneyness, tenor, and market capitalisation.

  HE = premium − payoff + hedge_pnl   (seller's perspective)
  HE_per_S = HE / S                   (scale-free metric)

Single-ticker tests  (Tests A, A2, B)  — uses config.TICKER
─────────────────────────────────────────────────────────────
  S5-13  Test A  │ std(HE/S) vs moneyness
  ───────────────┤ analyze_std_vs_quantile()
                 │ Theory: highest at ATM (Gamma ∝ N′(d₁) maximised there),
                 │ decreasing monotonically toward wings.
                 │
  S5-14  Test A2 │ mean(HE/S) vs moneyness  (seller bias)
  ───────────────┤ analyze_mean_vs_quantile()
                 │ + = seller systematically profitable at (T, k) node
                 │ − = seller systematically losing
                 │
  S5-15  Test B  │ std(HE/S) vs tenor  (jump accumulation)
  ───────────────┤ analyze_std_vs_tenor()
                 │ Theory: std increases with T due to jump accumulation.
                 │ OLS fit:  std²(T) = C₁ + C₂·T  per quantile label.

Cross-sectional tests  (Tests A_xs, B_xs, C, D, E)  — uses all CROSS_TICKERS
─────────────────────────────────────────────────────────────────────────────
  S5-13x Test A_xs │ Robustness of ATM-peak pattern across all tickers
  ────────────────┤ analyze_cross_std_pattern()
                  │ Mean σ_HE% heatmap across tickers (rows=tenors, cols=labels).
                  │ Per-ticker: % tenors where ATM label has highest σ.
                  │ Pass threshold: ATM is peak in ≥ 60% of tenors.
                  │ Summary: % tickers that pass → "robust" or "not robust".
                  │
  S5-15x Test B_xs │ Robustness of σ increasing with tenor across all tickers
  ────────────────┤ analyze_cross_std_pattern()
                  │ Per-ticker OLS fit: std²(T) = C₁ + C₂·T  (ATM label).
                  │ Summary: % tickers with C₂ > 0 and % with monotone series.
                  │ Robust if C₂ > 0 in ≥ 70% tickers and monotone in ≥ 60%.
                  │
  S5-16  Test C  │ HE bias per (T, q) node across all tickers
  ───────────────┤ analyze_cross_bias()
                 │ cross_mean_%           = mean over tickers of HE_per_S_mean × 100
                 │ cross_std_%            = between-ticker SD (denominator for t-test)
                 │ rms_per_ticker_std_%   = √(mean σ_j²) × 100, pooled within-ticker
                 │                         HE vol; reflects single-name uncertainty
                 │ t-stat                 = cross_mean / (cross_std / √n)
                 │                         Is average ticker-level mean HE
                 │                         significantly ≠ 0 across names?
                 │ |t| > 1.96 → significantly biased node  (++ or --)
                 │
  S5-17  Test D  │ Spearman(HE std, market-cap rank) per (T, q)
  ───────────────┤ analyze_he_vs_mktcap()
                 │ ρ < 0 → larger cap → lower HE std  (confirmed empirically)
                 │ ρ heatmap with * marking p < 0.10 significance
                 │ Per-tenor summary: avg ρ, % nodes negative, conclusion
                 │
  S5-18  Test E  │ Average HE std by market-cap quintile
  ───────────────┤ analyze_he_vs_mktcap()
                 │ Q1 = 5 smallest caps … Q4 = 5 largest caps
                 │ ATM-only staircase table (Q1→Q4, cols = tenors)
                 │ (Q1−Q4)/Q1 × 100 decline heatmap across all (T, q)

Reads  (Results/)
-----------------
  {TICKER}_{STRAT}_he_detail.csv   ← long-format HE stats (Stage 3 → S3-10)
  {TICKER}_{STRAT}_he_stats.csv    ← wide-format fallback  (parsed on the fly)

Outputs  (Results/)
-------------------
Single-ticker (Tests A, A2, B):
  {TICKER}_{STRAT}_std_vs_quantile.csv
  {TICKER}_{STRAT}_mean_vs_quantile.csv
  {TICKER}_{STRAT}_std_vs_tenor.csv
  {TICKER}_{STRAT}_std_tenor_fit.csv

Cross-section  (Tests A_xs, B_xs):
  cross_std_heatmap.csv                ← mean σ_HE% across tickers (tenors × labels)
  cross_test_a_robust.csv              ← per-ticker ATM-peak check
  cross_test_a_summary.csv             ← aggregate % tickers where ATM is peak
  cross_test_b_atm_fit.csv             ← per-ticker ATM OLS fit (C1, C2, R², mono)
  cross_test_b_summary.csv             ← % C2>0, % monotone, mean C2 across tickers

Cross-section  (Test C):
  cross_he_bias.csv                    ← long format, all columns per (T, q)
  cross_he_mean_heatmap.csv            ← cross_mean_% pivot  (tenors × q nodes)
  cross_he_tstat_heatmap.csv           ← t-stat pivot
  cross_he_signmap.csv                 ← bias sign map:  ++ / -- / ·

Cross-section  (Test D — Spearman):
  cross_spearman_mktcap.csv            ← long format, all columns per (T, q)
  cross_spearman_rho_heatmap.csv       ← ρ heatmap (tenors × q),  * = p<0.10
  cross_spearman_tenor_summary.csv     ← per-tenor: avg ρ, % nodes negative,
                                          conclusion (larger cap → lower/higher HE)

Cross-section  (Test E — Quintile):
  cross_quintile_std.csv               ← long format, Q1…Q4 avg std per (T, q)
  cross_quintile_wide_atm.csv          ← ATM only, rows=Q1→Q4+decline%, cols=tenors
  cross_quintile_q1q4_decline_heatmap.csv  ← (Q1−Q4)/Q1×100 for all (T, q)

Run
---
    cd project_updates
    python3 Vol/backtest/stage5_analysis.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from typing import Any

# ── package path ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG)
sys.path.insert(0, _PKG)
sys.path.insert(0, _ROOT)

from backtest.config   import TICKER, TENORS_DAYS, Q_GRID, TRADING_DAYS
from backtest.trading  import STRATEGY_NAME
from backtest.analysis import run_analysis

from backtest.stage3_stats import (load_checkpoint as _load_stage3,
                                   load_checkpoint_for_ticker as _load3_ticker)

_SEP = "═" * 65

# ── Cross-section ticker list: ordered small → large market cap (02/24) ──────
CROSS_TICKERS = [
    "688092", "600128", "000677", "300701", "300932",
    "000798", "002449", "600984", "002390", "688488",
    "300019", "600586", "688366", "600755", "603612",
    "605099", "603067", "600909", "002648", "601006",
]
# index 0 = smallest mktcap, index 19 = largest mktcap


# ── shared helper ─────────────────────────────────────────────────────────────

def _bias_sign(cross_mean_pct: float, t_stat: Any) -> str:
    """
    Return a symbol for the systematic bias of one (T, q) node.
      ++  significant seller profit  (|t| > 1.96, mean > 0)
      --  significant seller loss    (|t| > 1.96, mean < 0)
      ·   not statistically significant
    """
    t = float(t_stat) if (t_stat is not None and np.isfinite(float(t_stat))) else 0.0
    if abs(t) <= 1.96:
        return "·"
    return "++" if cross_mean_pct > 0 else "--"


# ── single-ticker Stage 5 ─────────────────────────────────────────────────────

def run_stage5(
        he_stats: pd.DataFrame,
        verbose:  bool = True,
) -> dict:
    """
    Execute Stage 5 single-ticker tests (Tests A, A2, B) in memory.

    Returns dict with keys: std_vs_quantile, mean_vs_quantile,
                            std_vs_tenor, fit_table
    """
    t0 = time.time()

    if verbose:
        print(f"\n{_SEP}")
        print("  STAGE 5 — Hedging Error Pattern Analysis")
        print(f"  Strategy : {STRATEGY_NAME}")
        print(_SEP)
        print("\n── Step 14 + 15: Run HE Pattern Analysis ───────────────────────")
        print("  Test A   — std  vs moneyness  (should peak near ATM)")
        print("  Test A2  — mean vs moneyness  (+ = seller profitable)")
        print("  Test B   — std  vs tenor      (std² ≈ C₁ + C₂·T)")

    result = run_analysis(
        he_stats=he_stats,
        tenors_days=TENORS_DAYS,
        q_grid=Q_GRID,
        out_dir=None,
        trading_days=TRADING_DAYS,
        verbose=verbose,
    )

    if verbose:
        print(f"\n  [Stage 5 total: {time.time()-t0:.1f}s]")

    return result


# ── Cross-section helpers ─────────────────────────────────────────────────────

def _load_he_all(
        results_dir: str,
        tickers: list[str],
        strategy: str,
) -> dict[str, pd.DataFrame]:
    """
    Load HE stats for each ticker as a (T_days, q)-indexed DataFrame.

    Priority:
      1. {ticker}_{strategy}_he_detail.csv  (long format, direct load)
      2. {ticker}_{strategy}_he_stats.csv   (wide format, parsed via
         load_checkpoint_for_ticker)
    Skips tickers where neither file exists.
    """
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        detail_path = os.path.join(results_dir, f"{t}_{strategy}_he_detail.csv")
        stats_path  = os.path.join(results_dir, f"{t}_{strategy}_he_stats.csv")
        if os.path.exists(detail_path):
            df = pd.read_csv(detail_path, index_col=["T_days", "q"])
            out[t] = df
        elif os.path.exists(stats_path):
            try:
                df = _load3_ticker(results_dir, t)
                out[t] = df
            except Exception as e:
                print(f"  [SKIP] {t}: failed to parse he_stats.csv — {e}")
        else:
            print(f"  [SKIP] {t}: no he_stats/he_detail file found")
    return out


def _node_values(
        he_dict: dict[str, pd.DataFrame],
        T: int,
        q: float,
        col: str,
        tickers: list[str] | None = None,
) -> tuple[list[float], list[str]]:
    """Return (values, ticker_labels) for a given (T, q) node and column."""
    src = tickers if tickers is not None else list(he_dict.keys())
    vals, labels = [], []
    for t in src:
        if t not in he_dict:
            continue
        df = he_dict[t]
        if (T, q) in df.index and col in df.columns:
            vals.append(float(df.loc[(T, q), col]))
            labels.append(t)
    return vals, labels


# ── Test C: cross-ticker HE bias per (T, q) ──────────────────────────────────

def analyze_cross_bias(
        he_dict:     dict[str, pd.DataFrame],
        tenors_days: tuple | list,
        q_grid:      np.ndarray,
        verbose:     bool = True,
) -> pd.DataFrame:
    """
    Test C — for each (T, q) node, test whether mean(HE_per_S_mean) across
    tickers is significantly different from zero.

    cross_std_% (between-ticker)
        Sample SD of the n values {HE_per_S_mean_j} (one per ticker), in %.
        Correct denominator for SE(mean): t = cross_mean / (cross_std / √n).

    rms_per_ticker_std_% (pooled within-ticker volatility)
        √( (1/n) Σ_j σ_j² ) × 100, where σ_j = HE_per_S_std per ticker.
        Summarises typical bucket-level HE dispersion; not used in the t-test.

    Nodes with |t| > 1.96 are flagged as significant.

    Returns DataFrame: T_days, q, n_tickers, cross_mean_%, cross_std_%,
                       rms_per_ticker_std_%, t_stat, significant
    """
    q_vals = list(q_grid) + [-1.0]   # -1.0 = ATM
    rows = []
    for T in tenors_days:
        for q in q_vals:
            vals, labels = _node_values(he_dict, T, q, "HE_per_S_mean")
            if len(vals) < 3:
                continue
            arr = np.array(vals) * 100
            n   = len(arr)
            m   = float(np.mean(arr))
            s   = float(np.std(arr, ddof=1))
            t   = m / (s / np.sqrt(n)) if s > 1e-12 else np.nan

            std_fracs: list[float] = []
            for tkr in labels:
                df_t = he_dict[tkr]
                if (T, q) in df_t.index and "HE_per_S_std" in df_t.columns:
                    std_fracs.append(float(df_t.loc[(T, q), "HE_per_S_std"]))
            rms_pct = (
                float(np.sqrt(np.mean(np.square(std_fracs))) * 100.0)
                if len(std_fracs) > 0 else float("nan")
            )

            rows.append({
                "T_days":               T,
                "q":                    q,
                "n_tickers":            n,
                "cross_mean_%":         round(m, 4),
                "cross_std_%":          round(s, 4),
                "rms_per_ticker_std_%": round(rms_pct, 4),
                "t_stat":               round(t, 3) if np.isfinite(t) else np.nan,
                "significant":          bool(np.isfinite(t) and abs(t) > 1.96),
            })

    df_out = pd.DataFrame(rows)

    if verbose:
        print(f"\n{_SEP}")
        print("  Test C: Cross-ticker HE bias per (T, q) node")
        print("  HE = premium − payoff + hedge_pnl  (seller perspective)")
        print("  cross_mean > 0 → seller profitable;  < 0 → seller losing")
        print("  t-stat = cross-mean / (cross-std / √n);  flag |t| > 1.96")
        print(_SEP)
        if df_out.empty:
            print("  No data.")
        else:
            sig = df_out[df_out["significant"]].copy()
            sig = sig.iloc[abs(sig["t_stat"]).argsort()[::-1]]
            if sig.empty:
                print("  No significantly biased nodes found (all |t| ≤ 1.96).")
            else:
                print(f"  {len(sig)} significantly biased nodes (|t| > 1.96):\n")
                cols_show = ["T_days", "q", "n_tickers",
                             "cross_mean_%", "cross_std_%", "t_stat"]
                print(sig[cols_show].to_string(index=False))

            # Heatmap 1: cross_mean_%  (economic magnitude)
            pivot_m = df_out.pivot(index="T_days", columns="q", values="cross_mean_%")
            print("\n  cross_mean_% heatmap  (+ = seller profitable, − = losing):")
            print(pivot_m.round(3).to_string())

            # Heatmap 2: t-stat
            pivot_t = df_out.pivot(index="T_days", columns="q", values="t_stat")
            print("\n  t-stat heatmap  (|t|>1.96 flagged):")
            print(pivot_t.round(2).to_string())

            # Sign map (use spaces for terminal alignment)
            def _sign_aligned(row: "pd.Series[float]") -> str:  # type: ignore[type-arg]
                return "  ·" if _bias_sign(
                    float(row["cross_mean_%"]), row["t_stat"]) == "·" else (
                    " ++" if float(row["cross_mean_%"]) > 0 else " --"
                )

            pivot_s = df_out.assign(
                _sgn=df_out.apply(_sign_aligned, axis=1)
            ).pivot(index="T_days", columns="q", values="_sgn")
            print("\n  Systematic bias map  (++ sig. seller profitable, "
                  "-- sig. seller losing, · not sig.):")
            print(pivot_s.to_string())

    return df_out


# ── Test D: Spearman(HE std, mktcap rank) per (T, q) ─────────────────────────

def analyze_he_vs_mktcap(
        he_dict:     dict[str, pd.DataFrame],
        tickers:     list[str],          # ordered small → large
        tenors_days: tuple | list,
        q_grid:      np.ndarray,
        verbose:     bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test D — Spearman rank correlation between HE metric and market-cap rank
             per (T, q) node.

    Test E — Average HE std by market-cap quintile (Q1=5 smallest … Q4=5 largest).

    Returns (corr_df, quintile_df)
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("  [WARN] scipy not installed; Spearman tests skipped.")
        return pd.DataFrame(), pd.DataFrame()

    mktcap_rank = {t: i + 1 for i, t in enumerate(tickers)}
    q_vals      = list(q_grid) + [-1.0]

    # ── Test D ────────────────────────────────────────────────────────────────
    corr_rows = []
    for T in tenors_days:
        for q in q_vals:
            std_vals, tkrs = _node_values(he_dict, T, q, "HE_per_S_std", tickers)
            mean_vals, _   = _node_values(he_dict, T, q, "HE_per_S_mean", tickers)
            if len(std_vals) < 5:
                continue
            ranks   = [mktcap_rank[t] for t in tkrs]
            _sr = spearmanr(ranks, [v * 100 for v in std_vals])
            _sm = spearmanr(ranks, [abs(v) * 100 for v in mean_vals])
            r_std  = float(np.asarray(_sr)[0])
            p_std  = float(np.asarray(_sr)[1])
            r_mean = float(np.asarray(_sm)[0])
            p_mean = float(np.asarray(_sm)[1])
            corr_rows.append({
                "T_days":           T,
                "q":                q,
                "n":                len(std_vals),
                "spearman_std":     round(float(r_std),  3),
                "p_std":            round(float(p_std),  3),
                "sig_std":          float(p_std) < 0.10,
                "spearman_absmean": round(float(r_mean), 3),
                "p_mean":           round(float(p_mean), 3),
                "sig_mean":         float(p_mean) < 0.10,
            })

    corr_df = pd.DataFrame(corr_rows)

    # ── Test E ────────────────────────────────────────────────────────────────
    n_per_q  = 5
    quintile = {t: (i // n_per_q) + 1 for i, t in enumerate(tickers)}

    quint_rows = []
    for T in tenors_days:
        for q in q_vals:
            by_q: dict[int, list] = {1: [], 2: [], 3: [], 4: []}
            for t in tickers:
                if t not in he_dict:
                    continue
                df = he_dict[t]
                if (T, q) in df.index and "HE_per_S_std" in df.columns:
                    v   = float(df.loc[(T, q), "HE_per_S_std"]) * 100
                    qnt = quintile[t]
                    by_q[qnt].append(v)
            row: dict = {"T_days": T, "q": q}
            for qnt in [1, 2, 3, 4]:
                row[f"Q{qnt}_avg_std_%"] = (
                    round(np.mean(by_q[qnt]), 4) if by_q[qnt] else np.nan
                )
            quint_rows.append(row)

    quintile_df = pd.DataFrame(quint_rows)

    if verbose:
        # ── Test D printout ───────────────────────────────────────────────────
        print(f"\n{_SEP}")
        print("  Test D: Spearman(HE std, mktcap rank) per (T, q)")
        print("  positive ρ → larger mktcap = higher HE std")
        print(_SEP)
        if corr_df.empty:
            print("  No data.")
        else:
            sig_d = corr_df[corr_df["sig_std"]].copy()
            sig_d = sig_d.sort_values("spearman_std", ascending=False)  # type: ignore[call-overload]
            print(f"  Nodes with p_std < 0.10  ({len(sig_d)} / {len(corr_df)}):")
            if sig_d.empty:
                print("    None.")
            else:
                cols_d = ["T_days", "q", "n", "spearman_std", "p_std"]
                print(pd.DataFrame(sig_d)[cols_d].to_string(index=False))

            pos = (corr_df["spearman_std"] > 0).sum()
            neg = (corr_df["spearman_std"] < 0).sum()
            print(f"\n  Direction summary: {pos} nodes ρ>0, {neg} nodes ρ<0 "
                  f"(out of {len(corr_df)} total)")

            pivot_r = corr_df.pivot(index="T_days", columns="q", values="spearman_std")
            print("\n  Spearman ρ (std) heatmap (T_days × q):")
            print(pivot_r.round(2).to_string())

        # ── Test E printout ───────────────────────────────────────────────────
        print(f"\n{_SEP}")
        print("  Test E: Average HE std (%) by mktcap quintile")
        print("  Q1 = smallest 5 tickers  …  Q4 = largest 5 tickers")
        print(_SEP)
        if quintile_df.empty:
            print("  No data.")
        else:
            atm = quintile_df[quintile_df["q"] == -1.0].copy()
            if not atm.empty:
                print("  ATM nodes (q = -1.0 / ATM):")
                print(atm.drop(columns="q").to_string(index=False))

            print("\n  Monotonicity (Q1_avg < Q4_avg at ATM):")
            for _, qrow in atm.iterrows():
                q1 = float(qrow.get("Q1_avg_std_%", np.nan))  # type: ignore[arg-type]
                q4 = float(qrow.get("Q4_avg_std_%", np.nan))  # type: ignore[arg-type]
                if np.isfinite(q1) and np.isfinite(q4):
                    flag = "↑ larger cap → more HE" if q4 > q1 else "↓ no clear size effect"
                else:
                    flag = "?"
                print(f"    T={int(qrow['T_days']):3d}d  Q1={q1:.4f}%  Q4={q4:.4f}%  {flag}")

    return corr_df, quintile_df


# ── Tests A_xs + B_xs: cross-stock std pattern robustness ────────────────────

def analyze_cross_std_pattern(
        he_dict:      dict[str, pd.DataFrame],
        tenors_days:  tuple | list,
        q_grid:       np.ndarray,
        trading_days: int  = TRADING_DAYS,
        verbose:      bool = True,
) -> dict:
    """
    Test A (cross-stock) + Test B (cross-stock) — robustness across all tickers.

    Test A_xs  — does σ_HE peak at ATM across ALL tickers?
    ────────────────────────────────────────────────────────
    For each (T, q) node: collect σ_HE from all tickers, compute cross-mean.
    For each ticker × tenor: check if ATM label has the highest σ.
    Aggregate % tenors where ATM is peak, per ticker → overall robustness score.

    Test B_xs  — is C₂ > 0 (σ grows with tenor) across ALL tickers?
    ────────────────────────────────────────────────────────────────
    For each ticker × label: fit std²(T) = C₁ + C₂·T via OLS.
    Record C₂ sign and monotone flag for ATM label.
    Aggregate % tickers with C₂ > 0 and % with monotone increasing series.

    Returns dict
    ------------
    std_heatmap   : DataFrame (T_days × labels) — mean σ_HE% across tickers
    test_a_robust : DataFrame (per-ticker) — n tenors checked, n ATM-peak, %
    test_a_summary: DataFrame (1 row)     — aggregate % ATM-peak across tickers
    test_b_fit    : DataFrame (per-ticker) — ATM OLS results (C1, C2, R², mono)
    test_b_summary: DataFrame (1 row)     — % C2>0, % monotone, mean C2/R²
    """
    q_vals  = list(q_grid) + [-1.0]
    q_label = {q: (f"q{int(q*100):02d}" if q >= 0 else "ATM") for q in q_vals}
    labels  = [f"q{int(q*100):02d}" for q in q_grid] + ["ATM"]
    tickers = list(he_dict.keys())

    # ── Test A_xs: mean σ heatmap + ATM-peak robustness ──────────────────────

    # 1. Mean σ_HE heatmap across tickers
    heatmap_rows: dict[int, dict[str, float]] = {}
    for T in tenors_days:
        row: dict[str, float] = {}
        for q in q_vals:
            lbl = q_label[q]
            vals, _ = _node_values(he_dict, T, q, "HE_per_S_std")
            row[lbl] = round(float(np.mean(vals)) * 100, 4) if vals else float("nan")
        heatmap_rows[int(T)] = row

    std_heatmap = pd.DataFrame(heatmap_rows).T.reindex(columns=labels)
    std_heatmap.index.name   = "tenor_days"
    std_heatmap.columns.name = "label"

    # 2. Per-ticker ATM-peak check
    robust_rows = []
    for tkr in tickers:
        df_t = he_dict[tkr]
        n_checked = n_atm_peak = 0
        for T in tenors_days:
            T_int = int(T)
            σ_by_lbl: dict[str, float] = {}
            for q in q_vals:
                lbl = q_label[q]
                if (T_int, q) in df_t.index and "HE_per_S_std" in df_t.columns:
                    σ_by_lbl[lbl] = float(df_t.loc[(T_int, q), "HE_per_S_std"])
            if not σ_by_lbl or "ATM" not in σ_by_lbl:
                continue
            n_checked += 1
            # ATM is peak if σ_ATM >= all other labels
            σ_atm = σ_by_lbl["ATM"]
            if all(σ_atm >= v for v in σ_by_lbl.values()):
                n_atm_peak += 1
        pct = round(n_atm_peak / n_checked * 100, 1) if n_checked > 0 else float("nan")
        robust_rows.append({
            "ticker":           tkr,
            "n_tenors_checked": n_checked,
            "n_tenors_atm_peak": n_atm_peak,
            "pct_atm_peak_%":   pct,
            "pass":             pct >= 60 if np.isfinite(pct) else False,
        })
    test_a_robust = pd.DataFrame(robust_rows)

    n_pass  = int(test_a_robust["pass"].sum()) if not test_a_robust.empty else 0
    n_total = len(test_a_robust)
    test_a_summary = pd.DataFrame([{
        "n_tickers":        n_total,
        "n_pass_atm_peak":  n_pass,
        "pct_pass_%":       round(n_pass / n_total * 100, 1) if n_total > 0 else float("nan"),
        "mean_pct_atm_%":   round(float(test_a_robust["pct_atm_peak_%"].mean()), 1)
                            if not test_a_robust.empty else float("nan"),
        "conclusion":       ("ATM-peak pattern robust" if n_pass / max(n_total, 1) >= 0.6
                             else "ATM-peak pattern NOT robust across tickers"),
    }])

    # ── Test B_xs: per-ticker OLS fit for ATM ─────────────────────────────────

    T_arr = np.array(sorted(tenors_days), dtype=float)
    fit_rows_b = []
    for tkr in tickers:
        df_t = he_dict[tkr]
        # Collect σ_HE for ATM across tenors
        σ_vals = np.array([
            float(df_t.loc[(int(T), -1.0), "HE_per_S_std"])
            if (int(T), -1.0) in df_t.index else float("nan")
            for T in T_arr
        ])
        ok = np.isfinite(σ_vals) & (σ_vals > 0)
        if ok.sum() < 2:
            fit_rows_b.append({
                "ticker": tkr, "C1_pct2": np.nan, "C2_pct2_per_day": np.nan,
                "R2": np.nan, "monotone": False, "C2_positive": False,
            })
            continue
        T_ok = T_arr[ok]
        y_ok = (σ_vals[ok]) ** 2          # std² in fraction² units
        X    = np.column_stack([np.ones_like(T_ok), T_ok])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y_ok, rcond=None)
        except Exception:
            fit_rows_b.append({
                "ticker": tkr, "C1_pct2": np.nan, "C2_pct2_per_day": np.nan,
                "R2": np.nan, "monotone": False, "C2_positive": False,
            })
            continue
        C1, C2 = float(beta[0]), float(beta[1])
        y_hat  = C1 + C2 * T_ok
        ss_tot = float(np.var(y_ok)) * len(y_ok)
        ss_res = float(np.sum((y_ok - y_hat) ** 2))
        R2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else float("nan")
        mono   = all(σ_vals[ok][i] <= σ_vals[ok][i + 1] + 1e-6
                     for i in range(ok.sum() - 1))
        fit_rows_b.append({
            "ticker":              tkr,
            "C1_pct2":             round(C1 * 1e4, 6),
            "C2_pct2_per_day":     round(C2 * 1e4, 8),
            "R2":                  round(R2, 4),
            "monotone":            mono,
            "C2_positive":         C2 > 0,
        })

    test_b_fit = pd.DataFrame(fit_rows_b)
    if not test_b_fit.empty:
        n_b       = len(test_b_fit)
        pct_c2    = round(float(test_b_fit["C2_positive"].sum()) / n_b * 100, 1)
        pct_mono  = round(float(test_b_fit["monotone"].sum())    / n_b * 100, 1)
        mean_c2   = round(float(test_b_fit["C2_pct2_per_day"].mean()), 6)
        mean_r2   = round(float(test_b_fit["R2"].mean()), 3)
    else:
        pct_c2 = pct_mono = mean_c2 = mean_r2 = float("nan")

    test_b_summary = pd.DataFrame([{
        "n_tickers":        len(test_b_fit),
        "pct_C2_positive_%": pct_c2,
        "pct_monotone_%":   pct_mono,
        "mean_C2_pct2_per_day": mean_c2,
        "mean_R2":          mean_r2,
        "conclusion": (
            f"C2>0 in {pct_c2:.0f}% tickers, monotone in {pct_mono:.0f}% → "
            + ("std-vs-tenor pattern robust" if pct_c2 >= 70 and pct_mono >= 60
               else "std-vs-tenor pattern NOT fully robust across tickers")
        ),
    }])

    # ── Verbose output ─────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{_SEP}")
        print("  Test A (cross-stock) — std(HE/S) peaks at ATM across tickers")
        print(_SEP)
        print("\n  Mean σ_HE% across all tickers (rows=tenors, cols=labels):")
        print(std_heatmap.round(3).to_string())
        print("\n  ATM-peak robustness per ticker (pass = ATM peak ≥ 60% of tenors):")
        print(test_a_robust.to_string(index=False))
        print()
        print(test_a_summary.to_string(index=False))

        print(f"\n{_SEP}")
        print("  Test B (cross-stock) — std increases with tenor  (ATM label)")
        print(_SEP)
        print("\n  Per-ticker OLS fit:  std²(T) = C1 + C2·T  (ATM label)")
        print(test_b_fit.to_string(index=False))
        print()
        print(test_b_summary.to_string(index=False))

    return {
        "std_heatmap":    std_heatmap,
        "test_a_robust":  test_a_robust,
        "test_a_summary": test_a_summary,
        "test_b_fit":     test_b_fit,
        "test_b_summary": test_b_summary,
    }


# ── combined cross-section runner ─────────────────────────────────────────────

def run_cross_section_analysis(
        results_dir:  str,
        tickers:      list[str],
        strategy:     str,
        tenors_days:  tuple | list,
        q_grid:       np.ndarray,
        verbose:      bool = True,
) -> dict:
    """
    Load he_detail for all tickers and run Tests A_xs, B_xs, C, D, E.

    Returns dict with keys:
        he_dict, std_pattern, bias_df, corr_df, quintile_df
    """
    t0 = time.time()
    if verbose:
        print(f"\n{_SEP}")
        print("  Cross-Section Analysis  (Tests A_xs · B_xs · C · D · E)")
        print(f"  Tickers  : {len(tickers)} stocks  (ordered small → large mktcap)")
        print(f"  Strategy : {strategy}")
        print(_SEP)

    he_dict = _load_he_all(results_dir, tickers, strategy)
    loaded  = list(he_dict.keys())
    if verbose:
        print(f"\n  Loaded {len(loaded)}/{len(tickers)} tickers")
        missing = sorted(set(tickers) - set(loaded))
        if missing:
            print(f"  Missing : {missing}")

    std_pattern       = analyze_cross_std_pattern(he_dict, tenors_days, q_grid,
                                                   verbose=verbose)
    bias_df           = analyze_cross_bias(he_dict, tenors_days, q_grid, verbose)
    corr_df, quint_df = analyze_he_vs_mktcap(he_dict, tickers, tenors_days,
                                              q_grid, verbose)

    if verbose:
        print(f"\n  [Cross-section total: {time.time()-t0:.1f}s]")

    return {
        "he_dict":      he_dict,
        "std_pattern":  std_pattern,
        "bias_df":      bias_df,
        "corr_df":      corr_df,
        "quintile_df":  quint_df,
    }


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    results_dir = os.path.join(_ROOT, "Results")
    _s          = f"_{STRATEGY_NAME}"

    # ── Part 1: single-ticker analysis (config.TICKER) ───────────────────────
    print(f"\n{_SEP}")
    print(f"  Loading Stage 3 checkpoint for {TICKER} …")
    he_stats = _load_stage3(results_dir)
    print(f"  he_stats loaded : {len(he_stats)} nodes  "
          f"(tenors: {sorted(he_stats.index.get_level_values('T_days').unique())})")

    required = {"HE_per_S_mean", "HE_per_S_std"}
    missing  = required - set(he_stats.columns)
    if missing:
        print(f"[ERROR] he_stats checkpoint is missing columns: {missing}")
        print("        Please re-run stage3_stats.py first.")
        sys.exit(1)

    result = run_stage5(he_stats, verbose=True)

    print("\n── Saving Stage 5 Single-Ticker Outputs ────────────────────────")
    result["std_vs_quantile"].to_csv(
        os.path.join(results_dir, f"{TICKER}{_s}_std_vs_quantile.csv"))
    result["mean_vs_quantile"].to_csv(
        os.path.join(results_dir, f"{TICKER}{_s}_mean_vs_quantile.csv"))
    result["std_vs_tenor"].to_csv(
        os.path.join(results_dir, f"{TICKER}{_s}_std_vs_tenor.csv"))
    result["fit_table"].to_csv(
        os.path.join(results_dir, f"{TICKER}{_s}_std_tenor_fit.csv"))
    print(f"  [Saved] {TICKER}{_s}_std_vs_quantile.csv")
    print(f"  [Saved] {TICKER}{_s}_mean_vs_quantile.csv")
    print(f"  [Saved] {TICKER}{_s}_std_vs_tenor.csv")
    print(f"  [Saved] {TICKER}{_s}_std_tenor_fit.csv")
    print(_SEP)

    # ── Part 2: cross-section analysis over CROSS_TICKERS ────────────────────
    xs = run_cross_section_analysis(
        results_dir  = results_dir,
        tickers      = CROSS_TICKERS,
        strategy     = STRATEGY_NAME,
        tenors_days  = TENORS_DAYS,
        q_grid       = Q_GRID,
        verbose      = True,
    )

    print("\n── Saving Cross-Section Outputs ────────────────────────────────")

    # ── Test A_xs + B_xs outputs ─────────────────────────────────────────────
    sp = xs["std_pattern"]

    p = os.path.join(results_dir, "cross_std_heatmap.csv")
    sp["std_heatmap"].to_csv(p)
    print(f"  [Saved] cross_std_heatmap.csv  "
          f"(mean σ_HE% across tickers, rows=tenors × cols=labels)")

    if not sp["test_a_robust"].empty:
        p = os.path.join(results_dir, "cross_test_a_robust.csv")
        sp["test_a_robust"].to_csv(p, index=False)
        print(f"  [Saved] cross_test_a_robust.csv  "
              f"(per-ticker ATM-peak check)")

        p = os.path.join(results_dir, "cross_test_a_summary.csv")
        sp["test_a_summary"].to_csv(p, index=False)
        print(f"  [Saved] cross_test_a_summary.csv  "
              f"(aggregate ATM-peak robustness: "
              f"{sp['test_a_summary']['pct_pass_%'].iloc[0]:.0f}% tickers pass)")

    if not sp["test_b_fit"].empty:
        p = os.path.join(results_dir, "cross_test_b_atm_fit.csv")
        sp["test_b_fit"].to_csv(p, index=False)
        print(f"  [Saved] cross_test_b_atm_fit.csv  "
              f"(per-ticker ATM OLS fit: C1, C2, R², monotone)")

        p = os.path.join(results_dir, "cross_test_b_summary.csv")
        sp["test_b_summary"].to_csv(p, index=False)
        pct_c2   = sp["test_b_summary"]["pct_C2_positive_%"].iloc[0]
        pct_mono = sp["test_b_summary"]["pct_monotone_%"].iloc[0]
        print(f"  [Saved] cross_test_b_summary.csv  "
              f"(C2>0 in {pct_c2:.0f}%, monotone in {pct_mono:.0f}% of tickers)")

    if not xs["bias_df"].empty:
        df_bias = xs["bias_df"].copy()

        # 1. Long-format full table (all columns)
        p = os.path.join(results_dir, "cross_he_bias.csv")
        df_bias.to_csv(p, index=False)
        print(f"  [Saved] cross_he_bias.csv  ({len(df_bias)} rows, long format)")

        # 2. cross_mean_% heatmap  (T_days × q)
        pivot_m = df_bias.pivot(index="T_days", columns="q", values="cross_mean_%")
        pivot_m.index.name   = "tenor_days"
        pivot_m.columns.name = "q"
        p = os.path.join(results_dir, "cross_he_mean_heatmap.csv")
        pivot_m.round(4).to_csv(p)
        print(f"  [Saved] cross_he_mean_heatmap.csv  "
              f"({pivot_m.shape[0]} tenors × {pivot_m.shape[1]} q nodes)")

        # 3. t-stat heatmap  (T_days × q)
        pivot_t = df_bias.pivot(index="T_days", columns="q", values="t_stat")
        pivot_t.index.name   = "tenor_days"
        pivot_t.columns.name = "q"
        p = os.path.join(results_dir, "cross_he_tstat_heatmap.csv")
        pivot_t.round(3).to_csv(p)
        print(f"  [Saved] cross_he_tstat_heatmap.csv")

        # 4. Bias sign map  (++ / -- / ·) using shared _bias_sign helper
        df_bias["bias_sign"] = df_bias.apply(
            lambda row: _bias_sign(float(row["cross_mean_%"]), row["t_stat"]),
            axis=1,
        )
        pivot_s = df_bias.pivot(index="T_days", columns="q", values="bias_sign")
        pivot_s.index.name   = "tenor_days"
        pivot_s.columns.name = "q"
        p = os.path.join(results_dir, "cross_he_signmap.csv")
        pivot_s.to_csv(p)
        print(f"  [Saved] cross_he_signmap.csv  "
              f"(++ seller profitable, -- seller losing, · not sig.)")

    # ── Test D outputs ───────────────────────────────────────────────────────
    if not xs["corr_df"].empty:
        df_corr = xs["corr_df"].copy()

        # 1. Long-format full table
        p = os.path.join(results_dir, "cross_spearman_mktcap.csv")
        df_corr.to_csv(p, index=False)
        print(f"  [Saved] cross_spearman_mktcap.csv  ({len(df_corr)} rows, long format)")

        # 2. ρ heatmap with * marking significant nodes  (|T_days| × q)
        #    Cell format: "−0.81 *" for sig, "−0.81" for not sig
        def _rho_cell(row: Any) -> str:
            rho = float(row["spearman_std"])
            sig = bool(row["sig_std"])
            return f"{rho:+.2f} *" if sig else f"{rho:+.2f}"

        df_corr["rho_cell"] = df_corr.apply(_rho_cell, axis=1)
        pivot_rho = df_corr.pivot(index="T_days", columns="q", values="rho_cell")
        pivot_rho.index.name   = "tenor_days"
        pivot_rho.columns.name = "q"
        p = os.path.join(results_dir, "cross_spearman_rho_heatmap.csv")
        pivot_rho.to_csv(p)
        print(f"  [Saved] cross_spearman_rho_heatmap.csv  "
              f"(ρ heatmap, * = p < 0.10 significant)")

        # 3. Per-tenor summary: avg_rho, % nodes negative, n significant-negative
        #    → one row per tenor → immediate conclusion on size effect direction
        tenor_rows = []
        for T in sorted(df_corr["T_days"].unique()):
            sub = df_corr[df_corr["T_days"] == T]
            n_nodes     = len(sub)
            avg_rho     = round(float(sub["spearman_std"].mean()), 3)
            pct_neg     = round(float((sub["spearman_std"] < 0).sum()) / n_nodes * 100, 1)
            n_sig_neg   = int(((sub["spearman_std"] < 0) & sub["sig_std"]).sum())
            conclusion  = (
                "larger cap → lower HE std" if pct_neg >= 70
                else ("larger cap → higher HE std" if pct_neg <= 30
                      else "mixed / no clear pattern")
            )
            tenor_rows.append({
                "tenor_days":       T,
                "n_q_nodes":        n_nodes,
                "avg_rho":          avg_rho,
                "pct_nodes_neg_%":  pct_neg,
                "n_sig_neg_nodes":  n_sig_neg,
                "conclusion":       conclusion,
            })
        df_tenor_sum = pd.DataFrame(tenor_rows)
        p = os.path.join(results_dir, "cross_spearman_tenor_summary.csv")
        df_tenor_sum.to_csv(p, index=False)
        print(f"  [Saved] cross_spearman_tenor_summary.csv  "
              f"(per-tenor avg ρ, % negative, conclusion)")

    # ── Test E outputs ───────────────────────────────────────────────────────
    if not xs["quintile_df"].empty:
        df_quint = xs["quintile_df"].copy()

        # 1. Long-format full table
        p = os.path.join(results_dir, "cross_quintile_std.csv")
        df_quint.to_csv(p, index=False)
        print(f"  [Saved] cross_quintile_std.csv  ({len(df_quint)} rows, long format)")

        # 2. ATM-only wide table: rows = quintile, cols = tenor
        #    Makes the Q1 > Q2 > Q3 > Q4 staircase immediately visible
        atm_quint = df_quint[df_quint["q"] == -1.0].copy()
        q_cols    = [f"Q{i}_avg_std_%" for i in [1, 2, 3, 4]]
        if not atm_quint.empty:
            # Pivot: rows = Q1…Q4, cols = tenors
            wide = atm_quint.set_index("T_days")[q_cols].T
            wide.index = pd.Index(
                ["Q1_smallest", "Q2", "Q3", "Q4_largest"], name="quintile"
            )
            # Append Q1→Q4 decline % column
            for T in wide.columns:
                q1 = float(wide.loc["Q1_smallest", T])
                q4 = float(wide.loc["Q4_largest",  T])
                wide.loc["Q1_minus_Q4_decline_%", T] = (
                    round((q1 - q4) / q1 * 100, 1) if q1 > 0 else float("nan")
                )
            p = os.path.join(results_dir, "cross_quintile_wide_atm.csv")
            wide.round(4).to_csv(p)
            print(f"  [Saved] cross_quintile_wide_atm.csv  "
                  f"(ATM only, rows=Q1→Q4+decline, cols=tenors)")

        # 3. (Q1 − Q4) / Q1 × 100 decline heatmap for ALL (T, q) nodes
        #    Positive = smaller cap has higher HE std than larger cap
        def _decline(row: Any) -> float:
            q1 = float(row["Q1_avg_std_%"]) if pd.notna(row["Q1_avg_std_%"]) else float("nan")
            q4 = float(row["Q4_avg_std_%"]) if pd.notna(row["Q4_avg_std_%"]) else float("nan")
            if np.isfinite(q1) and np.isfinite(q4) and q1 > 0:
                return round((q1 - q4) / q1 * 100, 1)
            return float("nan")

        df_quint["Q1_Q4_decline_%"] = df_quint.apply(_decline, axis=1)
        pivot_dec = df_quint.pivot(index="T_days", columns="q",
                                   values="Q1_Q4_decline_%")
        pivot_dec.index.name   = "tenor_days"
        pivot_dec.columns.name = "q"
        p = os.path.join(results_dir, "cross_quintile_q1q4_decline_heatmap.csv")
        pivot_dec.to_csv(p)
        print(f"  [Saved] cross_quintile_q1q4_decline_heatmap.csv  "
              f"((Q1−Q4)/Q1 × 100, positive = smaller cap has higher HE std)")

    print(_SEP)
