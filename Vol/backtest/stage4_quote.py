"""
stage4_quote.py
===============
Stage 4 — Bid / Mid / Ask Price Surface   [S4-10, S4-10b, S4-11, S4-12]

Three-step quoting engine.  All tunable parameters live in config.py.

Separation of concerns
----------------------
  μ_HE (systematic bias in model pricing)  →  handled by S4-10b (mid correction)
  σ_HE (per-trade hedge uncertainty)        →  handled by S4-11  (spread width)

Pipeline
--------
  S4-10  │ MC repricing with jump-param adjustments  →  _build_adj_iv_surface()
  ───────┤ For EACH quote side (ask / bid):
         │   1. Adjust base jump params by config multipliers:
         │        lam     ← lam     × S4_{SIDE}_LAM_MULT
         │        mu_j    ← mu_j    + S4_{SIDE}_MU_J_ADD
         │        sigma_j ← sigma_j × S4_{SIDE}_SIGMA_J_MULT
         │   2. Generate N=S4_N_PATHS random paths from (S_t, h_t).
         │   3. Price options at the FAIR K-grid (same strikes as Stage 2).
         │   4. Invert MC prices to raw IV; fit eSSVI to smooth.
         │   → iv_ask_mc, iv_bid_mc  (step-1 IV surfaces)
         │
         │ Step-1 spread = p_ask_mc − p_bid_mc  (pure jump-param effect)
         │
  S4-10b │ HE bias correction  →  _apply_he_bias_correction()
  ───────┤ Shifts the MID vol (and both ask_mc / bid_mc identically) to correct
         │ for the systematic mispricing identified in Stage 3.
         │
         │   Δσ_bias[T,q] = clip( −S4_HE_BIAS_COEF × μ_HE / vega_eff,
         │                         −S4_HE_BIAS_CLIP, +S4_HE_BIAS_CLIP )
         │   iv_mid_adj   = iv_model  + Δσ_bias     ← bias-corrected fair mid
         │   iv_ask_mc_adj = iv_ask_mc + Δσ_bias    ← preserves step-1 spread
         │   iv_bid_mc_adj = iv_bid_mc + Δσ_bias
         │
         │ Sign:  μ_HE > 0 (overpriced) → Δσ < 0 → lower mid vol
         │        μ_HE < 0 (underpriced) → Δσ > 0 → raise mid vol
         │
  S4-11  │ HE vol-space spread (σ_HE ONLY)  →  _apply_he_spread()
  ───────┤ Per-node σ_HE converted to vol-point spread via regularised vega.
         │ μ_HE is NOT used here (fully absorbed by S4-10b).
         │
         │   vega_eff(T,K) = √[ (φ(d₁)·√T)²  +  ε(T)² ]
         │       ε(T) = S4_VEGA_REG_ALPHA × 0.4 × √T_years
         │
         │   Δσ_ask = max( S4_ASK_SIGMA_COEF × σ_HE / vega_eff,  0 )
         │   Δσ_bid = max( S4_BID_SIGMA_COEF × σ_HE / vega_eff,  0 )
         │   σ_ask_final = σ_ask_mc_adj + Δσ_ask   →  BS  →  p_ask
         │   σ_bid_final = σ_bid_mc_adj − Δσ_bid   →  BS  →  p_bid (floored)
         │
  S4-12  │ Butterfly arb check + fix  (Method B)  →  _fix_butterfly_arb()
  ───────┤ Per tenor: p(K₁) + p(K₃) ≥ 2·p(K₂).
         │ Fix: p(K₂) ← (p(K₁) + p(K₃)) / 2 − ε.
         │ Applied to both ask and bid surfaces independently.

All parameters (jump multipliers, bias coef/clip, HE-spread coefficients,
arb-fix settings, TARGET_POINTS) are set in config.py — no edits needed here.

Reads  (Results/)
-----------------
  {TICKER}_model.json              ← GARCH+Jump params       (Stage 1 → S1-03)
  {TICKER}_h_series.csv            ← GARCH filter states     (Stage 1 → S1-04)
  {TICKER}_{STRAT}_iv_surface.csv  ← fair IV surface         (Stage 2 → S2-07)
  {TICKER}_{STRAT}_k_surface.csv   ← strike grid per tenor   (Stage 2 → S2-07)
  {TICKER}_{STRAT}_he_detail.csv   ← per-node HE stats       (Stage 3 → S3-10)

Outputs  (Results/)
-------------------
  {TICKER}_{STRAT}_price_surface.csv
      MultiIndex table (tenor_days × metric).
      metric ∈ {K_over_S, p_fair, p_ask_mc, p_bid_mc,
                p_ask, p_bid, spread, s1_spread, s2_spread}

Run
---
    cd project_updates
    python3 Vol/backtest/stage4_quote.py
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
    DATA_DIR, PRINT_VOL_SURFACE_DATE,
    R_ANNUAL, Q_ANNUAL, TRADING_DAYS, SEED,
    TENORS_DAYS, Q_GRID,
    S4_ASK_LAM_MULT, S4_ASK_MU_J_ADD, S4_ASK_SIGMA_J_MULT,
    S4_BID_LAM_MULT, S4_BID_MU_J_ADD, S4_BID_SIGMA_J_MULT,
    S4_N_PATHS,
    S4_HE_BIAS_COEF, S4_HE_BIAS_CLIP,
    S4_VEGA_REG_ALPHA,
    S4_ASK_SIGMA_COEF,
    S4_BID_SIGMA_COEF,
    S4_SPREAD_FLOOR_COEF,
    P_BID_MIN,
    ARB_FIX_MAX_ITERS, ARB_FIX_EPS,
    TARGET_POINTS,
)
from backtest.trading    import STRATEGY_NAME
from backtest.data       import _load_price_df
from backtest.pricing    import _bs_call, _bs_put, _invert_iv, _bs_vega
from scipy.stats         import norm as _scipy_norm
from backtest.simulation import _make_rand_block, _simulate_from_block
from backtest.vol_surface import _calibrate_essvi_and_extract

from backtest.stage1_fit      import load_checkpoint as _load_stage1
from backtest.stage2_backtest import load_checkpoint as _load_stage2
from backtest.stage3_stats    import load_checkpoint as _load_stage3

_SEP  = "═" * 65
_SEP2 = "─" * 65


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _label_to_q(label: str) -> float:
    if label == "ATM":
        return -1.0
    return int(label[1:]) / 100.0


def _price_strategy(sigma: float, S: float, K: float, T_years: float,
                    strategy: str,
                    r: float = R_ANNUAL, q: float = Q_ANNUAL) -> float:
    if strategy == "short_call":
        return _bs_call(S, K, T_years, sigma, r, q)
    if strategy == "short_put":
        return _bs_put(S, K, T_years, sigma, r, q)
    return _bs_call(S, K, T_years, sigma, r, q) + _bs_put(S, K, T_years, sigma, r, q)


def _intrinsic(S: float, K: float, strategy: str) -> float:
    """Intrinsic value floor for the bid price."""
    if strategy == "short_call":
        return max(S - K, 0.0)
    if strategy == "short_put":
        return max(K - S, 0.0)
    return abs(S - K)


def _nearest_trading_date(price_df: pd.DataFrame, target: str) -> str:
    ts      = pd.Timestamp(target)
    dates   = list(price_df["date"])
    earlier = [d for d in dates if d <= ts]
    chosen  = earlier[-1] if earlier else dates[0]
    return str(pd.Timestamp(chosen).date())


# ═══════════════════════════════════════════════════════════════════════════════
#  S4-10 HELPERS: JUMP ADJUSTMENT + MC REPRICING
# ═══════════════════════════════════════════════════════════════════════════════

def _adj_jump_params(
        base: dict,
        lam_mult:     float,
        mu_j_add:     float,
        sigma_j_mult: float,
) -> dict:
    """Return a new jump_params dict with adjusted lambda / mu_j / sigma_j."""
    return {
        "lam":     base["lam"]     * lam_mult,
        "mu_j":    base["mu_j"]    + mu_j_add,
        "sigma_j": base["sigma_j"] * sigma_j_mult,
    }


def _get_S_h(
        pricing_date: str,
        price_df:     pd.DataFrame,
        h_series:     pd.Series,  # type: ignore[type-arg]
        garch_params: dict,
) -> tuple[float, float]:
    """
    Return (S_t, h_t) for the given pricing date.

    h_t = omega + alpha * r_t² + beta * h_pre, where r_t = log(S_t / S_prev).
    Falls back to the last available h_series value if the date is not found.
    """
    date_ts = pd.Timestamp(pricing_date)
    omega   = garch_params["omega"]
    alpha   = garch_params["alpha"]
    beta    = garch_params["beta"]

    # h_pre = GARCH state that prevailed at the START of pricing_date
    if date_ts in h_series.index:
        h_pre = float(h_series[date_ts])
    else:
        avail = h_series[h_series.index <= date_ts]  # type: ignore[operator]
        h_pre = float(avail.iloc[-1]) if not avail.empty else float(h_series.iloc[0])  # type: ignore[union-attr]

    # S_t and one-step h update using the return on pricing_date
    avail_px = price_df[price_df["date"] <= date_ts].sort_values("date")  # type: ignore[call-overload]
    S_t = float(avail_px["close"].iloc[-1])
    if len(avail_px) >= 2:
        S_prev = float(avail_px["close"].iloc[-2])
        r_t    = np.log(S_t / S_prev)
        h_t    = omega + alpha * r_t ** 2 + beta * h_pre
    else:
        h_t = h_pre

    return S_t, h_t


def _build_adj_iv_surface(
        S_t:          float,
        h_t:          float,
        k_surface_df: pd.DataFrame,   # fair K-grid (log-moneyness)
        garch_params: dict,
        adj_jump:     dict,
        n_paths:      int   = S4_N_PATHS,
        r:            float = R_ANNUAL,
        q:            float = Q_ANNUAL,
        trading_days: int   = TRADING_DAYS,
        seed:         int   = SEED,
) -> pd.DataFrame:
    """
    Build a raw IV surface using adjusted jump params at the FIXED K-grid
    from k_surface_df, then smooth with eSSVI.

    Steps
    -----
    1. Generate rand_block of shape (T_max, n_paths) using adj_jump.
    2. For each tenor T: simulate S_T paths, price calls at FAIR K-values,
       invert MC prices to IV.
    3. Fit eSSVI to the raw IV surface (same k-grid for all three surfaces).

    Returns
    -------
    iv_df : DataFrame  (tenor_days × label columns) — eSSVI-smoothed IV
    """
    tenors = list(k_surface_df.index)
    labels = list(k_surface_df.columns)
    T_max  = max(int(t) for t in tenors)

    lam     = adj_jump["lam"]
    mu_j    = adj_jump["mu_j"]
    sigma_j = adj_jump["sigma_j"]

    block = _make_rand_block(T_max, n_paths, lam, mu_j, sigma_j, seed=seed)

    iv_rows: dict[int, list[float]] = {}
    k_grid:  dict[int, np.ndarray]  = {}

    for T_days in tenors:
        T_int    = int(T_days)
        T_years  = T_int / trading_days
        discount = np.exp(-r * T_years)

        sub_block = {
            "Z": block["Z"][:T_int],
            "J": block["J"][:T_int],
        }
        S_T = _simulate_from_block(
            S_t, h_t, T_int, sub_block,
            garch_params, adj_jump, r, q,
        )

        iv_row: list[float] = []
        k_row:  list[float] = []
        for lbl in labels:
            k_val    = float(k_surface_df.loc[T_days, lbl])
            K        = S_t * np.exp(k_val)
            payoffs  = np.maximum(S_T - K, 0.0)
            price_mc = float(np.mean(payoffs)) * discount
            iv_val   = _invert_iv(price_mc, S_t, K, T_years, r, q)
            iv_row.append(iv_val if np.isfinite(iv_val) and iv_val > 0 else float("nan"))
            k_row.append(k_val)

        iv_rows[T_int] = iv_row
        k_grid[T_int]  = np.array(k_row)

    raw_iv = pd.DataFrame(iv_rows, index=labels).T  # type: ignore[arg-type]
    raw_iv.index.name   = "tenor_days"
    raw_iv.columns.name = "q"

    iv_fitted, _, _ = _calibrate_essvi_and_extract(raw_iv, k_grid, verbose=False)
    return iv_fitted


# ═══════════════════════════════════════════════════════════════════════════════
#  S4-10 → PRICES: Convert IV surface to price surface
# ═══════════════════════════════════════════════════════════════════════════════

def _prices_from_iv(
        iv_df:        pd.DataFrame,
        k_surface_df: pd.DataFrame,
        S:            float,
        strategy:     str,
        r:            float = R_ANNUAL,
        q:            float = Q_ANNUAL,
        trading_days: int   = TRADING_DAYS,
) -> pd.DataFrame:
    """
    Convert an IV surface to option prices via BS formula.

    Returns DataFrame with same index/columns as iv_df.
    """
    tenors = list(iv_df.index)
    labels = list(iv_df.columns)
    rows: dict[int, dict[str, float]] = {}

    for T_days in tenors:
        T_years = int(T_days) / trading_days
        row: dict[str, float] = {}
        for lbl in labels:
            sigma = float(iv_df.loc[T_days, lbl])
            if not (np.isfinite(sigma) and sigma > 0):
                row[lbl] = float("nan")
                continue
            k_val  = float(k_surface_df.loc[T_days, lbl])
            K      = S * np.exp(k_val)
            row[lbl] = round(_price_strategy(sigma, S, K, T_years, strategy, r, q), 6)
        rows[int(T_days)] = row

    df = pd.DataFrame(rows).T.reindex(columns=labels)
    df.index        = pd.Index([int(t) for t in tenors], name="tenor_days")
    df.columns.name = "label"
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  S4-10b: HE BIAS CORRECTION  (mid-vol shift)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_he_bias_correction(
        iv_df:        pd.DataFrame,   # IV surface to shift (decimal, any of mid/ask/bid)
        he_detail:    pd.DataFrame,   # (T_days, q) × {HE_per_S_mean, HE_per_S_std}
        k_surface_df: pd.DataFrame,   # log-moneyness grid (fair K-grid)
        S:            float,
        bias_coef:    float = S4_HE_BIAS_COEF,
        bias_clip:    float = S4_HE_BIAS_CLIP,
        r:            float = R_ANNUAL,
        q:            float = Q_ANNUAL,
        trading_days: int   = TRADING_DAYS,
) -> pd.DataFrame:
    """
    Shift every node of iv_df by the HE-implied bias correction.

    For each (T_days, label) node:
        Δσ_bias = clip( −bias_coef × μ_HE_per_S / vega_eff,
                        −bias_clip, +bias_clip )
        iv_adj  = max( iv + Δσ_bias, 1e-4 )

    Sign convention (seller perspective, HE = premium − payoff + hedge_pnl):
        μ_HE > 0  →  seller systematically overcharged
                  →  model vol was too HIGH  →  Δσ_bias < 0  (lower the mid)
        μ_HE < 0  →  seller systematically undercharged
                  →  model vol was too LOW   →  Δσ_bias > 0  (raise the mid)

    The same shift is applied identically to the ask_mc and bid_mc surfaces so
    the step-1 spread is preserved and only the centre of the market is moved.

    Parameters
    ----------
    iv_df      : IV surface (decimal) — typically iv_surface_df, iv_ask_mc, or
                 iv_bid_mc, all shifted by the *same* per-node Δσ_bias.
    bias_coef  : fraction of μ_HE to correct (0 = off, 1 = full correction).
    bias_clip  : hard cap on |Δσ_bias| in vol-point units (e.g. 0.10 = 10 vpts).

    Returns
    -------
    iv_adj : DataFrame — same shape / index as iv_df, bias-corrected.
    """
    tenors = list(iv_df.index)
    labels = list(iv_df.columns)

    mu_fallback = float(he_detail["HE_per_S_mean"].mean())
    adj_rows: dict[int, dict[str, float]] = {}

    for T_days in tenors:
        T_years = int(T_days) / trading_days
        row: dict[str, float] = {}

        for lbl in labels:
            q_val = _label_to_q(lbl)

            try:
                mu_he = float(he_detail.loc[(int(T_days), q_val), "HE_per_S_mean"])
            except KeyError:
                mu_he = mu_fallback

            k_val     = float(k_surface_df.loc[T_days, lbl])
            sigma_cur = float(iv_df.loc[T_days, lbl])
            v_eff     = _vega_eff(sigma_cur, k_val, T_years, r, q)

            # negative sign: overpricing (μ_HE > 0) → lower vol
            delta = float(np.clip(-bias_coef * mu_he / v_eff, -bias_clip, bias_clip))
            row[lbl] = max(sigma_cur + delta, 1e-4)

        adj_rows[int(T_days)] = row

    result = pd.DataFrame(adj_rows).T.reindex(columns=labels)
    result.index        = pd.Index([int(t) for t in tenors], name="tenor_days")
    result.columns.name = "label"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  S4-11: HE-BASED SPREAD OVERLAY  (σ_HE only)
# ═══════════════════════════════════════════════════════════════════════════════

def _vega_eff(
        sigma_mid: float,
        k_val:     float,
        T_years:   float,
        r:         float = R_ANNUAL,
        q:         float = Q_ANNUAL,
        alpha:     float = S4_VEGA_REG_ALPHA,
) -> float:
    """
    Regularised normalised vega:
        vega_eff = √[ (φ(d₁)·√T)²  +  ε(T)² ]
        ε(T)    = alpha × 0.4 × √T   (≈ alpha × ATM vega for a 40%-vol stock)

    Returns vega_eff using σ_mid (fair IV, decimal) and k = log(K/S).
    """
    if not (np.isfinite(sigma_mid) and sigma_mid > 0 and T_years > 0):
        return alpha * 0.4 * max(np.sqrt(T_years), 1e-6)

    # d₁ formula with log(S/K) = -k
    vt = sigma_mid * np.sqrt(T_years)
    d1 = (-k_val + (r - q + 0.5 * sigma_mid ** 2) * T_years) / vt
    vega_per_S = np.sqrt(T_years) * float(_scipy_norm.pdf(d1))
    eps        = alpha * 0.4 * np.sqrt(T_years)
    return float(np.sqrt(vega_per_S ** 2 + eps ** 2))


def _vega_eps(T_years: float, alpha: float = S4_VEGA_REG_ALPHA) -> float:
    """Regularisation term ε(T) = alpha × 0.4 × √T_years."""
    return alpha * 0.4 * np.sqrt(max(T_years, 1e-9))


def _apply_he_spread(
        iv_ask_mc:    pd.DataFrame,   # bias-corrected ask IV (S4-10b output, decimal)
        iv_bid_mc:    pd.DataFrame,   # bias-corrected bid IV (S4-10b output, decimal)
        iv_mid:       pd.DataFrame,   # bias-corrected mid IV (S4-10b output, decimal)
        he_detail:    pd.DataFrame,   # (T_days, q) × {HE_per_S_mean, HE_per_S_std}
        k_surface_df: pd.DataFrame,   # fair K-grid log-moneyness
        S:            float,
        strategy:     str,
        r:            float = R_ANNUAL,
        q:            float = Q_ANNUAL,
        trading_days: int   = TRADING_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply vol-space HE spread on top of bias-corrected IV surfaces (S4-11).

    Uses σ_HE ONLY to determine spread width.  μ_HE is handled upstream in
    S4-10b (_apply_he_bias_correction) and must NOT be reused here to avoid
    double-counting the systematic bias.

    For each (T_days, label) node:
        vega_eff = regularised normalised vega (see _vega_eff)
        ε(T)     = S4_VEGA_REG_ALPHA × 0.4 × √T_years  (regularisation floor)

        Δσ_HE_ask = S4_ASK_SIGMA_COEF × σ_HE / vega_eff
        Δσ_HE_bid = S4_BID_SIGMA_COEF × σ_HE / vega_eff

        Δσ_floor  = S4_SPREAD_FLOOR_COEF × ε(T) / vega_eff
            — OTM floor: at ATM, vega_eff ≈ φ(0)·√T >> ε, so floor ≈ 0.
              Deep OTM: vega_eff → ε, so floor → S4_SPREAD_FLOOR_COEF (constant).
            — This ensures the vol-spread widens toward the wings instead of
              being roughly flat (which happens when σ_HE ≈ const × vega_eff).

        Δσ_ask = max( Δσ_HE_ask + Δσ_floor,  0 )
        Δσ_bid = max( Δσ_HE_bid + Δσ_floor,  0 )

        σ_ask_final = σ_ask_mc + Δσ_ask
        σ_bid_final = max(σ_bid_mc − Δσ_bid,  1e-4)

    Bid price additionally floored at max(intrinsic, P_BID_MIN × S).
    Falls back to cross-node σ_HE average if a (T_days, q) key is missing.
    """
    tenors = list(iv_ask_mc.index)
    labels = list(iv_ask_mc.columns)

    std_fallback = float(he_detail["HE_per_S_std"].mean())

    ask_rows:    dict[int, dict[str, float]] = {}
    bid_rows:    dict[int, dict[str, float]] = {}
    iv_ask_rows: dict[int, dict[str, float]] = {}   # final ask IV (decimal)
    iv_bid_rows: dict[int, dict[str, float]] = {}   # final bid IV (decimal)

    for T_days in tenors:
        T_years = int(T_days) / trading_days
        ask_row:    dict[str, float] = {}
        bid_row:    dict[str, float] = {}
        iv_ask_row: dict[str, float] = {}
        iv_bid_row: dict[str, float] = {}

        for lbl in labels:
            q_val = _label_to_q(lbl)

            try:
                std_he = float(he_detail.loc[(int(T_days), q_val), "HE_per_S_std"])
            except KeyError:
                std_he = std_fallback

            k_val     = float(k_surface_df.loc[T_days, lbl])
            sigma_mid = float(iv_mid.loc[T_days, lbl]) if lbl in iv_mid.columns else float("nan")
            v_eff     = _vega_eff(sigma_mid, k_val, T_years, r, q)
            eps       = _vega_eps(T_years)

            # σ_HE-based component — μ_HE fully absorbed by S4-10b bias correction
            he_ask = S4_ASK_SIGMA_COEF * std_he / v_eff
            he_bid = S4_BID_SIGMA_COEF * std_he / v_eff

            # OTM intrinsic floor: grows as vega_eff→ε (i.e. deep OTM)
            floor = S4_SPREAD_FLOOR_COEF * eps / v_eff

            delta_ask = max(he_ask + floor, 0.0)
            delta_bid = max(he_bid + floor, 0.0)

            sigma_ask_base = float(iv_ask_mc.loc[T_days, lbl])
            sigma_bid_base = float(iv_bid_mc.loc[T_days, lbl])

            sigma_ask_final = sigma_ask_base + delta_ask
            sigma_bid_final = max(sigma_bid_base - delta_bid, 1e-4)

            K         = S * np.exp(k_val)
            bid_floor = max(_intrinsic(S, K, strategy), P_BID_MIN * S)

            p_ask_val = _price_strategy(sigma_ask_final, S, K, T_years, strategy, r, q)
            p_bid_val = _price_strategy(sigma_bid_final, S, K, T_years, strategy, r, q)
            p_bid_val = max(p_bid_val, bid_floor)

            ask_row[lbl]    = round(p_ask_val,       6)
            bid_row[lbl]    = round(p_bid_val,       6)
            iv_ask_row[lbl] = round(sigma_ask_final, 8)  # decimal
            iv_bid_row[lbl] = round(sigma_bid_final, 8)  # decimal

        ask_rows[int(T_days)]    = ask_row
        bid_rows[int(T_days)]    = bid_row
        iv_ask_rows[int(T_days)] = iv_ask_row
        iv_bid_rows[int(T_days)] = iv_bid_row

    def _to_df(d: dict) -> pd.DataFrame:
        df = pd.DataFrame(d).T.reindex(columns=labels)
        df.index        = pd.Index([int(t) for t in tenors], name="tenor_days")
        df.columns.name = "label"
        return df

    return _to_df(ask_rows), _to_df(bid_rows), _to_df(iv_ask_rows), _to_df(iv_bid_rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  S4-12: BUTTERFLY ARB FIX  (Method B)
# ═══════════════════════════════════════════════════════════════════════════════

def _fix_butterfly_arb(
        p_df:         pd.DataFrame,
        k_surface_df: pd.DataFrame,
        max_iters:    int   = ARB_FIX_MAX_ITERS,
        eps:          float = ARB_FIX_EPS,
) -> tuple[pd.DataFrame, int]:
    """
    Fix butterfly arbitrage violations in a price surface.

    Butterfly arb condition: p(K₁) + p(K₃) ≥ 2·p(K₂)  for K₁ < K₂ < K₃.
    If violated, the middle-strike price is too HIGH relative to its neighbours.

    Fix (Method B): p(K₂) ← (p(K₁) + p(K₃)) / 2 − ε
    Applied per-tenor, iterating up to max_iters passes until all clean.

    Returns
    -------
    fixed_df : corrected price surface (same shape as p_df)
    n_fixed  : total number of corrections across all tenors and passes
    """
    fixed   = p_df.copy()
    n_fixed = 0

    for T_days in fixed.index:
        k_row  = k_surface_df.loc[T_days]
        labels = list(fixed.columns)

        sorted_labels = sorted(
            [l for l in labels if l in k_row.index],
            key=lambda l: float(k_row[l]),
        )
        if len(sorted_labels) < 3:
            continue

        for _ in range(max_iters):
            changed = False
            p_vals  = [float(fixed.loc[T_days, l]) for l in sorted_labels]

            for i in range(1, len(sorted_labels) - 1):
                p1, p2, p3 = p_vals[i - 1], p_vals[i], p_vals[i + 1]
                if not (np.isfinite(p1) and np.isfinite(p2) and np.isfinite(p3)):
                    continue
                if p1 + p3 < 2.0 * p2 - eps:
                    new_p2 = (p1 + p3) / 2.0 - eps
                    new_p2 = max(new_p2, 0.0)
                    fixed.loc[T_days, sorted_labels[i]] = new_p2
                    p_vals[i] = new_p2
                    n_fixed  += 1
                    changed   = True

            if not changed:
                break

    return fixed, n_fixed


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def _format_price_table(
        p_fair:        pd.DataFrame,
        p_fair_adj:    pd.DataFrame,
        p_ask_mc:      pd.DataFrame,
        p_bid_mc:      pd.DataFrame,
        p_ask_mc_adj:  pd.DataFrame,
        p_bid_mc_adj:  pd.DataFrame,
        p_ask:         pd.DataFrame,
        p_bid:         pd.DataFrame,
        k_surface_df:  pd.DataFrame,
        S:             float,
        iv_mid:        pd.DataFrame | None = None,   # Stage-2 fair IV (decimal)
        iv_mid_adj:    pd.DataFrame | None = None,   # S4-10b bias-corrected mid IV
        iv_ask_mc_df:  pd.DataFrame | None = None,   # S4-10 ask IV (decimal)
        iv_bid_mc_df:  pd.DataFrame | None = None,   # S4-10 bid IV (decimal)
        iv_ask_df:     pd.DataFrame | None = None,   # final ask IV (decimal)
        iv_bid_df:     pd.DataFrame | None = None,   # final bid IV (decimal)
) -> pd.DataFrame:
    """
    Build a MultiIndex price surface table for CSV output.

    Row index  : (tenor_days, metric)
    Columns    : moneyness labels (q05 … ATM … q95)

    metric values
    -------------
    K_over_S     — K / S₀
    p_fair       — original Stage-2 fair BS price
    p_fair_adj   — bias-corrected fair price  (S4-10b)
    p_ask_mc     — S4-10 ask price  (jump-adjusted MC, before bias shift)
    p_bid_mc     — S4-10 bid price  (jump-adjusted MC, before bias shift)
    p_ask_mc_adj — S4-10b bias-shifted ask price  (after bias correction)
    p_bid_mc_adj — S4-10b bias-shifted bid price  (after bias correction)
    p_ask        — final ask price  (S4-11 spread + arb-fixed)
    p_bid        — final bid price  (S4-11 spread + arb-fixed)
    spread       — p_ask − p_bid
    s1_spread    — S4-10 jump-adj contribution = p_ask_mc − p_bid_mc
    s1b_spread   — S4-10b bias-shift contribution = p_ask_mc_adj − p_ask_mc
                   (same for ask and bid; represents centre-shift, not spread)
    s2_spread    — S4-11 HE σ-spread contribution = spread − s1_spread − s1b_spread
    iv_model     — Stage-2 fair IV %
    iv_model_adj — bias-corrected mid IV %  (S4-10b)
    iv_ask_mc    — S4-10 ask IV %
    iv_bid_mc    — S4-10 bid IV %
    iv_ask       — final ask IV %
    iv_bid       — final bid IV %
    """
    labels = list(p_fair.columns)
    rows: list[pd.Series] = []  # type: ignore[type-arg]
    include_iv = all(v is not None for v in
                     [iv_mid, iv_mid_adj, iv_ask_mc_df, iv_bid_mc_df,
                      iv_ask_df, iv_bid_df])

    for T_days in p_fair.index:
        k_row = k_surface_df.loc[T_days]

        ks       = [round(np.exp(float(k_row[l])), 4)                                for l in labels]
        fair     = [round(float(p_fair.loc[T_days, l]),        6)                    for l in labels]
        fair_adj = [round(float(p_fair_adj.loc[T_days, l]),    6)                    for l in labels]
        amc      = [round(float(p_ask_mc.loc[T_days, l]),      6)                    for l in labels]
        bmc      = [round(float(p_bid_mc.loc[T_days, l]),      6)                    for l in labels]
        amc_adj  = [round(float(p_ask_mc_adj.loc[T_days, l]),  6)                    for l in labels]
        bmc_adj  = [round(float(p_bid_mc_adj.loc[T_days, l]),  6)                    for l in labels]
        ask      = [round(float(p_ask.loc[T_days, l]),         6)                    for l in labels]
        bid      = [round(float(p_bid.loc[T_days, l]),         6)                    for l in labels]
        spr      = [round(float(p_ask.loc[T_days, l]) - float(p_bid.loc[T_days, l]), 6)
                    for l in labels]
        s1       = [round(float(p_ask_mc.loc[T_days, l]) - float(p_bid_mc.loc[T_days, l]), 6)
                    for l in labels]
        # s1b: bias shift (same direction on both sides — measures centre movement)
        s1b      = [round(float(p_ask_mc_adj.loc[T_days, l]) - float(p_ask_mc.loc[T_days, l]), 6)
                    for l in labels]
        s2       = [round(spr[i] - s1[i] - s1b[i], 6) for i in range(len(labels))]

        metrics_to_emit: list[tuple[str, list]] = [
            ("K_over_S",     ks),
            ("p_fair",       fair),
            ("p_fair_adj",   fair_adj),
            ("p_ask_mc",     amc),
            ("p_bid_mc",     bmc),
            ("p_ask_mc_adj", amc_adj),
            ("p_bid_mc_adj", bmc_adj),
            ("p_ask",        ask),
            ("p_bid",        bid),
            ("spread",       spr),
            ("s1_spread",    s1),
            ("s1b_spread",   s1b),
            ("s2_spread",    s2),
        ]

        if include_iv:
            def _iv_row(df: pd.DataFrame) -> list:  # type: ignore[misc]
                return [round(float(df.loc[T_days, l]) * 100, 4) for l in labels]  # type: ignore[arg-type]
            metrics_to_emit += [
                ("iv_model",     _iv_row(iv_mid)),        # type: ignore[arg-type]
                ("iv_model_adj", _iv_row(iv_mid_adj)),    # type: ignore[arg-type]
                ("iv_ask_mc",    _iv_row(iv_ask_mc_df)),  # type: ignore[arg-type]
                ("iv_bid_mc",    _iv_row(iv_bid_mc_df)),  # type: ignore[arg-type]
                ("iv_ask",       _iv_row(iv_ask_df)),     # type: ignore[arg-type]
                ("iv_bid",       _iv_row(iv_bid_df)),     # type: ignore[arg-type]
            ]

        for metric, vals in metrics_to_emit:
            rows.append(pd.Series(vals, index=labels, name=(int(T_days), metric)))

    df = pd.DataFrame(rows)
    if not df.empty:
        df.index = pd.MultiIndex.from_tuples(
            [r.name for r in rows],   # type: ignore[arg-type]
            names=["tenor_days", "metric"],
        )
    return df


def _print_price_summary(
        points:        list[tuple[int, str]],
        p_fair:        pd.DataFrame,
        p_fair_adj:    pd.DataFrame,
        p_ask_mc:      pd.DataFrame,
        p_bid_mc:      pd.DataFrame,
        p_ask_mc_adj:  pd.DataFrame,
        p_bid_mc_adj:  pd.DataFrame,
        p_ask:         pd.DataFrame,
        p_bid:         pd.DataFrame,
        k_surface_df:  pd.DataFrame,
        S:             float,
        strategy:      str,
        trading_days:  int = TRADING_DAYS,
) -> None:
    """
    Print a compact summary table at TARGET_POINTS with step-contribution breakdown.

    Columns
    -------
    p_fair, p_fair_adj — original and bias-corrected fair prices
    p_ask, p_bid, spread — final prices and total spread
    s1_spread, s1%   — S4-10 jump-adj contribution (absolute + % of total spread)
    s1b_bias         — S4-10b bias-shift on mid price (p_fair_adj − p_fair)
    s2_spread, s2%   — S4-11 σ_HE spread contribution
    spread/adj%      — total spread / p_fair_adj × 100
    """
    rows = []
    for T_days, lbl in points:
        if T_days not in p_fair.index or lbl not in p_fair.columns:
            continue
        k_val    = float(k_surface_df.loc[T_days, lbl])
        K_over_S = round(np.exp(k_val), 4)
        pf     = float(p_fair.loc[T_days, lbl])
        pf_adj = float(p_fair_adj.loc[T_days, lbl])
        pa     = float(p_ask.loc[T_days, lbl])
        pb     = float(p_bid.loc[T_days, lbl])
        amc    = float(p_ask_mc.loc[T_days, lbl])
        bmc    = float(p_bid_mc.loc[T_days, lbl])
        amc_adj = float(p_ask_mc_adj.loc[T_days, lbl])

        spr_total = pa - pb
        spr_s1    = amc - bmc
        spr_s1b   = amc_adj - amc          # bias shift (centre movement)
        spr_s2    = spr_total - spr_s1 - spr_s1b
        s1_pct    = round(spr_s1 / spr_total * 100, 1) if spr_total > 0 else float("nan")
        s2_pct    = round(spr_s2 / spr_total * 100, 1) if spr_total > 0 else float("nan")

        rows.append({
            "T":            T_days,
            "label":        lbl,
            "K/S":          K_over_S,
            "p_fair":       round(pf,        4),
            "p_fair_adj":   round(pf_adj,    4),
            "bias_Δ":       round(pf_adj - pf, 4),
            "p_ask":        round(pa,        4),
            "p_bid":        round(pb,        4),
            "spread":       round(spr_total, 4),
            "s1_spread":    round(spr_s1,    4),
            "s1%":          s1_pct,
            "s1b_bias":     round(spr_s1b,   4),
            "s2_spread":    round(spr_s2,    4),
            "s2%":          s2_pct,
            "spread/adj%":  round(spr_total / pf_adj * 100, 2) if pf_adj > 0 else float("nan"),
        })

    if rows:
        print(f"\n  Price summary  (S={S:.2f}  strategy={strategy})")
        print(f"  s1=jump-adj  |  s1b=bias-shift (mid corr)  |  s2=σ_HE spread")
        print(pd.DataFrame(rows).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN STAGE 4 RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage4(
        iv_surface_df: pd.DataFrame,
        k_surface_df:  pd.DataFrame,
        S_vol:         float,
        price_df:      pd.DataFrame,
        h_series:      pd.Series,  # type: ignore[type-arg]
        garch_params:  dict,
        jump_params:   dict,
        pricing_date:  str,
        he_detail:     pd.DataFrame,
        verbose:       bool = True,
) -> pd.DataFrame:
    """
    Execute Stage 4 in memory.

    Parameters
    ----------
    iv_surface_df : fair IV surface from Stage 2  (rows=T_days, cols=labels)
    k_surface_df  : log-moneyness surface          (same shape)
    S_vol         : closing price on the pricing date
    price_df      : full price history (for h_t computation in S4-10)
    h_series      : GARCH filter h_t series        (from Stage 1)
    garch_params  : GARCH(1,1) parameters          (from Stage 1)
    jump_params   : base jump parameters           (from Stage 1)
    pricing_date  : date string for the displayed surface (e.g. "2026-03-09")
    he_detail     : per-node HE stats from Stage 3
                    MultiIndex (T_days, q) × {HE_per_S_mean, HE_per_S_std}
    verbose       : print progress and summary tables

    Returns
    -------
    price_table : DataFrame (MultiIndex tenor_days × metric, cols=labels)
        metric ∈ {K_over_S,
                  p_fair, p_fair_adj,
                  p_ask_mc, p_bid_mc, p_ask_mc_adj, p_bid_mc_adj,
                  p_ask, p_bid, spread,
                  s1_spread, s1b_spread, s2_spread,
                  iv_model, iv_model_adj,
                  iv_ask_mc, iv_bid_mc, iv_ask, iv_bid}
    """
    t0       = time.time()
    strategy = STRATEGY_NAME

    if verbose:
        print(f"\n{_SEP}")
        print("  STAGE 4 — Bid / Mid / Ask Price Surface  (three-step engine)")
        print(f"  Strategy  : {strategy}")
        print(f"  Pricing   : {pricing_date}")
        print(f"  S_vol     : {S_vol:.4f}")
        print(f"  S4-10 ask : LAM×{S4_ASK_LAM_MULT}  MU_J+{S4_ASK_MU_J_ADD:+.3f}"
              f"  SIG×{S4_ASK_SIGMA_J_MULT}")
        print(f"  S4-10 bid : LAM×{S4_BID_LAM_MULT}  MU_J+{S4_BID_MU_J_ADD:+.3f}"
              f"  SIG×{S4_BID_SIGMA_J_MULT}")
        print(f"  S4-10b    : HE bias corr  coef={S4_HE_BIAS_COEF}  "
              f"clip=±{S4_HE_BIAS_CLIP:.0%} vol-pts")
        print(f"  S4-11     : spread ASK(σ×{S4_ASK_SIGMA_COEF})  "
              f"BID(σ×{S4_BID_SIGMA_COEF})  [σ_HE only]")
        print(_SEP)

    # ── S4-10: Build adjusted IV surfaces via MC + eSSVI ─────────────────────
    if verbose:
        print(f"\n── S4-10: MC Repricing  (N={S4_N_PATHS:,} paths per side) ────────")

    _, h_t = _get_S_h(pricing_date, price_df, h_series, garch_params)
    if verbose:
        print(f"  h_t at {pricing_date}: ann vol = "
              f"{np.sqrt(h_t * TRADING_DAYS) * 100:.2f}%")

    jump_ask = _adj_jump_params(
        jump_params, S4_ASK_LAM_MULT, S4_ASK_MU_J_ADD, S4_ASK_SIGMA_J_MULT)
    jump_bid = _adj_jump_params(
        jump_params, S4_BID_LAM_MULT, S4_BID_MU_J_ADD, S4_BID_SIGMA_J_MULT)

    if verbose:
        lam0, mj0, sj0 = jump_params["lam"], jump_params["mu_j"], jump_params["sigma_j"]
        print(f"  Base  jump: lam={lam0:.4f}  mu_j={mj0:.4f}  sigma_j={sj0:.4f}")
        print(f"  Ask   jump: lam={jump_ask['lam']:.4f}  mu_j={jump_ask['mu_j']:.4f}"
              f"  sigma_j={jump_ask['sigma_j']:.4f}")
        print(f"  Bid   jump: lam={jump_bid['lam']:.4f}  mu_j={jump_bid['mu_j']:.4f}"
              f"  sigma_j={jump_bid['sigma_j']:.4f}")

    t_mc = time.time()
    iv_ask = _build_adj_iv_surface(
        S_vol, h_t, k_surface_df, garch_params, jump_ask,
        n_paths=S4_N_PATHS, seed=SEED,
    )
    iv_bid = _build_adj_iv_surface(
        S_vol, h_t, k_surface_df, garch_params, jump_bid,
        n_paths=S4_N_PATHS, seed=SEED + 1,
    )
    if verbose:
        print(f"  [S4-10 MC time: {time.time()-t_mc:.1f}s]")

    # Convert original (unadjusted) fair IV to fair prices (for reference)
    p_fair   = _prices_from_iv(iv_surface_df, k_surface_df, S_vol, strategy)
    p_ask_mc = _prices_from_iv(iv_ask,        k_surface_df, S_vol, strategy)
    p_bid_mc = _prices_from_iv(iv_bid,        k_surface_df, S_vol, strategy)

    # ── S4-10b: HE Bias Correction ────────────────────────────────────────────
    # Shift mid, ask_mc, and bid_mc by the same per-node Δσ_bias = −μ_HE/vega_eff.
    # This corrects the centre of the market without changing the step-1 spread.
    if verbose:
        print(f"\n── S4-10b: HE Bias Correction (mid-vol shift) ───────────────────")
        print(f"  Δσ_bias = clip(−{S4_HE_BIAS_COEF}×μ_HE/vega_eff, "
              f"±{S4_HE_BIAS_CLIP:.0%})   applied to mid, ask_mc, bid_mc")

    iv_mid_adj  = _apply_he_bias_correction(
        iv_surface_df, he_detail, k_surface_df, S_vol)
    iv_ask_adj  = _apply_he_bias_correction(
        iv_ask,        he_detail, k_surface_df, S_vol)
    iv_bid_adj  = _apply_he_bias_correction(
        iv_bid,        he_detail, k_surface_df, S_vol)

    p_fair_adj   = _prices_from_iv(iv_mid_adj,  k_surface_df, S_vol, strategy)
    p_ask_mc_adj = _prices_from_iv(iv_ask_adj,  k_surface_df, S_vol, strategy)
    p_bid_mc_adj = _prices_from_iv(iv_bid_adj,  k_surface_df, S_vol, strategy)

    if verbose:
        # Show a few representative bias shifts at ATM
        atm_label = "ATM"
        if atm_label in iv_surface_df.columns:
            for T_days in list(iv_surface_df.index)[:3]:
                orig = float(iv_surface_df.loc[T_days, atm_label]) * 100
                adj  = float(iv_mid_adj.loc[T_days, atm_label])    * 100
                print(f"  T={int(T_days):3d}d  ATM: iv_model={orig:.2f}%  "
                      f"iv_model_adj={adj:.2f}%  Δ={adj-orig:+.2f} vol-pts")

    # ── S4-11: Apply HE spread overlay (σ_HE only) ────────────────────────────
    if verbose:
        print(f"\n── S4-11: HE Vol-Space Spread Overlay (σ_HE only) ───────────────")
        print(f"  Δσ_ask = {S4_ASK_SIGMA_COEF}×σ_HE/vega_eff  +  {S4_SPREAD_FLOOR_COEF}×ε/vega_eff")
        print(f"  Δσ_bid = {S4_BID_SIGMA_COEF}×σ_HE/vega_eff  +  {S4_SPREAD_FLOOR_COEF}×ε/vega_eff")
        print(f"  ε(T) = {S4_VEGA_REG_ALPHA} × 0.4 × √T   (OTM floor source; floor→0 at ATM)")
        print(f"  [μ_HE is NOT used here — fully absorbed by S4-10b]")

    p_ask, p_bid, iv_ask_final, iv_bid_final = _apply_he_spread(
        iv_ask_adj, iv_bid_adj, iv_mid_adj, he_detail, k_surface_df, S_vol, strategy,
    )

    # ── S4-12: Butterfly arb fix ──────────────────────────────────────────────
    if verbose:
        print(f"\n── S4-12: Butterfly Arb Check + Fix (Method B) ─────────────────")

    p_ask_fixed, n_ask = _fix_butterfly_arb(p_ask, k_surface_df)
    p_bid_fixed, n_bid = _fix_butterfly_arb(p_bid, k_surface_df)

    if verbose:
        if n_ask == 0 and n_bid == 0:
            print("  ✓ No butterfly violations found.")
        else:
            print(f"  Ask surface: {n_ask} correction(s) applied.")
            print(f"  Bid surface: {n_bid} correction(s) applied.")

    # ── Print summary at TARGET_POINTS ────────────────────────────────────────
    if verbose:
        _print_price_summary(
            TARGET_POINTS,
            p_fair, p_fair_adj,
            p_ask_mc, p_bid_mc,
            p_ask_mc_adj, p_bid_mc_adj,
            p_ask_fixed, p_bid_fixed,
            k_surface_df, S_vol, strategy,
        )

    # ── Full surface print ────────────────────────────────────────────────────
    if verbose:
        print(f"\n── Full Price Surface (p_fair / p_fair_adj / p_ask / p_bid / spread) ──")
        for T_days in p_fair.index:
            print(f"\n  T = {T_days}d")
            row_fair     = p_fair.loc[T_days].round(4)
            row_fair_adj = p_fair_adj.loc[T_days].round(4)
            row_ask      = p_ask_fixed.loc[T_days].round(4)
            row_bid      = p_bid_fixed.loc[T_days].round(4)
            row_spr      = (p_ask_fixed.loc[T_days] - p_bid_fixed.loc[T_days]).round(4)
            print(pd.DataFrame(
                [row_fair, row_fair_adj, row_ask, row_bid, row_spr],
                index=pd.Index(["p_fair", "p_fair_adj", "p_ask", "p_bid", "spread"]),
            ).to_string())

    # ── Assemble output table ──────────────────────────────────────────────────
    price_table = _format_price_table(
        p_fair, p_fair_adj,
        p_ask_mc, p_bid_mc,
        p_ask_mc_adj, p_bid_mc_adj,
        p_ask_fixed, p_bid_fixed,
        k_surface_df, S_vol,
        iv_mid=iv_surface_df,
        iv_mid_adj=iv_mid_adj,
        iv_ask_mc_df=iv_ask,
        iv_bid_mc_df=iv_bid,
        iv_ask_df=iv_ask_final,
        iv_bid_df=iv_bid_final,
    )

    if verbose:
        print(f"\n  [Stage 4 total: {time.time()-t0:.1f}s]")

    return price_table


# ── checkpoint path helper ────────────────────────────────────────────────────

def _price_surface_path(results_dir: str) -> str:
    return os.path.join(
        results_dir, f"{config.TICKER}_{STRATEGY_NAME}_price_surface.csv"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDALONE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results_dir = os.path.join(_ROOT, "Results")

    _pricing_date = PRINT_VOL_SURFACE_DATE
    if _pricing_date is None:
        print("[ERROR] Set PRINT_VOL_SURFACE_DATE in config.py.")
        sys.exit(1)

    # ── Load Stage 1 checkpoint (model params + h_series) ────────────────────
    print(f"\n{_SEP}")
    print("  Loading Stage 1 checkpoint …")
    garch_params, jump_params, _, h_series, _, _ = _load_stage1(results_dir)

    # ── Load Stage 2 checkpoint (IV surface + K surface) ─────────────────────
    print("  Loading Stage 2 checkpoint …")
    _, iv_surface_df, k_surface_df = _load_stage2(results_dir)

    # ── Load Stage 3 checkpoint (per-node HE stats) ───────────────────────────
    print("  Loading Stage 3 checkpoint …")
    he_detail = _load_stage3(results_dir)

    if iv_surface_df is None or k_surface_df is None:
        print("[ERROR] iv_surface.csv not found. Run stage2 with "
              "PRINT_VOL_SURFACE_DATE set.")
        sys.exit(1)

    # ── Resolve stock price on the pricing date ──────────────────────────────
    price_df    = _load_price_df(
        os.path.join(_ROOT, DATA_DIR, f"{config.TICKER}.csv"))
    actual_date = _nearest_trading_date(price_df, _pricing_date)
    if actual_date != _pricing_date:
        print(f"  [INFO] {_pricing_date} not a trading day → using {actual_date}")

    price_ts         = pd.Timestamp(actual_date)
    price_df_dates   = pd.to_datetime(price_df["date"]).dt.normalize()
    date_mask        = price_df_dates == price_ts
    if date_mask.any():
        S_vol = float(price_df.loc[date_mask, "close"].iloc[0])
    else:
        sorted_px = price_df.assign(_d=price_df_dates).sort_values("_d")
        S_vol     = float(sorted_px[sorted_px["_d"] <= price_ts]["close"].iloc[-1])  # type: ignore[index]

    print(f"  Pricing date : {actual_date}")
    print(f"  S on that day: {S_vol:.4f}")
    print(_SEP)

    # ── Run Stage 4 ──────────────────────────────────────────────────────────
    price_table = run_stage4(
        iv_surface_df = iv_surface_df,
        k_surface_df  = k_surface_df,
        S_vol         = S_vol,
        price_df      = price_df,
        h_series      = h_series,
        garch_params  = garch_params,
        jump_params   = jump_params,
        pricing_date  = actual_date,
        he_detail     = he_detail,
        verbose       = True,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    print("\n── Saving Stage 4 Outputs ──────────────────────────────────────")
    os.makedirs(results_dir, exist_ok=True)
    price_table.to_csv(_price_surface_path(results_dir))
    print(f"  [Saved] price_surface → {_price_surface_path(results_dir)}")
    print(_SEP)
