"""
vol_surface.py
==============
Step 3+4 — Vol surface construction:
  _build_daily_iv_surface_mc   daily surface via pre-generated rand block
  _build_ref_mc_surface        adaptive reference MC (determines N_block)
  _essvi_w                     eSSVI total-variance formula
  _calibrate_essvi_and_extract eSSVI calibration + iv_func builder
  get_vol_surface_df           query surface for any date
  plot_vol_surface             print + plot + return DataFrames
"""

from __future__ import annotations
import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .config import TRADING_DAYS, N_BATCH, N_MAX, R_ANNUAL, Q_ANNUAL, SEED, WEIGHT_WINGS
from .pricing import _bs_vega, _invert_iv
from .simulation import _make_rand_block, _simulate_from_block


# ── Daily IV surface (slices pre-generated block) ─────────────────────────────

def _build_daily_iv_surface_mc(
        S_t: float, h_t: float, day_idx: int,
        rand_block: dict,
        garch_params: dict, jump_params: dict,
        tenors_days: tuple | list, q_grid: np.ndarray,
        r: float = R_ANNUAL, q: float = Q_ANNUAL,
        n_paths: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Build today's raw IV surface by slicing the pre-generated random block.

    Quantile-based strike grid + ATM anchor
    ----------------------------------------
    For each tenor T:
      1. Simulate N paths of S_T using pre-generated random block rows
         [day_idx : day_idx+T] — no new RNG calls.
      2. K_arr = np.quantile(S_T, q_grid).
      3. K_atm = mean(S_T) ≈ E_Q[S_T] = F_T.  Added as "ATM" column.
      4. ATM inserted at its sorted k position so k_grid[T] and iv_surface
         columns always share the same ordering (required by eSSVI).
      5. Price each call from the same paths; invert to IV.

    n_paths : if given, only use the first n_paths columns of rand_block.
              Reduces simulation cost when full accuracy is not required
              (e.g. daily delta hedging in the backtest loop).

    Returns
    -------
    iv_surface : DataFrame  (index=tenor_days, columns sorted by k, ATM inline)
    k_grid     : {T_int: k_array}  — sorted log-moneyness, ATM at sorted position
    """
    q_labels   = [f"q{int(q_val*100):02d}" for q_val in q_grid]
    all_labels = None
    iv_rows    = []
    k_grid     = {}

    # Optionally cap the number of paths to reduce simulation cost.
    # Slicing columns of rand_block avoids any extra memory allocation.
    if n_paths is not None:
        n_paths = min(n_paths, rand_block["Z"].shape[1])
        rand_block = {
            "Z": rand_block["Z"][:, :n_paths],
            "J": rand_block["J"][:, :n_paths],
        }

    for T_days in tenors_days:
        T_int    = int(T_days)
        T_years  = T_int / TRADING_DAYS
        discount = np.exp(-r * T_years)

        end_idx = day_idx + T_int
        if end_idx > rand_block["Z"].shape[0]:
            raise ValueError(
                f"Pre-generated block too short: need row {end_idx}, "
                f"block has {rand_block['Z'].shape[0]} rows. "
                f"Increase T_total when calling _make_rand_block."
            )

        sub_block = {
            "Z": rand_block["Z"][day_idx:end_idx],
            "J": rand_block["J"][day_idx:end_idx],
        }
        S_T = _simulate_from_block(S_t, h_t, T_int, sub_block,
                                    garch_params, jump_params, r, q)

        K_arr = np.quantile(S_T, q_grid)
        k_arr = np.log(K_arr / S_t)

        K_atm      = float(np.mean(S_T))
        k_atm      = float(np.log(K_atm / S_t))
        ins        = int(np.searchsorted(k_arr, k_atm))
        k_arr_full = np.insert(k_arr, ins, k_atm)
        k_grid[T_int] = k_arr_full

        tenor_labels = q_labels[:ins] + ["ATM"] + q_labels[ins:]
        if all_labels is None:
            all_labels = tenor_labels

        iv_by_lbl = {}
        for iq in range(len(q_grid)):
            payoffs            = np.maximum(S_T - K_arr[iq], 0.0)
            price_mc           = float(np.mean(payoffs)) * discount
            iv_by_lbl[q_labels[iq]] = _invert_iv(price_mc, S_t, K_arr[iq],
                                                   T_years, r, q)
        payoffs_atm        = np.maximum(S_T - K_atm, 0.0)
        price_mc_atm       = float(np.mean(payoffs_atm)) * discount
        iv_by_lbl["ATM"]   = _invert_iv(price_mc_atm, S_t, K_atm, T_years, r, q)

        iv_rows.append([iv_by_lbl[lbl] for lbl in all_labels])

    if all_labels is None:
        all_labels = q_labels + ["ATM"]

    iv_surface = pd.DataFrame(iv_rows, index=list(tenors_days), columns=all_labels)
    iv_surface.index.name   = "tenor_days"
    iv_surface.columns.name = "q"
    return iv_surface, k_grid


# ── Adaptive reference MC (determines N_block) ────────────────────────────────

def _build_ref_mc_surface(S0: float, garch_params: dict, jump_params: dict,
                           tenors_days: tuple | list, q_grid: np.ndarray,
                           h0: float | None = None,
                           epoch_size: int = 30_000,
                           n_max: int = N_MAX,
                           tol_bps: float = 50.0,
                           r: float = R_ANNUAL, q: float = Q_ANNUAL,
                           seed: int = SEED, verbose: bool = True,
                           ) -> tuple[pd.DataFrame, dict, int]:
    """
    Adaptive epoch-based reference MC surface with quantile-based strikes.

    Returns
    -------
    iv_surface : DataFrame  (index=tenor_days, columns=q_labels)
    k_grid     : dict {T_int: k_array}
    n_required : int  — worst-case path count at convergence
    """
    omega   = garch_params["omega"]
    alpha   = garch_params["alpha"]
    beta    = garch_params["beta"]
    phi     = alpha + beta
    lam     = jump_params["lam"]
    mu_j    = jump_params["mu_j"]
    sigma_j = jump_params["sigma_j"]

    var_j  = lam * (mu_j**2 + sigma_j**2)
    m_j2   = var_j + (lam * mu_j)**2
    h_bar  = (omega + alpha * m_j2) / max(1.0 - phi, 1e-12)
    h_start = h0 if (h0 is not None and h0 > 0) else h_bar
    tol     = tol_bps * 1e-4
    n_q     = len(q_grid)
    q_labels = [f"q{int(q_val*100):02d}" for q_val in q_grid]

    if verbose:
        print(f"  epoch={epoch_size:,}  n_max={n_max:,}  tol={tol_bps:.0f}bps  "
              f"h_start={np.sqrt(h_start*TRADING_DAYS)*100:.1f}%ann  "
              f"tenors={list(tenors_days)}")

    iv_rows           = []
    k_grid_out        = {}
    n_paths_per_tenor = []
    t0_total          = time.time()

    for T_days in tenors_days:
        T_int    = int(T_days)
        T_years  = T_int / TRADING_DAYS
        discount = np.exp(-r * T_years) #r=0
        t0_t     = time.time()

        blk0  = _make_rand_block(T_int, epoch_size, lam, mu_j, sigma_j,
                                  seed=seed + T_int * 10_000)
        S_T0  = _simulate_from_block(S0, h_start, T_int, blk0,
                                      garch_params, jump_params, r, q)
        K_arr = np.quantile(S_T0, q_grid)

        all_S_T   = [S_T0]
        sum_pay   = np.array([np.maximum(S_T0 - K_arr[iq], 0.0).sum()
                               for iq in range(n_q)])
        sum_sq    = np.array([(np.maximum(S_T0 - K_arr[iq], 0.0)**2).sum()
                               for iq in range(n_q)])
        n_total   = epoch_size
        converged = np.zeros(n_q, dtype=bool)

        epoch_idx = 1
        while n_total < n_max:
            block = _make_rand_block(T_int, epoch_size, lam, mu_j, sigma_j,
                                      seed=seed + T_int * 10_000 + epoch_idx)
            S_T   = _simulate_from_block(S0, h_start, T_int, block,
                                          garch_params, jump_params, r, q)
            all_S_T.append(S_T)
            epoch_idx += 1
            n_total   += epoch_size

            for iq in range(n_q):
                pay = np.maximum(S_T - K_arr[iq], 0.0)
                sum_pay[iq] += pay.sum()
                sum_sq[iq]  += (pay**2).sum()

            for iq in range(n_q):
                if converged[iq]:
                    continue
                mean_pay  = sum_pay[iq] / n_total
                var_pay   = max(sum_sq[iq] / n_total - mean_pay**2, 0.0)
                se_pay    = np.sqrt(var_pay / n_total) * discount # but r =0

                crit_a = (se_pay / S0) < tol
                crit_b = False
                price_est = mean_pay * discount
                if price_est > 1e-10:
                    iv_est = _invert_iv(price_est, S0, K_arr[iq], T_years, r, q)
                    if np.isfinite(iv_est) and iv_est > 0:
                        vega   = _bs_vega(S0, K_arr[iq], T_years, iv_est, r, q)
                        crit_b = (se_pay / max(vega, 1e-8)) < tol
                if crit_a or crit_b:
                    converged[iq] = True

            if converged.all():
                break

        S_T_all  = np.concatenate(all_S_T)
        K_arr    = np.quantile(S_T_all, q_grid)
        k_arr    = np.log(K_arr / S0)
        k_grid_out[T_int] = k_arr

        iv_row = []
        for iq in range(n_q):
            payoffs   = np.maximum(S_T_all - K_arr[iq], 0.0)
            price_est = float(payoffs.mean()) * discount
            iv        = _invert_iv(price_est, S0, K_arr[iq], T_years, r, q)
            iv_row.append(iv)

        iv_rows.append(iv_row)
        n_paths_per_tenor.append(n_total)
        elapsed = time.time() - t0_t
        if verbose:
            print(
                  f"epochs={epoch_idx}({n_total:,}paths)  "
                  f"conv={converged.sum()}/{n_q}  [{elapsed:.1f}s]")

    iv_surface = pd.DataFrame(iv_rows, index=list(tenors_days), columns=q_labels)
    iv_surface.index.name   = "tenor_days"
    iv_surface.columns.name = "q"

    n_required = max(n_paths_per_tenor)
    if verbose:
        print(f"  Total ref-MC time: {time.time()-t0_total:.1f}s")
        print(f"  → N_required = {n_required:,}  (worst-case across all tenors)")

    return iv_surface, k_grid_out, n_required


# ── eSSVI calibration ─────────────────────────────────────────────────────────

def _essvi_w(k: np.ndarray, theta: float, rho: float, psi: float) -> np.ndarray:
    """
    eSSVI total variance:
        w(k) = ½ [θ + ρψk + √((ψk + ρθ)² + (1−ρ²)θ²)]
    """
    k      = np.asarray(k, float)
    inside = (psi * k + rho * theta)**2 + (1.0 - rho**2) * theta**2
    return 0.5 * (theta + rho * psi * k + np.sqrt(np.maximum(inside, 1e-24)))


def _calibrate_essvi_and_extract(iv_surface: pd.DataFrame, k_grid: dict,
                                  weight_wings: float = WEIGHT_WINGS,
                                  rho_bounds: tuple = (-0.999, 0.0),
                                  n_rho_grid: int = 20,
                                  verbose: bool = True,
                                  ) -> tuple[pd.DataFrame, dict, dict]:
    """
    eSSVI calibration.  For each tenor:
      1. θ = σ_ATM² × T_years  (from raw MC surface ATM col)
      2. Monotonise θ across tenors
      3. Grid-search ρ; optimise ψ at each ρ to minimise wing-weighted MSE
      4. Build iv_func(k, T_years) by linear interpolation

    Returns
    -------
    iv_fitted  : DataFrame
    essvi_info : dict  {iv_func, per_tenor_params}
    essvi_ref  : dict  {T_arr, theta_arr, psi_arr, rhopsi_arr}
    """
    tenors_days = np.asarray(iv_surface.index.values, float)
    T_arr_all   = tenors_days / TRADING_DAYS
    iv_mat      = iv_surface.values.astype(float)
    n_cols      = iv_mat.shape[1]

    if verbose:
        print("\n── STEP 3 (eSSVI): Calibrating smile shape ───────────────────")

    theta_arr = np.full(len(T_arr_all), np.nan)
    for i, Ti in enumerate(T_arr_all):
        if Ti <= 0:
            continue
        T_int = int(round(Ti * TRADING_DAYS))
        k_all = np.asarray(k_grid.get(T_int, []), float)
        row   = iv_mat[i, :]
        ok    = np.isfinite(row) & (row > 0)
        if not ok.any() or len(k_all) != n_cols:
            continue
        atm_idx = int(np.argmin(np.abs(k_all)))
        iv_atm  = float(row[atm_idx]) if ok[atm_idx] else float(row[ok][np.argmin(np.abs(k_all[ok]))])
        theta_arr[i] = iv_atm**2 * Ti

    # Enforce θ monotone-increasing via PAVA (Pool Adjacent Violators Algorithm).
    # PAVA finds the L2-closest monotone sequence to the raw θ values, so an
    # isolated anomalous spike is averaged down with its neighbours rather than
    # propagating forward and artificially inflating all subsequent tenors.
    # After PAVA we add a tiny gap (1e-6) between adjacent values to satisfy the
    # strict-inequality requirement of the calendar-arb condition.
    valid_mask = np.isfinite(theta_arr)
    if valid_mask.sum() >= 2:
        valid_idx = np.where(valid_mask)[0]
        raw_valid = theta_arr[valid_idx].copy()

        # --- isotonic regression (O(n) PAVA, no external dependency) ----------
        def _pava(y: np.ndarray) -> np.ndarray:
            """In-place PAVA for non-decreasing constraint."""
            y = y.copy().astype(float)
            n = len(y)
            # blocks: list of (sum, count) representing current pool
            blocks: list[list[float]] = [[y[0], 1.0]]
            for j in range(1, n):
                blocks.append([y[j], 1.0])
                while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                    s  = blocks[-2][0] + blocks[-1][0]
                    c  = blocks[-2][1] + blocks[-1][1]
                    blocks[-2:] = [[s, c]]
            out = np.empty(n)
            pos = 0
            for s, c in blocks:
                mean_val = s / c
                cnt = int(round(c))
                out[pos: pos + cnt] = mean_val
                pos += cnt
            return out

        theta_arr[valid_idx] = _pava(raw_valid)

    # Ensure strict monotonicity: each step must exceed previous by at least 1e-6
    last_th = 0.0
    for i in range(len(theta_arr)):
        if np.isfinite(theta_arr[i]):
            theta_arr[i] = max(theta_arr[i], last_th + 1e-6)
            last_th = theta_arr[i]

    rho_grid = np.linspace(float(rho_bounds[0]), float(rho_bounds[1]), n_rho_grid)
    fitted_T, fitted_th, fitted_rho, fitted_psi, fitted_rhopsi = [], [], [], [], []
    prev_psi, prev_rhopsi = 0.0, 0.0

    for i, Ti in enumerate(T_arr_all):
        if not np.isfinite(theta_arr[i]) or Ti <= 0:
            continue
        th_i  = theta_arr[i]
        T_int = int(round(Ti * TRADING_DAYS))
        k_all = np.asarray(k_grid.get(T_int, []), float)
        row   = iv_mat[i, :]
        ok    = np.isfinite(row) & (row > 0)
        if ok.sum() < 3 or len(k_all) != n_cols:
            continue

        k_obs  = k_all[ok];  iv_obs = row[ok];  w_obs = iv_obs**2 * Ti
        k_scale = max(float(np.max(np.abs(k_obs))), 1e-3)
        wgt     = 1.0 + (weight_wings - 1.0) * np.abs(k_obs) / k_scale

        def mse(psi, rho):
            if psi <= 0:
                return 1e12
            wm = _essvi_w(k_obs, th_i, rho, psi)
            return 1e12 if np.any(wm < 0) else float(((wgt * (wm - w_obs))**2).mean())

        best_cost = np.inf
        best_rho  = float(rho_bounds[0]) * 0.5
        best_psi  = max(prev_psi * 1.01, 1e-6)

        for rho in rho_grid:
            psi_bfly = min(4.0 / (1.0 + abs(rho)),
                           2.0 * np.sqrt(th_i / (1.0 + abs(rho))))
            if prev_psi > 0:
                rho_prev = prev_rhopsi / prev_psi
                a   = prev_psi * (1 - rho_prev) / (1 - rho) if rho < 1 - 1e-9 else 1e12
                b_v = prev_psi * (1 + rho_prev) / (1 + rho) if rho > -1 + 1e-9 else 1e12
                psi_lo = max(prev_psi, a, b_v, 1e-9)
            else:
                psi_lo = 1e-9
            psi_hi = psi_bfly * 0.9999
            if psi_lo >= psi_hi - 1e-12:
                continue
            try:
                res = minimize_scalar(lambda p: mse(p, rho),
                                      bounds=(psi_lo, psi_hi), method="bounded",
                                      options={"xatol": 1e-8, "maxiter": 100})
                if np.isfinite(res.fun) and res.fun < best_cost:
                    best_cost = res.fun
                    best_rho  = float(rho)
                    best_psi  = float(res.x)
            except Exception:
                pass

        fitted_T.append(Ti);      fitted_th.append(th_i)
        fitted_rho.append(best_rho);  fitted_psi.append(best_psi)
        fitted_rhopsi.append(best_rho * best_psi)
        prev_psi = best_psi;  prev_rhopsi = best_rho * best_psi

    if not fitted_T:
        raise ValueError("eSSVI calibration: no tenor succeeded.")

    ft_T  = np.array(fitted_T);   ft_th = np.array(fitted_th)
    ft_ps = np.array(fitted_psi); ft_rp = np.array(fitted_rhopsi)

    def _interp(T_years: float) -> tuple[float, float, float]:
        if T_years <= ft_T[0]:
            f = T_years / ft_T[0]
            return f * ft_th[0], f * ft_ps[0], f * ft_rp[0]
        if T_years >= ft_T[-1]:
            slope = ((ft_th[-1] - ft_th[-2]) / max(ft_T[-1] - ft_T[-2], 1e-9)
                     if len(ft_T) > 1 else ft_th[-1] / ft_T[-1])
            return ft_th[-1] + (T_years - ft_T[-1]) * slope, ft_ps[-1], ft_rp[-1]
        idx  = min(max(0, int(np.searchsorted(ft_T, T_years, "right")) - 1), len(ft_T) - 2)
        T_lo, T_hi = ft_T[idx], ft_T[idx + 1]
        f = (T_years - T_lo) / max(T_hi - T_lo, 1e-12)
        return ((1-f)*ft_th[idx] + f*ft_th[idx+1],
                (1-f)*ft_ps[idx] + f*ft_ps[idx+1],
                (1-f)*ft_rp[idx] + f*ft_rp[idx+1])

    def iv_func(k: float | np.ndarray, T_years: float) -> float | np.ndarray:
        T_years = float(T_years)
        k_arr   = np.asarray(k, float)
        if T_years <= 0:
            return np.full_like(k_arr, np.nan) if k_arr.ndim > 0 else float("nan")
        th, psi, rp = _interp(T_years)
        psi     = max(psi, 1e-12)
        rho_eff = float(np.clip(rp / psi, -0.9999, 0.9999))
        w       = _essvi_w(k_arr, max(th, 1e-12), rho_eff, psi)
        iv      = np.sqrt(np.maximum(w / T_years, 0.0))
        return float(iv) if iv.ndim == 0 else iv

    rows = []
    for i, Ti in enumerate(T_arr_all):
        T_int = int(round(Ti * TRADING_DAYS))
        k_row = np.asarray(k_grid.get(T_int, []), float)
        if len(k_row) != n_cols or not np.isfinite(theta_arr[i]):
            rows.append([np.nan] * n_cols)
            continue
        rows.append([float(iv_func(float(kv), Ti)) for kv in k_row])
    iv_fitted = pd.DataFrame(rows, index=iv_surface.index, columns=iv_surface.columns)
    iv_fitted = iv_fitted.where(iv_surface.notna())

    records = [{"T_days": int(round(Ti * TRADING_DAYS)), "theta": round(th, 6),
                "rho": round(rho, 4), "psi": round(psi, 4),
                "ATM_IV%": round(np.sqrt(max(th / max(Ti, 1e-9), 0)) * 100, 2)}
               for Ti, th, rho, psi in zip(fitted_T, fitted_th, fitted_rho, fitted_psi)]
    params_df = pd.DataFrame(records).set_index("T_days") if records else pd.DataFrame()

    if verbose:
        print(params_df.to_string())

    essvi_info = {"per_tenor_params": params_df, "iv_func": iv_func}
    essvi_ref  = {"T_arr": ft_T, "theta_arr": ft_th,
                  "psi_arr": ft_ps, "rhopsi_arr": ft_rp}
    return iv_fitted, essvi_info, essvi_ref


# ── Vol surface query for a specific date ─────────────────────────────────────

def get_vol_surface_df(date_str: str, price_df: pd.DataFrame,
                        h_series: pd.Series, garch_params: dict,
                        jump_params: dict,
                        tenors_days: tuple | list, q_grid: np.ndarray,
                        n_paths: int = N_BATCH,
                        r: float = R_ANNUAL, q: float = Q_ANNUAL,
                        seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the model IV surface and log-moneyness surface for a given date.

    Returns
    -------
    iv_df : DataFrame  (tenor_days × q-label columns) — annualised IV (decimal)
    k_df  : DataFrame  (tenor_days × q-label columns) — log-moneyness log(K/S_t)
    """
    date_ts  = pd.Timestamp(date_str)
    omega    = garch_params["omega"]
    alpha    = garch_params["alpha"]
    beta     = garch_params["beta"]

    if date_ts in h_series.index:
        h_pre = float(h_series[date_ts])
    else:
        avail = h_series[h_series.index <= date_ts]
        if avail.empty:
            lam, mu_j, sigma_j = (jump_params["lam"], jump_params["mu_j"],
                                   jump_params["sigma_j"])
            var_j = lam * (mu_j**2 + sigma_j**2)
            m_j2  = var_j + (lam * mu_j)**2
            h_pre = (omega + alpha * m_j2) / max(1 - alpha - beta, 1e-12)
        else:
            h_pre = float(avail.iloc[-1])

    avail_px = price_df[price_df["date"] <= date_ts].sort_values("date")
    if avail_px.empty:
        S_t = 100.0
        h_t = h_pre
    else:
        S_t = float(avail_px["close"].iloc[-1])
        if len(avail_px) >= 2:
            S_prev = float(avail_px["close"].iloc[-2])
            r_t   = np.log(S_t / S_prev)
            h_t    = omega + alpha * r_t**2 + beta * h_pre
        else:
            h_t = h_pre

    T_max   = max(int(t) for t in tenors_days)
    lam     = jump_params["lam"]
    mu_j    = jump_params["mu_j"]
    sigma_j = jump_params["sigma_j"]
    block   = _make_rand_block(T_max, n_paths, lam, mu_j, sigma_j, seed=seed)

    raw_iv, k_grid = _build_daily_iv_surface_mc(
        S_t, h_t, 0, block,
        garch_params, jump_params,
        tenors_days, q_grid, r, q,
    )
    _, essvi_info, _ = _calibrate_essvi_and_extract(raw_iv, k_grid, verbose=False)
    iv_func = essvi_info["iv_func"]

    all_labels = list(raw_iv.columns)
    iv_rows    = {}
    k_rows     = {}
    for T_days in tenors_days:
        T_int   = int(T_days)
        T_years = T_days / TRADING_DAYS
        if T_int not in k_grid:
            continue
        k_arr = k_grid[T_int]
        iv_row = {}
        k_row  = {}
        for iq, lbl in enumerate(all_labels):
            k_val       = float(k_arr[iq])
            iv_row[lbl] = float(iv_func(k_val, T_years))
            k_row[lbl]  = round(k_val, 6)
        iv_rows[T_days] = iv_row
        k_rows[T_days]  = k_row

    iv_df = pd.DataFrame(iv_rows).T[all_labels]
    k_df  = pd.DataFrame(k_rows).T[all_labels]
    iv_df.index.name = "tenor_days";  iv_df.columns.name = "q"
    k_df.index.name  = "tenor_days";  k_df.columns.name  = "q"
    return iv_df, k_df


def plot_vol_surface(date_str: str, price_df: pd.DataFrame,
                     h_series: pd.Series, garch_params: dict,
                     jump_params: dict,
                     tenors_days: tuple | list, q_grid: np.ndarray,
                     ticker: str = "",
                     out_dir: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Print and plot the model IV surface for a given date.
    Returns (iv_df, k_df) where k_df contains log(K/S_t) log-moneyness values.
    """
    iv_df, k_df = get_vol_surface_df(date_str, price_df, h_series, garch_params,
                                      jump_params, tenors_days, q_grid)

    print(f"\n── Vol Surface for {ticker} on {date_str} ─────────────────────")
    print("  IV (% annualised):")
    print((iv_df * 100).round(2).to_string())
    print("\n  Log-moneyness k = log(K/S_t):")
    print(k_df.round(4).to_string())

    try:
        import matplotlib.pyplot as plt
        col_labels = list(iv_df.columns)
        x_ticks    = np.arange(len(col_labels))
        fig, ax = plt.subplots(figsize=(9, 5))
        for T_days in tenors_days:
            if T_days in iv_df.index:
                row = iv_df.loc[T_days].values * 100
                ax.plot(x_ticks, row, marker="o", label=f"T={T_days}d")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(col_labels, rotation=45)
        ax.set_xlabel("Moneyness  (ATM = MC mean strike)")
        ax.set_ylabel("Implied vol  (%)")
        ax.set_title(f"{ticker} — IV Surface on {date_str}")
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        if out_dir:
            out_path = os.path.join(out_dir, f"{ticker}_vol_surface_{date_str}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"  [Saved] {out_path}")
    except ImportError:
        print("  [matplotlib not available — skipping plot]")

    return iv_df, k_df
