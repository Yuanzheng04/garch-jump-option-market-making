"""
garch.py
========
Step 2 — GARCH(1,1)+Jump MLE and state filtering.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import TRADING_DAYS


def _garch_filter(r: np.ndarray, omega: float, alpha: float, beta: float,
                  lam: float = 0.0, mu_j: float = 0.0,
                  sigma_j: float = 0.0,
                  h0: float | None = None) -> np.ndarray:
    """
    GARCH(1,1) + Compound-Poisson-Jump conditional variance path.

    Returns h[t] = h_{t|t-1}  (pre-update: variance forecast for period t
    computed BEFORE observing r[t]).  Specifically:
        h[0] = h0  if provided, else h̄  (stationary variance)
        h[t] = ω + α·r[t-1]² + β·h[t-1]   for t ≥ 1

    where h̄ = (ω + α·m_{J2}) / (1 − α − β),  m_{J2} = λ(μ_J²+σ_J²)+(λμ_J)².

    To obtain the post-update state after observing r[t], compute:
        h_{t+1|t} = ω + α·r[t]² + β·h[t]
    """
    phi   = alpha + beta
    denom = max(1.0 - phi, 1e-12)
    var_j = lam * (mu_j ** 2 + sigma_j ** 2)
    m_j2  = var_j + (lam * mu_j) ** 2
    h_bar = (omega + alpha * m_j2) / denom
    #基于r^对和return等长的序列进行更新
    h = np.empty(len(r))
    h_prev = max(float(h0), 1e-12) if h0 is not None else h_bar
    for t, rt in enumerate(r):
        h[t]   = h_prev
        h_prev = omega + alpha * rt ** 2 + beta * h_prev
    return np.maximum(h, 1e-12)


def _nll(params: np.ndarray, r: np.ndarray) -> float:
    """
    Negative log-likelihood for GARCH(1,1) + Compound-Poisson-Jump (Method A).

    Conditional density (Gaussian mixture):
        p(r_t | F_{t-1}) = Σ_{n=0}^{N_max} P(N_t=n) · N(r_t; n·μ_J, h_t + n·σ_J²)

    Parameters: [omega, alpha, beta, lam, mu_j, sigma_j]
    """
    omega, alpha, beta, lam, mu_j, sigma_j = params
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1.0:
        return 1e12
    if lam < 0 or sigma_j < 0:
        return 1e12

    h      = _garch_filter(r, omega, alpha, beta, lam, mu_j, sigma_j)
    N_max  = max(10, int(lam * 20 + 5))

    log_pn = -lam
    log_ll = log_pn - 0.5 * (np.log(2.0 * np.pi * h) + r ** 2 / h)

    for n in range(1, N_max + 1):
        log_pn  += np.log(max(lam, 1e-300)) - np.log(n)
        mean_n   = n * mu_j
        var_n    = h + n * sigma_j ** 2
        log_g    = -0.5 * (np.log(2.0 * np.pi * var_n) + (r - mean_n) ** 2 / var_n)
        log_term = log_pn + log_g
        lmax   = np.maximum(log_ll, log_term)
        log_ll = lmax + np.log(np.exp(log_ll - lmax) + np.exp(log_term - lmax))

    return -float(np.sum(log_ll))


def fit_garch_jump(r: np.ndarray, verbose: bool = True) -> tuple[dict, dict, dict]:
    """
    Fit GARCH(1,1)+Jump by joint MLE using SLSQP with a stationarity constraint.

    Returns
    -------
    garch_params : {omega, alpha, beta, last_var}
                   last_var = h_{T+1|T}  (post-update after the last training return)
    jump_params  : {lam, mu_j, sigma_j}
    fit_info     : {loglike, n_obs, persistence, n_jumps_implied, success}
    """
    r    = np.asarray(r, float)
    var0 = float(np.var(r))

    bounds = [
        (1e-12, 1.0   ),
        (1e-8,  0.999 ),
        (1e-8,  0.999 ),
        (1e-6,  10.0  ),
        (-0.50, 0.50  ),
        (1e-4,  0.50  ),
    ]

    _stat = {"type": "ineq", "fun": lambda x: (1.0 - 1e-6) - x[1] - x[2]}

    starts = [
        [var0 * 0.05, 0.05, 0.88, 0.008, -0.015, 0.030],
        [var0 * 0.02, 0.08, 0.85, 0.015, -0.030, 0.050],
        [var0 * 0.05, 0.10, 0.80, 0.025, -0.050, 0.080],
        [var0 * 0.03, 0.07, 0.87, 0.005, -0.010, 0.020],
    ]

    best = None
    for x0_raw in starts:
        x0 = np.clip(x0_raw, [b[0] + 1e-10 for b in bounds],
                              [b[1] - 1e-10 for b in bounds])
        if x0[1] + x0[2] >= 1.0 - 1e-6:
            x0[2] = max(bounds[2][0], 0.999 - 1e-6 - x0[1])
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                res = minimize(_nll, x0, args=(r,), bounds=bounds,
                               constraints=[_stat], method="SLSQP",
                               options={"maxiter": 3000, "ftol": 1e-13})
            if best is None or (res.success and res.fun < best.fun):
                best = res
            elif not best.success and res.fun < best.fun:
                best = res
        except Exception:
            pass

    if best is None:
        raise RuntimeError("GARCH+Jump MLE: all starting points failed.")

    omega, alpha, beta, lam, mu_j, sigma_j = best.x
    phi   = alpha + beta
    h_arr = _garch_filter(r, omega, alpha, beta, lam, mu_j, sigma_j)

    h_last_pre  = float(h_arr[-1])
    h_last_post = float(omega + alpha * r[-1]**2 + beta * h_last_pre)

    garch_params = {"omega": omega, "alpha": alpha, "beta": beta,
                    "last_var": h_last_post}
    jump_params  = {"lam": lam, "mu_j": mu_j, "sigma_j": sigma_j}
    n_jumps      = int(round(lam * len(r)))
    fit_info     = {"loglike": -best.fun, "n_obs": len(r),
                    "persistence": phi, "n_jumps_implied": n_jumps,
                    "success": best.success, "message": best.message}

    if verbose:
        var_j    = lam * (mu_j**2 + sigma_j**2)
        m_j2     = var_j + (lam * mu_j)**2
        h_bar    = (omega + alpha * m_j2) / max(1 - phi, 1e-12)
        lr_vol   = np.sqrt((h_bar + var_j) * TRADING_DAYS) * 100
        print("\n── STEP 2: GARCH+Jump MLE Results ───────────────────────────")
        print(f"  GARCH : ω={omega:.2e}  α={alpha:.4f}  β={beta:.4f}  "
              f"α+β={phi:.4f}")
        print(f"  Jump  : λ={lam:.4f}  μ_J={mu_j:.4f}  σ_J={sigma_j:.4f}")
        print(f"  h̄ (GARCH stationary, daily) = {h_bar:.6f}")
        print(f"  Long-run vol (annualised)   = {lr_vol:.2f}%")
        print(f"  Implied jumps over sample   = {n_jumps}")
        print(f"  Log-likelihood              = {-best.fun:.2f}")
        print(f"  Persistence (α+β)           = {phi:.4f}")
        rv_ann = float(np.std(r) * np.sqrt(TRADING_DAYS) * 100)
        print(f"  Realised vol (annualised)   = {rv_ann:.2f}%")

    return garch_params, jump_params, fit_info


def _filter_garch_states(price_df: pd.DataFrame, garch_params: dict,
                          jump_params: dict) -> pd.Series:
    """
    Run the GARCH filter over price_df and return h_{t|t-1} (pre-update)
    indexed by date.

    h[0] is initialised to r[0]² (first day's squared return) rather than
    the unconditional h̄, since after ≥ 6 months of burn-in the initial value
    has no influence on the backtest window.

    post-update h_{t+1|t} = ω + α·r_t² + β·h_{t|t-1}  is computed on-the-fly
    in run_backtest() and get_vol_surface_df().
    """
    prices = price_df["close"].values.astype(float)
    from .data import _log_returns
    r      = _log_returns(prices)
    h_arr  = _garch_filter(r,
                            garch_params["omega"],
                            garch_params["alpha"],
                            garch_params["beta"],
                            jump_params["lam"],
                            jump_params["mu_j"],
                            jump_params["sigma_j"],
                            h0=float(r[0] ** 2))
    dates  = price_df["date"].iloc[1:].values
    return pd.Series(h_arr, index=pd.DatetimeIndex(dates), name="h_t")
