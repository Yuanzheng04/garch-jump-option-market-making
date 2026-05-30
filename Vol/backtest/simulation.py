"""
simulation.py
=============
GARCH+Jump Monte Carlo simulation helpers.
"""

from __future__ import annotations
import numpy as np


def _make_rand_block(T_days: int, n_paths: int,
                      lam: float, mu_j: float, sigma_j: float,
                      seed: int) -> dict:
    """
    Pre-generate all random variables for T_days × n_paths in one shot.

    Uses the aggregate Gaussian property of Compound Poisson:
        J_t = N_t·μ_J + √N_t·σ_J·ξ_t,    ξ_t ~ N(0,1)

    Returns
    -------
    dict with keys:
        Z : (T_days, n_paths)  diffusion shocks ~ N(0,1)
        J : (T_days, n_paths)  aggregate jump per step
    """
    rng    = np.random.default_rng(seed)
    Z      = rng.standard_normal((T_days, n_paths))
    N_jump = rng.poisson(lam, (T_days, n_paths))
    Xi     = rng.standard_normal((T_days, n_paths))

    sqrt_N = np.sqrt(N_jump.astype(float))
    J = np.where(N_jump > 0,
                 N_jump * mu_j + sqrt_N * sigma_j * Xi,
                 0.0)
    return {"Z": Z, "J": J}


def _simulate_from_block(S0: float, h0: float, T_days: int,
                          rand_block: dict,
                          garch_params: dict, jump_params: dict,
                          r: float = 0.0, q: float = 0.0) -> np.ndarray:
    """
    Simulate terminal stock prices using a pre-generated random block.

    Q-measure log-return (full martingale correction):
        Δlog S_t = √h_t·Z_t + J_t  −  ½h_t  −  μ_comp_jump

    Returns
    -------
    np.ndarray  shape (n_paths,)  — terminal stock prices S_{t+T}
    """
    Z = rand_block["Z"][:T_days]
    J = rand_block["J"][:T_days]
    n_paths = Z.shape[1]

    omega   = garch_params["omega"]
    alpha   = garch_params["alpha"]
    beta    = garch_params["beta"]
    mu_j    = jump_params["mu_j"]
    sigma_j = jump_params["sigma_j"]
    lam     = jump_params["lam"]

    mu_comp_jump = lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1.0)

    h     = np.full(n_paths, max(h0, 1e-12), dtype=float)
    log_S = np.full(n_paths, np.log(S0), dtype=float)

    for t in range(T_days):
        eps_total  = np.sqrt(h) * Z[t] + J[t]
        log_S     += eps_total - 0.5 * h - mu_comp_jump
        h          = omega + alpha * eps_total**2 + beta * h
        h          = np.maximum(h, 1e-12)

    return np.exp(log_S)
