"""
pricing.py
==========
Black-Scholes option pricing helpers.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .config import TRADING_DAYS


def _bs_call(S: float, K: float, T: float, sigma: float,
             r: float = 0.0, q: float = 0.0) -> float:
    """Black-Scholes European call price.  Returns intrinsic value when T≤0."""
    if T <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if sigma <= 0:
        return max(np.exp(-r * T) * (S * np.exp((r - q) * T) - K), 0.0)
    vt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vt
    d2 = d1 - vt
    return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _bs_put(S: float, K: float, T: float, sigma: float,
            r: float = 0.0, q: float = 0.0) -> float:
    """
    Black-Scholes European put price via put-call parity.
    p = c - S·e^{-qT} + K·e^{-rT}
    """
    return _bs_call(S, K, T, sigma, r, q) - S * np.exp(-q * T) + K * np.exp(-r * T)


def _bs_delta(S: float, K: float, T: float, sigma: float,
              r: float = 0.0, q: float = 0.0) -> float:
    """Black-Scholes delta for a European call (= e^{-q T} N(d1))."""
    if T <= 1e-8 or sigma <= 0:
        return float(np.exp(-q * T) * (1.0 if S > K else 0.0))
    vt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vt
    return float(np.exp(-q * T) * norm.cdf(d1))


def _bs_vega(S: float, K: float, T: float, sigma: float,
             r: float = 0.0, q: float = 0.0) -> float:
    """Black-Scholes vega (∂C/∂σ).  Returns 0 when T≤0 or sigma≤0."""
    if T <= 1e-8 or sigma <= 0:
        return 0.0
    vt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vt
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


def _invert_iv(price: float, S: float, K: float, T: float,
               r: float = 0.0, q: float = 0.0,
               lo: float = 1e-6, hi: float = 10.0) -> float:
    """
    Invert BS call price to implied vol via Brentq bisection.
    Returns NaN if price is outside the arbitrage-free range.
    """
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if T <= 0 or price <= intrinsic + 1e-10:
        return float("nan")
    try:
        def f(sig):
            return _bs_call(S, K, T, sig, r, q) - price
        return float(brentq(f, lo, hi, xtol=1e-8, maxiter=200))
    except Exception:
        return float("nan")
