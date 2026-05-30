"""
trading.py
==========
Strategy module — defines the position opened and closed each day.

┌─────────────────────────────────────────────────────────────────────────┐
│  TO SWITCH STRATEGY: change ACTIVE_STRATEGY below.  Nothing else        │
│  in the codebase needs editing.                                          │
│                                                                          │
│  Available strategies:                                                   │
│    "short_call"      — sell call only at strike K                        │
│    "short_put"       — sell put  only at strike K                        │
│    "short_straddle"  — sell call + put at same K  (symmetric payoff)    │
└─────────────────────────────────────────────────────────────────────────┘

Seller perspective throughout:
    HE = premium − payoff + hedge_pnl
    HE > 0  →  position profitable for the seller

Delta sign convention:
    short_call   : delta = N(d1)       > 0   (buy shares to hedge)
    short_put    : delta = N(d1) − 1   < 0   (short shares to hedge)
    short_straddle: delta = 2·N(d1)−1         (small positive near ATM)
"""

from __future__ import annotations
from .pricing import _bs_call, _bs_put, _bs_delta, _bs_vega

# ══════════════════════════════════════════════════════════════════════════════
# ▶  CHANGE THIS LINE TO SWITCH STRATEGY
ACTIVE_STRATEGY: str = "short_call"
# ══════════════════════════════════════════════════════════════════════════════

_VALID = {"short_straddle", "short_call", "short_put"}
if ACTIVE_STRATEGY not in _VALID:
    raise ValueError(f"Unknown strategy '{ACTIVE_STRATEGY}'. Choose from {_VALID}")

STRATEGY_NAME: str = ACTIVE_STRATEGY   # used for print / CSV labels


# ── Private implementation: short straddle ────────────────────────────────────

def _open_short_straddle(S: float, K: float, T_years: float,
                          iv: float, r: float, q: float,
                          ) -> tuple[float, float, float]:
    """Sell one call + one put at K.  premium = C + P,  delta = 2·N(d1)−1."""
    call_prem = _bs_call(S, K, T_years, iv, r, q)
    put_prem  = _bs_put( S, K, T_years, iv, r, q)
    premium   = call_prem + put_prem
    delta     = 2.0 * _bs_delta(S, K, T_years, iv, r, q) - 1.0
    vega      = 2.0 * _bs_vega( S, K, T_years, iv, r, q)
    return premium, delta, vega


def _expiry_payoff_straddle(S_T: float, K: float) -> float:
    """Seller pays |S_T − K|."""
    return abs(S_T - K)


def _live_delta_straddle(S: float, K: float, tau: float,
                          iv: float, r: float, q: float) -> float:
    return 2.0 * _bs_delta(S, K, tau, iv, r, q) - 1.0


# ── Private implementation: short put ────────────────────────────────────────

def _open_short_put(S: float, K: float, T_years: float,
                    iv: float, r: float, q: float,
                    ) -> tuple[float, float, float]:
    """Sell one put at K.  premium = P,  delta = N(d1) − 1  (negative)."""
    premium = _bs_put(  S, K, T_years, iv, r, q)
    delta   = _bs_delta(S, K, T_years, iv, r, q) - 1.0   # put delta
    vega    = _bs_vega( S, K, T_years, iv, r, q)
    return premium, delta, vega


def _expiry_payoff_put(S_T: float, K: float) -> float:
    """Seller pays max(K − S_T, 0)."""
    return max(K - S_T, 0.0)


def _live_delta_put(S: float, K: float, tau: float,
                    iv: float, r: float, q: float) -> float:
    return _bs_delta(S, K, tau, iv, r, q) - 1.0


# ── Private implementation: short call ───────────────────────────────────────

def _open_short_call(S: float, K: float, T_years: float,
                      iv: float, r: float, q: float,
                      ) -> tuple[float, float, float]:
    """Sell one call at K.  premium = C,  delta = N(d1)."""
    premium = _bs_call( S, K, T_years, iv, r, q)
    delta   = _bs_delta(S, K, T_years, iv, r, q)
    vega    = _bs_vega( S, K, T_years, iv, r, q)
    return premium, delta, vega


def _expiry_payoff_call(S_T: float, K: float) -> float:
    """Seller pays max(S_T − K, 0)."""
    return max(S_T - K, 0.0)


def _live_delta_call(S: float, K: float, tau: float,
                      iv: float, r: float, q: float) -> float:
    return _bs_delta(S, K, tau, iv, r, q)


# ── Dispatch table ────────────────────────────────────────────────────────────

_IMPL = {
    "short_straddle": {
        "open":          _open_short_straddle,
        "expiry_payoff": _expiry_payoff_straddle,
        "live_delta":    _live_delta_straddle,
    },
    "short_call": {
        "open":          _open_short_call,
        "expiry_payoff": _expiry_payoff_call,
        "live_delta":    _live_delta_call,
    },
    "short_put": {
        "open":          _open_short_put,
        "expiry_payoff": _expiry_payoff_put,
        "live_delta":    _live_delta_put,
    },
}


# ── Public interface — called by backtest.py ──────────────────────────────────

def open_position(S: float, K: float, T_years: float, iv: float,
                  r: float, q: float) -> tuple[float, float, float]:
    """
    Open a short position.

    Returns
    -------
    premium : float   total premium received
    delta   : float   BS hedge delta at entry
    vega    : float   BS vega at entry
    """
    return _IMPL[ACTIVE_STRATEGY]["open"](S, K, T_years, iv, r, q)


def expiry_payoff(S_T: float, K: float) -> float:
    """Amount the seller pays to the buyer at expiry."""
    return _IMPL[ACTIVE_STRATEGY]["expiry_payoff"](S_T, K)


def live_delta(S: float, K: float, tau_years: float, iv: float,
               r: float, q: float) -> float:
    """Delta for daily rebalancing during the life of the trade."""
    return _IMPL[ACTIVE_STRATEGY]["live_delta"](S, K, tau_years, iv, r, q)


