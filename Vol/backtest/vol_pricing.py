"""
vol_pricing.py
==============
Vol surface quoting utilities used by Stage 4:
  - _fix_arb_grid           : fast calendar + butterfly fix directly on discrete grid
  - format_vol_quotes_table : reshape bid/ask surface into a display table
  - test_quoted_surface     : proxy win-rate test using Stage-2 fair-hedge P&L
  - _label_to_q             : label string → q float helper
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from .config  import TRADING_DAYS, R_ANNUAL, Q_ANNUAL
from .pricing import _bs_call, _bs_put, _bs_vega, _invert_iv


# ── helpers ──────────────────────────────────────────────────────────────────

def _label_to_q(label: str) -> float:
    """Convert column label → q value used as index in he_stats."""
    if label == "ATM":
        return -1.0
    return int(label[1:]) / 100.0   # "q75" → 0.75


def _fix_arb_grid(
        iv_df:        pd.DataFrame,
        k_df:         pd.DataFrame,
        trading_days: int = TRADING_DAYS,
) -> pd.DataFrame:
    """
    Fast arb-free correction directly on a discrete IV grid (no eSSVI).

    Pass 1 — calendar arb-free:
        For each moneyness label (column), enforce total variance
        w(k, T) = σ(k,T)² × T_years  non-decreasing in T.
        Violated points are raised to match the previous tenor's w.

    Pass 2 — butterfly arb-free (convexity):
        For each tenor (row), enforce w(k) is convex in k, which is a
        necessary condition for non-negative risk-neutral density.
        A concave point is raised to the linear interpolant of its neighbours.

    Both passes are O(n_tenors × n_labels) — microseconds for 66 points.
    Suitable for win-rate validation; the final quoted surface still uses eSSVI.
    """
    result = iv_df.copy().astype(float)
    tenors = sorted(int(t) for t in result.index)
    labels = list(result.columns)

    # Pass 1: calendar arb — forward scan per label (column)
    for lbl in labels:
        w_prev = 0.0
        for T in tenors:
            T_y = T / trading_days
            s   = float(result.loc[T, lbl])
            if not (np.isfinite(s) and s > 0):
                continue
            w = s ** 2 * T_y
            if w < w_prev - 1e-14:
                result.loc[T, lbl] = float(np.sqrt(w_prev / T_y))
            else:
                w_prev = w

    # Pass 2: butterfly arb — convexity in k per tenor (row)
    for T in tenors:
        T_y   = T / trading_days
        k_arr = np.array([float(k_df.loc[T, lbl]) for lbl in labels], dtype=float)
        w_arr = np.array(
            [float(result.loc[T, lbl]) ** 2 * T_y for lbl in labels], dtype=float
        )
        n       = len(labels)
        changed = False
        for i in range(1, n - 1):
            if not (np.isfinite(w_arr[i - 1]) and
                    np.isfinite(w_arr[i]) and
                    np.isfinite(w_arr[i + 1])):
                continue
            dk_l = float(k_arr[i]     - k_arr[i - 1])
            dk_r = float(k_arr[i + 1] - k_arr[i])
            if dk_l <= 0 or dk_r <= 0:
                continue
            # Minimum w at k[i] that preserves convexity: linear interp of neighbours
            w_min = (w_arr[i - 1] * dk_r + w_arr[i + 1] * dk_l) / (dk_l + dk_r)
            if w_arr[i] < w_min - 1e-14:
                w_arr[i] = w_min
                changed  = True
        if changed:
            s_arr = np.sqrt(np.maximum(w_arr / T_y, 0.0))
            for j, lbl in enumerate(labels):
                result.loc[T, lbl] = float(s_arr[j])

    return result



# ── core: build quoted vol surface ───────────────────────────────────────────

# ── empirical win-rate test ───────────────────────────────────────────────────

def test_quoted_surface(
        quotes:              dict[str, pd.DataFrame],
        trades:              pd.DataFrame,
        strategy:            str   = "short_call",
        target_ask_win_rate: float = 0.95,
        target_bid_win_rate: float = 0.95,
        r:                   float = R_ANNUAL,
        q_div:               float = Q_ANNUAL,
        trading_days:        int   = TRADING_DAYS,
        verbose:             bool  = True,
) -> pd.DataFrame:
    """
    Proxy win-rate test using the original fair-vol hedge P&L from Stage 2.

    Both ask and bid tests share the same Stage-2 hedge P&L (hedged at σ_fair),
    and check whether the quoted overlay premium is sufficient to cover the
    historical hedging residual.

    Ask test — seller receives prem_ask, hedges at σ_fair:
      residual  = payoff − hedge_pnl_fair     (net cost after hedging)
      Win_ask   = 1  if  prem_ask > residual
      Interpretation: the ask overlay premium covers the fair-hedge residual.

    Bid test — buyer pays prem_bid, benefits from σ_fair hedge running in reverse:
      residual  = payoff − hedge_pnl_fair
      Win_bid   = 1  if  residual > prem_bid
      Interpretation: fair-hedge residual exceeds bid premium (buyer earns net).

    Both pass at ≥ 95% target win rate.

    Returns DataFrame with columns:
        T_days, q_label, q_val, n_trades,
        ask_win_rate, ask_target, ask_pass,
        bid_win_rate, bid_target, bid_pass
    """
    if trades.empty:
        return pd.DataFrame()

    ask_df = quotes["ask"]
    bid_df = quotes["bid"]

    results = []
    tenors  = sorted(trades["T_days"].unique())

    for T_days in tenors:
        T_years = int(T_days) / trading_days
        mask_T  = trades["T_days"] == T_days
        qs      = sorted(trades.loc[mask_T, "q"].unique())

        for q_val in qs:
            mask = mask_T & (trades["q"] == q_val)
            sub  = trades.loc[mask]
            if sub.empty:
                continue

            lbl = "ATM" if q_val == -1.0 else f"q{int(q_val*100):02d}"
            if T_days not in ask_df.index or lbl not in ask_df.columns:
                continue

            sigma_ask = float(ask_df.loc[T_days, lbl])
            sigma_bid = float(bid_df.loc[T_days, lbl]) if (
                T_days in bid_df.index and lbl in bid_df.columns) else float("nan")

            if not (np.isfinite(sigma_ask) and sigma_ask > 0):
                continue

            wins_ask = []
            wins_bid = []

            for _, row in sub.iterrows():
                S_e    = float(row["S_entry"])
                K      = float(row["K"])
                payoff = float(row["payoff"])

                # ── Premium at ask and bid ────────────────────────────────────
                call_ask = _bs_call(S_e, K, T_years, sigma_ask, r, q_div)
                put_ask  = _bs_put( S_e, K, T_years, sigma_ask, r, q_div)
                if strategy == "short_straddle":
                    prem_ask = call_ask + put_ask
                elif strategy == "short_put":
                    prem_ask = put_ask
                else:
                    prem_ask = call_ask

                if np.isfinite(sigma_bid) and sigma_bid > 0:
                    call_bid = _bs_call(S_e, K, T_years, sigma_bid, r, q_div)
                    put_bid  = _bs_put( S_e, K, T_years, sigma_bid, r, q_div)
                    if strategy == "short_straddle":
                        prem_bid = call_bid + put_bid
                    elif strategy == "short_put":
                        prem_bid = put_bid
                    else:
                        prem_bid = call_bid
                else:
                    prem_bid = float("nan")

                # ── Fair-vol hedge P&L (from Stage 2) ────────────────────────
                hedge_pnl_fair = float(row["hedge_pnl"])
                residual = payoff - hedge_pnl_fair

                # ── Ask: seller wins if overlay premium covers residual ───────
                wins_ask.append(1 if prem_ask > residual else 0)

                # ── Bid: buyer wins if residual exceeds bid premium ──────────
                if np.isfinite(prem_bid):
                    wins_bid.append(1 if residual > prem_bid else 0)

            n        = len(wins_ask)
            wr_ask   = float(np.mean(wins_ask))
            wr_bid   = float(np.mean(wins_bid)) if wins_bid else float("nan")
            pass_ask = wr_ask >= target_ask_win_rate - 0.01
            pass_bid = (wr_bid >= target_bid_win_rate - 0.01
                        if np.isfinite(wr_bid) else False)

            results.append({
                "T_days":      T_days,
                "q_label":     lbl,
                "q_val":       q_val,
                "n_trades":    n,
                "ask_win_rate":round(wr_ask, 4),
                "ask_target":  target_ask_win_rate,
                "ask_pass":    pass_ask,
                "bid_win_rate":round(wr_bid, 4) if np.isfinite(wr_bid) else float("nan"),
                "bid_target":  target_bid_win_rate,
                "bid_pass":    pass_bid,
            })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    if verbose:
        print("\n── Vol Surface Win-Rate Test (proxy: fair-hedge residual) ─────")
        print(f"  Hedge: Stage-2 σ_fair hedge P&L  |  Strategy: {strategy}")
        print(f"  Ask: prem_ask > residual ≥ {target_ask_win_rate:.0%}  |  "
              f"Bid: residual > prem_bid ≥ {target_bid_win_rate:.0%}")
        n_ask = int(df["ask_pass"].sum())
        n_bid = int(df["bid_pass"].sum())
        n_tot = len(df)
        print(f"  Ask nodes passing: {n_ask} / {n_tot}   "
              f"Bid nodes passing: {n_bid} / {n_tot}")
        cols = ["T_days", "q_label", "n_trades",
                "ask_win_rate", "ask_pass",
                "bid_win_rate", "bid_pass"]
        print(df[cols].to_string(index=False))

    return df


# ── formatted output table ───────────────────────────────────────────────────

def format_vol_quotes_table(
        quotes:      dict[str, pd.DataFrame],
        include_mid: bool = False,
) -> pd.DataFrame:
    """
    Reshape bid/ask (and optionally mid) vol surfaces into a formatted table
    identical in structure to the IV surface CSV:
        rows    = (tenor_days, metric)   e.g. (21, 'bid_vol%'), (21, 'ask_vol%')
        columns = moneyness labels       e.g. q05, q10, ..., ATM, ..., q95

    Parameters
    ----------
    quotes      : dict with keys 'mid', 'ask', 'bid'  (DataFrames T_days × label)
    include_mid : if True, add a 'mid_vol%' row between bid and ask

    Returns
    -------
    pd.DataFrame with MultiIndex rows (tenor_days, metric)
    """
    layers = ["bid", "ask"] if not include_mid else ["bid", "mid", "ask"]
    rows = []
    for T_days in sorted(quotes["bid"].index):
        for layer in layers:
            df = quotes[layer]
            if T_days not in df.index:
                continue
            vals = (df.loc[T_days] * 100).round(2)
            rows.append(vals.rename((T_days, f"{layer}_vol%")))

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result.index = pd.MultiIndex.from_tuples(
        result.index, names=["tenor_days", "metric"])
    return result


