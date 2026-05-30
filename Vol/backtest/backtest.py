"""
backtest.py
===========
Step 5 — In-sample backtest with daily MC vol surface + delta hedging.

Calls trading.py for all position-specific logic.
To change strategy: edit trading.py only.
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd

from .config import TRADING_DAYS, ENTRY_FREQ_DAYS, R_ANNUAL, Q_ANNUAL, N_BACKTEST
from .vol_surface import _build_daily_iv_surface_mc, _calibrate_essvi_and_extract
from .trading import open_position, expiry_payoff, live_delta, STRATEGY_NAME


def run_backtest(price_df: pd.DataFrame, h_series: pd.Series,
                 garch_params: dict, jump_params: dict,
                 rand_block: dict,
                 backtest_start: str, backtest_end: str,
                 tenors_days: tuple | list, q_grid: np.ndarray,
                 entry_freq: int = ENTRY_FREQ_DAYS,
                 r: float = R_ANNUAL, q: float = Q_ANNUAL,
                 verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    In-sample backtest: open short positions across all (q, T) every
    `entry_freq` trading days, hold to expiry with daily BS delta hedging.

    Strategy is defined in trading.py (currently: short straddle).

    Seller perspective:
        HE = premium − payoff + hedge_pnl
        HE > 0  →  seller profitable

    Returns
    -------
    trades_df : pd.DataFrame
        One row per closed trade:
        trade_id, entry_date, expiry_date, q, T_days, K, S_entry,
        premium, payoff, hedge_pnl, HE, HE_per_S

    daily_log_df : pd.DataFrame
        One row per (trade × day) covering entry through expiry:
        trade_id, date, day_in_trade, S_t, K, tau_days, iv_t,
        delta_used, dS, sub_hedge_pnl, cum_hedge_pnl
        hedge_pnl in trades_df == daily_log_df.groupby("trade_id")["sub_hedge_pnl"].sum()
    """
    bt_df = price_df[(price_df["date"] >= pd.Timestamp(backtest_start)) &
                     (price_df["date"] <= pd.Timestamp(backtest_end))].reset_index(drop=True)

    if len(bt_df) < 2:
        raise ValueError("Backtest window has fewer than 2 trading days.")

    trade_log   = []
    daily_log   = []
    trade_id_ctr = 0
    dates       = bt_df["date"].values
    closes      = bt_df["close"].values.astype(float)

    pre_bt_rows = price_df[price_df["date"] < pd.Timestamp(backtest_start)]
    if pre_bt_rows.empty:
        raise ValueError(
            f"No price data before BACKTEST_START ({backtest_start}). "
            "BACKTEST_START must be at least one trading day after TRAIN_START."
        )
    S_prev_day0 = float(pre_bt_rows["close"].iloc[-1])

    n_bt = len(dates)
    T_max_bt = max(int(t) for t in tenors_days)
    T_needed = n_bt + T_max_bt
    if rand_block["Z"].shape[0] < T_needed:
        raise ValueError(
            f"rand_block has {rand_block['Z'].shape[0]} rows but backtest needs "
            f"{T_needed} (={n_bt} backtest days + {T_max_bt} max tenor). "
            f"Re-generate with T_total >= {T_needed}."
        )

    t0_bt = time.time()
    if verbose:
        print(f"\n── STEP 5: Backtest {backtest_start} → {backtest_end} ────────────")
        print(f"  Strategy: {STRATEGY_NAME}")
        print(f"  Trading days: {n_bt}  |  Entry every {entry_freq} days  "
              f"|  Tenors: {list(tenors_days)}  |  q-grid: {len(q_grid)} quantiles")
        print(f"  Daily MC paths: {rand_block['Z'].shape[1]:,}")

    omega = garch_params["omega"]
    alpha = garch_params["alpha"]
    beta  = garch_params["beta"]

    for day_idx, (date, S_t) in enumerate(zip(dates, closes)):
        date_ts = pd.Timestamp(date)

        if date_ts not in h_series.index:
            raise KeyError(
                f"h_series missing date {date_ts.date()}. "
                "Ensure BACKTEST_START >= h_series start date and re-run stage1."
            )
        h_pre = float(h_series[date_ts])

        S_prev = closes[day_idx - 1] if day_idx > 0 else S_prev_day0
        r_t    = np.log(S_t / S_prev)
        h_t    = omega + alpha * r_t**2 + beta * h_pre

        # Use N_BACKTEST paths (< N_block) — delta accuracy does not require
        # the full reference-surface path count.
        raw_iv_t, k_grid_t = _build_daily_iv_surface_mc(
            S_t, h_t, day_idx, rand_block,
            garch_params, jump_params,
            tenors_days, q_grid, r, q,
            n_paths=N_BACKTEST,
        )
        # Coarser rho grid (8 vs 20) — adequate for daily delta; saves ~2.5× eSSVI time.
        _, essvi_t, _ = _calibrate_essvi_and_extract(
            raw_iv_t, k_grid_t, verbose=False, n_rho_grid=8,
        )
        iv_func_t = essvi_t["iv_func"]

        # ── Open new trades on entry days ──────────────────────────────────────
        if day_idx % entry_freq == 0:
            for T_days in tenors_days:
                T_years  = T_days / TRADING_DAYS
                T_int    = int(T_days)
                if T_int not in k_grid_t:
                    continue
                k_arr_t    = k_grid_t[T_int]
                col_labels = list(raw_iv_t.columns)
                for iq_all, col_lbl in enumerate(col_labels):
                    if col_lbl == "ATM":
                        q_val = -1.0
                    else:
                        q_val = int(col_lbl[1:]) / 100.0   # "q75" → 0.75
                    k   = float(k_arr_t[iq_all])
                    K   = float(S_t * np.exp(k))
                    iv  = float(iv_func_t(k, T_years))
                    if not np.isfinite(iv) or iv <= 0:
                        continue

                    # ── Strategy-specific: open_position from trading.py ──────
                    premium, delta, vega = open_position(S_t, K, T_years, iv, r, q)

                    tid = trade_id_ctr
                    trade_id_ctr += 1

                    trade_log.append({
                        "trade_id":   tid,
                        "open_day":   day_idx,
                        "entry_date": date,
                        "expiry_date":None,
                        "q":          float(q_val),
                        "T_days":     T_days,
                        "K":          K,
                        "S_entry":    S_t,
                        "premium":    premium,
                        "vega_entry": vega,
                        "payoff":     None,
                        "hedge_pnl":  0.0,
                        "delta_eod":  delta,   # current end-of-day hedge position
                        "closed":     False,
                        "days_left":  T_days,
                    })

                    # Entry-day log: records delta established at open (= EOD delta for day 0).
                    # dS=0 and sub_hedge_pnl=0 because no price movement before first hedge.
                    # Verification: sub_hedge_pnl[row d] == delta_eod[row d-1] * dS[row d]
                    daily_log.append({
                        "trade_id":      tid,
                        "date":          date,
                        "day_in_trade":  0,
                        "S_t":           S_t,
                        "q":             float(q_val),
                        "K":             K,
                        "tau_days":      T_days,
                        "iv_t":          iv,
                        "delta_eod":     delta,  # EOD delta: applied to NEXT day's dS
                        "dS":            0.0,
                        "sub_hedge_pnl": 0.0,
                        "cum_hedge_pnl": 0.0,
                    })

        # ── Update existing open trades ────────────────────────────────────────
        for trade in trade_log:
            if trade["closed"]:
                continue
            if trade["open_day"] == day_idx:
                continue

            trade["days_left"] -= 1
            tau_days  = trade["days_left"]
            K         = trade["K"]
            dS        = S_t - closes[day_idx - 1]
            sub_pnl   = trade["delta_eod"] * dS   # yesterday's EOD delta × today's dS
            trade["hedge_pnl"] += sub_pnl
            day_in_t  = int(trade["T_days"]) - tau_days

            if tau_days <= 0:
                # ── Strategy-specific: expiry payoff from trading.py ──────────
                payoff              = expiry_payoff(S_t, K)
                trade["payoff"]     = payoff
                trade["expiry_date"]= date
                trade["closed"]     = True
                delta_eod_new       = float("nan")  # position closed, no new delta
                iv_used             = float("nan")
            else:
                tau_years     = tau_days / TRADING_DAYS
                k_log         = np.log(S_t / K)
                iv_used       = float(iv_func_t(k_log, tau_years))
                if not np.isfinite(iv_used) or iv_used <= 0:
                    iv_used = 0.3
                # ── Strategy-specific: live_delta from trading.py ─────────────
                delta_eod_new       = live_delta(S_t, K, tau_years, iv_used, r, q)
                trade["delta_eod"]  = delta_eod_new   # update for next day

            # delta_eod = NEW end-of-day delta (what will be applied to NEXT day's dS).
            # Verification: sub_hedge_pnl == delta_eod[prev row] * dS[this row]
            daily_log.append({
                "trade_id":      trade["trade_id"],
                "date":          date,
                "day_in_trade":  day_in_t,
                "S_t":           S_t,
                "q":             trade["q"],
                "K":             K,
                "tau_days":      tau_days,
                "iv_t":          iv_used,
                "delta_eod":     delta_eod_new,
                "dS":            dS,
                "sub_hedge_pnl": sub_pnl,
                "cum_hedge_pnl": trade["hedge_pnl"],
            })

    # ── Assemble results ───────────────────────────────────────────────────────
    closed_trades = [t for t in trade_log if t["closed"]]
    if not closed_trades:
        print("  [WARNING] No trades were closed — check the backtest window.")
        return pd.DataFrame(), pd.DataFrame()

    records = []
    for t in closed_trades:
        HE       = t["premium"] - t["payoff"] + t["hedge_pnl"]
        S_e      = max(t["S_entry"], 1e-8)
        HE_per_S = HE / S_e
        records.append({
            "trade_id":    t["trade_id"],
            "entry_date":  t["entry_date"],
            "expiry_date": t["expiry_date"],
            "q":           t["q"],
            "T_days":      t["T_days"],
            "K":           round(t["K"], 4),
            "S_entry":     t["S_entry"],
            "premium":     t["premium"],
            "payoff":      t["payoff"],
            "hedge_pnl":   t["hedge_pnl"],
            "HE":          HE,
            "HE_per_S":    HE_per_S,
        })

    trades_df    = pd.DataFrame(records)
    closed_ids   = set(trades_df["trade_id"])
    daily_log_df = pd.DataFrame([r for r in daily_log if r["trade_id"] in closed_ids])

    if verbose:
        bt_elapsed = time.time() - t0_bt
        print(f"  Closed trades: {len(trades_df)}  |  "
              f"Open (not expired): {sum(1 for t in trade_log if not t['closed'])}")
        print(f"  [Step 5 time: {bt_elapsed:.1f}s]")
    return trades_df, daily_log_df
