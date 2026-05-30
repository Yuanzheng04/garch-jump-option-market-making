"""
config.py
=========
User configuration and global constants for the GARCH+Jump backtest.

Edit the USER CONFIGURATION section before running.
All other modules import from here — change a value once, it propagates.
"""

from __future__ import annotations
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION  ← edit here before running
# ═══════════════════════════════════════════════════════════════════════════════

# Tickers to run in the pipeline (run_all.py loops over this list).
# Use one or more tickers; each must have a <TICKER>.csv under DATA_DIR.
TICKERS: list[str]   = ["600128"]
    #"688092", "600128", "000677",
    # "300701", "300932", "000798", "002449", "600984", 
#"002390", "688488", "300019", "600586", "688366", "600755", "603612", "605099", "603067", "600909", "002648", "601006",      # e.g. ["603986"]  or  ["603986", "000058"]
# "603612", "605099", "603067", "600909", "002648", "601006"]       # e.g. ["603986"]  or  ["603986", "000058"]
TICKER               = TICKERS[0]          # default for single-stage scripts
DATA_DIR             = "Data/latest_data"  # directory containing <TICKER>.csv
TRAIN_START          = "2021-01-01"
TRAIN_END            = "2026-03-09"
BURN_IN_YEARS        = 0.5           # first 6 months of h_series discarded from backtest
BACKTEST_START       = "2021-07-01"  # = TRAIN_START + BURN_IN_YEARS (6-month burn-in)
BACKTEST_END         = "2026-03-09"
ENTRY_FREQ_DAYS      = 5                   # open a new option batch every N trading days
TENORS_DAYS          = (10, 21, 42, 63, 125, 252)

# Quantile moneyness grid — 5th to 95th percentile in steps of 10pp.
# Covers ≥ 90 % of all simulated paths (5 % tails trimmed on each side).
# q=0.45 is near-ATM; q=0.05 → deep ITM call; q=0.95 → deep OTM call.
Q_GRID = np.array([0.05, 0.15, 0.25, 0.35, 0.45,
                   0.55, 0.65, 0.75, 0.85, 0.95])

# Vol surface query: set to a date string to print/save that day's surface.
PRINT_VOL_SURFACE_DATE = "2026-03-09"   # or None to skip

# Monte Carlo parameters
N_BATCH       = 30_000     # epoch size for adaptive reference-surface MC
N_MAX         = 1_000_000  # hard cap on total paths (adaptive MC only)
# Paths used in the daily backtest loop (delta hedging only — fewer paths suffice).
# Slices the first N_BACKTEST columns from the pre-generated rand_block.
# Lower → faster backtest; 10_000 is sufficient for delta accuracy.
N_BACKTEST    = 10_000

# eSSVI calibration: wing emphasis weight (1 = no emphasis; 2 = wings weighted 2×)
# Higher value makes eSSVI fit OTM wings more tightly at the cost of mid-smile accuracy.
WEIGHT_WINGS  = 1

# Market rates
R_ANNUAL    = 0.0
Q_ANNUAL    = 0.0
SEED        = 42

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 PARAMETERS  (three-step quoting engine)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Three-step pipeline:
#   S4-10   MC repricing with jump-param adjustments → p_ask_mc / p_bid_mc
#   S4-10b  HE bias correction: shift the mid vol level using μ_HE
#           (μ_HE > 0 → model overpriced → lower mid vol; μ_HE < 0 → raise it)
#   S4-11   HE vol-spread overlay using σ_HE only → widen ask / tighten bid
#   S4-12   Butterfly arb fix
#
# Separation of concerns:
#   μ_HE (systematic bias)      → handled exclusively in S4-10b (mid correction)
#   σ_HE (hedge uncertainty)    → handled exclusively in S4-11  (spread width)

# ── S4-10: Jump-param adjustments for bid / ask MC repricing ──────────────────
# Ask side: heavier tails → fatter distribution → higher implied vol → higher ask
S4_ASK_LAM_MULT     = 1.20   # lambda multiplier  (20% more frequent jumps)
S4_ASK_MU_J_ADD     = -0.0   # additive shift to mu_J
S4_ASK_SIGMA_J_MULT = 1.0    # sigma_J multiplier

# Bid side: lighter tails → thinner distribution → lower implied vol → lower bid
S4_BID_LAM_MULT     = 0.80   # lambda multiplier  (20% fewer expected jumps)
S4_BID_MU_J_ADD     = 0.0    # additive shift to mu_J
S4_BID_SIGMA_J_MULT = 1.0    # sigma_J multiplier

# MC paths for Stage 4 one-shot surface repricing (no adaptive convergence needed)
S4_N_PATHS = 50_000

# ── S4-10b: HE bias correction ────────────────────────────────────────────────
# Shifts the mid vol (and both ask_mc / bid_mc) by –bias_coef × μ_HE / vega_eff
# so that the fair price corrects for the systematic model mispricing identified
# in Stage 3.  The spread produced by S4-10 and S4-11 is preserved unchanged.
#
#   Δσ_bias[T,q] = clip( –S4_HE_BIAS_COEF × μ_HE / vega_eff,
#                         –S4_HE_BIAS_CLIP, +S4_HE_BIAS_CLIP )
#   iv_mid_adj   = iv_model  + Δσ_bias     (bias-corrected mid, used as new fair)
#   iv_ask_mc_adj = iv_ask_mc + Δσ_bias    (shift preserves step-1 spread)
#   iv_bid_mc_adj = iv_bid_mc + Δσ_bias
#
S4_HE_BIAS_COEF = 1.0    # fraction of μ_HE bias to correct (0 = off, 1 = full)
S4_HE_BIAS_CLIP = 0.10   # hard cap on |Δσ_bias| in vol-point units (10 vol pts)

# ── S4-11: HE vol-spread overlay (σ_HE only, regularised vega) ────────────────
# Applied on top of the bias-corrected IV surfaces from S4-10b.
# μ_HE is NOT used here — it is fully absorbed by S4-10b.
#
#   vega_eff(T,K) = √[ (φ(d₁)·√T)²  +  ε(T)² ]
#       where  ε(T) = S4_VEGA_REG_ALPHA × 0.4 × √T_years
#
#   Δσ_ask[T,q] = max( S4_ASK_SIGMA_COEF × σ_HE / vega_eff,  0 )
#   Δσ_bid[T,q] = max( S4_BID_SIGMA_COEF × σ_HE / vega_eff,  0 )
#
#   σ_ask_final = σ_ask_mc_adj + Δσ_ask     → BS → p_ask
#   σ_bid_final = max(σ_bid_mc_adj − Δσ_bid, σ_MIN)  → BS → p_bid (also floored)
#
S4_VEGA_REG_ALPHA = 0.10  # ε(T) = alpha × 0.4 × √T  (regularised vega floor)
S4_ASK_SIGMA_COEF = 0.5   # weight on σ_HE for ask spread
S4_BID_SIGMA_COEF = 0.5   # weight on σ_HE for bid spread
P_BID_MIN         = 0.0001  # absolute bid floor = 0.01% × S₀  (prevents zero bid)

# ── S4-11 minimum vol-spread floor (ensures OTM wings are never narrower than ATM) ─
# Without this floor the bid-ask spread in vol-space is ~ σ_HE / vega_eff.
# Because vega_eff is regularised (floored by ε) the spread is roughly flat
# across strikes rather than widening at OTM.  A per-node floor of the form
#   Δσ_floor = S4_SPREAD_FLOOR_COEF × (ε_OTM / vega_eff)
# adds an intrinsic minimum that grows as vega_eff shrinks OTM.
# Setting S4_SPREAD_FLOOR_COEF = 0 disables the floor (original behaviour).
S4_SPREAD_FLOOR_COEF = 0.5   # intrinsic minimum vol-spread per unit ε/vega_eff

# ── S4-12: Butterfly arb fix ───────────────────────────────────────────────────
ARB_FIX_MAX_ITERS = 5    # max iterative passes per surface per tenor
ARB_FIX_EPS       = 1e-6 # convexity margin (push below midpoint by this amount)

# ── Stage 4 display: nodes shown in the summary print ─────────────────────────
TARGET_POINTS: list[tuple[int, str]] = [
    (21,  "ATM"), (21,  "q35"), (21,  "q65"),
    (42,  "ATM"), (42,  "q25"), (42,  "q75"),
    (63,  "ATM"), (125, "ATM"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

TRADING_DAYS = 252
