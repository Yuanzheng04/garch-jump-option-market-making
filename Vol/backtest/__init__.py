"""
backtest — GARCH+Jump dynamic backtest package.

Public API (mirrors the old dynamic_backtest.py for drop-in compatibility):
"""

from .config import TRADING_DAYS, Q_GRID, TENORS_DAYS, R_ANNUAL, Q_ANNUAL, SEED
from .data import _load_price_df, _slice_df, _log_returns
from .garch import fit_garch_jump, _garch_filter, _filter_garch_states
from .pricing import _bs_call, _bs_put, _bs_delta, _bs_vega, _invert_iv
from .simulation import _make_rand_block, _simulate_from_block
from .vol_surface import (
    _build_daily_iv_surface_mc,
    _build_ref_mc_surface,
    _essvi_w,
    _calibrate_essvi_and_extract,
    get_vol_surface_df,
    plot_vol_surface,
)
from .trading import open_position, expiry_payoff, live_delta, STRATEGY_NAME
from .backtest import run_backtest
from .stats import compute_he_stats
from .vol_pricing import (
    test_quoted_surface,
    format_vol_quotes_table,
)
from .analysis import run_analysis, analyze_std_vs_quantile, analyze_std_vs_tenor
