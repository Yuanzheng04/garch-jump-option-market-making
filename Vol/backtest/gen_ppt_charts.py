"""
gen_ppt_charts.py
=================
Generate six presentation charts for the GARCH+Jump option quoting project.
All output PNGs are saved to Results/ (same directory as other stage outputs).
Ticker is read from config.py — do not hardcode it here.

Chart list
----------
chart1_std_vs_moneyness.png  — Delta-sigma(HE) vs strike quantile (cross-stock mean, vega-normalised)
chart2_std_vs_tenor.png      — Delta-sigma(HE) vs tenor (ATM cross-stock mean + OLS fit)
chart3_tstat_heatmap.png     — Cross-stock HE mean t-stat heatmap (with significance stars)
chart4_spearman_heatmap.png  — Spearman rho (sigma_HE vs market-cap rank) heatmap
{ticker}_chart5_vol_surface.png  — eSSVI implied vol surface 3D for config.TICKER
{ticker}_chart6_bid_ask_smile.png — Bid/ask vol smile + price spread for config.TICKER

Runtime behaviour
-----------------
* Chart 3 and Chart 4 are cross-stock aggregates that are expensive to regenerate.
  They are skipped if the output file already exists.
* Chart 5 and Chart 6 are per-ticker and are always regenerated.

Dependencies
------------
pip install matplotlib scipy
"""

import glob
import os
import sys
import warnings

# ── package path setup (same pattern as all other backtest modules) ──────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG)
sys.path.insert(0, _PKG)
sys.path.insert(0, _ROOT)

import backtest.config as _cfg  # read TICKER and other settings from config.py

import matplotlib
matplotlib.use("Agg")           # headless rendering; switch to "TkAgg" for local preview
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.stats import norm as _scipy_norm

warnings.filterwarnings("ignore")

# ── output directory: all charts go to Results/Charts/ ───────────────────────
RESULTS_DIR = os.path.join(_ROOT, "Results", "Charts") + os.sep

# ── colour palette (aligned with PPT theme) ──────────────────────────────────
NAVY   = "#1F4E79"
TBLUE  = "#2E74B5"
LBLUE  = "#BDD7EE"
WHITE  = "#FFFFFF"
DARK   = "#1A1A2E"
GRAY   = "#595959"
GOLD   = "#FFBF00"
RED    = "#C0392B"
GREEN  = "#1E8B4C"
BGCOL  = "#F0F4F9"
LGRAY  = "#E0E4EA"

# ── Chinese font stack (macOS); swap to "SimHei" / "Microsoft YaHei" on Windows ─
plt.rcParams["font.sans-serif"] = [
    "STHeiti", "Heiti TC", "PingFang HK",
    "Arial Unicode MS", "SimHei", "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

# ── shared grid parameters ────────────────────────────────────────────────────
COL_ORDER = ["q05", "q15", "q25", "q35", "q45", "ATM",
             "q55", "q65", "q75", "q85", "q95"]
XLABELS   = ["5%", "15%", "25%", "35%", "45%", "ATM",
             "55%", "65%", "75%", "85%", "95%"]
TENORS    = [10, 21, 42, 63, 125, 252]
DPI       = 160   # output resolution; increase to 200 for sharper images
FIG_W     = 7.5   # single-chart width (inches)


# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════

def _style(ax, title="", xlabel="", ylabel=""):
    """Apply a uniform axis style (background, grid, spines, fonts)."""
    ax.set_facecolor(BGCOL)
    ax.set_title(title, color=NAVY, fontweight="bold", fontsize=11, pad=8)
    ax.set_xlabel(xlabel, color=GRAY, fontsize=9)
    ax.set_ylabel(ylabel, color=GRAY, fontsize=9)
    ax.tick_params(colors=DARK, labelsize=8.5)
    ax.yaxis.grid(True, color=WHITE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)


def _load_all_std() -> dict[str, pd.DataFrame]:
    """Load HE_per_S_std_% from all *_he_stats.csv files.

    Returns {ticker: DataFrame(index=tenors, columns=labels)}.
    """
    files = sorted(glob.glob(RESULTS_DIR + "*_short_call_he_stats.csv"))
    std_all: dict[str, pd.DataFrame] = {}
    for fp in files:
        tkr = os.path.basename(fp).replace("_short_call_he_stats.csv", "")
        df  = pd.read_csv(fp, index_col=0)
        sub = df[df["metric"] == "HE_per_S_std_%"].drop(columns="metric")
        sub.index = sub.index.astype(int)
        sub = sub.loc[[t for t in TENORS if t in sub.index]]
        sub = sub.reindex(columns=COL_ORDER)
        std_all[tkr] = sub.astype(float)
    return std_all


def _cross_mean_std(std_all: dict, T: int):
    """Cross-sectional mean and std of HE sigma at tenor T across all tickers."""
    vals = [df.loc[T, COL_ORDER].values for df in std_all.values()
            if T in df.index]
    if not vals:
        return None, None
    arr = np.array(vals, dtype=float)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _load_all_vega_std(clip_vol_pts: float = 10.0,
                       alpha: float = 0.10) -> dict:
    """
    Load sigma(HE/S) for all tickers and convert to vega-normalised vol points.

    Conversion formula (doc §2.2):
        Delta_sigma(T, K) = sigma(HE/S) / vega_eff(T, K)
        vega_eff = sqrt[ (phi(d1) * sqrt(T_years))^2  +  eps(T)^2 ]
        eps(T)   = alpha * 0.4 * sqrt(T_years)   (regularisation floor)
        d1       = (-k + sigma_mid^2*T/2) / (sigma_mid*sqrt(T))

    Sources:
        sigma(HE/S) from *_he_stats.csv  HE_per_S_std_%  column (in %)
        sigma_mid   from *_he_stats.csv  model_vol%      column (% -> /100)
        k           from *_iv_surface.csv  k row (log K/S)

    Returns vol points (%), clipped to clip_vol_pts.
    """
    files_he = sorted(glob.glob(RESULTS_DIR + "*_short_call_he_stats.csv"))
    result: dict = {}

    for fp in files_he:
        tkr = os.path.basename(fp).replace("_short_call_he_stats.csv", "")
        fp_iv = RESULTS_DIR + f"{tkr}_short_call_iv_surface.csv"
        if not os.path.exists(fp_iv):
            continue

        df_he = pd.read_csv(fp, index_col=0)
        df_iv = pd.read_csv(fp_iv, index_col=0)

        # sigma(HE/S) in %, model vol in % (needs /100 for BS formula)
        std_raw = df_he[df_he["metric"] == "HE_per_S_std_%"].drop(columns="metric")
        vol_raw = df_he[df_he["metric"] == "model_vol%"].drop(columns="metric")
        k_raw   = df_iv[df_iv["metric"] == "k"].drop(columns="metric")

        for df_ in (std_raw, vol_raw, k_raw):
            df_.index = df_.index.astype(int)

        tenors = [t for t in TENORS if t in std_raw.index]
        norm_rows: dict = {}
        for T in tenors:
            T_years = T / 252.0
            sqrt_T  = np.sqrt(T_years)
            eps     = alpha * 0.4 * sqrt_T
            row: dict = {}
            for q in COL_ORDER:
                try:
                    sigma_he_pct = float(std_raw.loc[T, q])        # e.g. 0.55 (%)
                    sigma_mid    = float(vol_raw.loc[T, q]) / 100.0 # decimal
                    k_val        = float(k_raw.loc[T, q]) if q in k_raw.columns else 0.0
                except (KeyError, ValueError):
                    row[q] = np.nan
                    continue

                if not (np.isfinite(sigma_mid) and sigma_mid > 0):
                    row[q] = np.nan
                    continue

                vt = sigma_mid * sqrt_T
                d1 = (-k_val + 0.5 * sigma_mid ** 2 * T_years) / vt
                vega_per_S = sqrt_T * float(_scipy_norm.pdf(d1))
                vega_eff   = np.sqrt(vega_per_S ** 2 + eps ** 2)

                # Delta_sigma in vol points (%): (sigma_HE% / 100) / vega_eff * 100
                delta_vol = (sigma_he_pct / 100.0) / vega_eff * 100.0
                row[q]    = float(np.clip(delta_vol, 0.0, clip_vol_pts))
            norm_rows[T] = row

        if norm_rows:
            df_norm = pd.DataFrame(norm_rows).T.reindex(columns=COL_ORDER)
            df_norm.index.name = "tenor_days"
            result[tkr] = df_norm.astype(float)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Chart 1 — Delta-sigma(HE) vs strike quantile (cross-stock mean)
# ═════════════════════════════════════════════════════════════════════════════

def plot_chart1(vega_std_all: dict, out_path: str) -> None:
    """
    Line chart: x = strike quantile, y = Delta-sigma(HE) vol points (cross-stock mean).
    Y-axis in vol points so traders can directly judge hedge-error magnitude.
    Delta-sigma = sigma(HE/S) / vega_eff(T,K); deep-OTM clipped to 10 vol pts.
    One line per tenor; shading = +/- 0.5 * cross-stock std.
    """
    fig, ax = plt.subplots(figsize=(FIG_W, 4.0))
    fig.patch.set_facecolor(WHITE)

    tenor_colors = ["#9DC3E6", "#5EA3D0", "#2E74B5", "#1F4E79", "#1A3A5C", DARK]
    x = np.arange(len(COL_ORDER))

    for i, T in enumerate(TENORS):
        mean_v, std_v = _cross_mean_std(vega_std_all, T)
        if mean_v is None:
            continue
        ax.plot(x, mean_v, color=tenor_colors[i], linewidth=1.8,
                marker="o", markersize=3.5, label=f"{T}d", zorder=4)
        ax.fill_between(x, mean_v - std_v * 0.5, mean_v + std_v * 0.5,
                        color=tenor_colors[i], alpha=0.12, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS, fontsize=8.5)
    ax.set_ylim(bottom=0)

    _style(ax,
           title=r"$\Delta\sigma$(HE) vs 行权价（20只股票均值，vol points）",
           xlabel="行权价分位数",
           ylabel=r"$\Delta\sigma$ (vol points, %)")
    ax.legend(title="期限", fontsize=8, title_fontsize=8,
              ncol=3, loc="upper left", framealpha=0.9, edgecolor=LBLUE)

    fig.text(0.02, -0.02,
             r"注：$\Delta\sigma = \sigma(HE/S) \,/\, vega_{eff}(T,K)$，单位 vol points；"
             "深度OTM（vega≈0）处已截断至10 vol pts；阴影 = 跨股票截面标准差×0.5",
             fontsize=7.5, color=GRAY, style="italic")

    plt.tight_layout(pad=1.0)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Chart 1 saved → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Chart 2 — Delta-sigma(HE) vs tenor (ATM cross-stock mean + OLS fit)
# ═════════════════════════════════════════════════════════════════════════════

def plot_chart2(vega_std_all: dict, out_path: str) -> None:
    """
    Line chart: x = tenor (trading days), y = ATM Delta-sigma(HE) vol points.
    OLS fit: Delta-sigma^2(T) = C1 + C2*T (verifies monotone term structure).
    """
    fig, ax = plt.subplots(figsize=(FIG_W, 4.0))
    fig.patch.set_facecolor(WHITE)
    T_arr = np.array(TENORS, dtype=float)

    atm_means, atm_stds = [], []
    for T in TENORS:
        vals = [float(df.loc[T, "ATM"]) for df in vega_std_all.values()
                if T in df.index and not np.isnan(df.loc[T, "ATM"])]
        atm_means.append(np.nanmean(vals) if vals else np.nan)
        atm_stds.append(np.nanstd(vals)  if vals else np.nan)
    atm_means = np.array(atm_means)
    atm_stds  = np.array(atm_stds)

    ax.fill_between(T_arr, atm_means - atm_stds, atm_means + atm_stds,
                    color=GOLD, alpha=0.20, zorder=3,
                    label=r"ATM $\pm$1 std（20股截面分散带）")
    ax.plot(T_arr, atm_means, color=GOLD, lw=2.5, marker="o", ms=5,
            label="ATM 均值", zorder=5)

    # OLS fit: Delta_sigma^2(T) = C1 + C2*T
    ok = np.isfinite(atm_means)
    if ok.sum() >= 2:
        X   = np.column_stack([np.ones(ok.sum()), T_arr[ok]])
        b, _, _, _ = np.linalg.lstsq(X, (atm_means[ok] / 100) ** 2, rcond=None)
        T_fit    = np.linspace(T_arr[0], T_arr[-1], 200)
        fit_line = np.sqrt(np.maximum(b[0] + b[1] * T_fit, 0)) * 100
        y_actual = (atm_means[ok] / 100) ** 2
        ss_res   = np.sum((y_actual - (b[0] + b[1] * T_arr[ok])) ** 2)
        ss_tot   = np.sum((y_actual - y_actual.mean()) ** 2)
        R2       = max(1 - ss_res / ss_tot if ss_tot > 0 else 0, 0)
        ols_label = r"OLS: $\Delta\sigma^2 \approx C_1+C_2 T$" + f"  ($R^2$={R2:.2f})"
        ax.plot(T_fit, fit_line, color=RED, lw=1.8, ls="--",
                label=ols_label, zorder=6)

    ax.set_xticks(TENORS)
    ax.set_xticklabels([str(t) for t in TENORS])
    ax.set_ylim(bottom=0)
    _style(ax,
           title=r"ATM $\Delta\sigma$(HE) 随期限增加（20股截面均值，vol points）",
           xlabel="期限（交易日）",
           ylabel=r"$\Delta\sigma$ (vol points, %)")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95, edgecolor=LBLUE)

    fig.text(0.02, -0.02,
             r"注：ATM $\Delta\sigma$ = ATM $\sigma(HE/S)$ / $vega_{eff}$(T, ATM)；"
             "阴影带 = 20只股票间的截面标准差 ±1σ，落在带内说明与市场均值接近",
             fontsize=7.5, color=GRAY, style="italic")

    plt.tight_layout(pad=1.0)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Chart 2 saved → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Chart 3 — Cross-stock HE mean t-stat heatmap (values + significance stars)
# ═════════════════════════════════════════════════════════════════════════════

def plot_chart3(out_path: str) -> None:
    """
    Continuous-value heatmap sourced from cross_he_tstat_heatmap.csv.
    Colour encodes the t-statistic (green = positive/significant, red = negative).
    Cell annotation: HE mean (% of S) + significance stars from t-stat:
        ***  |t| > 2.576  (99% CI)
        **   |t| > 1.96   (95% CI)
        *    |t| > 1.645  (90% CI)
    """
    df_ts   = pd.read_csv(RESULTS_DIR + "cross_he_tstat_heatmap.csv",  index_col=0)
    df_mean = pd.read_csv(RESULTS_DIR + "cross_he_mean_heatmap.csv",   index_col=0)

    col_map = {"0.05": "q05", "0.15": "q15", "0.25": "q25", "0.35": "q35",
               "0.45": "q45", "0.55": "q55", "0.65": "q65", "0.75": "q75",
               "0.85": "q85", "0.95": "q95"}
    # drop ATM column (-1.0) — quantile-only display
    df_ts   = df_ts.drop(columns=[c for c in df_ts.columns   if str(c) == "-1.0"], errors="ignore")
    df_mean = df_mean.drop(columns=[c for c in df_mean.columns if str(c) == "-1.0"], errors="ignore")
    df_ts.columns   = [col_map.get(str(c), str(c)) for c in df_ts.columns]
    df_mean.columns = [col_map.get(str(c), str(c)) for c in df_mean.columns]
    display_order = ["q05", "q15", "q25", "q35", "q45",
                     "q55", "q65", "q75", "q85", "q95"]
    df_ts      = df_ts.reindex(columns=display_order)
    df_mean    = df_mean.reindex(columns=display_order)
    x_labels   = ["5%", "15%", "25%", "35%", "45%",
                  "55%", "65%", "75%", "85%", "95%"]
    row_labels = [f"T={int(t)}d" for t in df_ts.index]
    data_t     = df_ts.values.astype(float)   # t-stats  → colourmap + stars
    data_m     = df_mean.values.astype(float) # HE means → cell text
    data       = data_t                       # alias for vmin/vmax scaling

    # colourmap: red (negative) → white (zero) → green (positive)
    cmap3 = LinearSegmentedColormap.from_list(
        "tstat", [RED, "#F5F5F5", GREEN], N=256
    )
    abs_max = max(np.nanmax(np.abs(data)), 3.0)

    fig, ax = plt.subplots(figsize=(FIG_W + 0.5, 3.8))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    im = ax.imshow(data, cmap=cmap3, aspect="auto",
                   vmin=-abs_max, vmax=abs_max, interpolation="nearest")

    # cell annotation: HE mean value + significance stars from t-stat
    for i in range(data_t.shape[0]):
        for j in range(data_t.shape[1]):
            t = data_t[i, j]
            m = data_m[i, j]
            if not np.isfinite(t):
                continue
            if   abs(t) > 2.576: stars = "***"
            elif abs(t) > 1.96:  stars = "**"
            elif abs(t) > 1.645: stars = "*"
            else:                 stars = ""
            cell_txt = f"{m:.2f}{stars}"
            ax.text(j, i, cell_txt, ha="center", va="center",
                    fontsize=7.5, color=DARK,
                    fontweight="bold" if stars else "normal")

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("跨股票 HE 均值（卖方视角）",
                 color=NAVY, fontweight="bold", fontsize=11, pad=8)
    ax.set_xlabel("行权价分位数", color=GRAY, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=20, pad=0.02)
    cbar.set_label("t 统计量", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    # significance threshold reference lines on the colorbar
    for thresh, label_ in [(1.645, "90%"), (1.96, "95%"), (2.576, "99%")]:
        cbar.ax.axhline(thresh,  color=DARK, lw=0.8, ls="--", alpha=0.6)
        cbar.ax.axhline(-thresh, color=DARK, lw=0.8, ls="--", alpha=0.6)

    fig.text(
        0.02, -0.03,
        "注：格中数值 = 跨股票 HE 均值 (% of S)；颜色/星号基于 t 统计量；"
        "*** p<0.01，** p<0.05，* p<0.10；"
        "绿色 = 卖方获利显著，红色 = 卖方亏损显著",
        fontsize=7, color=GRAY, style="italic",
    )

    plt.tight_layout(pad=1.0)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Chart 3 saved → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Chart 4 — Spearman rho (sigma_HE vs market-cap rank) heatmap
# ═════════════════════════════════════════════════════════════════════════════

def plot_chart4(out_path: str) -> None:
    """
    Heatmap: rows = tenor, cols = strike; colour = Spearman rho.
    Deep blue (rho < 0) means large-cap stocks have lower hedging costs.
    Source: cross_spearman_rho_heatmap.csv (cell format "-0.72*").
    """
    df_rho = pd.read_csv(RESULTS_DIR + "cross_spearman_rho_heatmap.csv", index_col=0)

    rho_num = pd.DataFrame(index=df_rho.index, columns=df_rho.columns, dtype=float)
    sig_map = pd.DataFrame(False, index=df_rho.index, columns=df_rho.columns)
    for ridx in df_rho.index:
        for cidx in df_rho.columns:
            cell = str(df_rho.loc[ridx, cidx])
            sig  = "*" in cell
            num  = float(cell.replace("*", "").replace("\u2212", "-").strip())
            rho_num.loc[ridx, cidx] = num
            sig_map.loc[ridx, cidx] = sig

    # drop ATM column (-1.0) — quantile-only display
    df_rho   = df_rho.drop(columns=[c for c in df_rho.columns if str(c) == "-1.0"], errors="ignore")
    rho_num  = rho_num.drop(columns=[c for c in rho_num.columns if str(c) == "-1.0"], errors="ignore")
    sig_map  = sig_map.drop(columns=[c for c in sig_map.columns if str(c) == "-1.0"], errors="ignore")
    col_labels = [f"q{int(float(c)*100):02d}" for c in df_rho.columns]
    row_labels = [f"{int(t)}d" for t in df_rho.index]
    data4 = rho_num.values.astype(float)

    cmap4 = LinearSegmentedColormap.from_list(
        "ppt", [RED, "#F0F0F0", NAVY], N=256
    )

    fig, ax = plt.subplots(figsize=(FIG_W, 3.6))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    im = ax.imshow(data4, cmap=cmap4, aspect="auto", vmin=-1, vmax=1)

    for i in range(data4.shape[0]):
        for j in range(data4.shape[1]):
            v   = data4[i, j]
            txt = f"{v:+.2f}" + ("*" if sig_map.iloc[i, j] else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color=DARK,
                    fontweight="bold" if sig_map.iloc[i, j] else "normal")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=45)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(
        "Spearman ρ (σ_HE, 市值排名)\n深蓝=ρ<0 大盘股对冲成本更低",
        color=NAVY, fontweight="bold", fontsize=10, pad=6,
    )
    ax.set_xlabel("行权价分位数", color=GRAY, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("ρ", rotation=0, labelpad=8, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.text(0.98, 0.01, "* p < 0.10", ha="right", fontsize=7.5,
             color=GRAY, style="italic")

    plt.tight_layout(pad=1.0)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Chart 4 saved → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Chart 5 — eSSVI implied vol surface 3D
# ═════════════════════════════════════════════════════════════════════════════

def plot_chart5(ticker: str, out_path: str,
                ref_date: str = "2026-03-09",
                ref_tenor: int = 63,
                elev: float = 22, azim: float = 225,
                n_q: int = 40, n_T: int = 60,
                ks_min: float = 0.81, ks_max: float = 1.15) -> None:
    """
    Bloomberg-style three-panel figure:
      - Main (left): 3D eSSVI vol surface
          x = K/S (left→right = ITM→OTM, OTM nearest to viewer)
          y = Tenor (front→back = short→long)
          z = IV %
      - Top-right: Vol vs Tenor (ATM term structure)
      - Bottom-right: Vol vs K/S (smile, ref_tenor)

    Wing spikes at short tenors are a real-data feature (not interpolation artefact).
    Mitigation:
      1. Lower grid density (40x60) to reduce matplotlib rendering noise
      2. Add lightweight wireframe for smooth appearance
      3. rcount/ccount limits to avoid jagged edges

    Parameters
    ----------
    ref_tenor : tenor (trading days) used for the bottom-right smile panel
    elev/azim : 3D viewing angle (azim=225 → left-front, OTM near viewer)
    n_q / n_T : interpolation grid density (lower = less rendering noise)
    """
    from scipy.interpolate import PchipInterpolator
    from matplotlib.gridspec import GridSpec

    df_iv     = pd.read_csv(RESULTS_DIR + f"{ticker}_short_call_iv_surface.csv",
                             index_col=0)
    tenors_iv = sorted(df_iv.index.unique())
    Q_LBLS_IV = ["q05","q15","q25","q35","q45","ATM","q55","q65","q75","q85","q95"]

    IV_data = np.zeros((len(tenors_iv), 11))
    K_data  = np.zeros((len(tenors_iv), 11))
    for i, T in enumerate(tenors_iv):
        sub = df_iv.loc[T]
        IV_data[i] = sub[sub["metric"]=="IV%"].drop(columns="metric")[Q_LBLS_IV].values.astype(float)
        K_data[i]  = sub[sub["metric"]=="k"  ].drop(columns="metric")[Q_LBLS_IV].values.astype(float)

    T_coarse = np.array(tenors_iv, dtype=float)
    q_coarse = np.arange(11, dtype=float)

    # interpolate along tenor axis, then along quantile axis
    T_fine = np.linspace(T_coarse[0], T_coarse[-1], n_T)
    q_fine = np.linspace(0.0, 10.0, n_q)

    IV_tenor = np.zeros((n_T, 11))
    for j in range(11):
        IV_tenor[:, j] = PchipInterpolator(T_coarse, IV_data[:, j])(T_fine)

    IV_full = np.zeros((n_T, n_q))
    for i in range(n_T):
        IV_full[i] = PchipInterpolator(q_coarse, IV_tenor[i])(q_fine)

    # light Gaussian smoothing to suppress Pchip short-end oscillations
    from scipy.ndimage import gaussian_filter
    IV_full = gaussian_filter(IV_full, sigma=(0.8, 0.4))

    # K/S reference at ref_tenor — used for y-axis tick labels and smile panel
    ref_idx  = list(tenors_iv).index(ref_tenor) if ref_tenor in tenors_iv else len(tenors_iv)//2
    ks_ref   = np.exp(K_data[ref_idx])   # shape (11,)

    # clip K/S range to [ks_min, ks_max] to remove extreme wing spikes
    k_ref_fine  = PchipInterpolator(q_coarse, K_data[ref_idx])(q_fine)
    ks_ref_fine = np.exp(k_ref_fine)
    q_mask      = (ks_ref_fine >= ks_min) & (ks_ref_fine <= ks_max)
    if q_mask.sum() < 6:  # fallback: keep all points if clip removes too many
        q_mask = np.ones(n_q, dtype=bool)

    q_fine_sub   = q_fine[q_mask]
    IV_full_sub  = IV_full[:, q_mask]   # shape (n_T, n_sub)
    ks_fine_sub  = ks_ref_fine[q_mask]  # K/S values at fine sub-grid for ref tenor

    n_q_sub = q_mask.sum()

    # axis orientation: OTM (high quantile) at front (small Y), ITM at back (large Y)
    q_rev    = q_fine_sub[::-1]   # reversed sub-grid
    IV_rev   = IV_full_sub[:, ::-1]
    ks_rev   = ks_fine_sub[::-1]

    # meshgrid: X = tenor (left→right short→long), Y = q_rev (front→back OTM→ITM)
    TT_m, QQ_m = np.meshgrid(T_fine, q_rev)   # shape (n_q_sub, n_T)
    IV_surf     = IV_rev.T                     # shape (n_q_sub, n_T)

    # figure layout: main 3D + top-right term structure + bottom-right smile
    cmap5 = LinearSegmentedColormap.from_list("vol", [LBLUE, TBLUE, NAVY, DARK], N=256)
    fig   = plt.figure(figsize=(13.5, 6.0))
    fig.patch.set_facecolor(WHITE)
    gs = GridSpec(2, 3, figure=fig,
                  width_ratios=[1.6, 0.02, 0.9],   # 3D | gap | side panels
                  hspace=0.50, wspace=0.12)

    ax3d   = fig.add_subplot(gs[:, 0], projection="3d")
    ax_trm = fig.add_subplot(gs[0, 2])   # Vol vs Tenor (ATM)
    ax_sml = fig.add_subplot(gs[1, 2])   # Vol vs K/S   (smile)

    # ── main 3D surface ───────────────────────────────────────────────────────
    ax3d.set_facecolor(BGCOL)
    surf = ax3d.plot_surface(
        TT_m, QQ_m, IV_surf,
        cmap=cmap5, edgecolor="none", alpha=0.90, antialiased=True,
        rcount=n_T, ccount=n_q_sub,
    )
    # lightweight wireframe for depth perception
    ax3d.plot_wireframe(
        TT_m, QQ_m, IV_surf,
        rstride=max(1, n_T//8), cstride=max(1, n_q_sub//6),
        color=WHITE, linewidth=0.25, alpha=0.35,
    )
    # contour projection at the base
    iv_floor = IV_full.min() - 0.8
    ax3d.contour(TT_m, QQ_m, IV_surf, zdir="z", offset=iv_floor,
                 levels=6, cmap=cmap5, alpha=0.25, linewidths=0.6)

    # x-axis: tenor
    ax3d.set_xticks([10, 42, 63, 125, 252])
    ax3d.set_xticklabels(["10d", "42d", "63d", "125d", "252d"], fontsize=7)

    # y-axis: K/S (ks_rev already covers the fine sub-grid from OTM to ITM)
    y_lo, y_hi = float(q_rev[0]), float(q_rev[-1])
    ytick_pos  = np.linspace(y_lo, y_hi, min(5, n_q_sub))
    ytick_ks   = []
    for yp in ytick_pos:
        idx_near = int(np.argmin(np.abs(q_rev - yp)))
        ytick_ks.append(f"{ks_rev[idx_near]:.2f}")
    ax3d.set_yticks(ytick_pos)
    ax3d.set_yticklabels(ytick_ks, fontsize=7)
    ax3d.tick_params(colors=DARK, labelsize=7)

    ax3d.set_xlabel("Tenor (days)", labelpad=10, fontsize=8.5, color=DARK)
    ax3d.set_ylabel("K/S",         labelpad=10, fontsize=8.5, color=DARK)
    ax3d.set_zlabel("IV (%)",       labelpad=6,  fontsize=8.5, color=DARK)

    ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor(LGRAY)
    ax3d.yaxis.pane.set_edgecolor(LGRAY)
    ax3d.zaxis.pane.set_edgecolor(LGRAY)
    ax3d.grid(True, color=LGRAY, linewidth=0.35)
    ax3d.view_init(elev=elev, azim=azim)
    # invert Y so that high K/S (OTM) is nearest the viewer
    ax3d.invert_yaxis()

    cbar = fig.colorbar(surf, ax=ax3d, shrink=0.38, aspect=14, pad=0.04, location="right")
    cbar.set_label("IV %", fontsize=8, labelpad=5)
    cbar.ax.tick_params(labelsize=7)

    ax3d.set_title(f"eSSVI 隐含波动率曲面  |  {ticker}  |  截止 {ref_date}",
                   color=NAVY, fontweight="bold", fontsize=10, pad=8)

    # ── top-right: Vol vs Tenor (ATM = quantile index 5) ─────────────────────
    ax_trm.set_facecolor(BGCOL)
    atm_iv_fine  = IV_tenor[:, 5]   # ATM IV at T_fine grid points
    ax_trm.plot(T_fine, atm_iv_fine, color=NAVY, lw=2.0, zorder=4)
    ax_trm.scatter(T_coarse, IV_data[:, 5], color=GOLD, s=40, zorder=5,
                   edgecolors=DARK, linewidths=0.5)
    ax_trm.set_xticks([10, 42, 63, 125, 252])
    ax_trm.set_xticklabels(["10", "42", "63", "125", "252"], fontsize=7.5)
    ax_trm.set_title("Vol vs Tenor  (ATM)", color=NAVY, fontweight="bold", fontsize=9)
    ax_trm.set_xlabel("Tenor (days)", color=GRAY, fontsize=8)
    ax_trm.set_ylabel("IV %",         color=GRAY, fontsize=8)
    ax_trm.yaxis.grid(True, color=WHITE, lw=0.7)
    ax_trm.tick_params(labelsize=7.5)
    for sp in ax_trm.spines.values(): sp.set_edgecolor(LGRAY)

    # ── bottom-right: Vol vs K/S smile at each tenor ─────────────────────────
    ax_sml.set_facecolor(BGCOL)
    smile_cols = ["#9DC3E6","#5EA3D0","#2E74B5","#1F4E79","#1A3A5C",DARK]
    for ci, T in enumerate(tenors_iv):
        t_fidx = int(np.argmin(np.abs(T_fine - T)))
        iv_row = IV_full_sub[t_fidx]   # IV along the fine K/S sub-grid for this tenor
        if T == ref_tenor:
            ax_sml.plot(ks_fine_sub, iv_row, color=GOLD,
                        lw=2.5, zorder=5, label=f"{ref_tenor}d (ref)")
        else:
            ax_sml.plot(ks_fine_sub, iv_row,
                        color=smile_cols[ci % len(smile_cols)],
                        lw=1.5, label=f"{int(T)}d", zorder=3)
    ks_sml = ks_fine_sub
    tick_candidates = ks_sml[[0, len(ks_sml)//2, -1]] if len(ks_sml) >= 3 else ks_sml
    ax_sml.set_xticks(tick_candidates)
    ax_sml.set_xticklabels([f"{v:.2f}" for v in tick_candidates], fontsize=7, rotation=30)
    ax_sml.set_title("Vol vs K/S  (Smile)", color=NAVY, fontweight="bold", fontsize=9)
    ax_sml.set_xlabel("K/S", color=GRAY, fontsize=8)
    ax_sml.set_ylabel("IV %", color=GRAY, fontsize=8)
    ax_sml.legend(fontsize=6.5, ncol=2, loc="upper center",
                  framealpha=0.9, edgecolor=LBLUE)
    ax_sml.yaxis.grid(True, color=WHITE, lw=0.7)
    ax_sml.tick_params(labelsize=7.5)
    for sp in ax_sml.spines.values(): sp.set_edgecolor(LGRAY)

    plt.tight_layout(pad=0.8)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Chart 5 saved → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Chart 6 — Bid-Ask Vol Smile + Price Spread (specified ticker and tenor)
# ═════════════════════════════════════════════════════════════════════════════

def _load_chart6_data(ticker: str, tenor: int = 63):
    """
    Read vol smile bid-ask data from Stage 4 price_surface.csv.

    Returns
    -------
    ks           : K/S array
    iv_model     : Stage-2 original fair IV (%)       — shown as reference line
    iv_model_adj : S4-10b bias-corrected mid IV (%)   — shown as actual mid line
    iv_ask       : Stage-4 final ask IV (%)
    iv_bid       : Stage-4 final bid IV (%)

    Falls back to iv_model if iv_model_adj is absent (older output files).
    """
    fp_ps = RESULTS_DIR + f"{ticker}_short_call_price_surface.csv"
    fp_iv = RESULTS_DIR + f"{ticker}_short_call_iv_surface.csv"
    _Q = ["q05","q15","q25","q35","q45","ATM","q55","q65","q75","q85","q95"]

    df_ps = pd.read_csv(fp_ps, index_col=[0, 1])
    df_iv = pd.read_csv(fp_iv, index_col=0)

    # K/S from log-moneyness stored in the iv_surface file
    sub_iv = df_iv.loc[[tenor]]
    k_vals = sub_iv[sub_iv["metric"] == "k"].drop(columns="metric")[_Q].values.flatten().astype(float)
    ks     = np.exp(k_vals)

    def _row(metric):
        return df_ps.loc[(tenor, metric), _Q].values.flatten().astype(float)

    iv_model = _row("iv_model")
    try:
        iv_model_adj = _row("iv_model_adj")   # bias-corrected mid from S4-10b
    except KeyError:
        iv_model_adj = iv_model               # backward-compatible fallback

    return ks, iv_model, iv_model_adj, _row("iv_ask"), _row("iv_bid")


def plot_chart6(ks, iv_model, iv_model_adj, iv_ask, iv_bid,
                ticker="", tenor=63, out_path=None):
    """
    Two-column dark-theme layout showing Stage 4 quoting results.

    Left panel — Vol Smile:
        ask / bid (dashed)    — Stage-4 final quotes
        mid (solid, bright blue) — S4-10b bias-corrected iv_model_adj
        model ref (dotted, grey) — Stage-2 original fair IV (for comparison)
        fill_between(bid, ask)   — bid-ask band

    Right panel — Price Spread = (p_ask - p_bid) / S (%),  computed via BS.

    Data source: Stage-4 price_surface.csv
        iv_model_adj → mid (bias-corrected)
        iv_model     → reference line (Stage-2, not used as quote mid)
        iv_ask / iv_bid → final quotes
    """
    from scipy.stats import norm as _norm

    n       = len(ks)
    T_years = tenor / 252.0

    def bs_call_pct(iv_pct, k_log):
        """Black-Scholes call price normalised by S (S=1, r=0)."""
        iv = iv_pct / 100.0
        if iv <= 0:
            return max(np.exp(k_log) - 1.0, 0.0)
        sqrtT = np.sqrt(T_years)
        d1 = (-k_log + 0.5 * iv**2 * T_years) / (iv * sqrtT)
        d2 = d1 - iv * sqrtT
        return _norm.cdf(d1) - np.exp(k_log) * _norm.cdf(d2)

    k_log = np.log(ks)
    p_adj = np.array([bs_call_pct(iv_model_adj[i], k_log[i]) for i in range(n)])
    p_ask = np.array([bs_call_pct(iv_ask[i],        k_log[i]) for i in range(n)])
    p_bid = np.array([bs_call_pct(iv_bid[i],        k_log[i]) for i in range(n)])

    price_spread_pct = (p_ask - p_bid) * 100.0

    # colour palette for dark-theme chart
    BG        = "#1a1a2e"
    FG        = "white"
    GRID      = (1, 1, 1, 0.08)
    COL_ASK   = "#FF6B6B"
    COL_MID   = "#74B9FF"
    COL_REF   = "#888888"   # Stage-2 original model IV reference line
    COL_BID   = "#55EFC4"
    COL_SPRD  = "#A29BFE"

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    for ax in (ax_l, ax_r):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=9)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)
        for sp in ax.spines.values():
            sp.set_edgecolor((1, 1, 1, 0.2))
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.xaxis.grid(True, color=GRID, lw=0.6)

    # left panel: Vol Smile
    ax_l.fill_between(ks, iv_bid, iv_ask, color=COL_MID, alpha=0.13, zorder=2)
    ax_l.plot(ks, iv_ask,       color=COL_ASK, lw=2.0, ls="--",
              label="Ask IV",                     zorder=5)
    ax_l.plot(ks, iv_model_adj, color=COL_MID, lw=2.5,
              label="Mid IV (HE-bias corrected)", zorder=6)
    ax_l.plot(ks, iv_bid,       color=COL_BID, lw=2.0, ls="--",
              label="Bid IV",                     zorder=5)
    ax_l.plot(ks, iv_model,     color=COL_REF, lw=1.2, ls=":",
              label="Model IV (Stage-2 ref)",     zorder=4, alpha=0.75)
    ax_l.set_xlabel("K/S")
    ax_l.set_ylabel("IV %")
    ax_l.set_title(f"{ticker}  |  Vol Smile Bid-Ask  (T={tenor}d)",
                   fontweight="bold", fontsize=10, pad=8, color=FG)
    ax_l.legend(fontsize=8, facecolor=BG, edgecolor=(1,1,1,0.2),
                labelcolor=FG, loc="upper center", ncol=2)

    # right panel: Price Spread (ask - bid) / S (%)
    ax_r.fill_between(ks, 0, price_spread_pct, color=COL_SPRD, alpha=0.20, zorder=2)
    ax_r.plot(ks, price_spread_pct, color=COL_SPRD, marker="o", ms=5,
              lw=1.8, label="(p_ask − p_bid) / S (%)", zorder=4)
    ax_r.set_xlabel("K/S")
    ax_r.set_ylabel("Δp/S (%)")
    ax_r.set_title("Price Spread  (ask − bid) / S", fontweight="bold",
                   fontsize=10, pad=8, color=FG)
    ax_r.set_ylim(bottom=0)
    ax_r.legend(fontsize=8, facecolor=BG, edgecolor=(1,1,1,0.2),
                labelcolor=FG, loc="upper center")

    fig.text(0.5, -0.02,
             f"Mid = HE-bias corrected (S4-10b)  |  "
             f"Stage-4 price_surface.csv  |  T={tenor}d = {T_years:.3f}yr",
             ha="center", fontsize=8, color=(1, 1, 1, 0.65), style="italic")

    plt.tight_layout(pad=1.2)
    if out_path:
        plt.savefig(out_path, dpi=DPI, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        print(f"Chart 6 saved → {out_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    tickers = list(_cfg.TICKERS)
    out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating charts for tickers: {tickers}")

    # Chart 1 / 2 are disabled by default (uncomment to enable)
    # vega_std_all = _load_all_vega_std()
    # print(f"Loaded {len(vega_std_all)} tickers for vega-normalised std.")
    # plot_chart1(vega_std_all, out_dir + "chart1_std_vs_moneyness.png")
    # plot_chart2(vega_std_all, out_dir + "chart2_std_vs_tenor.png")

    # Chart 3 / 4: cross-stock aggregates — generated once, skip if file exists
    c3_path = out_dir + "chart3_tstat_heatmap.png"
    if not os.path.exists(c3_path):
        print("Generating Chart 3 (cross-stock HE mean t-stat heatmap) ...")
        plot_chart3(c3_path)
    else:
        print(f"Chart 3 already exists, skipping: {c3_path}")

    c4_path = out_dir + "chart4_spearman_heatmap.png"
    if not os.path.exists(c4_path):
        print("Generating Chart 4 (Spearman rho heatmap) ...")
        plot_chart4(c4_path)
    else:
        print(f"Chart 4 already exists, skipping: {c4_path}")

    # Chart 5 / 6: per-ticker — loop over all tickers in config.TICKERS
    for ticker in tickers:
        _cfg.TICKER = ticker  # keep config in sync (used by helper functions)
        print(f"\n── {ticker} ──────────────────────────────")

        c5_path = out_dir + f"{ticker}_chart5_vol_surface.png"
        print(f"Generating Chart 5 (3D vol surface, {ticker}) ...")
        plot_chart5(ticker, c5_path)

        print(f"Generating Chart 6 (Vol Smile Bid-Ask, {ticker}) ...")
        ks6, ivm6, ivm6_adj, iva6, ivb6 = _load_chart6_data(ticker, tenor=63)
        c6_path = out_dir + f"{ticker}_chart6_bid_ask_smile.png"
        plot_chart6(ks6, ivm6, ivm6_adj, iva6, ivb6,
                    ticker=ticker, tenor=63, out_path=c6_path)

    print(f"\nAll charts written to: {out_dir}")


if __name__ == "__main__":
    main()
