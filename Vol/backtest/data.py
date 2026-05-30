"""
data.py
=======
Step 1 — Data loading helpers.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def _load_price_df(csv_path: str) -> pd.DataFrame:
    """
    Load a price CSV and return a clean DataFrame sorted by date.
    Expected columns: date, close (additional columns are ignored).

    Dates are normalized to midnight (date-only) to avoid timezone off-by-one
    errors common with Chinese market data stored in UTC+8.

    Root cause of the bug this fixes:
      pd.to_datetime(..., utc=True) treats "2021-07-01 00:00:00+08:00" as UTC,
      shifting it to "2021-06-30 16:00:00", which then falls BEFORE any
      pd.Timestamp("2021-07-01") filter and causes all prices to shift one day.

    Correct approach: convert tz-aware dates to Beijing time first, then strip tz,
    then normalize to midnight.  Tz-naive dates (no +08:00 suffix) are just
    normalized directly.
    """
    df = pd.read_csv(csv_path)
    parsed = pd.to_datetime(df["date"])
    if parsed.dt.tz is not None:
        # Tz-aware: convert to Beijing time first so that
        # "2021-07-01 00:00+08:00" stays on 2021-07-01, not 2021-06-30.
        df["date"] = (parsed
                      .dt.tz_convert("Asia/Shanghai")
                      .dt.tz_localize(None)
                      .dt.normalize())
    else:
        # Tz-naive (e.g. "2021-07-01 08:00:00"): just strip the time part.
        df["date"] = parsed.dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    df["close"] = df["close"].astype(float)
    return df


def _slice_df(df: pd.DataFrame,
              date_start: str | None = None,
              date_end:   str | None = None) -> pd.DataFrame:
    """Slice a date-indexed DataFrame to [date_start, date_end] (tz-naive)."""
    out = df.copy()
    if date_start:
        out = out[out["date"] >= pd.Timestamp(date_start)]
    if date_end:
        out = out[out["date"] <= pd.Timestamp(date_end)]
    return out.reset_index(drop=True)


def _log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log-returns from a price array."""
    return np.diff(np.log(prices))
