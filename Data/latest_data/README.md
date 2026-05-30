# latest_data/

Place one CSV file per stock here before running the pipeline.

**File naming:** `{TICKER}.csv` (6-digit code, e.g. `600909.csv`)

**Required format:**
```
date,close
2021-01-04,12.34
2021-01-05,12.56
```

- `date`: trading date (YYYY-MM-DD)
- `close`: adjusted closing price (split- and dividend-adjusted)

See `Data/data_yf.py` to download data automatically via yfinance.
