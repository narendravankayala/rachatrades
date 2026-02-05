# Copilot Instructions for rachatrades

## Project Overview

RachaTrades is an autonomous stock market scanner that identifies buy/sell signals using the **Rashemator Multi-Timeframe EMA Cloud Strategy** with MFI and Williams %R confirmation. Publishes results to rachatrades.com. Designed to run fully autonomously via GitHub Actions with zero operational cost.

## Architecture

```
rachatrades/
├── src/
│   ├── data/provider.py       # yfinance wrapper with MTF caching
│   ├── indicators/            # EMA Clouds (3 per timeframe) + MFI + Williams %R
│   ├── strategies/            # Rashemator MTF strategy logic
│   ├── signals/               # Position tracking (SQLite)
│   ├── scanner/               # Stock universe (Top 100 by market cap)
│   ├── web/                   # Static site generator (Jinja2)
│   ├── analyze.py             # Market analysis tool
│   └── main.py                # Scanner entry point
├── data/                      # SQLite database + JSON outputs
└── .github/workflows/         # Scheduled scanning (market hours)
```

## Strategy Rules (Rashemator MTF)

### EMA Clouds
Uses 3 EMA clouds on each timeframe:
- **Trend Cloud**: 50/120 (1-min) or 5/12 (15-min)
- **Midpoint Cloud**: 80/90 (1-min) or 8/9 (15-min)
- **Major S/R Cloud**: 340/500 (1-min) or 34/50 (15-min)

### Zones
- **LONG_ZONE**: Price > 50/120 cloud AND > 340/500 cloud → BUY PULLBACKS
- **SHORT_ZONE**: Price < 50/120 cloud AND < 340/500 cloud → SHORT RALLIES
- **FLAT_ZONE**: Price between clouds → NO TRADE

### BUY Signal (ALL conditions met):
1. In LONG_ZONE (above both clouds)
2. Pullback detected (price touching 50/120 or 80/90 cloud)
3. Oscillator confirms: MFI < 20 **OR** Williams %R < -80

### SELL Signal (ANY condition triggers):
- Zone changes to FLAT or SHORT
- 50/120 cloud turns bearish
- MFI > 80 OR Williams %R > -20 (overbought)

### SHORT Signal (ALL conditions met):
1. In SHORT_ZONE (below both clouds)
2. Oscillator confirms: MFI > 80 **OR** Williams %R > -20

### COVER Signal (ANY condition triggers):
- Zone changes to FLAT or LONG
- 50/120 cloud turns bullish
- MFI < 20 OR Williams %R < -80 (oversold)

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run scanner manually (forces run outside market hours)
python -m src.main --force

# Dry run (no trades executed)
python -m src.main --force --dry-run

# Analyze market (show zones, pullbacks, oscillators)
python -m src.analyze

# Generate website
python -m src.web.generate

# Test individual components
python -m src.indicators.ema_cloud
python -m src.strategies.ema_cloud_strategy
```

## Key Patterns

### EMA Cloud Indicator
The `ema_cloud.py` module provides:
- `calculate_rashemator_clouds_1min(df)` - 50/120, 80/90, 340/500 EMAs
- `calculate_rashemator_clouds_10min(df)` - 5/12, 8/9, 34/50 EMAs (uses 15m data)
- `get_rashemator_signal_mtf(df_10min, df_1min)` - Combined MTF signal

### Zone Detection
```python
from src.indicators import Zone, get_rashemator_signal_mtf

signal = get_rashemator_signal_mtf(df_15m, df_1m)
if signal.zone == Zone.LONG and signal.pullback_1m:
    # Check oscillator confirmation
```

### Multi-Timeframe Data
```python
from src.data import DataProvider

provider = DataProvider()
mtf_data = provider.get_batch_mtf_ohlcv(tickers)  # Returns 15-min + 1-min data
```

### Position Tracking
All trades stored in SQLite (`data/positions.db`). Use `PositionTracker`:
```python
tracker = PositionTracker()
tracker.open_position(ticker, price, timestamp, reason)
tracker.close_position(ticker, price, timestamp, reason)
```

## External Dependencies

- **Data**: yfinance (free, no API key needed)
- **Scheduling**: GitHub Actions cron (free tier: 2000 min/month)
- **Hosting**: GitHub Pages (free)

## Notes

- yfinance doesn't support 10-minute intervals, so we use 15-minute as the "signal" timeframe
- 1-minute data requires 500+ bars for 340/500 EMA, hence `period="7d"`
- Oscillators use OR logic: either MFI < 20 OR Williams %R < -80 confirms entry
