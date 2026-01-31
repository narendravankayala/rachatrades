# Copilot Instructions for rachatrades

## Project Overview

RachaTrades is an autonomous stock market scanner that identifies buy/sell signals using a multi-indicator strategy (EMA Cloud + MFI + Williams %R) and publishes results to rachatrades.com. Designed to run fully autonomously via GitHub Actions with zero operational cost.

## Architecture

```
rachatrades/
├── src/
│   ├── data/provider.py       # yfinance wrapper with caching
│   ├── indicators/            # Technical indicators (EMA, MFI, Williams %R)
│   ├── strategies/            # Trading strategy logic
│   ├── signals/               # Position tracking (SQLite)
│   ├── scanner/               # Stock universe (Top 100 by market cap)
│   ├── web/                   # Static site generator (Jinja2)
│   └── main.py                # Scanner entry point
├── data/                      # SQLite database + JSON outputs
└── .github/workflows/         # Scheduled scanning (every 15 min)
```

## Strategy Rules (EMA Cloud 5/12)

**BUY when ALL conditions met:**
- EMA Cloud bullish (EMA5 > EMA12)
- Price above cloud
- MFI < 20 (oversold)
- Williams %R < -80 (oversold)

**SELL when ANY condition flips:**
- EMA Cloud turns bearish OR MFI > 80 OR Williams %R > -20

**HOLD while no sell signal**

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run scanner manually (forces run outside market hours)
python -m src.main --force

# Dry run (no trades executed)
python -m src.main --force --dry-run

# Generate website
python -m src.web.generate

# Test individual components
python -m src.indicators.ema_cloud
python -m src.strategies.ema_cloud_strategy
```

## Key Patterns

### Adding a New Indicator

Create in `src/indicators/`, follow pattern from `ema_cloud.py`:
1. `calculate_<indicator>(df) -> df` - adds columns to DataFrame
2. `get_<indicator>_signal(df) -> dict` - returns current signal state

### Adding a New Strategy

Inherit from the pattern in `ema_cloud_strategy.py`:
- Use `StrategyConfig` dataclass for parameters
- Return `StrategyResult` with signal + reasoning
- Implement `evaluate()` and `scan_universe()` methods

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

## Future: Alpaca Integration

Architecture supports adding live trading. Add execution mode to position tracker:
```python
tracker = PositionTracker(execution_mode="live")  # or "paper"
```
