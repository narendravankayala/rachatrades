# rachatrades

Autonomous stock market scanner using EMA Cloud + MFI + Williams %R strategy.

## Features

- 📊 Scans top 100 US stocks by market cap every 10 minutes
- 🎯 Multi-indicator strategy: EMA Cloud (5/12), MFI, Williams %R
- 📈 Automatic position tracking with P&L calculation
- 🌐 Live dashboard at [rachatrades.com](https://rachatrades.com)
- ⚡ Fully autonomous via GitHub Actions (zero cost)

## Strategy

**BUY** when:
- EMA 5 crosses above EMA 12 (bullish cloud)
- Price is above the cloud
- MFI is oversold (< 20)
- Williams %R is oversold (< -80)

**SELL** when any condition reverses.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run scanner (with --force for outside market hours)
python -m src.main --force

# Generate website
python -m src.web.generate
```

## Architecture

```
src/
├── data/        # yfinance data provider
├── indicators/  # EMA Cloud, MFI, Williams %R
├── strategies/  # Strategy logic combining indicators
├── signals/     # SQLite position tracker
├── scanner/     # Stock universe (top 100)
├── web/         # Static site generator
└── main.py      # Entry point
```

## Deployment

The scanner runs automatically via GitHub Actions every 10 minutes during market hours. Results are published to GitHub Pages.

## License

MIT
