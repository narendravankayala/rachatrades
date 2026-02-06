# rachatrades

Autonomous stock market scanner powered by the **Rashemator EMA Cloud Strategy** — a multi-timeframe cloud flip system on 10-minute candles with trend filtering.

Live dashboard: [rachatrades.com](https://rachatrades.com)

## Features

- 📊 Scans ~100 US stocks by market cap every 10 minutes during market hours
- ☁️ Rashemator Cloud Flip Strategy: 5/12 EMA cloud flips with 34/50 trend filter
- 📈 Long & short position tracking with P&L calculation (SQLite)
- 📧 Real-time email alerts on BUY / SELL / SHORT / COVER signals
- 🌐 Auto-generated dashboard at [rachatrades.com](https://rachatrades.com)
- ⚡ Fully autonomous via GitHub Actions — zero cost to operate
- 🧩 Agent-based architecture — pluggable strategies for future expansion

## Strategy — Rashemator Cloud Flip

All analysis runs on **true 10-minute candles** (1-min data resampled from yfinance).

| Signal | Condition |
|--------|-----------|
| **BUY** | 5/12 EMA cloud flips bullish **AND** 34/50 major cloud is bullish |
| **SELL** | 5/12 EMA cloud flips bearish |
| **SHORT** | 5/12 EMA cloud flips bearish **AND** 34/50 major cloud is bearish |
| **COVER** | 5/12 EMA cloud flips bullish |

The **34/50 major cloud** acts as a trend filter — you only go long in uptrends and short in downtrends.

Oscillators (MFI 14, Williams %R 14) provide additional confirmation context.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run scanner (--force to run outside market hours)
python scripts/scan.py --force

# Dry run (no positions opened/closed)
python scripts/scan.py --force --dry-run

# Analyze current market zones & opportunities
python scripts/analyze.py

# Generate website
python -m rachatrades.web.generate
```

## Architecture

```
rachatrades/                  # Main package
├── core/                     # Shared infrastructure
│   ├── data/                 #   DataProvider (yfinance → 10min resampling)
│   ├── indicators/           #   EMA Clouds, MFI, Williams %R
│   └── signals/              #   PositionTracker (SQLite, long + short)
├── agents/                   # Pluggable trading strategies
│   ├── base.py               #   BaseAgent abstract class
│   └── rashemator/           #   Rashemator cloud flip strategy
├── scanner/                  # Stock universe (top ~100 by market cap)
├── notifications/            # Email alerts (Gmail SMTP)
└── web/                      # Static site generator (Jinja2)

scripts/                      # Entry points
├── scan.py                   #   Market scanner
└── analyze.py                #   Market analysis tool

docs/                         # Documentation
├── vision.md                 #   Grand vision & roadmap
└── strategies/               #   Strategy params & explainers

tests/                        # Test suite
```

## Deployment

The scanner runs automatically via **GitHub Actions** every 10 minutes during US market hours (9:30 AM – 4:00 PM ET, Mon–Fri). Results are published to GitHub Pages and trade alerts are emailed in real time.

### Environment Variables (GitHub Secrets)

| Secret | Description |
|--------|-------------|
| `SMTP_USER` | Gmail address for sending alerts |
| `SMTP_PASSWORD` | Gmail app password |
| `NOTIFY_EMAILS` | Comma-separated recipient emails |

## License

MIT
