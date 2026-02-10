# RachaTrades - AI Trading Agent Platform

## Project Overview
Autonomous stock market scanner and trading platform powered by the Rashemator EMA Cloud Strategy. Scans 98 stocks every 10 minutes during market hours via GitHub Actions.

## Architecture
- **Agents**: `rachatrades/agents/` — pluggable strategy framework (BaseAgent)
  - **Rashemator** (LIVE): EMA cloud flip strategy on 10-min candles (5/12 + 34/50 clouds)
  - **ORB** (built, not live): Opening Range Breakout (first hour)
- **Core**: `rachatrades/core/` — data (yfinance), indicators (EMA, MFI, Williams %R, order blocks), signals (SQLite position tracker), execution (Alpaca broker)
- **Scanner**: `scripts/scan.py` — main entry point, runs on GitHub Actions every 10 min
- **Notifications**: Email alerts via Gmail SMTP (working)
- **Web**: Static dashboard deployed to GitHub Pages (rachatrades.com)

## Current Live Config
Scanner uses `improved_no_stop` strategy config:
- Cloud spread filter (min 0.05% EMA 5/12 spread)
- No stop loss (exits via cloud flip)
- 3-bar cooldown (30 min) after exits
- Max 5 concurrent positions
- Skip first/last 30 min of session
- Smart EOD close at 3:50 PM: close losers, let winners ride overnight

## Key Scripts
- `scripts/scan.py` — live scanner (GitHub Actions)
- `scripts/simulate.py` — single-day backtester
- `scripts/backtest.py` — multi-day backtester with A/B comparison (`--compare --top 20 --period 60d`)
- `scripts/test_email.py` — email notification tester

## Infrastructure
- **Data**: yfinance (free) — 1-min data resampled to 10-min candles
- **Execution**: Alpaca paper trading (ALPACA_API_KEY, ALPACA_SECRET_KEY)
- **DB**: SQLite at `data/positions.db`
- **CI/CD**: GitHub Actions (`.github/workflows/scanner.yml`) — scans every 10 min during market hours
- **Secrets needed**: SMTP_USER, SMTP_PASSWORD, NOTIFY_EMAILS, ALPACA_API_KEY, ALPACA_SECRET_KEY

## Backtest Results (7-day, top 20 tickers)
- **baseline**: 65 trades, 43% WR, +7.53% P&L, 1.58 PF
- **improved_no_stop** (current): 17 trades, 65% WR, +8.22% P&L, 3.59 PF (best)
- **improved** (with 0.5% stop): 17 trades, 59% WR, +5.68% P&L — stops hurt performance

## Vision / Roadmap
See `grand_vision_startup.txt` for full vision. Next priorities:
1. Get ORB agent live alongside Rashemator
2. Add VIX filter to gate entries during high volatility
3. Economic calendar guard (skip FOMC/CPI days)
4. Telegram/Discord alerts (first monetizable feature)
5. Position sizing (currently qty=1)
6. ES/NQ futures expansion (needs IBKR integration)
7. 90-day verified track record -> audience -> paid signals

## Recent Session (2026-02-09)
- Switched scanner from baseline to improved_no_stop config
- Wired entry guards into scan.py (time-of-day, max positions, cooldown)
- Fixed email notifications (UTF-8 encoding + non-breaking space sanitization)
- Changed EOD close to smart close (only close losers, winners ride overnight)
- Started 60-day backtest comparing all configs vs SPY (+1.85% over 60 days) — rerun: `python3 scripts/backtest.py --compare --top 20 --period 60d`
- Repo should be made private (`gh repo edit --visibility private`)

## Coding Conventions
- Python 3.9+, dataclasses for configs/models
- Strategy configs use feature flags (StrategyConfig)
- New agents inherit from BaseAgent in `rachatrades/agents/base.py`
- Tests: not yet set up (TODO)
- Commits: conventional commits style (feat:, fix:, docs:)
