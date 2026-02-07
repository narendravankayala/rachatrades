# RachaTrades — Grand Vision & Roadmap

## The Goal

Build an **autonomous, zero-cost stock trading signal system** that:
1. Scans the top ~100 US stocks every 10 minutes during market hours
2. Generates BUY / SELL / SHORT / COVER signals based on a modular, testable strategy
3. Publishes results to [rachatrades.com](https://rachatrades.com) automatically
4. Continuously improves by A/B testing indicator combinations against real data

No paid APIs. No paid hosting. No manual intervention. Everything runs on GitHub Actions free tier.

---

## Architecture Philosophy

### Agent-Based Design
Each strategy is a self-contained **agent** under `rachatrades/agents/`. The first (and current) agent is **Rashemator** — a cloud flip system. Future agents can implement different strategies entirely (momentum, mean reversion, etc.) and run in parallel.

### Modular Indicators
Indicators live in `rachatrades/core/indicators/` and are shared across all agents. Each indicator is independent and stateless — give it a DataFrame, get a signal back.

**Current indicators:**
- EMA Clouds (5/12, 8/9, 34/50 on 10-min)
- MFI (Money Flow Index)
- Williams %R
- Order Blocks (institutional supply/demand zones — ported from LuxAlgo)

### Feature Flag System
Every indicator/filter is gated behind a boolean flag in `StrategyConfig`. This lets us:
- Toggle filters on/off without changing code
- A/B test combinations using the simulator
- Roll back any change by flipping a flag
- Keep the live scanner on a proven config while testing new ideas

```python
StrategyConfig(
    use_oscillator_filter=False,   # MFI/WR confirmation for entries
    use_order_blocks=False,        # OB confluence: only enter near S/R zones
    # Future:
    # use_volume_filter=False,     # Above-average volume required
    # use_vwap_filter=False,       # VWAP trend confirmation
    # use_atr_stops=False,         # ATR-based stop loss
    # use_rsi_filter=False,        # RSI divergence confirmation
)
```

---

## Current Strategy: Rashemator Cloud Flip (v0.2)

All analysis on **true 10-minute candles** (1-min data from yfinance, resampled).

### Core Logic (always active)
| Signal | Condition |
|--------|-----------|
| **BUY** | 5/12 cloud flips bullish AND 34/50 cloud is bullish (uptrend) |
| **SELL** | 5/12 cloud flips bearish |
| **SHORT** | 5/12 cloud flips bearish AND 34/50 cloud is bearish (downtrend) |
| **COVER** | 5/12 cloud flips bullish |

### Optional Filters (toggled via feature flags)
| Filter | What It Does | Flag |
|--------|-------------|------|
| Oscillator | Require MFI < 20 or WR < -80 for longs, MFI > 80 or WR > -20 for shorts | `use_oscillator_filter` |
| Order Blocks | Only enter near institutional support (longs) or resistance (shorts) | `use_order_blocks` |

---

## Simulation & Backtesting

The simulator (`scripts/simulate.py`) replays 10-min candles bar-by-bar, exactly matching live scanner logic.

```bash
# Single config, yesterday
python scripts/simulate.py

# A/B comparison — runs 4 configs on same data
python scripts/simulate.py --compare

# Specific date and tickers
python scripts/simulate.py --compare --date 2026-02-05 --tickers AAPL MSFT NVDA
```

### Comparison Mode
Runs these predefined configs side-by-side:
- **baseline** — pure cloud flip (no filters)
- **oscillator** — + MFI/WR confirmation
- **order_blocks** — + OB confluence
- **osc+ob** — both filters combined

Output includes trade counts, win rates, P&L, and filter impact analysis vs baseline.

---

## Roadmap

### Phase 1 — Foundation ✅
- [x] Rashemator cloud flip strategy on 10-min candles
- [x] Long + short position tracking (SQLite)
- [x] Email notifications on signals
- [x] GitHub Actions scanner (every 10 min, market hours)
- [x] Auto-published dashboard at rachatrades.com
- [x] Agent-based package architecture (v0.2.0)

### Phase 2 — Modular Testing ✅
- [x] Feature flag system in StrategyConfig
- [x] Order Block indicator (LuxAlgo port)
- [x] A/B comparison simulator
- [x] Filter impact analysis

### Phase 3 — More Indicators 🔜
- [ ] Volume filter (above-average volume for entries)
- [ ] VWAP filter (trend confirmation)
- [ ] RSI divergence detection
- [ ] ATR-based dynamic stops (instead of cloud flip exits)
- [ ] Fair Value Gaps (FVG) as entry zones
- [ ] Multi-day backtesting (run simulate across a date range)

### Phase 4 — Smarter Exits
- [ ] Trailing stops using ATR
- [ ] Partial profit taking at key levels
- [ ] Time-based exits (max hold period)
- [ ] EOD auto-close for day trading mode

### Phase 5 — Multi-Strategy
- [ ] Second agent (momentum / mean reversion)
- [ ] Strategy allocation (which agent gets which tickers)
- [ ] Ensemble signals (multiple agents must agree)
- [ ] Performance dashboard comparing agents over time

### Phase 6 — Beyond Signals
- [ ] Paper trading integration (Alpaca API)
- [ ] Risk management (position sizing, max exposure)
- [ ] Portfolio-level analytics
- [ ] Mobile push notifications

---

## Design Principles

1. **Zero cost** — No paid APIs, no servers, no subscriptions
2. **Fully autonomous** — Runs without human intervention
3. **Reversible** — Every change is behind a feature flag
4. **Testable** — Every change can be backtested before going live
5. **Simple > clever** — Prefer clear code over complex optimizations
6. **Data-driven** — Let the simulator tell us what works, not gut feelings