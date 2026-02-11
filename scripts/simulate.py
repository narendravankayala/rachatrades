"""
Simulate the Rashemator Cloud Flip Strategy on a specific day.

Replays 10-min candles bar-by-bar, tracking positions and P&L
exactly as the live scanner would.

Usage:
    python scripts/simulate.py                       # yesterday, default config
    python scripts/simulate.py --date 2026-02-05
    python scripts/simulate.py --tickers AAPL MSFT
    python scripts/simulate.py --compare             # A/B test all configs
    python scripts/simulate.py --compare --date 2026-02-05
"""

import argparse
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytz

from rachatrades.core.data import DataProvider, MTFData
from rachatrades.core.indicators import (
    Zone,
    calculate_rashemator_clouds_10min,
)
from rachatrades.agents.rashemator import get_strategy, Signal
from rachatrades.agents.rashemator.strategy import StrategyConfig
from rachatrades.scanner import get_universe

ET = pytz.timezone("America/New_York")


# ═══════════════════════════════════════════════════════════════════════
#  Predefined configs for A/B comparison
# ═══════════════════════════════════════════════════════════════════════

CONFIGS: Dict[str, StrategyConfig] = {
    # ── V1: Cloud Flip (frozen baseline) ─────────────────────────
    "v1_baseline": StrategyConfig(
        name="v1_baseline", version="v1",
    ),
    "v1_improved": StrategyConfig(
        name="v1_improved", version="v1",
        use_cloud_spread_filter=True, min_cloud_spread_pct=0.05,
        cooldown_bars=3, max_positions=5,
        skip_first_minutes=30, skip_last_minutes=30,
    ),

    # ── V2: Pullback Reclaim (Rash's actual system) ──────────────
    "v2_baseline": StrategyConfig(
        name="v2_baseline", version="v2",
    ),
    "v2_spread": StrategyConfig(
        name="v2_spread", version="v2",
        use_cloud_spread_filter=True, min_cloud_spread_pct=0.05,
        cooldown_bars=3, max_positions=5,
        skip_first_minutes=30, skip_last_minutes=30,
    ),
    "v2_oscillator": StrategyConfig(
        name="v2_oscillator", version="v2",
        use_oscillator_filter=True,
    ),
    "v2_ob": StrategyConfig(
        name="v2_ob", version="v2",
        use_order_blocks=True,
    ),
}


# ═══════════════════════════════════════════════════════════════════════
#  Data loading (shared across configs)
# ═══════════════════════════════════════════════════════════════════════

def load_data(
    tickers: Optional[List[str]] = None,
    period: str = "7d",
) -> Dict[str, pd.DataFrame]:
    """Fetch intraday data and return dict of ticker -> full 10-min resampled DF.
    
    For period <= 7d: fetches 1-min data, resamples to 10-min.
    For period > 7d:  fetches 5-min data, resamples to 10-min.
    
    yfinance limits:
      - 1m data: max 7 calendar days
      - 5m data: max 60 calendar days (~40 trading days)
    """
    provider = DataProvider(cache_minutes=0)

    if tickers is None:
        tickers = get_universe()

    # Determine optimal source interval based on requested period
    period_days = int(period.replace("d", "")) if period.endswith("d") else 7
    if period_days <= 7:
        src_interval = "1m"
        src_period = period
    else:
        src_interval = "5m"
        src_period = f"{min(period_days, 60)}d"

    print(f"Fetching {src_interval} data for {len(tickers)} tickers ({src_period})...")
    raw = provider.get_batch_ohlcv(tickers, interval=src_interval, period=src_period)
    print(f"Got data for {len(raw)} tickers\n")

    result: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = raw.get(ticker)
        if df is None or df.empty:
            continue
        df_10m = provider._resample_to_10min(df)
        if df_10m is not None and len(df_10m) >= 55:
            result[ticker] = df_10m

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Core simulation (runs one config on pre-loaded data)
# ═══════════════════════════════════════════════════════════════════════

def run_simulation(
    target_date: datetime,
    data_10m: Dict[str, pd.DataFrame],
    config: Optional[StrategyConfig] = None,
    verbose: bool = False,
) -> List[dict]:
    """
    Run the strategy bar-by-bar for one day using a given config.
    Returns list of trade dicts.
    
    Supports:
    - Cooldown timer: block re-entry for N bars after exit
    - Max concurrent positions: cap total open positions
    - Time-of-day filter: skip first/last N minutes of session
    - Stop loss: pass entry price to strategy for stop loss evaluation
    """
    if config is None:
        config = StrategyConfig()
    strategy = get_strategy(config)
    all_trades: List[dict] = []
    open_longs: Dict[str, dict] = {}
    open_shorts: Dict[str, dict] = {}
    
    # Cooldown tracking: ticker -> bar index when cooldown expires
    cooldown_until: Dict[str, int] = {}
    
    # Market hours for time-of-day filtering
    market_open_minutes = 9 * 60 + 30   # 9:30 AM in minutes
    market_close_minutes = 16 * 60       # 4:00 PM in minutes
    skip_before = market_open_minutes + config.skip_first_minutes
    skip_after = market_close_minutes - config.skip_last_minutes

    for ticker, df_10m_full in data_10m.items():
        # Localize to ET for date filtering
        if df_10m_full.index.tz is None:
            idx = df_10m_full.index.tz_localize("America/New_York")
        else:
            idx = df_10m_full.index.tz_convert("America/New_York")

        target_mask = idx.date == target_date.date()
        day_indices = [i for i, m in enumerate(target_mask) if m]

        if not day_indices:
            continue

        has_long = False
        has_short = False

        for bar_idx in day_indices:
            window = df_10m_full.iloc[: bar_idx + 1]
            if len(window) < 52:
                continue

            bar_time = df_10m_full.index[bar_idx]
            if hasattr(bar_time, "tz") and bar_time.tz is not None:
                bar_time_et = bar_time.tz_convert("America/New_York")
            else:
                bar_time_et = bar_time

            # ── Time-of-day filter: skip entries outside allowed window ──
            bar_minutes = bar_time_et.hour * 60 + bar_time_et.minute
            in_trading_window = skip_before <= bar_minutes < skip_after
            
            # Get entry price for stop loss evaluation
            entry_price = 0.0
            if has_long and ticker in open_longs:
                entry_price = open_longs[ticker]["entry_price"]
            elif has_short and ticker in open_shorts:
                entry_price = open_shorts[ticker]["entry_price"]

            mtf = MTFData(ticker=ticker, df_10min=window, df_1min=None)
            result = strategy.evaluate_mtf(
                ticker, mtf,
                has_open_position=has_long,
                has_short_position=has_short,
                entry_price=entry_price,
            )

            # ── Max positions cap: block new entries if at capacity ──
            total_open = len(open_longs) + len(open_shorts)
            at_capacity = total_open >= config.max_positions
            
            # ── Cooldown check: block re-entry if ticker on cooldown ──
            on_cooldown = ticker in cooldown_until and bar_idx < cooldown_until[ticker]

            if result.signal == Signal.BUY and not has_long:
                if not in_trading_window:
                    pass  # Skip entry outside time window
                elif at_capacity:
                    pass  # Skip entry when at max positions
                elif on_cooldown:
                    pass  # Skip entry during cooldown
                else:
                    has_long = True
                    open_longs[ticker] = {
                        "entry_price": result.price,
                        "entry_time": bar_time_et,
                        "reason": result.reason,
                    }
                    if verbose:
                        print(f"  {bar_time_et.strftime('%H:%M')} BUY  {ticker:<6} @ ${result.price:.2f}  — {result.reason}")

            elif result.signal == Signal.SELL and has_long:
                entry = open_longs.pop(ticker)
                pnl = result.price - entry["entry_price"]
                pnl_pct = (pnl / entry["entry_price"]) * 100
                has_long = False
                # Set cooldown
                cooldown_until[ticker] = bar_idx + config.cooldown_bars
                all_trades.append({
                    "ticker": ticker,
                    "type": "LONG",
                    "entry_time": entry["entry_time"],
                    "entry_price": entry["entry_price"],
                    "exit_time": bar_time_et,
                    "exit_price": result.price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "reason_in": entry["reason"],
                    "reason_out": result.reason,
                })
                if verbose:
                    print(f"  {bar_time_et.strftime('%H:%M')} SELL {ticker:<6} @ ${result.price:.2f}  P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)  — {result.reason}")

            elif result.signal == Signal.SHORT and not has_short:
                if not in_trading_window:
                    pass  # Skip entry outside time window
                elif at_capacity:
                    pass  # Skip entry when at max positions
                elif on_cooldown:
                    pass  # Skip entry during cooldown
                else:
                    has_short = True
                    open_shorts[ticker] = {
                        "entry_price": result.price,
                        "entry_time": bar_time_et,
                        "reason": result.reason,
                    }
                    if verbose:
                        print(f"  {bar_time_et.strftime('%H:%M')} SHORT {ticker:<6} @ ${result.price:.2f}  — {result.reason}")

            elif result.signal == Signal.COVER and has_short:
                entry = open_shorts.pop(ticker)
                pnl = entry["entry_price"] - result.price
                pnl_pct = (pnl / entry["entry_price"]) * 100
                has_short = False
                # Set cooldown
                cooldown_until[ticker] = bar_idx + config.cooldown_bars
                all_trades.append({
                    "ticker": ticker,
                    "type": "SHORT",
                    "entry_time": entry["entry_time"],
                    "entry_price": entry["entry_price"],
                    "exit_time": bar_time_et,
                    "exit_price": result.price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "reason_in": entry["reason"],
                    "reason_out": result.reason,
                })
                if verbose:
                    print(f"  {bar_time_et.strftime('%H:%M')} COVER {ticker:<6} @ ${result.price:.2f}  P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)  — {result.reason}")

        # Close any positions still open at end of day
        if has_long and ticker in open_longs:
            last_bar = df_10m_full.iloc[day_indices[-1]]
            close_price = float(last_bar["Close"])
            entry = open_longs.pop(ticker)
            pnl = close_price - entry["entry_price"]
            pnl_pct = (pnl / entry["entry_price"]) * 100
            all_trades.append({
                "ticker": ticker, "type": "LONG",
                "entry_time": entry["entry_time"], "entry_price": entry["entry_price"],
                "exit_time": df_10m_full.index[day_indices[-1]], "exit_price": close_price,
                "pnl": pnl, "pnl_pct": pnl_pct,
                "reason_in": entry["reason"], "reason_out": "EOD close (open position)",
            })

        if has_short and ticker in open_shorts:
            last_bar = df_10m_full.iloc[day_indices[-1]]
            close_price = float(last_bar["Close"])
            entry = open_shorts.pop(ticker)
            pnl = entry["entry_price"] - close_price
            pnl_pct = (pnl / entry["entry_price"]) * 100
            all_trades.append({
                "ticker": ticker, "type": "SHORT",
                "entry_time": entry["entry_time"], "entry_price": entry["entry_price"],
                "exit_time": df_10m_full.index[day_indices[-1]], "exit_price": close_price,
                "pnl": pnl, "pnl_pct": pnl_pct,
                "reason_in": entry["reason"], "reason_out": "EOD close (open position)",
            })

    return all_trades


# ═══════════════════════════════════════════════════════════════════════
#  Summary helpers
# ═══════════════════════════════════════════════════════════════════════

def summarize(trades: List[dict]) -> dict:
    """Compute summary stats from a list of trades."""
    if not trades:
        return {
            "total": 0, "longs": 0, "shorts": 0,
            "winners": 0, "losers": 0, "win_rate": 0.0,
            "total_pnl": 0.0, "total_pnl_pct": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0,
            "best": None, "worst": None,
        }

    longs = [t for t in trades if t["type"] == "LONG"]
    shorts = [t for t in trades if t["type"] == "SHORT"]
    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    total_pnl_pct = sum(t["pnl_pct"] for t in trades)
    win_rate = len(winners) / len(trades) * 100

    return {
        "total": len(trades),
        "longs": len(longs),
        "shorts": len(shorts),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "avg_win": sum(t["pnl"] for t in winners) / len(winners) if winners else 0,
        "avg_loss": sum(t["pnl"] for t in losers) / len(losers) if losers else 0,
        "best": max(trades, key=lambda t: t["pnl"]) if trades else None,
        "worst": min(trades, key=lambda t: t["pnl"]) if trades else None,
    }


def print_single_report(date_str: str, trades: List[dict], data_10m: Dict[str, pd.DataFrame]):
    """Print detailed report for a single config run."""
    skipped = 0  # we don't track this here, but it's fine
    total_tickers = len(data_10m)
    s = summarize(trades)

    print(f"\n{'=' * 70}")
    print(f"  SIMULATION RESULTS — {date_str}")
    print(f"{'=' * 70}")
    print(f"  Tickers scanned : {total_tickers}")
    print(f"  Total trades    : {s['total']}")

    if not trades:
        print("\n  No trades triggered.\n")
        return

    print(f"  Long trades     : {s['longs']}")
    print(f"  Short trades    : {s['shorts']}")
    print(f"  Winners         : {s['winners']}")
    print(f"  Losers          : {s['losers']}")
    print(f"  Win rate        : {s['win_rate']:.1f}%")
    print(f"  Total P&L ($)   : ${s['total_pnl']:+.2f}")
    print(f"  Total P&L (%)   : {s['total_pnl_pct']:+.2f}%")

    if s["best"]:
        b = s["best"]
        print(f"  Best trade      : {b['ticker']} ${b['pnl']:+.2f} ({b['pnl_pct']:+.1f}%)")
    if s["worst"]:
        w = s["worst"]
        print(f"  Worst trade     : {w['ticker']} ${w['pnl']:+.2f} ({w['pnl_pct']:+.1f}%)")

    print(f"  Avg winner ($)  : ${s['avg_win']:+.2f}")
    print(f"  Avg loser ($)   : ${s['avg_loss']:+.2f}")

    # Trade log
    print(f"\n{'=' * 70}")
    print(f"  TRADE LOG")
    print(f"{'=' * 70}")
    print(f"  {'Type':<6} {'Ticker':<7} {'Entry':>8} {'Exit':>8} {'P&L':>9} {'%':>7}  Entry→Exit Time")
    print(f"  {'-' * 65}")

    for t in sorted(trades, key=lambda x: x["entry_time"]):
        entry_ts = t["entry_time"].strftime("%H:%M") if hasattr(t["entry_time"], "strftime") else str(t["entry_time"])
        exit_ts = t["exit_time"].strftime("%H:%M") if hasattr(t["exit_time"], "strftime") else str(t["exit_time"])
        marker = "✓" if t["pnl"] > 0 else "✗"
        print(
            f"  {marker} {t['type']:<5} {t['ticker']:<7} "
            f"${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
            f"${t['pnl']:>+8.2f} {t['pnl_pct']:>+6.1f}%  "
            f"{entry_ts}→{exit_ts}"
        )

    # P&L by side
    longs = [t for t in trades if t["type"] == "LONG"]
    shorts = [t for t in trades if t["type"] == "SHORT"]
    if longs:
        long_pnl = sum(t["pnl"] for t in longs)
        long_pnl_pct = sum(t["pnl_pct"] for t in longs)
        long_wins = sum(1 for t in longs if t["pnl"] > 0)
        print(f"\n  LONG  trades: {len(longs)} | W/L: {long_wins}/{len(longs)-long_wins} | P&L: ${long_pnl:+.2f} ({long_pnl_pct:+.1f}%)")

    if shorts:
        short_pnl = sum(t["pnl"] for t in shorts)
        short_pnl_pct = sum(t["pnl_pct"] for t in shorts)
        short_wins = sum(1 for t in shorts if t["pnl"] > 0)
        print(f"  SHORT trades: {len(shorts)} | W/L: {short_wins}/{len(shorts)-short_wins} | P&L: ${short_pnl:+.2f} ({short_pnl_pct:+.1f}%)")

    print()


# ═══════════════════════════════════════════════════════════════════════
#  Comparison mode — run all configs, print side-by-side
# ═══════════════════════════════════════════════════════════════════════

def compare_configs(
    target_date: datetime,
    data_10m: Dict[str, pd.DataFrame],
    configs: Optional[Dict[str, StrategyConfig]] = None,
    verbose: bool = False,
):
    """Run multiple strategy configs on the same data and compare results."""
    if configs is None:
        configs = CONFIGS

    date_str = target_date.strftime("%Y-%m-%d")
    print(f"\n{'=' * 80}")
    print(f"  A/B COMPARISON — {date_str}")
    print(f"{'=' * 80}")
    print(f"  Tickers: {len(data_10m)} | Configs: {len(configs)}")
    print(f"  {'-' * 76}")

    for name, cfg in configs.items():
        print(f"  • {name:20s} → {cfg.describe()}")

    results: Dict[str, dict] = {}
    for name, cfg in configs.items():
        print(f"\n  Running [{name}]...", end="", flush=True)
        trades = run_simulation(target_date, data_10m, config=cfg, verbose=False)
        results[name] = {"trades": trades, "summary": summarize(trades)}
        s = results[name]["summary"]
        print(f" {s['total']} trades, ${s['total_pnl']:+.2f}")

    # ── Comparison table ──
    print(f"\n{'=' * 80}")
    print(f"  COMPARISON TABLE")
    print(f"{'=' * 80}")

    header = f"  {'Config':<20s} {'Trades':>6} {'Longs':>6} {'Shorts':>6} {'WinR':>6} {'P&L $':>9} {'P&L %':>8} {'AvgW':>7} {'AvgL':>7}"
    print(header)
    print(f"  {'-' * 76}")

    best_pnl = max(r["summary"]["total_pnl"] for r in results.values())

    for name, r in results.items():
        s = r["summary"]
        marker = " ★" if s["total_pnl"] == best_pnl and s["total"] > 0 else "  "
        print(
            f"{marker}{name:<20s} "
            f"{s['total']:>6} {s['longs']:>6} {s['shorts']:>6} "
            f"{s['win_rate']:>5.1f}% "
            f"${s['total_pnl']:>+8.2f} {s['total_pnl_pct']:>+7.2f}% "
            f"${s['avg_win']:>+6.2f} ${s['avg_loss']:>+6.2f}"
        )

    # ── Filtered trades analysis ──
    print(f"\n{'=' * 80}")
    print(f"  FILTER IMPACT ANALYSIS")
    print(f"{'=' * 80}")

    baseline_trades = results.get("baseline", {}).get("trades", [])
    if baseline_trades:
        bl = summarize(baseline_trades)
        for name, r in results.items():
            if name == "baseline":
                continue
            s = r["summary"]
            trade_diff = s["total"] - bl["total"]
            pnl_diff = s["total_pnl"] - bl["total_pnl"]
            wr_diff = s["win_rate"] - bl["win_rate"]
            print(
                f"  {name:20s} vs baseline: "
                f"trades {trade_diff:+d}, "
                f"P&L ${pnl_diff:+.2f}, "
                f"win rate {wr_diff:+.1f}pp"
            )

    print()

    # If verbose, print the full trade log for the best config
    if verbose:
        best_name = max(results, key=lambda n: results[n]["summary"]["total_pnl"])
        best_trades = results[best_name]["trades"]
        print(f"  ★ Best config: [{best_name}] — detailed trade log:")
        print_single_report(date_str, best_trades, data_10m)


# ═══════════════════════════════════════════════════════════════════════
#  Legacy entrypoint (single config)
# ═══════════════════════════════════════════════════════════════════════

def simulate_day(
    target_date: datetime,
    tickers: Optional[List[str]] = None,
    verbose: bool = False,
):
    """Simulate with default config — backward compatible."""
    date_str = target_date.strftime("%Y-%m-%d")
    print(f"{'=' * 70}")
    print(f"  RASHEMATOR CLOUD FLIP SIMULATION — {date_str}")
    print(f"{'=' * 70}")

    data_10m = load_data(tickers)
    trades = run_simulation(target_date, data_10m, config=None, verbose=verbose)
    print_single_report(date_str, trades, data_10m)


def main():
    parser = argparse.ArgumentParser(description="Simulate Rashemator strategy on a specific day")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date in YYYY-MM-DD format (default: yesterday)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Specific tickers to simulate (default: full universe)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print each signal as it occurs",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all predefined configs and show A/B comparison",
    )

    args = parser.parse_args()

    if args.date:
        target = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        target = datetime.now() - timedelta(days=1)

    if args.compare:
        data_10m = load_data(args.tickers)
        compare_configs(target, data_10m, verbose=args.verbose)
    else:
        simulate_day(target, tickers=args.tickers, verbose=args.verbose)


if __name__ == "__main__":
    main()
