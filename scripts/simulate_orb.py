"""
Simulate the Opening Range Breakout (ORB) strategy on a specific day.

Replays 10-min candles bar-by-bar, tracking positions and P&L.

Usage:
    python scripts/simulate_orb.py                        # yesterday
    python scripts/simulate_orb.py --date 2026-02-05
    python scripts/simulate_orb.py --tickers AAPL MSFT
    python scripts/simulate_orb.py -v                     # verbose bar-by-bar
"""

import argparse
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytz

from rachatrades.core.data import DataProvider, MTFData
from rachatrades.agents.opening_range import ORBStrategy, ORBConfig
from rachatrades.agents.opening_range.strategy import ORBSignal
from rachatrades.scanner import get_universe

ET = pytz.timezone("America/New_York")


def load_data(tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Fetch 1-min data → resample to 10-min."""
    provider = DataProvider(cache_minutes=0)
    if tickers is None:
        tickers = get_universe()

    print(f"Fetching 1-min data for {len(tickers)} tickers (7 days)...")
    raw_1m = provider.get_batch_ohlcv(tickers, interval="1m", period="7d")
    print(f"Got data for {len(raw_1m)} tickers\n")

    result: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df_1m = raw_1m.get(ticker)
        if df_1m is None or df_1m.empty:
            continue
        df_10m = provider._resample_to_10min(df_1m)
        if df_10m is not None and len(df_10m) >= 20:
            result[ticker] = df_10m
    return result


def run_orb_simulation(
    target_date: datetime,
    data_10m: Dict[str, pd.DataFrame],
    config: Optional[ORBConfig] = None,
    verbose: bool = False,
) -> List[dict]:
    """Run ORB strategy bar-by-bar for one day."""
    strategy = ORBStrategy(config)
    all_trades: List[dict] = []
    open_longs: Dict[str, dict] = {}
    open_shorts: Dict[str, dict] = {}

    for ticker, df_10m_full in data_10m.items():
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
            if len(window) < 10:
                continue

            mtf = MTFData(ticker=ticker, df_10min=window, df_1min=None)
            result = strategy.evaluate_mtf(
                ticker, mtf,
                has_open_position=has_long,
                has_short_position=has_short,
            )

            bar_time = df_10m_full.index[bar_idx]
            bar_time_et = bar_time.tz_convert("America/New_York") if bar_time.tzinfo else bar_time

            if result.signal == ORBSignal.BUY and not has_long:
                has_long = True
                open_longs[ticker] = {
                    "entry_price": result.price,
                    "entry_time": bar_time_et,
                    "reason": result.reason,
                }
                if verbose:
                    print(f"  {bar_time_et.strftime('%H:%M')} BUY  {ticker:<6} @ ${result.price:.2f}  — {result.reason}")

            elif result.signal == ORBSignal.SELL and has_long:
                entry = open_longs.pop(ticker)
                pnl = result.price - entry["entry_price"]
                pnl_pct = (pnl / entry["entry_price"]) * 100
                has_long = False
                all_trades.append({
                    "ticker": ticker, "type": "LONG",
                    "entry_time": entry["entry_time"], "entry_price": entry["entry_price"],
                    "exit_time": bar_time_et, "exit_price": result.price,
                    "pnl": pnl, "pnl_pct": pnl_pct,
                    "reason_in": entry["reason"], "reason_out": result.reason,
                })
                if verbose:
                    print(f"  {bar_time_et.strftime('%H:%M')} SELL {ticker:<6} @ ${result.price:.2f}  P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")

            elif result.signal == ORBSignal.SHORT and not has_short:
                has_short = True
                open_shorts[ticker] = {
                    "entry_price": result.price,
                    "entry_time": bar_time_et,
                    "reason": result.reason,
                }
                if verbose:
                    print(f"  {bar_time_et.strftime('%H:%M')} SHORT {ticker:<6} @ ${result.price:.2f}  — {result.reason}")

            elif result.signal == ORBSignal.COVER and has_short:
                entry = open_shorts.pop(ticker)
                pnl = entry["entry_price"] - result.price
                pnl_pct = (pnl / entry["entry_price"]) * 100
                has_short = False
                all_trades.append({
                    "ticker": ticker, "type": "SHORT",
                    "entry_time": entry["entry_time"], "entry_price": entry["entry_price"],
                    "exit_time": bar_time_et, "exit_price": result.price,
                    "pnl": pnl, "pnl_pct": pnl_pct,
                    "reason_in": entry["reason"], "reason_out": result.reason,
                })
                if verbose:
                    print(f"  {bar_time_et.strftime('%H:%M')} COVER {ticker:<6} @ ${result.price:.2f}  P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")

        # EOD close
        if has_long and ticker in open_longs:
            last = df_10m_full.iloc[day_indices[-1]]
            cp = float(last["Close"])
            entry = open_longs.pop(ticker)
            pnl = cp - entry["entry_price"]
            pnl_pct = (pnl / entry["entry_price"]) * 100
            all_trades.append({
                "ticker": ticker, "type": "LONG",
                "entry_time": entry["entry_time"], "entry_price": entry["entry_price"],
                "exit_time": df_10m_full.index[day_indices[-1]], "exit_price": cp,
                "pnl": pnl, "pnl_pct": pnl_pct,
                "reason_in": entry["reason"], "reason_out": "EOD close",
            })

        if has_short and ticker in open_shorts:
            last = df_10m_full.iloc[day_indices[-1]]
            cp = float(last["Close"])
            entry = open_shorts.pop(ticker)
            pnl = entry["entry_price"] - cp
            pnl_pct = (pnl / entry["entry_price"]) * 100
            all_trades.append({
                "ticker": ticker, "type": "SHORT",
                "entry_time": entry["entry_time"], "entry_price": entry["entry_price"],
                "exit_time": df_10m_full.index[day_indices[-1]], "exit_price": cp,
                "pnl": pnl, "pnl_pct": pnl_pct,
                "reason_in": entry["reason"], "reason_out": "EOD close",
            })

    return all_trades


def print_report(date_str: str, trades: List[dict], data_10m: Dict[str, pd.DataFrame]):
    """Print simulation report."""
    print(f"\n{'=' * 70}")
    print(f"  ORB SIMULATION RESULTS — {date_str}")
    print(f"{'=' * 70}")
    print(f"  Tickers scanned : {len(data_10m)}")
    print(f"  Total trades    : {len(trades)}")

    if not trades:
        print("\n  No trades triggered.\n")
        return

    longs = [t for t in trades if t["type"] == "LONG"]
    shorts = [t for t in trades if t["type"] == "SHORT"]
    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    total_pnl_pct = sum(t["pnl_pct"] for t in trades)
    win_rate = len(winners) / len(trades) * 100

    print(f"  Long trades     : {len(longs)}")
    print(f"  Short trades    : {len(shorts)}")
    print(f"  Winners         : {len(winners)}")
    print(f"  Losers          : {len(losers)}")
    print(f"  Win rate        : {win_rate:.1f}%")
    print(f"  Total P&L ($)   : ${total_pnl:+.2f}")
    print(f"  Total P&L (%)   : {total_pnl_pct:+.2f}%")

    if winners:
        best = max(trades, key=lambda t: t["pnl"])
        print(f"  Best trade      : {best['ticker']} ${best['pnl']:+.2f} ({best['pnl_pct']:+.1f}%)")
    if losers:
        worst = min(trades, key=lambda t: t["pnl"])
        print(f"  Worst trade     : {worst['ticker']} ${worst['pnl']:+.2f} ({worst['pnl_pct']:+.1f}%)")

    avg_win = sum(t["pnl"] for t in winners) / len(winners) if winners else 0
    avg_loss = sum(t["pnl"] for t in losers) / len(losers) if losers else 0
    print(f"  Avg winner ($)  : ${avg_win:+.2f}")
    print(f"  Avg loser ($)   : ${avg_loss:+.2f}")

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


def main():
    parser = argparse.ArgumentParser(description="Simulate ORB strategy on a specific day")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    target = datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime.now() - timedelta(days=1)

    date_str = target.strftime("%Y-%m-%d")
    print(f"{'=' * 70}")
    print(f"  OPENING RANGE BREAKOUT (ORB) SIMULATION — {date_str}")
    print(f"{'=' * 70}")

    data_10m = load_data(args.tickers)
    trades = run_orb_simulation(target, data_10m, verbose=args.verbose)
    print_report(date_str, trades, data_10m)


if __name__ == "__main__":
    main()
