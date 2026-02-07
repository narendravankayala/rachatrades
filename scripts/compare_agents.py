"""
Compare agents head-to-head on the same day's data.

Runs Rashemator and ORB (and future agents) side-by-side
on identical data to see which performs best.

Usage:
    python scripts/compare_agents.py                      # yesterday
    python scripts/compare_agents.py --date 2026-02-05
    python scripts/compare_agents.py --tickers AAPL MSFT
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

# Reuse the simulation engines
from simulate import load_data, run_simulation, summarize
from simulate_orb import run_orb_simulation


def compare_agents(
    target_date: datetime,
    tickers: Optional[List[str]] = None,
):
    """Run all agents on the same data and compare."""
    date_str = target_date.strftime("%Y-%m-%d")

    # Load data once — shared by all agents
    data_10m = load_data(tickers)

    print(f"\n{'=' * 80}")
    print(f"  AGENT HEAD-TO-HEAD — {date_str}")
    print(f"{'=' * 80}")
    print(f"  Tickers: {len(data_10m)}")
    print(f"  Agents: Rashemator (cloud flip), ORB (opening range breakout)")

    # Run each agent
    results = {}

    print(f"\n  Running [Rashemator]...", end="", flush=True)
    rash_trades = run_simulation(target_date, data_10m, config=None, verbose=False)
    results["Rashemator"] = {"trades": rash_trades, "summary": summarize(rash_trades)}
    s = results["Rashemator"]["summary"]
    print(f" {s['total']} trades, ${s['total_pnl']:+.2f}")

    print(f"  Running [ORB]...", end="", flush=True)
    orb_trades = run_orb_simulation(target_date, data_10m, verbose=False)
    results["ORB"] = {"trades": orb_trades, "summary": summarize(orb_trades)}
    s = results["ORB"]["summary"]
    print(f" {s['total']} trades, ${s['total_pnl']:+.2f}")

    # Comparison table
    print(f"\n{'=' * 80}")
    print(f"  COMPARISON TABLE")
    print(f"{'=' * 80}")

    header = f"  {'Agent':<15s} {'Trades':>6} {'Longs':>6} {'Shorts':>6} {'WinR':>6} {'P&L $':>9} {'P&L %':>8} {'AvgW':>7} {'AvgL':>7}"
    print(header)
    print(f"  {'-' * 74}")

    best_pnl = max(r["summary"]["total_pnl"] for r in results.values())

    for name, r in results.items():
        s = r["summary"]
        marker = " ★" if s["total_pnl"] == best_pnl and s["total"] > 0 else "  "
        print(
            f"{marker}{name:<15s} "
            f"{s['total']:>6} {s['longs']:>6} {s['shorts']:>6} "
            f"{s['win_rate']:>5.1f}% "
            f"${s['total_pnl']:>+8.2f} {s['total_pnl_pct']:>+7.2f}% "
            f"${s['avg_win']:>+6.2f} ${s['avg_loss']:>+6.2f}"
        )

    # Overlap analysis — did both agents trade the same tickers?
    rash_tickers = {t["ticker"] for t in rash_trades}
    orb_tickers = {t["ticker"] for t in orb_trades}
    overlap = rash_tickers & orb_tickers
    rash_only = rash_tickers - orb_tickers
    orb_only = orb_tickers - rash_tickers

    print(f"\n{'=' * 80}")
    print(f"  OVERLAP ANALYSIS")
    print(f"{'=' * 80}")
    print(f"  Both traded        : {len(overlap)} tickers {sorted(overlap) if overlap else ''}")
    print(f"  Rashemator only    : {len(rash_only)} tickers")
    print(f"  ORB only           : {len(orb_only)} tickers")

    # If both traded the same ticker, show head-to-head on those
    if overlap:
        print(f"\n  Head-to-head on shared tickers:")
        for ticker in sorted(overlap):
            r_trades = [t for t in rash_trades if t["ticker"] == ticker]
            o_trades = [t for t in orb_trades if t["ticker"] == ticker]
            r_pnl = sum(t["pnl"] for t in r_trades)
            o_pnl = sum(t["pnl"] for t in o_trades)
            winner = "Rash" if r_pnl > o_pnl else "ORB" if o_pnl > r_pnl else "TIE"
            print(f"    {ticker:<6s}  Rash: ${r_pnl:+.2f}  ORB: ${o_pnl:+.2f}  → {winner}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Compare agents head-to-head")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--tickers", nargs="+", default=None)

    args = parser.parse_args()
    target = datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime.now() - timedelta(days=1)

    compare_agents(target, tickers=args.tickers)


if __name__ == "__main__":
    main()
