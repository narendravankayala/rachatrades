"""
Multi-Day Backtester for the Rashemator Strategy.

Runs the simulator across ALL trading days available in the data
(yfinance gives ~5 trading days of 1-min data), aggregates results,
and produces a comprehensive performance report.

Usage:
    python scripts/backtest.py                       # All days, default config
    python scripts/backtest.py --compare             # A/B test all configs
    python scripts/backtest.py --tickers AAPL MSFT   # Specific tickers
    python scripts/backtest.py -v                    # Verbose trade log
    python scripts/backtest.py --top 20              # Top 20 by market cap

Examples:
    # Quick sanity check on a few tickers
    python scripts/backtest.py --tickers AAPL NVDA TSLA META

    # Full universe A/B comparison
    python scripts/backtest.py --compare --top 50
"""

import argparse
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytz

from rachatrades.core.data import DataProvider
from rachatrades.agents.rashemator.strategy import StrategyConfig
from rachatrades.scanner import get_universe

# Reuse simulation engine from simulate.py
from scripts.simulate import (
    CONFIGS,
    load_data,
    run_simulation,
    summarize,
)

ET = pytz.timezone("America/New_York")


# ═══════════════════════════════════════════════════════════════════════
#  Multi-day helpers
# ═══════════════════════════════════════════════════════════════════════

def find_trading_days(data_10m: Dict[str, pd.DataFrame]) -> List[datetime]:
    """
    Discover all unique trading days present in the data.

    Returns sorted list of datetime objects representing each trading day.
    """
    all_dates = set()
    for ticker, df in data_10m.items():
        if df.index.tz is None:
            idx = df.index.tz_localize("America/New_York")
        else:
            idx = df.index.tz_convert("America/New_York")
        for dt in idx:
            all_dates.add(dt.date())

    # Convert to datetime and sort
    trading_days = sorted([
        datetime(d.year, d.month, d.day) for d in all_dates
        if d.weekday() < 5  # Mon-Fri
    ])

    return trading_days


def compute_equity_curve(
    daily_results: List[dict],
    starting_capital: float = 100_000.0,
) -> List[dict]:
    """
    Build an equity curve from daily P&L results.

    Assumes $1,000 position per trade (1% of capital) to normalize P&L
    from percentage returns.
    """
    curve = []
    equity = starting_capital
    peak = starting_capital
    position_size = 1_000.0  # $1k per trade for P&L normalization

    for day in daily_results:
        # Convert percentage P&L to dollar P&L using fixed position size
        day_pnl = sum(
            t["pnl_pct"] / 100.0 * position_size
            for t in day["trades"]
        )
        equity += day_pnl
        peak = max(peak, equity)
        drawdown = (equity - peak) / peak * 100 if peak > 0 else 0

        curve.append({
            "date": day["date"],
            "equity": equity,
            "day_pnl": day_pnl,
            "peak": peak,
            "drawdown_pct": drawdown,
            "trades": day["num_trades"],
            "win_rate": day["win_rate"],
        })

    return curve


def compute_advanced_stats(
    all_trades: List[dict],
    equity_curve: List[dict],
) -> dict:
    """Compute advanced backtest statistics."""
    if not all_trades:
        return {
            "total_trades": 0,
            "profit_factor": 0,
            "max_drawdown_pct": 0,
            "avg_trades_per_day": 0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "expectancy": 0,
            "sharpe_approx": 0,
        }

    winners = [t for t in all_trades if t["pnl"] > 0]
    losers = [t for t in all_trades if t["pnl"] <= 0]

    gross_profit = sum(t["pnl_pct"] for t in winners) if winners else 0
    gross_loss = abs(sum(t["pnl_pct"] for t in losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    max_dd = min(e["drawdown_pct"] for e in equity_curve) if equity_curve else 0

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    curr_wins = 0
    curr_losses = 0
    for t in all_trades:
        if t["pnl"] > 0:
            curr_wins += 1
            curr_losses = 0
            max_consec_wins = max(max_consec_wins, curr_wins)
        else:
            curr_losses += 1
            curr_wins = 0
            max_consec_losses = max(max_consec_losses, curr_losses)

    # Expectancy (avg pnl per trade in %)
    expectancy = sum(t["pnl_pct"] for t in all_trades) / len(all_trades)

    # Approximate Sharpe (daily returns)
    if len(equity_curve) > 1:
        daily_returns = [e["day_pnl"] for e in equity_curve]
        import numpy as np
        dr = np.array(daily_returns)
        if dr.std() > 0:
            sharpe = (dr.mean() / dr.std()) * (252 ** 0.5)  # Annualized
        else:
            sharpe = 0
    else:
        sharpe = 0

    num_days = len(equity_curve) if equity_curve else 1

    return {
        "total_trades": len(all_trades),
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd,
        "avg_trades_per_day": len(all_trades) / num_days,
        "consecutive_wins": max_consec_wins,
        "consecutive_losses": max_consec_losses,
        "expectancy": expectancy,
        "sharpe_approx": sharpe,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Single-config multi-day backtest
# ═══════════════════════════════════════════════════════════════════════

def run_multiday_backtest(
    data_10m: Dict[str, pd.DataFrame],
    config: Optional[StrategyConfig] = None,
    verbose: bool = False,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Run the strategy across all available trading days.

    Returns:
        (all_trades, daily_results, equity_curve)
    """
    trading_days = find_trading_days(data_10m)

    if not trading_days:
        print("  No trading days found in data!")
        return [], [], []

    config_name = config.name if config else "default"
    all_trades: List[dict] = []
    daily_results: List[dict] = []

    for day in trading_days:
        date_str = day.strftime("%Y-%m-%d (%a)")
        trades = run_simulation(day, data_10m, config=config, verbose=verbose)

        s = summarize(trades)
        daily_results.append({
            "date": day,
            "date_str": date_str,
            "trades": trades,
            "num_trades": s["total"],
            "longs": s["longs"],
            "shorts": s["shorts"],
            "winners": s["winners"],
            "losers": s["losers"],
            "win_rate": s["win_rate"],
            "total_pnl": s["total_pnl"],
            "total_pnl_pct": s["total_pnl_pct"],
        })
        all_trades.extend(trades)

    equity_curve = compute_equity_curve(daily_results)

    return all_trades, daily_results, equity_curve


# ═══════════════════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════════════════

def print_multiday_report(
    all_trades: List[dict],
    daily_results: List[dict],
    equity_curve: List[dict],
    config_name: str = "default",
    data_10m: Dict[str, pd.DataFrame] = None,
    verbose: bool = False,
):
    """Print comprehensive multi-day backtest report."""

    num_days = len(daily_results)
    num_tickers = len(data_10m) if data_10m else 0
    s = summarize(all_trades)
    adv = compute_advanced_stats(all_trades, equity_curve)

    print()
    print(f"{'═' * 72}")
    print(f"  MULTI-DAY BACKTEST REPORT — [{config_name}]")
    print(f"{'═' * 72}")

    if num_days == 0:
        print("  No trading days found.\n")
        return

    first_day = daily_results[0]["date_str"]
    last_day = daily_results[-1]["date_str"]

    print(f"  Period         : {first_day} → {last_day} ({num_days} trading days)")
    print(f"  Tickers        : {num_tickers}")
    print(f"  Total trades   : {s['total']}")
    print(f"  Avg trades/day : {adv['avg_trades_per_day']:.1f}")
    print()

    # ── Win/Loss Stats ──────────────────────────────────────────────
    print(f"  {'─' * 44}")
    print(f"  Long trades    : {s['longs']}")
    print(f"  Short trades   : {s['shorts']}")
    print(f"  Winners        : {s['winners']}")
    print(f"  Losers         : {s['losers']}")
    print(f"  Win rate       : {s['win_rate']:.1f}%")
    print()

    # ── P&L Stats ───────────────────────────────────────────────────
    print(f"  {'─' * 44}")
    print(f"  Total P&L ($)  : ${s['total_pnl']:+.2f}")
    print(f"  Total P&L (%)  : {s['total_pnl_pct']:+.2f}%")
    print(f"  Avg winner ($) : ${s['avg_win']:+.2f}")
    print(f"  Avg loser ($)  : ${s['avg_loss']:+.2f}")
    if s["best"]:
        b = s["best"]
        print(f"  Best trade     : {b['ticker']} ${b['pnl']:+.2f} ({b['pnl_pct']:+.1f}%)")
    if s["worst"]:
        w = s["worst"]
        print(f"  Worst trade    : {w['ticker']} ${w['pnl']:+.2f} ({w['pnl_pct']:+.1f}%)")
    print()

    # ── Advanced Stats ──────────────────────────────────────────────
    print(f"  {'─' * 44}")
    pf_str = f"{adv['profit_factor']:.2f}" if adv['profit_factor'] != float('inf') else "∞"
    print(f"  Profit factor  : {pf_str}")
    print(f"  Expectancy (%) : {adv['expectancy']:+.3f}% per trade")
    print(f"  Max drawdown   : {adv['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe (approx): {adv['sharpe_approx']:.2f}")
    print(f"  Max consec wins: {adv['consecutive_wins']}")
    print(f"  Max consec loss: {adv['consecutive_losses']}")
    print()

    # ── Per-Day Breakdown ───────────────────────────────────────────
    print(f"{'═' * 72}")
    print(f"  PER-DAY BREAKDOWN")
    print(f"{'═' * 72}")
    print(f"  {'Date':<20s} {'Trades':>6} {'W':>4} {'L':>4} {'WR%':>6} {'P&L $':>10} {'P&L %':>8}  Equity")
    print(f"  {'─' * 68}")

    for i, day in enumerate(daily_results):
        eq = equity_curve[i]["equity"] if i < len(equity_curve) else 0
        dd = equity_curve[i]["drawdown_pct"] if i < len(equity_curve) else 0

        day_marker = "+" if day["total_pnl_pct"] > 0 else ("-" if day["total_pnl_pct"] < 0 else " ")
        dd_str = f" (DD:{dd:.1f}%)" if dd < -0.5 else ""

        print(
            f"  {day_marker} {day['date_str']:<18s} "
            f"{day['num_trades']:>6} "
            f"{day['winners']:>4} {day['losers']:>4} "
            f"{day['win_rate']:>5.1f}% "
            f"${day['total_pnl']:>+9.2f} {day['total_pnl_pct']:>+7.2f}% "
            f"${eq:>10,.0f}{dd_str}"
        )

    print()

    # ── Equity Curve (ASCII) ────────────────────────────────────────
    if equity_curve:
        print(f"{'═' * 72}")
        print(f"  EQUITY CURVE")
        print(f"{'═' * 72}")
        _print_ascii_equity_curve(equity_curve)
        print()

    # ── P&L by Ticker ───────────────────────────────────────────────
    if all_trades:
        print(f"{'═' * 72}")
        print(f"  P&L BY TICKER (top 10 most active)")
        print(f"{'═' * 72}")
        _print_ticker_breakdown(all_trades)
        print()

    # ── Long vs Short Analysis ──────────────────────────────────────
    if all_trades:
        print(f"{'═' * 72}")
        print(f"  LONG vs SHORT ANALYSIS")
        print(f"{'═' * 72}")
        _print_side_analysis(all_trades)
        print()

    # ── Trade Log (verbose) ─────────────────────────────────────────
    if verbose and all_trades:
        print(f"{'═' * 72}")
        print(f"  FULL TRADE LOG ({len(all_trades)} trades)")
        print(f"{'═' * 72}")
        print(f"  {'Date':<12} {'Time':>5} {'Type':<6} {'Ticker':<7} {'Entry':>8} {'Exit':>8} {'P&L':>9} {'%':>7}  Reason")
        print(f"  {'─' * 68}")

        for t in sorted(all_trades, key=lambda x: x["entry_time"]):
            entry_date = t["entry_time"].strftime("%Y-%m-%d") if hasattr(t["entry_time"], "strftime") else ""
            entry_time = t["entry_time"].strftime("%H:%M") if hasattr(t["entry_time"], "strftime") else ""
            marker = "✓" if t["pnl"] > 0 else "✗"
            reason_short = t["reason_out"][:30] if len(t["reason_out"]) > 30 else t["reason_out"]
            print(
                f"  {marker} {entry_date} {entry_time:>5} {t['type']:<5} {t['ticker']:<7} "
                f"${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
                f"${t['pnl']:>+8.2f} {t['pnl_pct']:>+6.1f}%  "
                f"{reason_short}"
            )
        print()

    # ── Verdict ─────────────────────────────────────────────────────
    print(f"{'═' * 72}")
    _print_verdict(s, adv, num_days)
    print(f"{'═' * 72}\n")


def _print_ascii_equity_curve(equity_curve: List[dict], width: int = 50):
    """Print a simple ASCII equity curve."""
    values = [e["equity"] for e in equity_curve]
    if not values:
        return

    min_val = min(values)
    max_val = max(values)
    value_range = max_val - min_val if max_val != min_val else 1

    for i, e in enumerate(equity_curve):
        bar_len = int((e["equity"] - min_val) / value_range * width)
        bar = "█" * bar_len
        day_label = e["date"].strftime("%m/%d")
        pnl_marker = "▲" if e["day_pnl"] > 0 else ("▼" if e["day_pnl"] < 0 else "─")
        print(f"  {day_label} │{bar} ${e['equity']:,.0f} {pnl_marker}")


def _print_ticker_breakdown(all_trades: List[dict]):
    """Print P&L breakdown by ticker."""
    ticker_stats = defaultdict(lambda: {"trades": 0, "pnl": 0, "pnl_pct": 0, "wins": 0})

    for t in all_trades:
        ts = ticker_stats[t["ticker"]]
        ts["trades"] += 1
        ts["pnl"] += t["pnl"]
        ts["pnl_pct"] += t["pnl_pct"]
        if t["pnl"] > 0:
            ts["wins"] += 1

    # Sort by number of trades (most active)
    sorted_tickers = sorted(ticker_stats.items(), key=lambda x: x[1]["trades"], reverse=True)[:10]

    print(f"  {'Ticker':<7} {'Trades':>6} {'W':>4} {'L':>4} {'WR%':>6} {'P&L $':>10} {'P&L %':>8}")
    print(f"  {'─' * 50}")

    for ticker, ts in sorted_tickers:
        wr = ts["wins"] / ts["trades"] * 100 if ts["trades"] > 0 else 0
        losses = ts["trades"] - ts["wins"]
        marker = "+" if ts["pnl_pct"] > 0 else "-" if ts["pnl_pct"] < 0 else " "
        print(
            f"  {marker} {ticker:<6} {ts['trades']:>6} {ts['wins']:>4} {losses:>4} "
            f"{wr:>5.1f}% ${ts['pnl']:>+9.2f} {ts['pnl_pct']:>+7.2f}%"
        )


def _print_side_analysis(all_trades: List[dict]):
    """Print long vs short comparison."""
    longs = [t for t in all_trades if t["type"] == "LONG"]
    shorts = [t for t in all_trades if t["type"] == "SHORT"]

    for label, trades in [("LONG", longs), ("SHORT", shorts)]:
        if not trades:
            print(f"  {label:5s}: No trades")
            continue

        wins = sum(1 for t in trades if t["pnl"] > 0)
        total_pnl = sum(t["pnl"] for t in trades)
        total_pnl_pct = sum(t["pnl_pct"] for t in trades)
        wr = wins / len(trades) * 100

        avg_hold_bars = []
        for t in trades:
            if hasattr(t["entry_time"], "timestamp") and hasattr(t["exit_time"], "timestamp"):
                try:
                    hold_mins = (t["exit_time"] - t["entry_time"]).total_seconds() / 60
                    avg_hold_bars.append(hold_mins)
                except:
                    pass

        avg_hold = sum(avg_hold_bars) / len(avg_hold_bars) if avg_hold_bars else 0

        print(
            f"  {label:5s}: {len(trades)} trades | "
            f"W/L: {wins}/{len(trades)-wins} ({wr:.0f}%) | "
            f"P&L: ${total_pnl:+.2f} ({total_pnl_pct:+.1f}%) | "
            f"Avg hold: {avg_hold:.0f} min"
        )


def _print_verdict(s: dict, adv: dict, num_days: int):
    """Print a human-readable verdict on strategy health."""
    issues = []
    strengths = []

    # Win rate assessment
    if s["win_rate"] >= 55:
        strengths.append(f"Win rate {s['win_rate']:.0f}% is solid")
    elif s["win_rate"] >= 45:
        issues.append(f"Win rate {s['win_rate']:.0f}% is marginal — near coin flip")
    elif s["total"] > 0:
        issues.append(f"Win rate {s['win_rate']:.0f}% is LOW — strategy is losing more than winning")

    # Profit factor
    if adv["profit_factor"] >= 1.5:
        strengths.append(f"Profit factor {adv['profit_factor']:.2f} — winners > losers by good margin")
    elif adv["profit_factor"] >= 1.0:
        issues.append(f"Profit factor {adv['profit_factor']:.2f} — barely profitable")
    elif s["total"] > 0:
        issues.append(f"Profit factor {adv['profit_factor']:.2f} — LOSING money overall")

    # Trade frequency
    if adv["avg_trades_per_day"] < 1 and num_days > 1:
        issues.append(f"Only {adv['avg_trades_per_day']:.1f} trades/day — strategy may be too selective")
    elif adv["avg_trades_per_day"] > 20:
        issues.append(f"{adv['avg_trades_per_day']:.0f} trades/day — overtrading, commission drag will hurt")

    # Drawdown
    if adv["max_drawdown_pct"] < -5:
        issues.append(f"Max drawdown {adv['max_drawdown_pct']:.1f}% — significant capital risk")

    # Expectancy
    if adv["expectancy"] > 0:
        strengths.append(f"Positive expectancy: {adv['expectancy']:+.3f}% per trade")
    elif s["total"] > 0:
        issues.append(f"Negative expectancy: {adv['expectancy']:+.3f}% per trade")

    # Consecutive losses
    if adv["consecutive_losses"] >= 5:
        issues.append(f"Max {adv['consecutive_losses']} consecutive losses — will test discipline")

    # No trades
    if s["total"] == 0:
        issues.append("ZERO trades generated — strategy never triggered")

    print(f"  VERDICT")
    print(f"  {'─' * 44}")
    if strengths:
        for item in strengths:
            print(f"  ✓ {item}")
    if issues:
        for item in issues:
            print(f"  ✗ {item}")

    if not issues and strengths:
        print(f"\n  → Strategy looks HEALTHY. Ready for paper trading.")
    elif not issues and not strengths:
        print(f"\n  → Insufficient data to judge.")
    elif len(issues) <= 1 and strengths:
        print(f"\n  → Strategy is OKAY but has room to improve.")
    else:
        print(f"\n  → Strategy needs WORK. Review the issues above before going live.")


# ═══════════════════════════════════════════════════════════════════════
#  Multi-config comparison across all days
# ═══════════════════════════════════════════════════════════════════════

def compare_configs_multiday(
    data_10m: Dict[str, pd.DataFrame],
    configs: Optional[Dict[str, StrategyConfig]] = None,
    verbose: bool = False,
):
    """Run all configs across all trading days and compare."""
    if configs is None:
        configs = CONFIGS

    trading_days = find_trading_days(data_10m)
    num_days = len(trading_days)
    num_tickers = len(data_10m)

    print()
    print(f"{'═' * 80}")
    print(f"  MULTI-DAY A/B COMPARISON")
    print(f"{'═' * 80}")

    if not trading_days:
        print("  No trading days found.\n")
        return

    first_day = trading_days[0].strftime("%Y-%m-%d (%a)")
    last_day = trading_days[-1].strftime("%Y-%m-%d (%a)")

    print(f"  Period  : {first_day} → {last_day} ({num_days} trading days)")
    print(f"  Tickers : {num_tickers}")
    print(f"  Configs : {len(configs)}")
    print(f"  {'─' * 76}")

    for name, cfg in configs.items():
        print(f"  • {name:20s} → {cfg.describe()}")

    # Run each config
    results: Dict[str, dict] = {}
    for name, cfg in configs.items():
        print(f"\n  Running [{name}] across {num_days} days...", end="", flush=True)
        all_trades, daily_results, equity_curve = run_multiday_backtest(
            data_10m, config=cfg, verbose=False
        )
        s = summarize(all_trades)
        adv = compute_advanced_stats(all_trades, equity_curve)
        results[name] = {
            "trades": all_trades,
            "daily": daily_results,
            "equity": equity_curve,
            "summary": s,
            "advanced": adv,
        }
        print(f" {s['total']} trades, ${s['total_pnl']:+.2f}, WR: {s['win_rate']:.0f}%")

    # ── Overall Comparison Table ────────────────────────────────────
    print(f"\n{'═' * 80}")
    print(f"  OVERALL COMPARISON ({num_days} days)")
    print(f"{'═' * 80}")

    header = (
        f"  {'Config':<20s} {'Trades':>6} {'WR%':>6} "
        f"{'P&L %':>8} {'PF':>6} {'Sharpe':>7} "
        f"{'MaxDD':>7} {'E[R]':>8} {'CW':>3} {'CL':>3}"
    )
    print(header)
    print(f"  {'─' * 76}")

    best_pnl = max(r["summary"]["total_pnl_pct"] for r in results.values())

    for name, r in results.items():
        s = r["summary"]
        a = r["advanced"]
        marker = " ★" if s["total_pnl_pct"] == best_pnl and s["total"] > 0 else "  "
        pf_str = f"{a['profit_factor']:.2f}" if a['profit_factor'] != float('inf') else "  ∞"
        print(
            f"{marker}{name:<20s} "
            f"{s['total']:>6} {s['win_rate']:>5.1f}% "
            f"{s['total_pnl_pct']:>+7.2f}% {pf_str:>6} "
            f"{a['sharpe_approx']:>+6.2f} "
            f"{a['max_drawdown_pct']:>+6.2f}% "
            f"{a['expectancy']:>+7.3f}% "
            f"{a['consecutive_wins']:>3} {a['consecutive_losses']:>3}"
        )

    # ── Per-Day Config Comparison ───────────────────────────────────
    print(f"\n{'═' * 80}")
    print(f"  PER-DAY P&L BY CONFIG")
    print(f"{'═' * 80}")

    config_names = list(results.keys())
    header_configs = "  " + f"{'Date':<18s} " + " ".join(f"{n:>12s}" for n in config_names)
    print(header_configs)
    print(f"  {'─' * (18 + 13 * len(config_names))}")

    for day_idx in range(num_days):
        day_str = trading_days[day_idx].strftime("%Y-%m-%d (%a)")
        row = f"  {day_str:<18s} "
        for name in config_names:
            daily = results[name]["daily"]
            if day_idx < len(daily):
                pnl_pct = daily[day_idx]["total_pnl_pct"]
                num_t = daily[day_idx]["num_trades"]
                marker = "+" if pnl_pct > 0 else ("-" if pnl_pct < 0 else " ")
                row += f" {marker}{pnl_pct:>+6.2f}%({num_t:>2})"
            else:
                row += f" {'—':>12s}"
        print(row)

    print()

    # ── Filter Impact Analysis (vs baseline) ────────────────────────
    baseline = results.get("baseline")
    if baseline:
        print(f"{'═' * 80}")
        print(f"  FILTER IMPACT vs BASELINE ({num_days} days)")
        print(f"{'═' * 80}")

        bl_s = baseline["summary"]
        bl_a = baseline["advanced"]
        for name, r in results.items():
            if name == "baseline":
                continue
            s = r["summary"]
            a = r["advanced"]
            trade_diff = s["total"] - bl_s["total"]
            pnl_diff = s["total_pnl_pct"] - bl_s["total_pnl_pct"]
            wr_diff = s["win_rate"] - bl_s["win_rate"]
            pf_diff = a["profit_factor"] - bl_a["profit_factor"]

            impact = "BETTER" if pnl_diff > 0 and wr_diff >= 0 else (
                "MIXED" if pnl_diff > 0 or wr_diff > 0 else "WORSE"
            )

            print(
                f"  {name:20s} vs baseline: "
                f"trades {trade_diff:+d}, "
                f"P&L {pnl_diff:+.2f}%, "
                f"WR {wr_diff:+.1f}pp, "
                f"PF {pf_diff:+.2f} "
                f"→ {impact}"
            )

        print()

    # ── Best Config Recommendation ──────────────────────────────────
    print(f"{'═' * 80}")
    print(f"  RECOMMENDATION")
    print(f"{'═' * 80}")

    # Score each config (weighted: profit factor 40%, win rate 30%, P&L 20%, drawdown 10%)
    scores = {}
    for name, r in results.items():
        s = r["summary"]
        a = r["advanced"]
        if s["total"] == 0:
            scores[name] = -999
            continue
        pf = min(a["profit_factor"], 5)  # Cap PF at 5 to avoid infinity
        score = (
            pf * 40 +
            s["win_rate"] * 0.3 +
            min(max(s["total_pnl_pct"], -50), 50) * 0.2 +
            max(a["max_drawdown_pct"], -10) * 1  # Less negative = better
        )
        scores[name] = score

    best_name = max(scores, key=scores.get)
    best_r = results[best_name]

    print(f"  ★ Best config: [{best_name}]")
    print(f"    → {best_r['summary']['total']} trades, "
          f"WR: {best_r['summary']['win_rate']:.0f}%, "
          f"P&L: {best_r['summary']['total_pnl_pct']:+.2f}%, "
          f"PF: {best_r['advanced']['profit_factor']:.2f}")
    print()

    # If verbose, print the full report for the best config
    if verbose:
        print_multiday_report(
            best_r["trades"], best_r["daily"], best_r["equity"],
            config_name=best_name, data_10m=data_10m, verbose=True,
        )


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-day backtest for the Rashemator strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/backtest.py                          # All days, default config
  python scripts/backtest.py --compare                # A/B test all configs
  python scripts/backtest.py --tickers AAPL NVDA TSLA # Specific tickers
  python scripts/backtest.py --top 20 -v              # Top 20, verbose
  python scripts/backtest.py --compare --top 50       # Full comparison
        """,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Specific tickers to backtest (default: full universe)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Use top N tickers from universe (e.g., --top 20)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed trade log",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="7d",
        help="Data period: 7d (1-min data) or up to 60d (5-min data). Default: 7d",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all predefined configs and show multi-day A/B comparison",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=list(CONFIGS.keys()),
        default=None,
        help="Run specific config (default: baseline)",
    )

    args = parser.parse_args()

    # Determine tickers
    tickers = args.tickers
    if tickers is None and args.top:
        all_tickers = get_universe()
        tickers = all_tickers[:args.top]

    # Load data once (shared across all configs)
    print(f"\n{'═' * 72}")
    print(f"  RASHEMATOR MULTI-DAY BACKTESTER")
    print(f"{'═' * 72}")
    data_10m = load_data(tickers, period=args.period)

    if not data_10m:
        print("  No data loaded. Check tickers and network.\n")
        return

    trading_days = find_trading_days(data_10m)
    print(f"  Found {len(trading_days)} trading days in data")
    for d in trading_days:
        print(f"    • {d.strftime('%Y-%m-%d (%A)')}")
    print()

    if args.compare:
        compare_configs_multiday(data_10m, verbose=args.verbose)
    else:
        config = CONFIGS.get(args.config) if args.config else None
        config_name = args.config or "default"
        all_trades, daily_results, equity_curve = run_multiday_backtest(
            data_10m, config=config, verbose=args.verbose
        )
        print_multiday_report(
            all_trades, daily_results, equity_curve,
            config_name=config_name, data_10m=data_10m, verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
