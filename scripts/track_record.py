"""
Generate track record data for the website.

Reads positions.db and outputs JSON that the website template uses
to render a daily P&L chart and agent performance table.

Run by GitHub Actions after each scan, or manually:
    python scripts/track_record.py
    python scripts/track_record.py --db data/positions.db --out data/track_record.json
"""

import argparse
import json
import logging
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def generate_track_record(
    db_path: str = "data/positions.db",
    output_path: str = "data/track_record.json",
) -> dict:
    """
    Generate track record JSON from positions database.

    Returns dict with:
    - daily_pnl: list of {date, trades, winners, losers, pnl, cumulative_pnl}
    - agents: list of {name, trades, win_rate, total_pnl, avg_pnl, best_trade, worst_trade}
    - summary: overall stats
    """
    db = Path(db_path)
    if not db.exists():
        logger.warning(f"Database not found: {db_path}")
        return _empty_record()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # ── Daily P&L ────────────────────────────────────────────────────
    rows = conn.execute("""
        SELECT
            DATE(exit_time) as trade_date,
            ticker,
            position_type,
            entry_price,
            exit_price,
            pnl,
            pnl_percent,
            entry_reason,
            exit_reason
        FROM positions
        WHERE status = 'CLOSED' AND exit_time IS NOT NULL
        ORDER BY exit_time ASC
    """).fetchall()

    daily_data = defaultdict(lambda: {
        "trades": 0, "winners": 0, "losers": 0,
        "pnl": 0.0, "pnl_pct": 0.0,
        "long_trades": 0, "short_trades": 0,
        "long_pnl": 0.0, "short_pnl": 0.0,
    })

    all_trades = []
    for row in rows:
        d = dict(row)
        date = d["trade_date"]
        pnl = d["pnl"] or 0
        pnl_pct = d["pnl_percent"] or 0
        pos_type = d["position_type"] or "LONG"

        daily_data[date]["trades"] += 1
        daily_data[date]["pnl"] += pnl
        daily_data[date]["pnl_pct"] += pnl_pct
        if pnl > 0:
            daily_data[date]["winners"] += 1
        else:
            daily_data[date]["losers"] += 1

        if pos_type == "LONG":
            daily_data[date]["long_trades"] += 1
            daily_data[date]["long_pnl"] += pnl
        else:
            daily_data[date]["short_trades"] += 1
            daily_data[date]["short_pnl"] += pnl

        all_trades.append(d)

    # Build daily P&L with cumulative
    cumulative = 0.0
    daily_pnl = []
    for date in sorted(daily_data.keys()):
        day = daily_data[date]
        cumulative += day["pnl"]
        daily_pnl.append({
            "date": date,
            "trades": day["trades"],
            "winners": day["winners"],
            "losers": day["losers"],
            "win_rate": round(day["winners"] / day["trades"] * 100, 1) if day["trades"] > 0 else 0,
            "pnl": round(day["pnl"], 2),
            "pnl_pct": round(day["pnl_pct"], 2),
            "cumulative_pnl": round(cumulative, 2),
            "long_trades": day["long_trades"],
            "short_trades": day["short_trades"],
            "long_pnl": round(day["long_pnl"], 2),
            "short_pnl": round(day["short_pnl"], 2),
        })

    # ── Overall Summary ──────────────────────────────────────────────
    total_trades = len(all_trades)
    winners = sum(1 for t in all_trades if (t["pnl"] or 0) > 0)
    total_pnl = sum(t["pnl"] or 0 for t in all_trades)
    total_pnl_pct = sum(t["pnl_percent"] or 0 for t in all_trades)

    # Open positions count
    open_count = conn.execute(
        "SELECT COUNT(*) FROM positions WHERE status = 'OPEN'"
    ).fetchone()[0]

    # Best/worst
    best = max(all_trades, key=lambda t: t["pnl"] or 0) if all_trades else None
    worst = min(all_trades, key=lambda t: t["pnl"] or 0) if all_trades else None

    # Max drawdown from cumulative curve
    peak = 0.0
    max_dd = 0.0
    for day in daily_pnl:
        peak = max(peak, day["cumulative_pnl"])
        dd = peak - day["cumulative_pnl"]
        max_dd = max(max_dd, dd)

    summary = {
        "total_trades": total_trades,
        "winners": winners,
        "losers": total_trades - winners,
        "win_rate": round(winners / total_trades * 100, 1) if total_trades > 0 else 0,
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "avg_pnl": round(total_pnl / total_trades, 2) if total_trades > 0 else 0,
        "avg_pnl_pct": round(total_pnl_pct / total_trades, 2) if total_trades > 0 else 0,
        "open_positions": open_count,
        "trading_days": len(daily_pnl),
        "max_drawdown": round(max_dd, 2),
        "best_trade": {
            "ticker": best["ticker"], "pnl": round(best["pnl"], 2)
        } if best else None,
        "worst_trade": {
            "ticker": worst["ticker"], "pnl": round(worst["pnl"], 2)
        } if worst else None,
        "generated_at": datetime.now().isoformat(),
    }

    # ── Agent Breakdown (from entry_reason) ──────────────────────────
    # Currently all trades are from Rashemator, but this future-proofs
    # for multi-agent by parsing agent name from entry_reason or adding
    # an agent column to positions table
    agents = [{
        "name": "Rashemator",
        "description": "5/12 Cloud Flip + 34/50 Trend Filter",
        "trades": total_trades,
        "win_rate": summary["win_rate"],
        "total_pnl": summary["total_pnl"],
        "avg_pnl": summary["avg_pnl"],
        "status": "live",
    }]

    conn.close()

    record = {
        "daily_pnl": daily_pnl,
        "agents": agents,
        "summary": summary,
    }

    # Write JSON
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(f"Track record written to {out} ({total_trades} trades, {len(daily_pnl)} days)")

    return record


def _empty_record() -> dict:
    return {
        "daily_pnl": [],
        "agents": [],
        "summary": {
            "total_trades": 0, "winners": 0, "losers": 0,
            "win_rate": 0, "total_pnl": 0, "total_pnl_pct": 0,
            "avg_pnl": 0, "avg_pnl_pct": 0, "open_positions": 0,
            "trading_days": 0, "max_drawdown": 0,
            "best_trade": None, "worst_trade": None,
            "generated_at": datetime.now().isoformat(),
        },
    }


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate track record JSON from positions DB")
    parser.add_argument("--db", default="data/positions.db", help="Path to positions database")
    parser.add_argument("--out", default="data/track_record.json", help="Output JSON path")
    args = parser.parse_args()

    record = generate_track_record(args.db, args.out)
    s = record["summary"]
    print(f"\nTrack Record Generated:")
    print(f"  Trading days : {s['trading_days']}")
    print(f"  Total trades : {s['total_trades']}")
    print(f"  Win rate     : {s['win_rate']}%")
    print(f"  Total P&L    : ${s['total_pnl']:+.2f}")
    print(f"  Max drawdown : ${s['max_drawdown']:.2f}")
    if s["best_trade"]:
        print(f"  Best trade   : {s['best_trade']['ticker']} ${s['best_trade']['pnl']:+.2f}")
    if s["worst_trade"]:
        print(f"  Worst trade  : {s['worst_trade']['ticker']} ${s['worst_trade']['pnl']:+.2f}")


if __name__ == "__main__":
    main()
