"""Main scanner - orchestrates the entire scanning pipeline."""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pytz

from src.data import DataProvider
from src.scanner import get_universe
from src.signals import PositionTracker
from src.strategies import EMACloudStrategy, Signal, StrategyConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Market timezone
ET = pytz.timezone("America/New_York")


def is_market_hours() -> bool:
    """Check if US market is currently open (roughly)."""
    now = datetime.now(ET)

    # Weekend check
    if now.weekday() >= 5:
        return False

    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now <= market_close


def run_scan(
    db_path: str = "data/positions.db",
    dry_run: bool = False,
) -> dict:
    """
    Run a single scan of the universe.

    Args:
        db_path: Path to the positions database
        dry_run: If True, don't actually open/close positions

    Returns:
        Dictionary with scan results
    """
    logger.info("=" * 60)
    logger.info("Starting market scan...")
    logger.info(f"Time: {datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Market hours: {is_market_hours()}")
    logger.info("=" * 60)

    # Initialize components
    provider = DataProvider(cache_minutes=5)
    tracker = PositionTracker(db_path)
    strategy = EMACloudStrategy()

    # Get universe and open positions
    universe = get_universe()
    open_positions = tracker.get_open_tickers()

    logger.info(f"Scanning {len(universe)} stocks...")
    logger.info(f"Currently holding {len(open_positions)} positions: {open_positions}")

    # Fetch data for all tickers
    logger.info("Fetching 15-minute data...")
    data = provider.get_batch_ohlcv(universe, interval="15m", period="5d")
    logger.info(f"Got data for {len(data)} tickers")

    # Run strategy on all tickers
    results = strategy.scan_universe(data, open_positions)

    # Process signals
    buys = []
    sells = []
    holds = []

    for result in results:
        if result.signal == Signal.BUY:
            buys.append(result)
            if not dry_run:
                tracker.open_position(
                    ticker=result.ticker,
                    price=result.price,
                    timestamp=datetime.now(ET),
                    reason=result.reason,
                )

        elif result.signal == Signal.SELL:
            sells.append(result)
            if not dry_run:
                tracker.close_position(
                    ticker=result.ticker,
                    price=result.price,
                    timestamp=datetime.now(ET),
                    reason=result.reason,
                )

        elif result.signal == Signal.HOLD:
            holds.append(result)

    # Update daily summary
    if not dry_run:
        tracker.update_daily_summary(datetime.now(ET))

    # Get stats
    stats = tracker.get_stats()

    # Log summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SCAN COMPLETE")
    logger.info("=" * 60)
    logger.info(f"New BUY signals: {len(buys)}")
    for b in buys:
        logger.info(f"  📈 {b.ticker} @ ${b.price:.2f} - {b.reason}")

    logger.info(f"SELL signals: {len(sells)}")
    for s in sells:
        logger.info(f"  📉 {s.ticker} @ ${s.price:.2f} - {s.reason}")

    logger.info(f"HOLD positions: {len(holds)}")
    logger.info("")
    logger.info("Portfolio Stats:")
    logger.info(f"  Open positions: {stats['open_positions']}")
    logger.info(f"  Total trades: {stats['total_trades']}")
    logger.info(f"  Win rate: {stats['win_rate']:.1f}%")
    logger.info(f"  Total P&L: ${stats['total_pnl']:.2f}")
    logger.info("=" * 60)

    return {
        "timestamp": datetime.now(ET).isoformat(),
        "buys": [{"ticker": r.ticker, "price": r.price, "reason": r.reason} for r in buys],
        "sells": [{"ticker": r.ticker, "price": r.price, "reason": r.reason} for r in sells],
        "holds": [{"ticker": r.ticker, "price": r.price} for r in holds],
        "stats": stats,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="rachatrades market scanner")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run scan without executing trades",
    )
    parser.add_argument(
        "--db",
        default="data/positions.db",
        help="Path to positions database",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even outside market hours",
    )

    args = parser.parse_args()

    # Check market hours unless forced
    if not args.force and not is_market_hours():
        logger.info("Market is closed. Use --force to run anyway.")
        return

    # Run the scan
    results = run_scan(db_path=args.db, dry_run=args.dry_run)

    # Save results to JSON for website generation
    import json
    output_path = Path("data/latest_scan.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
