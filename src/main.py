"""
Main scanner - orchestrates the Rashemator MTF scanning pipeline.

Uses multi-timeframe analysis:
- 10-min charts for trend/zone detection
- 1-min charts for entry timing and oscillator confirmation
"""

import argparse
import json
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
    Run a single scan of the universe using Rashemator MTF strategy.

    Args:
        db_path: Path to the positions database
        dry_run: If True, don't actually open/close positions

    Returns:
        Dictionary with scan results
    """
    logger.info("=" * 60)
    logger.info("Starting Rashemator MTF scan...")
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

    # Fetch MTF data (10-min + 1-min) for all tickers
    logger.info("Fetching multi-timeframe data (10-min + 1-min)...")
    mtf_data = provider.get_batch_mtf_ohlcv(universe)
    logger.info(f"Got MTF data for {len(mtf_data)} tickers")

    # Run strategy on all tickers
    results = strategy.scan_universe_mtf(mtf_data, open_positions)

    # Process signals
    buys = []
    sells = []
    shorts = []
    covers = []
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
        
        elif result.signal == Signal.SHORT:
            shorts.append(result)
            # Note: PositionTracker would need short position support
            # For now, just track them in output
        
        elif result.signal == Signal.COVER:
            covers.append(result)
            # Note: PositionTracker would need short position support

        elif result.signal == Signal.HOLD or result.signal == Signal.HOLD_SHORT:
            holds.append(result)

    # Count zones
    zone_counts = {"LONG": 0, "SHORT": 0, "FLAT": 0}
    for result in results:
        zone_counts[result.zone.value] = zone_counts.get(result.zone.value, 0) + 1

    # Update daily summary
    if not dry_run:
        tracker.update_daily_summary(datetime.now(ET))

    # Get stats
    stats = tracker.get_stats()

    # Log summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("RASHEMATOR SCAN COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Zone Distribution: LONG={zone_counts['LONG']} | FLAT={zone_counts['FLAT']} | SHORT={zone_counts['SHORT']}")
    logger.info("")
    logger.info(f"New BUY signals: {len(buys)}")
    for b in buys:
        logger.info(f"  📈 {b.ticker} @ ${b.price:.2f} - {b.reason}")

    logger.info(f"New SHORT signals: {len(shorts)}")
    for s in shorts:
        logger.info(f"  📉 {s.ticker} @ ${s.price:.2f} - {s.reason}")

    logger.info(f"SELL signals: {len(sells)}")
    for s in sells:
        logger.info(f"  💰 {s.ticker} @ ${s.price:.2f} - {s.reason}")

    logger.info(f"COVER signals: {len(covers)}")
    for c in covers:
        logger.info(f"  🔄 {c.ticker} @ ${c.price:.2f} - {c.reason}")

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
        "buys": [
            {
                "ticker": r.ticker, 
                "price": r.price, 
                "reason": r.reason,
                "zone": r.zone.value,
                "pullback_type": r.pullback_type.value,
                "mfi": r.mfi_value,
                "williams_r": r.williams_r_value,
            } 
            for r in buys
        ],
        "shorts": [
            {
                "ticker": r.ticker, 
                "price": r.price, 
                "reason": r.reason,
                "zone": r.zone.value,
                "rally_type": r.rally_type.value,
                "mfi": r.mfi_value,
                "williams_r": r.williams_r_value,
            } 
            for r in shorts
        ],
        "sells": [
            {
                "ticker": r.ticker, 
                "price": r.price, 
                "reason": r.reason,
            } 
            for r in sells
        ],
        "covers": [
            {
                "ticker": r.ticker, 
                "price": r.price, 
                "reason": r.reason,
            } 
            for r in covers
        ],
        "holds": [
            {
                "ticker": r.ticker, 
                "price": r.price,
                "zone": r.zone.value,
            } 
            for r in holds
        ],
        "stats": stats,
        "zone_counts": zone_counts,
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
    output_path = Path("data/latest_scan.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
