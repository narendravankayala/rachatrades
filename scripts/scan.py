"""
Main scanner - Rashemator Cloud Flip Strategy.

Cloud flip system on 10-min candles:
- BUY when 5/12 cloud flips bullish AND 34/50 major cloud is bullish
- SELL when 5/12 cloud flips bearish
- SHORT when 5/12 cloud flips bearish AND 34/50 major cloud is bearish
- COVER when 5/12 cloud flips bullish
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on path when script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import timedelta

import pytz

from rachatrades.core.data import DataProvider
from rachatrades.core.execution import AlpacaBroker
from rachatrades.notifications import EmailNotifier
from rachatrades.scanner import get_universe
from rachatrades.core.signals import PositionTracker
from rachatrades.agents.rashemator import EMACloudStrategy, Signal, StrategyConfig

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


def is_eod_close_time() -> bool:
    """Check if it's time to close all positions (3:50 PM ET or later)."""
    now = datetime.now(ET)
    eod_cutoff = now.replace(hour=15, minute=50, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return eod_cutoff <= now <= market_close


def run_scan(
    db_path: str = "data/positions.db",
    dry_run: bool = False,
) -> dict:
    """
    Run a single scan of the universe using Rashemator cloud flip strategy.

    Args:
        db_path: Path to the positions database
        dry_run: If True, don't actually open/close positions

    Returns:
        Dictionary with scan results
    """
    scan_time = datetime.now(ET)
    
    logger.info("=" * 60)
    logger.info("Starting Rashemator Cloud Flip scan...")
    logger.info(f"Time: {scan_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Market hours: {is_market_hours()}")
    logger.info("=" * 60)

    # Initialize components
    provider = DataProvider(cache_minutes=5)
    tracker = PositionTracker(db_path)
    strategy = EMACloudStrategy(StrategyConfig(
        name="improved_no_stop",
        use_cloud_spread_filter=True,
        min_cloud_spread_pct=0.05,
        use_stop_loss=False,
        cooldown_bars=3,
        max_positions=5,
        skip_first_minutes=30,
        skip_last_minutes=30,
    ))
    notifier = EmailNotifier()
    broker = AlpacaBroker()

    if notifier.is_configured:
        logger.info("Email notifications: ENABLED")
    else:
        logger.info("Email notifications: DISABLED (set SMTP_USER, SMTP_PASSWORD, NOTIFY_EMAILS)")

    if broker.is_configured:
        mode = "PAPER" if broker.paper else "LIVE"
        logger.info(f"Alpaca execution: ENABLED ({mode})")
        if broker.is_connected():
            acct = broker.get_account_info()
            logger.info(f"  Account equity: ${acct.get('equity', 0):,.2f}  |  Buying power: ${acct.get('buying_power', 0):,.2f}")
        else:
            logger.warning("Alpaca broker configured but connection failed")
    else:
        logger.info("Alpaca execution: DISABLED (set ALPACA_API_KEY, ALPACA_SECRET_KEY)")

    # Get universe and open positions
    universe = get_universe()
    open_long_positions = tracker.get_open_tickers()
    open_short_positions = tracker.get_open_short_tickers()

    logger.info(f"Scanning {len(universe)} stocks...")
    logger.info(f"Open LONG positions ({len(open_long_positions)}): {open_long_positions}")
    logger.info(f"Open SHORT positions ({len(open_short_positions)}): {open_short_positions}")

    # ── EOD CLOSE: close LOSING positions, let winners ride overnight ─
    if is_eod_close_time() and (open_long_positions or open_short_positions):
        logger.info("=" * 60)
        logger.info("EOD CLOSE: Closing losing positions — winners ride overnight")
        logger.info("=" * 60)

        eod_sells = []
        eod_covers = []
        eod_held = []

        # Build entry price lookup from open positions
        open_positions = tracker.get_open_positions()
        entry_prices = {p.ticker: p for p in open_positions}

        provider_eod = DataProvider(cache_minutes=0)

        # Check longs: close losers, keep winners
        for ticker in list(open_long_positions):
            try:
                price = provider_eod.get_latest_price(ticker)
                if not price:
                    continue
                pos = entry_prices.get(ticker)
                if not pos:
                    continue
                unrealized_pnl = price - pos.entry_price

                if unrealized_pnl < 0:
                    # Losing — close it
                    if not dry_run:
                        closed = tracker.close_position(
                            ticker=ticker, price=price,
                            timestamp=scan_time, reason=f"EOD close — losing (${unrealized_pnl:+.2f})",
                        )
                        if broker.is_configured:
                            broker.execute_sell(ticker)
                        pnl_info = f" (P&L: ${closed.pnl:+.2f})" if closed and closed.pnl else ""
                        logger.info(f"  EOD SELL {ticker} @ ${price:.2f}{pnl_info} — LOSING")
                        eod_sells.append({"ticker": ticker, "price": price, "pnl": unrealized_pnl})
                else:
                    # Winning — let it ride
                    logger.info(f"  EOD HOLD {ticker} @ ${price:.2f} (unrealized: ${unrealized_pnl:+.2f}) — WINNER rides overnight")
                    eod_held.append({"ticker": ticker, "price": price, "pnl": unrealized_pnl})
            except Exception as e:
                logger.error(f"  Failed to evaluate LONG {ticker}: {e}")

        # Check shorts: close losers, keep winners
        for ticker in list(open_short_positions):
            try:
                price = provider_eod.get_latest_price(ticker)
                if not price:
                    continue
                pos = entry_prices.get(ticker)
                if not pos:
                    continue
                unrealized_pnl = pos.entry_price - price  # SHORT: profit when price drops

                if unrealized_pnl < 0:
                    # Losing — close it
                    if not dry_run:
                        closed = tracker.close_short_position(
                            ticker=ticker, price=price,
                            timestamp=scan_time, reason=f"EOD close — losing (${unrealized_pnl:+.2f})",
                        )
                        if broker.is_configured:
                            broker.execute_cover(ticker)
                        pnl_info = f" (P&L: ${closed.pnl:+.2f})" if closed and closed.pnl else ""
                        logger.info(f"  EOD COVER {ticker} @ ${price:.2f}{pnl_info} — LOSING")
                        eod_covers.append({"ticker": ticker, "price": price, "pnl": unrealized_pnl})
                else:
                    # Winning — let it ride
                    logger.info(f"  EOD HOLD SHORT {ticker} @ ${price:.2f} (unrealized: ${unrealized_pnl:+.2f}) — WINNER rides overnight")
                    eod_held.append({"ticker": ticker, "price": price, "pnl": unrealized_pnl})
            except Exception as e:
                logger.error(f"  Failed to evaluate SHORT {ticker}: {e}")

        logger.info(f"EOD: Closed {len(eod_sells)} losing longs, {len(eod_covers)} losing shorts, {len(eod_held)} winners riding overnight")

        # Skip normal scanning after EOD close
        return {
            "timestamp": scan_time.isoformat(),
            "scan_time": scan_time.strftime("%Y-%m-%d %H:%M:%S ET"),
            "eod_close": True,
            "eod_sells": eod_sells,
            "eod_covers": eod_covers,
            "eod_held": eod_held,
            "buys": [], "shorts": [], "sells": [], "covers": [], "holds": [],
            "stats": tracker.get_stats(),
            "zone_counts": {"LONG": 0, "SHORT": 0, "FLAT": 0},
        }

    # Fetch 1-min data and resample to true 10-min candles
    logger.info("Fetching 1-min data and resampling to 10-min candles...")
    mtf_data = provider.get_batch_mtf_ohlcv(universe)
    logger.info(f"Got 10-min data for {len(mtf_data)} tickers")

    # Run strategy on all tickers
    results = strategy.scan_universe_mtf(
        mtf_data, 
        open_long_positions,
        short_positions=open_short_positions,
    )

    # Process signals
    buys = []
    sells = []
    shorts = []
    covers = []
    holds = []

    # ── Entry guards from config ─────────────────────────────────────
    cfg = strategy.config

    # Time-of-day filter: only allow new entries within trading window
    now_et = datetime.now(ET)
    now_minutes = now_et.hour * 60 + now_et.minute
    market_open_min = 9 * 60 + 30   # 9:30 AM
    market_close_min = 16 * 60      # 4:00 PM
    entry_window_open = market_open_min + cfg.skip_first_minutes
    entry_window_close = market_close_min - cfg.skip_last_minutes
    in_entry_window = entry_window_open <= now_minutes < entry_window_close

    if not in_entry_window:
        logger.info(f"Outside entry window ({entry_window_open//60}:{entry_window_open%60:02d}-{entry_window_close//60}:{entry_window_close%60:02d}) — new entries blocked")

    # Max positions cap
    total_open = len(open_long_positions) + len(open_short_positions)
    at_capacity = total_open >= cfg.max_positions
    if at_capacity:
        logger.info(f"At max positions ({total_open}/{cfg.max_positions}) — new entries blocked")

    # Cooldown: time-based (cooldown_bars * 10 minutes)
    cooldown_duration = timedelta(minutes=cfg.cooldown_bars * 10)

    for result in results:
        if result.signal == Signal.BUY:
            # ── Entry guards ──
            if not in_entry_window:
                logger.info(f"  SKIP BUY {result.ticker} — outside entry window")
                holds.append(result)
                continue
            if at_capacity:
                logger.info(f"  SKIP BUY {result.ticker} — at max positions ({cfg.max_positions})")
                holds.append(result)
                continue
            last_exit = tracker.get_last_exit_time(result.ticker)
            if last_exit and (scan_time - last_exit) < cooldown_duration:
                remaining = cooldown_duration - (scan_time - last_exit)
                logger.info(f"  SKIP BUY {result.ticker} — cooldown ({remaining.seconds // 60}m remaining)")
                holds.append(result)
                continue

            buys.append(result)
            total_open += 1
            at_capacity = total_open >= cfg.max_positions
            logger.info(f">>> BUY {result.ticker} @ ${result.price:.2f} at {scan_time.strftime('%H:%M:%S')} - {result.reason}")
            if not dry_run:
                tracker.open_position(
                    ticker=result.ticker,
                    price=result.price,
                    timestamp=scan_time,
                    reason=result.reason,
                )
                # Execute via broker
                if broker.is_configured:
                    order = broker.execute_buy(result.ticker)
                    logger.info(f"  Alpaca: {order}")
                notifier.send_trade_alert(
                    signal_type="BUY",
                    ticker=result.ticker,
                    price=result.price,
                    reason=result.reason,
                    zone=result.zone.value,
                    timestamp=scan_time,
                )

        elif result.signal == Signal.SELL:
            sells.append(result)
            total_open -= 1
            at_capacity = total_open >= cfg.max_positions
            logger.info(f">>> SELL {result.ticker} @ ${result.price:.2f} at {scan_time.strftime('%H:%M:%S')} - {result.reason}")
            if not dry_run:
                closed = tracker.close_position(
                    ticker=result.ticker,
                    price=result.price,
                    timestamp=scan_time,
                    reason=result.reason,
                )
                # Execute via broker
                if broker.is_configured:
                    order = broker.execute_sell(result.ticker)
                    logger.info(f"  Alpaca: {order}")
                pnl_info = f" (P&L: ${closed.pnl:+.2f})" if closed and closed.pnl else ""
                notifier.send_trade_alert(
                    signal_type="SELL",
                    ticker=result.ticker,
                    price=result.price,
                    reason=result.reason + pnl_info,
                    zone=result.zone.value,
                    timestamp=scan_time,
                )

        elif result.signal == Signal.SHORT:
            # ── Entry guards ──
            if not in_entry_window:
                logger.info(f"  SKIP SHORT {result.ticker} — outside entry window")
                holds.append(result)
                continue
            if at_capacity:
                logger.info(f"  SKIP SHORT {result.ticker} — at max positions ({cfg.max_positions})")
                holds.append(result)
                continue
            last_exit = tracker.get_last_exit_time(result.ticker)
            if last_exit and (scan_time - last_exit) < cooldown_duration:
                remaining = cooldown_duration - (scan_time - last_exit)
                logger.info(f"  SKIP SHORT {result.ticker} — cooldown ({remaining.seconds // 60}m remaining)")
                holds.append(result)
                continue

            shorts.append(result)
            total_open += 1
            at_capacity = total_open >= cfg.max_positions
            logger.info(f">>> SHORT {result.ticker} @ ${result.price:.2f} at {scan_time.strftime('%H:%M:%S')} - {result.reason}")
            if not dry_run:
                tracker.open_short_position(
                    ticker=result.ticker,
                    price=result.price,
                    timestamp=scan_time,
                    reason=result.reason,
                )
                # Execute via broker
                if broker.is_configured:
                    order = broker.execute_short(result.ticker)
                    logger.info(f"  Alpaca: {order}")
                notifier.send_trade_alert(
                    signal_type="SHORT",
                    ticker=result.ticker,
                    price=result.price,
                    reason=result.reason,
                    zone=result.zone.value,
                    timestamp=scan_time,
                )

        elif result.signal == Signal.COVER:
            covers.append(result)
            total_open -= 1
            at_capacity = total_open >= cfg.max_positions
            logger.info(f">>> COVER {result.ticker} @ ${result.price:.2f} at {scan_time.strftime('%H:%M:%S')} - {result.reason}")
            if not dry_run:
                closed = tracker.close_short_position(
                    ticker=result.ticker,
                    price=result.price,
                    timestamp=scan_time,
                    reason=result.reason,
                )
                # Execute via broker
                if broker.is_configured:
                    order = broker.execute_cover(result.ticker)
                    logger.info(f"  Alpaca: {order}")
                pnl_info = f" (P&L: ${closed.pnl:+.2f})" if closed and closed.pnl else ""
                notifier.send_trade_alert(
                    signal_type="COVER",
                    ticker=result.ticker,
                    price=result.price,
                    reason=result.reason + pnl_info,
                    zone=result.zone.value,
                    timestamp=scan_time,
                )

        elif result.signal in (Signal.HOLD, Signal.HOLD_SHORT):
            holds.append(result)

    # Count zones
    zone_counts = {"LONG": 0, "SHORT": 0, "FLAT": 0}
    for result in results:
        zone_counts[result.zone.value] = zone_counts.get(result.zone.value, 0) + 1

    # Update daily summary
    if not dry_run:
        tracker.update_daily_summary(scan_time)

    # Send scan summary email (only if there were signals)
    stats = tracker.get_stats()
    if not dry_run:
        notifier.send_scan_summary(buys, sells, shorts, covers, stats, zone_counts, scan_time)

    # Log summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("RASHEMATOR CLOUD FLIP SCAN COMPLETE")
    logger.info(f"Scan time: {scan_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info("=" * 60)
    logger.info(f"Zone Distribution: LONG={zone_counts['LONG']} | FLAT={zone_counts['FLAT']} | SHORT={zone_counts['SHORT']}")
    logger.info("")
    logger.info(f"New BUY signals: {len(buys)}")
    for b in buys:
        logger.info(f"  BUY {b.ticker} @ ${b.price:.2f} - {b.reason}")

    logger.info(f"New SHORT signals: {len(shorts)}")
    for s in shorts:
        logger.info(f"  SHORT {s.ticker} @ ${s.price:.2f} - {s.reason}")

    logger.info(f"SELL signals: {len(sells)}")
    for s in sells:
        logger.info(f"  SELL {s.ticker} @ ${s.price:.2f} - {s.reason}")

    logger.info(f"COVER signals: {len(covers)}")
    for c in covers:
        logger.info(f"  COVER {c.ticker} @ ${c.price:.2f} - {c.reason}")

    logger.info(f"HOLD positions: {len(holds)}")
    logger.info("")
    logger.info("Portfolio Stats:")
    logger.info(f"  Open positions: {stats['open_positions']}")
    logger.info(f"  Total trades: {stats['total_trades']}")
    logger.info(f"  Win rate: {stats['win_rate']:.1f}%")
    logger.info(f"  Total P&L: ${stats['total_pnl']:.2f}")
    logger.info("=" * 60)

    return {
        "timestamp": scan_time.isoformat(),
        "scan_time": scan_time.strftime("%Y-%m-%d %H:%M:%S ET"),
        "buys": [
            {
                "ticker": r.ticker, 
                "price": r.price, 
                "reason": r.reason,
                "zone": r.zone.value,
                "time": scan_time.strftime("%H:%M:%S"),
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
                "time": scan_time.strftime("%H:%M:%S"),
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
                "time": scan_time.strftime("%H:%M:%S"),
            } 
            for r in sells
        ],
        "covers": [
            {
                "ticker": r.ticker, 
                "price": r.price, 
                "reason": r.reason,
                "time": scan_time.strftime("%H:%M:%S"),
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
