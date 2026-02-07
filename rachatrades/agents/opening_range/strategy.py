"""
Opening Range Breakout (ORB) Strategy — Agent #2.

The first hour of trading (9:30–10:30 ET) sets the day's range.
Breakout above = long. Breakdown below = short.

This strategy is widely used by professional day traders and is
documented in Toby Crabel's "Day Trading with Short-Term Price Patterns."

How it works:
1. OPENING RANGE: Collect High/Low of first hour (9:30–10:30 ET)
2. After 10:30 ET, monitor for breakout:
   - Price closes above OR high → BUY
   - Price closes below OR low → SHORT
3. Trend filter: 34/50 EMA cloud on 10-min for direction bias
4. Exit: Price re-enters range OR cloud flips against you

The ORB is built on 10-min candles (same data as Rashemator).
Opening range = first 6 bars of the day (6 × 10 min = 60 min).
"""

import logging
from dataclasses import dataclass, field
from datetime import time
from enum import Enum
from typing import Dict, List, Optional, Set

import pandas as pd
import pytz

from rachatrades.core.data import MTFData
from rachatrades.core.indicators import (
    Zone,
    calculate_rashemator_clouds_10min,
    get_rashemator_signal_10min,
)

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")

# Market hours
MARKET_OPEN = time(9, 30)
OR_END = time(10, 30)  # Opening range ends after 1 hour
MARKET_CLOSE = time(16, 0)


class ORBSignal(Enum):
    """ORB signal types."""
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"
    HOLD = "HOLD"
    HOLD_SHORT = "HOLD_SHORT"
    NO_POSITION = "NO_POSITION"
    BUILDING_RANGE = "BUILDING_RANGE"  # Still in first hour


@dataclass
class ORBConfig:
    """Configuration for the ORB strategy.

    Feature flags match the Rashemator pattern for A/B testing.
    """
    # ── Feature Flags ────────────────────────────────────────────────
    use_trend_filter: bool = True     # Require 34/50 cloud alignment
    use_volume_confirm: bool = False  # Require above-avg volume on breakout bar
    use_retest_entry: bool = False    # Wait for breakout + retest of range edge

    # ── Range Settings ───────────────────────────────────────────────
    or_minutes: int = 60             # Opening range duration (minutes)
    breakout_buffer_pct: float = 0.0 # Extra % above/below range for breakout (0 = exact)

    # ── Exit Settings ────────────────────────────────────────────────
    exit_on_range_reentry: bool = True   # Exit if price goes back into the range
    exit_on_cloud_flip: bool = True      # Exit if 5/12 cloud flips against
    stop_loss_pct: Optional[float] = None  # Fixed stop loss %

    # ── Config metadata ─────────────────────────────────────────────
    name: str = "orb_default"

    def describe(self) -> str:
        """Human-readable summary of active features."""
        flags = ["ORB"]
        if self.use_trend_filter:
            flags.append("34/50_filter")
        if self.use_volume_confirm:
            flags.append("volume")
        if self.use_retest_entry:
            flags.append("retest")
        return f"[{self.name}] " + " + ".join(flags)


@dataclass
class ORBResult:
    """Result from ORB strategy evaluation."""
    ticker: str
    signal: ORBSignal
    price: float
    timestamp: pd.Timestamp

    # Opening range
    or_high: Optional[float] = None
    or_low: Optional[float] = None
    or_range_pct: Optional[float] = None  # Range as % of price
    or_complete: bool = False

    # Breakout state
    broke_above: bool = False
    broke_below: bool = False
    bars_since_breakout: int = 0

    # Trend filter
    cloud_34_50_bullish: bool = False

    # Config
    active_filters: str = ""

    # Reasoning
    reason: str = ""


def _to_et(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert a timestamp to Eastern Time."""
    if ts.tzinfo is None:
        return ts.tz_localize(ET)
    return ts.tz_convert(ET)


def _get_or_bars(df_10m: pd.DataFrame, target_date) -> pd.DataFrame:
    """Get just the opening range bars (first hour) for a given date."""
    bars = []
    for i in range(len(df_10m)):
        ts = _to_et(df_10m.index[i])
        if ts.date() != target_date:
            continue
        t = ts.time()
        # Bars from 9:30 up to (but not including) 10:30
        if MARKET_OPEN <= t < OR_END:
            bars.append(i)
    return bars


class ORBStrategy:
    """
    Opening Range Breakout on 10-min candles.

    Uses the same 10-min data as Rashemator. The first 6 bars
    of each day define the opening range.
    """

    def __init__(self, config: Optional[ORBConfig] = None):
        self.config = config or ORBConfig()

    def evaluate_mtf(
        self,
        ticker: str,
        mtf_data: MTFData,
        has_open_position: bool = False,
        has_short_position: bool = False,
    ) -> ORBResult:
        """
        Evaluate ORB strategy on 10-min data.

        Args:
            ticker: Stock ticker
            mtf_data: MTFData with 10-min DataFrame
            has_open_position: Whether we hold a long
            has_short_position: Whether we hold a short
        """
        df_10m = mtf_data.df_10min

        if df_10m is None or df_10m.empty or len(df_10m) < 10:
            return ORBResult(
                ticker=ticker,
                signal=ORBSignal.NO_POSITION,
                price=0.0,
                timestamp=pd.Timestamp.now(),
                reason="Insufficient data",
            )

        latest = df_10m.iloc[-1]
        current_price = float(latest["Close"])
        current_time = df_10m.index[-1]
        current_et = _to_et(current_time)
        today = current_et.date()

        # ── Step 1: Build opening range ──────────────────────────────
        or_bar_indices = _get_or_bars(df_10m, today)

        if not or_bar_indices:
            return ORBResult(
                ticker=ticker,
                signal=ORBSignal.NO_POSITION,
                price=current_price,
                timestamp=current_time,
                reason="No bars in opening range window for today",
            )

        or_bars = df_10m.iloc[or_bar_indices]
        or_high = float(or_bars["High"].max())
        or_low = float(or_bars["Low"].min())
        or_range_pct = ((or_high - or_low) / or_low * 100) if or_low > 0 else 0

        # Check if we're still building the range
        or_complete = current_et.time() >= OR_END
        if not or_complete:
            return ORBResult(
                ticker=ticker,
                signal=ORBSignal.BUILDING_RANGE,
                price=current_price,
                timestamp=current_time,
                or_high=or_high,
                or_low=or_low,
                or_range_pct=or_range_pct,
                or_complete=False,
                reason=f"Building opening range: ${or_low:.2f}–${or_high:.2f} ({or_range_pct:.1f}%)",
            )

        # ── Step 2: Get trend filter ─────────────────────────────────
        rash_signal = get_rashemator_signal_10min(df_10m)
        cloud_34_50_bull = (
            rash_signal.cloud_34_50.bullish if rash_signal.cloud_34_50 else False
        )
        cloud_34_50_bear = (
            rash_signal.cloud_34_50.bearish if rash_signal.cloud_34_50 else False
        )

        # Breakout thresholds with optional buffer
        buffer = or_high * self.config.breakout_buffer_pct / 100
        breakout_level = or_high + buffer
        breakdown_level = or_low - buffer

        # Check if current bar breaks the range
        broke_above = current_price > breakout_level
        broke_below = current_price < breakdown_level

        # Build base result
        result = ORBResult(
            ticker=ticker,
            signal=ORBSignal.NO_POSITION,
            price=current_price,
            timestamp=current_time,
            or_high=or_high,
            or_low=or_low,
            or_range_pct=or_range_pct,
            or_complete=True,
            broke_above=broke_above,
            broke_below=broke_below,
            cloud_34_50_bullish=cloud_34_50_bull,
            active_filters=self.config.describe(),
        )

        # ── Step 3: Signal logic ─────────────────────────────────────

        if has_open_position:
            # Currently long — check exit conditions
            if self.config.exit_on_range_reentry and current_price < or_high:
                result.signal = ORBSignal.SELL
                result.reason = f"SELL: Price ${current_price:.2f} fell back below OR high ${or_high:.2f}"
            elif self.config.exit_on_cloud_flip and rash_signal.cloud_5_12_cross_down:
                result.signal = ORBSignal.SELL
                result.reason = "SELL: 5/12 cloud flipped bearish"
            else:
                result.signal = ORBSignal.HOLD
                result.reason = f"HOLD: Above OR high ${or_high:.2f}"

        elif has_short_position:
            # Currently short — check exit conditions
            if self.config.exit_on_range_reentry and current_price > or_low:
                result.signal = ORBSignal.COVER
                result.reason = f"COVER: Price ${current_price:.2f} rose back above OR low ${or_low:.2f}"
            elif self.config.exit_on_cloud_flip and rash_signal.cloud_5_12_cross_up:
                result.signal = ORBSignal.COVER
                result.reason = "COVER: 5/12 cloud flipped bullish"
            else:
                result.signal = ORBSignal.HOLD_SHORT
                result.reason = f"HOLD SHORT: Below OR low ${or_low:.2f}"

        else:
            # No position — look for breakout/breakdown
            if broke_above:
                if self.config.use_trend_filter and not cloud_34_50_bull:
                    result.signal = ORBSignal.NO_POSITION
                    result.reason = f"Breakout above ${or_high:.2f} but 34/50 bearish — no long"
                else:
                    result.signal = ORBSignal.BUY
                    result.reason = (
                        f"BUY: Breakout above OR high ${or_high:.2f} "
                        f"(range: {or_range_pct:.1f}%"
                        f"{', 34/50 bullish' if cloud_34_50_bull else ''})"
                    )

            elif broke_below:
                if self.config.use_trend_filter and not cloud_34_50_bear:
                    result.signal = ORBSignal.NO_POSITION
                    result.reason = f"Breakdown below ${or_low:.2f} but 34/50 bullish — no short"
                else:
                    result.signal = ORBSignal.SHORT
                    result.reason = (
                        f"SHORT: Breakdown below OR low ${or_low:.2f} "
                        f"(range: {or_range_pct:.1f}%"
                        f"{', 34/50 bearish' if cloud_34_50_bear else ''})"
                    )

            else:
                result.signal = ORBSignal.NO_POSITION
                result.reason = (
                    f"Inside range ${or_low:.2f}–${or_high:.2f}, "
                    f"waiting for breakout"
                )

        return result

    def scan_universe_mtf(
        self,
        mtf_data: Dict[str, MTFData],
        open_positions: Set[str],
        short_positions: Optional[Set[str]] = None,
    ) -> List[ORBResult]:
        """Scan multiple tickers."""
        if short_positions is None:
            short_positions = set()

        results = []
        for ticker, data in mtf_data.items():
            try:
                result = self.evaluate_mtf(
                    ticker, data,
                    has_open_position=(ticker in open_positions),
                    has_short_position=(ticker in short_positions),
                )
                results.append(result)
                if result.signal in (ORBSignal.BUY, ORBSignal.SELL, ORBSignal.SHORT, ORBSignal.COVER):
                    logger.info(f"[ORB] {ticker}: {result.signal.value} — {result.reason}")
            except Exception as e:
                logger.error(f"[ORB] Error evaluating {ticker}: {e}")

        return results


if __name__ == "__main__":
    import yfinance as yf
    from rachatrades.core.data import DataProvider

    logging.basicConfig(level=logging.INFO)

    strategy = ORBStrategy()
    provider = DataProvider()

    print("Fetching AAPL data (1-min → resample to 10-min)...")
    mtf_data = provider.get_mtf_ohlcv("AAPL")

    if mtf_data.df_10min is not None:
        print(f"Got {len(mtf_data.df_10min)} true 10-min bars")

    result = strategy.evaluate_mtf("AAPL", mtf_data)

    print(f"\n{'=' * 60}")
    print(f"ORB Strategy Result for AAPL")
    print(f"{'=' * 60}")
    print(f"  Signal: {result.signal.value}")
    print(f"  Price: ${result.price:.2f}")
    if result.or_high and result.or_low:
        print(f"  Opening Range: ${result.or_low:.2f} – ${result.or_high:.2f} ({result.or_range_pct:.1f}%)")
    print(f"  OR Complete: {result.or_complete}")
    print(f"  Broke Above: {result.broke_above}")
    print(f"  Broke Below: {result.broke_below}")
    print(f"  34/50 Bullish: {result.cloud_34_50_bullish}")
    print(f"  Reason: {result.reason}")
