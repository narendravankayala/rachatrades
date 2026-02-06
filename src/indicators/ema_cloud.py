"""
Rashemator EMA Cloud System - Multi-Timeframe Trend Following.

Three EMA clouds for trend identification and entry timing:
- 10-min: 5/12 (trend), 8/9 (midpoint), 34/50 (major S/R)
- 1-min:  50/120 (trend), 80/90 (midpoint), 340/500 (major S/R)

The 1-min EMAs are 10x the 10-min EMAs (50 = 5*10, 120 = 12*10, etc.)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

# Silence pandas FutureWarning about fillna downcasting
pd.set_option('future.no_silent_downcasting', True)
import numpy as np


class Zone(Enum):
    """Market zone based on price position relative to clouds."""
    LONG = "LONG"      # Above both clouds - buy dips
    SHORT = "SHORT"    # Below both clouds - sell rips
    FLAT = "FLAT"      # Between clouds - no trade


class PullbackType(Enum):
    """Type of pullback detected (for longs - price dipping to support)."""
    NONE = "NONE"
    SHALLOW = "SHALLOW"  # Pullback to trend cloud (50/120 or 5/12)
    MIDPOINT = "MIDPOINT"  # Pullback to midpoint (80/90 or 8/9)
    DEEP = "DEEP"  # Pullback to major S/R cloud (340/500 or 34/50)


class RallyType(Enum):
    """Type of rally detected (for shorts - price rising to resistance)."""
    NONE = "NONE"
    SHALLOW = "SHALLOW"  # Rally to trend cloud (50/120 or 5/12)
    MIDPOINT = "MIDPOINT"  # Rally to midpoint (80/90 or 8/9)
    DEEP = "DEEP"  # Rally to major S/R cloud (340/500 or 34/50)


@dataclass
class CloudState:
    """State of a single EMA cloud."""
    ema_fast: float
    ema_slow: float
    bullish: bool
    cloud_top: float
    cloud_bottom: float
    
    @property
    def bearish(self) -> bool:
        return not self.bullish


@dataclass
class RashematorSignal:
    """Complete Rashemator signal from multi-timeframe analysis."""
    # Zone
    zone: Zone
    
    # 10-min cloud states
    cloud_5_12: Optional[CloudState] = None
    cloud_8_9: Optional[CloudState] = None
    cloud_34_50: Optional[CloudState] = None
    
    # 1-min cloud states  
    cloud_50_120: Optional[CloudState] = None
    cloud_80_90: Optional[CloudState] = None
    cloud_340_500: Optional[CloudState] = None
    
    # Price info
    price: float = 0.0
    
    # Pullback detection (for longs)
    pullback_type: PullbackType = PullbackType.NONE
    pullback_10m: bool = False  # Pullback detected on 10-min
    pullback_1m: bool = False   # Pullback detected on 1-min
    
    # Rally detection (for shorts - price rising into resistance)
    rally_type: RallyType = RallyType.NONE
    rally_10m: bool = False  # Rally detected on 10-min
    rally_1m: bool = False   # Rally detected on 1-min
    
    # Reclaim detection (bullish when price reclaims support)
    reclaim_detected: bool = False
    
    # Rejection detection (bearish when price rejects resistance)
    rejection_detected: bool = False
    
    # Cloud flip detection (EMA crossover on current bar)
    cloud_5_12_cross_up: bool = False    # 5/12 just flipped bullish
    cloud_5_12_cross_down: bool = False  # 5/12 just flipped bearish
    
    # Alignment
    clouds_aligned_10m: bool = False  # All 10-min clouds bullish
    clouds_aligned_1m: bool = False   # All 1-min clouds bullish
    mtf_aligned: bool = False         # Both timeframes aligned


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        series: Price series (typically Close prices)
        period: EMA period

    Returns:
        EMA values as pandas Series
    """
    return series.ewm(span=period, adjust=False).mean()


def _add_cloud_columns(
    df: pd.DataFrame,
    fast_period: int,
    slow_period: int,
    prefix: str,
) -> pd.DataFrame:
    """Add EMA cloud columns with a prefix."""
    result = df.copy()
    
    # Calculate EMAs
    ema_fast = calculate_ema(df["Close"], fast_period)
    ema_slow = calculate_ema(df["Close"], slow_period)
    
    result[f"{prefix}_ema_fast"] = ema_fast
    result[f"{prefix}_ema_slow"] = ema_slow
    
    # Cloud direction
    result[f"{prefix}_bullish"] = ema_fast > ema_slow
    
    # Cloud boundaries
    result[f"{prefix}_top"] = result[[f"{prefix}_ema_fast", f"{prefix}_ema_slow"]].max(axis=1)
    result[f"{prefix}_bottom"] = result[[f"{prefix}_ema_fast", f"{prefix}_ema_slow"]].min(axis=1)
    
    # Price position
    result[f"{prefix}_price_above"] = result["Close"] > result[f"{prefix}_top"]
    result[f"{prefix}_price_below"] = result["Close"] < result[f"{prefix}_bottom"]
    result[f"{prefix}_price_in"] = (
        (result["Close"] >= result[f"{prefix}_bottom"]) & 
        (result["Close"] <= result[f"{prefix}_top"])
    )
    
    # Crossovers
    result[f"{prefix}_cross_up"] = (
        result[f"{prefix}_bullish"] & 
        ~result[f"{prefix}_bullish"].shift(1).fillna(False).astype(bool)
    )
    result[f"{prefix}_cross_down"] = (
        ~result[f"{prefix}_bullish"] & 
        result[f"{prefix}_bullish"].shift(1).fillna(True).astype(bool)
    )
    
    # Reclaim detection: was below/in cloud, now above
    prev_below_or_in = (
        result[f"{prefix}_price_below"].shift(1).fillna(False).astype(bool) |
        result[f"{prefix}_price_in"].shift(1).fillna(False).astype(bool)
    )
    result[f"{prefix}_reclaim"] = prev_below_or_in & result[f"{prefix}_price_above"]
    
    return result


def calculate_rashemator_clouds_10min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all Rashemator clouds for 10-minute timeframe.
    
    Clouds: 5/12 (trend), 8/9 (midpoint), 34/50 (major S/R)
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with cloud columns added
    """
    result = df.copy()
    
    # Trend cloud: 5/12
    result = _add_cloud_columns(result, 5, 12, "c5_12")
    
    # Midpoint cloud: 8/9
    result = _add_cloud_columns(result, 8, 9, "c8_9")
    
    # Major S/R cloud: 34/50
    result = _add_cloud_columns(result, 34, 50, "c34_50")
    
    return result


def calculate_rashemator_clouds_1min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all Rashemator clouds for 1-minute timeframe.
    
    Clouds: 50/120 (trend), 80/90 (midpoint), 340/500 (major S/R)
    These are 10x the 10-minute EMAs.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with cloud columns added
    """
    result = df.copy()
    
    # Trend cloud: 50/120 (= 5/12 * 10)
    result = _add_cloud_columns(result, 50, 120, "c50_120")
    
    # Midpoint cloud: 80/90 (= 8/9 * 10)
    result = _add_cloud_columns(result, 80, 90, "c80_90")
    
    # Major S/R cloud: 340/500 (= 34/50 * 10)
    result = _add_cloud_columns(result, 340, 500, "c340_500")
    
    return result


def _get_cloud_state(df: pd.DataFrame, prefix: str) -> Optional[CloudState]:
    """Extract cloud state from DataFrame."""
    if df.empty:
        return None
    
    latest = df.iloc[-1]
    
    ema_fast = latest.get(f"{prefix}_ema_fast")
    ema_slow = latest.get(f"{prefix}_ema_slow")
    
    if pd.isna(ema_fast) or pd.isna(ema_slow):
        return None
    
    return CloudState(
        ema_fast=float(ema_fast),
        ema_slow=float(ema_slow),
        bullish=bool(latest.get(f"{prefix}_bullish", False)),
        cloud_top=float(latest.get(f"{prefix}_top", 0)),
        cloud_bottom=float(latest.get(f"{prefix}_bottom", 0)),
    )


def _detect_zone(
    price: float,
    trend_cloud: Optional[CloudState],
    major_cloud: Optional[CloudState],
) -> Zone:
    """Determine trading zone based on price position relative to clouds."""
    if trend_cloud is None or major_cloud is None:
        return Zone.FLAT
    
    above_trend = price > trend_cloud.cloud_top
    above_major = price > major_cloud.cloud_top
    below_trend = price < trend_cloud.cloud_bottom
    below_major = price < major_cloud.cloud_bottom
    
    if above_trend and above_major:
        return Zone.LONG
    elif below_trend and below_major:
        return Zone.SHORT
    else:
        return Zone.FLAT


def _detect_pullback(
    df: pd.DataFrame,
    trend_prefix: str,
    mid_prefix: str,
    major_prefix: str,
) -> PullbackType:
    """Detect pullback type based on which cloud price is touching."""
    if df.empty:
        return PullbackType.NONE
    
    latest = df.iloc[-1]
    
    # Check if price is in or touching each cloud
    in_major = latest.get(f"{major_prefix}_price_in", False)
    in_mid = latest.get(f"{mid_prefix}_price_in", False)
    in_trend = latest.get(f"{trend_prefix}_price_in", False)
    
    # Also check if price just touched from above (shallow pullback)
    above_major = latest.get(f"{major_prefix}_price_above", True)
    
    if in_major or (not above_major and latest.get(f"{major_prefix}_price_above", False) == False):
        return PullbackType.DEEP
    elif in_mid:
        return PullbackType.MIDPOINT
    elif in_trend:
        return PullbackType.SHALLOW
    
    return PullbackType.NONE


def _detect_rally(
    df: pd.DataFrame,
    trend_prefix: str,
    mid_prefix: str,
    major_prefix: str,
) -> RallyType:
    """Detect rally type for shorts - price rising into resistance clouds."""
    if df.empty:
        return RallyType.NONE
    
    latest = df.iloc[-1]
    
    # Check if price is in or touching each cloud (from below = resistance)
    in_major = latest.get(f"{major_prefix}_price_in", False)
    in_mid = latest.get(f"{mid_prefix}_price_in", False)
    in_trend = latest.get(f"{trend_prefix}_price_in", False)
    
    # Check if price is below (for shorts, we want price touching cloud from below)
    below_major = latest.get(f"{major_prefix}_price_below", True)
    
    # Deep rally = hitting major cloud from below
    if in_major or (not below_major and latest.get(f"{major_prefix}_price_below", False) == False):
        return RallyType.DEEP
    elif in_mid:
        return RallyType.MIDPOINT
    elif in_trend:
        return RallyType.SHALLOW
    
    return RallyType.NONE


def _detect_rejection(df: pd.DataFrame, trend_prefix: str) -> bool:
    """Detect rejection from resistance (bearish - for short entries)."""
    if df.empty or len(df) < 2:
        return False
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Was in or above cloud, now below
    was_in_or_above = prev.get(f"{trend_prefix}_price_in", False) or prev.get(f"{trend_prefix}_price_above", False)
    now_below = latest.get(f"{trend_prefix}_price_below", False)
    
    return was_in_or_above and now_below


def get_rashemator_signal_10min(df: pd.DataFrame) -> RashematorSignal:
    """
    Get Rashemator signal from 10-minute data.
    
    Args:
        df: DataFrame with 10-min OHLCV data (clouds will be calculated)
        
    Returns:
        RashematorSignal with 10-min analysis
    """
    if df.empty:
        return RashematorSignal(zone=Zone.FLAT)
    
    # Calculate clouds
    df_clouds = calculate_rashemator_clouds_10min(df)
    
    latest = df_clouds.iloc[-1]
    price = float(latest["Close"])
    
    # Get cloud states
    c5_12 = _get_cloud_state(df_clouds, "c5_12")
    c8_9 = _get_cloud_state(df_clouds, "c8_9")
    c34_50 = _get_cloud_state(df_clouds, "c34_50")
    
    # Determine zone
    zone = _detect_zone(price, c5_12, c34_50)
    
    # Detect pullback (for longs)
    pullback = _detect_pullback(df_clouds, "c5_12", "c8_9", "c34_50")
    
    # Detect rally (for shorts)
    rally = _detect_rally(df_clouds, "c5_12", "c8_9", "c34_50")
    
    # Check alignment
    clouds_aligned = all([
        c5_12 and c5_12.bullish,
        c8_9 and c8_9.bullish,
        c34_50 and c34_50.bullish,
    ])
    
    # Check reclaim (bullish)
    reclaim = bool(latest.get("c5_12_reclaim", False))
    
    # Check rejection (bearish)
    rejection = _detect_rejection(df_clouds, "c5_12")
    
    # Cloud flip detection (5/12 crossover on this bar)
    cross_up = bool(latest.get("c5_12_cross_up", False))
    cross_down = bool(latest.get("c5_12_cross_down", False))
    
    return RashematorSignal(
        zone=zone,
        cloud_5_12=c5_12,
        cloud_8_9=c8_9,
        cloud_34_50=c34_50,
        price=price,
        pullback_type=pullback,
        pullback_10m=(pullback != PullbackType.NONE),
        rally_type=rally,
        rally_10m=(rally != RallyType.NONE),
        reclaim_detected=reclaim,
        rejection_detected=rejection,
        cloud_5_12_cross_up=cross_up,
        cloud_5_12_cross_down=cross_down,
        clouds_aligned_10m=clouds_aligned,
    )


def get_rashemator_signal_1min(df: pd.DataFrame) -> RashematorSignal:
    """
    Get Rashemator signal from 1-minute data.
    
    Args:
        df: DataFrame with 1-min OHLCV data (clouds will be calculated)
        
    Returns:
        RashematorSignal with 1-min analysis
    """
    if df.empty:
        return RashematorSignal(zone=Zone.FLAT)
    
    # Calculate clouds
    df_clouds = calculate_rashemator_clouds_1min(df)
    
    latest = df_clouds.iloc[-1]
    price = float(latest["Close"])
    
    # Get cloud states
    c50_120 = _get_cloud_state(df_clouds, "c50_120")
    c80_90 = _get_cloud_state(df_clouds, "c80_90")
    c340_500 = _get_cloud_state(df_clouds, "c340_500")
    
    # Determine zone
    zone = _detect_zone(price, c50_120, c340_500)
    
    # Detect pullback (for longs)
    pullback = _detect_pullback(df_clouds, "c50_120", "c80_90", "c340_500")
    
    # Detect rally (for shorts)
    rally = _detect_rally(df_clouds, "c50_120", "c80_90", "c340_500")
    
    # Check alignment
    clouds_aligned = all([
        c50_120 and c50_120.bullish,
        c80_90 and c80_90.bullish,
        c340_500 and c340_500.bullish,
    ])
    
    # Check reclaim (bullish)
    reclaim = bool(latest.get("c50_120_reclaim", False))
    
    # Check rejection (bearish)
    rejection = _detect_rejection(df_clouds, "c50_120")
    
    return RashematorSignal(
        zone=zone,
        cloud_50_120=c50_120,
        cloud_80_90=c80_90,
        cloud_340_500=c340_500,
        price=price,
        pullback_type=pullback,
        pullback_1m=(pullback != PullbackType.NONE),
        rally_type=rally,
        rally_1m=(rally != RallyType.NONE),
        reclaim_detected=reclaim,
        rejection_detected=rejection,
        clouds_aligned_1m=clouds_aligned,
    )


def get_rashemator_signal_mtf(
    df_10min: pd.DataFrame,
    df_1min: pd.DataFrame,
) -> RashematorSignal:
    """
    Get combined multi-timeframe Rashemator signal.
    
    Uses 10-min for trend/zone and 1-min for entry timing.
    
    Args:
        df_10min: DataFrame with 10-min OHLCV data
        df_1min: DataFrame with 1-min OHLCV data
        
    Returns:
        RashematorSignal with combined MTF analysis
    """
    # Get individual signals
    sig_10m = get_rashemator_signal_10min(df_10min)
    sig_1m = get_rashemator_signal_1min(df_1min)
    
    # Use 10-min zone as primary (higher timeframe)
    zone = sig_10m.zone
    
    # Check MTF alignment
    mtf_aligned = (
        sig_10m.clouds_aligned_10m and 
        sig_1m.clouds_aligned_1m and
        sig_10m.zone == sig_1m.zone
    )
    
    # Combine the signals
    return RashematorSignal(
        zone=zone,
        # 10-min clouds
        cloud_5_12=sig_10m.cloud_5_12,
        cloud_8_9=sig_10m.cloud_8_9,
        cloud_34_50=sig_10m.cloud_34_50,
        # 1-min clouds
        cloud_50_120=sig_1m.cloud_50_120,
        cloud_80_90=sig_1m.cloud_80_90,
        cloud_340_500=sig_1m.cloud_340_500,
        # Price (use 1-min for precision)
        price=sig_1m.price if sig_1m.price > 0 else sig_10m.price,
        # Pullback (for longs - use 1-min for entry)
        pullback_type=sig_1m.pullback_type,
        pullback_10m=sig_10m.pullback_10m,
        pullback_1m=sig_1m.pullback_1m,
        # Rally (for shorts - use 1-min for entry)
        rally_type=sig_1m.rally_type,
        rally_10m=sig_10m.rally_10m,
        rally_1m=sig_1m.rally_1m,
        # Reclaim (bullish - use 1-min for entry trigger)
        reclaim_detected=sig_1m.reclaim_detected,
        # Rejection (bearish - use 1-min for entry trigger)
        rejection_detected=sig_1m.rejection_detected,
        # Alignment
        clouds_aligned_10m=sig_10m.clouds_aligned_10m,
        clouds_aligned_1m=sig_1m.clouds_aligned_1m,
        mtf_aligned=mtf_aligned,
    )


# =============================================================================
# Legacy functions for backward compatibility
# =============================================================================

def calculate_ema_cloud(
    df: pd.DataFrame,
    fast_period: int = 5,
    slow_period: int = 12,
) -> pd.DataFrame:
    """
    Calculate EMA Cloud indicator (legacy compatibility).
    
    For new code, use calculate_rashemator_clouds_10min() or 
    calculate_rashemator_clouds_1min() instead.
    """
    result = df.copy()

    # Calculate EMAs
    result["ema_fast"] = calculate_ema(df["Close"], fast_period)
    result["ema_slow"] = calculate_ema(df["Close"], slow_period)

    # Cloud direction
    result["cloud_bullish"] = result["ema_fast"] > result["ema_slow"]
    result["cloud_bearish"] = result["ema_fast"] < result["ema_slow"]

    # Price position relative to cloud
    cloud_top = result[["ema_fast", "ema_slow"]].max(axis=1)
    cloud_bottom = result[["ema_fast", "ema_slow"]].min(axis=1)

    result["price_above_cloud"] = result["Close"] > cloud_top
    result["price_below_cloud"] = result["Close"] < cloud_bottom
    result["price_in_cloud"] = (result["Close"] >= cloud_bottom) & (
        result["Close"] <= cloud_top
    )

    # Crossovers (signal changes)
    result["cloud_crossover_up"] = (result["cloud_bullish"]) & (
        ~result["cloud_bullish"].shift(1).fillna(False)
    )
    result["cloud_crossover_down"] = (result["cloud_bearish"]) & (
        ~result["cloud_bearish"].shift(1).fillna(False)
    )

    return result


def get_ema_cloud_signal(df: pd.DataFrame) -> dict:
    """
    Get the current EMA Cloud signal (legacy compatibility).
    
    For new code, use get_rashemator_signal_10min() or 
    get_rashemator_signal_1min() instead.
    """
    if df.empty:
        return {
            "bullish": False,
            "price_above": False,
            "crossover_up": False,
            "crossover_down": False,
            "ema_fast": None,
            "ema_slow": None,
        }

    latest = df.iloc[-1]

    return {
        "bullish": bool(latest.get("cloud_bullish", False)),
        "price_above": bool(latest.get("price_above_cloud", False)),
        "crossover_up": bool(latest.get("cloud_crossover_up", False)),
        "crossover_down": bool(latest.get("cloud_crossover_down", False)),
        "ema_fast": float(latest["ema_fast"]) if pd.notna(latest.get("ema_fast")) else None,
        "ema_slow": float(latest["ema_slow"]) if pd.notna(latest.get("ema_slow")) else None,
    }


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    print("Testing Rashemator EMA Cloud System...")
    print("=" * 60)
    
    # Fetch both timeframes (use 15m since yfinance doesn't support 10m)
    ticker = yf.Ticker("AAPL")
    df_10m = ticker.history(period="5d", interval="15m")
    df_1m = ticker.history(period="7d", interval="1m")
    
    print(f"\nFetched {len(df_10m)} 10-min bars and {len(df_1m)} 1-min bars")
    
    # Get MTF signal
    signal = get_rashemator_signal_mtf(df_10m, df_1m)
    
    print(f"\n{'='*60}")
    print(f"AAPL Rashemator Signal")
    print(f"{'='*60}")
    print(f"  Price: ${signal.price:.2f}")
    print(f"  Zone: {signal.zone.value}")
    print(f"  MTF Aligned: {signal.mtf_aligned}")
    print(f"\n  10-MIN CLOUDS:")
    if signal.cloud_5_12:
        print(f"    5/12:  {'BULL' if signal.cloud_5_12.bullish else 'BEAR'} "
              f"(Fast: {signal.cloud_5_12.ema_fast:.2f}, Slow: {signal.cloud_5_12.ema_slow:.2f})")
    if signal.cloud_8_9:
        print(f"    8/9:   {'BULL' if signal.cloud_8_9.bullish else 'BEAR'}")
    if signal.cloud_34_50:
        print(f"    34/50: {'BULL' if signal.cloud_34_50.bullish else 'BEAR'}")
    print(f"    Aligned: {signal.clouds_aligned_10m}")
    print(f"    Pullback: {signal.pullback_10m}")
    print(f"\n  1-MIN CLOUDS:")
    if signal.cloud_50_120:
        print(f"    50/120:  {'BULL' if signal.cloud_50_120.bullish else 'BEAR'} "
              f"(Fast: {signal.cloud_50_120.ema_fast:.2f}, Slow: {signal.cloud_50_120.ema_slow:.2f})")
    if signal.cloud_80_90:
        print(f"    80/90:   {'BULL' if signal.cloud_80_90.bullish else 'BEAR'}")
    if signal.cloud_340_500:
        print(f"    340/500: {'BULL' if signal.cloud_340_500.bullish else 'BEAR'}")
    print(f"    Aligned: {signal.clouds_aligned_1m}")
    print(f"    Pullback: {signal.pullback_1m} ({signal.pullback_type.value})")
    print(f"    Reclaim: {signal.reclaim_detected}")
