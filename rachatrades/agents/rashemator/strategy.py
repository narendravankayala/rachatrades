"""
Rashemator Strategy - 10-Minute EMA Cloud System.

Everything runs on 10-min candles (resampled from 1-min).

Three EMA clouds on 10-min:
- 5/12 (trend), 8/9 (midpoint), 34/50 (major S/R)

Trade Zones:
- LONG_ZONE: Price > 5/12 cloud AND > 34/50 cloud → buy pullbacks
- SHORT_ZONE: Price < 5/12 cloud AND < 34/50 cloud → sell rips
- FLAT_ZONE: Price between clouds → NO TRADE

Entry Confirmation:
- MFI < 20 OR Williams %R < -80 (14-period on 10-min = 140 min lookback)
- Pullback to cloud support detected
- Candle closes back above cloud (reclaim)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import pandas as pd

from rachatrades.core.indicators import (
    Zone,
    PullbackType,
    RallyType,
    RashematorSignal,
    get_rashemator_signal_10min,
    get_rashemator_signal_1min,
    get_rashemator_signal_mtf,
    calculate_mfi,
    calculate_williams_r,
    get_mfi_signal,
    get_williams_r_signal,
    # Legacy compatibility
    calculate_ema_cloud,
    get_ema_cloud_signal,
)
from rachatrades.core.indicators.order_blocks import detect_order_blocks, OrderBlockSignal
from rachatrades.core.data import MTFData

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"  # Open short position
    COVER = "COVER"  # Close short position
    HOLD = "HOLD"
    HOLD_SHORT = "HOLD_SHORT"  # Hold short position
    NO_POSITION = "NO_POSITION"


@dataclass
class StrategyConfig:
    """Configuration for the Rashemator strategy.
    
    Feature flags control which filters are active. Toggle them on/off
    to A/B test combinations in the simulator. The base strategy (cloud
    flips + 34/50 trend filter) always runs.
    """
    
    # ── Feature Flags ────────────────────────────────────────────────
    # Each flag enables/disables an independent filter layer.
    # Default: all OFF — only the core cloud flip logic is active.
    
    use_oscillator_filter: bool = False     # Require MFI/WR confirmation for entries
    use_order_blocks: bool = False          # OB confluence: only enter near support/resistance
    use_cloud_spread_filter: bool = False   # Require min EMA5/12 spread to avoid chop
    use_stop_loss: bool = False             # Hard stop loss on open positions
    # Future flags:
    # use_volume_filter: bool = False       # Require above-average volume
    # use_vwap_filter: bool = False         # VWAP trend confirmation
    # use_rsi_filter: bool = False          # RSI divergence confirmation
    
    # ── Oscillator Settings ──────────────────────────────────────────
    mfi_period: int = 14
    mfi_oversold: float = 20.0
    mfi_overbought: float = 80.0

    williams_r_period: int = 14
    williams_r_oversold: float = -80.0
    williams_r_overbought: float = -20.0
    
    # Require oscillator confirmation for entries (legacy, replaced by use_oscillator_filter)
    require_oscillator: bool = True
    
    # Use OR logic for oscillators (either MFI or WR confirms)
    oscillator_or_logic: bool = True
    
    # ── Order Block Settings ────────────────────────────────────────
    ob_volume_pivot_length: int = 5      # Lookback for volume pivot detection
    ob_mitigation_method: str = "wick"   # "wick" or "close"
    ob_max_active: int = 10              # Max OBs to track per side
    ob_proximity_pct: float = 0.3        # How close price must be to OB (%)
    
    # ── Cloud Spread Settings ────────────────────────────────────────
    min_cloud_spread_pct: float = 0.1        # Min |EMA5 - EMA12| as % of price
    
    # ── Stop Loss Settings ───────────────────────────────────────────
    stop_loss_pct: float = 0.5               # Stop loss as % of entry price
    
    # ── Cooldown Settings (simulation-level) ─────────────────────────
    cooldown_bars: int = 3                   # Bars to wait after exit before re-entry (10-min bars = 30 min)
    max_positions: int = 5                   # Max concurrent positions
    skip_first_minutes: int = 30             # Skip first N minutes of market open
    skip_last_minutes: int = 30              # Skip last N minutes before close
    
    # ── Config metadata ─────────────────────────────────────────────
    name: str = "default"                # Config name for A/B comparison
    version: str = "v1"                  # Strategy version: "v1" (cloud flip) or "v2" (pullback)
    
    def describe(self) -> str:
        """Human-readable summary of active features."""
        base = "pullback_reclaim" if self.version == "v2" else "cloud_flip"
        flags = [f"{base} + 34/50_filter"]
        if self.use_oscillator_filter:
            flags.append("oscillator")
        if self.use_order_blocks:
            flags.append("order_blocks")
        if self.use_cloud_spread_filter:
            flags.append(f"spread>{self.min_cloud_spread_pct}%")
        if self.use_stop_loss:
            flags.append(f"stop{self.stop_loss_pct}%")
        return f"[{self.name}] " + " + ".join(flags)


@dataclass
class StrategyResult:
    """Result from strategy evaluation."""
    ticker: str
    signal: Signal
    price: float
    timestamp: pd.Timestamp

    # Zone (from Rashemator)
    zone: Zone = Zone.FLAT
    
    # Cloud states (10-min)
    cloud_5_12_bullish: bool = False
    cloud_8_9_bullish: bool = False
    cloud_34_50_bullish: bool = False
    clouds_aligned_10m: bool = False
    
    # Cloud states (1-min)
    cloud_50_120_bullish: bool = False
    cloud_80_90_bullish: bool = False
    cloud_340_500_bullish: bool = False
    clouds_aligned_1m: bool = False
    
    # MTF alignment
    mtf_aligned: bool = False
    
    # Pullback detection (for longs)
    pullback_type: PullbackType = PullbackType.NONE
    pullback_10m: bool = False
    pullback_1m: bool = False
    
    # Rally detection (for shorts)
    rally_type: RallyType = RallyType.NONE
    rally_10m: bool = False
    rally_1m: bool = False
    
    # Reclaim signal (bullish)
    reclaim_detected: bool = False
    
    # Rejection signal (bearish)
    rejection_detected: bool = False
    
    # Oscillator values
    mfi_value: Optional[float] = None
    mfi_oversold: bool = False
    mfi_overbought: bool = False

    williams_r_value: Optional[float] = None
    williams_r_oversold: bool = False
    williams_r_overbought: bool = False
    
    # Combined oscillator confirmation
    oscillator_confirms: bool = False

    # Order block state
    near_bullish_ob: bool = False
    near_bearish_ob: bool = False
    inside_bullish_ob: bool = False
    inside_bearish_ob: bool = False
    nearest_support_price: Optional[float] = None
    nearest_resistance_price: Optional[float] = None

    # Which filters were active for this result
    active_filters: str = ""

    # Reasoning
    reason: str = ""
    
    # Legacy compatibility
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    ema_cloud_bullish: bool = False
    price_above_cloud: bool = False


def get_strategy(config: Optional[StrategyConfig] = None):
    """
    Factory: create the right strategy class based on config.version.

    Args:
        config: StrategyConfig with version field ("v1" or "v2")

    Returns:
        Strategy instance (CloudFlipStrategy or PullbackStrategy)
    """
    config = config or StrategyConfig()

    from rachatrades.agents.rashemator.v1_cloud_flip import CloudFlipStrategy
    from rachatrades.agents.rashemator.v2_pullback import PullbackStrategy

    VERSIONS = {
        "v1": CloudFlipStrategy,
        "v2": PullbackStrategy,
    }

    cls = VERSIONS.get(config.version)
    if cls is None:
        raise ValueError(f"Unknown strategy version: {config.version!r}. Available: {list(VERSIONS.keys())}")

    return cls(config)


# Backward compatibility alias
def EMACloudStrategy(config: Optional[StrategyConfig] = None):
    """Legacy alias — returns the strategy for the given config version."""
    return get_strategy(config)
