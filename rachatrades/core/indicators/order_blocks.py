"""
Order Block Detector — institutional supply/demand zones.

Ported from LuxAlgo's TradingView indicator (Pine Script → Python).
Source: https://www.tradingview.com/v/KvGhxGxY/

Order blocks are price zones where significant volume occurred at swing
highs/lows, indicating institutional activity. They act as future
support (bullish OB) and resistance (bearish OB) until "mitigated"
(price breaks through them).

Detection logic:
1. Find volume pivots (bars where volume peaks vs neighbors)
2. During swing lows → bullish OB (zone: low → hl2 of that bar)
3. During swing highs → bearish OB (zone: hl2 → high of that bar)
4. OBs are removed when price mitigates them (breaks through)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OBType(Enum):
    """Order block type."""
    BULLISH = "BULLISH"   # Support zone (buy here)
    BEARISH = "BEARISH"   # Resistance zone (sell here)


@dataclass
class OrderBlock:
    """A single order block zone."""
    ob_type: OBType
    top: float          # Upper boundary of the zone
    bottom: float       # Lower boundary of the zone
    avg: float          # Midpoint (average line)
    bar_index: int      # Bar index where OB formed
    timestamp: Optional[pd.Timestamp] = None
    volume: float = 0.0  # Volume at formation
    mitigated: bool = False

    @property
    def size(self) -> float:
        """Zone size in price."""
        return self.top - self.bottom

    @property
    def size_pct(self) -> float:
        """Zone size as % of avg price."""
        return (self.size / self.avg * 100) if self.avg > 0 else 0

    def contains_price(self, price: float) -> bool:
        """Check if price is inside this order block zone."""
        return self.bottom <= price <= self.top

    def price_near(self, price: float, tolerance_pct: float = 0.5) -> bool:
        """Check if price is near (within tolerance %) of this OB."""
        margin = self.avg * tolerance_pct / 100
        return (self.bottom - margin) <= price <= (self.top + margin)


@dataclass
class OrderBlockSignal:
    """Result of order block analysis for a single point in time."""
    bullish_obs: List[OrderBlock] = field(default_factory=list)
    bearish_obs: List[OrderBlock] = field(default_factory=list)

    # Convenience flags for the strategy
    near_bullish_ob: bool = False     # Price is at/near a bullish (support) OB
    near_bearish_ob: bool = False     # Price is at/near a bearish (resistance) OB
    inside_bullish_ob: bool = False   # Price is inside a bullish OB
    inside_bearish_ob: bool = False   # Price is inside a bearish OB

    # Mitigation events on this bar
    bull_ob_mitigated: bool = False   # A bullish OB was just broken (bearish)
    bear_ob_mitigated: bool = False   # A bearish OB was just broken (bullish)

    # Nearest OB info
    nearest_support: Optional[OrderBlock] = None   # Closest bullish OB below
    nearest_resistance: Optional[OrderBlock] = None  # Closest bearish OB above


def _find_pivot_highs(series: pd.Series, length: int) -> pd.Series:
    """
    Find pivot highs: values higher than `length` bars on both sides.
    Returns boolean Series, True at pivot bars.
    """
    result = pd.Series(False, index=series.index)
    values = series.values

    for i in range(length, len(values) - length):
        is_pivot = True
        for j in range(1, length + 1):
            if values[i] <= values[i - j] or values[i] <= values[i + j]:
                is_pivot = False
                break
        if is_pivot:
            result.iloc[i] = True

    return result


def detect_order_blocks(
    df: pd.DataFrame,
    volume_pivot_length: int = 5,
    mitigation_method: str = "wick",
    max_obs: int = 10,
) -> OrderBlockSignal:
    """
    Detect order blocks in OHLCV data.

    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns
        volume_pivot_length: Lookback for volume pivot detection (default: 5)
        mitigation_method: "wick" (default) or "close" — how to check mitigation
        max_obs: Maximum number of active OBs to track per side

    Returns:
        OrderBlockSignal with current OB state and flags
    """
    if df is None or df.empty or len(df) < volume_pivot_length * 3:
        return OrderBlockSignal()

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    volume = df["Volume"].values
    hl2 = (high + low) / 2

    n = len(df)
    length = volume_pivot_length

    # Track swing direction: 0 = swing high area, 1 = swing low area
    # Uses highest high / lowest low over lookback
    os_state = np.zeros(n, dtype=int)
    for i in range(length, n):
        upper = np.max(high[max(0, i - length):i])
        lower = np.min(low[max(0, i - length):i])

        if i >= length and high[i - length] > upper:
            os_state[i] = 0  # swing high
        elif i >= length and low[i - length] < lower:
            os_state[i] = 1  # swing low
        else:
            os_state[i] = os_state[i - 1] if i > 0 else 0

    # Find volume pivot highs
    vol_series = pd.Series(volume, index=df.index)
    pivot_mask = _find_pivot_highs(vol_series, length)

    # Collect order blocks
    bullish_obs: List[OrderBlock] = []
    bearish_obs: List[OrderBlock] = []

    for i in range(length, n - length):
        if not pivot_mask.iloc[i]:
            continue

        bar_idx = i - length  # The bar the OB references (offset by length)
        if bar_idx < 0:
            continue

        ts = df.index[bar_idx] if hasattr(df.index, '__getitem__') else None

        if os_state[i] == 1:
            # Swing low → Bullish OB (support)
            ob = OrderBlock(
                ob_type=OBType.BULLISH,
                top=float(hl2[bar_idx]),
                bottom=float(low[bar_idx]),
                avg=float((hl2[bar_idx] + low[bar_idx]) / 2),
                bar_index=bar_idx,
                timestamp=ts,
                volume=float(volume[i]),
            )
            bullish_obs.append(ob)

        elif os_state[i] == 0:
            # Swing high → Bearish OB (resistance)
            ob = OrderBlock(
                ob_type=OBType.BEARISH,
                top=float(high[bar_idx]),
                bottom=float(hl2[bar_idx]),
                avg=float((high[bar_idx] + hl2[bar_idx]) / 2),
                bar_index=bar_idx,
                timestamp=ts,
                volume=float(volume[i]),
            )
            bearish_obs.append(ob)

    # Mitigate order blocks (check current price against OBs)
    bull_mitigated = False
    bear_mitigated = False

    if mitigation_method == "close":
        target_bull = float(np.min(close[max(0, n - length):n]))
        target_bear = float(np.max(close[max(0, n - length):n]))
    else:  # wick
        target_bull = float(np.min(low[max(0, n - length):n]))
        target_bear = float(np.max(high[max(0, n - length):n]))

    # Remove mitigated bullish OBs (price broke below OB bottom)
    active_bull = []
    for ob in bullish_obs:
        if target_bull < ob.bottom:
            ob.mitigated = True
            bull_mitigated = True
        else:
            active_bull.append(ob)
    bullish_obs = active_bull[-max_obs:] if len(active_bull) > max_obs else active_bull

    # Remove mitigated bearish OBs (price broke above OB top)
    active_bear = []
    for ob in bearish_obs:
        if target_bear > ob.top:
            ob.mitigated = True
            bear_mitigated = True
        else:
            active_bear.append(ob)
    bearish_obs = active_bear[-max_obs:] if len(active_bear) > max_obs else active_bear

    # Analyze current price vs active OBs
    current_price = float(close[-1])

    near_bull = False
    near_bear = False
    inside_bull = False
    inside_bear = False
    nearest_support = None
    nearest_resistance = None

    # Check bullish OBs (support below)
    for ob in reversed(bullish_obs):  # Most recent first
        if ob.contains_price(current_price):
            inside_bull = True
            near_bull = True
            if nearest_support is None:
                nearest_support = ob
        elif ob.price_near(current_price, tolerance_pct=0.3):
            near_bull = True
            if nearest_support is None and ob.avg <= current_price:
                nearest_support = ob
        elif ob.top <= current_price and nearest_support is None:
            nearest_support = ob

    # Check bearish OBs (resistance above)
    for ob in reversed(bearish_obs):
        if ob.contains_price(current_price):
            inside_bear = True
            near_bear = True
            if nearest_resistance is None:
                nearest_resistance = ob
        elif ob.price_near(current_price, tolerance_pct=0.3):
            near_bear = True
            if nearest_resistance is None and ob.avg >= current_price:
                nearest_resistance = ob
        elif ob.bottom >= current_price and nearest_resistance is None:
            nearest_resistance = ob

    return OrderBlockSignal(
        bullish_obs=bullish_obs,
        bearish_obs=bearish_obs,
        near_bullish_ob=near_bull,
        near_bearish_ob=near_bear,
        inside_bullish_ob=inside_bull,
        inside_bearish_ob=inside_bear,
        bull_ob_mitigated=bull_mitigated,
        bear_ob_mitigated=bear_mitigated,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
    )


if __name__ == "__main__":
    """Quick test with AAPL data."""
    import logging
    logging.basicConfig(level=logging.INFO)

    from rachatrades.core.data import DataProvider

    provider = DataProvider()
    mtf = provider.get_mtf_ohlcv("AAPL")

    if mtf.df_10min is not None:
        signal = detect_order_blocks(mtf.df_10min)

        print(f"Active Bullish OBs (support): {len(signal.bullish_obs)}")
        for ob in signal.bullish_obs[-3:]:
            print(f"  ${ob.bottom:.2f} - ${ob.top:.2f} (avg ${ob.avg:.2f})")

        print(f"\nActive Bearish OBs (resistance): {len(signal.bearish_obs)}")
        for ob in signal.bearish_obs[-3:]:
            print(f"  ${ob.bottom:.2f} - ${ob.top:.2f} (avg ${ob.avg:.2f})")

        price = float(mtf.df_10min["Close"].iloc[-1])
        print(f"\nCurrent price: ${price:.2f}")
        print(f"Near bullish OB: {signal.near_bullish_ob}")
        print(f"Near bearish OB: {signal.near_bearish_ob}")
        if signal.nearest_support:
            print(f"Nearest support: ${signal.nearest_support.bottom:.2f} - ${signal.nearest_support.top:.2f}")
        if signal.nearest_resistance:
            print(f"Nearest resistance: ${signal.nearest_resistance.bottom:.2f} - ${signal.nearest_resistance.top:.2f}")
