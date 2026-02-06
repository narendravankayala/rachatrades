"""Rashemator agent - EMA Cloud Flip Strategy.

The first and flagship agent. Uses 5/12 EMA cloud flips on 10-min candles
with 34/50 major cloud as trend filter.

Rules:
- BUY: 5/12 flips bullish + 34/50 bullish
- SELL: 5/12 flips bearish
- SHORT: 5/12 flips bearish + 34/50 bearish
- COVER: 5/12 flips bullish
"""

from .strategy import EMACloudStrategy, StrategyConfig, StrategyResult, Signal

__all__ = ["EMACloudStrategy", "StrategyConfig", "StrategyResult", "Signal"]
