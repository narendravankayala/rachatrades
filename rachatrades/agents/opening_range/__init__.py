"""Opening Range Breakout (ORB) agent.

Trades breakouts/breakdowns from the first hour's high and low.
One of the simplest and most proven intraday strategies.

Rules:
- Track the first hour's high/low (9:30–10:30 ET)
- BUY when price breaks above the opening range high + 34/50 bullish
- SHORT when price breaks below the opening range low + 34/50 bearish
- SELL / COVER when price re-enters the range (failed breakout)
"""

from .strategy import ORBStrategy, ORBConfig, ORBResult

__all__ = ["ORBStrategy", "ORBConfig", "ORBResult"]
