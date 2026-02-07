"""Trading agents - pluggable strategy modules.

Each agent implements BaseAgent and encapsulates:
- A specific trading strategy (indicators + rules)
- Entry/exit logic
- Risk parameters

Current agents:
- rashemator: EMA cloud flip strategy with 34/50 trend filter

Future agents:
- price_action: King bar, engulfing patterns, etc.
- momentum: RSI/MACD/VWAP-based entries
- vix: Volatility-aware trading
- news: Event-driven trading (CPI, FOMC, earnings)
- opening_bell: First hour scalping
- power_hour: Last 30 min momentum
- futures: Overnight futures trading
"""

from .base import BaseAgent, AgentConfig, AgentSignal

# Agent registry — maps agent name to its strategy class
AGENT_REGISTRY = {
    "rashemator": "rachatrades.agents.rashemator.EMACloudStrategy",
    "orb": "rachatrades.agents.opening_range.ORBStrategy",
}

__all__ = ["BaseAgent", "AgentConfig", "AgentSignal", "AGENT_REGISTRY"]
