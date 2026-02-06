"""Base agent interface for all trading strategies.

Every trading agent must implement this interface. This enables:
- Pluggable strategies (swap agents without changing infrastructure)
- Agent portfolios (run multiple agents with different risk profiles)
- A/B testing (compare agents on same universe)
- Subscription tiers (users pick which agents to follow)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import pandas as pd

from rachatrades.core.data import MTFData

logger = logging.getLogger(__name__)


class AgentSignal(Enum):
    """Universal signal types across all agents."""
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"
    HOLD = "HOLD"
    HOLD_SHORT = "HOLD_SHORT"
    NO_POSITION = "NO_POSITION"


@dataclass
class AgentConfig:
    """Base configuration shared by all agents.
    
    Subclass this per agent to add strategy-specific params.
    """
    name: str = "base"
    description: str = ""
    version: str = "1.0.0"
    
    # Risk parameters
    max_positions: int = 10
    max_position_pct: float = 10.0  # Max % of portfolio per position
    stop_loss_pct: Optional[float] = None  # Stop loss percentage
    take_profit_pct: Optional[float] = None  # Take profit percentage
    
    # Market hours
    trade_premarket: bool = False
    trade_afterhours: bool = False
    
    # Timeframes this agent operates on
    primary_timeframe: str = "10min"
    secondary_timeframe: Optional[str] = None


@dataclass
class AgentResult:
    """Result from an agent's evaluation of a single ticker."""
    ticker: str
    signal: AgentSignal
    price: float
    timestamp: pd.Timestamp
    agent_name: str = ""
    confidence: float = 0.0  # 0-1 confidence score
    reason: str = ""
    metadata: dict = field(default_factory=dict)  # Agent-specific data


class BaseAgent(ABC):
    """Abstract base class for all trading agents.
    
    To create a new agent:
    1. Subclass BaseAgent
    2. Implement evaluate() 
    3. Register in rachatrades/agents/__init__.py
    
    Example:
        class MyAgent(BaseAgent):
            def evaluate(self, ticker, data, **kwargs):
                # Your strategy logic here
                return AgentResult(...)
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.name = self.config.name

    @abstractmethod
    def evaluate(
        self,
        ticker: str,
        data: MTFData,
        has_open_position: bool = False,
        has_short_position: bool = False,
    ) -> AgentResult:
        """Evaluate a single ticker and return a trading signal.
        
        Args:
            ticker: Stock symbol
            data: Multi-timeframe market data
            has_open_position: Whether we hold a long position
            has_short_position: Whether we hold a short position
            
        Returns:
            AgentResult with signal, price, reason, etc.
        """
        ...

    def scan(
        self,
        universe_data: Dict[str, MTFData],
        long_positions: Set[str],
        short_positions: Optional[Set[str]] = None,
    ) -> List[AgentResult]:
        """Scan entire universe and return results.
        
        Default implementation calls evaluate() on each ticker.
        Override for batch-optimized strategies.
        """
        short_positions = short_positions or set()
        results = []
        
        for ticker, data in universe_data.items():
            try:
                result = self.evaluate(
                    ticker=ticker,
                    data=data,
                    has_open_position=(ticker in long_positions),
                    has_short_position=(ticker in short_positions),
                )
                result.agent_name = self.name
                results.append(result)
            except Exception as e:
                logger.error(f"[{self.name}] Error evaluating {ticker}: {e}")
        
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
