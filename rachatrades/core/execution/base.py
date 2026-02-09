"""Base broker abstraction — interface for all brokers."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderResult:
    """Result from submitting an order."""
    success: bool
    order_id: Optional[str] = None
    ticker: str = ""
    side: str = ""
    qty: int = 0
    filled_price: Optional[float] = None
    filled_qty: Optional[int] = None
    status: str = ""
    error: Optional[str] = None
    timestamp: Optional[datetime] = None
    raw: Optional[dict] = None  # broker-specific raw response

    def __str__(self):
        if self.success:
            price_str = f" @ ${self.filled_price:.2f}" if self.filled_price else ""
            return f"[OK] {self.side.upper()} {self.qty} {self.ticker}{price_str} (id={self.order_id})"
        return f"[FAIL] {self.side.upper()} {self.qty} {self.ticker}: {self.error}"


@dataclass
class BrokerPosition:
    """A position held at the broker."""
    ticker: str
    qty: int
    side: str  # "long" or "short"
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float
    current_price: float


class BaseBroker(ABC):
    """Abstract broker interface.

    All brokers (Alpaca, IBKR, Schwab, etc.) implement this interface
    so the scanner doesn't care which broker is being used.
    """

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if broker connection is active."""
        ...

    @abstractmethod
    def get_account_info(self) -> dict:
        """Get account info (cash, buying power, equity, etc.)."""
        ...

    @abstractmethod
    def submit_order(
        self,
        ticker: str,
        side: OrderSide,
        qty: int = 1,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> OrderResult:
        """Submit an order to the broker.

        Args:
            ticker: Stock symbol
            side: BUY or SELL
            qty: Number of shares
            order_type: "market", "limit", etc.
            time_in_force: "day", "gtc", "ioc", etc.

        Returns:
            OrderResult with success/failure and fill details
        """
        ...

    @abstractmethod
    def get_positions(self) -> List[BrokerPosition]:
        """Get all open positions at the broker."""
        ...

    @abstractmethod
    def get_position(self, ticker: str) -> Optional[BrokerPosition]:
        """Get a specific position."""
        ...

    @abstractmethod
    def close_position(self, ticker: str) -> OrderResult:
        """Close an entire position (long or short)."""
        ...

    def buy(self, ticker: str, qty: int = 1) -> OrderResult:
        """Convenience: submit a market buy order."""
        return self.submit_order(ticker, OrderSide.BUY, qty)

    def sell(self, ticker: str, qty: int = 1) -> OrderResult:
        """Convenience: submit a market sell order."""
        return self.submit_order(ticker, OrderSide.SELL, qty)

    def short(self, ticker: str, qty: int = 1) -> OrderResult:
        """Short sell a stock (sell without owning)."""
        return self.submit_order(ticker, OrderSide.SELL, qty)

    def cover(self, ticker: str, qty: int = 1) -> OrderResult:
        """Cover a short position (buy to close)."""
        return self.submit_order(ticker, OrderSide.BUY, qty)
