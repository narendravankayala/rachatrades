"""Execution layer — broker abstraction for order execution.

Supports:
- Paper trading (Alpaca sandbox)
- Live trading (future: Alpaca live, IBKR, Schwab)

Usage:
    from rachatrades.core.execution import AlpacaBroker

    broker = AlpacaBroker()  # reads ALPACA_API_KEY, ALPACA_SECRET_KEY from env
    broker.submit_order("AAPL", "buy", qty=1)
"""

from .base import BaseBroker, OrderResult, OrderSide
from .alpaca_broker import AlpacaBroker

__all__ = ["BaseBroker", "AlpacaBroker", "OrderResult", "OrderSide"]
