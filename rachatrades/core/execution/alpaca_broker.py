"""Alpaca broker implementation — paper and live trading.

Uses the alpaca-py SDK to submit orders to Alpaca's paper trading API.
Paper trading is free, no real money involved, and provides a full
simulation of live trading with real market data.

Setup:
    1. Sign up at https://alpaca.markets (free)
    2. Go to Paper Trading → API Keys
    3. Set environment variables:
       export ALPACA_API_KEY="your-paper-api-key"
       export ALPACA_SECRET_KEY="your-paper-secret-key"
       export ALPACA_PAPER=true  (default, use false for live)

Usage:
    from rachatrades.core.execution import AlpacaBroker

    broker = AlpacaBroker()
    result = broker.buy("AAPL", qty=1)
    print(result)  # [OK] BUY 1 AAPL @ $185.23 (id=abc-123)
"""

import logging
import os
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lazy import — alpaca-py is optional until the user configures it
_alpaca_available = None


def _check_alpaca():
    """Check if alpaca-py is installed."""
    global _alpaca_available
    if _alpaca_available is None:
        try:
            import alpaca  # noqa: F401
            _alpaca_available = True
        except ImportError:
            _alpaca_available = False
    return _alpaca_available


class AlpacaBroker:
    """Alpaca paper/live trading broker.

    Reads configuration from environment variables:
    - ALPACA_API_KEY: Your Alpaca API key
    - ALPACA_SECRET_KEY: Your Alpaca secret key
    - ALPACA_PAPER: "true" (default) for paper trading, "false" for live
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: Optional[bool] = None,
    ):
        """Initialize Alpaca broker.

        Args:
            api_key: Alpaca API key (or set ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (or set ALPACA_SECRET_KEY env var)
            paper: True for paper trading (default), False for live
        """
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.paper = paper if paper is not None else os.environ.get("ALPACA_PAPER", "true").lower() == "true"

        self._client = None
        self._configured = bool(self.api_key and self.secret_key)

        if self._configured:
            self._init_client()
        else:
            logger.info(
                "Alpaca broker not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "to enable paper trading execution."
            )

    @property
    def is_configured(self) -> bool:
        """Check if Alpaca credentials are set."""
        return self._configured

    def _init_client(self):
        """Initialize the Alpaca TradingClient."""
        if not _check_alpaca():
            logger.error(
                "alpaca-py is not installed. Run: pip install alpaca-py"
            )
            self._configured = False
            return

        try:
            from alpaca.trading.client import TradingClient

            self._client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper,
            )
            mode = "PAPER" if self.paper else "LIVE"
            logger.info(f"Alpaca broker initialized ({mode} trading)")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            self._configured = False

    def is_connected(self) -> bool:
        """Check if broker connection is active by pinging the account endpoint."""
        if not self._client:
            return False
        try:
            account = self._client.get_account()
            return account.status == "ACTIVE"  # type: ignore
        except Exception as e:
            logger.error(f"Alpaca connection check failed: {e}")
            return False

    def get_account_info(self) -> dict:
        """Get account details: cash, equity, buying power, etc."""
        if not self._client:
            return {"error": "Not configured"}

        try:
            account = self._client.get_account()
            return {
                "status": str(account.status),
                "cash": float(account.cash),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "day_trade_count": int(account.daytrade_count),
                "pattern_day_trader": bool(account.pattern_day_trader),
                "trading_blocked": bool(account.trading_blocked),
                "paper": self.paper,
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {"error": str(e)}

    def submit_order(
        self,
        ticker: str,
        side: str,
        qty: int = 1,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> dict:
        """Submit an order to Alpaca.

        Args:
            ticker: Stock symbol (e.g., "AAPL")
            side: "buy" or "sell"
            qty: Number of shares
            order_type: "market" (default), "limit", etc.
            time_in_force: "day" (default), "gtc", "ioc"

        Returns:
            dict with order result
        """
        from .base import OrderResult

        if not self._client:
            return OrderResult(
                success=False,
                ticker=ticker,
                side=side,
                qty=qty,
                error="Alpaca broker not configured",
            )

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            # Map string to enums
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif = {
                "day": TimeInForce.DAY,
                "gtc": TimeInForce.GTC,
                "ioc": TimeInForce.IOC,
            }.get(time_in_force.lower(), TimeInForce.DAY)

            # Submit market order
            order_data = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=tif,
            )

            order = self._client.submit_order(order_data=order_data)

            filled_price = float(order.filled_avg_price) if order.filled_avg_price else None
            filled_qty = int(order.filled_qty) if order.filled_qty else None

            result = OrderResult(
                success=True,
                order_id=str(order.id),
                ticker=ticker,
                side=side,
                qty=qty,
                filled_price=filled_price,
                filled_qty=filled_qty,
                status=str(order.status),
                timestamp=datetime.now(),
                raw={
                    "id": str(order.id),
                    "client_order_id": str(order.client_order_id),
                    "status": str(order.status),
                    "type": str(order.type),
                    "side": str(order.side),
                    "symbol": order.symbol,
                },
            )

            mode = "PAPER" if self.paper else "LIVE"
            logger.info(f"[{mode}] Order submitted: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to submit order {side.upper()} {qty} {ticker}: {e}")
            return OrderResult(
                success=False,
                ticker=ticker,
                side=side,
                qty=qty,
                error=str(e),
            )

    def get_positions(self) -> list:
        """Get all open positions from Alpaca."""
        from .base import BrokerPosition

        if not self._client:
            return []

        try:
            positions = self._client.get_all_positions()
            return [
                BrokerPosition(
                    ticker=p.symbol,
                    qty=abs(int(p.qty)),
                    side="long" if int(p.qty) > 0 else "short",
                    avg_entry_price=float(p.avg_entry_price),
                    market_value=float(p.market_value),
                    unrealized_pnl=float(p.unrealized_pl),
                    current_price=float(p.current_price),
                )
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_position(self, ticker: str):
        """Get a specific position by ticker."""
        from .base import BrokerPosition

        if not self._client:
            return None

        try:
            p = self._client.get_open_position(ticker)
            return BrokerPosition(
                ticker=p.symbol,
                qty=abs(int(p.qty)),
                side="long" if int(p.qty) > 0 else "short",
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl),
                current_price=float(p.current_price),
            )
        except Exception:
            return None

    def close_position(self, ticker: str) -> dict:
        """Close an entire position at market."""
        from .base import OrderResult

        if not self._client:
            return OrderResult(
                success=False,
                ticker=ticker,
                side="close",
                error="Alpaca broker not configured",
            )

        try:
            order = self._client.close_position(ticker)

            result = OrderResult(
                success=True,
                order_id=str(order.id) if hasattr(order, "id") else None,
                ticker=ticker,
                side="close",
                status=str(order.status) if hasattr(order, "status") else "submitted",
                timestamp=datetime.now(),
            )

            mode = "PAPER" if self.paper else "LIVE"
            logger.info(f"[{mode}] Position closed: {ticker}")
            return result

        except Exception as e:
            logger.error(f"Failed to close position {ticker}: {e}")
            return OrderResult(
                success=False,
                ticker=ticker,
                side="close",
                error=str(e),
            )

    def close_all_positions(self) -> list:
        """Close all open positions. Useful for EOD cleanup."""
        from .base import OrderResult

        if not self._client:
            return []

        try:
            responses = self._client.close_all_positions(cancel_orders=True)
            results = []
            for r in responses:
                # Each response has .status (int) and .body (order or error)
                if hasattr(r, "status") and r.status == 200:
                    results.append(OrderResult(
                        success=True,
                        ticker=r.symbol if hasattr(r, "symbol") else "unknown",
                        side="close",
                        status="closed",
                    ))
                else:
                    results.append(OrderResult(
                        success=False,
                        side="close",
                        error=str(r),
                    ))

            mode = "PAPER" if self.paper else "LIVE"
            logger.info(f"[{mode}] Closed all positions: {len(results)} orders")
            return results
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return []

    # ── Convenience methods matching signal types ────────────────────

    def execute_buy(self, ticker: str, qty: int = 1) -> dict:
        """Execute a BUY signal."""
        return self.submit_order(ticker, "buy", qty)

    def execute_sell(self, ticker: str, qty: int = 1) -> dict:
        """Execute a SELL signal (close long position)."""
        # Use close_position to close the entire long
        return self.close_position(ticker)

    def execute_short(self, ticker: str, qty: int = 1) -> dict:
        """Execute a SHORT signal (sell short)."""
        return self.submit_order(ticker, "sell", qty)

    def execute_cover(self, ticker: str, qty: int = 1) -> dict:
        """Execute a COVER signal (close short position)."""
        # Use close_position to close the entire short
        return self.close_position(ticker)
