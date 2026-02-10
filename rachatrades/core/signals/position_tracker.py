"""Position tracker using SQLite for persistence."""

import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Position status."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class PositionType(Enum):
    """Position type."""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    """Represents a trading position."""
    id: Optional[int]
    ticker: str
    status: PositionStatus
    position_type: PositionType
    entry_time: datetime
    entry_price: float
    quantity: int = 1  # Virtual quantity for tracking
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    entry_reason: str = ""
    exit_reason: str = ""


class PositionTracker:
    """Track positions in SQLite database."""

    def __init__(self, db_path: str = "data/positions.db"):
        """
        Initialize position tracker.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    status TEXT NOT NULL,
                    position_type TEXT NOT NULL DEFAULT 'LONG',
                    entry_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity INTEGER DEFAULT 1,
                    exit_time TEXT,
                    exit_price REAL,
                    pnl REAL,
                    pnl_percent REAL,
                    entry_reason TEXT,
                    exit_reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_ticker
                ON positions(ticker)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_status
                ON positions(status)
            """)

            # Migrate old DB: add position_type column if missing
            cursor = conn.execute("PRAGMA table_info(positions)")
            columns = {row[1] for row in cursor.fetchall()}
            if "position_type" not in columns:
                conn.execute("ALTER TABLE positions ADD COLUMN position_type TEXT NOT NULL DEFAULT 'LONG'")
                logger.info("Migrated DB: added position_type column")

            # Create daily summary table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    open_positions INTEGER DEFAULT 0,
                    scanned_at TEXT
                )
            """)

            conn.commit()

    def open_position(
        self,
        ticker: str,
        price: float,
        timestamp: datetime,
        reason: str = "",
    ) -> Position:
        """
        Open a new position.

        Args:
            ticker: Stock ticker
            price: Entry price
            timestamp: Entry time
            reason: Why the position was opened

        Returns:
            The created Position
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO positions (ticker, status, position_type, entry_time, entry_price, entry_reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ticker, PositionStatus.OPEN.value, PositionType.LONG.value, timestamp.isoformat(), price, reason),
            )
            conn.commit()

            position = Position(
                id=cursor.lastrowid,
                ticker=ticker,
                status=PositionStatus.OPEN,
                position_type=PositionType.LONG,
                entry_time=timestamp,
                entry_price=price,
                entry_reason=reason,
            )

            logger.info(f"Opened LONG position: {ticker} @ ${price:.2f}")
            return position

    def close_position(
        self,
        ticker: str,
        price: float,
        timestamp: datetime,
        reason: str = "",
    ) -> Optional[Position]:
        """
        Close an open position.

        Args:
            ticker: Stock ticker
            price: Exit price
            timestamp: Exit time
            reason: Why the position was closed

        Returns:
            The closed Position, or None if no open position found
        """
        with sqlite3.connect(self.db_path) as conn:
            # Find the open LONG position
            cursor = conn.execute(
                """
                SELECT id, entry_price, entry_time FROM positions
                WHERE ticker = ? AND status = ? AND position_type = ?
                ORDER BY entry_time DESC LIMIT 1
                """,
                (ticker, PositionStatus.OPEN.value, PositionType.LONG.value),
            )
            row = cursor.fetchone()

            if not row:
                logger.warning(f"No open position found for {ticker}")
                return None

            position_id, entry_price, entry_time_str = row

            # Calculate P&L
            pnl = price - entry_price
            pnl_percent = (pnl / entry_price) * 100

            # Update the position
            conn.execute(
                """
                UPDATE positions
                SET status = ?, exit_time = ?, exit_price = ?,
                    pnl = ?, pnl_percent = ?, exit_reason = ?
                WHERE id = ?
                """,
                (
                    PositionStatus.CLOSED.value,
                    timestamp.isoformat(),
                    price,
                    pnl,
                    pnl_percent,
                    reason,
                    position_id,
                ),
            )
            conn.commit()

            position = Position(
                id=position_id,
                ticker=ticker,
                status=PositionStatus.CLOSED,
                position_type=PositionType.LONG,
                entry_time=datetime.fromisoformat(entry_time_str),
                entry_price=entry_price,
                exit_time=timestamp,
                exit_price=price,
                pnl=pnl,
                pnl_percent=pnl_percent,
                exit_reason=reason,
            )

            logger.info(
                f"Closed LONG position: {ticker} @ ${price:.2f} "
                f"(P&L: ${pnl:.2f}, {pnl_percent:.2f}%)"
            )
            return position

    def open_short_position(
        self,
        ticker: str,
        price: float,
        timestamp: datetime,
        reason: str = "",
    ) -> Position:
        """Open a new SHORT position."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO positions (ticker, status, position_type, entry_time, entry_price, entry_reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ticker, PositionStatus.OPEN.value, PositionType.SHORT.value, timestamp.isoformat(), price, reason),
            )
            conn.commit()

            position = Position(
                id=cursor.lastrowid,
                ticker=ticker,
                status=PositionStatus.OPEN,
                position_type=PositionType.SHORT,
                entry_time=timestamp,
                entry_price=price,
                entry_reason=reason,
            )

            logger.info(f"Opened SHORT position: {ticker} @ ${price:.2f}")
            return position

    def close_short_position(
        self,
        ticker: str,
        price: float,
        timestamp: datetime,
        reason: str = "",
    ) -> Optional[Position]:
        """Close a SHORT position (cover). P&L = entry - exit (profit when price drops)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, entry_price, entry_time FROM positions
                WHERE ticker = ? AND status = ? AND position_type = ?
                ORDER BY entry_time DESC LIMIT 1
                """,
                (ticker, PositionStatus.OPEN.value, PositionType.SHORT.value),
            )
            row = cursor.fetchone()

            if not row:
                logger.warning(f"No open SHORT position found for {ticker}")
                return None

            position_id, entry_price, entry_time_str = row

            # SHORT P&L: profit when price drops
            pnl = entry_price - price
            pnl_percent = (pnl / entry_price) * 100

            conn.execute(
                """
                UPDATE positions
                SET status = ?, exit_time = ?, exit_price = ?,
                    pnl = ?, pnl_percent = ?, exit_reason = ?
                WHERE id = ?
                """,
                (
                    PositionStatus.CLOSED.value,
                    timestamp.isoformat(),
                    price,
                    pnl,
                    pnl_percent,
                    reason,
                    position_id,
                ),
            )
            conn.commit()

            position = Position(
                id=position_id,
                ticker=ticker,
                status=PositionStatus.CLOSED,
                position_type=PositionType.SHORT,
                entry_time=datetime.fromisoformat(entry_time_str),
                entry_price=entry_price,
                exit_time=timestamp,
                exit_price=price,
                pnl=pnl,
                pnl_percent=pnl_percent,
                exit_reason=reason,
            )

            logger.info(
                f"Closed SHORT position: {ticker} @ ${price:.2f} "
                f"(P&L: ${pnl:.2f}, {pnl_percent:.2f}%)"
            )
            return position

    def get_open_short_tickers(self) -> Set[str]:
        """Get set of tickers with open SHORT positions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT DISTINCT ticker FROM positions WHERE status = ? AND position_type = ?",
                (PositionStatus.OPEN.value, PositionType.SHORT.value),
            )
            return {row[0] for row in cursor.fetchall()}

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM positions WHERE status = ?
                ORDER BY entry_time DESC
                """,
                (PositionStatus.OPEN.value,),
            )

            positions = []
            for row in cursor.fetchall():
                try:
                    pos_type = PositionType(row["position_type"])
                except (KeyError, IndexError):
                    pos_type = PositionType.LONG
                positions.append(
                    Position(
                        id=row["id"],
                        ticker=row["ticker"],
                        status=PositionStatus(row["status"]),
                        position_type=pos_type,
                        entry_time=datetime.fromisoformat(row["entry_time"]),
                        entry_price=row["entry_price"],
                        quantity=row["quantity"],
                        entry_reason=row["entry_reason"] or "",
                    )
                )

            return positions

    def get_open_tickers(self) -> Set[str]:
        """Get set of tickers with open positions."""
        positions = self.get_open_positions()
        return {p.ticker for p in positions}

    def has_open_position(self, ticker: str) -> bool:
        """Check if ticker has an open position."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM positions
                WHERE ticker = ? AND status = ?
                """,
                (ticker, PositionStatus.OPEN.value),
            )
            count = cursor.fetchone()[0]
            return count > 0

    def get_closed_positions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Position]:
        """Get closed positions within date range."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM positions WHERE status = ?"
            params = [PositionStatus.CLOSED.value]

            if start_date:
                query += " AND exit_time >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND exit_time <= ?"
                params.append(end_date.isoformat())

            query += " ORDER BY exit_time DESC"

            cursor = conn.execute(query, params)

            positions = []
            for row in cursor.fetchall():
                try:
                    pos_type = PositionType(row["position_type"])
                except (KeyError, IndexError):
                    pos_type = PositionType.LONG
                positions.append(
                    Position(
                        id=row["id"],
                        ticker=row["ticker"],
                        status=PositionStatus(row["status"]),
                        position_type=pos_type,
                        entry_time=datetime.fromisoformat(row["entry_time"]),
                        entry_price=row["entry_price"],
                        quantity=row["quantity"],
                        exit_time=datetime.fromisoformat(row["exit_time"]) if row["exit_time"] else None,
                        exit_price=row["exit_price"],
                        pnl=row["pnl"],
                        pnl_percent=row["pnl_percent"],
                        entry_reason=row["entry_reason"] or "",
                        exit_reason=row["exit_reason"] or "",
                    )
                )

            return positions

    def get_total_pnl(self) -> float:
        """Get total P&L from all closed positions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT COALESCE(SUM(pnl), 0) FROM positions
                WHERE status = ?
                """,
                (PositionStatus.CLOSED.value,),
            )
            return cursor.fetchone()[0]

    def get_last_exit_time(self, ticker: str) -> Optional[datetime]:
        """Get the most recent exit time for a ticker (for cooldown checks)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT exit_time FROM positions
                WHERE ticker = ? AND status = ? AND exit_time IS NOT NULL
                ORDER BY exit_time DESC LIMIT 1
                """,
                (ticker, PositionStatus.CLOSED.value),
            )
            row = cursor.fetchone()
            if row and row[0]:
                return datetime.fromisoformat(row[0])
            return None

    def get_stats(self) -> dict:
        """Get overall trading statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(CASE WHEN pnl = 0 THEN 1 ELSE 0 END) as breakeven_trades,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(AVG(pnl), 0) as avg_pnl,
                    COALESCE(AVG(pnl_percent), 0) as avg_pnl_percent
                FROM positions
                WHERE status = ?
                """,
                (PositionStatus.CLOSED.value,),
            )
            row = cursor.fetchone()

            open_cursor = conn.execute(
                "SELECT COUNT(*) FROM positions WHERE status = ?",
                (PositionStatus.OPEN.value,),
            )
            open_count = open_cursor.fetchone()[0]

            total = row[0] or 0
            winning = row[1] or 0

            return {
                "total_trades": total,
                "winning_trades": winning,
                "losing_trades": row[2] or 0,
                "breakeven_trades": row[3] or 0,
                "win_rate": (winning / total * 100) if total > 0 else 0,
                "total_pnl": row[4],
                "avg_pnl": row[5],
                "avg_pnl_percent": row[6],
                "open_positions": open_count,
            }

    def update_daily_summary(self, date: datetime):
        """Update the daily summary table."""
        stats = self.get_stats()
        date_str = date.strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_summary
                (date, total_trades, winning_trades, losing_trades, total_pnl, open_positions, scanned_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    date_str,
                    stats["total_trades"],
                    stats["winning_trades"],
                    stats["losing_trades"],
                    stats["total_pnl"],
                    stats["open_positions"],
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()


if __name__ == "__main__":
    # Test the position tracker
    logging.basicConfig(level=logging.INFO)

    tracker = PositionTracker("data/test_positions.db")

    # Open a test position
    pos = tracker.open_position(
        ticker="AAPL",
        price=150.00,
        timestamp=datetime.now(),
        reason="Test buy signal",
    )
    print(f"Opened: {pos}")

    # Check open positions
    print(f"Open tickers: {tracker.get_open_tickers()}")

    # Close the position
    closed = tracker.close_position(
        ticker="AAPL",
        price=155.00,
        timestamp=datetime.now(),
        reason="Test sell signal",
    )
    print(f"Closed: {closed}")

    # Get stats
    stats = tracker.get_stats()
    print(f"Stats: {stats}")
