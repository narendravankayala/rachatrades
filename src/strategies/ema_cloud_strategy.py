"""EMA Cloud Strategy - combines EMA Cloud, MFI, and Williams %R."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

import pandas as pd

from src.indicators import (
    calculate_ema_cloud,
    calculate_mfi,
    calculate_williams_r,
    get_ema_cloud_signal,
    get_mfi_signal,
    get_williams_r_signal,
)

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NO_POSITION = "NO_POSITION"


@dataclass
class StrategyConfig:
    """Configuration for the EMA Cloud strategy."""
    # EMA Cloud settings
    ema_fast_period: int = 5
    ema_slow_period: int = 12

    # MFI settings
    mfi_period: int = 14
    mfi_oversold: float = 20.0
    mfi_overbought: float = 80.0

    # Williams %R settings
    williams_r_period: int = 14
    williams_r_oversold: float = -80.0
    williams_r_overbought: float = -20.0


@dataclass
class StrategyResult:
    """Result from strategy evaluation."""
    ticker: str
    signal: Signal
    price: float
    timestamp: pd.Timestamp

    # Individual indicator values
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    ema_cloud_bullish: bool = False
    price_above_cloud: bool = False

    mfi_value: Optional[float] = None
    mfi_oversold: bool = False
    mfi_overbought: bool = False

    williams_r_value: Optional[float] = None
    williams_r_oversold: bool = False
    williams_r_overbought: bool = False

    # Reasoning
    reason: str = ""


class EMACloudStrategy:
    """
    EMA Cloud + MFI + Williams %R Strategy.

    BUY when ALL conditions are met:
    - EMA Cloud is bullish (EMA5 > EMA12)
    - Price is above the EMA cloud
    - MFI is oversold (< 20)
    - Williams %R is oversold (< -80)

    SELL when ANY condition flips:
    - EMA Cloud turns bearish (EMA5 < EMA12)
    - OR MFI becomes overbought (> 80)
    - OR Williams %R becomes overbought (> -20)

    HOLD while in position and no sell signal.
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        """Initialize the strategy with configuration."""
        self.config = config or StrategyConfig()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators for the strategy.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all indicator columns added
        """
        if df.empty:
            return df

        # Calculate EMA Cloud
        df = calculate_ema_cloud(
            df,
            fast_period=self.config.ema_fast_period,
            slow_period=self.config.ema_slow_period,
        )

        # Calculate MFI
        df = calculate_mfi(df, period=self.config.mfi_period)

        # Calculate Williams %R
        df = calculate_williams_r(df, period=self.config.williams_r_period)

        return df

    def evaluate(
        self,
        ticker: str,
        df: pd.DataFrame,
        has_open_position: bool = False,
    ) -> StrategyResult:
        """
        Evaluate the strategy for a single ticker.

        Args:
            ticker: Stock ticker symbol
            df: DataFrame with OHLCV data (indicators will be calculated)
            has_open_position: Whether we currently hold this stock

        Returns:
            StrategyResult with signal and indicator values
        """
        if df.empty or len(df) < max(
            self.config.ema_slow_period,
            self.config.mfi_period,
            self.config.williams_r_period,
        ):
            return StrategyResult(
                ticker=ticker,
                signal=Signal.NO_POSITION,
                price=0.0,
                timestamp=pd.Timestamp.now(),
                reason="Insufficient data",
            )

        # Calculate all indicators
        df_with_indicators = self.calculate_indicators(df)

        # Get latest values
        latest = df_with_indicators.iloc[-1]
        current_price = float(latest["Close"])
        current_time = df_with_indicators.index[-1]

        # Get individual signals
        ema_signal = get_ema_cloud_signal(df_with_indicators)
        mfi_signal = get_mfi_signal(
            df_with_indicators,
            oversold_threshold=self.config.mfi_oversold,
            overbought_threshold=self.config.mfi_overbought,
        )
        williams_signal = get_williams_r_signal(
            df_with_indicators,
            oversold_threshold=self.config.williams_r_oversold,
            overbought_threshold=self.config.williams_r_overbought,
        )

        # Build result with indicator values
        result = StrategyResult(
            ticker=ticker,
            signal=Signal.NO_POSITION,
            price=current_price,
            timestamp=current_time,
            ema_fast=ema_signal["ema_fast"],
            ema_slow=ema_signal["ema_slow"],
            ema_cloud_bullish=ema_signal["bullish"],
            price_above_cloud=ema_signal["price_above"],
            mfi_value=mfi_signal["value"],
            mfi_oversold=mfi_signal["oversold"],
            mfi_overbought=mfi_signal["overbought"],
            williams_r_value=williams_signal["value"],
            williams_r_oversold=williams_signal["oversold"],
            williams_r_overbought=williams_signal["overbought"],
        )

        # Determine signal
        if has_open_position:
            # Check for SELL conditions (any one triggers sell)
            sell_reasons = []

            if not ema_signal["bullish"]:
                sell_reasons.append("EMA Cloud turned bearish")

            if mfi_signal["overbought"]:
                sell_reasons.append(f"MFI overbought ({mfi_signal['value']:.1f})")

            if williams_signal["overbought"]:
                sell_reasons.append(f"Williams %R overbought ({williams_signal['value']:.1f})")

            if sell_reasons:
                result.signal = Signal.SELL
                result.reason = "SELL: " + ", ".join(sell_reasons)
            else:
                result.signal = Signal.HOLD
                result.reason = "HOLD: All conditions still valid"

        else:
            # Check for BUY conditions (all must be true)
            buy_conditions = {
                "EMA Cloud bullish": ema_signal["bullish"],
                "Price above cloud": ema_signal["price_above"],
                "MFI oversold": mfi_signal["oversold"],
                "Williams %R oversold": williams_signal["oversold"],
            }

            met_conditions = [k for k, v in buy_conditions.items() if v]
            unmet_conditions = [k for k, v in buy_conditions.items() if not v]

            if all(buy_conditions.values()):
                result.signal = Signal.BUY
                result.reason = "BUY: All conditions met - " + ", ".join(met_conditions)
            else:
                result.signal = Signal.NO_POSITION
                result.reason = f"NO BUY: Missing conditions - {', '.join(unmet_conditions)}"

        return result

    def scan_universe(
        self,
        data: Dict[str, pd.DataFrame],
        open_positions: Set[str],
    ) -> List[StrategyResult]:
        """
        Scan multiple tickers and return strategy results.

        Args:
            data: Dictionary mapping ticker -> OHLCV DataFrame
            open_positions: Set of tickers we currently hold

        Returns:
            List of StrategyResult for each ticker
        """
        results = []

        for ticker, df in data.items():
            try:
                has_position = ticker in open_positions
                result = self.evaluate(ticker, df, has_open_position=has_position)
                results.append(result)

                if result.signal in (Signal.BUY, Signal.SELL):
                    logger.info(f"{ticker}: {result.signal.value} - {result.reason}")

            except Exception as e:
                logger.error(f"Error evaluating {ticker}: {e}")
                continue

        return results


if __name__ == "__main__":
    # Test the strategy
    import yfinance as yf

    logging.basicConfig(level=logging.INFO)

    strategy = EMACloudStrategy()

    # Test with a single stock
    df = yf.Ticker("AAPL").history(period="1mo", interval="10m")
    result = strategy.evaluate("AAPL", df, has_open_position=False)

    print(f"\nStrategy Result for AAPL:")
    print(f"  Signal: {result.signal.value}")
    print(f"  Price: ${result.price:.2f}")
    print(f"  EMA Cloud: {'Bullish' if result.ema_cloud_bullish else 'Bearish'}")
    print(f"  Price Above Cloud: {result.price_above_cloud}")
    print(f"  MFI: {result.mfi_value:.1f} (Oversold: {result.mfi_oversold})")
    print(f"  Williams %R: {result.williams_r_value:.1f} (Oversold: {result.williams_r_oversold})")
    print(f"  Reason: {result.reason}")
