"""
Rashemator Strategy - 10-Minute EMA Cloud System.

Everything runs on 10-min candles (resampled from 1-min).

Three EMA clouds on 10-min:
- 5/12 (trend), 8/9 (midpoint), 34/50 (major S/R)

Trade Zones:
- LONG_ZONE: Price > 5/12 cloud AND > 34/50 cloud → buy pullbacks
- SHORT_ZONE: Price < 5/12 cloud AND < 34/50 cloud → sell rips
- FLAT_ZONE: Price between clouds → NO TRADE

Entry Confirmation:
- MFI < 20 OR Williams %R < -80 (14-period on 10-min = 140 min lookback)
- Pullback to cloud support detected
- Candle closes back above cloud (reclaim)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import pandas as pd

from src.indicators import (
    Zone,
    PullbackType,
    RallyType,
    RashematorSignal,
    get_rashemator_signal_10min,
    get_rashemator_signal_1min,
    get_rashemator_signal_mtf,
    calculate_mfi,
    calculate_williams_r,
    get_mfi_signal,
    get_williams_r_signal,
    # Legacy compatibility
    calculate_ema_cloud,
    get_ema_cloud_signal,
)
from src.data import MTFData

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"  # Open short position
    COVER = "COVER"  # Close short position
    HOLD = "HOLD"
    HOLD_SHORT = "HOLD_SHORT"  # Hold short position
    NO_POSITION = "NO_POSITION"


@dataclass
class StrategyConfig:
    """Configuration for the Rashemator strategy."""
    
    # MFI settings (tight thresholds)
    mfi_period: int = 14
    mfi_oversold: float = 20.0
    mfi_overbought: float = 80.0

    # Williams %R settings (tight thresholds)
    williams_r_period: int = 14
    williams_r_oversold: float = -80.0
    williams_r_overbought: float = -20.0
    
    # Require oscillator confirmation for entries
    require_oscillator: bool = True
    
    # Use OR logic for oscillators (either MFI or WR confirms)
    oscillator_or_logic: bool = True


@dataclass
class StrategyResult:
    """Result from strategy evaluation."""
    ticker: str
    signal: Signal
    price: float
    timestamp: pd.Timestamp

    # Zone (from Rashemator)
    zone: Zone = Zone.FLAT
    
    # Cloud states (10-min)
    cloud_5_12_bullish: bool = False
    cloud_8_9_bullish: bool = False
    cloud_34_50_bullish: bool = False
    clouds_aligned_10m: bool = False
    
    # Cloud states (1-min)
    cloud_50_120_bullish: bool = False
    cloud_80_90_bullish: bool = False
    cloud_340_500_bullish: bool = False
    clouds_aligned_1m: bool = False
    
    # MTF alignment
    mtf_aligned: bool = False
    
    # Pullback detection (for longs)
    pullback_type: PullbackType = PullbackType.NONE
    pullback_10m: bool = False
    pullback_1m: bool = False
    
    # Rally detection (for shorts)
    rally_type: RallyType = RallyType.NONE
    rally_10m: bool = False
    rally_1m: bool = False
    
    # Reclaim signal (bullish)
    reclaim_detected: bool = False
    
    # Rejection signal (bearish)
    rejection_detected: bool = False
    
    # Oscillator values
    mfi_value: Optional[float] = None
    mfi_oversold: bool = False
    mfi_overbought: bool = False

    williams_r_value: Optional[float] = None
    williams_r_oversold: bool = False
    williams_r_overbought: bool = False
    
    # Combined oscillator confirmation
    oscillator_confirms: bool = False

    # Reasoning
    reason: str = ""
    
    # Legacy compatibility
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    ema_cloud_bullish: bool = False
    price_above_cloud: bool = False


class EMACloudStrategy:
    """
    Rashemator 10-Minute Strategy.
    
    All analysis on 10-min candles: zones, pullbacks, oscillators.
    
    BUY when 5/12 cloud flips bullish AND 34/50 major cloud is bullish (uptrend).
    
    SELL when 5/12 cloud flips bearish (crosses down).
    
    SHORT when 5/12 cloud flips bearish AND 34/50 major cloud is bearish (downtrend).
    
    COVER when 5/12 cloud flips bullish (crosses up).
    
    34/50 major cloud = trend filter. 5/12 trend cloud = entry/exit timing.
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        """Initialize the strategy with configuration."""
        self.config = config or StrategyConfig()

    def _calculate_oscillators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MFI and Williams %R indicators."""
        if df.empty:
            return df
        
        result = df.copy()
        result = calculate_mfi(result, period=self.config.mfi_period)
        result = calculate_williams_r(result, period=self.config.williams_r_period)
        return result

    def _get_oscillator_signals(
        self,
        df: pd.DataFrame,
    ) -> tuple:
        """Get MFI and Williams %R signals from DataFrame."""
        mfi_signal = get_mfi_signal(
            df,
            oversold_threshold=self.config.mfi_oversold,
            overbought_threshold=self.config.mfi_overbought,
        )
        williams_signal = get_williams_r_signal(
            df,
            oversold_threshold=self.config.williams_r_oversold,
            overbought_threshold=self.config.williams_r_overbought,
        )
        return mfi_signal, williams_signal

    def _check_oscillator_confirmation(
        self,
        mfi_signal: dict,
        williams_signal: dict,
    ) -> bool:
        """Check if oscillators confirm entry (OR logic) - for longs (oversold)."""
        if not self.config.require_oscillator:
            return True
        
        if self.config.oscillator_or_logic:
            # Either MFI OR Williams %R confirms
            return mfi_signal["oversold"] or williams_signal["oversold"]
        else:
            # Both must confirm (original strict logic)
            return mfi_signal["oversold"] and williams_signal["oversold"]

    def _check_overbought_confirmation(
        self,
        mfi_signal: dict,
        williams_signal: dict,
    ) -> bool:
        """Check if oscillators confirm entry (OR logic) - for shorts (overbought)."""
        if not self.config.require_oscillator:
            return True
        
        if self.config.oscillator_or_logic:
            # Either MFI OR Williams %R confirms
            return mfi_signal["overbought"] or williams_signal["overbought"]
        else:
            # Both must confirm
            return mfi_signal["overbought"] and williams_signal["overbought"]

    def evaluate_mtf(
        self,
        ticker: str,
        mtf_data: MTFData,
        has_open_position: bool = False,
        has_short_position: bool = False,
    ) -> StrategyResult:
        """
        Evaluate using multi-timeframe Rashemator strategy.
        
        Args:
            ticker: Stock ticker symbol
            mtf_data: MTFData with both 10-min and 1-min DataFrames
            has_open_position: Whether we currently hold a long position
            has_short_position: Whether we currently hold a short position
            
        Returns:
            StrategyResult with signal and indicator values
        """
        df_10m = mtf_data.df_10min
        
        # Need at least 50 bars of 10-min data for 34/50 EMA stability
        if df_10m is None or df_10m.empty or len(df_10m) < 50:
            return StrategyResult(
                ticker=ticker,
                signal=Signal.NO_POSITION,
                price=0.0,
                timestamp=pd.Timestamp.now(),
                reason="Insufficient 10-min data (need 50+ bars for 34/50 EMA)",
            )
        
        # Calculate oscillators on 10-min (14 period = 140 minutes)
        df_10m_osc = self._calculate_oscillators(df_10m)
        mfi_signal, williams_signal = self._get_oscillator_signals(df_10m_osc)
        
        # Get Rashemator signal on 10-min
        rash_signal = get_rashemator_signal_10min(df_10m)
        
        # Get latest values from 10-min
        latest = df_10m.iloc[-1]
        current_price = float(latest["Close"])
        current_time = df_10m.index[-1]
        
        # Check oscillator confirmation (OR logic: MFI < 20 OR WR < -80)
        oscillator_confirms = self._check_oscillator_confirmation(mfi_signal, williams_signal)
        
        # Check overbought confirmation for shorts (OR logic: MFI > 80 OR WR > -20)
        overbought_confirms = self._check_overbought_confirmation(mfi_signal, williams_signal)
        
        # Build result
        result = StrategyResult(
            ticker=ticker,
            signal=Signal.NO_POSITION,
            price=current_price,
            timestamp=current_time,
            # Zone
            zone=rash_signal.zone,
            # 10-min clouds
            cloud_5_12_bullish=rash_signal.cloud_5_12.bullish if rash_signal.cloud_5_12 else False,
            cloud_8_9_bullish=rash_signal.cloud_8_9.bullish if rash_signal.cloud_8_9 else False,
            cloud_34_50_bullish=rash_signal.cloud_34_50.bullish if rash_signal.cloud_34_50 else False,
            clouds_aligned_10m=rash_signal.clouds_aligned_10m,
            # 1-min clouds
            cloud_50_120_bullish=rash_signal.cloud_50_120.bullish if rash_signal.cloud_50_120 else False,
            cloud_80_90_bullish=rash_signal.cloud_80_90.bullish if rash_signal.cloud_80_90 else False,
            cloud_340_500_bullish=rash_signal.cloud_340_500.bullish if rash_signal.cloud_340_500 else False,
            clouds_aligned_1m=rash_signal.clouds_aligned_1m,
            # MTF
            mtf_aligned=rash_signal.mtf_aligned,
            # Pullback (for longs)
            pullback_type=rash_signal.pullback_type,
            pullback_10m=rash_signal.pullback_10m,
            pullback_1m=rash_signal.pullback_1m,
            # Rally (for shorts)
            rally_type=rash_signal.rally_type,
            rally_10m=rash_signal.rally_10m,
            rally_1m=rash_signal.rally_1m,
            # Reclaim/Rejection
            reclaim_detected=rash_signal.reclaim_detected,
            rejection_detected=rash_signal.rejection_detected,
            # Oscillators
            mfi_value=mfi_signal["value"],
            mfi_oversold=mfi_signal["oversold"],
            mfi_overbought=mfi_signal["overbought"],
            williams_r_value=williams_signal["value"],
            williams_r_oversold=williams_signal["oversold"],
            williams_r_overbought=williams_signal["overbought"],
            oscillator_confirms=oscillator_confirms,
            # Legacy
            ema_fast=rash_signal.cloud_50_120.ema_fast if rash_signal.cloud_50_120 else None,
            ema_slow=rash_signal.cloud_50_120.ema_slow if rash_signal.cloud_50_120 else None,
            ema_cloud_bullish=rash_signal.cloud_50_120.bullish if rash_signal.cloud_50_120 else False,
            price_above_cloud=(rash_signal.zone == Zone.LONG),
        )
        
        # Determine signal based on position state
        if has_open_position:
            # Currently long - sell when 5/12 flips bearish
            result = self._evaluate_sell(result, rash_signal, mfi_signal, williams_signal)
        elif has_short_position:
            # Currently short - cover when 5/12 flips bullish
            result = self._evaluate_cover(result, rash_signal, mfi_signal, williams_signal)
        else:
            # No position - check for cloud flips with 34/50 trend filter
            if rash_signal.cloud_5_12_cross_up:
                # Only go LONG if 34/50 major cloud confirms uptrend
                if rash_signal.cloud_34_50 and rash_signal.cloud_34_50.bullish:
                    result = self._evaluate_buy(result, rash_signal, oscillator_confirms)
                else:
                    result.signal = Signal.NO_POSITION
                    result.reason = "5/12 flipped bullish but 34/50 bearish - no long against downtrend"
            elif rash_signal.cloud_5_12_cross_down:
                # Only go SHORT if 34/50 major cloud confirms downtrend
                if rash_signal.cloud_34_50 and rash_signal.cloud_34_50.bearish:
                    result = self._evaluate_short(result, rash_signal, overbought_confirms)
                else:
                    result.signal = Signal.NO_POSITION
                    result.reason = "5/12 flipped bearish but 34/50 bullish - no short against uptrend"
            else:
                result.signal = Signal.NO_POSITION
                cloud_dir = "bearish" if rash_signal.cloud_5_12 and rash_signal.cloud_5_12.bearish else "bullish"
                result.reason = f"No cloud flip, 5/12 currently {cloud_dir}"
        
        return result

    def _evaluate_buy(
        self,
        result: StrategyResult,
        rash_signal: RashematorSignal,
        oscillator_confirms: bool,
    ) -> StrategyResult:
        """Evaluate BUY conditions: 5/12 cloud flips bullish."""
        
        # BUY entry: 5/12 cloud just flipped bullish
        if rash_signal.cloud_5_12_cross_up:
            result.signal = Signal.BUY
            result.reason = f"BUY: 5/12 cloud flipped bullish (zone: {rash_signal.zone.value})"
        else:
            result.signal = Signal.NO_POSITION
            cloud_state = "bullish" if rash_signal.cloud_5_12 and rash_signal.cloud_5_12.bullish else "bearish"
            result.reason = f"Waiting for 5/12 cloud flip, currently {cloud_state}"
        
        return result

    def _evaluate_sell(
        self,
        result: StrategyResult,
        rash_signal: RashematorSignal,
        mfi_signal: dict,
        williams_signal: dict,
    ) -> StrategyResult:
        """Evaluate SELL conditions: 5/12 cloud flips bearish."""
        
        # SELL: 5/12 cloud just flipped bearish
        if rash_signal.cloud_5_12_cross_down:
            result.signal = Signal.SELL
            result.reason = f"SELL: 5/12 cloud flipped bearish (zone: {rash_signal.zone.value})"
        else:
            result.signal = Signal.HOLD
            result.reason = f"HOLD: {rash_signal.zone.value} zone, 5/12 still bullish"
        
        return result

    def _evaluate_short(
        self,
        result: StrategyResult,
        rash_signal: RashematorSignal,
        overbought_confirms: bool,
    ) -> StrategyResult:
        """Evaluate SHORT conditions: 5/12 cloud flips bearish."""
        
        # SHORT entry: 5/12 cloud just flipped bearish
        if rash_signal.cloud_5_12_cross_down:
            result.signal = Signal.SHORT
            result.reason = f"SHORT: 5/12 cloud flipped bearish (zone: {rash_signal.zone.value})"
        else:
            result.signal = Signal.NO_POSITION
            cloud_state = "bearish" if rash_signal.cloud_5_12 and rash_signal.cloud_5_12.bearish else "bullish"
            result.reason = f"SHORT ZONE, 5/12 already {cloud_state} - waiting for flip"
        
        return result

    def _evaluate_cover(
        self,
        result: StrategyResult,
        rash_signal: RashematorSignal,
        mfi_signal: dict,
        williams_signal: dict,
    ) -> StrategyResult:
        """Evaluate COVER conditions: 5/12 cloud flips bullish."""
        
        # COVER: 5/12 cloud just flipped bullish
        if rash_signal.cloud_5_12_cross_up:
            result.signal = Signal.COVER
            result.reason = f"COVER: 5/12 cloud flipped bullish (zone: {rash_signal.zone.value})"
        else:
            result.signal = Signal.HOLD_SHORT
            result.reason = f"HOLD SHORT: {rash_signal.zone.value} zone, 5/12 still bearish"
        
        return result

    def evaluate(
        self,
        ticker: str,
        df: pd.DataFrame,
        has_open_position: bool = False,
    ) -> StrategyResult:
        """
        Evaluate using single timeframe (legacy compatibility).
        
        For full MTF analysis, use evaluate_mtf() instead.
        """
        # Wrap in MTFData and use 1-min logic
        mtf_data = MTFData(ticker=ticker, df_1min=df, df_10min=None)
        return self.evaluate_mtf(ticker, mtf_data, has_open_position)

    def scan_universe_mtf(
        self,
        mtf_data: Dict[str, MTFData],
        open_positions: Set[str],
        short_positions: Optional[Set[str]] = None,
    ) -> List[StrategyResult]:
        """
        Scan multiple tickers using MTF data.
        
        Args:
            mtf_data: Dictionary mapping ticker -> MTFData
            open_positions: Set of tickers we currently hold long
            short_positions: Set of tickers we currently hold short
            
        Returns:
            List of StrategyResult for each ticker
        """
        if short_positions is None:
            short_positions = set()
            
        results = []
        
        for ticker, data in mtf_data.items():
            try:
                has_long = ticker in open_positions
                has_short = ticker in short_positions
                result = self.evaluate_mtf(
                    ticker, 
                    data, 
                    has_open_position=has_long,
                    has_short_position=has_short,
                )
                results.append(result)
                
                if result.signal in (Signal.BUY, Signal.SELL):
                    logger.info(f"{ticker}: {result.signal.value} - {result.reason}")
                    
            except Exception as e:
                logger.error(f"Error evaluating {ticker}: {e}")
                continue
        
        return results

    def scan_universe(
        self,
        data: Dict[str, pd.DataFrame],
        open_positions: Set[str],
    ) -> List[StrategyResult]:
        """
        Scan multiple tickers (legacy compatibility).
        
        For MTF analysis, use scan_universe_mtf() instead.
        """
        # Convert to MTFData format
        mtf_data = {
            ticker: MTFData(ticker=ticker, df_1min=df, df_10min=None)
            for ticker, df in data.items()
        }
        return self.scan_universe_mtf(mtf_data, open_positions)


if __name__ == "__main__":
    # Test the Rashemator strategy
    import yfinance as yf
    from src.data import DataProvider

    logging.basicConfig(level=logging.INFO)

    strategy = EMACloudStrategy()
    provider = DataProvider()

    # Fetch 1-min data and resample to true 10-min
    print("Fetching AAPL data (1-min -> resample to 10-min)...")
    mtf_data = provider.get_mtf_ohlcv("AAPL")
    
    if mtf_data.df_10min is not None:
        print(f"Got {len(mtf_data.df_10min)} true 10-min bars")
    
    # Evaluate
    result = strategy.evaluate_mtf("AAPL", mtf_data, has_open_position=False)

    print(f"\n{'='*60}")
    print(f"Rashemator Strategy Result for AAPL")
    print(f"{'='*60}")
    print(f"  Signal: {result.signal.value}")
    print(f"  Price: ${result.price:.2f}")
    print(f"  Zone: {result.zone.value}")
    print(f"\n  10-MIN CLOUDS:")
    print(f"    5/12:  {'BULL' if result.cloud_5_12_bullish else 'BEAR'}")
    print(f"    8/9:   {'BULL' if result.cloud_8_9_bullish else 'BEAR'}")
    print(f"    34/50: {'BULL' if result.cloud_34_50_bullish else 'BEAR'}")
    print(f"    Aligned: {result.clouds_aligned_10m}")
    print(f"    Pullback: {result.pullback_10m} ({result.pullback_type.value})")
    print(f"    Reclaim: {result.reclaim_detected}")
    print(f"\n  OSCILLATORS (14-period on 10-min = 140 min):")
    if result.mfi_value is not None:
        print(f"    MFI: {result.mfi_value:.1f} (Oversold: {result.mfi_oversold})")
    if result.williams_r_value is not None:
        print(f"    Williams %R: {result.williams_r_value:.1f} (Oversold: {result.williams_r_oversold})")
    print(f"    Confirms: {result.oscillator_confirms}")
    print(f"\n  Reason: {result.reason}")
