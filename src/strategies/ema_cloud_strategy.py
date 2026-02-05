"""
Rashemator Strategy - Multi-Timeframe EMA Cloud System.

Uses 10-min for signal (trend/zone detection) and 1-min for entry (pullback + oscillator).

Three EMA clouds per timeframe:
- 10-min: 5/12 (trend), 8/9 (midpoint), 34/50 (major S/R)
- 1-min:  50/120 (trend), 80/90 (midpoint), 340/500 (major S/R)

Trade Zones:
- LONG_ZONE: Price > trend cloud AND > major cloud → buy pullbacks
- SHORT_ZONE: Price < trend cloud AND < major cloud → sell rips
- FLAT_ZONE: Price between clouds → NO TRADE

Entry Confirmation:
- MFI < 20 OR Williams %R < -80 (tight oscillator filter)
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
    Rashemator Multi-Timeframe Strategy.
    
    Uses 10-min charts for trend/zone detection and 1-min for entry timing.
    
    BUY when ALL conditions are met:
    - LONG_ZONE: Price above both trend cloud (5/12 or 50/120) and major cloud (34/50 or 340/500)
    - Pullback detected: Price touching trend or midpoint cloud
    - Oscillator confirmation: MFI < 20 OR Williams %R < -80
    - Reclaim: Candle closes back above cloud after dipping
    
    SELL when ANY condition triggers:
    - Price closes below trend cloud (5/12 or 50/120)
    - Zone changes to FLAT or SHORT
    - MFI > 80 OR Williams %R > -20 (overbought)
    
    NO TRADE when in FLAT_ZONE (between clouds).
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
        df_1m = mtf_data.df_1min
        
        # Need at least 1-min data for entry
        if df_1m is None or df_1m.empty or len(df_1m) < 500:
            return StrategyResult(
                ticker=ticker,
                signal=Signal.NO_POSITION,
                price=0.0,
                timestamp=pd.Timestamp.now(),
                reason="Insufficient 1-min data (need 500+ bars for 340/500 EMA)",
            )
        
        # Calculate oscillators on 1-min (for entry timing)
        df_1m_osc = self._calculate_oscillators(df_1m)
        mfi_signal, williams_signal = self._get_oscillator_signals(df_1m_osc)
        
        # Get Rashemator MTF signal
        if df_10m is not None and not df_10m.empty and len(df_10m) >= 50:
            rash_signal = get_rashemator_signal_mtf(df_10m, df_1m)
        else:
            # Fallback to 1-min only
            rash_signal = get_rashemator_signal_1min(df_1m)
        
        # Get latest values
        latest = df_1m.iloc[-1]
        current_price = float(latest["Close"])
        current_time = df_1m.index[-1]
        
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
            # Currently long - check for sell
            result = self._evaluate_sell(result, rash_signal, mfi_signal, williams_signal)
        elif has_short_position:
            # Currently short - check for cover
            result = self._evaluate_cover(result, rash_signal, mfi_signal, williams_signal)
        else:
            # No position - check for buy OR short entry
            if rash_signal.zone == Zone.LONG:
                result = self._evaluate_buy(result, rash_signal, oscillator_confirms)
            elif rash_signal.zone == Zone.SHORT:
                result = self._evaluate_short(result, rash_signal, overbought_confirms)
            else:
                result.signal = Signal.NO_POSITION
                result.reason = "FLAT ZONE: Price between clouds - no trade"
        
        return result

    def _evaluate_buy(
        self,
        result: StrategyResult,
        rash_signal: RashematorSignal,
        oscillator_confirms: bool,
    ) -> StrategyResult:
        """Evaluate BUY conditions."""
        
        # FLAT zone = NO TRADE (critical Rashemator rule)
        if rash_signal.zone == Zone.FLAT:
            result.signal = Signal.NO_POSITION
            result.reason = "FLAT ZONE: Price between clouds - no trade"
            return result
        
        # SHORT zone = no longs (we're long-only for now)
        if rash_signal.zone == Zone.SHORT:
            result.signal = Signal.NO_POSITION
            result.reason = "SHORT ZONE: Bearish trend - no long entries"
            return result
        
        # LONG zone - check for pullback entry
        buy_conditions = {
            "LONG_ZONE": rash_signal.zone == Zone.LONG,
            "Pullback to cloud": rash_signal.pullback_1m or rash_signal.pullback_10m,
            "Oscillator confirms": oscillator_confirms,
        }
        
        # Reclaim is bonus, not required
        if rash_signal.reclaim_detected:
            buy_conditions["Cloud reclaim"] = True
        
        met = [k for k, v in buy_conditions.items() if v]
        unmet = [k for k, v in buy_conditions.items() if not v]
        
        # BUY: LONG zone + Pullback + Oscillator (reclaim is bonus)
        core_conditions = [
            rash_signal.zone == Zone.LONG,
            rash_signal.pullback_1m or rash_signal.pullback_10m,
            oscillator_confirms,
        ]
        
        if all(core_conditions):
            result.signal = Signal.BUY
            pb_type = rash_signal.pullback_type.value.lower()
            osc_detail = []
            if result.mfi_oversold:
                osc_detail.append(f"MFI={result.mfi_value:.0f}")
            if result.williams_r_oversold:
                osc_detail.append(f"WR={result.williams_r_value:.0f}")
            result.reason = f"BUY: {pb_type} pullback + {' '.join(osc_detail)}"
            if rash_signal.reclaim_detected:
                result.reason += " + reclaim"
        else:
            result.signal = Signal.NO_POSITION
            result.reason = f"LONG ZONE but missing: {', '.join(unmet)}"
        
        return result

    def _evaluate_sell(
        self,
        result: StrategyResult,
        rash_signal: RashematorSignal,
        mfi_signal: dict,
        williams_signal: dict,
    ) -> StrategyResult:
        """Evaluate SELL conditions."""
        
        sell_reasons = []
        
        # Zone changed to FLAT or SHORT
        if rash_signal.zone == Zone.FLAT:
            sell_reasons.append("Zone changed to FLAT")
        elif rash_signal.zone == Zone.SHORT:
            sell_reasons.append("Zone changed to SHORT")
        
        # Trend cloud turned bearish (50/120 on 1-min)
        if rash_signal.cloud_50_120 and not rash_signal.cloud_50_120.bullish:
            sell_reasons.append("50/120 cloud bearish")
        
        # Oscillator overbought (profit taking)
        if mfi_signal["overbought"]:
            sell_reasons.append(f"MFI overbought ({mfi_signal['value']:.0f})")
        if williams_signal["overbought"]:
            sell_reasons.append(f"Williams %R overbought ({williams_signal['value']:.0f})")
        
        if sell_reasons:
            result.signal = Signal.SELL
            result.reason = "SELL: " + ", ".join(sell_reasons)
        else:
            result.signal = Signal.HOLD
            result.reason = f"HOLD: {rash_signal.zone.value} zone, trend intact"
        
        return result

    def _evaluate_short(
        self,
        result: StrategyResult,
        rash_signal: RashematorSignal,
        overbought_confirms: bool,
    ) -> StrategyResult:
        """Evaluate SHORT conditions: SHORT zone + overbought oscillators."""
        
        # SHORT zone + overbought confirmation (rally/rejection are bonus)
        short_conditions = {
            "SHORT_ZONE": rash_signal.zone == Zone.SHORT,
            "Oscillator confirms": overbought_confirms,
        }
        
        met = [k for k, v in short_conditions.items() if v]
        unmet = [k for k, v in short_conditions.items() if not v]
        
        # SHORT: SHORT zone + Overbought
        if rash_signal.zone == Zone.SHORT and overbought_confirms:
            result.signal = Signal.SHORT
            osc_detail = []
            if result.mfi_overbought:
                osc_detail.append(f"MFI={result.mfi_value:.0f}")
            if result.williams_r_overbought:
                osc_detail.append(f"WR={result.williams_r_value:.0f}")
            result.reason = f"SHORT: overbought {' '.join(osc_detail)}"
            # Add rally/rejection info if present
            if rash_signal.rally_1m or rash_signal.rally_10m:
                result.reason += f" + {rash_signal.rally_type.value.lower()} rally"
            if rash_signal.rejection_detected:
                result.reason += " + rejection"
        else:
            result.signal = Signal.NO_POSITION
            result.reason = f"SHORT ZONE but missing: {', '.join(unmet)}"
        
        return result

    def _evaluate_cover(
        self,
        result: StrategyResult,
        rash_signal: RashematorSignal,
        mfi_signal: dict,
        williams_signal: dict,
    ) -> StrategyResult:
        """Evaluate COVER conditions (exit short position)."""
        
        cover_reasons = []
        
        # Zone changed to FLAT or LONG
        if rash_signal.zone == Zone.FLAT:
            cover_reasons.append("Zone changed to FLAT")
        elif rash_signal.zone == Zone.LONG:
            cover_reasons.append("Zone changed to LONG")
        
        # Trend cloud turned bullish (50/120 on 1-min)
        if rash_signal.cloud_50_120 and rash_signal.cloud_50_120.bullish:
            cover_reasons.append("50/120 cloud bullish")
        
        # Oscillator oversold (short profit taking)
        if mfi_signal["oversold"]:
            cover_reasons.append(f"MFI oversold ({mfi_signal['value']:.0f})")
        if williams_signal["oversold"]:
            cover_reasons.append(f"Williams %R oversold ({williams_signal['value']:.0f})")
        
        if cover_reasons:
            result.signal = Signal.COVER
            result.reason = "COVER: " + ", ".join(cover_reasons)
        else:
            result.signal = Signal.HOLD_SHORT
            result.reason = f"HOLD SHORT: {rash_signal.zone.value} zone, downtrend intact"
        
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

    logging.basicConfig(level=logging.INFO)

    strategy = EMACloudStrategy()

    # Fetch MTF data (use 15m since yfinance doesn't support 10m)
    print("Fetching AAPL data...")
    ticker = yf.Ticker("AAPL")
    df_10m = ticker.history(period="5d", interval="15m")
    df_1m = ticker.history(period="7d", interval="1m")
    
    print(f"Got {len(df_10m)} 10-min bars, {len(df_1m)} 1-min bars")
    
    # Create MTFData
    mtf_data = MTFData(ticker="AAPL", df_10min=df_10m, df_1min=df_1m)
    
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
    print(f"    Pullback: {result.pullback_10m}")
    print(f"\n  1-MIN CLOUDS:")
    print(f"    50/120:  {'BULL' if result.cloud_50_120_bullish else 'BEAR'}")
    print(f"    80/90:   {'BULL' if result.cloud_80_90_bullish else 'BEAR'}")
    print(f"    340/500: {'BULL' if result.cloud_340_500_bullish else 'BEAR'}")
    print(f"    Aligned: {result.clouds_aligned_1m}")
    print(f"    Pullback: {result.pullback_1m} ({result.pullback_type.value})")
    print(f"\n  OSCILLATORS:")
    print(f"    MFI: {result.mfi_value:.1f} (Oversold: {result.mfi_oversold})")
    print(f"    Williams %R: {result.williams_r_value:.1f} (Oversold: {result.williams_r_oversold})")
    print(f"    Confirms: {result.oscillator_confirms}")
    print(f"\n  MTF Aligned: {result.mtf_aligned}")
    print(f"  Reclaim: {result.reclaim_detected}")
    print(f"\n  Reason: {result.reason}")
