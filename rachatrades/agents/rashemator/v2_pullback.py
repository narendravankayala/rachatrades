"""
Rashemator V2 — Pullback Reclaim Strategy.

Based on Rash's actual system from his video + images:
- Clouds define TREND DIRECTION, not entry signals
- Entry: price pulls back to cloud support and reclaims (bounces above)
- Exit: 5/12 cloud flips (crossover = trend change = exit)
- Deep pullback: price breaks 5/12, bounces off 34/50

Key rules from Rash:
- Above 5/12 & 34/50 → buy every dip
- Below 5/12 & 34/50 → sell every rip
- Between clouds → FLAT, stay out
- Highest probability: price above/below BOTH clouds
"""

import logging
from typing import Dict, List, Optional, Set

import pandas as pd

from rachatrades.core.indicators import (
    Zone,
    RashematorSignal,
    get_rashemator_signal_10min,
    calculate_mfi,
    calculate_williams_r,
    get_mfi_signal,
    get_williams_r_signal,
)
from rachatrades.core.indicators.order_blocks import detect_order_blocks, OrderBlockSignal
from rachatrades.core.data import MTFData
from rachatrades.agents.rashemator.strategy import Signal, StrategyConfig, StrategyResult

logger = logging.getLogger(__name__)


class PullbackStrategy:
    """
    V2: Pullback Reclaim Strategy (Rash's actual system).

    LONG ENTRY:
      - 5/12 cloud bullish (uptrend) + 34/50 bullish (major trend)
      - Price pulls back to 5/12 cloud and reclaims above it
      - NOT on the crossover bar itself (trend must be established)

    LONG ENTRY (deep pullback):
      - 34/50 cloud still bullish (major trend intact)
      - Price pulled back through 5/12 to 34/50 and reclaims above 34/50

    SHORT ENTRY:
      - 5/12 cloud bearish + 34/50 bearish
      - Price rallies to 5/12 cloud and rejects below it

    SHORT ENTRY (deep rally):
      - 34/50 cloud still bearish
      - Price rallied through 5/12 to 34/50 and rejects below 34/50

    EXIT:
      - SELL: 5/12 cloud flips bearish (trend over)
      - COVER: 5/12 cloud flips bullish (trend over)
      - Or stop loss
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()

    def _calculate_oscillators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        result = df.copy()
        result = calculate_mfi(result, period=self.config.mfi_period)
        result = calculate_williams_r(result, period=self.config.williams_r_period)
        return result

    def _get_oscillator_signals(self, df: pd.DataFrame) -> tuple:
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

    def evaluate_mtf(
        self,
        ticker: str,
        mtf_data: MTFData,
        has_open_position: bool = False,
        has_short_position: bool = False,
        entry_price: float = 0.0,
    ) -> StrategyResult:
        df_10m = mtf_data.df_10min

        if df_10m is None or df_10m.empty or len(df_10m) < 50:
            return StrategyResult(
                ticker=ticker, signal=Signal.NO_POSITION, price=0.0,
                timestamp=pd.Timestamp.now(),
                reason="Insufficient 10-min data (need 50+ bars for 34/50 EMA)",
            )

        # Calculate indicators
        df_10m_osc = self._calculate_oscillators(df_10m)
        mfi_signal, williams_signal = self._get_oscillator_signals(df_10m_osc)
        rash_signal = get_rashemator_signal_10min(df_10m)

        latest = df_10m.iloc[-1]
        current_price = float(latest["Close"])
        current_time = df_10m.index[-1]

        # Order blocks (optional)
        ob_signal: Optional[OrderBlockSignal] = None
        if self.config.use_order_blocks:
            try:
                ob_signal = detect_order_blocks(
                    df_10m,
                    volume_pivot_length=self.config.ob_volume_pivot_length,
                    mitigation_method=self.config.ob_mitigation_method,
                    max_obs=self.config.ob_max_active,
                )
            except Exception as e:
                logger.warning(f"Order block detection failed for {ticker}: {e}")

        # Build base result
        result = StrategyResult(
            ticker=ticker, signal=Signal.NO_POSITION, price=current_price,
            timestamp=current_time, zone=rash_signal.zone,
            cloud_5_12_bullish=rash_signal.cloud_5_12.bullish if rash_signal.cloud_5_12 else False,
            cloud_8_9_bullish=rash_signal.cloud_8_9.bullish if rash_signal.cloud_8_9 else False,
            cloud_34_50_bullish=rash_signal.cloud_34_50.bullish if rash_signal.cloud_34_50 else False,
            clouds_aligned_10m=rash_signal.clouds_aligned_10m,
            pullback_type=rash_signal.pullback_type,
            pullback_10m=rash_signal.pullback_10m,
            rally_type=rash_signal.rally_type,
            rally_10m=rash_signal.rally_10m,
            reclaim_detected=rash_signal.reclaim_detected,
            rejection_detected=rash_signal.rejection_detected,
            mfi_value=mfi_signal["value"],
            mfi_oversold=mfi_signal["oversold"],
            mfi_overbought=mfi_signal["overbought"],
            williams_r_value=williams_signal["value"],
            williams_r_oversold=williams_signal["oversold"],
            williams_r_overbought=williams_signal["overbought"],
            near_bullish_ob=ob_signal.near_bullish_ob if ob_signal else False,
            near_bearish_ob=ob_signal.near_bearish_ob if ob_signal else False,
            inside_bullish_ob=ob_signal.inside_bullish_ob if ob_signal else False,
            inside_bearish_ob=ob_signal.inside_bearish_ob if ob_signal else False,
            nearest_support_price=(
                ob_signal.nearest_support.avg if ob_signal and ob_signal.nearest_support else None
            ),
            nearest_resistance_price=(
                ob_signal.nearest_resistance.avg if ob_signal and ob_signal.nearest_resistance else None
            ),
            active_filters=self.config.describe(),
        )

        # ── Signal Logic: Pullback Reclaim ───────────────────────────
        if has_open_position:
            result = self._evaluate_sell(result, rash_signal, entry_price)
        elif has_short_position:
            result = self._evaluate_cover(result, rash_signal, entry_price)
        else:
            result = self._evaluate_entry(result, rash_signal, ob_signal, mfi_signal, williams_signal)

        return result

    def _evaluate_entry(
        self,
        result: StrategyResult,
        rash_signal: RashematorSignal,
        ob_signal: Optional[OrderBlockSignal],
        mfi_signal: dict,
        williams_signal: dict,
    ) -> StrategyResult:
        """Evaluate entry conditions based on pullback reclaim / rally rejection."""

        c5_12 = rash_signal.cloud_5_12
        c34_50 = rash_signal.cloud_34_50

        # ── LONG ENTRIES ─────────────────────────────────────────────
        # Shallow pullback: 5/12 bullish + 34/50 bullish + reclaim on 5/12
        if (c5_12 and c5_12.bullish
                and c34_50 and c34_50.bullish
                and rash_signal.reclaim_detected
                and not rash_signal.cloud_5_12_cross_up):  # Not the crossover bar itself

            # Apply optional filters
            if not self._passes_long_filters(rash_signal, ob_signal, mfi_signal, williams_signal):
                return result

            result.signal = Signal.BUY
            result.reason = f"BUY: pullback reclaim on 5/12 cloud (zone: {rash_signal.zone.value})"
            return result

        # Deep pullback: 34/50 still bullish + reclaim on 34/50
        if (c34_50 and c34_50.bullish
                and rash_signal.deep_reclaim_detected
                and not rash_signal.cloud_5_12_cross_up):

            if not self._passes_long_filters(rash_signal, ob_signal, mfi_signal, williams_signal):
                return result

            result.signal = Signal.BUY
            result.reason = f"BUY: deep pullback reclaim on 34/50 cloud (zone: {rash_signal.zone.value})"
            return result

        # ── SHORT ENTRIES ────────────────────────────────────────────
        # Shallow rally: 5/12 bearish + 34/50 bearish + rejection on 5/12
        if (c5_12 and c5_12.bearish
                and c34_50 and c34_50.bearish
                and rash_signal.rejection_detected
                and not rash_signal.cloud_5_12_cross_down):  # Not the crossover bar itself

            if not self._passes_short_filters(rash_signal, ob_signal, mfi_signal, williams_signal):
                return result

            result.signal = Signal.SHORT
            result.reason = f"SHORT: rally rejection on 5/12 cloud (zone: {rash_signal.zone.value})"
            return result

        # Deep rally: 34/50 still bearish + rejection on 34/50
        if (c34_50 and c34_50.bearish
                and rash_signal.deep_rejection_detected
                and not rash_signal.cloud_5_12_cross_down):

            if not self._passes_short_filters(rash_signal, ob_signal, mfi_signal, williams_signal):
                return result

            result.signal = Signal.SHORT
            result.reason = f"SHORT: deep rally rejection on 34/50 cloud (zone: {rash_signal.zone.value})"
            return result

        # ── NO ENTRY ─────────────────────────────────────────────────
        if c5_12 and c34_50:
            if c5_12.bullish and c34_50.bullish:
                result.reason = "Uptrend: waiting for pullback to 5/12 cloud"
            elif c5_12.bearish and c34_50.bearish:
                result.reason = "Downtrend: waiting for rally to 5/12 cloud"
            else:
                result.reason = f"Mixed trend (5/12: {'bull' if c5_12.bullish else 'bear'}, 34/50: {'bull' if c34_50.bullish else 'bear'}) — FLAT"
        else:
            result.reason = "Insufficient cloud data"

        return result

    def _passes_long_filters(
        self,
        rash_signal: RashematorSignal,
        ob_signal: Optional[OrderBlockSignal],
        mfi_signal: dict,
        williams_signal: dict,
    ) -> bool:
        """Check optional filters for long entries. Returns False if filtered out."""
        # Cloud spread filter
        if self.config.use_cloud_spread_filter and rash_signal.cloud_5_12:
            spread_pct = abs(rash_signal.cloud_5_12.ema_fast - rash_signal.cloud_5_12.ema_slow) / rash_signal.price * 100
            if spread_pct < self.config.min_cloud_spread_pct:
                return False

        # Oscillator filter (oversold = good time to buy the dip)
        if self.config.use_oscillator_filter:
            oversold = mfi_signal["oversold"] or williams_signal["oversold"]
            if not oversold:
                return False

        # Order block filter
        if self.config.use_order_blocks and ob_signal:
            if not (ob_signal.near_bullish_ob or ob_signal.inside_bullish_ob):
                return False

        return True

    def _passes_short_filters(
        self,
        rash_signal: RashematorSignal,
        ob_signal: Optional[OrderBlockSignal],
        mfi_signal: dict,
        williams_signal: dict,
    ) -> bool:
        """Check optional filters for short entries. Returns False if filtered out."""
        if self.config.use_cloud_spread_filter and rash_signal.cloud_5_12:
            spread_pct = abs(rash_signal.cloud_5_12.ema_fast - rash_signal.cloud_5_12.ema_slow) / rash_signal.price * 100
            if spread_pct < self.config.min_cloud_spread_pct:
                return False

        if self.config.use_oscillator_filter:
            overbought = mfi_signal["overbought"] or williams_signal["overbought"]
            if not overbought:
                return False

        if self.config.use_order_blocks and ob_signal:
            if not (ob_signal.near_bearish_ob or ob_signal.inside_bearish_ob):
                return False

        return True

    def _evaluate_sell(self, result, rash_signal, entry_price=0.0):
        """Exit long: 5/12 cloud flips bearish or stop loss."""
        # Stop loss
        if self.config.use_stop_loss and entry_price > 0:
            loss_pct = (rash_signal.price - entry_price) / entry_price * 100
            if loss_pct <= -self.config.stop_loss_pct:
                result.signal = Signal.SELL
                result.reason = f"SELL: stop loss hit ({loss_pct:.2f}% <= -{self.config.stop_loss_pct}%)"
                return result

        # 5/12 cloud flips bearish = trend over
        if rash_signal.cloud_5_12_cross_down:
            result.signal = Signal.SELL
            result.reason = f"SELL: 5/12 cloud flipped bearish (zone: {rash_signal.zone.value})"
        else:
            result.signal = Signal.HOLD
            result.reason = f"HOLD: {rash_signal.zone.value} zone, 5/12 still bullish"
        return result

    def _evaluate_cover(self, result, rash_signal, entry_price=0.0):
        """Exit short: 5/12 cloud flips bullish or stop loss."""
        if self.config.use_stop_loss and entry_price > 0:
            loss_pct = (entry_price - rash_signal.price) / entry_price * 100
            if loss_pct <= -self.config.stop_loss_pct:
                result.signal = Signal.COVER
                result.reason = f"COVER: stop loss hit ({loss_pct:.2f}% <= -{self.config.stop_loss_pct}%)"
                return result

        if rash_signal.cloud_5_12_cross_up:
            result.signal = Signal.COVER
            result.reason = f"COVER: 5/12 cloud flipped bullish (zone: {rash_signal.zone.value})"
        else:
            result.signal = Signal.HOLD_SHORT
            result.reason = f"HOLD SHORT: {rash_signal.zone.value} zone, 5/12 still bearish"
        return result

    def scan_universe_mtf(
        self,
        mtf_data: Dict[str, MTFData],
        open_positions: Set[str],
        short_positions: Optional[Set[str]] = None,
    ) -> List[StrategyResult]:
        if short_positions is None:
            short_positions = set()
        results = []
        for ticker, data in mtf_data.items():
            try:
                result = self.evaluate_mtf(
                    ticker, data,
                    has_open_position=(ticker in open_positions),
                    has_short_position=(ticker in short_positions),
                )
                results.append(result)
                if result.signal in (Signal.BUY, Signal.SELL, Signal.SHORT, Signal.COVER):
                    logger.info(f"{ticker}: {result.signal.value} - {result.reason}")
            except Exception as e:
                logger.error(f"Error evaluating {ticker}: {e}")
        return results
