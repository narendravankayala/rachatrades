"""
Rashemator V1 — Cloud Flip Strategy (FROZEN).

Entry: 5/12 EMA cloud crossover + 34/50 trend filter.
Exit:  5/12 cloud flips opposite direction.

This file is FROZEN — do not modify. It serves as the baseline
for A/B testing against newer strategy versions.
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


class CloudFlipStrategy:
    """
    V1: Cloud Flip Strategy (FROZEN).

    BUY when 5/12 cloud flips bullish AND 34/50 is bullish.
    SELL when 5/12 cloud flips bearish.
    SHORT when 5/12 cloud flips bearish AND 34/50 is bearish.
    COVER when 5/12 cloud flips bullish.
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

    def _check_oscillator_confirmation(self, mfi_signal: dict, williams_signal: dict) -> bool:
        if not self.config.require_oscillator:
            return True
        if self.config.oscillator_or_logic:
            return mfi_signal["oversold"] or williams_signal["oversold"]
        else:
            return mfi_signal["oversold"] and williams_signal["oversold"]

    def _check_overbought_confirmation(self, mfi_signal: dict, williams_signal: dict) -> bool:
        if not self.config.require_oscillator:
            return True
        if self.config.oscillator_or_logic:
            return mfi_signal["overbought"] or williams_signal["overbought"]
        else:
            return mfi_signal["overbought"] and williams_signal["overbought"]

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

        df_10m_osc = self._calculate_oscillators(df_10m)
        mfi_signal, williams_signal = self._get_oscillator_signals(df_10m_osc)
        rash_signal = get_rashemator_signal_10min(df_10m)

        latest = df_10m.iloc[-1]
        current_price = float(latest["Close"])
        current_time = df_10m.index[-1]

        oscillator_confirms = self._check_oscillator_confirmation(mfi_signal, williams_signal)
        overbought_confirms = self._check_overbought_confirmation(mfi_signal, williams_signal)

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

        result = StrategyResult(
            ticker=ticker, signal=Signal.NO_POSITION, price=current_price,
            timestamp=current_time, zone=rash_signal.zone,
            cloud_5_12_bullish=rash_signal.cloud_5_12.bullish if rash_signal.cloud_5_12 else False,
            cloud_8_9_bullish=rash_signal.cloud_8_9.bullish if rash_signal.cloud_8_9 else False,
            cloud_34_50_bullish=rash_signal.cloud_34_50.bullish if rash_signal.cloud_34_50 else False,
            clouds_aligned_10m=rash_signal.clouds_aligned_10m,
            pullback_type=rash_signal.pullback_type,
            pullback_10m=rash_signal.pullback_10m,
            pullback_1m=rash_signal.pullback_1m,
            rally_type=rash_signal.rally_type,
            rally_10m=rash_signal.rally_10m,
            rally_1m=rash_signal.rally_1m,
            reclaim_detected=rash_signal.reclaim_detected,
            rejection_detected=rash_signal.rejection_detected,
            mfi_value=mfi_signal["value"],
            mfi_oversold=mfi_signal["oversold"],
            mfi_overbought=mfi_signal["overbought"],
            williams_r_value=williams_signal["value"],
            williams_r_oversold=williams_signal["oversold"],
            williams_r_overbought=williams_signal["overbought"],
            oscillator_confirms=oscillator_confirms,
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
            ema_fast=rash_signal.cloud_50_120.ema_fast if rash_signal.cloud_50_120 else None,
            ema_slow=rash_signal.cloud_50_120.ema_slow if rash_signal.cloud_50_120 else None,
            ema_cloud_bullish=rash_signal.cloud_50_120.bullish if rash_signal.cloud_50_120 else False,
            price_above_cloud=(rash_signal.zone == Zone.LONG),
        )

        # ── Signal Logic: Cloud Flip ─────────────────────────────────
        if has_open_position:
            result = self._evaluate_sell(result, rash_signal, entry_price)
        elif has_short_position:
            result = self._evaluate_cover(result, rash_signal, entry_price)
        else:
            if rash_signal.cloud_5_12_cross_up:
                if rash_signal.cloud_34_50 and rash_signal.cloud_34_50.bullish:
                    if self.config.use_cloud_spread_filter and rash_signal.cloud_5_12:
                        spread_pct = abs(rash_signal.cloud_5_12.ema_fast - rash_signal.cloud_5_12.ema_slow) / rash_signal.price * 100
                        if spread_pct < self.config.min_cloud_spread_pct:
                            result.signal = Signal.NO_POSITION
                            result.reason = f"5/12 spread too thin ({spread_pct:.3f}% < {self.config.min_cloud_spread_pct}%)"
                            return result
                    if self.config.use_oscillator_filter and not oscillator_confirms:
                        result.signal = Signal.NO_POSITION
                        result.reason = "5/12 flipped bullish + 34/50 bullish but oscillator not oversold"
                    elif self.config.use_order_blocks and ob_signal and not (ob_signal.near_bullish_ob or ob_signal.inside_bullish_ob):
                        result.signal = Signal.NO_POSITION
                        result.reason = "5/12 flipped bullish but no bullish OB support nearby"
                    else:
                        result.signal = Signal.BUY
                        result.reason = f"BUY: 5/12 cloud flipped bullish (zone: {rash_signal.zone.value})"
                else:
                    result.signal = Signal.NO_POSITION
                    result.reason = "5/12 flipped bullish but 34/50 bearish - no long against downtrend"
            elif rash_signal.cloud_5_12_cross_down:
                if rash_signal.cloud_34_50 and rash_signal.cloud_34_50.bearish:
                    if self.config.use_cloud_spread_filter and rash_signal.cloud_5_12:
                        spread_pct = abs(rash_signal.cloud_5_12.ema_fast - rash_signal.cloud_5_12.ema_slow) / rash_signal.price * 100
                        if spread_pct < self.config.min_cloud_spread_pct:
                            result.signal = Signal.NO_POSITION
                            result.reason = f"5/12 spread too thin ({spread_pct:.3f}% < {self.config.min_cloud_spread_pct}%)"
                            return result
                    if self.config.use_oscillator_filter and not overbought_confirms:
                        result.signal = Signal.NO_POSITION
                        result.reason = "5/12 flipped bearish + 34/50 bearish but oscillator not overbought"
                    elif self.config.use_order_blocks and ob_signal and not (ob_signal.near_bearish_ob or ob_signal.inside_bearish_ob):
                        result.signal = Signal.NO_POSITION
                        result.reason = "5/12 flipped bearish but no bearish OB resistance nearby"
                    else:
                        result.signal = Signal.SHORT
                        result.reason = f"SHORT: 5/12 cloud flipped bearish (zone: {rash_signal.zone.value})"
                else:
                    result.signal = Signal.NO_POSITION
                    result.reason = "5/12 flipped bearish but 34/50 bullish - no short against uptrend"
            else:
                result.signal = Signal.NO_POSITION
                cloud_dir = "bearish" if rash_signal.cloud_5_12 and rash_signal.cloud_5_12.bearish else "bullish"
                result.reason = f"No cloud flip, 5/12 currently {cloud_dir}"

        return result

    def _evaluate_sell(self, result, rash_signal, entry_price=0.0):
        if self.config.use_stop_loss and entry_price > 0:
            loss_pct = (rash_signal.price - entry_price) / entry_price * 100
            if loss_pct <= -self.config.stop_loss_pct:
                result.signal = Signal.SELL
                result.reason = f"SELL: stop loss hit ({loss_pct:.2f}% <= -{self.config.stop_loss_pct}%)"
                return result
        if rash_signal.cloud_5_12_cross_down:
            result.signal = Signal.SELL
            result.reason = f"SELL: 5/12 cloud flipped bearish (zone: {rash_signal.zone.value})"
        else:
            result.signal = Signal.HOLD
            result.reason = f"HOLD: {rash_signal.zone.value} zone, 5/12 still bullish"
        return result

    def _evaluate_cover(self, result, rash_signal, entry_price=0.0):
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
