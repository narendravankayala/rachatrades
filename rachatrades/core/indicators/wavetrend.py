"""WaveTrend Oscillator (based on LazyBear's WaveTrend implementation).

A leading oscillator that identifies overbought/oversold conditions and
crossover signals. Less prone to false signals in choppy markets compared
to traditional oscillators like RSI.

Usage:
    df = calculate_wavetrend(df)
    signal = get_wavetrend_signal(df)
    if signal["cross_up"] and signal["oversold"]:
        # Bullish crossover from oversold territory -> strong long signal
"""

import pandas as pd
import numpy as np


def calculate_wavetrend(
    df: pd.DataFrame,
    channel_period: int = 10,
    avg_period: int = 21,
    overbought: float = 60.0,
    oversold: float = -60.0,
) -> pd.DataFrame:
    """
    Calculate WaveTrend Oscillator.

    Based on LazyBear's TradingView implementation:
    1. Calculate typical price (HLC/3)
    2. Smooth with EMA → ESA
    3. Calculate deviation: EMA of |close - ESA| → D
    4. Normalize: CI = (close - ESA) / (0.015 * D)
    5. Smooth CI with EMA → WT1 (main line)
    6. SMA(WT1, 4) → WT2 (signal line)

    Args:
        df: DataFrame with OHLCV data
        channel_period: Period for ESA and deviation EMAs
        avg_period: Period for the final smoothing EMA (WT1)
        overbought: Overbought threshold (typically 60)
        oversold: Oversold threshold (typically -60)

    Returns:
        DataFrame with additional columns:
        - wt1: Main WaveTrend line
        - wt2: Signal line (SMA of WT1)
        - wt_cross_up: True when WT1 crosses above WT2 (bullish)
        - wt_cross_down: True when WT1 crosses below WT2 (bearish)
        - wt_overbought: True when WT1 > overbought threshold
        - wt_oversold: True when WT1 < oversold threshold
    """
    result = df.copy()
    hlc3 = (result["High"] + result["Low"] + result["Close"]) / 3

    # Step 1: EMA of typical price
    esa = hlc3.ewm(span=channel_period, adjust=False).mean()

    # Step 2: EMA of absolute deviation
    d = (hlc3 - esa).abs().ewm(span=channel_period, adjust=False).mean()

    # Step 3: Normalize (CI = channel index)
    # Guard against division by zero
    ci = pd.Series(
        np.where(d != 0, (hlc3 - esa) / (0.015 * d), 0.0),
        index=df.index,
    )

    # Step 4: WT1 = EMA of CI
    result["wt1"] = ci.ewm(span=avg_period, adjust=False).mean()

    # Step 5: WT2 = SMA of WT1 (signal line)
    result["wt2"] = result["wt1"].rolling(4).mean()

    # Crossovers
    wt1_prev = result["wt1"].shift(1)
    wt2_prev = result["wt2"].shift(1)
    result["wt_cross_up"] = (wt1_prev <= wt2_prev) & (result["wt1"] > result["wt2"])
    result["wt_cross_down"] = (wt1_prev >= wt2_prev) & (result["wt1"] < result["wt2"])

    # Overbought / oversold zones
    result["wt_overbought"] = result["wt1"] > overbought
    result["wt_oversold"] = result["wt1"] < oversold

    return result


def get_wavetrend_signal(df: pd.DataFrame) -> dict:
    """
    Get the current WaveTrend signal from the latest data.

    Returns:
        Dictionary with current signal state:
        - wt1: Main WaveTrend value
        - wt2: Signal line value
        - cross_up: True if bullish crossover on latest bar
        - cross_down: True if bearish crossover on latest bar
        - overbought: True if WT1 in overbought zone
        - oversold: True if WT1 in oversold zone
        - bullish: True if WT1 > WT2 (above signal line)
        - increasing: True if WT1 is rising
    """
    if df.empty or "wt1" not in df.columns:
        return {
            "wt1": None,
            "wt2": None,
            "cross_up": False,
            "cross_down": False,
            "overbought": False,
            "oversold": False,
            "bullish": False,
            "increasing": False,
        }

    latest = df.iloc[-1]
    wt1 = latest.get("wt1", np.nan)
    wt2 = latest.get("wt2", np.nan)

    if pd.isna(wt1) or pd.isna(wt2):
        return {
            "wt1": None,
            "wt2": None,
            "cross_up": False,
            "cross_down": False,
            "overbought": False,
            "oversold": False,
            "bullish": False,
            "increasing": False,
        }

    prev_wt1 = df["wt1"].iloc[-2] if len(df) > 1 else np.nan

    return {
        "wt1": float(wt1),
        "wt2": float(wt2),
        "cross_up": bool(latest.get("wt_cross_up", False)),
        "cross_down": bool(latest.get("wt_cross_down", False)),
        "overbought": bool(latest.get("wt_overbought", False)),
        "oversold": bool(latest.get("wt_oversold", False)),
        "bullish": float(wt1) > float(wt2),
        "increasing": float(wt1) > float(prev_wt1) if not pd.isna(prev_wt1) else False,
    }
