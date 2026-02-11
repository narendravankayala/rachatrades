"""Squeeze Momentum Indicator (based on LazyBear / John Carter TTM Squeeze).

Detects low-volatility "squeeze" periods (Bollinger Bands inside Keltner Channels)
that often precede explosive moves. The momentum histogram shows the direction
and strength of the move when the squeeze releases.

Usage:
    df = calculate_squeeze_momentum(df)
    signal = get_squeeze_signal(df)
    if signal["squeeze_releasing"] and signal["momentum_bullish"]:
        # Squeeze just released with bullish momentum → strong long entry
"""

import pandas as pd
import numpy as np


def calculate_squeeze_momentum(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
    mom_period: int = 20,
) -> pd.DataFrame:
    """
    Calculate Squeeze Momentum Indicator.

    Squeeze: Bollinger Bands inside Keltner Channels = low volatility.
    Momentum: Linear regression of (close - midline of highest/lowest).

    Args:
        df: DataFrame with OHLCV data
        bb_period: Bollinger Bands SMA period
        bb_mult: Bollinger Bands standard deviation multiplier
        kc_period: Keltner Channel EMA period
        kc_mult: Keltner Channel ATR multiplier
        mom_period: Momentum lookback period

    Returns:
        DataFrame with additional columns:
        - sqz_on: True when squeeze is active (BB inside KC)
        - sqz_off: True when squeeze is off (BB outside KC)
        - sqz_momentum: Momentum value (positive = bullish)
        - sqz_momentum_increasing: True when momentum is increasing
        - sqz_releasing: True on the bar the squeeze releases
    """
    result = df.copy()
    close = result["Close"]
    high = result["High"]
    low = result["Low"]

    # ── Bollinger Bands ──────────────────────────────────────────
    bb_sma = close.rolling(bb_period).mean()
    bb_std = close.rolling(bb_period).std()
    bb_upper = bb_sma + bb_mult * bb_std
    bb_lower = bb_sma - bb_mult * bb_std

    # ── Keltner Channels ─────────────────────────────────────────
    kc_ema = close.ewm(span=kc_period, adjust=False).mean()
    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(kc_period).mean()
    kc_upper = kc_ema + kc_mult * atr
    kc_lower = kc_ema - kc_mult * atr

    # ── Squeeze Detection ────────────────────────────────────────
    result["sqz_on"] = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    result["sqz_off"] = ~result["sqz_on"]

    # Squeeze just released: was on previous bar, off this bar
    result["sqz_releasing"] = result["sqz_off"] & result["sqz_on"].shift(1).fillna(False)

    # ── Momentum (Linear Regression) ─────────────────────────────
    # Midline of highest high and lowest low
    highest = high.rolling(mom_period).max()
    lowest = low.rolling(mom_period).min()
    midline = (highest + lowest) / 2

    # Average of midline and BB SMA
    avg_ml = (midline + bb_sma) / 2

    # Momentum = close - avg_ml, smoothed with linear regression
    delta = close - avg_ml

    # Simple linear regression value (slope * period/2 approximation)
    # Using rolling linregress for accuracy
    result["sqz_momentum"] = _linreg(delta, mom_period)

    # Momentum direction
    result["sqz_momentum_increasing"] = result["sqz_momentum"] > result["sqz_momentum"].shift(1)

    return result


def _linreg(series: pd.Series, period: int) -> pd.Series:
    """Calculate linear regression value (endpoint) over rolling window."""
    values = []
    arr = series.values
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    for i in range(len(arr)):
        if i < period - 1 or np.isnan(arr[max(0, i - period + 1):i + 1]).any():
            values.append(np.nan)
        else:
            y = arr[i - period + 1:i + 1]
            y_mean = y.mean()
            slope = ((x - x_mean) * (y - y_mean)).sum() / x_var if x_var != 0 else 0
            intercept = y_mean - slope * x_mean
            values.append(intercept + slope * (period - 1))

    return pd.Series(values, index=series.index)


def get_squeeze_signal(df: pd.DataFrame) -> dict:
    """
    Get the current Squeeze Momentum signal from the latest data.

    Returns:
        Dictionary with current signal state:
        - squeeze_on: True if squeeze is currently active
        - squeeze_releasing: True if squeeze just released this bar
        - momentum: Current momentum value
        - momentum_bullish: True if momentum > 0
        - momentum_increasing: True if momentum is increasing
        - bars_in_squeeze: How many consecutive bars the squeeze has been on
    """
    if df.empty or "sqz_on" not in df.columns:
        return {
            "squeeze_on": False,
            "squeeze_releasing": False,
            "momentum": None,
            "momentum_bullish": False,
            "momentum_increasing": False,
            "bars_in_squeeze": 0,
        }

    latest = df.iloc[-1]
    momentum = latest.get("sqz_momentum", np.nan)

    # Count consecutive squeeze bars
    bars_in_squeeze = 0
    if "sqz_on" in df.columns:
        sqz_col = df["sqz_on"].values
        for i in range(len(sqz_col) - 1, -1, -1):
            if sqz_col[i]:
                bars_in_squeeze += 1
            else:
                break

    if pd.isna(momentum):
        return {
            "squeeze_on": False,
            "squeeze_releasing": False,
            "momentum": None,
            "momentum_bullish": False,
            "momentum_increasing": False,
            "bars_in_squeeze": bars_in_squeeze,
        }

    return {
        "squeeze_on": bool(latest.get("sqz_on", False)),
        "squeeze_releasing": bool(latest.get("sqz_releasing", False)),
        "momentum": float(momentum),
        "momentum_bullish": float(momentum) > 0,
        "momentum_increasing": bool(latest.get("sqz_momentum_increasing", False)),
        "bars_in_squeeze": bars_in_squeeze,
    }
