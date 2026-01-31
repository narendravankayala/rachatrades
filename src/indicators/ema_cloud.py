"""EMA Cloud indicator - 5/12 EMA with cloud visualization logic."""

import pandas as pd
import numpy as np


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        series: Price series (typically Close prices)
        period: EMA period

    Returns:
        EMA values as pandas Series
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_ema_cloud(
    df: pd.DataFrame,
    fast_period: int = 5,
    slow_period: int = 12,
) -> pd.DataFrame:
    """
    Calculate EMA Cloud indicator.

    The cloud is formed between fast EMA and slow EMA.
    - Bullish: Fast EMA > Slow EMA (price trending up)
    - Bearish: Fast EMA < Slow EMA (price trending down)

    Args:
        df: DataFrame with OHLCV data (must have 'Close' column)
        fast_period: Fast EMA period (default 5)
        slow_period: Slow EMA period (default 12)

    Returns:
        DataFrame with additional columns:
        - ema_fast: Fast EMA values
        - ema_slow: Slow EMA values
        - cloud_bullish: True when fast > slow
        - cloud_bearish: True when fast < slow
        - price_above_cloud: True when close > max(fast, slow)
        - price_below_cloud: True when close < min(fast, slow)
        - cloud_crossover_up: True on bullish crossover (fast crosses above slow)
        - cloud_crossover_down: True on bearish crossover (fast crosses below slow)
    """
    result = df.copy()

    # Calculate EMAs
    result["ema_fast"] = calculate_ema(df["Close"], fast_period)
    result["ema_slow"] = calculate_ema(df["Close"], slow_period)

    # Cloud direction
    result["cloud_bullish"] = result["ema_fast"] > result["ema_slow"]
    result["cloud_bearish"] = result["ema_fast"] < result["ema_slow"]

    # Price position relative to cloud
    cloud_top = result[["ema_fast", "ema_slow"]].max(axis=1)
    cloud_bottom = result[["ema_fast", "ema_slow"]].min(axis=1)

    result["price_above_cloud"] = result["Close"] > cloud_top
    result["price_below_cloud"] = result["Close"] < cloud_bottom
    result["price_in_cloud"] = (result["Close"] >= cloud_bottom) & (
        result["Close"] <= cloud_top
    )

    # Crossovers (signal changes)
    result["cloud_crossover_up"] = (result["cloud_bullish"]) & (
        ~result["cloud_bullish"].shift(1).fillna(False)
    )
    result["cloud_crossover_down"] = (result["cloud_bearish"]) & (
        ~result["cloud_bearish"].shift(1).fillna(False)
    )

    return result


def get_ema_cloud_signal(df: pd.DataFrame) -> dict:
    """
    Get the current EMA Cloud signal from the latest data.

    Args:
        df: DataFrame with EMA Cloud calculations (from calculate_ema_cloud)

    Returns:
        Dictionary with current signal state:
        - bullish: True if cloud is bullish
        - price_above: True if price is above cloud
        - crossover_up: True if just crossed up
        - crossover_down: True if just crossed down
        - ema_fast: Current fast EMA value
        - ema_slow: Current slow EMA value
    """
    if df.empty:
        return {
            "bullish": False,
            "price_above": False,
            "crossover_up": False,
            "crossover_down": False,
            "ema_fast": None,
            "ema_slow": None,
        }

    latest = df.iloc[-1]

    return {
        "bullish": bool(latest.get("cloud_bullish", False)),
        "price_above": bool(latest.get("price_above_cloud", False)),
        "crossover_up": bool(latest.get("cloud_crossover_up", False)),
        "crossover_down": bool(latest.get("cloud_crossover_down", False)),
        "ema_fast": float(latest["ema_fast"]) if pd.notna(latest["ema_fast"]) else None,
        "ema_slow": float(latest["ema_slow"]) if pd.notna(latest["ema_slow"]) else None,
    }


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf

    df = yf.Ticker("AAPL").history(period="5d", interval="10m")
    df_with_cloud = calculate_ema_cloud(df)
    signal = get_ema_cloud_signal(df_with_cloud)

    print("EMA Cloud Signal for AAPL:")
    print(f"  Bullish Cloud: {signal['bullish']}")
    print(f"  Price Above Cloud: {signal['price_above']}")
    print(f"  EMA Fast (5): {signal['ema_fast']:.2f}")
    print(f"  EMA Slow (12): {signal['ema_slow']:.2f}")
