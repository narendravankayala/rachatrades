"""Williams %R indicator."""

import pandas as pd
import numpy as np


def calculate_williams_r(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Calculate Williams %R indicator.

    Williams %R measures overbought/oversold levels on a scale of -100 to 0.
    - Williams %R < -80: Oversold (potential buy signal)
    - Williams %R > -20: Overbought (potential sell signal)

    Args:
        df: DataFrame with OHLCV data (must have High, Low, Close)
        period: Lookback period (default 14)

    Returns:
        DataFrame with additional columns:
        - williams_r: Williams %R value (-100 to 0)
        - williams_r_oversold: True when Williams %R < -80
        - williams_r_overbought: True when Williams %R > -20
    """
    result = df.copy()

    # Calculate highest high and lowest low over the period
    highest_high = df["High"].rolling(window=period).max()
    lowest_low = df["Low"].rolling(window=period).min()

    # Calculate Williams %R
    # Formula: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    denominator = highest_high - lowest_low
    result["williams_r"] = ((highest_high - df["Close"]) / denominator.replace(0, np.nan)) * -100

    # Signal levels
    result["williams_r_oversold"] = result["williams_r"] < -80
    result["williams_r_overbought"] = result["williams_r"] > -20

    return result


def get_williams_r_signal(
    df: pd.DataFrame,
    oversold_threshold: float = -80,
    overbought_threshold: float = -20,
) -> dict:
    """
    Get the current Williams %R signal from the latest data.

    Args:
        df: DataFrame with Williams %R calculations (from calculate_williams_r)
        oversold_threshold: Level considered oversold (default -80)
        overbought_threshold: Level considered overbought (default -20)

    Returns:
        Dictionary with current signal state:
        - value: Current Williams %R value
        - oversold: True if Williams %R < oversold_threshold
        - overbought: True if Williams %R > overbought_threshold
        - neutral: True if between thresholds
    """
    if df.empty or "williams_r" not in df.columns:
        return {
            "value": None,
            "oversold": False,
            "overbought": False,
            "neutral": True,
        }

    latest_wr = df["williams_r"].iloc[-1]

    if pd.isna(latest_wr):
        return {
            "value": None,
            "oversold": False,
            "overbought": False,
            "neutral": True,
        }

    return {
        "value": float(latest_wr),
        "oversold": latest_wr < oversold_threshold,
        "overbought": latest_wr > overbought_threshold,
        "neutral": oversold_threshold <= latest_wr <= overbought_threshold,
    }


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf

    df = yf.Ticker("AAPL").history(period="1mo", interval="10m")
    df_with_wr = calculate_williams_r(df)
    signal = get_williams_r_signal(df_with_wr)

    print("Williams %R Signal for AAPL:")
    print(f"  Williams %R Value: {signal['value']:.2f}")
    print(f"  Oversold (<-80): {signal['oversold']}")
    print(f"  Overbought (>-20): {signal['overbought']}")
