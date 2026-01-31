"""Money Flow Index (MFI) indicator."""

import pandas as pd
import numpy as np


def calculate_mfi(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Calculate Money Flow Index (MFI).

    MFI is a volume-weighted RSI that measures buying and selling pressure.
    - MFI < 20: Oversold (potential buy signal)
    - MFI > 80: Overbought (potential sell signal)

    Args:
        df: DataFrame with OHLCV data (must have High, Low, Close, Volume)
        period: MFI calculation period (default 14)

    Returns:
        DataFrame with additional columns:
        - typical_price: (High + Low + Close) / 3
        - money_flow: typical_price * volume
        - mfi: Money Flow Index (0-100)
        - mfi_oversold: True when MFI < 20
        - mfi_overbought: True when MFI > 80
    """
    result = df.copy()

    # Calculate Typical Price
    result["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3

    # Calculate Raw Money Flow
    result["money_flow"] = result["typical_price"] * df["Volume"]

    # Determine if money flow is positive or negative
    result["price_change"] = result["typical_price"].diff()
    result["positive_flow"] = np.where(
        result["price_change"] > 0, result["money_flow"], 0
    )
    result["negative_flow"] = np.where(
        result["price_change"] < 0, result["money_flow"], 0
    )

    # Calculate Money Flow Ratio over the period
    positive_sum = result["positive_flow"].rolling(window=period).sum()
    negative_sum = result["negative_flow"].rolling(window=period).sum()

    # Avoid division by zero
    money_ratio = positive_sum / negative_sum.replace(0, np.nan)

    # Calculate MFI
    result["mfi"] = 100 - (100 / (1 + money_ratio))

    # Signal levels
    result["mfi_oversold"] = result["mfi"] < 20
    result["mfi_overbought"] = result["mfi"] > 80

    # Clean up intermediate columns
    result = result.drop(
        columns=["price_change", "positive_flow", "negative_flow"], errors="ignore"
    )

    return result


def get_mfi_signal(df: pd.DataFrame, oversold_threshold: float = 20, overbought_threshold: float = 80) -> dict:
    """
    Get the current MFI signal from the latest data.

    Args:
        df: DataFrame with MFI calculations (from calculate_mfi)
        oversold_threshold: MFI level considered oversold (default 20)
        overbought_threshold: MFI level considered overbought (default 80)

    Returns:
        Dictionary with current signal state:
        - value: Current MFI value
        - oversold: True if MFI < oversold_threshold
        - overbought: True if MFI > overbought_threshold
        - neutral: True if between thresholds
    """
    if df.empty or "mfi" not in df.columns:
        return {
            "value": None,
            "oversold": False,
            "overbought": False,
            "neutral": True,
        }

    latest_mfi = df["mfi"].iloc[-1]

    if pd.isna(latest_mfi):
        return {
            "value": None,
            "oversold": False,
            "overbought": False,
            "neutral": True,
        }

    return {
        "value": float(latest_mfi),
        "oversold": latest_mfi < oversold_threshold,
        "overbought": latest_mfi > overbought_threshold,
        "neutral": oversold_threshold <= latest_mfi <= overbought_threshold,
    }


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf

    df = yf.Ticker("AAPL").history(period="1mo", interval="10m")
    df_with_mfi = calculate_mfi(df)
    signal = get_mfi_signal(df_with_mfi)

    print("MFI Signal for AAPL:")
    print(f"  MFI Value: {signal['value']:.2f}")
    print(f"  Oversold (<20): {signal['oversold']}")
    print(f"  Overbought (>80): {signal['overbought']}")
