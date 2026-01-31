"""Data provider using yfinance for 10-minute OHLCV data."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataProvider:
    """Fetches and caches stock data from yfinance."""

    def __init__(self, cache_minutes: int = 5):
        """
        Initialize the data provider.

        Args:
            cache_minutes: Number of minutes to cache data before refetching
        """
        self.cache_minutes = cache_minutes
        self._cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}

    def get_ohlcv(
        self,
        ticker: str,
        interval: str = "15m",
        period: str = "5d",
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single ticker.

        Args:
            ticker: Stock ticker symbol
            interval: Data interval (1m, 5m, 10m, 15m, 30m, 1h, 1d)
            period: How far back to fetch (1d, 5d, 1mo, etc.)
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            Returns None if fetch fails
        """
        cache_key = f"{ticker}_{interval}_{period}"

        # Check cache
        if not force_refresh and cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=self.cache_minutes):
                return cached_data

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None

            # Clean up the dataframe
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df.index = pd.to_datetime(df.index)

            # Cache the result
            self._cache[cache_key] = (datetime.now(), df)

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None

    def get_batch_ohlcv(
        self,
        tickers: List[str],
        interval: str = "15m",
        period: str = "5d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            interval: Data interval
            period: How far back to fetch

        Returns:
            Dictionary mapping ticker -> DataFrame
        """
        results = {}

        # yfinance supports batch downloads
        try:
            # Download all at once for efficiency
            data = yf.download(
                tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                progress=False,
                threads=True,
            )

            if data.empty:
                logger.warning("Batch download returned no data")
                return results

            # Parse the multi-level column DataFrame
            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        # Single ticker doesn't have multi-level columns
                        df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
                    else:
                        df = data[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()

                    df = df.dropna()
                    if not df.empty:
                        results[ticker] = df
                        # Update cache
                        cache_key = f"{ticker}_{interval}_{period}"
                        self._cache[cache_key] = (datetime.now(), df)
                except KeyError:
                    logger.warning(f"No data for {ticker} in batch download")
                    continue

        except Exception as e:
            logger.error(f"Batch download failed: {e}")
            # Fallback to individual fetches
            for ticker in tickers:
                df = self.get_ohlcv(ticker, interval, period)
                if df is not None:
                    results[ticker] = df

        return results

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Get the most recent closing price for a ticker."""
        df = self.get_ohlcv(ticker, interval="1m", period="1d")
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
        return None

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()


# Singleton instance for convenience
_provider: Optional[DataProvider] = None


def get_provider() -> DataProvider:
    """Get the singleton data provider instance."""
    global _provider
    if _provider is None:
        _provider = DataProvider()
    return _provider


if __name__ == "__main__":
    # Test the data provider
    logging.basicConfig(level=logging.INFO)

    provider = DataProvider()
    df = provider.get_ohlcv("AAPL", interval="10m", period="5d")

    if df is not None:
        print(f"Fetched {len(df)} bars for AAPL")
        print(df.tail())
    else:
        print("Failed to fetch data")
