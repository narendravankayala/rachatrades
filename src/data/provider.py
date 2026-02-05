"""Data provider using yfinance for multi-timeframe OHLCV data.

Supports:
- 10-minute bars for signal timeframe
- 1-minute bars for entry timeframe
- Multi-timeframe fetching for Rashemator strategy
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class MTFData:
    """Multi-timeframe data container."""
    df_10min: Optional[pd.DataFrame] = None
    df_1min: Optional[pd.DataFrame] = None
    ticker: str = ""


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

    def get_mtf_ohlcv(
        self,
        ticker: str,
        force_refresh: bool = False,
    ) -> MTFData:
        """
        Fetch multi-timeframe OHLCV data for Rashemator strategy.
        
        Fetches both 15-min (signal) and 1-min (entry) data for the same ticker.
        Note: yfinance doesn't support 10m, so we use 15m as the signal timeframe.
        
        Args:
            ticker: Stock ticker symbol
            force_refresh: Bypass cache and fetch fresh data
            
        Returns:
            MTFData with df_10min (actually 15-min) and df_1min DataFrames
        """
        # 15-min: 5d period for signal timeframe (using 15m since 10m not supported)
        df_10min = self.get_ohlcv(
            ticker, 
            interval="15m", 
            period="5d",
            force_refresh=force_refresh,
        )
        
        # 1-min: 7d period for 500 EMA stability
        df_1min = self.get_ohlcv(
            ticker,
            interval="1m",
            period="7d",
            force_refresh=force_refresh,
        )
        
        return MTFData(
            ticker=ticker,
            df_10min=df_10min,
            df_1min=df_1min,
        )

    def get_batch_mtf_ohlcv(
        self,
        tickers: List[str],
    ) -> Dict[str, MTFData]:
        """
        Fetch multi-timeframe data for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary mapping ticker -> MTFData
        """
        results: Dict[str, MTFData] = {}
        
        # Fetch 15-min data in batch (signal timeframe)
        data_10min = self.get_batch_ohlcv(tickers, interval="15m", period="5d")
        
        # Fetch 1-min data in batch (entry timeframe)
        data_1min = self.get_batch_ohlcv(tickers, interval="1m", period="7d")
        
        # Combine into MTFData objects
        for ticker in tickers:
            df_10m = data_10min.get(ticker)
            df_1m = data_1min.get(ticker)
            
            if df_10m is not None or df_1m is not None:
                results[ticker] = MTFData(
                    ticker=ticker,
                    df_10min=df_10m,
                    df_1min=df_1m,
                )
        
        return results

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
    
    # Test single timeframe
    df = provider.get_ohlcv("AAPL", interval="10m", period="5d")
    if df is not None:
        print(f"Fetched {len(df)} 10-min bars for AAPL")
        print(df.tail(3))
    
    # Test multi-timeframe
    print("\n" + "=" * 60)
    print("Testing Multi-Timeframe Data Fetch")
    print("=" * 60)
    
    mtf = provider.get_mtf_ohlcv("AAPL")
    if mtf.df_10min is not None:
        print(f"\n10-min: {len(mtf.df_10min)} bars")
    if mtf.df_1min is not None:
        print(f"1-min:  {len(mtf.df_1min)} bars")
        print(f"\nLatest 1-min bar:")
        print(mtf.df_1min.tail(1))
