"""Analyze current market conditions and indicator values."""
import warnings
warnings.filterwarnings('ignore')

from src.data import DataProvider
from src.strategies import EMACloudStrategy
from src.scanner import get_universe

def main():
    provider = DataProvider()
    strategy = EMACloudStrategy()
    
    # Get all data
    tickers = get_universe()
    print(f"Fetching data for {len(tickers)} stocks...")
    data = provider.get_batch_ohlcv(tickers, interval="15m", period="5d")
    print(f"Got data for {len(data)} stocks\n")
    
    # Analyze all stocks
    print("=" * 90)
    print("TOP STOCKS - INDICATOR VALUES (Latest 15-min Bar)")
    print("=" * 90)
    print(f"{'Ticker':<8} {'Price':>10} {'EMA':>8} {'Above':>6} {'MFI':>8} {'Will%R':>8} {'Status':<30}")
    print("-" * 90)
    
    # Check major tech stocks first
    majors = ["AAPL", "NVDA", "TSLA", "META", "GOOGL", "AMD", "MSFT", "AMZN"]
    for ticker in majors:
        if ticker not in data:
            continue
        result = strategy.evaluate(ticker, data[ticker], has_open_position=False)
        
        ema = "BULL" if result.ema_cloud_bullish else "BEAR"
        above = "Y" if result.price_above_cloud else "N"
        mfi = f"{result.mfi_value:.1f}" if result.mfi_value else "N/A"
        wr = f"{result.williams_r_value:.1f}" if result.williams_r_value else "N/A"
        
        # Determine status
        conditions_met = []
        if result.ema_cloud_bullish:
            conditions_met.append("EMA✓")
        if result.price_above_cloud:
            conditions_met.append("Above✓")
        if result.mfi_value and result.mfi_value < 20:
            conditions_met.append("MFI✓")
        if result.williams_r_value and result.williams_r_value < -80:
            conditions_met.append("WR✓")
        
        status = " ".join(conditions_met) if conditions_met else "No conditions met"
        
        print(f"{ticker:<8} ${result.price:>9.2f} {ema:>8} {above:>6} {mfi:>8} {wr:>8} {status:<30}")
    
    print("-" * 90)
    print("\nBUY Signal requires ALL 4: EMA✓ + Above✓ + MFI✓ (<20) + WR✓ (<-80)")
    
    # Find closest to buy
    print("\n" + "=" * 90)
    print("STOCKS CLOSEST TO BUY SIGNALS (ranked by conditions met)")
    print("=" * 90)
    
    candidates = []
    for ticker, df in data.items():
        result = strategy.evaluate(ticker, df, has_open_position=False)
        if result.mfi_value and result.williams_r_value:
            score = 0
            if result.ema_cloud_bullish:
                score += 1
            if result.price_above_cloud:
                score += 1
            if result.mfi_value < 30:
                score += 0.5
            if result.mfi_value < 20:
                score += 0.5
            if result.williams_r_value < -60:
                score += 0.5
            if result.williams_r_value < -80:
                score += 0.5
            candidates.append((ticker, result, score))
    
    candidates.sort(key=lambda x: (-x[2], x[1].mfi_value))
    
    print(f"{'Ticker':<8} {'Price':>10} {'EMA':>8} {'Above':>6} {'MFI':>8} {'Will%R':>8} {'Score':>6}")
    print("-" * 90)
    
    for ticker, result, score in candidates[:20]:
        ema = "BULL" if result.ema_cloud_bullish else "BEAR"
        above = "Y" if result.price_above_cloud else "N"
        print(f"{ticker:<8} ${result.price:>9.2f} {ema:>8} {above:>6} {result.mfi_value:>8.1f} {result.williams_r_value:>8.1f} {score:>6.1f}")
    
    # Find any oversold stocks
    print("\n" + "=" * 90)
    print("OVERSOLD STOCKS (MFI < 30 OR Williams %R < -70)")
    print("=" * 90)
    
    oversold = []
    for ticker, df in data.items():
        result = strategy.evaluate(ticker, df, has_open_position=False)
        if result.mfi_value and result.williams_r_value:
            if result.mfi_value < 30 or result.williams_r_value < -70:
                oversold.append((ticker, result))
    
    if oversold:
        print(f"{'Ticker':<8} {'Price':>10} {'EMA':>8} {'Above':>6} {'MFI':>8} {'Will%R':>8}")
        print("-" * 90)
        for ticker, result in sorted(oversold, key=lambda x: x[1].mfi_value):
            ema = "BULL" if result.ema_cloud_bullish else "BEAR"
            above = "Y" if result.price_above_cloud else "N"
            print(f"{ticker:<8} ${result.price:>9.2f} {ema:>8} {above:>6} {result.mfi_value:>8.1f} {result.williams_r_value:>8.1f}")
    else:
        print("No oversold stocks found - market may be in strong uptrend")


if __name__ == "__main__":
    main()
