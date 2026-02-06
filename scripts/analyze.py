"""
Analyze current market conditions using Rashemator MTF strategy.

Shows zone classification, pullback opportunities, and oscillator readings.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Ensure project root is on path when script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rachatrades.core.data import DataProvider
from rachatrades.agents.rashemator import EMACloudStrategy, Signal
from rachatrades.scanner import get_universe
from rachatrades.core.indicators import Zone, PullbackType, RallyType


def main():
    provider = DataProvider()
    strategy = EMACloudStrategy()
    
    # Get universe
    tickers = get_universe()
    print(f"Fetching data for {len(tickers)} stocks...")
    print("(1-min data resampled to 10-min candles)")
    print()
    
    # Fetch MTF data
    mtf_data = provider.get_batch_mtf_ohlcv(tickers)
    print(f"Got MTF data for {len(mtf_data)} stocks\n")
    
    # Analyze all stocks
    results = []
    for ticker, data in mtf_data.items():
        try:
            result = strategy.evaluate_mtf(ticker, data, has_open_position=False)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
    
    # Separate by zone
    long_zone = [r for r in results if r.zone == Zone.LONG]
    short_zone = [r for r in results if r.zone == Zone.SHORT]
    flat_zone = [r for r in results if r.zone == Zone.FLAT]
    
    # Print zone summary
    print("=" * 100)
    print("RASHEMATOR ZONE ANALYSIS")
    print("=" * 100)
    print(f"🟢 LONG ZONE:  {len(long_zone):3d} stocks (buy pullbacks)")
    print(f"🔴 SHORT ZONE: {len(short_zone):3d} stocks (sell rips)")
    print(f"⚪ FLAT ZONE:  {len(flat_zone):3d} stocks (NO TRADE)")
    print()
    
    # Print LONG zone stocks with best opportunities
    if long_zone:
        print("=" * 100)
        print("🟢 LONG ZONE - PULLBACK OPPORTUNITIES")
        print("=" * 100)
        print(f"{'Ticker':<8} {'Price':>10} {'Clouds':>10} {'Pullback':>10} {'MFI':>6} {'WR':>6} {'Status':<20}")
        print("-" * 80)
        
        # Sort by pullback presence and oscillator values
        long_sorted = sorted(
            long_zone,
            key=lambda r: (
                -(1 if r.pullback_10m else 0),  # Pullback first
                -(1 if r.oscillator_confirms else 0),  # Oscillator confirms
                r.mfi_value if r.mfi_value else 100,  # Lower MFI better
            )
        )
        
        for r in long_sorted[:30]:
            # Cloud alignment
            align = "✓ aligned" if r.clouds_aligned_10m else "— mixed"
            
            # Pullback type
            pb_type = r.pullback_type.value[:4] if r.pullback_type != PullbackType.NONE else "—"
            
            # MFI and WR
            mfi = f"{r.mfi_value:.0f}" if r.mfi_value else "—"
            wr = f"{r.williams_r_value:.0f}" if r.williams_r_value else "—"
            
            # Status
            status_parts = []
            if r.pullback_10m:
                status_parts.append("PULLBACK")
            if r.oscillator_confirms:
                status_parts.append("OSC✓")
            if r.reclaim_detected:
                status_parts.append("RECLAIM")
            status = " ".join(status_parts) if status_parts else "Waiting"
            
            print(f"{r.ticker:<8} ${r.price:>9.2f} {align:>10} {pb_type:>10} {mfi:>6} {wr:>6} {status:<20}")
    
    # Print active BUY signals
    buy_signals = [r for r in long_zone if r.signal.value == "BUY"]
    if buy_signals:
        print()
        print("=" * 100)
        print("🚀 ACTIVE BUY SIGNALS")
        print("=" * 100)
        for r in buy_signals:
            print(f"  {r.ticker:<8} ${r.price:.2f} - {r.reason}")
    
    # Print stocks closest to buy (pullback active, waiting for reclaim/oscillator)
    print()
    print("=" * 100)
    print("⏳ WATCHLIST - Pullback Active, Waiting for Reclaim + Oscillator")
    print("=" * 100)
    
    watchlist = [
        r for r in long_zone 
        if r.pullback_10m and (not r.oscillator_confirms or not r.reclaim_detected)
    ]
    watchlist_sorted = sorted(watchlist, key=lambda r: r.mfi_value if r.mfi_value else 100)
    
    if watchlist_sorted:
        print(f"{'Ticker':<8} {'Price':>10} {'Pullback':>10} {'Reclaim':>8} {'MFI':>8} {'WR':>8} {'Need':<30}")
        print("-" * 90)
        for r in watchlist_sorted[:15]:
            pb = r.pullback_type.value
            reclaim = "✓" if r.reclaim_detected else "—"
            mfi = f"{r.mfi_value:.1f}" if r.mfi_value else "—"
            wr = f"{r.williams_r_value:.1f}" if r.williams_r_value else "—"
            
            # What's needed
            needs = []
            if not r.reclaim_detected:
                needs.append("Reclaim (close above cloud)")
            if r.mfi_value and r.mfi_value >= 20 and (r.williams_r_value and r.williams_r_value >= -80):
                needs.append(f"MFI<20 or WR<-80")
            need_str = " + ".join(needs) if needs else "Almost ready"
            
            print(f"{r.ticker:<8} ${r.price:>9.2f} {pb:>10} {reclaim:>8} {mfi:>8} {wr:>8} {need_str:<30}")
    else:
        print("No stocks with active pullbacks in LONG zone")
    
    # Print SHORT zone summary
    if short_zone:
        print()
        print("=" * 100)
        print("🔴 SHORT ZONE - RALLY OPPORTUNITIES (sell rips)")
        print("=" * 100)
        print(f"{'Ticker':<8} {'Price':>10} {'Clouds':>10} {'Rally':>10} {'MFI':>6} {'WR':>6} {'Status':<20}")
        print("-" * 80)
        
        # Sort by rally presence and oscillator values
        short_sorted = sorted(
            short_zone,
            key=lambda r: (
                -(1 if r.rally_10m else 0),  # Rally first
                -(1 if r.mfi_overbought or r.williams_r_overbought else 0),  # Overbought confirms
                -(r.mfi_value if r.mfi_value else 0),  # Higher MFI better for shorts
            )
        )
        
        for r in short_sorted[:20]:
            # Cloud bearish alignment
            align = "✓ bearish" if not r.clouds_aligned_10m else "— mixed"
            
            # Rally type
            rally_type = r.rally_type.value[:4] if r.rally_type != RallyType.NONE else "—"
            
            # MFI and WR
            mfi = f"{r.mfi_value:.0f}" if r.mfi_value else "—"
            wr = f"{r.williams_r_value:.0f}" if r.williams_r_value else "—"
            
            # Status
            status_parts = []
            if r.rally_10m:
                status_parts.append("RALLY")
            if r.mfi_overbought or r.williams_r_overbought:
                status_parts.append("OSC✓")
            if r.rejection_detected:
                status_parts.append("REJECT")
            status = " ".join(status_parts) if status_parts else "Waiting"
            
            print(f"{r.ticker:<8} ${r.price:>9.2f} {align:>10} {rally_type:>10} {mfi:>6} {wr:>6} {status:<20}")
    
    # Print active SHORT signals
    short_signals = [r for r in short_zone if r.signal == Signal.SHORT]
    if short_signals:
        print()
        print("=" * 100)
        print("📉 ACTIVE SHORT SIGNALS")
        print("=" * 100)
        for r in short_signals:
            print(f"  {r.ticker:<8} ${r.price:.2f} - {r.reason}")
    
    # Print stocks closest to short (in rally with oscillator close to overbought)
    print()
    print("=" * 100)
    print("⏳ SHORT WATCHLIST - Rally Active, Waiting for Overbought")
    print("=" * 100)
    
    short_watchlist = [
        r for r in short_zone 
        if r.rally_10m and not (r.mfi_overbought or r.williams_r_overbought)
    ]
    short_watchlist_sorted = sorted(short_watchlist, key=lambda r: -(r.mfi_value if r.mfi_value else 0))
    
    if short_watchlist_sorted:
        print(f"{'Ticker':<8} {'Price':>10} {'Rally':>10} {'MFI':>8} {'WR':>8} {'Need':<30}")
        print("-" * 80)
        for r in short_watchlist_sorted[:10]:
            rally = r.rally_type.value
            mfi = f"{r.mfi_value:.1f}" if r.mfi_value else "—"
            wr = f"{r.williams_r_value:.1f}" if r.williams_r_value else "—"
            
            # What's needed
            needs = []
            if r.mfi_value and r.mfi_value <= 80:
                needs.append(f"MFI > 80 (now {r.mfi_value:.0f})")
            if r.williams_r_value and r.williams_r_value <= -20:
                needs.append(f"WR > -20 (now {r.williams_r_value:.0f})")
            need_str = " or ".join(needs) if needs else "?"
            
            print(f"{r.ticker:<8} ${r.price:>9.2f} {rally:>10} {mfi:>8} {wr:>8} {need_str:<30}")
    else:
        print("No stocks with active rallies in SHORT zone")
    
    # Key insight
    print()
    print("=" * 100)
    print("KEY INSIGHT")
    print("=" * 100)
    print("BUY when:   LONG_ZONE + Clouds aligned + Pullback + Reclaim + (MFI<20 OR WR<-80)")
    print("SHORT when: SHORT_ZONE + Rally + Rejection + (MFI>80 OR WR>-20)")
    print("SELL when:  Zone changes OR 5/12 cloud flips OR price < 34/50 (stop-loss)")
    print("STAY OUT:   FLAT zone (between 5/12 and 34/50 clouds)")
    print("NOTE:       All on 10-min candles (14-period oscillators = 140 min lookback)")


if __name__ == "__main__":
    main()
