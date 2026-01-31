"""Stock universe management - Top 100 US stocks by market cap."""

from typing import List, Optional

# Top 100 US stocks by market cap (as of 2024)
# This list can be updated periodically or fetched dynamically
TOP_100_STOCKS = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "ADBE",
    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "C",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT", "LOW",
    # Industrial
    "CAT", "GE", "RTX", "HON", "UPS", "BA", "LMT", "DE", "MMM", "UNP",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "KMI",
    # Tech (more)
    "CRM", "AMD", "INTC", "QCOM", "TXN", "IBM", "NOW", "INTU", "AMAT", "MU",
    # Communication
    "DIS", "NFLX", "CMCSA", "VZ", "T", "TMUS", "CHTR", "EA", "ATVI", "WBD",
    # Real Estate & Utilities
    "AMT", "PLD", "CCI", "EQIX", "PSA", "NEE", "DUK", "SO", "D", "AEP",
    # Other
    "SPGI", "BLK", "CB", "MMC", "PNC", "USB", "SCHW", "TFC", "COF", "AIG",
]


def get_universe() -> List[str]:
    """Return the list of stock tickers to scan."""
    return TOP_100_STOCKS.copy()


def get_universe_subset(start: int = 0, end: Optional[int] = None) -> List[str]:
    """Return a subset of the universe for batched processing."""
    return TOP_100_STOCKS[start:end]


if __name__ == "__main__":
    print(f"Universe contains {len(get_universe())} stocks")
    print(get_universe()[:10])
