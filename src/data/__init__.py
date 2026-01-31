"""Data module for fetching market data."""

from .provider import DataProvider, get_provider

__all__ = ["DataProvider", "get_provider"]
