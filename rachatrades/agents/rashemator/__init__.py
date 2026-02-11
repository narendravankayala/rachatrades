"""Rashemator agent — Versioned EMA Cloud Strategy.

Strategy versions:
  v1 (CloudFlipStrategy)  — Cloud crossover = entry. FROZEN baseline.
  v2 (PullbackStrategy)   — Pullback reclaim = entry. Rash's actual system.

Use get_strategy(config) to create the right version based on config.version.
"""

from .strategy import EMACloudStrategy, StrategyConfig, StrategyResult, Signal, get_strategy
from .v1_cloud_flip import CloudFlipStrategy
from .v2_pullback import PullbackStrategy

__all__ = [
    "get_strategy",
    "EMACloudStrategy",  # backward compat alias
    "CloudFlipStrategy",
    "PullbackStrategy",
    "StrategyConfig",
    "StrategyResult",
    "Signal",
]
