"""Trading agents for the Figment arena."""

from .adaptive import AdaptiveMarketMaker, MMParams
from .base import Agent
from .fundamental import FundamentalAgent
from .market_maker import BayesianMarketMaker
from .random_agent import RandomAgent

__all__ = [
    "Agent",
    "RandomAgent",
    "FundamentalAgent",
    "BayesianMarketMaker",
    "AdaptiveMarketMaker",
    "MMParams",
]
