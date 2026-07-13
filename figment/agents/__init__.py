"""Trading agents for the Figment arena."""

from .adaptive import AdaptiveMarketMaker, MMParams
from .base import Agent
from .fundamental import FundamentalAgent
from .market_maker import BayesianMarketMaker
from .momentum import MomentumAgent
from .random_agent import RandomAgent

__all__ = [
    "Agent",
    "RandomAgent",
    "FundamentalAgent",
    "MomentumAgent",
    "BayesianMarketMaker",
    "AdaptiveMarketMaker",
    "MMParams",
]
