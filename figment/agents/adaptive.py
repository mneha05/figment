"""The adaptive market maker: a :class:`BayesianMarketMaker` whose parameters
are learned by the evolutionary self-play tuner (see :mod:`figment.evolve`).

Behaviourally identical to the Bayesian maker; the distinction is that its
``MMParams`` are treated as an evolvable genome rather than hand-set constants.
"""

from __future__ import annotations

from .market_maker import BayesianMarketMaker, MMParams

__all__ = ["AdaptiveMarketMaker", "MMParams"]


class AdaptiveMarketMaker(BayesianMarketMaker):
    name = "AdaptiveMM"

    def __init__(self, params: MMParams | None = None):
        super().__init__(params)
