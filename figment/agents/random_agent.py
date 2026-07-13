"""A noise trader: quotes and lifts at random. The market's raw liquidity."""

from __future__ import annotations

import numpy as np

from ..cards import Suit
from ..engine import GameConfig, Observation, Order


class RandomAgent:
    name = "Random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, player: int, hand: np.ndarray, config: GameConfig) -> None:
        self.me = player

    def act(self, obs: Observation) -> Order | None:
        if self.rng.random() < 0.25:
            return None
        suit = Suit(int(self.rng.integers(4)))
        price = int(self.rng.integers(1, 15))
        is_buy = bool(self.rng.random() < 0.5)
        if not is_buy and obs.hand[int(suit)] < 1:
            is_buy = True  # cannot sell what we do not hold
        if is_buy and obs.cash < price:
            return None
        return Order(player=obs.me, suit=suit, price=price, is_buy=is_buy)
