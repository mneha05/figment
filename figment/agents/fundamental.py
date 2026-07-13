"""A value investor.

The fundamental agent computes each suit's risk-neutral fair value from its
private hand (via the hypergeometric posterior) and trades only when the market
offers a clear edge: it lifts asks below value and hits bids above value.  It
provides passive quotes a fixed margin away from value but never chases.
"""

from __future__ import annotations

import numpy as np

from ..belief import card_values, posterior_goal
from ..cards import Suit
from ..engine import GameConfig, Observation, Order


class FundamentalAgent:
    name = "Fundamental"

    def __init__(self, edge: float = 3.0):
        self.edge = edge

    def reset(self, player: int, hand: np.ndarray, config: GameConfig) -> None:
        self.me = player
        self.pot = config.pot
        self.values = card_values(posterior_goal(hand), self.pot)

    def act(self, obs: Observation) -> Order | None:
        for s in range(4):
            v = self.values[s]
            ask = obs.best_ask[s]
            if ask is not None and ask.player != obs.me and ask.price <= v - self.edge \
                    and obs.cash >= ask.price:
                return Order(obs.me, Suit(s), ask.price, is_buy=True)  # buy cheap
            bid = obs.best_bid[s]
            if bid is not None and bid.player != obs.me and bid.price >= v + self.edge \
                    and obs.hand[s] >= 1:
                return Order(obs.me, Suit(s), bid.price, is_buy=False)  # sell rich

        # Otherwise, rest a passive quote on the most valuable suit we can price.
        s = int(np.argmax(self.values))
        v = self.values[s]
        if obs.hand[s] >= 1 and self.rng_bit(s, obs.step):
            return Order(obs.me, Suit(s), max(1, round(v + self.edge)), is_buy=False)
        bid_price = max(1, round(v - self.edge))
        if obs.cash >= bid_price:
            return Order(obs.me, Suit(s), bid_price, is_buy=True)
        return None

    @staticmethod
    def rng_bit(s: int, step: int) -> bool:
        return (step + s) % 2 == 0
