"""A momentum / order-flow-chasing agent.

Where the market maker forms its own view and fades the crowd, the momentum
agent *follows* it: it buys whichever suit is being aggressively bought (reading
sustained buying as a signal that suit is the goal) and lightens up on suits the
crowd is dumping. It is deliberately naive about value — a useful foil that
shows why chasing flow without a fair-value anchor bleeds money against a
disciplined maker.
"""

from __future__ import annotations

import numpy as np

from ..cards import Suit
from ..engine import GameConfig, Observation, Order


class MomentumAgent:
    name = "Momentum"

    def __init__(self, aggression: float = 1.5):
        self.aggression = aggression

    def reset(self, player: int, hand: np.ndarray, config: GameConfig) -> None:
        self.me = player

    def act(self, obs: Observation) -> Order | None:
        flow = obs.net_bought.astype(float) - obs.net_sold.astype(float)
        hot = int(np.argmax(flow))

        # Chase the hottest suit: lift its ask if one is resting and affordable.
        ask = obs.best_ask[hot]
        if flow[hot] > 0 and ask is not None and ask.player != obs.me and obs.cash >= ask.price:
            return Order(obs.me, Suit(hot), ask.price, is_buy=True)

        # Otherwise, bid a touch below the last print on the hot suit.
        last = obs.last_price[hot]
        if flow[hot] > 0 and last is not None:
            price = max(1, int(round(last - self.aggression)))
            if obs.cash >= price:
                return Order(obs.me, Suit(hot), price, is_buy=True)

        # Dump a suit the crowd is selling, if we are holding it.
        cold = int(np.argmin(flow))
        if flow[cold] < 0 and obs.hand[cold] >= 1:
            bid = obs.best_bid[cold]
            if bid is not None and bid.player != obs.me:
                return Order(obs.me, Suit(cold), bid.price, is_buy=False)
        return None
