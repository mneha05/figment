"""The Bayesian market maker — Figment's flagship agent.

The agent maintains a posterior over the goal suit, blends in the order flow it
observes (aggressive buying of a suit is weak evidence that suit scores), and
quotes a two-sided market around each suit's fair value.  Quoting follows an
Avellaneda-Stoikov-style rule: the mid is skewed against inventory so the agent
naturally offloads risk, and the half-spread widens with belief uncertainty and
narrows as the closing bell approaches.  When a resting order is mispriced by
more than ``take_edge``, the agent crosses and takes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..belief import EXPECTED_GOAL_CARDS, belief_entropy, posterior_goal
from ..cards import Suit
from ..engine import GameConfig, Observation, Order


@dataclass
class MMParams:
    """Tunable knobs for the market maker (targets of the evolutionary tuner)."""

    base_spread: float = 2.5      # baseline half-spread, in dollars
    risk_aversion: float = 1.1    # inventory skew per excess card
    uncertainty_weight: float = 1.6  # extra half-spread per bit of belief entropy
    signal_weight: float = 0.35   # how strongly order flow tilts the belief
    take_edge: float = 2.0        # edge required to cross the spread and take
    time_pressure: float = 0.6    # fraction the spread tightens by the close

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.base_spread, self.risk_aversion, self.uncertainty_weight,
            self.signal_weight, self.take_edge, self.time_pressure,
        ])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "MMParams":
        return cls(*[float(x) for x in v])

    @staticmethod
    def bounds() -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
        hi = np.array([8.0, 4.0, 5.0, 2.0, 8.0, 0.95])
        return lo, hi


class BayesianMarketMaker:
    name = "BayesianMM"

    def __init__(self, params: MMParams | None = None):
        self.params = params or MMParams()

    def reset(self, player: int, hand: np.ndarray, config: GameConfig) -> None:
        self.me = player
        self.hand0 = hand.copy()
        self.config = config
        self.pot = config.pot
        self.target = hand.sum() / 4.0  # a balanced pile holds hand_size/4 per suit
        self.prior_goal = posterior_goal(hand)

    # ---- valuation ---------------------------------------------------------
    def _belief(self, obs: Observation) -> np.ndarray:
        """Hand posterior tilted by observed aggressor order flow."""
        flow = obs.net_bought.astype(float) - obs.net_sold.astype(float)
        if np.any(flow):
            flow = flow / (np.abs(flow).max() + 1e-9)
        tilt = np.exp(self.params.signal_weight * flow)
        p = self.prior_goal * tilt
        return p / p.sum()

    def _values(self, p_goal: np.ndarray) -> np.ndarray:
        return p_goal * (self.pot / EXPECTED_GOAL_CARDS)

    # ---- policy ------------------------------------------------------------
    def act(self, obs: Observation) -> Order | None:
        p_goal = self._belief(obs)
        values = self._values(p_goal)
        entropy = belief_entropy(p_goal)
        progress = obs.step / max(1, obs.total_steps)
        half_spread = self.params.base_spread + self.params.uncertainty_weight * entropy
        half_spread *= (1.0 - self.params.time_pressure * progress)
        half_spread = max(1.0, half_spread)

        # 1) Take the single most-mispriced resting order, if any clears take_edge.
        best_take: tuple[float, Order] | None = None
        for s in range(4):
            v = values[s]
            ask = obs.best_ask[s]
            if ask is not None and ask.player != obs.me and obs.cash >= ask.price:
                edge = v - ask.price
                if edge >= self.params.take_edge and (best_take is None or edge > best_take[0]):
                    best_take = (edge, Order(obs.me, Suit(s), ask.price, is_buy=True))
            bid = obs.best_bid[s]
            if bid is not None and bid.player != obs.me and obs.hand[s] >= 1:
                edge = bid.price - v
                if edge >= self.params.take_edge and (best_take is None or edge > best_take[0]):
                    best_take = (edge, Order(obs.me, Suit(s), bid.price, is_buy=False))
        if best_take is not None:
            return best_take[1]

        # 2) Otherwise refresh a two-sided quote on one suit (round-robin by step).
        s = obs.step % 4
        v = values[s]
        excess = obs.hand[s] - self.target
        reservation = v - self.params.risk_aversion * excess

        # Offload when long, accumulate when short and the suit looks like the goal.
        if excess > 0 and obs.hand[s] >= 1:
            ask_price = max(1, int(np.ceil(reservation + half_spread)))
            return Order(obs.me, Suit(s), ask_price, is_buy=False)

        bid_price = max(1, int(np.floor(reservation - half_spread)))
        if v > 1.0 and obs.cash >= bid_price:
            return Order(obs.me, Suit(s), bid_price, is_buy=True)
        if obs.hand[s] >= 1:
            ask_price = max(1, int(np.ceil(reservation + half_spread)))
            return Order(obs.me, Suit(s), ask_price, is_buy=False)
        return None
