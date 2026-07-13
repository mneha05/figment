"""Agent protocol.

An agent is any object exposing ``act(observation) -> Order | None``.  It may
optionally implement ``reset(player, hand, config)`` to initialise per-game
state (belief priors, remembered hand, RNG).  Agents never see hidden state:
the goal suit, other players' hands, or the true deck are not in the observation.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..engine import GameConfig, Observation, Order


class Agent(Protocol):
    name: str

    def reset(self, player: int, hand: np.ndarray, config: GameConfig) -> None: ...

    def act(self, obs: Observation) -> Order | None: ...
