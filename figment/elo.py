"""Multiplayer Elo ratings.

Elo is defined for pairwise games; we extend it to an N-player free-for-all by
treating each game as all C(N, 2) pairwise duels, where the player with the
higher profit "beats" the other (ties score 0.5).  Every player's rating is
nudged by the average of their pairwise surprises.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


class EloBook:
    def __init__(self, names: list[str], k: float = 24.0, base: float = 1500.0):
        self.names = names
        self.k = k
        self.rating = {n: base for n in names}
        self.games = {n: 0 for n in names}

    @staticmethod
    def _expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update(self, names: list[str], profits: np.ndarray) -> None:
        deltas = {n: 0.0 for n in names}
        for i, j in combinations(range(len(names)), 2):
            a, b = names[i], names[j]
            if profits[i] > profits[j]:
                sa = 1.0
            elif profits[i] < profits[j]:
                sa = 0.0
            else:
                sa = 0.5
            ea = self._expected(self.rating[a], self.rating[b])
            deltas[a] += self.k * (sa - ea)
            deltas[b] += self.k * ((1 - sa) - (1 - ea))
        for n in names:
            self.rating[n] += deltas[n]
            self.games[n] += 1

    def table(self) -> list[tuple[str, float, int]]:
        rows = [(n, self.rating[n], self.games[n]) for n in self.names]
        return sorted(rows, key=lambda r: r[1], reverse=True)
