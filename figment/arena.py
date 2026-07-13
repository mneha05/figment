"""The arena: repeated games, seat rotation, and aggregate statistics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .elo import EloBook
from .engine import GameConfig, GameResult, play_game


@dataclass
class TournamentResult:
    names: list[str]
    mean_profit: np.ndarray
    std_profit: np.ndarray
    elo: list[tuple[str, float, int]]
    profit_history: np.ndarray  # (n_games, n_agents) profit per agent per game
    goal_accuracy: np.ndarray   # fraction of games each agent held the most goal cards
    n_games: int


def run_tournament(
    agents: list,
    n_games: int = 200,
    pot: int = 200,
    steps: int = 240,
    seed: int = 0,
    rotate_seats: bool = True,
) -> TournamentResult:
    """Play ``n_games`` games among ``agents``, rotating seats to remove position
    bias, and return per-agent profit statistics plus Elo ratings."""
    names = [a.name for a in agents]
    n = len(agents)
    elo = EloBook(names)
    rng = np.random.default_rng(seed)

    profit_history = np.zeros((n_games, n))
    goal_wins = np.zeros(n)

    for g in range(n_games):
        order = rng.permutation(n) if rotate_seats else np.arange(n)
        seated = [agents[k] for k in order]
        config = GameConfig(n_players=n, pot=pot, steps=steps, seed=seed + 1 + g)
        result: GameResult = play_game(seated, config)

        # Map seat-indexed outcomes back to stable agent indices.
        profits = np.zeros(n)
        top = result.goal_holdings.max()
        winners = result.goal_holdings == top
        for pos, k in enumerate(order):
            profits[k] = result.profits[pos]
            if winners[pos]:
                goal_wins[k] += 1.0 / winners.sum()
        profit_history[g] = profits
        elo.update([names[k] for k in order], result.profits)

    return TournamentResult(
        names=names,
        mean_profit=profit_history.mean(axis=0),
        std_profit=profit_history.std(axis=0),
        elo=elo.table(),
        profit_history=profit_history,
        goal_accuracy=goal_wins / n_games,
        n_games=n_games,
    )
