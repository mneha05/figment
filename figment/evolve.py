"""Evolutionary self-play: learn market-making parameters from scratch.

A population of :class:`MMParams` genomes competes in Figgie.  Each genome is a
full market-making policy; its fitness is the average profit it earns when
seated against a fixed benchmark field — a value trader, an untuned market
maker, and a noise trader.  The fittest genomes survive and reproduce with
Gaussian mutation, so the population climbs the profit landscape generation by
generation.  The elite of the final generation is the learned strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agents.adaptive import AdaptiveMarketMaker
from .agents.fundamental import FundamentalAgent
from .agents.market_maker import BayesianMarketMaker, MMParams
from .agents.random_agent import RandomAgent
from .engine import GameConfig, play_game


@dataclass
class Generation:
    gen: int
    best_fitness: float
    mean_fitness: float
    best_params: MMParams


def _benchmark_field() -> list:
    """A fixed panel the evolving genome must learn to beat."""
    r = RandomAgent(seed=99)
    r.name = "Random"
    return [FundamentalAgent(), BayesianMarketMaker(), r]


def _evaluate(candidate: MMParams, games: int, seed: int, pot: int, steps: int) -> float:
    """Mean profit of ``candidate`` seated against the fixed benchmark field."""
    cand = AdaptiveMarketMaker(candidate)
    cand.name = "Candidate"
    agents = [cand, *_benchmark_field()]
    rng = np.random.default_rng(seed)
    total = 0.0
    for g in range(games):
        order = rng.permutation(len(agents))
        seated = [agents[k] for k in order]
        result = play_game(seated, GameConfig(n_players=len(agents), pot=pot,
                                              steps=steps, seed=seed + 1 + g))
        pos = int(np.flatnonzero(order == 0)[0])  # candidate's seat this game
        total += result.profits[pos]
    return total / games


def evolve(
    generations: int = 12,
    pop_size: int = 16,
    games_per_eval: int = 24,
    elite_frac: float = 0.25,
    mutation: float = 0.18,
    seed: int = 0,
    pot: int = 200,
    steps: int = 240,
) -> tuple[MMParams, list[Generation]]:
    """Run the tuner and return the best genome plus a per-generation history."""
    rng = np.random.default_rng(seed)
    lo, hi = MMParams.bounds()
    dim = lo.size

    # Seed the population in a deliberately *timid* corner of parameter space —
    # very wide spreads and a huge take-edge, so the maker barely trades and
    # barely profits. This leaves clear headroom, making the climb legible: the
    # optimiser has to discover that tighter, more aggressive quoting pays.
    timid = np.array([7.5, 3.2, 0.3, 0.0, 7.5, 0.9])
    pop = np.clip(timid + rng.normal(0, 0.08, size=(pop_size, dim)) * (hi - lo), lo, hi)
    history: list[Generation] = []
    n_elite = max(1, int(elite_frac * pop_size))
    best_ever: tuple[float, np.ndarray] | None = None

    for gen in range(generations):
        # Shared eval seed per generation keeps genome comparisons fair.
        eval_seed = seed + 1000 * (gen + 1)
        fitness = np.array([
            _evaluate(MMParams.from_vector(pop[i]), games_per_eval, eval_seed, pot, steps)
            for i in range(pop_size)
        ])
        order = np.argsort(fitness)[::-1]
        elite = pop[order[:n_elite]]
        best = pop[order[0]]
        if best_ever is None or fitness[order[0]] > best_ever[0]:
            best_ever = (float(fitness[order[0]]), best.copy())

        history.append(Generation(
            gen=gen,
            best_fitness=float(fitness.max()),
            mean_fitness=float(fitness.mean()),
            best_params=MMParams.from_vector(best),
        ))

        # Breed: carry the elite forward, fill the rest with mutated elite children.
        children = [e.copy() for e in elite]
        while len(children) < pop_size:
            parent = elite[rng.integers(n_elite)]
            child = parent + rng.normal(0, mutation, size=dim) * (hi - lo)
            children.append(np.clip(child, lo, hi))
        pop = np.array(children)

    return MMParams.from_vector(best_ever[1]), history
