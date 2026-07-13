"""Benchmark the full agent field over many games and print a leaderboard.

    python scripts/benchmark.py --games 1000
"""

from __future__ import annotations

import argparse
import time

from figment.agents import (
    BayesianMarketMaker,
    FundamentalAgent,
    MomentumAgent,
    RandomAgent,
)
from figment.arena import run_tournament


def build_field() -> list:
    r = RandomAgent(seed=17)
    r.name = "Random"
    return [BayesianMarketMaker(), FundamentalAgent(), MomentumAgent(), r]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    tr = run_tournament(build_field(), n_games=args.games, seed=args.seed)
    dt = time.perf_counter() - t0

    print(f"\n{args.games} games in {dt:.2f}s  ({args.games / dt:,.0f} games/s)\n")
    print(f"{'agent':<14}{'Elo':>7}{'profit/game':>14}{'goal-win %':>12}")
    print("-" * 47)
    for name, rating, _ in tr.elo:
        i = tr.names.index(name)
        print(f"{name:<14}{rating:>7.0f}{tr.mean_profit[i]:>+14.1f}"
              f"{100 * tr.goal_accuracy[i]:>11.1f}%")


if __name__ == "__main__":
    main()
