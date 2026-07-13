"""Regenerate every asset the README embeds: charts, the animated replay GIF,
and a machine-readable results summary.  Deterministic given the seeds below.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .agents import BayesianMarketMaker, FundamentalAgent, RandomAgent
from .arena import run_tournament
from .cards import SUITS
from .engine import FiggieGame, GameConfig
from .evolve import evolve
from . import viz


def _focal_lineup() -> list:
    mm = BayesianMarketMaker()
    fund = FundamentalAgent()
    r1, r2 = RandomAgent(seed=3), RandomAgent(seed=8)
    r1.name, r2.name = "Random-A", "Random-B"
    return [mm, fund, r1, r2]


def _pick_illustrative_game(seeds, steps=240):
    """Choose a seed where the market maker holds the goal suit and beliefs move —
    it makes the clearest story for the hero animation."""
    best = None
    for seed in seeds:
        agents = _focal_lineup()
        game = FiggieGame(agents, GameConfig(n_players=4, steps=steps, seed=seed))
        result = game.play()
        beliefs = viz.reconstruct_belief(result, 0)
        conviction = beliefs[-1, int(result.deck.goal_suit)]  # final belief in truth
        mm_won = result.goal_holdings[0] == result.goal_holdings.max()
        score = conviction + (0.5 if mm_won else 0) + len(result.trades) / 400
        if best is None or score > best[0]:
            best = (score, seed, result)
    return best[1], best[2]


def generate_assets(out_dir: str = "assets") -> None:
    os.makedirs(out_dir, exist_ok=True)
    names = [a.name for a in _focal_lineup()]

    print("· selecting an illustrative round…")
    seed, result = _pick_illustrative_game(range(1, 60))
    print(f"  chosen seed={seed}  goal={result.deck.goal_suit.name}  "
          f"trades={len(result.trades)}")

    print("· rendering hero replay GIF (this takes a moment)…")
    viz.animate_game(result, focal_player=0, agent_names=names,
                     path=os.path.join(out_dir, "replay.gif"))
    viz.plot_belief_convergence(result, 0, os.path.join(out_dir, "belief.png"))

    print("· running tournament (400 games)…")
    tr = run_tournament(_focal_lineup(), n_games=400, seed=0)
    viz.plot_tournament(tr, os.path.join(out_dir, "leaderboard.png"))

    print("· running evolutionary self-play…")
    best, history = evolve(generations=14, pop_size=20, games_per_eval=40, seed=0)
    viz.plot_evolution(history, os.path.join(out_dir, "evolution.png"))

    summary = {
        "chosen_seed": int(seed),
        "goal_suit": result.deck.goal_suit.name,
        "deck_counts": {s.name: int(result.deck.counts[int(s)]) for s in SUITS},
        "tournament": {
            "n_games": tr.n_games,
            "leaderboard": [
                {"agent": n, "elo": round(r, 0), "mean_profit": round(float(tr.mean_profit[tr.names.index(n)]), 2),
                 "goal_win_pct": round(100 * float(tr.goal_accuracy[tr.names.index(n)]), 1)}
                for n, r, _ in tr.elo
            ],
        },
        "evolution": {
            "generations": len(history),
            "start_fitness": round(history[0].best_fitness, 2),
            "final_fitness": round(history[-1].best_fitness, 2),
            "best_params": {k: round(v, 3) for k, v in vars(best).items()},
        },
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"· wrote assets to {out_dir}/  (replay.gif, belief.png, leaderboard.png, "
          f"evolution.png, results.json)")
    return summary
