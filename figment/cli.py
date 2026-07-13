"""Command-line interface: ``figment {play,tournament,evolve,demo}``."""

from __future__ import annotations

import argparse
import sys

import numpy as np

# Suit glyphs (♠♣♥♦) need a UTF-8 stream; Windows consoles default to cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

from .agents import (
    BayesianMarketMaker,
    FundamentalAgent,
    RandomAgent,
)
from .arena import run_tournament
from .cards import SUITS
from .engine import GameConfig, play_game
from .evolve import evolve


def default_lineup() -> list:
    """A representative four-handed table used across the CLI and demos."""
    mm = BayesianMarketMaker()
    fund = FundamentalAgent()
    r1, r2 = RandomAgent(seed=11), RandomAgent(seed=29)
    r1.name, r2.name = "Random-A", "Random-B"
    return [mm, fund, r1, r2]


def cmd_play(args: argparse.Namespace) -> None:
    agents = default_lineup()
    result = play_game(agents, GameConfig(n_players=len(agents), seed=args.seed))
    deck = result.deck
    print(f"\nDeck sizes  : " + "  ".join(
        f"{s.glyph}{deck.counts[int(s)]}" for s in SUITS))
    print(f"Common suit : {deck.common_suit.glyph} {deck.common_suit.name.title()} (12 cards)")
    print(f"GOAL suit   : {deck.goal_suit.glyph} {deck.goal_suit.name.title()} "
          f"({deck.goal_count} cards)\n")
    print(f"{'agent':<14}{'goal cards':>12}{'profit ($)':>14}")
    print("-" * 40)
    for a, g, p in zip(agents, result.goal_holdings, result.profits):
        print(f"{a.name:<14}{g:>12}{p:>14.1f}")
    print(f"\nTrades executed: {len(result.trades)}   (zero-sum check: "
          f"{result.profits.sum():+.2f})")


def cmd_tournament(args: argparse.Namespace) -> None:
    agents = default_lineup()
    tr = run_tournament(agents, n_games=args.games, seed=args.seed)
    print(f"\nFIGMENT leaderboard  ·  {tr.n_games} games\n")
    print(f"{'rank':<5}{'agent':<14}{'Elo':>7}{'profit/game':>14}{'goal-win %':>12}")
    print("-" * 52)
    acc = {n: a for n, a in zip(tr.names, tr.goal_accuracy)}
    for rank, (name, rating, _) in enumerate(tr.elo, 1):
        i = tr.names.index(name)
        print(f"{rank:<5}{name:<14}{rating:>7.0f}{tr.mean_profit[i]:>+14.1f}"
              f"{100 * acc[name]:>11.1f}%")
    print()


def cmd_evolve(args: argparse.Namespace) -> None:
    best, history = evolve(generations=args.generations, seed=args.seed)
    print("\nEvolutionary self-play\n")
    print(f"{'gen':<5}{'best fitness':>14}{'mean fitness':>14}")
    print("-" * 33)
    for h in history:
        print(f"{h.gen:<5}{h.best_fitness:>+14.2f}{h.mean_fitness:>+14.2f}")
    print("\nBest genome (MMParams):")
    for k, v in vars(best).items():
        print(f"  {k:<20}{v:.3f}")


def cmd_demo(args: argparse.Namespace) -> None:
    from .scripts_demo import generate_assets
    generate_assets(args.out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="figment",
                                description="A self-play market-making arena for "
                                            "Figgie-style card markets.")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("play", help="play one round and print the outcome")
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(func=cmd_play)

    st = sub.add_parser("tournament", help="run a multi-game tournament + Elo")
    st.add_argument("--games", type=int, default=300)
    st.add_argument("--seed", type=int, default=0)
    st.set_defaults(func=cmd_tournament)

    se = sub.add_parser("evolve", help="evolve market-maker params via self-play")
    se.add_argument("--generations", type=int, default=12)
    se.add_argument("--seed", type=int, default=0)
    se.set_defaults(func=cmd_evolve)

    sd = sub.add_parser("demo", help="regenerate README assets (charts + GIF)")
    sd.add_argument("--out", type=str, default="assets")
    sd.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
