"""Minimal end-to-end example: play one round and inspect the outcome.

    python examples/quickstart.py
"""

from figment import GameConfig, play_game, posterior_goal
from figment.agents import (
    BayesianMarketMaker,
    FundamentalAgent,
    MomentumAgent,
    RandomAgent,
)


def main() -> None:
    agents = [BayesianMarketMaker(), FundamentalAgent(), MomentumAgent(), RandomAgent(seed=1)]
    result = play_game(agents, GameConfig(seed=7))

    print(f"Goal suit was {result.deck.goal_suit.glyph} {result.deck.goal_suit.name.title()}")
    print(f"Player 0's opening read: {posterior_goal(result.hands_start[0]).round(3)}\n")

    for agent, profit, goal in zip(agents, result.profits, result.goal_holdings):
        print(f"  {agent.name:<14} profit ${profit:+6.1f}   goal cards held: {goal}")

    print(f"\nZero-sum check: {result.profits.sum():+.2f}")


if __name__ == "__main__":
    main()
