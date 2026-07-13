import numpy as np

from figment.agents import BayesianMarketMaker, FundamentalAgent, RandomAgent
from figment.engine import GameConfig, play_game


def _lineup():
    r1, r2 = RandomAgent(seed=1), RandomAgent(seed=2)
    r1.name, r2.name = "R1", "R2"
    return [BayesianMarketMaker(), FundamentalAgent(), r1, r2]


def test_same_seed_reproduces_the_game():
    a = play_game(_lineup(), GameConfig(seed=123, steps=200))
    b = play_game(_lineup(), GameConfig(seed=123, steps=200))
    assert a.deck.goal_suit == b.deck.goal_suit
    assert np.allclose(a.profits, b.profits)
    assert list(a.goal_holdings) == list(b.goal_holdings)
    assert len(a.trades) == len(b.trades)


def test_different_seeds_generally_differ():
    a = play_game(_lineup(), GameConfig(seed=1, steps=200))
    b = play_game(_lineup(), GameConfig(seed=2, steps=200))
    # Overwhelmingly likely to differ in deck or outcome.
    assert (a.deck.counts != b.deck.counts) or (not np.allclose(a.profits, b.profits))
