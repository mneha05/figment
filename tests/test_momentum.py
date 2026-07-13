import numpy as np

from figment.agents import BayesianMarketMaker, FundamentalAgent, MomentumAgent, RandomAgent
from figment.arena import run_tournament
from figment.engine import GameConfig, play_game


def _lineup():
    a = [BayesianMarketMaker(), MomentumAgent(), FundamentalAgent(), RandomAgent(seed=4)]
    a[3].name = "Random"
    return a


def test_momentum_agent_plays_legally():
    for seed in range(15):
        result = play_game(_lineup(), GameConfig(seed=seed, steps=200))
        # No illegal state: cards conserved, nobody short, profit zero-sum.
        assert list(np.sum(result.hands_end, axis=0)) == list(result.deck.counts)
        assert all(np.all(h >= 0) for h in result.hands_end)
        assert abs(result.profits.sum()) < 1e-6


def test_disciplined_maker_beats_flow_chaser():
    tr = run_tournament(_lineup(), n_games=120, seed=2)
    profit = {n: tr.mean_profit[i] for i, n in enumerate(tr.names)}
    # A fair-value maker should out-earn a naive momentum chaser over the long run.
    assert profit["BayesianMM"] > profit["Momentum"]
