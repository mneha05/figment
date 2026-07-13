import numpy as np

from figment.agents import BayesianMarketMaker, FundamentalAgent, RandomAgent
from figment.arena import run_tournament
from figment.evolve import evolve


def _lineup():
    mm, fund = BayesianMarketMaker(), FundamentalAgent()
    r1, r2 = RandomAgent(seed=1), RandomAgent(seed=2)
    r1.name, r2.name = "Random-A", "Random-B"
    return [mm, fund, r1, r2]


def test_tournament_runs_and_is_zero_sum():
    tr = run_tournament(_lineup(), n_games=40, seed=0)
    assert len(tr.elo) == 4
    assert tr.profit_history.shape == (40, 4)
    # Profit is zero-sum every single game.
    assert np.allclose(tr.profit_history.sum(axis=1), 0.0, atol=1e-6)
    # Elo is conserved around the 1500 base (multiplayer Elo is zero-sum in points).
    assert abs(sum(r for _, r, _ in tr.elo) - 4 * 1500) < 1.0


def test_market_maker_beats_random_over_many_games():
    tr = run_tournament(_lineup(), n_games=120, seed=1)
    profit = {n: tr.mean_profit[i] for i, n in enumerate(tr.names)}
    # The belief-driven maker should out-earn a coin-flipping noise trader.
    assert profit["BayesianMM"] > profit["Random-A"]
    assert profit["BayesianMM"] > profit["Random-B"]


def test_evolution_runs_and_stays_in_bounds():
    from figment.agents.market_maker import MMParams

    best, history = evolve(generations=5, pop_size=8, games_per_eval=10, seed=0)
    assert len(history) == 5
    assert all(np.isfinite(h.best_fitness) for h in history)
    # The learned genome must respect the parameter bounds.
    lo, hi = MMParams.bounds()
    assert np.all(best.as_vector() >= lo - 1e-9)
    assert np.all(best.as_vector() <= hi + 1e-9)
