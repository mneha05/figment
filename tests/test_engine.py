import numpy as np

from figment.agents import BayesianMarketMaker, FundamentalAgent, RandomAgent
from figment.cards import Suit
from figment.engine import FiggieGame, GameConfig, Observation, Order, play_game


def _lineup():
    a = [RandomAgent(seed=i) for i in range(4)]
    for i, x in enumerate(a):
        x.name = f"R{i}"
    return a


def test_cards_and_cash_are_conserved():
    for seed in range(25):
        game = FiggieGame(_lineup(), GameConfig(seed=seed, steps=200))
        start_cash = game.cash.sum()
        result = game.play()
        # Every suit's card count is preserved end to end.
        totals = np.sum(result.hands_end, axis=0)
        assert list(totals) == list(result.deck.counts)
        # Trading only moves cash between players; the total is invariant.
        assert abs(game.cash.sum() - start_cash) < 1e-6
        # No player is ever short a card.
        assert all(np.all(h >= 0) for h in result.hands_end)


def test_profits_are_zero_sum():
    for seed in range(25):
        result = play_game(_lineup(), GameConfig(seed=seed, steps=200))
        assert abs(result.profits.sum()) < 1e-6


def test_settlement_pays_out_the_whole_pot():
    game = FiggieGame(_lineup(), GameConfig(pot=200, seed=1))
    goal = int(game.deck.goal_suit)
    # Hand-craft goal holdings: player 0 holds 5 goal cards, player 1 holds 2, others 0.
    for h in game.hands:
        h[goal] = 0
    game.hands[0][goal] = 5
    game.hands[1][goal] = 2
    payout, holdings = game._settle()
    assert list(holdings) == [5, 2, 0, 0]
    # 7 goal cards x $10 = $70 to holders; majority (player 0) takes the $130 remainder.
    assert abs(payout.sum() - 200) < 1e-9
    assert abs(payout[0] - (50 + 130)) < 1e-9
    assert abs(payout[1] - 20) < 1e-9


def test_tie_splits_the_remainder():
    game = FiggieGame(_lineup(), GameConfig(pot=200, seed=2))
    goal = int(game.deck.goal_suit)
    for h in game.hands:
        h[goal] = 0
    game.hands[0][goal] = 3
    game.hands[1][goal] = 3
    payout, _ = game._settle()
    # 6 goal cards -> $60 to holders, $140 remainder split between the two leaders.
    assert abs(payout[0] - (30 + 70)) < 1e-9
    assert abs(payout[1] - (30 + 70)) < 1e-9


class _Seller:
    """Rests a single ask on spades at price 6, then passes."""

    name = "Seller"

    def __init__(self):
        self._done = False

    def reset(self, player, hand, config):
        self.me = player

    def act(self, obs: Observation):
        if self._done:
            return None
        self._done = True
        return Order(self.me, Suit.SPADES, 6, is_buy=False)


class _Buyer:
    """Waits for a resting spades ask, then crosses with a bid of 9."""

    name = "Buyer"

    def reset(self, player, hand, config):
        self.me = player

    def act(self, obs: Observation):
        if obs.best_ask[int(Suit.SPADES)] is not None:
            return Order(self.me, Suit.SPADES, 9, is_buy=True)
        return None


class _Passer:
    name = "Passer"

    def reset(self, player, hand, config):
        self.me = player

    def act(self, obs: Observation):
        return None


def test_crossing_orders_match_at_the_resting_price():
    # Seller (player 0) rests an ask at 6; the buyer waits for it and lifts at 6.
    seller, buyer = _Seller(), _Buyer()
    game = FiggieGame([seller, buyer, _Passer(), _Passer()], GameConfig(seed=0, steps=12))
    # Guarantee the seller can actually deliver a spade.
    game.hands[0][int(Suit.SPADES)] = max(game.hands[0][int(Suit.SPADES)], 1)
    before = game.hands[0][int(Suit.SPADES)]
    result = game.play()
    tape = [t for t in result.trades if t.suit == Suit.SPADES]
    assert len(tape) == 1
    assert tape[0].price == 6                       # trade at the resting ask, not the bid
    assert tape[0].buyer == 1 and tape[0].seller == 0
    assert game.hands[0][int(Suit.SPADES)] == before - 1
