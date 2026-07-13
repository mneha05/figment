import numpy as np

from figment.cards import DECK_SHAPE, Suit, deal, make_deck


def test_deck_shape_and_goal_rule():
    rng = np.random.default_rng(0)
    for _ in range(200):
        deck = make_deck(rng)
        assert sorted(deck.counts) == sorted(DECK_SHAPE)          # always {12,10,10,8}
        assert sum(deck.counts) == 40
        assert deck.counts[deck.common_suit] == 12                # common suit has 12
        assert deck.goal_suit == deck.common_suit.partner         # goal is same-colour partner
        assert deck.goal_suit.is_red == deck.common_suit.is_red   # same colour
        assert deck.goal_count in (8, 10)                         # goal never has 12


def test_partners_are_involutive():
    for s in Suit:
        assert s.partner.partner == s
        assert s.partner != s


def test_deal_conserves_cards():
    rng = np.random.default_rng(1)
    deck = make_deck(rng)
    for n_players in (4, 5, 8):
        hands = deal(deck, n_players, rng)
        total = np.sum(hands, axis=0)
        assert list(total) == list(deck.counts)                   # every card dealt exactly once
        assert all(h.sum() == 40 // n_players for h in hands)     # even deal
