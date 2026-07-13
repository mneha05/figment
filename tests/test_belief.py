import numpy as np

from figment.belief import (
    EXPECTED_GOAL_CARDS,
    belief_entropy,
    card_values,
    posterior_common,
    posterior_goal,
)


def test_posteriors_are_distributions():
    rng = np.random.default_rng(0)
    for _ in range(50):
        hand = rng.integers(0, 4, size=4)
        for post in (posterior_common(hand), posterior_goal(hand)):
            assert np.all(post >= 0)
            assert abs(post.sum() - 1.0) < 1e-9


def test_empty_hand_is_uniform():
    # With no private information, every suit is equally likely to score.
    p = posterior_goal(np.zeros(4, dtype=int))
    assert np.allclose(p, 0.25, atol=1e-9)


def test_seeing_many_of_a_suit_implies_it_is_common_not_goal():
    # Holding a lot of spades => spades is probably the 12-card common suit,
    # so the GOAL suit is probably spades' partner (clubs), not spades.
    hand = np.array([8, 0, 1, 1])  # tons of spades
    p_common = posterior_common(hand)
    p_goal = posterior_goal(hand)
    assert np.argmax(p_common) == 0          # spades most likely common
    assert np.argmax(p_goal) == 1            # clubs (spades' partner) most likely goal
    assert p_goal[0] < p_goal[1]


def test_entropy_bounds_and_values():
    uniform = np.full(4, 0.25)
    assert abs(belief_entropy(uniform) - 2.0) < 1e-9        # max uncertainty = 2 bits
    certain = np.array([1.0, 0.0, 0.0, 0.0])
    assert belief_entropy(certain) < 1e-6

    vals = card_values(uniform, pot=200)
    assert np.allclose(vals, (200 / EXPECTED_GOAL_CARDS) * 0.25)
    assert 4.5 < vals[0] < 6.5                               # ~5.36 each under uncertainty
