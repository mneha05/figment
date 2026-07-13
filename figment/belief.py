"""Bayesian inference over the hidden goal suit.

The deck assigns the sizes ``{12, 10, 10, 8}`` to the four suits.  There are
exactly ``4 x 3 = 12`` equally likely assignments (pick which suit is the 12,
then which is the 8; the remaining two are 10s).  Conditioned on an assignment
``n = (n_0, n_1, n_2, n_3)``, the probability of observing a private hand with
suit counts ``h`` is the multivariate hypergeometric likelihood

    P(h | n)  =  [ prod_s C(n_s, h_s) ] / C(40, |h|),

and the denominator is constant across assignments, so the posterior is simply
proportional to ``prod_s C(n_s, h_s)``.  The goal suit is the same-colour
partner of the 12-card suit, which turns a posterior over "which suit is common"
into a posterior over "which suit scores".
"""

from __future__ import annotations

from itertools import permutations
from math import comb

import numpy as np

from .cards import DECK_SHAPE, SUITS, Suit

# All 12 distinct suit-size assignments, as tuples indexed by suit.
_ASSIGNMENTS: tuple[tuple[int, int, int, int], ...] = tuple(
    dict.fromkeys(permutations(DECK_SHAPE))  # preserves order, drops duplicate 10/10 perms
)

# Expected number of goal cards. The common suit's partner is, uniformly over
# the three non-common suits {8, 10, 10}, the 8-suit w.p. 1/3 and a 10-suit
# w.p. 2/3, giving E[goal] = 8/3 + 20/3 = 28/3 ~= 9.33.
EXPECTED_GOAL_CARDS: float = 8 / 3 + 20 / 3


def posterior_common(hand: np.ndarray, extra_counts: np.ndarray | None = None) -> np.ndarray:
    """Posterior probability that each suit is the *common* (12-card) suit.

    ``hand`` is a length-4 vector of privately known suit counts.  ``extra_counts``
    optionally folds in additional observed cards (e.g. cards seen changing hands),
    which sharpens the estimate of the true suit populations.
    """
    observed = hand.astype(int)
    if extra_counts is not None:
        observed = observed + extra_counts.astype(int)

    weights = np.zeros(len(_ASSIGNMENTS))
    for i, n in enumerate(_ASSIGNMENTS):
        # Unnormalised likelihood prod_s C(n_s, h_s); zero if a suit is impossible.
        w = 1.0
        for s in range(4):
            if observed[s] > n[s]:
                w = 0.0
                break
            w *= comb(n[s], observed[s])
        weights[i] = w

    total = weights.sum()
    if total == 0:  # numerically impossible hand -> fall back to uniform
        weights[:] = 1.0
        total = weights.sum()
    weights /= total

    p_common = np.zeros(4)
    for i, n in enumerate(_ASSIGNMENTS):
        common = int(np.argmax(n))  # the 12 in this assignment
        p_common[common] += weights[i]
    return p_common


def posterior_goal(hand: np.ndarray, extra_counts: np.ndarray | None = None) -> np.ndarray:
    """Posterior probability that each suit is the *goal* suit.

    ``goal = partner(common)``, so ``P(goal = k) = P(common = partner(k))``.
    """
    p_common = posterior_common(hand, extra_counts)
    p_goal = np.zeros(4)
    for s in SUITS:
        p_goal[s] = p_common[Suit(s).partner]
    return p_goal


def belief_entropy(p_goal: np.ndarray) -> float:
    """Shannon entropy (bits) of the goal-suit posterior; 2.0 = maximal doubt."""
    p = np.clip(p_goal, 1e-12, 1.0)
    return float(-(p * np.log2(p)).sum())


def card_values(p_goal: np.ndarray, pot: float) -> np.ndarray:
    """Risk-neutral fair value of one card of each suit.

    A goal card is worth, on average, ``pot / E[goal cards]`` once the majority
    bonus is amortised across the winning pile.  A card of suit ``s`` is a goal
    card with probability ``P(goal = s)``, so its expected value is that share of
    the per-goal-card fair value.
    """
    fair_per_goal_card = pot / EXPECTED_GOAL_CARDS
    return p_goal * fair_per_goal_card
