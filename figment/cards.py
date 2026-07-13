"""Cards, suits, and Figgie deck construction.

A Figgie deck is a 40-card deck assembled from a standard deck by dealing the
four suits a random permutation of the multiset ``{12, 10, 10, 8}``.  The suit
that receives 12 cards is the *common suit*; the **goal suit** is always the
other suit of the same colour as the common suit, and therefore holds 8 or 10
cards.  Whoever ends the round holding the most goal-suit cards wins the pot.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Suit(IntEnum):
    SPADES = 0
    CLUBS = 1
    HEARTS = 2
    DIAMONDS = 3

    @property
    def glyph(self) -> str:
        return {Suit.SPADES: "♠", Suit.CLUBS: "♣", Suit.HEARTS: "♥", Suit.DIAMONDS: "♦"}[self]

    @property
    def is_red(self) -> bool:
        return self in (Suit.HEARTS, Suit.DIAMONDS)

    @property
    def partner(self) -> "Suit":
        """The other suit of the same colour (spades<->clubs, hearts<->diamonds)."""
        return {
            Suit.SPADES: Suit.CLUBS,
            Suit.CLUBS: Suit.SPADES,
            Suit.HEARTS: Suit.DIAMONDS,
            Suit.DIAMONDS: Suit.HEARTS,
        }[self]


SUITS: tuple[Suit, ...] = (Suit.SPADES, Suit.CLUBS, Suit.HEARTS, Suit.DIAMONDS)

# The multiset of suit sizes in every Figgie deck.
DECK_SHAPE: tuple[int, ...] = (12, 10, 10, 8)


@dataclass(frozen=True)
class Deck:
    """A concrete deck realisation for one round."""

    counts: tuple[int, int, int, int]  # cards of each suit, indexed by Suit
    common_suit: Suit                  # the 12-card suit
    goal_suit: Suit                    # scoring suit (partner of the common suit)

    @property
    def goal_count(self) -> int:
        return self.counts[self.goal_suit]


def make_deck(rng: np.random.Generator) -> Deck:
    """Build a random Figgie deck: a permutation of ``{12,10,10,8}`` over suits."""
    sizes = rng.permutation(np.array(DECK_SHAPE))
    counts = tuple(int(x) for x in sizes)
    common_suit = Suit(int(np.argmax(sizes)))  # the unique 12-card suit
    goal_suit = common_suit.partner
    return Deck(counts=counts, common_suit=common_suit, goal_suit=goal_suit)  # type: ignore[arg-type]


def deal(deck: Deck, n_players: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Deal all 40 cards evenly; returns per-player suit-count vectors (len 4)."""
    if 40 % n_players != 0:
        raise ValueError(f"{n_players} players do not divide 40 cards evenly.")
    # Build the physical pile of suit labels, shuffle, and deal round-robin.
    pile = np.repeat(np.arange(4), deck.counts)
    rng.shuffle(pile)
    hands = [np.zeros(4, dtype=int) for _ in range(n_players)]
    for i, suit in enumerate(pile):
        hands[i % n_players][suit] += 1
    return hands
