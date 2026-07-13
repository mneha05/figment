"""Show how the goal-suit posterior sharpens as a hand fills with one suit.

    python examples/belief_demo.py
"""

import numpy as np

from figment.belief import belief_entropy, posterior_goal
from figment.cards import SUITS


def main() -> None:
    print("Dealing spades one at a time and watching the posterior move:\n")
    print(f"{'hand':<22}{'  '.join(s.glyph for s in SUITS)}     entropy(bits)")
    print("-" * 60)
    for extra in range(0, 9, 2):
        hand = np.array([extra, 1, 1, 1])  # more and more spades
        p = posterior_goal(hand)
        bars = "  ".join(f"{x:0.2f}" for x in p)
        label = f"{extra}x spades, 1 each"
        print(f"{label:<22}{bars}     {belief_entropy(p):0.2f}")

    print("\nNote: as spades pile up, the model bets the GOAL is clubs (spades'")
    print("same-colour partner) — because spades is probably the 12-card suit.")


if __name__ == "__main__":
    main()
