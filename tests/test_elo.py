import numpy as np

from figment.elo import EloBook


def test_elo_points_are_conserved():
    book = EloBook(["A", "B", "C", "D"])
    rng = np.random.default_rng(0)
    for _ in range(200):
        book.update(["A", "B", "C", "D"], rng.normal(size=4))
    # Multiplayer Elo is zero-sum in rating points: the total never drifts.
    assert abs(sum(book.rating.values()) - 4 * 1500) < 1e-6


def test_consistent_winner_climbs():
    book = EloBook(["Strong", "Weak"])
    for _ in range(30):
        book.update(["Strong", "Weak"], np.array([10.0, -10.0]))
    assert book.rating["Strong"] > book.rating["Weak"]
    table = book.table()
    assert table[0][0] == "Strong"  # sorted best-first


def test_ties_do_not_move_ratings():
    book = EloBook(["A", "B"])
    book.update(["A", "B"], np.array([5.0, 5.0]))
    assert abs(book.rating["A"] - 1500) < 1e-9
    assert abs(book.rating["B"] - 1500) < 1e-9
