"""Figment — a self-play market-making arena for Figgie-style card markets."""

from .arena import TournamentResult, run_tournament
from .belief import card_values, posterior_common, posterior_goal
from .cards import Deck, Suit, make_deck
from .engine import FiggieGame, GameConfig, GameResult, play_game
from .evolve import evolve

__version__ = "0.1.0"

__all__ = [
    "Suit",
    "Deck",
    "make_deck",
    "GameConfig",
    "GameResult",
    "FiggieGame",
    "play_game",
    "posterior_goal",
    "posterior_common",
    "card_values",
    "run_tournament",
    "TournamentResult",
    "evolve",
    "__version__",
]
