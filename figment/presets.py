"""Named game and table presets, so experiments are reproducible by name."""

from __future__ import annotations

from .engine import GameConfig

# Standard Figgie table shapes (players must divide 40 cards evenly).
FOUR_HANDED = GameConfig(n_players=4, pot=200, steps=240)
FIVE_HANDED = GameConfig(n_players=5, pot=200, steps=300)
BLITZ = GameConfig(n_players=4, pot=200, steps=90)          # short, frantic round
DEEP = GameConfig(n_players=4, pot=200, steps=600)          # long price-discovery

PRESETS: dict[str, GameConfig] = {
    "four": FOUR_HANDED,
    "five": FIVE_HANDED,
    "blitz": BLITZ,
    "deep": DEEP,
}


def get_preset(name: str, seed: int | None = None) -> GameConfig:
    """Return a copy of a named preset with an optional seed applied."""
    base = PRESETS[name]
    return GameConfig(n_players=base.n_players, pot=base.pot,
                      starting_cash=base.starting_cash, steps=base.steps, seed=seed)
