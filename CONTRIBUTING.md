# Contributing to Figment

Thanks for taking a look! Figment is a small, dependency-light research codebase,
and contributions that keep it that way are very welcome.

## Setup

```bash
pip install -e ".[dev]"
pytest -q          # all tests should pass
ruff check .       # lint clean
```

## Ground rules

- **Conserve invariants.** The engine must never create or destroy a card or a
  dollar, and profit must stay zero-sum. If you touch `engine.py`, the tests in
  `tests/test_engine.py` are your contract — extend them, don't weaken them.
- **Keep it deterministic.** Everything is seeded through `np.random.Generator`.
  Avoid global RNG state and wall-clock/`random.random()` calls.
- **Small, focused PRs.** One idea per pull request, with a test that would fail
  without it.
- **Agents are pure policies.** An agent may only read its `Observation`; it must
  not reach into hidden game state.

## Good first issues

- A new agent (a proper RL policy, a Kelly-style sizer, an opponent-modelling maker).
- A belief over opponents' hands, not just the goal suit.
- A terminal UI to play a hand against the bots.

See the roadmap in the [README](README.md) for more.
