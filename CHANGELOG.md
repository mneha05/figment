# Changelog

All notable changes are documented here, following
[Keep a Changelog](https://keepachangelog.com/) and
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `MomentumAgent`, an order-flow-chasing foil for the market maker.
- `python -m figment` entry point.
- `figment.presets` with named table configs (`four`, `five`, `blitz`, `deep`).
- `--json` output for `figment tournament`.
- Runnable examples in `examples/` and a benchmark in `scripts/`.
- Theory, agent, and results docs under `docs/`.
- Tests for Elo conservation, reproducibility, and the momentum agent.

## [0.1.0] - 2026-07-13

### Added
- Figgie engine: random deck, dealing, continuous double auction with four order
  books, price-time priority, and exact settlement.
- Bayesian goal-suit inference via the multivariate hypergeometric posterior.
- Agents: `RandomAgent`, `FundamentalAgent`, `BayesianMarketMaker`, `AdaptiveMarketMaker`.
- Arena with seat rotation and multiplayer Elo; evolutionary self-play tuner.
- Dark-themed visualisations and an animated round replay.
- Test suite, CI, and packaging.
