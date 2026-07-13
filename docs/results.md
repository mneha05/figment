# Reproducible results

All numbers below regenerate deterministically. Run:

```bash
python scripts/benchmark.py --games 400        # leaderboard
python -m figment.cli tournament --games 400 --json
python -m figment.cli demo                      # charts + replay GIF
```

## Four-agent tournament (400 games, seed 0)

| Agent | Elo | Profit / game | Goal-suit win % |
|---|--:|--:|--:|
| BayesianMM | 1677 | +$57.9 | 20.3% |
| Fundamental | 1617 | +$42.9 | 19.0% |
| Random-A | 1355 | −$50.4 | 30.7% |
| Random-B | 1351 | −$50.4 | 30.0% |

**Read the last two columns together.** The noise traders win the pot *more*
often — they hoard goal cards blindly — yet lose money every game, because they
overpay. The maker wins less often but profits consistently on spread and
mispriced fills. Edge is not the same thing as outcome.

## Evolutionary self-play (14 generations)

Starting from a deliberately timid population (huge spreads, huge take-edge:
**+$6/game**), the champion climbs to **+$37/game**, converging on tight, aggressive
quoting that only widens under genuine uncertainty.

| Parameter | Timid start | Evolved champion |
|---|--:|--:|
| `base_spread` | 7.5 | 2.0 |
| `take_edge` | 7.5 | 0.7 |
| `risk_aversion` | 3.2 | 0.0 |
| `uncertainty_weight` | 0.3 | 1.5 |
| `time_pressure` | 0.9 | 0.7 |
