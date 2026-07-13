# The agents

Every agent implements one method — `act(observation) -> Order | None` — and sees
only public state plus its own hand. None of them can see the goal suit, the deck,
or anyone else's cards.

| Agent | One-liner | Edge |
|---|---|---|
| `RandomAgent` | quotes and lifts at random | none — it is the liquidity/noise |
| `FundamentalAgent` | values cards from its hand, trades only on a clear edge | patient value capture |
| `MomentumAgent` | chases aggressive order flow | none — a foil that overpays |
| `BayesianMarketMaker` | belief + EV + inventory-aware quoting | the disciplined maker |
| `AdaptiveMarketMaker` | same maker, parameters learned by evolution | tuned by self-play |

## BayesianMarketMaker

The flagship. Each turn it:

1. **Updates belief.** Starts from the hypergeometric posterior over the goal suit
   (see [`theory.md`](theory.md)) and tilts it by observed order flow — aggressive
   buying of a suit is weak evidence that suit scores.
2. **Prices.** Converts belief into a fair value $v_s$ per suit.
3. **Quotes (Avellaneda–Stoikov style).** Sets a reservation price skewed against
   inventory, $r_s = v_s - \gamma(q_s - \bar q)$, and a half-spread that widens
   with belief entropy $H(p)$ and narrows into the close:
   $\delta = (\delta_0 + \lambda H(p))(1 - \tau\,t/T)$. It rests a bid at
   $r_s - \delta$ and an ask at $r_s + \delta$.
4. **Takes.** Crosses to lift any resting order mispriced by more than its edge
   threshold.

### Parameters (`MMParams`)

| Field | Meaning |
|---|---|
| `base_spread` | baseline half-spread in dollars |
| `risk_aversion` | inventory skew per excess card ($\gamma$) |
| `uncertainty_weight` | extra half-spread per bit of entropy ($\lambda$) |
| `signal_weight` | how strongly order flow tilts the belief |
| `take_edge` | edge required to cross and take |
| `time_pressure` | how much the spread tightens by the close ($\tau$) |

These are exactly the knobs the [evolutionary tuner](../figment/evolve.py) learns.
