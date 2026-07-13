# The math behind Figment

This note derives the two equations that drive every agent: the **posterior over
the goal suit** and the **fair value of a card**.

## 1. The deck as a latent variable

A Figgie deck assigns the sizes $\{12, 10, 10, 8\}$ to the four suits. Choosing
which suit gets 12 (4 ways) and which gets 8 (3 ways) fixes the two 10s, so there
are exactly $4 \times 3 = 12$ equally likely size-assignments. Call one such
assignment $n = (n_0, n_1, n_2, n_3)$.

The **goal suit** is the same-colour partner of the 12-card suit, so once we know
which suit is the common (12) suit, the goal suit is determined:

$$\text{goal} = \text{partner}(\text{common}).$$

## 2. Inferring the deck from a private hand

Your hand $h = (h_0, \dots, h_3)$ is a draw *without replacement* from the true
suit populations. Under a hypothesised assignment $n$, the probability of seeing
$h$ is the **multivariate hypergeometric**:

$$
P(h \mid n) = \frac{\prod_s \binom{n_s}{h_s}}{\binom{40}{|h|}}.
$$

The denominator does not depend on $n$, so with a uniform prior over the 12
assignments the posterior is proportional to the numerator alone:

$$
P(n \mid h) \;\propto\; \prod_s \binom{n_s}{h_s}.
$$

Summing this over the assignments in which suit $k$ is the common suit gives
$P(\text{common} = k \mid h)$, and the partner map turns it into the quantity we
actually want:

$$
P(\text{goal} = k \mid h) = P\big(\text{common} = \text{partner}(k) \mid h\big).
$$

**Intuition.** The suit you hold *most* of is probably the 12-card common suit —
which means its *partner*, not itself, is probably the goal. Holding a void in a
suit is evidence it is the 8-card suit.

## 3. Expected goal-card count

Given the common suit, its partner is — uniformly over the three remaining suits
$\{8, 10, 10\}$ — the 8-suit with probability $\tfrac13$ and a 10-suit with
probability $\tfrac23$. Hence

$$
\mathbb{E}[\text{goal cards}] = \tfrac13 \cdot 8 + \tfrac23 \cdot 10 = \tfrac{28}{3} \approx 9.33.
$$

## 4. Fair value of a card

The $200 pot is fully paid out: $10 per goal card plus a majority bonus. Amortising
the whole pot across the goal cards gives a per-goal-card fair value of
$\text{pot} / \mathbb{E}[\text{goal cards}]$. A card of suit $s$ is a goal card
with probability $P(\text{goal}=s)$, so

$$
v_s = P(\text{goal}=s)\cdot \frac{\text{pot}}{\mathbb{E}[\text{goal cards}]}.
$$

This is the anchor the market maker quotes around — see
[`agents.md`](agents.md) for how inventory and uncertainty turn a value into a
two-sided quote.
