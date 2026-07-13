"""Figgie market engine: a continuous double auction plus settlement.

The engine models one round of Figgie as a sequence of discrete decision steps.
At each step a randomly chosen player observes the public market state and their
own hand and submits an :class:`Order` (or passes).  Orders are matched against
resting quotes with price-time priority; each fill moves exactly one card.  When
the clock runs out the goal suit is revealed and the pot is paid out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cards import Deck, Suit, deal, make_deck


@dataclass(frozen=True)
class Order:
    """A single-card limit order. ``is_buy`` True = bid, False = ask."""

    player: int
    suit: Suit
    price: int
    is_buy: bool


PASS: Order | None = None


@dataclass(frozen=True)
class Trade:
    step: int
    suit: Suit
    price: int
    buyer: int
    seller: int
    aggressor_bought: bool  # True if the incoming order was a buy that lifted an ask


@dataclass
class Quote:
    price: int
    player: int


@dataclass
class GameConfig:
    n_players: int = 4
    pot: int = 200
    starting_cash: int = 350
    steps: int = 240          # decision steps in the trading period
    seed: int | None = None


@dataclass
class Observation:
    """Everything a player is allowed to see when it is their turn to act."""

    me: int
    hand: np.ndarray                      # my suit counts (len 4)
    cash: int                             # my available cash
    best_bid: list[Quote | None]          # per suit
    best_ask: list[Quote | None]          # per suit
    net_bought: np.ndarray                # cumulative aggressor-buy volume per suit
    net_sold: np.ndarray                  # cumulative aggressor-sell volume per suit
    last_price: list[int | None]          # per suit
    step: int
    total_steps: int
    config: GameConfig


@dataclass
class GameResult:
    deck: Deck
    profits: np.ndarray                   # net profit per player (zero-sum)
    goal_holdings: np.ndarray             # goal cards held per player at settlement
    hands_start: list[np.ndarray]
    hands_end: list[np.ndarray]
    trades: list[Trade]
    # Per-step trajectories for visualisation.
    mid_history: np.ndarray               # (steps+1, 4) mid prices (nan if no market)
    pnl_history: np.ndarray               # (steps+1, n_players) mark-to-fundamental P&L


class Market:
    """Four independent single-name order books with price-time priority.

    Each player may keep at most one resting bid and one resting ask per suit; a
    new quote on the same side replaces the old one.  Shorting is disallowed and
    bids are capped by available cash, both enforced at submission time.
    """

    def __init__(self, n_players: int):
        self.n_players = n_players
        # bids[suit] / asks[suit]: dict player -> Order (their resting quote)
        self.bids: list[dict[int, Order]] = [dict() for _ in range(4)]
        self.asks: list[dict[int, Order]] = [dict() for _ in range(4)]

    def best_bid(self, suit: int) -> Quote | None:
        book = self.bids[suit]
        if not book:
            return None
        o = max(book.values(), key=lambda o: o.price)
        return Quote(price=o.price, player=o.player)

    def best_ask(self, suit: int) -> Quote | None:
        book = self.asks[suit]
        if not book:
            return None
        o = min(book.values(), key=lambda o: o.price)
        return Quote(price=o.price, player=o.player)

    def clear_player_side(self, suit: int, player: int, is_buy: bool) -> None:
        (self.bids if is_buy else self.asks)[suit].pop(player, None)


class FiggieGame:
    """Owns deck, hands, cash, the market, and the settlement rules."""

    def __init__(self, agents, config: GameConfig | None = None):
        self.config = config or GameConfig()
        if len(agents) != self.config.n_players:
            raise ValueError("Number of agents must equal config.n_players.")
        self.agents = agents
        self.rng = np.random.default_rng(self.config.seed)

        self.deck = make_deck(self.rng)
        self.hands = deal(self.deck, self.config.n_players, self.rng)
        self.hands_start = [h.copy() for h in self.hands]
        self.cash = np.full(self.config.n_players, self.config.starting_cash, dtype=float)
        self.market = Market(self.config.n_players)

        self.trades: list[Trade] = []
        self.net_bought = np.zeros(4, dtype=int)
        self.net_sold = np.zeros(4, dtype=int)
        self.last_price: list[int | None] = [None, None, None, None]

    # ---- observation -------------------------------------------------------
    def _observe(self, player: int, step: int) -> Observation:
        return Observation(
            me=player,
            hand=self.hands[player].copy(),
            cash=int(self.cash[player]),
            best_bid=[self.market.best_bid(s) for s in range(4)],
            best_ask=[self.market.best_ask(s) for s in range(4)],
            net_bought=self.net_bought.copy(),
            net_sold=self.net_sold.copy(),
            last_price=list(self.last_price),
            step=step,
            total_steps=self.config.steps,
            config=self.config,
        )

    # ---- order handling ----------------------------------------------------
    def _resting_asks_of(self, player: int, suit: int) -> int:
        return 1 if player in self.market.asks[suit] else 0

    def _submit(self, order: Order, step: int) -> None:
        suit = int(order.suit)
        p = order.player

        if order.is_buy:
            if order.price <= 0 or self.cash[p] < order.price:
                return  # cannot afford this bid
            ask = self.market.best_ask(suit)
            if ask is not None and ask.player != p and ask.price <= order.price:
                self._execute(buyer=p, seller=ask.player, suit=suit,
                              price=ask.price, step=step, aggressor_bought=True)
                self.market.clear_player_side(suit, ask.player, is_buy=False)
                return
            self.market.bids[suit][p] = order  # rest / replace
        else:
            # Need a real card to sell that is not already committed to a resting ask.
            if self.hands[p][suit] - self._resting_asks_of(p, suit) < 1:
                return
            bid = self.market.best_bid(suit)
            if bid is not None and bid.player != p and bid.price >= order.price:
                self._execute(buyer=bid.player, seller=p, suit=suit,
                              price=bid.price, step=step, aggressor_bought=False)
                self.market.clear_player_side(suit, bid.player, is_buy=True)
                return
            self.market.asks[suit][p] = order

    def _execute(self, buyer: int, seller: int, suit: int, price: int, step: int,
                 aggressor_bought: bool) -> None:
        self.cash[buyer] -= price
        self.cash[seller] += price
        self.hands[buyer][suit] += 1
        self.hands[seller][suit] -= 1
        self.last_price[suit] = price
        if aggressor_bought:
            self.net_bought[suit] += 1
        else:
            self.net_sold[suit] += 1
        # A filled resting quote on the losing side may now be stale; drop any of
        # the seller's asks / buyer's bids that they can no longer honour.
        if self.hands[seller][suit] - self._resting_asks_of(seller, suit) < 0:
            self.market.clear_player_side(suit, seller, is_buy=False)
        self.trades.append(Trade(step, Suit(suit), price, buyer, seller, aggressor_bought))

    # ---- settlement --------------------------------------------------------
    def _settle(self) -> tuple[np.ndarray, np.ndarray]:
        goal = int(self.deck.goal_suit)
        holdings = np.array([h[goal] for h in self.hands], dtype=int)
        payout = holdings * 10.0
        remainder = self.config.pot - int(holdings.sum()) * 10
        winners = np.flatnonzero(holdings == holdings.max())
        # If nobody holds a goal card, the max is 0 and every player "ties"; the
        # remainder (the whole pot) is split evenly, matching the ante refund.
        payout[winners] += remainder / len(winners)
        return payout, holdings

    def _fundamental_pnl(self, values: np.ndarray) -> np.ndarray:
        """Mark hands to their risk-neutral fundamental value for a P&L curve."""
        marks = np.array([float(h @ values) for h in self.hands])
        return (self.cash - self.config.starting_cash) + marks

    # ---- main loop ---------------------------------------------------------
    def play(self) -> GameResult:
        from .belief import card_values, posterior_goal  # local import avoids cycle

        cfg = self.config
        for agent, player in zip(self.agents, range(cfg.n_players)):
            if hasattr(agent, "reset"):
                agent.reset(player, self.hands[player].copy(), cfg)

        mid_history = np.full((cfg.steps + 1, 4), np.nan)
        pnl_history = np.zeros((cfg.steps + 1, cfg.n_players))

        # A neutral observer's fundamental value (public prior, no private hand)
        # used purely to mark P&L consistently for every player.
        neutral_values = card_values(posterior_goal(np.zeros(4, dtype=int)), cfg.pot)

        order = self.rng.permutation(cfg.n_players)
        for step in range(cfg.steps):
            player = int(order[step % cfg.n_players])
            if step % cfg.n_players == cfg.n_players - 1:
                order = self.rng.permutation(cfg.n_players)
            action = self.agents[player].act(self._observe(player, step))
            if action is not None:
                self._submit(action, step)
            self._record(step + 1, mid_history, pnl_history, neutral_values)

        payout, holdings = self._settle()
        ante = cfg.pot / cfg.n_players
        profits = (self.cash - cfg.starting_cash) + payout - ante

        return GameResult(
            deck=self.deck,
            profits=profits,
            goal_holdings=holdings,
            hands_start=self.hands_start,
            hands_end=[h.copy() for h in self.hands],
            trades=self.trades,
            mid_history=mid_history,
            pnl_history=pnl_history,
        )

    def _record(self, t: int, mid_history: np.ndarray, pnl_history: np.ndarray,
                values: np.ndarray) -> None:
        for s in range(4):
            bid = self.market.best_bid(s)
            ask = self.market.best_ask(s)
            if bid is not None and ask is not None:
                mid_history[t, s] = (bid.price + ask.price) / 2
            elif self.last_price[s] is not None:
                mid_history[t, s] = self.last_price[s]
        pnl_history[t] = self._fundamental_pnl(values)


def play_game(agents, config: GameConfig | None = None) -> GameResult:
    """Convenience wrapper to run a single game."""
    return FiggieGame(agents, config).play()
