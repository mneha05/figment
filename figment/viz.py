"""Rendering: dark-themed static charts and an animated game replay (GIF).

All figures share one palette so the generated assets read as a single system.
Nothing here is required to *run* the arena; it exists to make results legible.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless rendering, safe in CI

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .belief import posterior_goal
from .cards import SUITS, Suit
from .engine import GameResult
from .evolve import Generation

BG = "#0d1117"
PANEL = "#0f1620"
GRID = "#212b3a"
TEXT = "#c9d1d9"
MUTED = "#8b949e"
ACCENT = "#58a6ff"

SUIT_COLOR = {
    Suit.SPADES: "#58a6ff",
    Suit.CLUBS: "#3fb950",
    Suit.HEARTS: "#f85149",
    Suit.DIAMONDS: "#d6a017",
}
SUIT_LABEL = {s: f"{s.glyph} {s.name.title()}" for s in SUITS}


def _style_ax(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _new_fig(figsize):
    fig = plt.figure(figsize=figsize, facecolor=BG)
    return fig


# --------------------------------------------------------------------------- #
# Belief reconstruction
# --------------------------------------------------------------------------- #
def reconstruct_belief(result: GameResult, player: int, signal_weight: float = 0.35) -> np.ndarray:
    """Recompute the focal player's goal-suit posterior at every step.

    Mirrors :class:`BayesianMarketMaker`: a static hand posterior tilted by the
    cumulative aggressor order flow observed so far.
    """
    steps = result.mid_history.shape[0]
    prior = posterior_goal(result.hands_start[player])
    nb = np.zeros((steps, 4))
    ns = np.zeros((steps, 4))
    for tr in result.trades:
        t = tr.step + 1
        if t < steps:
            if tr.aggressor_bought:
                nb[t:, int(tr.suit)] += 1
            else:
                ns[t:, int(tr.suit)] += 1
    beliefs = np.zeros((steps, 4))
    for t in range(steps):
        flow = nb[t] - ns[t]
        m = np.abs(flow).max()
        if m > 0:
            flow = flow / m
        p = prior * np.exp(signal_weight * flow)
        beliefs[t] = p / p.sum()
    return beliefs


# --------------------------------------------------------------------------- #
# Animated game replay
# --------------------------------------------------------------------------- #
def animate_game(result: GameResult, focal_player: int, agent_names: list[str],
                 path: str, max_frames: int = 72, fps: int = 14) -> None:
    steps = result.mid_history.shape[0]
    beliefs = reconstruct_belief(result, focal_player)
    goal = result.deck.goal_suit

    stride = max(1, steps // max_frames)
    frames = list(range(0, steps, stride)) + [steps - 1] * max(4, fps // 2)

    fig = _new_fig((11, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.24,
                          left=0.07, right=0.975, top=0.9, bottom=0.1)
    ax_price = fig.add_subplot(gs[0, :])
    ax_belief = fig.add_subplot(gs[1, 0])
    ax_pnl = fig.add_subplot(gs[1, 1])
    for ax in (ax_price, ax_belief, ax_pnl):
        _style_ax(ax)

    fig.suptitle("FIGMENT  ·  one Figgie round, replayed",
                 color=TEXT, fontsize=14, fontweight="bold", x=0.07, ha="left")
    fig.text(0.07, 0.925, f"goal suit revealed at the bell:  {goal.glyph} {goal.name.title()}",
             color=MUTED, fontsize=9, ha="left")

    x = np.arange(steps)
    price_lines = {}
    for s in SUITS:
        lw = 2.6 if s == goal else 1.4
        (line,) = ax_price.plot([], [], color=SUIT_COLOR[s], lw=lw,
                                label=SUIT_LABEL[s] + ("  ★" if s == goal else ""))
        price_lines[s] = line
    ax_price.set_xlim(0, steps)
    finite = result.mid_history[np.isfinite(result.mid_history)]
    ymax = float(np.nanmax(finite)) if finite.size else 20.0
    ax_price.set_ylim(0, max(12, ymax * 1.15))
    ax_price.set_ylabel("mid price ($)")
    ax_price.set_title("Market prices per suit", loc="left", fontsize=10)
    ax_price.legend(loc="upper left", fontsize=7.5, facecolor=PANEL,
                    edgecolor=GRID, labelcolor=TEXT, ncol=4)

    ax_belief.set_ylim(0, 1)
    ax_belief.set_xticks(range(4))
    ax_belief.set_xticklabels([s.glyph for s in SUITS], fontsize=13)
    ax_belief.set_ylabel("P(goal suit)")
    ax_belief.set_title(f"Player {focal_player}'s Bayesian belief", loc="left", fontsize=10)
    bars = ax_belief.bar(range(4), np.zeros(4), color=[SUIT_COLOR[s] for s in SUITS],
                         edgecolor=BG, linewidth=1.5)
    ax_belief.axhline(0.25, color=MUTED, lw=0.8, ls=(0, (4, 4)), alpha=0.7)
    goal_star = ax_belief.text(int(goal), 0.02, "★", ha="center", color=BG, fontsize=11)

    pnl = result.pnl_history
    pnl_lines = []
    palette = [ACCENT, "#3fb950", "#f85149", "#d6a017", "#bc8cff", "#39c5cf"]
    for i in range(pnl.shape[1]):
        (line,) = ax_pnl.plot([], [], color=palette[i % len(palette)], lw=1.8,
                              label=agent_names[i])
        pnl_lines.append(line)
    ax_pnl.set_xlim(0, steps)
    ax_pnl.set_ylim(float(pnl.min()) * 1.1 - 5, float(pnl.max()) * 1.1 + 5)
    ax_pnl.axhline(0, color=MUTED, lw=0.8, alpha=0.6)
    ax_pnl.set_ylabel("P&L ($)")
    ax_pnl.set_title("Mark-to-fundamental P&L", loc="left", fontsize=10)
    ax_pnl.legend(loc="upper left", fontsize=7, facecolor=PANEL, edgecolor=GRID,
                  labelcolor=TEXT, ncol=2)

    clock = ax_price.text(0.99, 0.06, "", transform=ax_price.transAxes, ha="right",
                          color=MUTED, fontsize=9)

    def update(t):
        for s in SUITS:
            price_lines[s].set_data(x[: t + 1], result.mid_history[: t + 1, int(s)])
        for i, s in enumerate(SUITS):
            bars[i].set_height(beliefs[t, int(s)])
        goal_star.set_y(min(0.92, beliefs[t, int(goal)] / 2))
        for i, line in enumerate(pnl_lines):
            line.set_data(x[: t + 1], pnl[: t + 1, i])
        pct = int(100 * t / (steps - 1))
        clock.set_text("closing bell" if t >= steps - 1 else f"trading… {pct}%")
        return []

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), savefig_kwargs={"facecolor": BG})
    plt.close(fig)


def plot_belief_convergence(result: GameResult, focal_player: int, path: str) -> None:
    beliefs = reconstruct_belief(result, focal_player)
    goal = result.deck.goal_suit
    fig = _new_fig((9, 4.2))
    ax = fig.add_subplot(111)
    _style_ax(ax)
    x = np.arange(beliefs.shape[0])
    for s in SUITS:
        lw = 2.8 if s == goal else 1.5
        ax.plot(x, beliefs[:, int(s)], color=SUIT_COLOR[s], lw=lw,
                label=SUIT_LABEL[s] + ("  ★ goal" if s == goal else ""))
    ax.axhline(0.25, color=MUTED, lw=0.8, ls=(0, (4, 4)), alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xlabel("trading step")
    ax.set_ylabel("P(goal suit)")
    ax.set_title("Belief convergence from private hand + order flow", loc="left",
                 color=TEXT, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG, dpi=120)
    plt.close(fig)


def plot_tournament(tr, path: str) -> None:
    order = np.argsort(tr.mean_profit)[::-1]
    names = [tr.names[i] for i in order]
    means = tr.mean_profit[order]
    stds = tr.std_profit[order]
    elo_map = {n: r for n, r, _ in tr.elo}
    elos = [elo_map[n] for n in names]

    fig = _new_fig((10, 4.6))
    gs = fig.add_gridspec(1, 2, wspace=0.28, left=0.09, right=0.97, top=0.84, bottom=0.14)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for ax in (ax1, ax2):
        _style_ax(ax)

    colors = [ACCENT if "MM" in n else MUTED for n in names]
    ax1.barh(range(len(names)), means, xerr=stds, color=colors, edgecolor=BG,
             error_kw={"ecolor": GRID, "elinewidth": 1})
    ax1.axvline(0, color=MUTED, lw=0.8)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, color=TEXT, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("mean profit per game ($)")
    ax1.set_title(f"Profitability over {tr.n_games} games", loc="left", fontsize=11)

    ax2.barh(range(len(names)), elos, color=colors, edgecolor=BG)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, color=TEXT, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlim(min(elos) - 40, max(elos) + 40)
    ax2.set_xlabel("Elo rating")
    ax2.set_title("Skill (multiplayer Elo)", loc="left", fontsize=11)

    fig.suptitle("FIGMENT  ·  tournament leaderboard", color=TEXT, fontsize=13,
                 fontweight="bold", x=0.09, ha="left")
    fig.savefig(path, facecolor=BG, dpi=120)
    plt.close(fig)


def plot_evolution(history: list[Generation], path: str) -> None:
    gens = [h.gen for h in history]
    best = np.array([h.best_fitness for h in history])
    mean = np.array([h.mean_fitness for h in history])
    best_so_far = np.maximum.accumulate(best)
    fig = _new_fig((9, 4.2))
    ax = fig.add_subplot(111)
    _style_ax(ax)
    ax.fill_between(gens, mean, best, color=ACCENT, alpha=0.10)
    ax.plot(gens, best_so_far, color=ACCENT, lw=2.8, marker="o", ms=4,
            label="champion (best so far)")
    ax.plot(gens, best, color="#6ea8fe", lw=1.3, alpha=0.8, label="generation best")
    ax.plot(gens, mean, color=MUTED, lw=1.5, ls="--", marker=".", label="population mean")
    ax.axhline(0, color=MUTED, lw=0.8, alpha=0.6)
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness  (mean profit / game, $)")
    ax.set_title("Evolutionary self-play: learning to make markets", loc="left",
                 color=TEXT, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG, dpi=120)
    plt.close(fig)
