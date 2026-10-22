"""Pygame visualiser: warehouse grid, robots, truck countdowns, order panel and attention overlay."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .env import WarehouseEnv

CELL = 34
PANEL = 340
TOP = 44
COLORS = {
    "bg": (24, 26, 33),
    "floor": (236, 238, 242),
    "wall": (52, 58, 70),
    "shelf": (176, 137, 97),
    "shelf_hot": (233, 152, 60),
    "dock": (66, 133, 244),
    "charger": (250, 204, 21),
    "packing": (148, 163, 184),
    "text": (230, 233, 240),
    "muted": (148, 155, 170),
    "attn": (168, 85, 247),
    "tiers": ((239, 68, 68), (59, 130, 246), (120, 130, 145)),
}


def _lerp(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def slack_color(slack_ticks: float) -> tuple[int, int, int]:
    """Green when there is plenty of slack, red when the order is about to miss its truck."""
    red, amber, green = (220, 38, 38), (245, 158, 11), (34, 197, 94)
    t = slack_ticks / 120.0
    return _lerp(red, amber, t * 2) if t < 0.5 else _lerp(amber, green, (t - 0.5) * 2)


class Renderer:
    def __init__(self, env: WarehouseEnv) -> None:
        if env.render_mode == "rgb_array":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        self.pg = pygame
        pygame.init()
        self.env = env
        rows, cols = env.layout.shape
        self.size = (cols * CELL + PANEL, max(rows * CELL, 16 * 30 + 60) + TOP)
        if env.render_mode == "human":
            self.screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption("DockRush")
            self.clock = pygame.time.Clock()
        else:
            self.screen = pygame.Surface(self.size)
        self.font = pygame.font.Font(None, 20)
        self.small = pygame.font.Font(None, 16)
        self.big = pygame.font.Font(None, 26)

    def close(self) -> None:
        self.pg.quit()

    def draw(self, mode: str):
        pg, env, screen = self.pg, self.env, self.screen
        screen.fill(COLORS["bg"])
        layout = env.layout
        attention = env.attention if env.attention is not None else np.zeros(env.max_orders)
        attn_max = max(1e-6, float(attention.max()))

        hot: dict[tuple[int, int], float] = {}
        for index, order in enumerate(env.slots):
            if order is None:
                continue
            for item, state in enumerate(order.item_state):
                if state == 0:
                    shelf = layout.shelves[order.spec.items[item]]
                    hot[shelf] = max(hot.get(shelf, 0.0), attention[index] / attn_max if index < len(attention) else 0.0)

        for r, row in enumerate(layout.grid):
            for c, mark in enumerate(row):
                rect = pg.Rect(c * CELL, TOP + r * CELL, CELL, CELL)
                if mark == "#" or (r, c) in layout.blocked:
                    pg.draw.rect(screen, COLORS["wall"], rect)
                elif mark == "S":
                    pg.draw.rect(screen, COLORS["floor"], rect)
                    color = COLORS["shelf_hot"] if (r, c) in hot else COLORS["shelf"]
                    pg.draw.rect(screen, color, rect.inflate(-6, -6), border_radius=4)
                    if hot.get((r, c), 0) > 0.05:
                        pg.draw.rect(
                            screen, COLORS["attn"], rect.inflate(-2, -2), width=max(1, int(4 * hot[(r, c)])), border_radius=5
                        )
                else:
                    pg.draw.rect(screen, COLORS["floor"], rect)
                pg.draw.rect(screen, (210, 214, 222), rect, width=1)

        for dock_index, cell in enumerate(layout.docks):
            r, c = layout.positions[cell]
            rect = pg.Rect(c * CELL, TOP + r * CELL, CELL, CELL)
            pg.draw.rect(screen, COLORS["dock"], rect.inflate(-4, -4), border_radius=4)
            self._text(f"D{dock_index}", rect.center, COLORS["text"], self.small, center=True)
            trucks = env._dock_trucks[dock_index]
            nxt = env._dock_next[dock_index]
            if nxt < len(trucks):
                left = trucks[nxt].scheduled_departure - env.time
                bar = pg.Rect(rect.x, rect.y - 7, int(CELL * np.clip(left / 180.0, 0, 1)), 5)
                pg.draw.rect(screen, slack_color(left), bar)
        if layout.charger is not None:
            r, c = layout.positions[layout.charger]
            rect = pg.Rect(c * CELL, TOP + r * CELL, CELL, CELL)
            pg.draw.rect(screen, COLORS["charger"], rect.inflate(-4, -4), border_radius=4)
            self._text("C", rect.center, (40, 40, 40), self.font, center=True)
        r, c = layout.positions[layout.packing]
        rect = pg.Rect(c * CELL, TOP + r * CELL, CELL, CELL)
        pg.draw.rect(screen, COLORS["packing"], rect.inflate(-4, -4), border_radius=4)
        self._text("P", rect.center, (30, 30, 30), self.font, center=True)

        for robot in env.robots:
            r, c = layout.positions[robot.cell]
            tr, tc = layout.positions[robot.target]
            center = (c * CELL + CELL // 2, TOP + r * CELL + CELL // 2)
            target = (tc * CELL + CELL // 2, TOP + tr * CELL + CELL // 2)
            if target != center:
                pg.draw.line(screen, (99, 102, 241), center, target, 2)
            charge = robot.battery / env.cfg.battery_capacity if env.battery_on else 1.0
            pg.draw.circle(screen, _lerp((220, 38, 38), (34, 197, 94), charge), center, CELL // 2 - 3)
            pg.draw.circle(screen, (20, 20, 20), center, CELL // 2 - 3, width=2)
            self._text(str(robot.load), center, (15, 15, 15), self.font, center=True)

        done = env.stats
        self._text(
            f"t = {env.time:5.0f} / {env.horizon:.0f}    return {env.episode_return:7.1f}    "
            f"fulfilled {done['fulfilled']}  lost {done['lost']}  deferred {done['deferred']}",
            (10, 14),
            COLORS["text"],
            self.big,
        )
        self._draw_panel(layout.shape[1] * CELL, attention, attn_max)

        if mode == "human":
            pg.event.pump()
            pg.display.flip()
            self.clock.tick(self.env.metadata["render_fps"])
            return None
        return np.transpose(pg.surfarray.array3d(screen), (1, 0, 2)).copy()

    def _draw_panel(self, x0: int, attention: np.ndarray, attn_max: float) -> None:
        pg, env, screen = self.pg, self.env, self.screen
        self._text("orders  (tier  value  items  slack)", (x0 + 12, TOP + 6), COLORS["muted"], self.small)
        robot = env.robots[env.active]
        for index, order in enumerate(env.slots):
            y = TOP + 26 + index * 30
            row = pg.Rect(x0 + 8, y, PANEL - 16, 26)
            pg.draw.rect(screen, (36, 39, 48), row, border_radius=4)
            if order is None:
                continue
            if index < len(attention) and attention[index] > 0:
                width = int((PANEL - 16) * attention[index] / attn_max)
                pg.draw.rect(screen, (*COLORS["attn"][:3],), pg.Rect(x0 + 8, y + 22, width, 4), border_radius=2)
            pg.draw.rect(screen, COLORS["tiers"][order.tier], pg.Rect(x0 + 12, y + 5, 16, 16), border_radius=3)
            label = "VIP" if order.spec.vip else "ESB"[order.tier]
            self._text(label, (x0 + 20, y + 13), COLORS["text"], self.small, center=True)
            self._text(
                f"#{order.oid:<3d} ${order.value:5.1f}  {order.unpicked}/{order.n}  D{order.dock}",
                (x0 + 36, y + 7),
                COLORS["text"],
                self.font,
            )
            slack = order.truck.scheduled_departure - env.time
            bar_w = int(90 * np.clip(slack / 200.0, 0.03, 1.0))
            pg.draw.rect(screen, slack_color(slack), pg.Rect(x0 + PANEL - 108, y + 7, bar_w, 12), border_radius=3)
        footer = TOP + 26 + env.max_orders * 30 + 6
        battery = f"{100 * robot.battery / env.cfg.battery_capacity:3.0f}%" if env.battery_on else "off"
        self._text(
            f"robot {robot.rid}  load {robot.load}/{env.capacity}  battery {battery}",
            (x0 + 12, footer),
            COLORS["muted"],
            self.font,
        )

    def _text(self, text, pos, color, font, center=False):
        surface = font.render(text, True, color)
        rect = surface.get_rect(center=pos) if center else surface.get_rect(topleft=pos)
        self.screen.blit(surface, rect)


def record_gif(
    env: WarehouseEnv, policy, path: str, seed: int = 0, fps: int = 6, max_frames: int = 400, attention_fn=None
) -> int:
    """Roll out ``policy`` (obs -> action) in an ``rgb_array`` env and save an animated GIF."""
    import imageio.v2 as imageio

    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    for _ in range(max_frames):
        if attention_fn is not None:
            env.set_attention(attention_fn(obs))
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        frames.append(env.render())
        if terminated or truncated:
            break
    imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)
    return len(frames)
