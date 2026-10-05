"""God Mode Sandbox Tools & Environmental Intervention Panel (§38, §41)."""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pygame

from ...evolution.genome import (
    G_ATTACK,
    G_DIET,
    G_METAB,
    G_MUT,
    G_SIZE,
    G_SPEED,
    G_VIS,
    N_TRAITS,
    carnivore_template,
    herbivore_template,
)
from ...neural.network import init_params
from ...simulation.world import ROCK, GRASS
from .theme import (
    BORDER_ACCENT,
    BORDER_DEFAULT,
    BORDER_GLOW,
    BTN_ACTIVE,
    BTN_NORMAL,
    TEXT_ACCENT,
    TEXT_DANGER,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_SUCCESS,
    TEXT_WARNING,
    TEXT_PURPLE,
    Theme,
)
from .widgets import Button, Slider


class GodModePanel:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.visible = False
        self.active_tool: str = "inspect"  # "inspect", "food", "herb", "pred", "laser", "rock", "clear_rock"
        self.brush_radius = 40.0

        # Close button
        self.btn_close = Button(pygame.Rect(rect.right - 26, rect.y + 6, 20, 20), "X", self.toggle)

        # Tool buttons
        bx = rect.x + 12
        by = rect.y + 36
        bw = (rect.w - 30) // 2

        self.tool_buttons: Dict[str, Button] = {
            "inspect": Button(pygame.Rect(bx, by, bw, 22), "CURSOR", lambda: self.set_tool("inspect")),
            "food": Button(pygame.Rect(bx + bw + 6, by, bw, 22), "PLANT FOOD", lambda: self.set_tool("food")),
            "herb": Button(pygame.Rect(bx, by + 28, bw, 22), "+ HERBIVORE", lambda: self.set_tool("herb")),
            "pred": Button(pygame.Rect(bx + bw + 6, by + 28, bw, 22), "+ PREDATOR", lambda: self.set_tool("pred")),
            "laser": Button(pygame.Rect(bx, by + 56, bw, 22), "KILL LASER", lambda: self.set_tool("laser"), color=(60, 20, 25)),
            "rock": Button(pygame.Rect(bx + bw + 6, by + 56, bw, 22), "ADD WALL", lambda: self.set_tool("rock")),
        }

        # Disaster trigger buttons
        dy = by + 90
        self.disaster_buttons: List[Tuple[str, Button]] = [
            ("drought", Button(pygame.Rect(bx, dy, bw, 20), "DROUGHT")),
            ("flood", Button(pygame.Rect(bx + bw + 6, dy, bw, 20), "FLOOD")),
            ("heatwave", Button(pygame.Rect(bx, dy + 24, bw, 20), "HEATWAVE")),
            ("ice_age", Button(pygame.Rect(bx + bw + 6, dy + 24, bw, 20), "ICE AGE")),
            ("meteor", Button(pygame.Rect(bx, dy + 48, bw, 20), "METEOR IMPACT", color=(80, 28, 20))),
            ("toxic", Button(pygame.Rect(bx + bw + 6, dy + 48, bw, 20), "TOXIC SPILL", color=(50, 60, 20))),
        ]

        # Sliders
        sy = dy + 82
        self.slider_temp = Slider(
            pygame.Rect(bx, sy, rect.w - 24, 22),
            "TEMP OFFSET",
            -0.5,
            0.5,
            0.0,
            format_str="{:+.2f}",
        )
        self.slider_food_rate = Slider(
            pygame.Rect(bx, sy + 28, rect.w - 24, 22),
            "FOOD RATE",
            0.1,
            4.0,
            1.0,
            format_str="{:.1f}x",
        )
        self.slider_mutation = Slider(
            pygame.Rect(bx, sy + 56, rect.w - 24, 22),
            "MUTATION",
            0.1,
            5.0,
            1.0,
            format_str="{:.1f}x",
        )

        # Status text
        self.last_msg = ""
        self.last_msg_timer = 0.0

    def toggle(self) -> None:
        self.visible = not self.visible

    def set_tool(self, tool: str) -> None:
        self.active_tool = tool

    def handle_event(self, event: pygame.event.Event, eco=None) -> bool:
        if not self.visible:
            return False

        if self.btn_close.handle_event(event):
            return True

        for k, btn in self.tool_buttons.items():
            if btn.handle_event(event):
                return True

        if eco:
            for kind, btn in self.disaster_buttons:
                if btn.handle_event(event):
                    try:
                        eco.apply_action("disaster", kind=kind)
                        self.last_msg = f"TRIGGERED: {kind.upper()}"
                        self.last_msg_timer = 2.5
                    except Exception as ex:
                        self.last_msg = f"ERROR: {ex}"
                    return True

        if self.slider_temp.handle_event(event):
            if eco:
                eco.apply_action("temperature_offset", value=self.slider_temp.val)
            return True

        if self.slider_food_rate.handle_event(event):
            if eco:
                eco.apply_action("food_scale", value=self.slider_food_rate.val)
            return True

        if self.slider_mutation.handle_event(event):
            if eco:
                eco.apply_action("set_mutation_rate", value=self.slider_mutation.val * eco.cfg.mutation_rate)
            return True

        # Consume panel events
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            if self.rect.collidepoint(event.pos):
                return True

        return False

    def apply_brush(self, eco, wx: float, wy: float) -> bool:
        """Apply active sandbox brush at world coordinates (wx, wy)."""
        if self.active_tool == "inspect":
            return False

        if self.active_tool == "food":
            eco.apply_action("spawn_food", x=wx, y=wy, n=8, radius=self.brush_radius)
            return True

        elif self.active_tool == "herb":
            eco.apply_action("spawn_organism", x=wx, y=wy, kind="herbivore", n=1)
            return True

        elif self.active_tool == "pred":
            eco.apply_action("spawn_organism", x=wx, y=wy, kind="predator", n=1)
            return True

        elif self.active_tool == "laser":
            eco.apply_action("kill_organism", x=wx, y=wy, radius=self.brush_radius)
            return True

        elif self.active_tool == "rock":
            # Paint rock obstacle on world grid
            w = eco.world
            gx = int(wx / w.cell)
            gy = int(wy / w.cell)
            rad_cells = max(1, int(self.brush_radius / w.cell))
            for iy in range(max(0, gy - rad_cells), min(w.gh, gy + rad_cells + 1)):
                for ix in range(max(0, gx - rad_cells), min(w.gw, gx + rad_cells + 1)):
                    w.terrain[iy, ix] = ROCK
                    w.blocked[iy, ix] = True
            w.version += 1
            return True

        return False

    def draw(self, surface: pygame.Surface, theme: Theme, dt: float) -> None:
        if not self.visible:
            return

        theme.draw_glass_panel(
            surface,
            self.rect,
            glow=True,
            header_height=30,
            title="GOD MODE — SANDBOX CONTROLS",
        )

        self.btn_close.draw(surface, theme)

        # Draw tool buttons
        for k, btn in self.tool_buttons.items():
            btn.active = self.active_tool == k
            btn.draw(surface, theme)

        # Section header for disasters
        dy_lbl = theme.font_xs.render("TRIGGER DISASTER EVENTS", True, TEXT_MUTED)
        surface.blit(dy_lbl, (self.rect.x + 12, self.rect.y + 114))

        for _, btn in self.disaster_buttons:
            btn.draw(surface, theme)

        # Section header for sliders
        sy_lbl = theme.font_xs.render("GLOBAL CLIMATE & PHYSICS", True, TEXT_MUTED)
        surface.blit(sy_lbl, (self.rect.x + 12, self.rect.y + 204))

        self.slider_temp.draw(surface, theme)
        self.slider_food_rate.draw(surface, theme)
        self.slider_mutation.draw(surface, theme)

        # Status text feedback
        if self.last_msg_timer > 0.0:
            self.last_msg_timer -= dt
            msg_s = theme.font_xs.render(self.last_msg, True, TEXT_SUCCESS)
            surface.blit(msg_s, (self.rect.x + 14, self.rect.bottom - 22))
