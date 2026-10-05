"""Laboratory Boot Sequence & Controls Quick-Start Modal (§41)."""
from __future__ import annotations

from typing import Callable, Optional
import pygame

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
from .widgets import Button


class IntroModal:
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        mw, mh = 640, 480
        self.rect = pygame.Rect((screen_w - mw) // 2, (screen_h - mh) // 2, mw, mh)
        self.visible = True  # Shown on initial startup

        self.btn_start = Button(
            pygame.Rect(self.rect.centerx - 100, self.rect.bottom - 48, 200, 32),
            "ENTER LABORATORY",
            self.dismiss,
            color=(16, 75, 110),
        )

    def dismiss(self) -> None:
        self.visible = False

    def show(self) -> None:
        self.visible = True

    def toggle(self) -> None:
        self.visible = not self.visible

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False

        if self.btn_start.handle_event(event):
            return True

        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_ESCAPE):
                self.dismiss()
                return True

        # Consume clicks on modal background
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            if self.rect.collidepoint(event.pos):
                return True

        return False

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        if not self.visible:
            return

        # Dim world backdrop
        dim_surf = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        dim_surf.fill((5, 8, 14, 180))
        surface.blit(dim_surf, (0, 0))

        # Sci-Fi dialog box
        theme.draw_glass_panel(
            surface,
            self.rect,
            glow=True,
            header_height=42,
            title="DIGITAL EVOLUTION LAB — INITIALIZATION",
        )

        # Subtitle
        sub = theme.font_xs.render("ARTIFICIAL LIFE & NEURAL EVOLUTION SIMULATION SUITE", True, TEXT_ACCENT)
        surface.blit(sub, (self.rect.x + 14, self.rect.y + 48))

        # Description
        desc_lines = [
            "Welcome to the laboratory sandbox. Organisms in this ecosystem are governed entirely",
            "by emergent neural networks and genetics. Behaviors are NOT hard-coded: foraging,",
            "predation, schooling, and evasion evolve dynamically through natural selection.",
        ]
        for i, line in enumerate(desc_lines):
            t = theme.font_xs.render(line, True, TEXT_SECONDARY)
            surface.blit(t, (self.rect.x + 14, self.rect.y + 75 + i * 18))

        # Controls cheat-sheet
        ctrl_y = self.rect.y + 140
        h_ctrl = theme.font_sm.render("LABORATORY KEYBOARD SHORTCUTS", True, TEXT_PRIMARY)
        surface.blit(h_ctrl, (self.rect.x + 14, ctrl_y))

        shortcuts = [
            ("SPACE", "Pause / resume simulation"),
            ("1 - 5", "Simulation speed (0.5x, 1x, 2x, 5x, 20x)"),
            ("L-CLICK", "Inspect organism / Apply God-mode brush"),
            ("R-DRAG", "Pan simulation camera across world"),
            ("SCROLL", "Zoom camera in / out centered on cursor"),
            ("TAB", "Cycle focus across living organisms"),
            ("D", "Toggle Analytics & Telemetry Dashboard"),
            ("G", "Toggle God Mode Sandbox & Climate tools"),
            ("T", "Toggle Evolutionary Milestones Timeline"),
            ("S", "Toggle Phylogenetic Species Tree"),
            ("O", "Toggle Vision Cones & Movement Trails"),
            ("F1 / H", "Toggle this help screen"),
        ]

        col1 = shortcuts[:6]
        col2 = shortcuts[6:]

        for i, (k, desc) in enumerate(col1):
            ky = ctrl_y + 26 + i * 22
            theme.draw_badge(surface, k, self.rect.x + 14, ky, bg_color=(28, 42, 60), text_color=BORDER_GLOW)
            txt = theme.font_xs.render(desc, True, TEXT_SECONDARY)
            surface.blit(txt, (self.rect.x + 85, ky + 2))

        for i, (k, desc) in enumerate(col2):
            ky = ctrl_y + 26 + i * 22
            theme.draw_badge(surface, k, self.rect.x + 315, ky, bg_color=(28, 42, 60), text_color=BORDER_GLOW)
            txt = theme.font_xs.render(desc, True, TEXT_SECONDARY)
            surface.blit(txt, (self.rect.x + 386, ky + 2))

        # Start button
        self.btn_start.draw(surface, theme)
