"""Interactive UI Widgets: Buttons, Toggles, Sliders, and Tabs."""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple
import pygame

from .theme import (
    BORDER_ACCENT,
    BORDER_DEFAULT,
    BORDER_GLOW,
    BTN_ACTIVE,
    BTN_BORDER,
    BTN_HOVER,
    BTN_NORMAL,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    Theme,
)


class Button:
    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        on_click: Optional[Callable[[], None]] = None,
        tooltip: str = "",
        color: Optional[Tuple[int, int, int]] = None,
        active: bool = False,
    ):
        self.rect = rect
        self.text = text
        self.on_click = on_click
        self.tooltip = tooltip
        self.base_color = color or BTN_NORMAL
        self.active = active
        self.hovered = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.on_click:
                    self.on_click()
                return True
        return False

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        bg = BTN_ACTIVE if self.active else (BTN_HOVER if self.hovered else self.base_color)
        border = BORDER_GLOW if (self.hovered or self.active) else BTN_BORDER

        pygame.draw.rect(surface, bg, self.rect, border_radius=4)
        pygame.draw.rect(surface, border, self.rect, 1, border_radius=4)

        txt_col = TEXT_PRIMARY if (self.hovered or self.active) else TEXT_SECONDARY
        txt = theme.font_sm.render(self.text, True, txt_col)
        tx = self.rect.x + (self.rect.w - txt.get_width()) // 2
        ty = self.rect.y + (self.rect.h - txt.get_height()) // 2
        surface.blit(txt, (tx, ty))


class Slider:
    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        val_min: float,
        val_max: float,
        current_val: float,
        on_change: Optional[Callable[[float], None]] = None,
        format_str: str = "{:.2f}",
    ):
        self.rect = rect
        self.label = label
        self.val_min = val_min
        self.val_max = val_max
        self.val = current_val
        self.on_change = on_change
        self.format_str = format_str
        self.dragging = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        track_rect = pygame.Rect(self.rect.x + 80, self.rect.y + 4, self.rect.w - 140, self.rect.h - 8)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self._update_val(event.pos[0], track_rect)
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_val(event.pos[0], track_rect)
            return True
        return False

    def _update_val(self, mouse_x: int, track_rect: pygame.Rect) -> None:
        frac = max(0.0, min(1.0, (mouse_x - track_rect.x) / track_rect.w))
        self.val = self.val_min + frac * (self.val_max - self.val_min)
        if self.on_change:
            self.on_change(self.val)

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        # Label
        lbl = theme.font_xs.render(self.label, True, TEXT_SECONDARY)
        surface.blit(lbl, (self.rect.x, self.rect.y + 4))

        # Track
        track_w = self.rect.w - 140
        track_rect = pygame.Rect(self.rect.x + 80, self.rect.y + 8, track_w, 4)
        pygame.draw.rect(surface, BTN_NORMAL, track_rect, border_radius=2)

        # Fill
        frac = (self.val - self.val_min) / (self.val_max - self.val_min + 1e-9)
        fill_w = int(track_w * max(0.0, min(1.0, frac)))
        if fill_w > 0:
            fill_rect = pygame.Rect(track_rect.x, track_rect.y, fill_w, 4)
            pygame.draw.rect(surface, BORDER_GLOW, fill_rect, border_radius=2)

        # Handle knob
        knob_x = track_rect.x + fill_w
        knob_y = track_rect.y + 2
        pygame.draw.circle(surface, TEXT_PRIMARY, (knob_x, knob_y), 6)
        pygame.draw.circle(surface, BORDER_GLOW, (knob_x, knob_y), 6, 1)

        # Readout text
        val_txt = theme.font_xs.render(self.format_str.format(self.val), True, TEXT_PRIMARY)
        surface.blit(val_txt, (self.rect.x + self.rect.w - 50, self.rect.y + 4))


class Toggle:
    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        checked: bool = False,
        on_toggle: Optional[Callable[[bool], None]] = None,
    ):
        self.rect = rect
        self.label = label
        self.checked = checked
        self.on_toggle = on_toggle

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.checked = not self.checked
                if self.on_toggle:
                    self.on_toggle(self.checked)
                return True
        return False

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        box_size = 14
        box_rect = pygame.Rect(self.rect.x, self.rect.y + (self.rect.h - box_size) // 2, box_size, box_size)
        pygame.draw.rect(surface, BTN_NORMAL, box_rect, border_radius=3)
        pygame.draw.rect(surface, BORDER_GLOW if self.checked else BORDER_DEFAULT, box_rect, 1, border_radius=3)

        if self.checked:
            inner = pygame.Rect(box_rect.x + 3, box_rect.y + 3, box_size - 6, box_size - 6)
            pygame.draw.rect(surface, BORDER_GLOW, inner, border_radius=2)

        txt = theme.font_xs.render(self.label, True, TEXT_PRIMARY if self.checked else TEXT_SECONDARY)
        surface.blit(txt, (box_rect.right + 8, self.rect.y + (self.rect.h - txt.get_height()) // 2))
