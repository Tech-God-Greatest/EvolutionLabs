"""Live Evolutionary Event Feed & Timeline Panel (§33, §41)."""
from __future__ import annotations

from typing import List, Optional
import pygame

from ...analysis.events import Event
from .theme import (
    BORDER_ACCENT,
    BORDER_DEFAULT,
    BORDER_GLOW,
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


SEVERITY_COLORS = {
    1: TEXT_SECONDARY,
    2: TEXT_ACCENT,
    3: TEXT_WARNING,
    4: TEXT_DANGER,
}

KIND_BADGES = {
    "speciation": ("SPEC", TEXT_ACCENT),
    "extinction": ("EXTN", TEXT_DANGER),
    "dominance": ("DOM", TEXT_PURPLE),
    "boom": ("BOOM", TEXT_SUCCESS),
    "crash": ("CRSH", TEXT_WARNING),
    "bottleneck": ("BTLN", TEXT_DANGER),
    "disaster": ("DISA", TEXT_DANGER),
    "behavior": ("BEHV", TEXT_SUCCESS),
    "genetic_drift": ("DRFT", TEXT_WARNING),
}


class TimelinePanel:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.visible = False
        self.scroll_offset = 0

        # Close button
        self.btn_close = Button(pygame.Rect(rect.right - 26, rect.y + 6, 20, 20), "X", self.toggle)
        # Clear filter / scroll to top button
        self.btn_top = Button(pygame.Rect(rect.x + 12, rect.y + 35, 60, 20), "TOP", self.scroll_to_top)

        # Toasts list (active floating notifications on screen)
        self.toasts: List[dict] = []
        self._last_event_count = 0

    def toggle(self) -> None:
        self.visible = not self.visible

    def scroll_to_top(self) -> None:
        self.scroll_offset = 0

    def update_toasts(self, eco, dt: float) -> None:
        """Update active toast notifications shown in the main view."""
        events = eco.events.events
        n = len(events)
        if n > self._last_event_count:
            # New events arrived! Add latest to toast
            for ev in events[self._last_event_count : n]:
                if ev.severity >= 2:  # only prominent events as toasts
                    self.toasts.append({
                        "event": ev,
                        "life": 4.5,
                        "max_life": 4.5,
                    })
            self._last_event_count = n

        # Age toasts
        for t in self.toasts:
            t["life"] -= dt
        self.toasts = [t for t in self.toasts if t["life"] > 0]

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False

        if self.btn_close.handle_event(event):
            return True
        if self.btn_top.handle_event(event):
            return True

        if event.type == pygame.MOUSEWHEEL:
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll_offset = max(0, self.scroll_offset - event.y * 3)
                return True

        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            if self.rect.collidepoint(event.pos):
                return True

        return False

    def draw(self, surface: pygame.Surface, theme: Theme, eco) -> None:
        # First draw floating toasts if any
        self._draw_toasts(surface, theme)

        if not self.visible:
            return

        theme.draw_glass_panel(
            surface,
            self.rect,
            glow=True,
            header_height=30,
            title="TIMELINE & MILESTONES FEED",
        )

        self.btn_close.draw(surface, theme)
        self.btn_top.draw(surface, theme)

        # Event list
        events = eco.events.events
        n_ev = len(events)
        cnt_txt = theme.font_xs.render(f"EVENTS LOGGED: {n_ev}", True, TEXT_MUTED)
        surface.blit(cnt_txt, (self.rect.x + 80, self.rect.y + 38))

        list_area = pygame.Rect(self.rect.x + 10, self.rect.y + 64, self.rect.w - 20, self.rect.h - 74)
        item_h = 36
        visible_items = list_area.h // item_h

        # Most recent events first
        rev_events = list(reversed(events))
        start_idx = self.scroll_offset
        end_idx = min(len(rev_events), start_idx + visible_items)

        for i, ev in enumerate(rev_events[start_idx:end_idx]):
            iy = list_area.y + i * item_h
            item_rect = pygame.Rect(list_area.x, iy, list_area.w, item_h - 4)

            # Row background
            pygame.draw.rect(surface, (16, 24, 36), item_rect, border_radius=3)
            pygame.draw.rect(surface, BORDER_DEFAULT, item_rect, 1, border_radius=3)

            # Badge
            b_tag, b_col = KIND_BADGES.get(ev.kind, (ev.kind[:4].upper(), TEXT_SECONDARY))
            theme.draw_badge(surface, b_tag, item_rect.x + 6, iy + 6, bg_color=(24, 32, 48), text_color=b_col)

            # Gen / Time
            time_txt = theme.font_xs.render(f"G{ev.generation:.1f}", True, TEXT_MUTED)
            surface.blit(time_txt, (item_rect.x + 56, iy + 8))

            # Main text description
            col = SEVERITY_COLORS.get(ev.severity, TEXT_PRIMARY)
            msg_txt = theme.font_xs.render(ev.text[:65], True, col)
            surface.blit(msg_txt, (item_rect.x + 106, iy + 8))

    def _draw_toasts(self, surface: pygame.Surface, theme: Theme) -> None:
        """Render subtle floating notification banners in top-center of screen."""
        if not self.toasts:
            return

        sw = surface.get_width()
        toast_w = 420
        toast_h = 28
        start_y = 50

        for i, t in enumerate(self.toasts[-3:]):
            ev = t["event"]
            frac = t["life"] / t["max_life"]
            alpha = int(255 * min(1.0, frac * 2.0))

            tx = (sw - toast_w) // 2
            ty = start_y + i * (toast_h + 6)

            t_surf = pygame.Surface((toast_w, toast_h), pygame.SRCALPHA)
            t_surf.fill((15, 22, 35, int(alpha * 0.9)))
            pygame.draw.rect(t_surf, (*BORDER_GLOW[:3], alpha), (0, 0, toast_w, toast_h), 1, border_radius=4)

            # Badge & text
            b_tag, b_col = KIND_BADGES.get(ev.kind, ("EVENT", TEXT_SECONDARY))
            badge_txt = theme.font_xs.render(f"[{b_tag}] G{ev.generation:.1f}", True, b_col)
            t_surf.blit(badge_txt, (10, 6))

            desc_txt = theme.font_xs.render(ev.text[:48], True, TEXT_PRIMARY)
            t_surf.blit(desc_txt, (100, 6))

            surface.blit(t_surf, (tx, ty))
