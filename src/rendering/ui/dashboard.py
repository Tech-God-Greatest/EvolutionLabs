"""Real-time Statistics Dashboard & Multi-Chart Telemetry Panel (§39, §41)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
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


class DashboardPanel:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.visible = False
        self.tab = "overview"  # "overview", "population", "genetics", "ecology"

        # Close button
        self.btn_close = Button(
            pygame.Rect(rect.right - 26, rect.y + 6, 20, 20),
            "X",
            self.toggle,
        )

        # Tab buttons
        tab_y = rect.y + 36
        tw = 95
        self.btn_tab_overview = Button(
            pygame.Rect(rect.x + 12, tab_y, tw, 22),
            "OVERVIEW",
            lambda: self.set_tab("overview"),
        )
        self.btn_tab_pop = Button(
            pygame.Rect(rect.x + 12 + tw + 6, tab_y, tw, 22),
            "POPULATION",
            lambda: self.set_tab("population"),
        )
        self.btn_tab_gen = Button(
            pygame.Rect(rect.x + 12 + (tw + 6) * 2, tab_y, tw, 22),
            "GENETICS",
            lambda: self.set_tab("genetics"),
        )
        self.btn_tab_eco = Button(
            pygame.Rect(rect.x + 12 + (tw + 6) * 3, tab_y, tw, 22),
            "ECOLOGY",
            lambda: self.set_tab("ecology"),
        )

    def toggle(self) -> None:
        self.visible = not self.visible

    def set_tab(self, tab_name: str) -> None:
        self.tab = tab_name

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False

        if self.btn_close.handle_event(event):
            return True

        if self.btn_tab_overview.handle_event(event):
            return True
        if self.btn_tab_pop.handle_event(event):
            return True
        if self.btn_tab_gen.handle_event(event):
            return True
        if self.btn_tab_eco.handle_event(event):
            return True

        # Consume clicks within dashboard panel so world doesn't receive them
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            if self.rect.collidepoint(event.pos):
                return True

        return False

    def draw(self, surface: pygame.Surface, theme: Theme, eco) -> None:
        if not self.visible:
            return

        theme.draw_glass_panel(
            surface,
            self.rect,
            glow=True,
            header_height=30,
            title="ANALYTICS & METRICS DASHBOARD",
        )

        self.btn_close.draw(surface, theme)

        # Draw tab bar
        self.btn_tab_overview.active = self.tab == "overview"
        self.btn_tab_pop.active = self.tab == "population"
        self.btn_tab_gen.active = self.tab == "genetics"
        self.btn_tab_eco.active = self.tab == "ecology"

        self.btn_tab_overview.draw(surface, theme)
        self.btn_tab_pop.draw(surface, theme)
        self.btn_tab_gen.draw(surface, theme)
        self.btn_tab_eco.draw(surface, theme)

        hist = eco.stats.history
        content_rect = pygame.Rect(self.rect.x + 12, self.rect.y + 68, self.rect.w - 24, self.rect.h - 80)

        if self.tab == "overview":
            self._draw_overview(surface, theme, eco, content_rect)
        elif self.tab == "population":
            self._draw_population(surface, theme, hist, content_rect)
        elif self.tab == "genetics":
            self._draw_genetics(surface, theme, hist, content_rect)
        elif self.tab == "ecology":
            self._draw_ecology(surface, theme, hist, content_rect)

    def _draw_overview(self, surface: pygame.Surface, theme: Theme, eco, area: pygame.Rect) -> None:
        # Top KPI Cards
        card_w = (area.w - 24) // 4
        card_h = 58
        y = area.y

        n_pop = eco.org.n
        n_food = eco.food.n
        sp_mgr = getattr(eco, "species", getattr(eco, "species_tracker", None))
        n_spec = len(sp_mgr.active_ids()) if sp_mgr else 0
        gen = eco.stats.history["generation"][-1] if eco.stats.history["generation"] else 0.0

        kpis = [
            ("POPULATION", f"{n_pop}", TEXT_PRIMARY),
            ("FOOD SUPPLY", f"{n_food}", TEXT_SUCCESS),
            ("ACTIVE SPECIES", f"{n_spec}", TEXT_ACCENT),
            ("GENERATION", f"{gen:.1f}", TEXT_PURPLE),
        ]

        for i, (label, val, col) in enumerate(kpis):
            cx = area.x + i * (card_w + 8)
            card_rect = pygame.Rect(cx, y, card_w, card_h)
            pygame.draw.rect(surface, BTN_NORMAL, card_rect, border_radius=4)
            pygame.draw.rect(surface, BORDER_DEFAULT, card_rect, 1, border_radius=4)

            lbl = theme.font_xs.render(label, True, TEXT_SECONDARY)
            vtxt = theme.font_lg.render(val, True, col)
            surface.blit(lbl, (cx + 8, y + 6))
            surface.blit(vtxt, (cx + 8, y + 24))

        # Two charts below
        chart_y = y + card_h + 12
        chart_h = (area.bottom - chart_y - 8) // 2

        c1_rect = pygame.Rect(area.x, chart_y, area.w, chart_h)
        self._draw_chart(
            surface,
            theme,
            c1_rect,
            title="POPULATION & FOOD CARRIER RATIO",
            series=[
                ("Population", eco.stats.history.get("population", []), TEXT_ACCENT),
                ("Food", eco.stats.history.get("food_count", []), TEXT_SUCCESS),
            ],
        )

        c2_rect = pygame.Rect(area.x, chart_y + chart_h + 8, area.w, chart_h)
        self._draw_chart(
            surface,
            theme,
            c2_rect,
            title="TROPHIC GUILDS (PREDATORS VS HERBIVORES)",
            series=[
                ("Herbivores", eco.stats.history.get("herbivores", []), TEXT_SUCCESS),
                ("Predators", eco.stats.history.get("predators", []), TEXT_DANGER),
            ],
        )

    def _draw_population(self, surface: pygame.Surface, theme: Theme, hist: Dict[str, List[float]], area: pygame.Rect) -> None:
        ch = (area.h - 8) // 2
        c1 = pygame.Rect(area.x, area.y, area.w, ch)
        c2 = pygame.Rect(area.x, area.y + ch + 8, area.w, ch)

        self._draw_chart(
            surface,
            theme,
            c1,
            title="SPECIES EMERGENCE & EXTINCTION RATES",
            series=[
                ("Active Species", hist.get("active_species", []), TEXT_ACCENT),
                ("Total Extinctions", hist.get("total_extinctions", []), TEXT_DANGER),
            ],
        )

        self._draw_chart(
            surface,
            theme,
            c2,
            title="BIRTHS VS DEATHS DYNAMICS",
            series=[
                ("Birth Rate", hist.get("birth_rate", []), TEXT_SUCCESS),
                ("Death Rate", hist.get("death_rate", []), TEXT_WARNING),
            ],
        )

    def _draw_genetics(self, surface: pygame.Surface, theme: Theme, hist: Dict[str, List[float]], area: pygame.Rect) -> None:
        ch = (area.h - 8) // 2
        c1 = pygame.Rect(area.x, area.y, area.w, ch)
        c2 = pygame.Rect(area.x, area.y + ch + 8, area.w, ch)

        self._draw_chart(
            surface,
            theme,
            c1,
            title="GENETIC DIVERSITY & FITNESS EVOLUTION",
            series=[
                ("Diversity", hist.get("genetic_diversity", []), TEXT_PURPLE),
                ("Avg Fitness", hist.get("avg_fitness", []), TEXT_ACCENT),
            ],
        )

        self._draw_chart(
            surface,
            theme,
            c2,
            title="MORPHOLOGICAL TRAIT DRIFT (NORM)",
            series=[
                ("Speed", hist.get("avg_speed", []), TEXT_WARNING),
                ("Vision", hist.get("avg_vision", []), TEXT_ACCENT),
                ("Size", hist.get("avg_size", []), TEXT_PRIMARY),
                ("Metabolism", hist.get("avg_metabolism", []), TEXT_DANGER),
            ],
        )

    def _draw_ecology(self, surface: pygame.Surface, theme: Theme, hist: Dict[str, List[float]], area: pygame.Rect) -> None:
        ch = (area.h - 8) // 2
        c1 = pygame.Rect(area.x, area.y, area.w, ch)
        c2 = pygame.Rect(area.x, area.y + ch + 8, area.w, ch)

        self._draw_chart(
            surface,
            theme,
            c1,
            title="ECOSYSTEM FOOD ENERGY RESERVES",
            series=[
                ("Food Energy", hist.get("food_energy", []), TEXT_SUCCESS),
                ("Organism Energy", hist.get("total_energy", []), TEXT_ACCENT),
            ],
        )

        self._draw_chart(
            surface,
            theme,
            c2,
            title="POPULATION LIFESPAN",
            series=[
                ("Lifespan", hist.get("avg_lifespan", []), TEXT_PRIMARY),
            ],
        )

    def _draw_chart(
        self,
        surface: pygame.Surface,
        theme: Theme,
        rect: pygame.Rect,
        title: str,
        series: List[Tuple[str, List[float], Tuple[int, int, int]]],
    ) -> None:
        # Background box
        pygame.draw.rect(surface, (14, 20, 30), rect, border_radius=4)
        pygame.draw.rect(surface, BORDER_DEFAULT, rect, 1, border_radius=4)

        # Title
        t_surf = theme.font_xs.render(title, True, TEXT_SECONDARY)
        surface.blit(t_surf, (rect.x + 8, rect.y + 6))

        # Legend on the right
        leg_x = rect.right - 10
        for name, _, col in reversed(series):
            lbl = theme.font_xs.render(name, True, col)
            leg_x -= lbl.get_width() + 14
            pygame.draw.circle(surface, col, (leg_x + 5, rect.y + 11), 3)
            surface.blit(lbl, (leg_x + 12, rect.y + 6))

        # Chart plot area
        plot_rect = pygame.Rect(rect.x + 36, rect.y + 24, rect.w - 46, rect.h - 32)
        pygame.draw.line(surface, (25, 36, 52), (plot_rect.x, plot_rect.bottom), (plot_rect.right, plot_rect.bottom), 1)
        pygame.draw.line(surface, (25, 36, 52), (plot_rect.x, plot_rect.y), (plot_rect.x, plot_rect.bottom), 1)

        # Find min/max across series
        all_vals = [v for _, s, _ in series for v in s[-120:] if v is not None]
        if not all_vals:
            no_data = theme.font_xs.render("ACQUIRING TELEMETRY...", True, TEXT_MUTED)
            surface.blit(no_data, (plot_rect.centerx - no_data.get_width() // 2, plot_rect.centery - 6))
            return

        min_val = min(all_vals)
        max_val = max(all_vals)
        if abs(max_val - min_val) < 1e-6:
            max_val += 1.0

        # Draw min/max text
        max_txt = theme.font_xs.render(f"{max_val:.0f}", True, TEXT_MUTED)
        min_txt = theme.font_xs.render(f"{min_val:.0f}", True, TEXT_MUTED)
        surface.blit(max_txt, (rect.x + 4, plot_rect.y - 2))
        surface.blit(min_txt, (rect.x + 4, plot_rect.bottom - 10))

        # Draw grid line
        mid_y = plot_rect.y + plot_rect.h // 2
        pygame.draw.line(surface, (20, 28, 40), (plot_rect.x, mid_y), (plot_rect.right, mid_y), 1)

        # Plot lines
        max_pts = 120
        for _, data, col in series:
            pts_slice = data[-max_pts:]
            if len(pts_slice) < 2:
                continue

            pts = []
            for i, val in enumerate(pts_slice):
                norm_x = i / (max_pts - 1)
                norm_y = (val - min_val) / (max_val - min_val)
                px = int(plot_rect.x + norm_x * plot_rect.w)
                py = int(plot_rect.bottom - norm_y * plot_rect.h)
                py = max(plot_rect.y, min(plot_rect.bottom, py))
                pts.append((px, py))

            if len(pts) >= 2:
                pygame.draw.lines(surface, col, False, pts, 2)
