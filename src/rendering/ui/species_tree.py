"""Phylogenetic Tree & Species Genealogy Panel (§27, §41)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import pygame

from ...evolution.genome import G_ATTACK, G_DIET, G_SIZE, G_SPEED, G_VIS
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


class SpeciesTreePanel:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.visible = False
        self.selected_sid: Optional[int] = None
        self.scroll_offset = 0

        # Close button
        self.btn_close = Button(pygame.Rect(rect.right - 26, rect.y + 6, 20, 20), "X", self.toggle)
        # Follow button
        self.btn_follow = Button(pygame.Rect(rect.x + 12, rect.y + 35, 110, 22), "FOLLOW SPECIES")

    def toggle(self) -> None:
        self.visible = not self.visible

    def handle_event(self, event: pygame.event.Event, eco=None, camera=None) -> bool:
        if not self.visible:
            return False

        if self.btn_close.handle_event(event):
            return True

        if self.btn_follow.handle_event(event):
            if camera and self.selected_sid is not None:
                camera.following_species_id = self.selected_sid
                camera.following_id = None
            return True

        if event.type == pygame.MOUSEWHEEL:
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll_offset = max(0, self.scroll_offset - event.y * 2)
                return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                # Click on species row
                list_y = self.rect.y + 68
                row_h = 32
                if event.pos[1] >= list_y and event.pos[1] <= self.rect.bottom - 100:
                    idx = (event.pos[1] - list_y) // row_h + self.scroll_offset
                    if eco:
                        sp_dict = getattr(eco.species, "species", {})
                        all_specs = list(sp_dict.values())
                        if 0 <= idx < len(all_specs):
                            self.selected_sid = all_specs[idx].id
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
            title="PHYLOGENETIC SPECIES TREE",
        )

        self.btn_close.draw(surface, theme)
        self.btn_follow.draw(surface, theme)

        sp_dict = getattr(eco.species, "species", {})
        all_specs = list(sp_dict.values())
        n_active = sum(1 for s in all_specs if s.alive)
        n_extinct = sum(1 for s in all_specs if not s.alive)

        stat_txt = theme.font_xs.render(f"ACTIVE: {n_active}  |  EXTINCT: {n_extinct}", True, TEXT_SECONDARY)
        surface.blit(stat_txt, (self.rect.x + 130, self.rect.y + 39))

        # List of species
        list_rect = pygame.Rect(self.rect.x + 10, self.rect.y + 68, self.rect.w - 20, self.rect.h - 170)
        pygame.draw.rect(surface, (12, 18, 28), list_rect, border_radius=3)
        pygame.draw.rect(surface, BORDER_DEFAULT, list_rect, 1, border_radius=3)

        row_h = 32
        visible_rows = list_rect.h // row_h
        start_idx = self.scroll_offset
        end_idx = min(len(all_specs), start_idx + visible_rows)

        for i, sp in enumerate(all_specs[start_idx:end_idx]):
            ry = list_rect.y + i * row_h
            r_rect = pygame.Rect(list_rect.x + 2, ry + 2, list_rect.w - 4, row_h - 4)

            # Selected highlight
            is_sel = self.selected_sid == sp.id
            if is_sel:
                pygame.draw.rect(surface, (28, 48, 72), r_rect, border_radius=2)
                pygame.draw.rect(surface, BORDER_GLOW, r_rect, 1, border_radius=2)

            # Color dot
            pygame.draw.circle(surface, sp.color, (r_rect.x + 12, r_rect.y + row_h // 2 - 1), 5)

            # Name and lineage
            p_str = f" <- Sp#{sp.parent}" if sp.parent is not None else " [FOUNDER]"
            name_txt = theme.font_xs.render(f"{sp.name}{p_str}", True, TEXT_PRIMARY if sp.alive else TEXT_MUTED)
            surface.blit(name_txt, (r_rect.x + 24, r_rect.y + 7))

            # Pop badge
            pop_col = TEXT_SUCCESS if sp.population > 0 else TEXT_DANGER
            pop_txt = theme.font_xs.render(f"Pop: {sp.population}", True, pop_col)
            surface.blit(pop_txt, (r_rect.right - 70, r_rect.y + 7))

        # Selected species details footer
        sp_dict = getattr(eco.species, "species", {})
        if self.selected_sid is not None and self.selected_sid in sp_dict:
            sel_sp = sp_dict[self.selected_sid]
            foot_rect = pygame.Rect(self.rect.x + 10, self.rect.bottom - 95, self.rect.w - 20, 85)
            pygame.draw.rect(surface, (16, 24, 38), foot_rect, border_radius=4)
            pygame.draw.rect(surface, BORDER_DEFAULT, foot_rect, 1, border_radius=4)

            # Title
            t_str = f"SPECIES: {sel_sp.name.upper()} (ID: {sel_sp.id})"
            ts = theme.font_sm.render(t_str, True, TEXT_ACCENT)
            surface.blit(ts, (foot_rect.x + 10, foot_rect.y + 8))

            # Traits
            diet_str = "Carnivore" if sel_sp.rep_t[G_DIET] > 0.5 else "Herbivore"
            spd = sel_sp.rep_t[G_SPEED]
            sz = sel_sp.rep_t[G_SIZE]
            vis = sel_sp.rep_t[G_VIS]
            atk = sel_sp.rep_t[G_ATTACK]

            line1 = f"Trophic Role: {diet_str}  |  Peak Pop: {sel_sp.peak_population}  |  Total Members: {sel_sp.total_members}"
            line2 = f"Traits: Speed: {spd:.1f} | Size: {sz:.1f} | Vision: {vis:.0f} | Attack: {atk:.1f}"
            t1 = theme.font_xs.render(line1, True, TEXT_SECONDARY)
            t2 = theme.font_xs.render(line2, True, TEXT_MUTED)
            surface.blit(t1, (foot_rect.x + 10, foot_rect.y + 34))
            surface.blit(t2, (foot_rect.x + 10, foot_rect.y + 56))
