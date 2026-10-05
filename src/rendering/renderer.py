"""Master Pygame Renderer coordinating Camera, World, Organisms, Overlays, and UI Panels."""
from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pygame

from ..core.util import hsv_rgb
from ..evolution.genome import G_ATTACK, G_DIET, G_SIZE, G_SPEED
from ..simulation.world import ROCK, TOXIC, TREE, WATER
from .camera import Camera
from .overlays import OverlayManager
from .particles import ParticleSystem
from .ui.dashboard import DashboardPanel
from .ui.god_mode import GodModePanel
from .ui.inspector import OrganismInspector
from .ui.intro import IntroModal
from .ui.species_tree import SpeciesTreePanel
from .ui.theme import (
    BG_DARK,
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
from .ui.timeline_panel import TimelinePanel
from .ui.widgets import Button


class MasterRenderer:
    def __init__(self, screen_w: int = 1440, screen_h: int = 860, title: str = "Digital Evolution Lab"):
        pygame.init()
        pygame.font.init()
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.screen = pygame.display.set_mode((screen_w, screen_h), pygame.RESIZABLE | pygame.DOUBLEBUF)
        pygame.display.set_caption(title)

        self.theme = Theme()
        self.camera: Optional[Camera] = None
        self.particles = ParticleSystem()
        self.overlays: Optional[OverlayManager] = None

        # Time controls
        self.paused = False
        self.speed_multiplier = 1.0
        self.speed_options = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

        # Top HUD buttons
        self.btn_pause = Button(pygame.Rect(10, 8, 70, 24), "PAUSE", self.toggle_pause)
        self.speed_buttons: List[Button] = []
        sx = 88
        for spd in self.speed_options:
            lbl = f"{spd}x" if spd >= 1.0 else f"{spd}x"
            b = Button(
                pygame.Rect(sx, 8, 42, 24),
                lbl,
                lambda s=spd: self.set_speed(s),
            )
            self.speed_buttons.append(b)
            sx += 46

        # Panel toggle buttons on HUD right
        self.btn_hud_dash = Button(pygame.Rect(screen_w - 530, 8, 90, 24), "DASHBOARD", self.toggle_dashboard)
        self.btn_hud_time = Button(pygame.Rect(screen_w - 434, 8, 80, 24), "TIMELINE", self.toggle_timeline)
        self.btn_hud_spec = Button(pygame.Rect(screen_w - 348, 8, 75, 24), "SPECIES", self.toggle_species)
        self.btn_hud_god = Button(pygame.Rect(screen_w - 267, 8, 80, 24), "GOD MODE", self.toggle_god_mode)
        self.btn_hud_over = Button(pygame.Rect(screen_w - 181, 8, 80, 24), "OVERLAYS", self.toggle_overlays)
        self.btn_hud_help = Button(pygame.Rect(screen_w - 95, 8, 85, 24), "HELP [F1]", self.toggle_help)

        # UI Panels
        self.inspector = OrganismInspector(pygame.Rect(screen_w - 340, 42, 330, 560))
        self.dashboard = DashboardPanel(pygame.Rect(20, 42, 700, 520))
        self.god_mode = GodModePanel(pygame.Rect(20, 42, 330, 440))
        self.timeline = TimelinePanel(pygame.Rect(screen_w - 450, 42, 430, 500))
        self.species_tree = SpeciesTreePanel(pygame.Rect(20, 42, 500, 540))
        self.intro = IntroModal(screen_w, screen_h)

        # Internal state
        self.selected_org_id: Optional[int] = None
        self.last_sim_time = 0.0

    def init_simulation(self, eco) -> None:
        self.camera = Camera(self.screen_w, self.screen_h, eco.world.width, eco.world.height)
        self.overlays = OverlayManager(eco.world.width, eco.world.height)

    def set_speed(self, spd: float) -> None:
        self.speed_multiplier = spd

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.btn_pause.text = "RESUME" if self.paused else "PAUSE"

    def toggle_dashboard(self) -> None:
        self.dashboard.toggle()

    def toggle_timeline(self) -> None:
        self.timeline.toggle()

    def toggle_species(self) -> None:
        self.species_tree.toggle()

    def toggle_god_mode(self) -> None:
        self.god_mode.toggle()

    def toggle_overlays(self) -> None:
        if self.overlays:
            self.overlays.show_trails = not self.overlays.show_trails
            self.overlays.show_temperature = not self.overlays.show_temperature

    def toggle_help(self) -> None:
        self.intro.toggle()

    def save_screenshot(self, filepath: str) -> None:
        """Capture strictly the Pygame window surface and save to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        pygame.image.save(self.screen, filepath)

    def handle_event(self, event: pygame.event.Event, eco) -> bool:
        if event.type == pygame.VIDEORESIZE:
            self.screen_w = max(800, event.w)
            self.screen_h = max(600, event.h)
            self.screen = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.RESIZABLE | pygame.DOUBLEBUF)
            if self.camera:
                self.camera.resize(self.screen_w, self.screen_h)
            return True

        # Let modal dialog consume first
        if self.intro.handle_event(event):
            return True

        # Top HUD buttons
        if self.btn_pause.handle_event(event):
            return True
        for b in self.speed_buttons:
            if b.handle_event(event):
                return True
        if self.btn_hud_dash.handle_event(event):
            return True
        if self.btn_hud_time.handle_event(event):
            return True
        if self.btn_hud_spec.handle_event(event):
            return True
        if self.btn_hud_god.handle_event(event):
            return True
        if self.btn_hud_over.handle_event(event):
            return True
        if self.btn_hud_help.handle_event(event):
            return True

        # Panels
        if self.inspector.handle_event(event, eco, self.camera):
            return True
        if self.dashboard.handle_event(event):
            return True
        if self.god_mode.handle_event(event, eco):
            return True
        if self.timeline.handle_event(event):
            return True
        if self.species_tree.handle_event(event, eco, self.camera):
            return True

        # Keyboard shortcuts
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.toggle_pause()
                return True
            elif event.key == pygame.K_1:
                self.set_speed(0.5)
                return True
            elif event.key == pygame.K_2:
                self.set_speed(1.0)
                return True
            elif event.key == pygame.K_3:
                self.set_speed(2.0)
                return True
            elif event.key == pygame.K_4:
                self.set_speed(5.0)
                return True
            elif event.key == pygame.K_5:
                self.set_speed(20.0)
                return True
            elif event.key == pygame.K_d:
                self.toggle_dashboard()
                return True
            elif event.key == pygame.K_t:
                self.toggle_timeline()
                return True
            elif event.key == pygame.K_s:
                self.toggle_species()
                return True
            elif event.key == pygame.K_g:
                self.toggle_god_mode()
                return True
            elif event.key == pygame.K_o:
                self.toggle_overlays()
                return True
            elif event.key in (pygame.K_h, pygame.K_F1):
                self.toggle_help()
                return True
            elif event.key == pygame.K_TAB:
                # Cycle focus to next live organism
                s = eco.org
                if s.n > 0:
                    curr_idx = s.index_of(self.selected_org_id) if self.selected_org_id is not None else -1
                    next_idx = (curr_idx + 1) % s.n
                    self.selected_org_id = int(s.id[next_idx])
                    self.inspector.set_organism(self.selected_org_id, eco)
                    if self.camera:
                        self.camera.following_id = self.selected_org_id
                return True

        # Mouse camera controls
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in (2, 3):  # Middle or Right drag pan
                if self.camera:
                    self.camera.dragging = True
                    self.camera.last_mouse = event.pos
                return True
            elif event.button == 1:
                # Left click in world space
                if self.camera:
                    wx, wy = self.camera.screen_to_world(event.pos[0], event.pos[1])
                    # If God Mode active with brush
                    if self.god_mode.visible and self.god_mode.active_tool != "inspect":
                        self.god_mode.apply_brush(eco, wx, wy)
                        return True
                    # Else select organism
                    near_idx = eco.nearest_organism(wx, wy, 25.0)
                    if near_idx >= 0:
                        self.selected_org_id = int(eco.org.id[near_idx])
                        self.inspector.set_organism(self.selected_org_id, eco)
                    else:
                        self.selected_org_id = None
                        self.inspector.set_organism(None, eco)
                return True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button in (2, 3):
                if self.camera:
                    self.camera.dragging = False
                return True

        elif event.type == pygame.MOUSEMOTION:
            if self.camera and self.camera.dragging:
                dx = event.pos[0] - self.camera.last_mouse[0]
                dy = event.pos[1] - self.camera.last_mouse[1]
                self.camera.x -= dx / self.camera.zoom
                self.camera.y -= dy / self.camera.zoom
                self.camera.last_mouse = event.pos
                self.camera.following_id = None
                self.camera.following_species_id = None
                return True

        elif event.type == pygame.MOUSEWHEEL:
            if self.camera:
                factor = 1.15 if event.y > 0 else 0.85
                mx, my = pygame.mouse.get_pos()
                self.camera.zoom_at(mx, my, factor)
                return True

        return False

    def render(self, eco, dt: float, sim_fps: float = 0.0) -> None:
        if self.camera is None or self.overlays is None:
            self.init_simulation(eco)

        # Process ecosystem particle events
        if eco.fx:
            self.particles.process_ecosystem_fx(eco.fx)
            eco.fx.clear()

        # Update visual systems
        self.camera.update(eco, dt)
        self.particles.update(dt)
        self.overlays.update_trails(eco, dt)
        self.timeline.update_toasts(eco, dt)

        # Clear background
        self.screen.fill(BG_DARK)

        # 1. World Terrain
        self._render_terrain(eco)

        # 2. Temperature Heatmap overlay
        if self.overlays.show_temperature:
            self.overlays.draw_temperature_heatmap(self.screen, eco, self.camera)

        # 3. Trails overlay
        if self.overlays.show_trails:
            self.overlays.draw_trails(self.screen, self.camera)

        # 4. Food store
        self._render_food(eco)

        # 5. Organisms
        self._render_organisms(eco)

        # 6. Particles
        self.particles.draw(self.screen, self.camera)

        # 7. World boundaries
        self._render_world_border(eco)

        # 8. Top Control Bar & HUD
        self._render_hud(eco, sim_fps)

        # 9. UI Panels
        self.inspector.draw(self.screen, eco, self.camera, self.theme)
        self.dashboard.draw(self.screen, self.theme, eco)
        self.god_mode.draw(self.screen, self.theme, dt)
        self.timeline.draw(self.screen, self.theme, eco)
        self.species_tree.draw(self.screen, self.theme, eco)
        self.intro.draw(self.screen, self.theme)

        pygame.display.flip()

    def _render_world_border(self, eco) -> None:
        w, h = eco.world.width, eco.world.height
        p1 = self.camera.world_to_screen(0, 0)
        p2 = self.camera.world_to_screen(w, 0)
        p3 = self.camera.world_to_screen(w, h)
        p4 = self.camera.world_to_screen(0, h)
        pygame.draw.lines(self.screen, BORDER_GLOW, True, [p1, p2, p3, p4], 2)

    def _render_terrain(self, eco) -> None:
        w = eco.world
        cam = self.camera
        cell = w.cell

        c_min_x, c_min_y = cam.screen_to_world(0, 0)
        c_max_x, c_max_y = cam.screen_to_world(cam.screen_w, cam.screen_h)

        ix0 = max(0, int(c_min_x / cell))
        ix1 = min(w.gw, int(c_max_x / cell) + 1)
        iy0 = max(0, int(c_min_y / cell))
        iy1 = min(w.gh, int(c_max_y / cell) + 1)

        block_w = max(1, int(cell * cam.zoom) + 1)
        block_h = max(1, int(cell * cam.zoom) + 1)

        grid = w.terrain
        for iy in range(iy0, iy1):
            for ix in range(ix0, ix1):
                val = grid[iy, ix]
                if val == 0:
                    continue
                sx, sy = cam.world_to_screen(ix * cell, iy * cell)

                if val == WATER:
                    col = (12, 28, 48)
                elif val == ROCK:
                    col = (45, 54, 66)
                elif val == TREE:
                    col = (16, 42, 28)
                elif val == TOXIC:
                    col = (42, 16, 45)
                else:
                    continue

                pygame.draw.rect(self.screen, col, (sx, sy, block_w, block_h))

    def _render_food(self, eco) -> None:
        f = eco.food
        n = f.n
        if n == 0:
            return

        cam = self.camera
        pos = f.pos[:n]
        energy = f.energy[:n]
        kind = f.kind[:n]

        KIND_COLORS = {
            0: (74, 222, 128),   # Berry / Grass (Emerald)
            1: (251, 191, 36),   # Fruit (Amber)
            2: (192, 132, 252),  # Toxic Bloom (Violet)
            3: (248, 113, 113),  # Carrion (Crimson)
        }

        zoom = cam.zoom
        for i in range(n):
            wx, wy = pos[i, 0], pos[i, 1]
            if cam.is_visible(wx, wy, 8.0):
                sx, sy = cam.world_to_screen(wx, wy)
                rad = max(2, int(min(6.0, math.sqrt(energy[i]) * 0.8) * zoom))
                c = KIND_COLORS.get(int(kind[i]), (74, 222, 128))
                pygame.draw.circle(self.screen, c, (sx, sy), rad)

    def _render_organisms(self, eco) -> None:
        s = eco.org
        n = s.n
        if n == 0:
            return

        cam = self.camera
        pos = s.pos[:n]
        ang = s.ang[:n]
        traits = s.traits[:n]
        health = s.health[:n]
        species_ids = s.species[:n]
        sp_dict = getattr(eco.species, "species", {})

        zoom = cam.zoom

        for i in range(n):
            wx, wy = pos[i, 0], pos[i, 1]
            sz = traits[i, G_SIZE]
            if not cam.is_visible(wx, wy, sz * 2.0):
                continue

            sx, sy = cam.world_to_screen(wx, wy)
            rad = max(3, int(sz * zoom))

            # Species color
            sid = int(species_ids[i])
            sp = sp_dict.get(sid)
            if sp:
                body_color = sp.color
            else:
                diet = traits[i, G_DIET]
                body_color = (
                    int(230 * diet + 50 * (1 - diet)),
                    int(80 * diet + 210 * (1 - diet)),
                    int(90 * diet + 140 * (1 - diet)),
                )

            # Draw body circle
            pygame.draw.circle(self.screen, body_color, (sx, sy), rad)

            # Heading pointer / orientation indicator
            a = ang[i]
            nose_len = rad + max(2, int(4 * zoom))
            nx = int(sx + math.cos(a) * nose_len)
            ny = int(sy + math.sin(a) * nose_len)
            pygame.draw.line(self.screen, TEXT_PRIMARY, (sx, sy), (nx, ny), max(1, int(zoom)))

            # If damaged, draw outer warning ring
            if health[i] < 60.0:
                pygame.draw.circle(self.screen, TEXT_DANGER, (sx, sy), rad + 2, 1)

            # Highlight selected organism
            if self.selected_org_id is not None and int(s.id[i]) == self.selected_org_id:
                # Reticle ring
                pygame.draw.circle(self.screen, BORDER_GLOW, (sx, sy), rad + 6, 2)
                # Vision cone overlay
                self.overlays.draw_vision_cone(self.screen, eco, i, cam)

    def _render_hud(self, eco, sim_fps: float) -> None:
        # Top banner glass bar
        hud_rect = pygame.Rect(0, 0, self.screen_w, 40)
        hud_surf = pygame.Surface((self.screen_w, 40), pygame.SRCALPHA)
        hud_surf.fill((10, 15, 24, 235))
        self.screen.blit(hud_surf, (0, 0))
        pygame.draw.line(self.screen, BORDER_DEFAULT, (0, 40), (self.screen_w, 40), 1)

        # Time controls
        self.btn_pause.draw(self.screen, self.theme)
        for b in self.speed_buttons:
            b.active = abs(float(b.text.replace("x", "")) - self.speed_multiplier) < 1e-4
            b.draw(self.screen, self.theme)

        # Telemetry metrics
        n_pop = eco.org.n
        n_food = eco.food.n
        gen = eco.stats.history["generation"][-1] if eco.stats.history["generation"] else 0.0
        sec = eco.time

        m_x = 370
        stat_items = [
            ("POP", f"{n_pop}", TEXT_PRIMARY),
            ("FOOD", f"{n_food}", TEXT_SUCCESS),
            ("GEN", f"{gen:.1f}", TEXT_ACCENT),
            ("TIME", f"{sec:.0f}s", TEXT_PURPLE),
            ("SIM TPS", f"{sim_fps:.0f}", TEXT_SECONDARY),
        ]

        for lbl, val, col in stat_items:
            t1 = self.theme.font_xs.render(lbl, True, TEXT_MUTED)
            t2 = self.theme.font_sm.render(val, True, col)
            self.screen.blit(t1, (m_x, 6))
            self.screen.blit(t2, (m_x, 19))
            m_x += max(t1.get_width(), t2.get_width()) + 18

        # Panel toggle buttons
        self.btn_hud_dash.active = self.dashboard.visible
        self.btn_hud_time.active = self.timeline.visible
        self.btn_hud_spec.active = self.species_tree.visible
        self.btn_hud_god.active = self.god_mode.visible
        self.btn_hud_help.active = self.intro.visible

        self.btn_hud_dash.draw(self.screen, self.theme)
        self.btn_hud_time.draw(self.screen, self.theme)
        self.btn_hud_spec.draw(self.screen, self.theme)
        self.btn_hud_god.draw(self.screen, self.theme)
        self.btn_hud_over.draw(self.screen, self.theme)
        self.btn_hud_help.draw(self.screen, self.theme)
