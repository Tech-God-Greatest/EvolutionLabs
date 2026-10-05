"""Scientific Overlays: Heatmaps, Vision Cones, Fading Trails, and Territories."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pygame


class OverlayManager:
    def __init__(self, world_w: float, world_h: float):
        self.world_w = world_w
        self.world_h = world_h
        # Toggles
        self.show_food = True
        self.show_temperature = False
        self.show_species = True
        self.show_territories = True
        self.show_vision_cones = False
        self.show_neural_activity = True
        self.show_trails = True
        self.show_density = False

        # Persistent surface for fading trails
        self.trail_cell = 6  # coarse grid for world trails
        self.tw = int(world_w / self.trail_cell)
        self.th = int(world_h / self.trail_cell)
        self.trail_grid = np.zeros((self.th, self.tw), dtype=np.float32)
        self.trail_surface: Optional[pygame.Surface] = None

        # Precomputed temperature surface
        self._temp_surf: Optional[pygame.Surface] = None
        self._temp_surf_version = -1

    def update_trails(self, eco, dt: float) -> None:
        if not self.show_trails:
            return
        # Fade existing trails
        self.trail_grid *= (0.96 ** (dt * 10))

        s = eco.org
        n = s.n
        if n == 0:
            return

        # Deposit trail energy
        pos = s.pos[:n]
        gx = np.clip((pos[:, 0] / self.trail_cell).astype(int), 0, self.tw - 1)
        gy = np.clip((pos[:, 1] / self.trail_cell).astype(int), 0, self.th - 1)
        np.add.at(self.trail_grid, (gy, gx), 0.15)
        np.clip(self.trail_grid, 0.0, 1.0, out=self.trail_grid)

    def draw_trails(self, surface: pygame.Surface, camera) -> None:
        if not self.show_trails or camera.zoom < 0.2:
            return

        # Render active cells
        active_y, active_x = np.nonzero(self.trail_grid > 0.08)
        if len(active_x) == 0:
            return

        # Sample a subset if too dense for speed
        step = 1 if len(active_x) < 4000 else 2
        for i in range(0, len(active_x), step):
            gx = active_x[i]
            gy = active_y[i]
            val = self.trail_grid[gy, gx]
            wx = (gx + 0.5) * self.trail_cell
            wy = (gy + 0.5) * self.trail_cell
            if camera.is_visible(wx, wy, self.trail_cell):
                sx, sy = camera.world_to_screen(wx, wy)
                rad = max(1, int(self.trail_cell * camera.zoom * 0.8))
                alpha = int(val * 90)
                color = (40, 180, 240, alpha)
                # Quick rect or circle
                s = pygame.Surface((rad * 2, rad * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (40, 180, 240, alpha), (rad, rad), rad)
                surface.blit(s, (sx - rad, sy - rad))

    def draw_temperature_heatmap(self, surface: pygame.Surface, eco, camera) -> None:
        if not self.show_temperature:
            return
        w = eco.world
        env = eco.env

        # Coarse render of temperature cells
        cell_size = w.cell
        # Draw on screen where visible
        c_min_x, c_min_y = camera.screen_to_world(0, 0)
        c_max_x, c_max_y = camera.screen_to_world(camera.screen_w, camera.screen_h)

        i_x0 = max(0, int(c_min_x / cell_size))
        i_x1 = min(w.gw, int(c_max_x / cell_size) + 1)
        i_y0 = max(0, int(c_min_y / cell_size))
        i_y1 = min(w.gh, int(c_max_y / cell_size) + 1)

        temp_sub = w.temp[i_y0:i_y1, i_x0:i_x1] + env.temp_offset
        norm_temp = np.clip(temp_sub, 0.0, 1.0)

        # Draw blended cells
        block_w = max(1, int(cell_size * camera.zoom) + 1)
        block_h = max(1, int(cell_size * camera.zoom) + 1)

        overlay = pygame.Surface((camera.screen_w, camera.screen_h), pygame.SRCALPHA)
        for iy in range(i_y0, i_y1):
            for ix in range(i_x0, i_x1):
                t_val = norm_temp[iy - i_y0, ix - i_x0]
                # Heatmap gradient: blue (cold) -> yellow -> red (hot)
                r = int(255 * t_val)
                g = int(255 * (1.0 - abs(t_val - 0.5) * 2.0))
                b = int(255 * (1.0 - t_val))
                sx, sy = camera.world_to_screen(ix * cell_size, iy * cell_size)
                pygame.draw.rect(overlay, (r, g, b, 50), (sx, sy, block_w, block_h))
        surface.blit(overlay, (0, 0))

    def draw_vision_cone(self, surface: pygame.Surface, eco, idx: int, camera) -> None:
        """Render the sensory vision cone and whiskers for an organism."""
        s = eco.org
        if idx < 0 or idx >= s.n:
            return

        wx, wy = float(s.pos[idx, 0]), float(s.pos[idx, 1])
        ang = float(s.ang[idx])
        vis_range = float(s.traits[idx, 2]) * camera.zoom
        vis_angle = math.radians(float(s.traits[idx, 3]))
        sx, sy = camera.world_to_screen(wx, wy)

        # Draw cone arc
        half_ang = vis_angle / 2.0
        n_pts = 16
        pts = [(sx, sy)]
        for i in range(n_pts + 1):
            a = ang - half_ang + (vis_angle * i / n_pts)
            px = sx + math.cos(a) * vis_range
            py = sy + math.sin(a) * vis_range
            pts.append((int(px), int(py)))

        cone_surf = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
        pygame.draw.polygon(cone_surf, (80, 200, 255, 30), pts)
        pygame.draw.lines(cone_surf, (80, 220, 255, 90), False, pts, 1)
        surface.blit(cone_surf, (0, 0))
