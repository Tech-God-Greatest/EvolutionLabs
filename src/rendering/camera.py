"""2D Camera with smooth zoom, pan, following, and screen-space conversions."""
from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np


class Camera:
    def __init__(self, screen_w: int, screen_h: int, world_w: float, world_h: float):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.world_w = world_w
        self.world_h = world_h
        # Position centered on the world
        self.x = world_w / 2.0
        self.y = world_h / 2.0
        # Zoom scale: pixels per world unit
        self.zoom = min(screen_w / world_w, screen_h / world_h) * 0.95
        self.target_zoom = self.zoom
        self.min_zoom = 0.08
        self.max_zoom = 8.0
        # Following target
        self.following_id: Optional[int] = None
        self.following_species_id: Optional[int] = None
        # Dragging state
        self.dragging = False
        self.last_mouse = (0, 0)

    def resize(self, w: int, h: int) -> None:
        self.screen_w = w
        self.screen_h = h

    def world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        sx = int((wx - self.x) * self.zoom + self.screen_w / 2.0)
        sy = int((wy - self.y) * self.zoom + self.screen_h / 2.0)
        return sx, sy

    def screen_to_world(self, sx: int, sy: int) -> Tuple[float, float]:
        wx = (sx - self.screen_w / 2.0) / self.zoom + self.x
        wy = (sy - self.screen_h / 2.0) / self.zoom + self.y
        return wx, wy

    def is_visible(self, wx: float, wy: float, radius: float = 10.0) -> bool:
        margin = radius * self.zoom
        sx, sy = self.world_to_screen(wx, wy)
        return -margin <= sx <= self.screen_w + margin and -margin <= sy <= self.screen_h + margin

    def zoom_at(self, sx: int, sy: int, factor: float) -> None:
        """Zoom centered at cursor position."""
        old_zoom = self.zoom
        new_zoom = max(self.min_zoom, min(self.max_zoom, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-5:
            return
        # Cursor world coordinates before zoom
        c_wx, c_wy = self.screen_to_world(sx, sy)
        self.zoom = new_zoom
        # Adjust camera center so cursor points to same world position
        self.x = c_wx - (sx - self.screen_w / 2.0) / new_zoom
        self.y = c_wy - (sy - self.screen_h / 2.0) / new_zoom

    def update(self, eco, dt: float) -> None:
        # Smooth follow organism
        if self.following_id is not None:
            idx = eco.org.index_of(self.following_id)
            if idx >= 0:
                tx = float(eco.org.pos[idx, 0])
                ty = float(eco.org.pos[idx, 1])
                self.x += (tx - self.x) * min(1.0, 10.0 * dt)
                self.y += (ty - self.y) * min(1.0, 10.0 * dt)
            else:
                self.following_id = None

        # Smooth follow species centroid
        elif self.following_species_id is not None:
            mask = eco.org.species[:eco.org.n] == self.following_species_id
            if mask.any():
                tx = float(eco.org.pos[:eco.org.n][mask, 0].mean())
                ty = float(eco.org.pos[:eco.org.n][mask, 1].mean())
                self.x += (tx - self.x) * min(1.0, 8.0 * dt)
                self.y += (ty - self.y) * min(1.0, 8.0 * dt)
            else:
                self.following_species_id = None

    def center_on(self, wx: float, wy: float) -> None:
        self.x = wx
        self.y = wy
        self.following_id = None
        self.following_species_id = None
