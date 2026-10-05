"""Visual particle effects system for combat, births, deaths, and disasters."""
from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np
import pygame


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "size", "color")

    def __init__(self, x: float, y: float, vx: float, vy: float, life: float, size: float, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.size = size
        self.color = color


class ParticleSystem:
    def __init__(self, max_particles: int = 1500):
        self.particles: List[Particle] = []
        self.max_particles = max_particles

    def add(self, x: float, y: float, vx: float, vy: float, life: float, size: float, color: Tuple[int, int, int]) -> None:
        if len(self.particles) < self.max_particles:
            self.particles.append(Particle(x, y, vx, vy, life, size, color))

    def emit_burst(self, x: float, y: float, count: int, speed: float, color: Tuple[int, int, int], size: float = 2.5) -> None:
        for _ in range(count):
            ang = float(np.random.uniform(0, 2 * math.pi))
            spd = float(np.random.uniform(0.3, 1.0) * speed)
            vx = math.cos(ang) * spd
            vy = math.sin(ang) * spd
            life = float(np.random.uniform(0.3, 0.7))
            self.add(x, y, vx, vy, life, size, color)

    def process_ecosystem_fx(self, fx_list: List[tuple]) -> None:
        """Translate simulation engine fx tuples to visual particles."""
        for item in fx_list:
            kind = item[0]
            if kind == "hit":
                _, x, y = item
                self.emit_burst(x, y, count=6, speed=45.0, color=(255, 80, 80), size=2.0)
            elif kind == "birth":
                _, x, y = item
                self.emit_burst(x, y, count=8, speed=25.0, color=(100, 240, 255), size=2.5)
            elif kind == "death":
                _, x, y, _ = item
                self.emit_burst(x, y, count=10, speed=35.0, color=(200, 200, 220), size=2.0)
            elif kind == "eat":
                _, x, y = item
                self.emit_burst(x, y, count=3, speed=15.0, color=(120, 255, 140), size=1.8)
            elif kind == "disaster":
                _, dkind, x, y, radius = item
                if dkind == "meteor":
                    self.emit_burst(x, y, count=50, speed=160.0, color=(255, 120, 20), size=5.0)

    def update(self, dt: float) -> None:
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life > 0:
                p.x += p.vx * dt
                p.y += p.vy * dt
                p.vx *= (1.0 - 1.5 * dt)
                p.vy *= (1.0 - 1.5 * dt)
                alive.append(p)
        self.particles = alive

    def draw(self, surface: pygame.Surface, camera) -> None:
        for p in self.particles:
            if camera.is_visible(p.x, p.y, p.size):
                sx, sy = camera.world_to_screen(p.x, p.y)
                alpha = max(0, min(255, int(255 * (p.life / p.max_life))))
                rad = max(1, int(p.size * camera.zoom))
                c = (p.color[0], p.color[1], p.color[2])
                pygame.draw.circle(surface, c, (sx, sy), rad)
