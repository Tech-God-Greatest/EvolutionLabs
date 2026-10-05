"""Procedurally generated 2-D world: terrain, temperature, fertility, hazards.

All fields live on a coarse grid (``terrain_cell`` units per cell, indexed
``[row=y, col=x]``).  Six world presets + a mazes preset are supported.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

GRASS, WATER, ROCK, TREE, SAND, SNOW, TOXIC = range(7)
TERRAIN_NAMES = ["grass", "water", "rock", "tree", "sand", "snow", "toxic soil"]
SPEED_MULT = np.array([1.0, 0.5, 0.0, 0.0, 0.85, 0.8, 0.9], dtype=np.float32)
TOXIC_DANGER = 2.5

WORLD_PARAMS: Dict[str, dict] = {
    "meadow": dict(water=0.20, rock=0.035, tree=0.05, temp=0.50, trange=0.22, fert=1.00, food_mult=1.0, food_value=1.0, toxic=0.02),
    "desert": dict(water=0.05, rock=0.07, tree=0.008, temp=0.82, trange=0.12, fert=0.50, food_mult=0.55, food_value=1.0, toxic=0.02),
    "forest": dict(water=0.14, rock=0.05, tree=0.26, temp=0.50, trange=0.15, fert=1.30, food_mult=1.40, food_value=1.1, toxic=0.01),
    "arctic": dict(water=0.10, rock=0.07, tree=0.02, temp=0.12, trange=0.10, fert=0.55, food_mult=0.50, food_value=1.0, toxic=0.01),
    "toxic": dict(water=0.06, rock=0.07, tree=0.02, temp=0.55, trange=0.20, fert=0.60, food_mult=0.28, food_value=3.2, toxic=0.34),
}


# ----------------------------------------------------------------------------
# Noise
# ----------------------------------------------------------------------------
def _smooth_resize(grid: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Bilinear (smoothstep) resize of a small random lattice to (gh, gw)."""
    h, w = grid.shape
    ys = np.linspace(0, h - 1, gh)
    xs = np.linspace(0, w - 1, gw)
    y0 = np.minimum(ys.astype(int), h - 2)
    x0 = np.minimum(xs.astype(int), w - 2)
    ty = ys - y0
    tx = xs - x0
    ty = ty * ty * (3 - 2 * ty)
    tx = tx * tx * (3 - 2 * tx)
    a = grid[np.ix_(y0, x0)]
    b = grid[np.ix_(y0, x0 + 1)]
    c = grid[np.ix_(y0 + 1, x0)]
    d = grid[np.ix_(y0 + 1, x0 + 1)]
    top = a + (b - a) * tx[None, :]
    bot = c + (d - c) * tx[None, :]
    return top + (bot - top) * ty[:, None]


def fbm(rng: np.random.Generator, gh: int, gw: int, octaves: int = 5, base: int = 3, persistence: float = 0.5) -> np.ndarray:
    out = np.zeros((gh, gw), dtype=np.float64)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        res = base * 2 ** o
        out += amp * _smooth_resize(rng.random((res + 1, res + 1)), gh, gw)
        tot += amp
        amp *= persistence
    out /= tot
    lo, hi = out.min(), out.max()
    return ((out - lo) / (hi - lo + 1e-12)).astype(np.float32)


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """Approximate Euclidean distance (in cells) to the nearest True cell."""
    if not mask.any():
        return np.full(mask.shape, 1e3, dtype=np.float32)
    d = np.where(mask, 0.0, 1e6).astype(np.float32)
    D = 1.41421356
    for _ in range(sum(mask.shape)):
        m = d.copy()
        np.minimum(m[1:, :], d[:-1, :] + 1, out=m[1:, :])
        np.minimum(m[:-1, :], d[1:, :] + 1, out=m[:-1, :])
        np.minimum(m[:, 1:], d[:, :-1] + 1, out=m[:, 1:])
        np.minimum(m[:, :-1], d[:, 1:] + 1, out=m[:, :-1])
        np.minimum(m[1:, 1:], d[:-1, :-1] + D, out=m[1:, 1:])
        np.minimum(m[:-1, :-1], d[1:, 1:] + D, out=m[:-1, :-1])
        np.minimum(m[1:, :-1], d[:-1, 1:] + D, out=m[1:, :-1])
        np.minimum(m[:-1, 1:], d[1:, :-1] + D, out=m[:-1, 1:])
        if np.array_equal(m, d):
            break
        d = m
    return d


class World:
    def __init__(self, cfg):
        self.cfg = cfg
        self.world_type = cfg.world_type
        self.cell = float(cfg.terrain_cell)
        if self.world_type == "maze":
            n = int(cfg.maze_size)
            self.gw = self.gh = 4 * n + 1
            self.width = self.height = self.gw * self.cell
        else:
            self.width = float(cfg.world_width)
            self.height = float(cfg.world_height)
            self.gw = int(np.ceil(self.width / self.cell))
            self.gh = int(np.ceil(self.height / self.cell))
        self.params: dict = {}
        self.start = None
        self.goal = None
        self.maze_dist: Optional[np.ndarray] = None
        shape = (self.gh, self.gw)
        self.terrain_base = np.zeros(shape, dtype=np.uint8)
        self.terrain = np.zeros(shape, dtype=np.uint8)
        self.blocked = np.zeros(shape, dtype=bool)
        self.temp = np.full(shape, 0.5, dtype=np.float32)
        self.fertility = np.zeros(shape, dtype=np.float32)
        self.danger = np.zeros(shape, dtype=np.float32)
        self.danger_extra = np.zeros(shape, dtype=np.float32)
        self.safe = np.zeros(shape, dtype=bool)
        self.water_base = np.zeros(shape, dtype=bool)
        self.depth_in = np.zeros(shape, dtype=np.float32)
        self.dist_out = np.zeros(shape, dtype=np.float32)
        self.water_dist = np.zeros(shape, dtype=np.float32)
        self.flooded = np.zeros(shape, dtype=bool)
        self.water_offset = 0
        self.fert_mult = 1.0
        self.spawn_cdf = np.zeros(self.gw * self.gh)
        self.mean_fertility = 0.5
        self.version = 0  # bumped when static visuals need re-rendering

    # ------------------------------------------------------------------
    def params_for(self) -> dict:
        c = self.cfg
        if self.world_type == "custom":
            return dict(water=c.custom_water, rock=c.custom_rock, tree=c.custom_tree, temp=c.custom_temperature,
                        trange=c.custom_temp_range, fert=c.custom_fertility, food_mult=1.0, food_value=1.0,
                        toxic=c.custom_toxic)
        return dict(WORLD_PARAMS[self.world_type])

    def generate(self, rng: np.random.Generator) -> None:
        if self.world_type == "maze":
            self._generate_maze(rng)
            return
        p = self.params = self.params_for()
        gh, gw = self.gh, self.gw
        elev = fbm(rng, gh, gw, 5, 3)
        moist = fbm(rng, gh, gw, 5, 4)
        tn = fbm(rng, gh, gw, 4, 2)
        rn = fbm(rng, gh, gw, 4, 7, 0.6)
        trn = fbm(rng, gh, gw, 4, 9, 0.6)
        tox = fbm(rng, gh, gw, 3, 4, 0.55)
        yfrac = np.linspace(0, 1, gh)[:, None]
        # water
        if p["water"] > 0:
            water = elev < np.quantile(elev, p["water"])
        else:
            water = np.zeros((gh, gw), dtype=bool)
        self.water_base = water
        dist_w = distance_transform(water)
        # temperature (north colder, noise, water cools)
        temp = p["temp"] + p["trange"] * (tn - 0.5) * 2.0 + (yfrac - 0.5) * 0.12
        temp = temp - 0.05 * np.exp(-dist_w / 3.0)
        self.temp = np.clip(temp, 0.0, 1.0).astype(np.float32)
        # fertility
        fert = (0.30 + 0.70 * moist) * (0.50 + 0.50 * np.exp(-dist_w / 6.0)) * p["fert"]
        # terrain types
        t = np.zeros((gh, gw), dtype=np.uint8)
        t[(self.temp > 0.72) & (moist < 0.55)] = SAND
        t[(dist_w <= 1.5) & ~water] = SAND
        t[self.temp < 0.22] = SNOW
        toxic_mask = np.zeros((gh, gw), dtype=bool)
        if p["toxic"] > 0:
            toxic_mask = tox > np.quantile(tox, 1.0 - p["toxic"])
            t[toxic_mask & ~water] = TOXIC
        rock = np.zeros((gh, gw), dtype=bool)
        if p["rock"] > 0:
            rock = (rn > np.quantile(rn, 1.0 - p["rock"])) & ~water
        tree = np.zeros((gh, gw), dtype=bool)
        if p["tree"] > 0:
            tree = (trn > np.quantile(trn, 1.0 - p["tree"])) & ~water & ~rock & (fert > 0.25)
        t[water] = WATER
        t[rock] = ROCK
        t[tree] = TREE
        fert = np.where(t == SAND, fert * 0.35, fert)
        fert = np.where(t == SNOW, fert * 0.8, fert)
        fert = np.where(t == TOXIC, fert * 1.2, fert)
        fert = np.where((t == WATER) | (t == ROCK), 0.0, fert)
        fert = np.where(t == TREE, fert * 0.5, fert)
        self.fertility = np.clip(fert, 0.0, 1.5).astype(np.float32)
        self.terrain_base = t
        self.blocked = (t == ROCK) | (t == TREE)
        self.danger = np.where(t == TOXIC, TOXIC_DANGER, 0.0).astype(np.float32)
        self.danger_extra = np.zeros_like(self.danger)
        # safe zones (sanctuaries)
        safe = np.zeros((gh, gw), dtype=bool)
        yy, xx = np.mgrid[0:gh, 0:gw]
        for _ in range(int(rng.integers(3, 6))):
            for _try in range(30):
                cy, cx = int(rng.integers(gh)), int(rng.integers(gw))
                if not self.blocked[cy, cx] and t[cy, cx] not in (WATER, TOXIC):
                    r = rng.uniform(3.5, 7.0)
                    safe |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
                    break
        self.safe = safe & ~self.blocked
        self._after_terrain_change()

    # ------------------------------------------------------------------
    def _generate_maze(self, rng: np.random.Generator) -> None:
        n = int(self.cfg.maze_size)
        H = W = 4 * n + 1
        grid = np.ones((H, W), dtype=bool)  # True = wall
        visited = np.zeros((n, n), dtype=bool)
        stack = [(0, 0)]
        visited[0, 0] = True

        def carve_cell(i, j):
            y0, x0 = 1 + 4 * i, 1 + 4 * j
            grid[y0:y0 + 3, x0:x0 + 3] = False

        carve_cell(0, 0)
        while stack:
            i, j = stack[-1]
            nb = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                  if 0 <= i + di < n and 0 <= j + dj < n and not visited[i + di, j + dj]]
            if not nb:
                stack.pop()
                continue
            ni, nj = nb[int(rng.integers(len(nb)))]
            carve_cell(ni, nj)
            # open the wall between (i,j) and (ni,nj)
            if ni != i:
                y = 1 + 4 * min(i, ni) + 3
                grid[y, 1 + 4 * j:1 + 4 * j + 3] = False
            else:
                x = 1 + 4 * min(j, nj) + 3
                grid[1 + 4 * i:1 + 4 * i + 3, x] = False
            visited[ni, nj] = True
            stack.append((ni, nj))
        t = np.where(grid, ROCK, GRASS).astype(np.uint8)
        self.terrain_base = t
        self.blocked = grid
        self.water_base = np.zeros((H, W), dtype=bool)
        self.temp = np.full((H, W), 0.5, dtype=np.float32)
        self.fertility = np.zeros((H, W), dtype=np.float32)
        self.danger = np.zeros((H, W), dtype=np.float32)
        self.danger_extra = np.zeros_like(self.danger)
        self.safe = np.zeros((H, W), dtype=bool)
        # start/goal: BFS from start cell, goal = farthest open cell
        sy, sx = 2, 2
        dist = self._bfs(sy, sx)
        far = np.unravel_index(np.argmax(np.where(np.isfinite(dist), dist, -1)), dist.shape)
        gy, gx = int(far[0]), int(far[1])
        self.start = ((sx + 0.5) * self.cell, (sy + 0.5) * self.cell)
        self.goal = ((gx + 0.5) * self.cell, (gy + 0.5) * self.cell)
        self.maze_dist = self._bfs(gy, gx)
        self.params = dict(food_mult=0.0, food_value=1.0)
        self._after_terrain_change()

    def _bfs(self, sy: int, sx: int) -> np.ndarray:
        H, W = self.blocked.shape
        dist = np.full((H, W), np.inf, dtype=np.float32)
        dist[sy, sx] = 0
        frontier = [(sy, sx)]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for y, x in frontier:
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not self.blocked[ny, nx] and dist[ny, nx] == np.inf:
                        dist[ny, nx] = d
                        nxt.append((ny, nx))
            frontier = nxt
        return dist

    # ------------------------------------------------------------------
    def _after_terrain_change(self) -> None:
        self.terrain = self.terrain_base.copy()
        self.depth_in = distance_transform(~self.water_base) * self.water_base
        self.dist_out = distance_transform(self.water_base)
        self.flooded = np.zeros_like(self.water_base)
        self.water_offset = 0
        self.refresh_water_dist()
        self.refresh_spawn_cdf()
        self.version += 1

    def refresh_water_dist(self) -> None:
        self.water_dist = distance_transform(self.terrain == WATER)

    def set_water_offset(self, offset: int) -> bool:
        """Positive = flooding (water expands), negative = drought (water shrinks)."""
        offset = int(offset)
        if offset == self.water_offset or self.world_type == "maze":
            return False
        self.water_offset = offset
        base = self.water_base
        cur = base & (self.depth_in > max(0, -offset))
        if offset > 0:
            cur = cur | ((~base) & (self.dist_out <= offset) & ~self.blocked)
        t = self.terrain_base.copy()
        dried = base & ~cur
        t[dried] = SAND
        t[cur] = WATER
        self.flooded = cur & ~base
        self.terrain = t
        self.refresh_water_dist()
        self.refresh_spawn_cdf()
        self.version += 1
        return True

    def refresh_spawn_cdf(self) -> None:
        w = (self.fertility * self.fert_mult).astype(np.float64)
        w = np.where(self.terrain == WATER, 0.0, w)
        w = np.where(self.blocked, 0.0, w)
        flat = w.ravel()
        tot = flat.sum()
        mask = (~self.blocked) & (self.terrain != WATER)
        self.mean_fertility = float(self.fertility[mask].mean()) if mask.any() else 0.0
        if tot <= 0:
            flat = np.ones_like(flat)
            tot = flat.sum()
        self.spawn_cdf = np.cumsum(flat) / tot

    def set_fertility_mult(self, m: float) -> None:
        m = float(max(0.0, m))
        if abs(m - self.fert_mult) > 0.02:
            self.fert_mult = m
            self.refresh_spawn_cdf()

    # ------------------------------------------------------------------
    # lookups
    # ------------------------------------------------------------------
    def cell_index(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ix = np.clip((pos[..., 0] / self.cell).astype(np.int32), 0, self.gw - 1)
        iy = np.clip((pos[..., 1] / self.cell).astype(np.int32), 0, self.gh - 1)
        return ix, iy

    def blocked_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """True where the point is inside an obstacle or outside the world."""
        ix = (x / self.cell).astype(np.int32)
        iy = (y / self.cell).astype(np.int32)
        out = (x < 0) | (y < 0) | (ix >= self.gw) | (iy >= self.gh)
        ixc = np.clip(ix, 0, self.gw - 1)
        iyc = np.clip(iy, 0, self.gh - 1)
        return out | self.blocked[iyc, ixc]

    def sample_cells(self, count: int, rng: np.random.Generator, fertile: bool = True) -> np.ndarray:
        """Random cell centres (jittered) – fertility-weighted when ``fertile``."""
        if fertile:
            idx = np.searchsorted(self.spawn_cdf, rng.random(count))
            idx = np.minimum(idx, self.gw * self.gh - 1)
        else:
            idx = rng.integers(0, self.gw * self.gh, size=count)
        iy, ix = np.divmod(idx, self.gw)
        jit = rng.random((count, 2)).astype(np.float32)
        x = (ix + jit[:, 0]) * self.cell
        y = (iy + jit[:, 1]) * self.cell
        return np.stack([x, y], axis=1).astype(np.float32)

    def random_free_positions(self, count: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform positions on walkable (non-blocked) terrain."""
        out = np.zeros((0, 2), dtype=np.float32)
        tries = 0
        while len(out) < count and tries < 50:
            cand = self.sample_cells(max(count * 2, 16), rng, fertile=False)
            ok = ~self.blocked_at(cand[:, 0], cand[:, 1])
            out = np.concatenate([out, cand[ok]], axis=0)
            tries += 1
        if len(out) < count:
            out = np.concatenate([out, np.tile(np.array([[self.width / 2, self.height / 2]], np.float32), (count - len(out), 1))])
        return out[:count]

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "terrain_base", "terrain", "blocked", "temp", "fertility", "danger", "danger_extra", "safe",
            "water_base", "depth_in", "dist_out", "flooded", "water_offset", "fert_mult")} | {
            "start": self.start, "goal": self.goal, "maze_dist": self.maze_dist, "params": self.params,
            "world_type": self.world_type, "width": self.width, "height": self.height}

    def load_state_dict(self, d: dict) -> None:
        for k in ("terrain_base", "terrain", "blocked", "temp", "fertility", "danger", "danger_extra", "safe",
                  "water_base", "depth_in", "dist_out", "flooded"):
            setattr(self, k, np.array(d[k]))
        self.water_offset = int(d["water_offset"])
        self.fert_mult = float(d["fert_mult"])
        self.start, self.goal = d["start"], d["goal"]
        self.maze_dist = None if d["maze_dist"] is None else np.array(d["maze_dist"])
        self.params = d["params"]
        self.refresh_water_dist()
        self.refresh_spawn_cdf()
        self.version += 1
