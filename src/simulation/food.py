"""Dynamic food: plants that grow, seed their surroundings, and carrion.

The crop follows logistic regrowth limited by a *carrying capacity* derived from
terrain fertility, the world preset and the current environment – there is no
infinite food.  Heavy grazing shrinks the crop (and therefore the herbivore
population), which is the feedback loop the predator-prey dynamics rely on.
"""
from __future__ import annotations

import numpy as np

from .world import TREE, TOXIC, WATER

KIND_BERRY, KIND_FRUIT, KIND_TOXIC, KIND_CARRION = range(4)
KIND_NAMES = ["berry", "fruit", "toxic bloom", "carrion"]
KIND_MULT = {KIND_BERRY: 1.0, KIND_FRUIT: 1.8, KIND_TOXIC: 1.4}
CARRION_DECAY = 0.015  # fraction per second
INITIAL_ENERGY = 2.0
SEED_PROB = 0.75       # share of new plants that sprout beside an existing mature plant


class FoodStore:
    def __init__(self, capacity: int = 8192):
        self.n = 0
        self._alloc(capacity)
        self._tree_adj = None
        self._tree_adj_version = -1

    def _alloc(self, cap: int) -> None:
        self.cap = cap
        self.pos = np.zeros((cap, 2), dtype=np.float32)
        self.energy = np.zeros(cap, dtype=np.float32)
        self.max_energy = np.zeros(cap, dtype=np.float32)
        self.growth = np.zeros(cap, dtype=np.float32)
        self.kind = np.zeros(cap, dtype=np.uint8)

    _FIELDS = ("pos", "energy", "max_energy", "growth", "kind")

    def _ensure(self, extra: int) -> None:
        if self.n + extra <= self.cap:
            return
        new_cap = max(self.cap * 2, self.n + extra + 64)
        for f in self._FIELDS:
            old = getattr(self, f)
            new = np.zeros((new_cap,) + old.shape[1:], dtype=old.dtype)
            new[: self.n] = old[: self.n]
            setattr(self, f, new)
        self.cap = new_cap

    def add(self, pos, energy, max_energy, growth, kind) -> None:
        k = len(pos)
        if k == 0:
            return
        self._ensure(k)
        s = slice(self.n, self.n + k)
        self.pos[s] = pos
        self.energy[s] = energy
        self.max_energy[s] = max_energy
        self.growth[s] = growth
        self.kind[s] = kind
        self.n += k

    def remove(self, mask: np.ndarray) -> float:
        """Remove masked items; returns total energy removed."""
        n = self.n
        mask = mask[:n]
        if not mask.any():
            return 0.0
        lost = float(self.energy[:n][mask].sum())
        keep = ~mask
        m = int(keep.sum())
        for f in self._FIELDS:
            a = getattr(self, f)
            a[:m] = a[:n][keep]
        self.n = m
        return lost

    def count_plants(self) -> int:
        return int(np.count_nonzero(self.kind[: self.n] != KIND_CARRION))

    def total_energy(self) -> float:
        return float(self.energy[: self.n].sum(dtype=np.float64))

    # ------------------------------------------------------------------
    def grow(self, dt: float, ledger) -> None:
        n = self.n
        if n == 0:
            return
        e = self.energy[:n]
        plant = self.kind[:n] != KIND_CARRION
        inc = np.where(plant, np.minimum(self.growth[:n] * dt, np.maximum(self.max_energy[:n] - e, 0.0)), 0.0)
        ledger.add("in_food_growth", float(inc.sum(dtype=np.float64)))
        e += inc
        carr = ~plant
        if carr.any():
            dec = e * CARRION_DECAY * dt * carr
            ledger.add("out_decay", float(dec.sum(dtype=np.float64)))
            e -= dec
            dead = carr & (e < 0.5)
            if dead.any():
                ledger.add("out_decay", float(e[dead].sum(dtype=np.float64)))
                self.remove(dead)

    def capacity_target(self, eco) -> float:
        cfg, w = eco.cfg, eco.world
        area = w.width * w.height
        return (cfg.food_density * 2.5 * (area / 1000.0) * w.mean_fertility * eco.env.food_mult
                * w.params.get("food_mult", 1.0))

    def _tree_adjacent(self, world) -> np.ndarray:
        if self._tree_adj_version != world.version:
            t = world.terrain_base == TREE
            adj = t.copy()
            adj[1:, :] |= t[:-1, :]
            adj[:-1, :] |= t[1:, :]
            adj[:, 1:] |= t[:, :-1]
            adj[:, :-1] |= t[:, 1:]
            self._tree_adj = adj
            self._tree_adj_version = world.version
        return self._tree_adj

    def spawn(self, eco, count: int, mature_fraction: float = 0.0, seeded: bool = True) -> int:
        """Try to place ``count`` new plants.  Returns how many were actually created."""
        if count <= 0:
            return 0
        world, rng, cfg = eco.world, eco.rng.spawn, eco.cfg
        k = int(count)
        plants = np.flatnonzero((self.kind[: self.n] != KIND_CARRION) & (self.energy[: self.n] >= 0.5 * self.max_energy[: self.n]))
        use_seed = (rng.random(k) < SEED_PROB) & (plants.size > 0) & seeded
        par = plants[rng.integers(0, max(plants.size, 1), size=k)] if plants.size else np.zeros(k, dtype=np.int64)
        p_seed = (self.pos[par] if plants.size else np.zeros((k, 2), np.float32)) + rng.normal(0, 25.0, size=(k, 2)).astype(np.float32)
        p_rand = world.sample_cells(k, rng, fertile=True)
        pos = np.where(use_seed[:, None], p_seed, p_rand).astype(np.float32)
        np.clip(pos[:, 0], 1, world.width - 1, out=pos[:, 0])
        np.clip(pos[:, 1], 1, world.height - 1, out=pos[:, 1])
        ix, iy = world.cell_index(pos)
        fert = world.fertility[iy, ix] * world.fert_mult
        ok = (~world.blocked[iy, ix]) & (world.terrain[iy, ix] != WATER)
        ok &= (~use_seed) | (rng.random(k) < np.minimum(1.0, fert / 0.5))
        if not ok.any():
            return 0
        pos, ix, iy = pos[ok], ix[ok], iy[ok]
        m = len(pos)
        tree_adj = self._tree_adjacent(world)[iy, ix]
        toxic = world.terrain_base[iy, ix] == TOXIC
        kind = np.full(m, KIND_BERRY, dtype=np.uint8)
        kind[tree_adj & (rng.random(m) < 0.5)] = KIND_FRUIT
        kind[toxic] = KIND_TOXIC
        base = cfg.food_max_energy * world.params.get("food_value", 1.0)
        mult = np.where(kind == KIND_FRUIT, KIND_MULT[KIND_FRUIT], np.where(kind == KIND_TOXIC, KIND_MULT[KIND_TOXIC], 1.0))
        max_e = (base * mult * (0.7 + 0.6 * rng.random(m))).astype(np.float32)
        grow_t = cfg.food_grow_time * np.where(kind == KIND_FRUIT, 1.5, 1.0)
        growth = (max_e / grow_t).astype(np.float32)
        energy = np.full(m, INITIAL_ENERGY, dtype=np.float32)
        if mature_fraction > 0:
            energy = (max_e * (mature_fraction * (0.4 + 0.6 * rng.random(m)))).astype(np.float32)
        eco.ledger.add("in_food_spawn", float(energy.sum(dtype=np.float64)))
        self.add(pos, energy, max_e, growth, kind)
        return m

    def spawn_step(self, eco, dt: float) -> None:
        K = self.capacity_target(eco)
        if K <= 0:
            return
        F = self.count_plants()
        rate = eco.cfg.food_regrowth * F * max(0.0, 1.0 - F / K) + (0.002 * K if F < K else 0.0)
        k = int(eco.rng.spawn.poisson(rate * dt))
        k = min(k, max(0, int(K * 1.05) - F))
        if k:
            self.spawn(eco, k)

    def seed_initial(self, eco, fraction: float = 0.7) -> None:
        K = self.capacity_target(eco)
        want = int(K * fraction)
        got = 0
        for _ in range(6):
            got += self.spawn(eco, max(int((want - got) * 1.3), 0) if got else int(want * 1.4), mature_fraction=0.8, seeded=False)
            if got >= want:
                break

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        n = self.n
        return {f: getattr(self, f)[:n].copy() for f in self._FIELDS}

    def load_state_dict(self, d: dict) -> None:
        n = len(d["energy"])
        self.n = 0
        self._alloc(max(8192, n * 2))
        for f in self._FIELDS:
            getattr(self, f)[:n] = d[f]
        self.n = n
