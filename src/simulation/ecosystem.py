"""The ecosystem: ties world, food, organisms, evolution and analysis together.

Nothing in this file scripts behaviour.  Each tick every organism *senses* its
surroundings (vision cone, whiskers, temperature, internal state, memory), a
neural network turns those sensations into continuous motor/action outputs, and
physics, metabolism, combat and reproduction decide the consequences.  Which
behaviours are useful is discovered by selection.

The module is independent of rendering so identical code runs headlessly.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional

import numpy as np

from ..core.config import Config
from ..core.errors import ConfigError, GenomeError
from ..core.rng import RngHub
from ..core.util import wrap_angle
from ..evolution.crossover import crossover
from ..evolution.evolutionary_algorithms import make_strategy
from ..evolution.genome import (G_ACCEL, G_ATTACK, G_DIET, G_EFF, G_METAB, G_MUT, G_REPRO, G_SIZE, G_SPEED,
                                G_TPREF, G_TTOL, G_VANG, G_VIS, IDX, N_TRAITS, TRAIT_DEFAULT, TRAIT_LO, TRAIT_HI,
                                TRAIT_NAMES, TRAIT_RANGE, Genome, carnivore_template, clamp_traits,
                                herbivore_template)
from ..evolution.mutation import burst_mutation, mutate
from ..evolution.species import SpeciesManager, pair_distance
from ..neural.backend import get_backend
from ..neural.network import (NetSpec, O_ATTACK, O_EAT, O_GAIT, O_REPRO, O_SPRINT, O_THRUST, O_TURN, forward_one,
                              init_params)
from .environment import Environment
from .food import KIND_CARRION, FoodStore
from .organism import (C_AGE, C_DISASTER, C_KILLED, C_STARVE, C_TEMP, C_TOXIC, CAUSES, Lineage, OrganismStore)
from .spatial_grid import SpatialGrid
from .world import SPEED_MULT, World

# -- sensors ---------------------------------------------------------------------------
SENSOR_GROUP_NAMES = {
    "basic": ["food_dir", "food_prox", "food_energy", "food_kind", "energy", "health", "age", "speed",
              "reward", "time_since_eat", "time_since_repro"],
    "social": ["other_dir", "other_prox", "other_size", "threat_dir", "threat_prox", "target_dir", "target_prox",
               "neigh_density", "align_cos", "align_sin"],
    "env": ["temperature", "noise", "wall_prox", "water_prox", "obs_left", "obs_front", "obs_right"],
    "memory": ["prev_turn", "prev_thrust", "recent_danger", "recent_attack", "recent_food"],
}
GROUP_ORDER = ["basic", "social", "env", "memory"]
ALL_SENSORS = [n for g in GROUP_ORDER for n in SENSOR_GROUP_NAMES[g]]

WHISKER_LEN = 60.0
REACH_EXTRA = 3.0
ATTACK_ARC = 0.9
STAGE_BITE_COST_FACTOR = 1.0


def sensor_names_for(groups) -> List[str]:
    return [n for g in GROUP_ORDER if g in groups for n in SENSOR_GROUP_NAMES[g]]


class Ledger:
    """Energy bookkeeping (used to verify energy conservation)."""
    KEYS = ["in_initial", "in_food_growth", "in_food_spawn", "in_gift",
            "out_metabolism", "out_attack", "out_assimilation", "out_overflow", "out_birth", "out_decay",
            "out_predation", "out_corpse", "out_removed", "out_temperature"]

    def __init__(self):
        self.v: Dict[str, float] = {k: 0.0 for k in self.KEYS}

    def add(self, key: str, x: float) -> None:
        self.v[key] += float(x)

    def inflow(self) -> float:
        return sum(v for k, v in self.v.items() if k.startswith("in_"))

    def outflow(self) -> float:
        return sum(v for k, v in self.v.items() if k.startswith("out_"))


class Ecosystem:
    def __init__(self, config=None, seed: Optional[int] = None, record_fx: bool = False, init: bool = True):
        if config is None:
            cfg = Config()
        elif isinstance(config, Config):
            cfg = config.copy()
        else:
            cfg = Config(config)
        if seed is not None:
            cfg.seed = seed
        if cfg.seed is None:
            cfg.seed = RngHub.random_seed()
        self.cfg = cfg
        self.record_fx = record_fx
        self.rng = RngHub(cfg.seed)
        self.tick = 0
        self.time = 0.0
        self.next_id = 0
        self.mean_generation = 0.0
        self.max_generation = 0
        self.fx: List[tuple] = []
        self.nan_events = 0
        self.maze_mode = cfg.world_type == "maze"
        self.action_log: List[list] = []
        self.ledger = Ledger()
        # --- network / sensors ---
        self.sensor_names = sensor_names_for(cfg.sensor_groups)
        self.sensor_index = [ALL_SENSORS.index(n) for n in self.sensor_names]
        self.spec = NetSpec(len(self.sensor_names), cfg.hidden_layers, recurrent=cfg.recurrent)
        self.backend = get_backend(cfg.backend)
        self.strategy = make_strategy(cfg.strategy)
        # --- world ---
        self.world = World(cfg)
        self.world.generate(self.rng.world)
        self.env = Environment(self)
        self.food = FoodStore()
        cap = max(cfg.population_cap, cfg.population) + 64
        self.org = OrganismStore(self.spec, cap)
        self.lineage = Lineage()
        self.species = SpeciesManager(cfg.seed)
        self.ogrid = SpatialGrid(self.world.width, self.world.height, 60.0)
        self.fgrid = SpatialGrid(self.world.width, self.world.height, 60.0)
        # --- counters / transient ---
        self.births_total = 0
        self.deaths_total = 0
        self.death_causes = np.zeros(6, dtype=np.int64)
        self.recent_lifespans: List[float] = []
        self.recent_births = 0
        self.recent_deaths = 0
        self._cap = np.zeros(0, dtype=np.float32)
        self._pending = None
        self._cache: dict = {}
        # analysis (imported lazily to keep the simulation core import-light)
        from ..analysis.events import EventDetector
        from ..analysis.statistics import StatsRecorder
        self.events = EventDetector(self)
        self.stats = StatsRecorder(self)
        self.behavior_labels = np.zeros(0, dtype=np.int16)
        self.behavior_conf = np.zeros(0, dtype=np.float32)
        if init:
            if self.maze_mode:
                self._init_maze()
            else:
                self._populate()
                self.food.seed_initial(self, 0.7)
            self.species.update(self)
            self.ledger.v = {k: 0.0 for k in self.ledger.KEYS}
            self.initial_energy = self.total_energy()
            self.ledger.v["in_initial"] = self.initial_energy
            self.stats.record(force=True)

    # ===================================================================================
    # initialisation
    # ===================================================================================
    def new_ids(self, k: int) -> np.ndarray:
        ids = np.arange(self.next_id, self.next_id + k, dtype=np.int64)
        self.next_id += k
        return ids

    def energy_capacity(self, size: np.ndarray) -> np.ndarray:
        return (100.0 * (size / 5.0) ** 1.5).astype(np.float32)

    def make_founder_genomes(self, count: int, carnivores: int, rng=None):
        """Founder lineage genomes (traits, weights, masks)."""
        rng = rng or self.rng.spawn
        cfg = self.cfg
        spec = self.spec
        traits = np.zeros((count, N_TRAITS), dtype=np.float32)
        for i in range(count):
            base = carnivore_template(cfg) if i < carnivores else herbivore_template(cfg)
            traits[i] = base + rng.normal(0, 0.10, N_TRAITS).astype(np.float32) * TRAIT_RANGE
            traits[i, G_DIET] = base[G_DIET] + rng.normal(0, 0.05)
            traits[i, G_MUT] = 1.0
        clamp_traits(traits)
        w = init_params(spec, rng, count)
        # carnivore founders get an attack-bias prior (seed genetics, not a rule)
        from ..neural.network import OUTPUT_BIAS_PRIOR
        wo = spec.offsets[-1]
        for i in range(carnivores):
            w[i, wo[1] + O_ATTACK] = 0.9
        m = np.ones((count, spec.n_params), dtype=np.uint8)
        return traits, w, m

    def _populate(self) -> None:
        cfg, s, rng = self.cfg, self.org, self.rng.spawn
        N = cfg.population
        L = max(1, min(cfg.founder_lineages, N))
        n_pred = int(round(N * cfg.predator_ratio))
        lin_pred = 0 if n_pred == 0 else max(1, int(round(L * 0.3)))
        lin_pred = min(lin_pred, L - 1) if L > 1 else (1 if n_pred else 0)
        if L == 1:
            lin_pred = 0
        fT, fW, fM = self.make_founder_genomes(L, lin_pred)
        herb_lin = np.arange(lin_pred, L) if lin_pred < L else np.arange(L)
        assign = np.zeros(N, dtype=np.int64)
        assign[: n_pred] = rng.integers(0, max(lin_pred, 1), size=n_pred) if lin_pred else 0
        assign[n_pred:] = herb_lin[rng.integers(0, len(herb_lin), size=N - n_pred)]
        traits, w, m = fT[assign].copy(), fW[assign].copy(), fM[assign].copy()
        # small jitter so lineages are clouds rather than clones
        rate = np.full(N, 0.25, dtype=np.float32)
        mutate(traits, w, m, rate, 0.04, rng, self.spec, 0.0, False)
        traits[:, G_DIET] = np.where(np.arange(N) < n_pred, np.clip(traits[:, G_DIET], 0.6, 1), np.clip(traits[:, G_DIET], 0, 0.35))
        pos = self.world.random_free_positions(N, rng)
        ids = self.new_ids(N)
        cap = self.energy_capacity(traits[:, G_SIZE])
        energy = cap * rng.uniform(0.5, 0.8, N).astype(np.float32)
        sl = s.add(N, id=ids, parent_a=-1, parent_b=-1, generation=0, species=-1,
                   age=rng.uniform(0, cfg.min_reproduction_age * 1.5, N), pos=pos, start_pos=pos,
                   ang=rng.uniform(-np.pi, np.pi, N), energy=energy, health=100.0, traits=traits, w=w, mask=m,
                   w_eff=w * m, cooldown=rng.uniform(0, cfg.reproduction_cooldown, N), finish_time=-1.0,
                   last_attacker=-1)
        self.lineage.add_founders(ids, np.zeros(N, dtype=np.int32), 0.0, np.full(N, -1))
        self.next_stats_time = 0.0

    # ----- maze --------------------------------------------------------------------
    def _init_maze(self) -> None:
        w = self.world
        # a single permanent "reward" item at the goal
        self.food.add(np.array([w.goal], dtype=np.float32), np.array([50.0], np.float32),
                      np.array([50.0], np.float32), np.array([0.0], np.float32), np.array([0], np.uint8))
        N = self.cfg.population
        T, W, M = self.make_founder_genomes(N, 0)
        T[:, G_DIET] = 0.0
        T[:, G_ATTACK] = 0.0
        self.maze_load_generation(T, W, M, np.zeros(N, dtype=np.int32))

    def maze_load_generation(self, T, W, M, gens) -> None:
        """Replace all organisms with a fresh cohort placed at the maze start."""
        s = self.org
        s.n = 0
        N = len(T)
        rng = self.rng.spawn
        sx, sy = self.world.start
        pos = np.tile(np.array([[sx, sy]], dtype=np.float32), (N, 1)) + rng.normal(0, 3.0, (N, 2)).astype(np.float32)
        ids = self.new_ids(N)
        s.add(N, id=ids, parent_a=-1, parent_b=-1, generation=gens, species=-1, pos=pos, start_pos=pos,
              ang=rng.uniform(-np.pi, np.pi, N), energy=100.0, health=100.0, traits=T, w=W, mask=M, w_eff=W * M,
              finish_time=-1.0, last_attacker=-1)
        self.lineage.recs = {}
        self.lineage.add_founders(ids, np.asarray(gens, dtype=np.int32), self.time, np.full(N, -1))
        self.maze_start_time = self.time

    # ===================================================================================
    # one simulation tick
    # ===================================================================================
    def step(self) -> None:
        cfg = self.cfg
        dt = cfg.dt
        self.tick += 1
        self.time += dt
        self._pending = None
        self.env.update(dt)
        if not self.maze_mode:
            self.food.grow(dt, self.ledger)
            self.food.spawn_step(self, dt)
        else:
            self.food.energy[0] = self.food.max_energy[0]
        n = self.org.n
        if n > 0:
            self._cap = self.energy_capacity(self.org.traits[:n, G_SIZE])
            self._sense()
            self._think()
            self._act(dt)
            self._physiology(dt)
            if not self.maze_mode:
                self._plan_reproduction()
        self._cull()
        self._spawn_pending()
        if self.tick % 10 == 0 and self.org.n:
            g = self.org.generation[: self.org.n]
            self.mean_generation = float(g.mean())
            self.max_generation = int(g.max())
        if self.tick % cfg.species_interval * (3 if cfg.optimized_mode else 1) == 0 or self.tick == 1:
            self.species.update(self)
        if self.tick % 1000 == 0:
            self.lineage.prune(self.org.id[: self.org.n].tolist())
        self.stats.maybe_record()

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    # ===================================================================================
    # sensing
    # ===================================================================================
    def _sense(self) -> None:
        s, W, cfg, env = self.org, self.world, self.cfg, self.env
        n = s.n
        pos, ang, T = s.pos[:n], s.ang[:n], s.traits[:n]
        vis, half, size = T[:, G_VIS], np.radians(T[:, G_VANG]) * 0.5, T[:, G_SIZE]
        near_r = size * 2.5 + 6.0
        f32 = np.float32
        vmax = float(vis.max())
        cell = max(40.0, vmax / 3.0)
        R = int(math.ceil(vmax / cell))
        # ---------------- organisms ----------------
        self.ogrid.set_cell(cell)
        self.ogrid.build(pos)
        op = self.ogrid.pairs(pos, R)
        src, dst = op.src, op.dst
        dx = pos[dst, 0] - pos[src, 0]
        dy = pos[dst, 1] - pos[src, 1]
        d = np.hypot(dx, dy).astype(f32)
        ang_to = np.arctan2(dy, dx)
        rel = wrap_angle(ang_to - ang[src]).astype(f32)
        notself = src != dst
        visible = notself & (d < vis[src]) & ((np.abs(rel) <= half[src]) | (d < near_r[src]))
        strength = size * (0.5 + T[:, G_ATTACK])
        inf = f32(np.inf)
        i_any, _ = op.segment_argmin(np.where(visible, d, inf))
        i_thr, _ = op.segment_argmin(np.where(visible & (strength[dst] > 1.15 * strength[src]), d, inf))
        i_tar, _ = op.segment_argmin(np.where(visible & (strength[dst] < 0.87 * strength[src]), d, inf))
        vcount = np.bincount(src, weights=visible.astype(f32), minlength=n)
        dang = ang[dst] - ang[src]
        acos = np.bincount(src, weights=np.cos(dang) * visible, minlength=n)
        asin = np.bincount(src, weights=np.sin(dang) * visible, minlength=n)
        social_near = np.bincount(src, weights=(visible & (d < 40.0)).astype(f32), minlength=n)

        def dirprox(idx, rng_arr=vis):
            has = idx >= 0
            di = np.zeros(n, f32)
            px = np.zeros(n, f32)
            if has.any():
                k = idx[has]
                di[has] = rel[k] / math.pi
                px[has] = np.clip(1.0 - d[k] / rng_arr[has], 0.0, 1.0)
            return di, px, has

        cols: Dict[str, np.ndarray] = {}
        cols["other_dir"], cols["other_prox"], has_any = dirprox(i_any)
        sz = np.zeros(n, f32)
        if has_any.any():
            sz[has_any] = np.tanh(np.log(size[dst[i_any[has_any]]] / size[has_any]))
        cols["other_size"] = sz
        cols["threat_dir"], cols["threat_prox"], has_thr = dirprox(i_thr)
        cols["target_dir"], cols["target_prox"], has_tar = dirprox(i_tar)
        cols["neigh_density"] = np.tanh(vcount / 4.0).astype(f32)
        cnt = np.maximum(vcount, 1.0)
        cols["align_cos"] = (acos / cnt).astype(f32)
        cols["align_sin"] = (asin / cnt).astype(f32)
        self._cache["op"] = (op, d, rel)
        # ---------------- food ----------------
        food = self.food
        fn = food.n
        if fn > 0:
            fpos = food.pos[:fn]
            self.fgrid.set_cell(cell)
            self.fgrid.build(fpos)
            fp = self.fgrid.pairs(pos, R)
            fs, fd_ = fp.src, fp.dst
            fdx = fpos[fd_, 0] - pos[fs, 0]
            fdy = fpos[fd_, 1] - pos[fs, 1]
            fdist = np.hypot(fdx, fdy).astype(f32)
            frel = wrap_angle(np.arctan2(fdy, fdx) - ang[fs]).astype(f32)
            in_cone = (fdist < vis[fs]) & ((np.abs(frel) <= half[fs]) | (fdist < near_r[fs]))
            i_food, _ = fp.segment_argmin(np.where(in_cone, fdist, inf))
            i_reach, _ = fp.segment_argmin(np.where(fdist < size[fs] + REACH_EXTRA, fdist, inf))
            has_food = i_food >= 0
            fdir = np.zeros(n, f32); fprox = np.zeros(n, f32); fen = np.zeros(n, f32); fkind = np.zeros(n, f32)
            if has_food.any():
                k = i_food[has_food]
                fdir[has_food] = frel[k] / math.pi
                fprox[has_food] = np.clip(1.0 - fdist[k] / vis[has_food], 0.0, 1.0)
                tgt = fd_[k]
                fen[has_food] = np.clip(food.energy[tgt] / (1.5 * cfg.food_max_energy), 0.0, 1.0)
                fkind[has_food] = (food.kind[tgt] == KIND_CARRION)
            self._cache["fp"] = (fp, i_reach)
        else:
            has_food = np.zeros(n, bool)
            fdir = np.zeros(n, f32); fprox = np.zeros(n, f32); fen = np.zeros(n, f32); fkind = np.zeros(n, f32)
            self._cache["fp"] = None
        cols["food_dir"], cols["food_prox"], cols["food_energy"], cols["food_kind"] = fdir, fprox, fen, fkind
        # ---------------- self / basic ----------------
        cap = self._cap
        cols["energy"] = np.clip(s.energy[:n] / cap, 0.0, 1.0)
        cols["health"] = np.clip(s.health[:n] / 100.0, 0.0, 1.0)
        cols["age"] = np.clip(s.age[:n] / cfg.max_age, 0.0, 1.0)
        vmax_eff = T[:, G_SPEED] * (0.6 + 0.4 * T[:, G_METAB])
        cols["speed"] = np.clip(s.spd[:n] / vmax_eff, 0.0, 1.0)
        cols["reward"] = np.tanh(s.reward[:n] / 5.0)
        cols["time_since_eat"] = 1.0 - np.exp(-s.t_eat[:n] / 15.0)
        cols["time_since_repro"] = 1.0 - np.exp(-s.t_repro[:n] / 30.0)
        # ---------------- environment ----------------
        ix, iy = W.cell_index(pos)
        Tloc = W.temp[iy, ix] + env.temp_offset
        self._cache["cell"] = (ix, iy, Tloc)
        cols["temperature"] = np.clip((Tloc - T[:, G_TPREF]) / 0.5, -1.0, 1.0)
        cols["noise"] = self.rng.noise.uniform(-1.0, 1.0, n).astype(f32)
        wall = np.minimum(np.minimum(pos[:, 0], W.width - pos[:, 0]), np.minimum(pos[:, 1], W.height - pos[:, 1]))
        cols["wall_prox"] = np.clip(1.0 - wall / 100.0, 0.0, 1.0)
        cols["water_prox"] = np.clip(1.0 - W.water_dist[iy, ix] * W.cell / 150.0, 0.0, 1.0)

        def whisker(off):
            a = ang + off
            best = np.zeros(n, f32)
            for fr in (1 / 3, 2 / 3, 1.0):
                px = pos[:, 0] + np.cos(a) * WHISKER_LEN * fr
                py = pos[:, 1] + np.sin(a) * WHISKER_LEN * fr
                b = W.blocked_at(px, py)
                best = np.maximum(best, b * (1.0 - (fr - 1 / 3)))
            return best.astype(f32)

        cols["obs_left"], cols["obs_front"], cols["obs_right"] = whisker(0.7), whisker(0.0), whisker(-0.7)
        # ---------------- memory ----------------
        cols["prev_turn"] = s.prev_turn[:n]
        cols["prev_thrust"] = s.prev_thr[:n]
        cols["recent_danger"] = np.tanh(s.danger[:n] / 15.0)
        cols["recent_attack"] = s.recent_atk[:n]
        cols["recent_food"] = s.recent_food[:n]
        # EMA of 'saw food'
        rf = s.recent_food[:n]
        rf += (has_food.astype(f32) - rf) * min(1.0, cfg.dt / 3.0)
        self._cols = cols
        self._has_food, self._has_thr, self._has_tar = has_food, has_thr, has_tar
        self._social_near = social_near
        X = np.stack([cols[nm] for nm in ALL_SENSORS], axis=1).astype(f32)
        self._Xall = X
        self._X = X[:, self.sensor_index]

    # ===================================================================================
    # thinking
    # ===================================================================================
    def _think(self) -> None:
        s = self.org
        n = s.n
        X = self._X
        hprev = s.hidden[:n] if self.spec.state_size else None
        out, h = self.backend.forward(self.spec, s.w_eff[:n], X, hprev)
        bad = ~np.isfinite(out)
        if bad.any():
            rows = np.flatnonzero(bad.any(axis=1))
            self.nan_events += len(rows)
            out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
            # quarantine corrupted genomes so the problem cannot spread silently
            s.w[rows] = np.nan_to_num(s.w[rows], nan=0.0, posinf=0.0, neginf=0.0)
            s.w_eff[rows] = s.w[rows] * s.mask[rows]
        if self.spec.state_size:
            s.hidden[:n] = np.nan_to_num(h)
        s.last_in[:n] = X
        s.last_out[:n] = out
        s.activity[:n] = np.mean(np.abs(out), axis=1)
        self._out = out

    # ===================================================================================
    # acting: movement, eating, attacking
    # ===================================================================================
    def _spend(self, idx, amount: np.ndarray, key: str) -> np.ndarray:
        """Deduct energy (clamped at available); returns the amount actually spent."""
        s = self.org
        e = s.energy[idx]
        actual = np.minimum(e, np.maximum(amount, 0.0))
        s.energy[idx] = e - actual
        s.spent[idx] += actual
        self.ledger.add(key, float(actual.sum(dtype=np.float64)))
        return actual

    def _act(self, dt: float) -> None:
        s, W, cfg, env = self.org, self.world, self.cfg, self.env
        n = s.n
        out = self._out
        T = s.traits[:n]
        pos, ang, spd = s.pos[:n], s.ang[:n], s.spd[:n]
        size = T[:, G_SIZE]
        frozen = s.frozen[:n]
        f32 = np.float32
        ix, iy, Tloc = self._cache["cell"]
        # ---- steering & speed ----
        turn = out[:, O_TURN]
        thr = out[:, O_THRUST]
        sprint = out[:, O_SPRINT] > 0.3
        gait = 0.35 + 0.65 * (out[:, O_GAIT] * 0.5 + 0.5)
        turn_rate = 3.0 * np.sqrt(5.0 / size)
        ang += turn * turn_rate * dt * (~frozen)
        ang[:] = wrap_angle(ang)
        terr_mult = SPEED_MULT[W.terrain[iy, ix]]
        vmax = T[:, G_SPEED] * (0.6 + 0.4 * T[:, G_METAB]) * gait * np.where(sprint, 1.35, 1.0) * np.maximum(terr_mult, 0.4)
        acc = T[:, G_ACCEL]
        spd += np.where(thr > 0, acc * thr, 2.0 * acc * thr) * dt
        spd *= (1.0 - 0.6 * dt)
        np.clip(spd, 0.0, vmax, out=spd)
        spd[frozen] = 0.0
        fwd = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        drift = env.gravity[None, :] * dt * (5.0 / size)[:, None]
        newp = pos + fwd * (spd * dt)[:, None] + drift * (~frozen)[:, None]
        # ---- collisions (axis-separated) ----
        bx = W.blocked_at(newp[:, 0], pos[:, 1])
        by = W.blocked_at(pos[:, 0], newp[:, 1])
        newp[bx, 0] = pos[bx, 0]
        newp[by, 1] = pos[by, 1]
        bump = bx | by
        lo, hi = 1.0, np.array([W.width - 1.0, W.height - 1.0], dtype=f32)
        outside = (newp[:, 0] < lo) | (newp[:, 1] < lo) | (newp[:, 0] > hi[0]) | (newp[:, 1] > hi[1])
        newp = np.clip(newp, lo, hi)
        bump |= outside
        spd[bump] *= 0.3
        moved = np.hypot(newp[:, 0] - pos[:, 0], newp[:, 1] - pos[:, 1])
        s.distance[:n] += moved
        pos[:] = newp
        disp = np.hypot(pos[:, 0] - s.start_pos[:n, 0], pos[:, 1] - s.start_pos[:n, 1])
        s.max_disp[:n] = np.maximum(s.max_disp[:n], disp)
        s.home_t[:n] += (disp < 120.0) * dt
        # ---- behavioural accumulators (measurements only; never fed back) ----
        c = self._cols
        fast = spd > 0.3 * vmax
        s.pursuit_t[:n] += ((self._has_tar & (np.abs(c["target_dir"]) < 0.2) & fast)) * dt
        s.forage_t[:n] += ((self._has_food & (np.abs(c["food_dir"]) < 0.2) & fast)) * dt
        s.flee_t[:n] += ((self._has_thr & (np.abs(c["threat_dir"]) > 0.7) & fast)) * dt
        s.social_t[:n] += (self._social_near > 0) * dt
        still = spd < 0.12 * np.maximum(vmax, 1.0)
        s.still_t[:n] += still * dt
        s.still_streak[:n] = np.where(still, s.still_streak[:n] + dt, 0.0)
        s.turn_abs[:n] += np.abs(turn) * dt
        s.turn_sq[:n] += turn * turn * dt
        s.prev_turn[:n] = turn
        s.prev_thr[:n] = thr
        # ---- eating ----
        if not self.maze_mode:
            self._eat(dt, out, T, size, frozen)
        # ---- attacking ----
        if not self.maze_mode:
            self._attack(dt, out, T, size, frozen, vmax)
        else:
            self._maze_check_goal()
        s.recent_atk[:n] *= (1.0 - min(1.0, dt / 2.0))

    # ----------------------------------------------------------------------------
    def _diet_efficiency(self, diet: np.ndarray, eff: np.ndarray, carrion: np.ndarray) -> np.ndarray:
        plant = np.clip(1.0 - 0.9 * diet, 0.08, 1.0)
        meat = np.clip(0.12 + 0.88 * diet, 0.12, 1.0)
        return np.where(carrion, meat, plant) * eff

    def _eat(self, dt, out, T, size, frozen) -> None:
        s, food, cfg = self.org, self.food, self.cfg
        n = s.n
        fc = self._cache["fp"]
        if fc is None or food.n == 0:
            s.t_eat[:n] += dt
            s.reward[:n] *= (1.0 - min(1.0, dt / 3.0))
            return
        fp, i_reach = fc
        cap = self._cap
        room = cap - s.energy[:n]
        cand = (i_reach >= 0) & (out[:, O_EAT] > 0) & ~frozen & (room > 0.5) & (s.health[:n] > 0)
        gain_now = np.zeros(n, dtype=np.float32)
        if cand.any():
            i = np.flatnonzero(cand)
            f = fp.dst[i_reach[i]]
            carrion = food.kind[f] == KIND_CARRION
            eff = self._diet_efficiency(T[i, G_DIET], T[i, G_EFF], carrion)
            bite = np.minimum(cfg.bite_rate * dt, room[i] / np.maximum(eff, 1e-3)).astype(np.float32)
            demand = np.bincount(f, weights=bite, minlength=food.n)
            scale = np.minimum(1.0, food.energy[: food.n] / np.maximum(demand, 1e-9))
            take = bite * scale[f]
            gain = take * eff
            food.energy[: food.n] -= np.bincount(f, weights=take, minlength=food.n).astype(np.float32)
            np.maximum(food.energy[: food.n], 0.0, out=food.energy[: food.n])
            s.energy[i] += gain
            self.ledger.add("out_assimilation", float(take.sum(dtype=np.float64) - gain.sum(dtype=np.float64)))
            s.gained[i] += gain
            np.add.at(s.food_carrion, i[carrion], gain[carrion])
            np.add.at(s.food_plant, i[~carrion], gain[~carrion])
            gain_now[i] = gain
            s.t_eat[i] = 0.0
            if self.record_fx and len(self.fx) < 1500:
                for k in range(min(len(i), 30)):
                    self.fx.append(("eat", float(s.pos[i[k], 0]), float(s.pos[i[k], 1])))
            # drop depleted items
            dead = food.energy[: food.n] < 0.4
            if dead.any():
                self.ledger.add("out_decay", food.remove(dead))
        eaten = gain_now > 0
        s.t_eat[:n] += dt * (~eaten)
        s.reward[:n] += (gain_now / dt - s.reward[:n]) * min(1.0, dt / 3.0)

    # ----------------------------------------------------------------------------
    def _attack(self, dt, out, T, size, frozen, vmax) -> None:
        s, cfg, W = self.org, self.cfg, self.world
        n = s.n
        s.atk_cd[:n] = np.maximum(s.atk_cd[:n] - dt, 0.0)
        swing = (out[:, O_ATTACK] > 0.2) & (s.atk_cd[:n] <= 0) & ~frozen & (s.energy[:n] > 2.0) & (s.health[:n] > 0) \
            & (T[:, G_ATTACK] > 0.02)
        if not swing.any():
            return
        op, d, rel = self._cache["op"]
        src, dst = op.src, op.dst
        sw = np.flatnonzero(swing)
        # swinging costs energy whether or not it connects
        self._spend(sw, cfg.attack_cost * T[sw, G_ATTACK], "out_attack")
        s.atk_cd[sw] = 0.5
        s.recent_atk[sw] = 1.0
        s.attacks[sw] += 1
        ok = swing[src] & (src != dst) & (d < size[src] + size[dst] + REACH_EXTRA) & (np.abs(rel) < ATTACK_ARC) \
            & (s.health[:n][dst] > 0)
        if not ok.any():
            return
        ps, pd, dd = src[ok], dst[ok], d[ok]
        order = np.lexsort((dd, ps))
        ps, pd, dd = ps[order], pd[order], dd[order]
        first = np.concatenate([[True], ps[1:] != ps[:-1]])
        a, t = ps[first], pd[first]
        pos = s.pos[:n]
        ang = s.ang[:n]
        spd = s.spd[:n]
        u = pos[t] - pos[a]
        un = u / np.maximum(np.hypot(u[:, 0], u[:, 1]), 1e-6)[:, None]
        va = np.stack([np.cos(ang[a]), np.sin(ang[a])], 1) * spd[a, None]
        vt = np.stack([np.cos(ang[t]), np.sin(ang[t])], 1) * spd[t, None]
        closing = np.einsum("ij,ij->i", va - vt, un)
        f_speed = 1.0 + 0.5 * np.clip(closing / 60.0, -0.5, 1.5)
        power = T[a, G_ATTACK] * (size[a] / 5.0) * f_speed
        # surprise: attacker lies outside the target's vision cone
        to_att = np.arctan2(-u[:, 1], -u[:, 0])
        off_axis = np.abs(wrap_angle(to_att - ang[t]))
        half_t = np.radians(T[t, G_VANG]) * 0.5
        surprise = np.where(off_axis > half_t, 1.6, 1.0)
        defense = np.sqrt(size[t] / 5.0)
        ix, iy, _ = self._cache["cell"]
        safe = W.safe[iy[t], ix[t]] | W.safe[iy[a], ix[a]]
        dmg = (cfg.attack_damage * power * surprise / defense * (~safe)).astype(np.float32)
        want = dmg * 0.8
        wsum = np.bincount(t, weights=want, minlength=n)
        scale = np.minimum(1.0, s.energy[:n] / np.maximum(wsum, 1e-9))
        stolen = want * scale[t]
        # apply to targets
        dsum = np.bincount(t, weights=dmg, minlength=n).astype(np.float32)
        ssum = np.bincount(t, weights=stolen, minlength=n).astype(np.float32)
        s.health[:n] -= dsum
        s.cause_dmg[:n, C_KILLED] += dsum
        s.energy[:n] -= ssum
        np.maximum(s.energy[:n], 0.0, out=s.energy[:n])
        s.dmg_taken[:n] += dsum
        s.danger[:n] += dsum
        s.last_attacker[t] = s.id[a]
        # attackers gain meat energy
        eff = self._diet_efficiency(T[a, G_DIET], T[a, G_EFF], np.ones(len(a), bool))
        gain = np.minimum(stolen * eff, np.maximum(self._cap[a] - s.energy[a], 0.0)).astype(np.float32)
        np.add.at(s.energy, a, gain)
        np.add.at(s.food_meat, a, gain)
        np.add.at(s.gained, a, gain)
        np.add.at(s.dmg_dealt, a, dmg)
        self.ledger.add("out_predation", float(stolen.sum(dtype=np.float64) - gain.sum(dtype=np.float64)))
        amb = s.still_streak[a] > 1.0
        np.add.at(s.ambush, a[amb], 1.0)
        s.reward[a] += gain / cfg.dt * 0.3
        if self.record_fx and len(self.fx) < 1500:
            for k in range(min(len(a), 20)):
                self.fx.append(("hit", float(pos[t[k], 0]), float(pos[t[k], 1])))

    # ----------------------------------------------------------------------------
    def _maze_check_goal(self) -> None:
        s = self.org
        n = s.n
        gx, gy = self.world.goal
        d = np.hypot(s.pos[:n, 0] - gx, s.pos[:n, 1] - gy)
        newly = (d < 18.0) & (s.finish_time[:n] < 0)
        if newly.any():
            s.finish_time[:n][newly] = self.time - self.maze_start_time
            s.frozen[:n][newly] = True

    # ===================================================================================
    # physiology: metabolism, temperature, hazards, ageing
    # ===================================================================================
    def _hurt(self, amount: np.ndarray, cause: int) -> None:
        s = self.org
        n = s.n
        s.health[:n] -= amount
        s.cause_dmg[:n, cause] += amount

    def _physiology(self, dt: float) -> None:
        s, cfg, W, env = self.org, self.cfg, self.world, self.env
        n = s.n
        T = s.traits[:n]
        s.age[:n] += dt
        if self.maze_mode:
            return
        size = T[:, G_SIZE]
        metab, eff = T[:, G_METAB], T[:, G_EFF]
        sc = (size / 5.0)
        spd = s.spd[:n]
        sprint = self._out[:, O_SPRINT] > 0.3
        basal = cfg.basal_cost * metab * sc ** 1.5 * eff ** 1.5
        move = cfg.move_cost * spd ** 2 * sc * (1.0 + 1.5 * sprint)
        vision = 0.003 * T[:, G_VIS] * (T[:, G_VANG] / 360.0)
        upkeep = 0.08 * T[:, G_ATTACK]
        idx = np.arange(n)
        self._spend(idx, (basal + move + vision + upkeep) * dt, "out_metabolism")
        # temperature stress
        ix, iy, Tloc = self._cache["cell"]
        excess = np.maximum(np.abs(Tloc - T[:, G_TPREF]) - T[:, G_TTOL], 0.0)
        if excess.any():
            self._spend(idx, 6.0 * excess * dt, "out_temperature")
            self._hurt((12.0 * excess * dt).astype(np.float32), C_TEMP)
        # hazards
        dz = (W.danger[iy, ix] + W.danger_extra[iy, ix]) * (0.6 + 0.4 * metab) * dt
        in_safe = W.safe[iy, ix]
        dz = np.where(in_safe, 0.0, dz)
        if dz.any():
            self._hurt(dz.astype(np.float32), C_TOXIC)
        if env.flood_level > 0:
            fl = W.flooded[iy, ix]
            if fl.any():
                slow = 1.0 - np.clip(T[:, G_SPEED] / 130.0, 0.0, 0.9)
                self._hurt((fl * 6.0 * slow * env.flood_level * dt).astype(np.float32), C_DISASTER)
        # starvation
        starving = s.energy[:n] <= 0.0
        if starving.any():
            self._hurt((starving * 8.0 * dt).astype(np.float32), C_STARVE)
        # senescence
        frac = s.age[:n] / cfg.max_age
        sen = 8.0 * np.maximum((frac - 0.7) / 0.3, 0.0) ** 2 * dt
        if sen.any():
            self._hurt(sen.astype(np.float32), C_AGE)
        # healing
        heal = (s.energy[:n] > 0.5 * self._cap) & (s.health[:n] < 100.0)
        s.health[:n] += (heal * (1.0 + 2.0 * in_safe) * dt).astype(np.float32)
        np.minimum(s.health[:n], 100.0, out=s.health[:n])
        s.cooldown[:n] = np.maximum(s.cooldown[:n] - dt * (0.6 + 0.4 * metab), 0.0)
        s.t_repro[:n] += dt
        s.danger[:n] *= (1.0 - min(1.0, dt / 3.0))

    # ===================================================================================
    # reproduction
    # ===================================================================================
    def _plan_reproduction(self) -> None:
        s, cfg = self.org, self.cfg
        n = s.n
        free = cfg.population_cap - n
        if free <= 0:
            return
        out = self._out
        T = s.traits[:n]
        cap = self._cap
        need = T[:, G_REPRO].copy()
        pen = self.strategy.repro_penalty(self, s.species[:n])
        if pen is not None:
            need = need + pen
        efrac = s.energy[:n] / cap
        ready = (out[:, O_REPRO] > 0) & (s.health[:n] > 0) & (efrac >= need) & (s.age[:n] >= cfg.min_reproduction_age) \
            & (s.cooldown[:n] <= 0) & ~s.frozen[:n]
        if not ready.any():
            return
        mode = self.strategy.reproduction_mode(cfg)
        ridx = np.flatnonzero(ready)
        order = ridx[np.argsort(-efrac[ridx], kind="stable")]
        pairs: List[tuple] = []
        used = set()
        if mode in ("sexual", "mixed"):
            pairs, used = self._find_mates(order, ready, efrac)
        singles = []
        if mode in ("asexual", "mixed"):
            singles = [int(a) for a in order if int(a) not in used]
        births_a = [p[0] for p in pairs] + singles
        births_b = [p[1] for p in pairs] + [-1] * len(singles)
        births_a, births_b = births_a[:free], births_b[:free]
        if not births_a:
            return
        pa = np.array(births_a, dtype=np.int64)
        pb = np.array(births_b, dtype=np.int64)
        cost = cfg.birth_cost
        sexual = pb >= 0
        child_e = np.full(len(pa), 0.85 * cost, dtype=np.float32)
        paid = np.zeros(len(pa))
        pay_a = np.where(sexual, 0.6 * cost, cost)
        paid += self._spend(pa, pay_a, "out_birth")
        if sexual.any():
            paid[sexual] += self._spend(pb[sexual], np.full(int(sexual.sum()), 0.6 * cost), "out_birth")
        # what is not delivered to the child is lost; _spend already booked everything as out_birth,
        # so credit back the child's energy as it re-enters the system as organism energy.
        child_e = np.minimum(child_e, paid.astype(np.float32))
        self.ledger.add("out_birth", -float(child_e.sum(dtype=np.float64)))
        cd = cfg.reproduction_cooldown
        for arr in (pa, pb[sexual]):
            s.cooldown[arr] = cd
            s.t_repro[arr] = 0.0
            s.offspring[arr] += 1
        self._pending = self._make_children(pa, pb, child_e)

    def _find_mates(self, order, ready, efrac):
        s, cfg = self.org, self.cfg
        n = s.n
        op, d, rel = self._cache["op"]
        T = s.traits[:n]
        cap = self._cap
        eligible = (efrac >= 0.4) & (s.age[:n] >= cfg.min_reproduction_age) & (s.cooldown[:n] <= 0) \
            & ~s.frozen[:n] & (s.health[:n] > 0)
        src, dst = op.src, op.dst
        sel = ready[src] & eligible[dst] & (src != dst) & (d < T[src, G_VIS])
        if not sel.any():
            return [], set()
        cs, cdst, cdist = src[sel], dst[sel], d[sel]
        comp = pair_distance(T[cs], s.w_eff[:n][cs], T[cdst], s.w_eff[:n][cdst]) <= cfg.species_threshold
        cs, cdst, cdist = cs[comp], cdst[comp], cdist[comp]
        if cs.size == 0:
            return [], set()
        if self.strategy.mate_choice == "fittest":
            key = -(efrac[cdst] * (0.5 + s.age[:n][cdst] / cfg.max_age))
        else:
            key = cdist
        o = np.lexsort((key, cs))
        cs, cdst = cs[o], cdst[o]
        first = np.concatenate([[True], cs[1:] != cs[:-1]])
        proposal = dict(zip(cs[first].tolist(), cdst[first].tolist()))
        used = set()
        pairs = []
        for a in order.tolist():
            if a in used:
                continue
            b = proposal.get(a)
            if b is None or b in used or b == a:
                continue
            used.add(a)
            used.add(b)
            pairs.append((a, b))
        return pairs, used

    def _make_children(self, pa: np.ndarray, pb: np.ndarray, child_e: np.ndarray) -> dict:
        s, cfg = self.org, self.cfg
        k = len(pa)
        b_idx = np.where(pb >= 0, pb, pa)
        tA, wA, mA = s.traits[pa], s.w[pa], s.mask[pa]
        tB, wB, mB = s.traits[b_idx], s.w[b_idx], s.mask[b_idx]
        cT, cW, cM = crossover(tA, wA, mA, tB, wB, mB, self.rng.mutation, cfg.crossover_mode)
        rate = self.strategy.mutation_rate(cfg) * cT[:, G_MUT]
        changed = mutate(cT, cW, cM, rate.astype(np.float32), cfg.mutation_strength, self.rng.mutation, self.spec,
                         self.strategy.structural_rate(cfg), cfg.evolvable_mutation)
        rel = (cT - tA) / TRAIT_RANGE
        mt = np.argmax(np.abs(rel), axis=1)
        md = rel[np.arange(k), mt]
        rs = self.rng.spawn
        ang = rs.uniform(-np.pi, np.pi, k)
        rad = rs.uniform(8.0, 18.0, k)
        pos = s.pos[pa] + np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32) * rad[:, None]
        pos = np.clip(pos, 1.0, [self.world.width - 1.0, self.world.height - 1.0]).astype(np.float32)
        blocked = self.world.blocked_at(pos[:, 0], pos[:, 1])
        pos[blocked] = s.pos[pa][blocked]
        gen = np.maximum(s.generation[pa], s.generation[b_idx]) + 1
        return dict(pa=s.id[pa].copy(), pb=np.where(pb >= 0, s.id[b_idx], -1), traits=cT, w=cW, mask=cM, pos=pos,
                    gen=gen.astype(np.int32), species=s.species[pa].copy(), energy=child_e, mut=changed,
                    mt=mt, md=md, ang=rs.uniform(-np.pi, np.pi, k))

    def _spawn_pending(self) -> None:
        p = self._pending
        if not p:
            return
        self._pending = None
        s = self.org
        k = len(p["pa"])
        ids = self.new_ids(k)
        s.add(k, id=ids, parent_a=p["pa"], parent_b=p["pb"], generation=p["gen"], species=p["species"],
              pos=p["pos"], start_pos=p["pos"], ang=p["ang"], energy=p["energy"], health=100.0,
              traits=p["traits"], w=p["w"], mask=p["mask"], w_eff=p["w"] * p["mask"], finish_time=-1.0,
              last_attacker=-1)
        self.lineage.add_births(ids, p["pa"], p["pb"], p["gen"], self.time, p["species"], p["mut"], p["mt"], p["md"])
        self.births_total += k
        self.recent_births += k
        if self.record_fx and len(self.fx) < 1500:
            for i in range(min(k, 20)):
                self.fx.append(("birth", float(p["pos"][i, 0]), float(p["pos"][i, 1])))

    # ===================================================================================
    # death
    # ===================================================================================
    def _cull(self) -> None:
        s = self.org
        n = s.n
        if n == 0:
            return
        dead = s.health[:n] <= 0
        if not dead.any():
            return
        idx = np.flatnonzero(dead)
        cause = np.argmax(s.cause_dmg[idx], axis=1)
        none = s.cause_dmg[idx].sum(axis=1) <= 0
        cause[none] = C_DISASTER
        # kill credit
        killed = cause == C_KILLED
        if killed.any():
            att = s.indices_of(s.last_attacker[idx[killed]])
            att = att[att >= 0]
            if att.size:
                s.kills[: s.n] += np.bincount(att, minlength=s.n).astype(np.int32)
        # carrion
        en = np.maximum(s.energy[idx], 0.0)
        mk = en > 2.0
        if mk.any():
            ce = (en[mk] * 0.7).astype(np.float32)
            self.food.add(s.pos[idx[mk]], ce, ce, np.zeros(mk.sum(), np.float32), np.full(int(mk.sum()), KIND_CARRION, np.uint8))
        self.ledger.add("out_corpse", float(en.sum(dtype=np.float64) - (en[mk] * 0.7).sum(dtype=np.float64)))
        ages = s.age[idx]
        self.lineage.note_death(s.id[idx], self.time, ages, cause)
        self.species.record_death(s.species[idx], ages)
        self.deaths_total += len(idx)
        self.recent_deaths += len(idx)
        self.death_causes += np.bincount(cause, minlength=6)
        self.recent_lifespans.extend(ages.tolist())
        if len(self.recent_lifespans) > 5000:
            self.recent_lifespans = self.recent_lifespans[-2500:]
        if self.record_fx and len(self.fx) < 1500:
            for i in idx[:20]:
                self.fx.append(("death", float(s.pos[i, 0]), float(s.pos[i, 1]), int(s.species[i])))
        s.remove(dead)

    # ===================================================================================
    # derived metrics
    # ===================================================================================
    def total_energy(self) -> float:
        n = self.org.n
        return float(self.org.energy[:n].sum(dtype=np.float64)) + self.food.total_energy()

    def energy_balance_error(self) -> float:
        """|actual - (inflow - outflow)|: should be ~0 (float rounding only)."""
        expected = self.ledger.inflow() - self.ledger.outflow()
        return abs(self.total_energy() - expected)

    def fitness(self, idx=None) -> np.ndarray:
        """Emergent, *descriptive* fitness in [0,1) – never used for selection."""
        s = self.org
        n = s.n
        sl = slice(0, n) if idx is None else idx
        off = s.offspring[sl].astype(np.float32)
        eff = s.gained[sl] / np.maximum(s.spent[sl], 1.0)
        surv = s.age[sl] / self.cfg.max_age
        raw = off * 1.0 + 0.8 * np.minimum(eff, 3.0) + 1.2 * surv
        return np.tanh(raw / 4.0)

    def state_hash(self) -> str:
        h = hashlib.sha256()
        s = self.org
        n = s.n
        for a in (s.id[:n], s.pos[:n], s.energy[:n], s.traits[:n], s.w[:n], s.health[:n]):
            h.update(np.ascontiguousarray(a).tobytes())
        f = self.food
        h.update(np.ascontiguousarray(f.pos[: f.n]).tobytes())
        h.update(np.ascontiguousarray(f.energy[: f.n]).tobytes())
        h.update(str(self.tick).encode())
        return h.hexdigest()[:16]

    def log_event(self, kind: str, text: str, severity: int = 1, data: Optional[dict] = None) -> None:
        self.events.log(kind, text, severity, data)

    # ===================================================================================
    # organism accessors (inspector / tools)
    # ===================================================================================
    def organism_info(self, idx: int) -> dict:
        s = self.org
        T = s.traits[idx]
        age = float(s.age[idx])
        sp = self.species.species.get(int(s.species[idx]))
        return {
            "id": int(s.id[idx]), "name": f"VX-{int(s.id[idx]):04d}",
            "species": sp.name if sp else "—", "species_id": int(s.species[idx]),
            "generation": int(s.generation[idx]), "age": age,
            "energy": float(s.energy[idx] / self.energy_capacity(T[G_SIZE])), "health": float(s.health[idx] / 100.0),
            "parents": [int(s.parent_a[idx]), int(s.parent_b[idx])], "offspring": int(s.offspring[idx]),
            "kills": int(s.kills[idx]), "fitness": float(self.fitness(np.array([idx]))[0]),
            "traits": {n: float(T[i]) for i, n in enumerate(TRAIT_NAMES)},
            "pos": (float(s.pos[idx, 0]), float(s.pos[idx, 1])),
        }

    def genome_of(self, idx: int) -> Genome:
        s = self.org
        sp = self.species.species.get(int(s.species[idx]))
        meta = {"id": int(s.id[idx]), "generation": int(s.generation[idx]),
                "species": sp.name if sp else None, "seed": self.cfg.seed}
        return Genome(s.traits[idx], s.w[idx], s.mask[idx], self.spec, meta)

    def network_trace(self, idx: int):
        """(outputs, activations per layer) for the organism's most recent inputs – for the visualiser."""
        s = self.org
        h = s.hidden[idx] if self.spec.state_size else None
        return forward_one(self.spec, s.w_eff[idx], s.last_in[idx], h)

    def nearest_organism(self, x: float, y: float, max_dist: float = 40.0) -> int:
        s = self.org
        n = s.n
        if n == 0:
            return -1
        d = np.hypot(s.pos[:n, 0] - x, s.pos[:n, 1] - y) - s.traits[:n, G_SIZE]
        i = int(np.argmin(d))
        return i if d[i] <= max_dist else -1

    # ===================================================================================
    # interventions (god mode, timeline, experiments).  All are logged for replay.
    # ===================================================================================
    def apply_action(self, name: str, log: bool = True, **p):
        fn = getattr(self, f"_act_{name}", None)
        if fn is None:
            raise ValueError(f"unknown action '{name}'")
        if log:
            self.action_log.append([self.tick, name, _jsonable(p)])
        return fn(**p)

    def _act_spawn_food(self, x, y, n=12, radius=45.0, **_):
        rng = self.rng.spawn
        pos = np.stack([x + rng.normal(0, radius * 0.5, n), y + rng.normal(0, radius * 0.5, n)], 1).astype(np.float32)
        pos = np.clip(pos, 1, [self.world.width - 1, self.world.height - 1])
        ok = ~self.world.blocked_at(pos[:, 0], pos[:, 1])
        pos = pos[ok]
        m = len(pos)
        if m:
            me = np.full(m, self.cfg.food_max_energy * self.world.params.get("food_value", 1.0), np.float32)
            self.food.add(pos, me, me, me / self.cfg.food_grow_time, np.zeros(m, np.uint8))
            self.ledger.add("in_gift", float(me.sum()))
        return m

    def _act_remove_food(self, x, y, radius=80.0, **_):
        f = self.food
        if f.n == 0:
            return 0
        m = np.hypot(f.pos[: f.n, 0] - x, f.pos[: f.n, 1] - y) < radius
        c = int(m.sum())
        self.ledger.add("out_removed", f.remove(m))
        return c

    def _act_spawn_organism(self, x, y, kind="random", n=1, genome=None, **_):
        s = self.org
        n = int(n)
        free = max(0, self.cfg.population_cap - s.n)
        n = min(n, free)
        if n <= 0:
            return 0
        rng = self.rng.spawn
        if genome is not None:
            g = Genome.from_dict(genome, self.spec)
            T = np.tile(g.traits, (n, 1)); W = np.tile(g.weights, (n, 1)); M = np.tile(g.mask, (n, 1))
        else:
            carn = n if kind == "predator" else 0
            T, W, M = self.make_founder_genomes(n, carn, rng)
            if kind == "herbivore":
                T[:, G_DIET] = np.clip(T[:, G_DIET], 0, 0.3)
        pos = np.stack([x + rng.normal(0, 12, n), y + rng.normal(0, 12, n)], 1).astype(np.float32)
        pos = np.clip(pos, 1, [self.world.width - 1, self.world.height - 1])
        bl = self.world.blocked_at(pos[:, 0], pos[:, 1])
        pos[bl] = self.world.random_free_positions(int(bl.sum()), rng) if bl.any() else pos[bl]
        cap = self.energy_capacity(T[:, G_SIZE])
        e = cap * 0.7
        gen0 = int(self.mean_generation)
        ids = self.new_ids(n)
        s.add(n, id=ids, parent_a=-1, parent_b=-1, generation=gen0, species=-1, pos=pos, start_pos=pos,
              ang=rng.uniform(-np.pi, np.pi, n), energy=e, health=100.0, traits=T, w=W, mask=M, w_eff=W * M,
              finish_time=-1.0, last_attacker=-1)
        self.lineage.add_founders(ids, np.full(n, gen0, np.int32), self.time, np.full(n, -1))
        self.ledger.add("in_gift", float(e.sum()))
        self.species.update(self)
        return n

    def _act_create_predator(self, x, y, n=3, **_):
        return self._act_spawn_organism(x, y, kind="predator", n=n)

    def _act_kill_organism(self, id=None, x=None, y=None, radius=30.0, **_):
        s = self.org
        n = s.n
        if id is not None:
            i = s.index_of(int(id))
            m = np.zeros(n, bool)
            if i >= 0:
                m[i] = True
        else:
            m = np.hypot(s.pos[:n, 0] - x, s.pos[:n, 1] - y) < radius
        s.health[:n][m] = 0.0
        s.cause_dmg[:n][m, C_DISASTER] += 100.0
        c = int(m.sum())
        self._cull()
        return c

    def _act_create_mutation(self, id=None, x=None, y=None, radius=60.0, **_):
        s = self.org
        n = s.n
        if id is not None:
            i = s.index_of(int(id))
            idx = np.array([i]) if i >= 0 else np.zeros(0, int)
        else:
            idx = np.flatnonzero(np.hypot(s.pos[:n, 0] - x, s.pos[:n, 1] - y) < radius)
        if idx.size:
            t, w, m = s.traits[idx].copy(), s.w[idx].copy(), s.mask[idx].copy()
            burst_mutation(t, w, m, self.rng.mutation, self.spec)
            s.traits[idx], s.w[idx], s.mask[idx], s.w_eff[idx] = t, w, m, w * m
        return int(idx.size)

    def _act_set_temperature(self, value, **_):
        self.env.base_temp_offset = float(np.clip(value, -1, 1))

    temperature_offset = None  # (name kept free for timeline alias below)

    def _act_temperature_offset(self, value, **_):
        self.env.base_temp_offset = float(np.clip(value, -1, 1))

    def _act_disaster(self, kind, strength=1.0, **_):
        return self.env.trigger(kind, strength)

    def _act_set_gravity(self, gx=0.0, gy=0.0, **_):
        self.env.gravity = np.array([gx, gy], dtype=np.float32)

    def _act_set_mutation_rate(self, value, **_):
        self.cfg.mutation_rate = float(value)

    def _act_set_population_cap(self, value, **_):
        self.cfg.population_cap = int(max(value, 1))
        self.org.ensure(max(0, self.cfg.population_cap - self.org.n) + 8)

    def _act_set_food_density(self, value, **_):
        self.cfg.food_density = float(value)

    def _act_food_scale(self, value, **_):
        self.env.base_food_mult = float(max(0.0, value))

    def _act_water_offset(self, value, **_):
        self.env.base_water_offset = int(value)

    def _act_spawn_predators(self, count=20, **_):
        rng = self.rng.spawn
        pos = self.world.random_free_positions(1, rng)[0]
        return self._act_spawn_organism(float(pos[0]), float(pos[1]), kind="predator", n=int(count))

    def _act_reshuffle_obstacles(self, **_):
        self.world.reshuffle_obstacles(self.rng.world)
        s = self.org
        n = s.n
        if n:
            bl = self.world.blocked_at(s.pos[:n, 0], s.pos[:n, 1])
            if bl.any():
                s.pos[:n][bl] = self.world.random_free_positions(int(bl.sum()), self.rng.world)
        f = self.food
        if f.n:
            bl = self.world.blocked_at(f.pos[: f.n, 0], f.pos[: f.n, 1])
            self.ledger.add("out_removed", f.remove(bl))

    def _act_teleport(self, id, x, y, **_):
        i = self.org.index_of(int(id))
        if i >= 0:
            self.org.pos[i] = [np.clip(x, 1, self.world.width - 1), np.clip(y, 1, self.world.height - 1)]
        return i >= 0

    def _act_freeze(self, id, on=True, **_):
        i = self.org.index_of(int(id))
        if i >= 0:
            self.org.frozen[i] = bool(on)
            if on:
                self.org.spd[i] = 0.0
        return i >= 0

    def _act_give_energy(self, id, amount=50.0, **_):
        s = self.org
        i = s.index_of(int(id))
        if i < 0:
            return False
        cap = float(self.energy_capacity(s.traits[i, G_SIZE]))
        add = min(float(amount), cap - float(s.energy[i]))
        s.energy[i] += add
        self.ledger.add("in_gift", add)
        return True

    def _act_edit_genome(self, id, traits=None, genome=None, **_):
        """Genetic surgery: overwrite traits (and optionally the whole genome) of a living organism."""
        s = self.org
        i = s.index_of(int(id))
        if i < 0:
            return False
        if genome is not None:
            g = Genome.from_dict(genome, self.spec)
            s.traits[i], s.w[i], s.mask[i] = g.traits, g.weights, g.mask
            s.w_eff[i] = g.weights * g.mask
        if traits:
            for k, v in traits.items():
                if k not in IDX:
                    raise GenomeError(f"unknown trait '{k}'")
                j = IDX[k]
                s.traits[i, j] = float(np.clip(v, TRAIT_LO[j], TRAIT_HI[j]))
        return True

    def _act_bottleneck(self, keep=10, **_):
        s = self.org
        n = s.n
        keep = int(min(keep, n))
        surv = self.rng.events.choice(n, size=keep, replace=False)
        dead = np.ones(n, dtype=bool)
        dead[surv] = False
        s.health[:n][dead] = 0.0
        s.cause_dmg[:n][dead, C_DISASTER] += 100.0
        # bottleneck victims leave no carrion in order not to distort the recovery
        s.energy[:n][dead] = 0.0
        self.ledger.add("out_removed", 0.0)
        self._cull()
        self.log_event("bottleneck", f"Population forced through a bottleneck: {n} -> {keep}", 3,
                       {"before": n, "after": keep})
        return keep

    # ===================================================================================
    # persistence hooks (see persistence.save)
    # ===================================================================================
    def state_dict(self) -> dict:
        return {
            "config": self.cfg.to_dict(), "tick": self.tick, "time": self.time, "next_id": self.next_id,
            "mean_generation": self.mean_generation, "max_generation": self.max_generation,
            "rng": self.rng.get_state(), "world": self.world.state_dict(), "env": self.env.state_dict(),
            "food": self.food.state_dict(), "org": self.org.state_dict(), "species": self.species.state_dict(),
            "lineage": self.lineage.state_dict(), "ledger": dict(self.ledger.v),
            "counters": {"births_total": self.births_total, "deaths_total": self.deaths_total,
                         "death_causes": self.death_causes.copy(), "nan_events": self.nan_events,
                         "recent_births": self.recent_births, "recent_deaths": self.recent_deaths,
                         "recent_lifespans": list(self.recent_lifespans),
                         "initial_energy": getattr(self, "initial_energy", 0.0),
                         "maze_start_time": getattr(self, "maze_start_time", 0.0)},
            "stats": self.stats.state_dict(), "events": self.events.state_dict(),
            "action_log": [list(a) for a in self.action_log],
        }

    def load_state_dict(self, d: dict) -> None:
        self.tick = int(d["tick"])
        self.time = float(d["time"])
        self.next_id = int(d["next_id"])
        self.mean_generation = float(d["mean_generation"])
        self.max_generation = int(d["max_generation"])
        self.rng.set_state(d["rng"])
        self.world.load_state_dict(d["world"])
        self.env.load_state_dict(d["env"])
        self.food.load_state_dict(d["food"])
        self.org.load_state_dict(d["org"])
        self.species.load_state_dict(d["species"])
        self.lineage.load_state_dict(d["lineage"])
        self.ledger.v = {k: float(v) for k, v in d["ledger"].items()}
        c = d["counters"]
        self.births_total, self.deaths_total = int(c["births_total"]), int(c["deaths_total"])
        self.death_causes = np.array(c["death_causes"], dtype=np.int64)
        self.nan_events = int(c["nan_events"])
        self.recent_births, self.recent_deaths = int(c["recent_births"]), int(c["recent_deaths"])
        self.recent_lifespans = list(c["recent_lifespans"])
        self.initial_energy = float(c["initial_energy"])
        self.maze_start_time = float(c["maze_start_time"])
        self.stats.load_state_dict(d["stats"])
        self.events.load_state_dict(d["events"])
        self.action_log = [list(a) for a in d["action_log"]]
        self._pending = None
        self._cache = {}
        self.fx = []


def _jsonable(p: dict) -> dict:
    out = {}
    for k, v in p.items():
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        out[k] = v
    return out
