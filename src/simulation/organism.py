"""Organism storage (structure-of-arrays) and lineage tracking.

Organisms are *not* Python objects: every attribute is a NumPy array indexed by
organism, which lets the simulation sense, think and act for the whole
population with vectorised operations.  Live organisms are always packed in the
first ``n`` slots (dead ones are compacted away each tick).  Organism ids are
strictly increasing, therefore ``ids[:n]`` is always sorted and id -> index
lookup is a binary search.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from ..evolution.genome import N_TRAITS, TRAIT_NAMES
from ..neural.network import NetSpec

CAUSES = ["old age", "starvation", "predation", "temperature", "toxic zone", "disaster"]
C_AGE, C_STARVE, C_KILLED, C_TEMP, C_TOXIC, C_DISASTER = range(6)


def _fields(spec: NetSpec):
    G, P, S = N_TRAITS, spec.n_params, spec.state_size
    f = np.float32
    return [
        ("id", np.int64, ()), ("parent_a", np.int64, ()), ("parent_b", np.int64, ()),
        ("generation", np.int32, ()), ("species", np.int32, ()),
        ("age", f, ()), ("birth_time", f, ()),
        ("pos", f, (2,)), ("start_pos", f, (2,)), ("ang", f, ()), ("spd", f, ()),
        ("energy", f, ()), ("health", f, ()),
        ("traits", f, (G,)), ("w", f, (P,)), ("mask", np.uint8, (P,)), ("w_eff", f, (P,)), ("hidden", f, (S,)),
        ("cooldown", f, ()), ("atk_cd", f, ()), ("frozen", bool, ()),
        ("t_eat", f, ()), ("t_repro", f, ()), ("reward", f, ()), ("prev_turn", f, ()), ("prev_thr", f, ()),
        ("danger", f, ()), ("recent_atk", f, ()), ("recent_food", f, ()), ("activity", f, ()),
        ("last_in", f, (spec.n_in,)), ("last_out", f, (spec.n_out,)),
        ("last_attacker", np.int64, ()), ("cause_dmg", f, (6,)),
        # behavioural / statistical accumulators ---------------------------------
        ("food_plant", f, ()), ("food_meat", f, ()), ("food_carrion", f, ()),
        ("dmg_dealt", f, ()), ("dmg_taken", f, ()), ("distance", f, ()),
        ("offspring", np.int32, ()), ("kills", np.int32, ()),
        ("spent", f, ()), ("gained", f, ()), ("attacks", np.int32, ()),
        ("pursuit_t", f, ()), ("forage_t", f, ()), ("flee_t", f, ()), ("social_t", f, ()),
        ("still_t", f, ()), ("still_streak", f, ()), ("turn_abs", f, ()), ("turn_sq", f, ()),
        ("ambush", f, ()), ("home_t", f, ()), ("max_disp", f, ()), ("finish_time", f, ()),
    ]


class OrganismStore:
    def __init__(self, spec: NetSpec, capacity: int):
        self.spec = spec
        self.n = 0
        self.fields = _fields(spec)
        self.names = [f[0] for f in self.fields]
        self._alloc(capacity)

    def _alloc(self, cap: int) -> None:
        self.cap = cap
        for name, dt, shp in self.fields:
            setattr(self, name, np.zeros((cap,) + shp, dtype=dt))

    def ensure(self, extra: int) -> None:
        if self.n + extra <= self.cap:
            return
        new_cap = max(int(self.cap * 1.5), self.n + extra + 32)
        for name, dt, shp in self.fields:
            old = getattr(self, name)
            new = np.zeros((new_cap,) + shp, dtype=dt)
            new[: self.n] = old[: self.n]
            setattr(self, name, new)
        self.cap = new_cap

    def add(self, k: int, **values) -> slice:
        """Append ``k`` organisms; unspecified fields are zero.  Returns the slice they occupy."""
        self.ensure(k)
        s = slice(self.n, self.n + k)
        for name, dt, shp in self.fields:
            getattr(self, name)[s] = 0
        for name, v in values.items():
            getattr(self, name)[s] = v
        self.n += k
        return s

    def remove(self, dead: np.ndarray) -> None:
        n = self.n
        keep = ~dead[:n]
        m = int(keep.sum())
        if m == n:
            return
        for name in self.names:
            a = getattr(self, name)
            a[:m] = a[:n][keep]
        self.n = m

    def index_of(self, oid: int) -> int:
        """Index of organism with id ``oid`` or -1."""
        ids = self.id[: self.n]
        i = int(np.searchsorted(ids, oid))
        return i if i < self.n and ids[i] == oid else -1

    def indices_of(self, oids: np.ndarray) -> np.ndarray:
        """Vectorised id -> index (-1 if absent)."""
        ids = self.id[: self.n]
        if self.n == 0:
            return np.full(len(oids), -1, dtype=np.int64)
        i = np.searchsorted(ids, oids)
        i = np.minimum(i, self.n - 1)
        return np.where(ids[i] == oids, i, -1)

    def state_dict(self) -> dict:
        n = self.n
        return {name: getattr(self, name)[:n].copy() for name in self.names}

    def load_state_dict(self, d: dict) -> None:
        n = len(d["id"])
        self.n = 0
        self._alloc(max(self.cap, n + 64))
        for name in self.names:
            if name in d:
                getattr(self, name)[:n] = d[name]
        self.n = n


# ----------------------------------------------------------------------------
class LineageRec:
    __slots__ = ("id", "parent_a", "parent_b", "generation", "birth_time", "species_hist", "children",
                 "death_time", "mut_count", "mut_trait", "mut_delta", "age", "cause", "offspring")

    def __init__(self, oid, pa, pb, gen, t, sid, mut_count=0, mut_trait=-1, mut_delta=0.0):
        self.id = int(oid)
        self.parent_a = int(pa)
        self.parent_b = int(pb)
        self.generation = int(gen)
        self.birth_time = float(t)
        self.species_hist: List[List[float]] = [[float(t), int(sid)]]
        self.children: List[int] = []
        self.death_time: Optional[float] = None
        self.mut_count = int(mut_count)
        self.mut_trait = int(mut_trait)
        self.mut_delta = float(mut_delta)
        self.age = 0.0
        self.cause = -1
        self.offspring = 0

    def to_list(self):
        return [self.id, self.parent_a, self.parent_b, self.generation, self.birth_time, self.species_hist,
                self.children, self.death_time, self.mut_count, self.mut_trait, self.mut_delta, self.age,
                self.cause, self.offspring]

    @staticmethod
    def from_list(v):
        r = LineageRec(v[0], v[1], v[2], v[3], v[4], -1)
        r.species_hist = [list(x) for x in v[5]]
        r.children = list(v[6])
        r.death_time = v[7]
        r.mut_count, r.mut_trait, r.mut_delta = v[8], v[9], v[10]
        r.age, r.cause, r.offspring = v[11], v[12], v[13]
        return r


class Lineage:
    """Ancestry records for family trees, mutation history and species transitions."""

    PRUNE_THRESHOLD = 40000

    def __init__(self):
        self.recs: Dict[int, LineageRec] = {}
        self.protected: set = set()

    def add_founders(self, ids, gens, t, sids):
        for i, g, s in zip(ids.tolist(), gens.tolist(), sids.tolist()):
            self.recs[i] = LineageRec(i, -1, -1, g, t, s)

    def add_births(self, ids, pa, pb, gens, t, sids, mut_count, mut_trait, mut_delta):
        recs = self.recs
        for i, a, b, g, s, mc, mt, md in zip(ids.tolist(), pa.tolist(), pb.tolist(), gens.tolist(), sids.tolist(),
                                              mut_count.tolist(), mut_trait.tolist(), mut_delta.tolist()):
            recs[i] = LineageRec(i, a, b, g, t, s, mc, mt, md)
            for p in (a, b):
                r = recs.get(p)
                if r is not None:
                    r.children.append(i)
                    r.offspring += 1

    def note_death(self, ids, t, ages, causes):
        for i, a, c in zip(ids.tolist(), ages.tolist(), causes.tolist()):
            r = self.recs.get(i)
            if r is not None:
                r.death_time = float(t)
                r.age = float(a)
                r.cause = int(c)

    def note_species(self, ids, sids, t):
        for i, s in zip(np.asarray(ids).tolist(), np.asarray(sids).tolist()):
            r = self.recs.get(i)
            if r is not None and (not r.species_hist or r.species_hist[-1][1] != s):
                r.species_hist.append([float(t), int(s)])

    def ancestors(self, oid: int, depth: int = 12) -> List[List[int]]:
        """Generations of ancestors: [[oid], [parents], [grandparents], ...]."""
        levels: List[List[int]] = [[oid]]
        for _ in range(depth):
            nxt = []
            for i in levels[-1]:
                r = self.recs.get(i)
                if r is None:
                    continue
                for p in (r.parent_a, r.parent_b):
                    if p >= 0 and p in self.recs and p not in nxt:
                        nxt.append(p)
            if not nxt:
                break
            levels.append(nxt)
        return levels

    def lineage_path(self, oid: int, max_len: int = 5000) -> List[int]:
        """Primary-parent chain from the founder down to ``oid``."""
        path = [oid]
        while len(path) < max_len:
            r = self.recs.get(path[-1])
            if r is None or r.parent_a < 0 or r.parent_a not in self.recs:
                break
            path.append(r.parent_a)
        return path[::-1]

    def prune(self, alive_ids: Iterable[int]) -> None:
        """Drop records that are neither alive, ancestors of the living, nor protected."""
        if len(self.recs) < self.PRUNE_THRESHOLD:
            return
        keep = set()
        stack = list(alive_ids) + list(self.protected)
        while stack:
            i = stack.pop()
            if i in keep or i < 0:
                continue
            r = self.recs.get(i)
            if r is None:
                continue
            keep.add(i)
            stack.append(r.parent_a)
            stack.append(r.parent_b)
        self.recs = {i: r for i, r in self.recs.items() if i in keep}

    def state_dict(self) -> dict:
        return {"recs": [r.to_list() for r in self.recs.values()], "protected": sorted(self.protected)}

    def load_state_dict(self, d: dict) -> None:
        self.recs = {}
        for v in d["recs"]:
            r = LineageRec.from_list(v)
            self.recs[r.id] = r
        self.protected = set(d.get("protected", []))
