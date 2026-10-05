"""Automatic species detection.

Species are clusters of genomes: an organism belongs to the species whose
representative genome is within ``species_threshold`` (genetic distance).  If no
representative is close enough a **new species is founded**.  Representatives
drift slowly toward the centroid of their members so a population that evolves
*together* stays one species, while a subpopulation that diverges eventually
splits away.  The species a new one split from is recorded (parent species) so
a phylogenetic tree can be drawn.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.util import hsv_rgb, species_name
from .genome import TRAIT_RANGE, G_SPEED, G_SIZE, G_VIS, G_DIET, G_ATTACK

TRAIT_NORM = 0.25   # typical mean normalised trait distance of unrelated genomes
WEIGHT_NORM = 0.45  # typical rms weight distance of unrelated genomes
W_TRAIT, W_WEIGHT = 0.4, 0.6
MAX_ACTIVE_SPECIES = 160
REP_DRIFT = 0.2


def pair_distance(tA, wA, tB, wB) -> np.ndarray:
    """Genetic distance between aligned genome rows (broadcasting allowed)."""
    dt = np.mean(np.abs(tA - tB) / TRAIT_RANGE, axis=-1) / TRAIT_NORM
    dw = np.sqrt(np.mean((wA - wB) ** 2, axis=-1)) / WEIGHT_NORM
    return W_TRAIT * dt + W_WEIGHT * dw


def matrix_distance(T: np.ndarray, W: np.ndarray, repT: np.ndarray, repW: np.ndarray) -> np.ndarray:
    """(n, S) distance matrix between n genomes and S representatives (matmul-accelerated)."""
    n, P = W.shape
    dt = np.zeros((n, repT.shape[0]), dtype=np.float32)
    for s in range(repT.shape[0]):
        dt[:, s] = np.mean(np.abs(T - repT[s]) / TRAIT_RANGE, axis=1)
    a2 = np.einsum("ij,ij->i", W, W)
    b2 = np.einsum("ij,ij->i", repW, repW)
    sq = a2[:, None] + b2[None, :] - 2.0 * (W @ repW.T)
    dw = np.sqrt(np.maximum(sq, 0.0) / P)
    return (W_TRAIT * dt / TRAIT_NORM + W_WEIGHT * dw / WEIGHT_NORM).astype(np.float32)


class Species:
    def __init__(self, sid: int, name: str, color: Tuple[int, int, int], hue: float, parent: Optional[int],
                 founded_time: float, founded_gen: int, rep_t: np.ndarray, rep_w: np.ndarray):
        self.id = sid
        self.name = name
        self.color = color
        self.hue = hue
        self.parent = parent
        self.founded_time = founded_time
        self.founded_gen = founded_gen
        self.rep_t = rep_t.astype(np.float32).copy()
        self.rep_w = rep_w.astype(np.float32).copy()
        self.extinct_time: Optional[float] = None
        self.population = 0
        self.peak_population = 0
        self.life_sum = 0.0
        self.life_count = 0
        self.total_members = 0

    @property
    def alive(self) -> bool:
        return self.extinct_time is None

    @property
    def avg_lifespan(self) -> float:
        return self.life_sum / self.life_count if self.life_count else 0.0

    def info(self) -> dict:
        return {"id": self.id, "name": self.name, "color": list(self.color), "hue": self.hue, "parent": self.parent,
                "founded_time": self.founded_time, "founded_gen": self.founded_gen,
                "extinct_time": self.extinct_time, "population": self.population,
                "peak_population": self.peak_population, "life_sum": self.life_sum,
                "life_count": self.life_count}


class SpeciesManager:
    def __init__(self, seed: int):
        self.seed = seed
        self.species: Dict[int, Species] = {}
        self.next_id = 1
        self.names_used: set = set()
        self.new_this_epoch: List[int] = []
        self.extinct_this_epoch: List[int] = []
        self.total_created = 0
        self.total_extinct = 0

    # ------------------------------------------------------------------
    def active_ids(self) -> List[int]:
        return [sid for sid, sp in self.species.items() if sp.alive]

    def _create(self, eco, t, w, parent: Optional[int], gen: int) -> Species:
        sid = self.next_id
        self.next_id += 1
        name = species_name(self.seed, sid, self.names_used)
        self.names_used.add(name)
        if parent is not None and parent in self.species:
            ph = self.species[parent].hue
            hue = (ph + (0.07 + 0.06 * ((sid * 0.618) % 1.0)) * (1 if sid % 2 else -1)) % 1.0
        else:
            hue = (sid * 0.61803398875) % 1.0
        sat = 0.55 + 0.35 * ((sid * 0.37) % 1.0)
        val = 0.85 + 0.15 * ((sid * 0.53) % 1.0)
        sp = Species(sid, name, hsv_rgb(hue, sat, val), hue, parent, eco.time, gen, t, w)
        self.species[sid] = sp
        self.new_this_epoch.append(sid)
        self.total_created += 1
        return sp

    # ------------------------------------------------------------------
    def update(self, eco) -> None:
        s = eco.org
        n = s.n
        self.new_this_epoch = []
        self.extinct_this_epoch = []
        thr = float(eco.cfg.species_threshold)
        if n == 0:
            for sp in self.species.values():
                if sp.alive:
                    self._mark_extinct(sp, eco.time)
            return
        T = s.traits[:n]
        W = s.w_eff[:n]
        ids = self.active_ids()
        assigned = np.full(n, -1, dtype=np.int32)
        D = None
        if ids:
            repT = np.stack([self.species[i].rep_t for i in ids])
            repW = np.stack([self.species[i].rep_w for i in ids])
            D = matrix_distance(T, W, repT, repW)
            j = np.argmin(D, axis=1)
            dmin = D[np.arange(n), j]
            id_arr = np.array(ids, dtype=np.int32)
            ok = dmin <= thr
            assigned[ok] = id_arr[j[ok]]
            if len(ids) >= MAX_ACTIVE_SPECIES:  # runaway guard – absorb stragglers
                assigned[~ok] = id_arr[j[~ok]]
        un = np.flatnonzero(assigned < 0)
        while un.size:
            i = int(un[0])
            parent = None
            if D is not None:
                parent = ids[int(np.argmin(D[i]))]
            sp = self._create(eco, T[i], W[i], parent, int(s.generation[i]))
            d = pair_distance(T[un], W[un], T[i], W[i])
            take = un[d <= thr]
            assigned[take] = sp.id
            un = un[d > thr]
            if len(self.active_ids()) >= MAX_ACTIVE_SPECIES and un.size:
                assigned[un] = sp.id
                break
        old = s.species[:n]
        changed = np.flatnonzero((old != assigned) & (old >= 0))
        if changed.size:
            eco.lineage.note_species(s.id[changed], assigned[changed], eco.time)
        s.species[:n] = assigned
        # population + representative drift
        order = np.argsort(assigned, kind="stable")
        sa = assigned[order]
        uniq, starts, counts = np.unique(sa, return_index=True, return_counts=True)
        sumT = np.add.reduceat(T[order], starts, axis=0)
        sumW = np.add.reduceat(W[order], starts, axis=0)
        present = set()
        for k, sid in enumerate(uniq):
            sp = self.species[int(sid)]
            present.add(int(sid))
            sp.population = int(counts[k])
            sp.peak_population = max(sp.peak_population, sp.population)
            sp.total_members = max(sp.total_members, sp.population)
            sp.rep_t += REP_DRIFT * (sumT[k] / counts[k] - sp.rep_t)
            sp.rep_w += REP_DRIFT * (sumW[k] / counts[k] - sp.rep_w)
        for sid in self.active_ids():
            if sid not in present:
                self._mark_extinct(self.species[sid], eco.time)

    def _mark_extinct(self, sp: Species, t: float) -> None:
        sp.population = 0
        sp.extinct_time = t
        self.extinct_this_epoch.append(sp.id)
        self.total_extinct += 1

    def record_death(self, species_ids: np.ndarray, ages: np.ndarray) -> None:
        if species_ids.size == 0:
            return
        for sid in np.unique(species_ids):
            sp = self.species.get(int(sid))
            if sp is not None:
                m = species_ids == sid
                sp.life_sum += float(ages[m].sum())
                sp.life_count += int(m.sum())

    # ------------------------------------------------------------------
    def tree_edges(self) -> List[Tuple[Optional[int], int]]:
        return [(sp.parent, sp.id) for sp in self.species.values()]

    def top(self, k: int = 10) -> List[Species]:
        return sorted([sp for sp in self.species.values() if sp.alive], key=lambda s: -s.population)[:k]

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "next_id": self.next_id, "names_used": sorted(self.names_used),
            "total_created": self.total_created, "total_extinct": self.total_extinct,
            "species": [dict(sp.info(), rep_t=sp.rep_t, rep_w=sp.rep_w) for sp in self.species.values()],
        }

    def load_state_dict(self, d: dict) -> None:
        self.next_id = int(d["next_id"])
        self.names_used = set(d["names_used"])
        self.total_created = int(d["total_created"])
        self.total_extinct = int(d["total_extinct"])
        self.species = {}
        for e in d["species"]:
            sp = Species(int(e["id"]), e["name"], tuple(e["color"]), float(e["hue"]),
                         None if e["parent"] is None else int(e["parent"]),
                         float(e["founded_time"]), int(e["founded_gen"]),
                         np.asarray(e["rep_t"], dtype=np.float32), np.asarray(e["rep_w"], dtype=np.float32))
            sp.extinct_time = e["extinct_time"]
            sp.population = int(e["population"])
            sp.peak_population = int(e["peak_population"])
            sp.life_sum = float(e["life_sum"])
            sp.life_count = int(e["life_count"])
            self.species[sp.id] = sp
