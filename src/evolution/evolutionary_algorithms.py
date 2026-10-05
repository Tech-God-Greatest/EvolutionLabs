"""Pluggable evolutionary strategies.

Two flavours of every algorithm are provided so they can be compared:

* **Continuous** (``Strategy``): hooks used by the open-ended ecosystem.  The
  ecosystem itself performs natural selection; a strategy only decides *how
  reproduction is organised* (mate choice, crossover vs cloning, mutation scale,
  speciation-based crowding penalties, topology mutation).
* **Generational** (``GenerationalAlgorithm``): classic "evaluate everybody, then
  breed the next generation" loops used by the maze experiment and the
  algorithm-comparison experiments.

Strategies: genetic algorithm (GA), NEAT-style, mutation + selection.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..neural.network import NetSpec
from .crossover import crossover
from .mutation import mutate
from .selection import tournament, truncation
from .species import matrix_distance, pair_distance


# ----------------------------------------------------------------------------
# Continuous (ecosystem) strategies
# ----------------------------------------------------------------------------
class Strategy:
    name = "mutation_selection"
    label = "Mutation + Selection"
    description = "Asexual/mixed reproduction; mutation supplies variation, resource competition selects."
    mutation_scale = 1.0
    mate_choice = "nearest"      # 'nearest' | 'fittest'
    force_sexual = False

    def mutation_rate(self, cfg) -> float:
        return cfg.mutation_rate * self.mutation_scale

    def structural_rate(self, cfg) -> float:
        return cfg.structural_mutation

    def reproduction_mode(self, cfg) -> str:
        return "sexual" if self.force_sexual else cfg.reproduction_mode

    def repro_penalty(self, eco, species_ids: np.ndarray) -> Optional[np.ndarray]:
        """Extra energy-fraction required to reproduce (crowding / fitness sharing)."""
        return None


class MutationSelection(Strategy):
    pass


class GeneticAlgorithmStrategy(Strategy):
    name = "genetic_algorithm"
    label = "Genetic Algorithm"
    description = "Sexual reproduction with crossover; mates are chosen by viability (energy x age)."
    mutation_scale = 0.6
    mate_choice = "fittest"
    force_sexual = True

    def reproduction_mode(self, cfg) -> str:
        return "mixed"   # sexual preferred, asexual fallback prevents extinction from mate scarcity


class NeatStyleStrategy(Strategy):
    name = "neat"
    label = "NEAT-style"
    description = ("Topology (connections) evolves with weights; speciation-based fitness sharing penalises "
                   "reproduction in crowded species, protecting innovation.")
    mutation_scale = 0.7

    def structural_rate(self, cfg) -> float:
        return max(cfg.structural_mutation, 0.6)

    def reproduction_mode(self, cfg) -> str:
        return "mixed"

    def repro_penalty(self, eco, species_ids):
        n = len(species_ids)
        if n == 0:
            return None
        uniq, inv, counts = np.unique(species_ids, return_inverse=True, return_counts=True)
        share = counts[inv] / n
        return 0.12 * share        # explicit fitness sharing: big species must save more energy


STRATEGIES = {
    "mutation_selection": MutationSelection,
    "genetic_algorithm": GeneticAlgorithmStrategy,
    "neat": NeatStyleStrategy,
}


def make_strategy(name: str) -> Strategy:
    return STRATEGIES[name]()


# ----------------------------------------------------------------------------
# Generational algorithms
# ----------------------------------------------------------------------------
def _speciate(T, W, threshold: float) -> np.ndarray:
    n = len(T)
    labels = np.full(n, -1, dtype=np.int64)
    un = np.arange(n)
    s = 0
    while un.size:
        i = un[0]
        d = pair_distance(T[un], W[un], T[i], W[i])
        take = un[d <= threshold]
        labels[take] = s
        un = un[d > threshold]
        s += 1
    return labels


class GenerationalAlgorithm:
    name = "base"
    label = "base"

    def __init__(self, spec: NetSpec, cfg, rng: np.random.Generator):
        self.spec = spec
        self.cfg = cfg
        self.rng = rng

    def _mutate(self, T, W, M, rate_mult=1.0, structural=0.0):
        k = len(T)
        rate = np.full(k, self.cfg.mutation_rate * rate_mult, dtype=np.float32)
        mutate(T, W, M, rate, self.cfg.mutation_strength, self.rng, self.spec, structural=structural)

    def next_generation(self, T, W, M, fitness) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class MutationSelectionGen(GenerationalAlgorithm):
    """(mu + lambda): the best fraction survives unchanged and seeds mutated offspring."""
    name = "mutation_selection"
    label = "Mutation + Selection"

    def next_generation(self, T, W, M, fitness):
        n = len(T)
        top = truncation(fitness, 0.2)
        elite = top[: max(2, n // 50)]
        parents = np.concatenate([elite, top[np.arange(n - len(elite)) % len(top)]])
        nT, nW, nM = T[parents].copy(), W[parents].copy(), M[parents].copy()
        self._mutate(nT[len(elite):], nW[len(elite):], nM[len(elite):], 1.0, self.cfg.structural_mutation)
        return nT, nW, nM, parents


class GeneticAlgorithmGen(GenerationalAlgorithm):
    name = "genetic_algorithm"
    label = "Genetic Algorithm"

    def next_generation(self, T, W, M, fitness):
        n = len(T)
        n_elite = max(2, n // 50)
        order = np.argsort(-fitness, kind="stable")
        elite = order[:n_elite]
        m = n - n_elite
        pa = tournament(self.rng, fitness, m, 4)
        pb = tournament(self.rng, fitness, m, 4)
        cT, cW, cM = crossover(T[pa], W[pa], M[pa], T[pb], W[pb], M[pb], self.rng, self.cfg.crossover_mode)
        self._mutate(cT, cW, cM, 0.6, self.cfg.structural_mutation)
        nT = np.concatenate([T[elite], cT])
        nW = np.concatenate([W[elite], cW])
        nM = np.concatenate([M[elite], cM])
        return nT, nW, nM, np.concatenate([elite, pa])


class NeatStyleGen(GenerationalAlgorithm):
    """Speciation + explicit fitness sharing + topology mutation."""
    name = "neat"
    label = "NEAT-style"

    def next_generation(self, T, W, M, fitness):
        n = len(T)
        eff = W * M
        labels = _speciate(T, eff, self.cfg.species_threshold)
        nsp = labels.max() + 1
        sizes = np.bincount(labels, minlength=nsp)
        shifted = fitness - fitness.min() + 1e-6
        adj = shifted / sizes[labels]
        sp_score = np.bincount(labels, weights=adj, minlength=nsp)
        alloc = np.floor(sp_score / sp_score.sum() * n).astype(int)
        while alloc.sum() < n:                       # distribute remainder to best species
            alloc[int(np.argmax(sp_score - alloc * sp_score.mean() / max(1, n / nsp)))] += 1
        parents_a, parents_b, is_elite = [], [], []
        for s in range(nsp):
            if alloc[s] == 0:
                continue
            members = np.flatnonzero(labels == s)
            f = fitness[members]
            best = members[int(np.argmax(f))]
            parents_a.append(best); parents_b.append(best); is_elite.append(True)
            for _ in range(alloc[s] - 1):
                a = members[tournament(self.rng, f, 1, 3)[0]]
                b = members[tournament(self.rng, f, 1, 3)[0]]
                parents_a.append(a); parents_b.append(b); is_elite.append(False)
        pa = np.array(parents_a); pb = np.array(parents_b); elite = np.array(is_elite)
        cT, cW, cM = crossover(T[pa], W[pa], M[pa], T[pb], W[pb], M[pb], self.rng, "uniform")
        ne = ~elite
        if ne.any():
            # boolean-mask indexing returns copies, so mutate then write back explicitly
            tt, ww, mm = cT[ne].copy(), cW[ne].copy(), cM[ne].copy()
            self._mutate(tt, ww, mm, 0.8, max(self.cfg.structural_mutation, 0.6))
            cT[ne], cW[ne], cM[ne] = tt, ww, mm
        return cT, cW, cM, pa


GENERATIONAL = {"mutation_selection": MutationSelectionGen, "genetic_algorithm": GeneticAlgorithmGen, "neat": NeatStyleGen}


def make_generational(name: str, spec: NetSpec, cfg, rng) -> GenerationalAlgorithm:
    return GENERATIONAL[name](spec, cfg, rng)
