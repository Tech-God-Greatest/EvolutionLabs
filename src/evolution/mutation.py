"""Mutation operators (batched, in-place on child arrays)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..neural.network import NetSpec
from ..neural.neuroevolution import structural_mutation
from .genome import G_MUT, TRAIT_LO, TRAIT_HI, TRAIT_RANGE, clamp_traits

TRAIT_SIGMA = 0.06          # std-dev of a trait mutation as a fraction of its range
WEIGHT_RESET_PROB = 0.02    # fraction of mutated weights that are re-drawn rather than nudged


def mutate(traits: np.ndarray, weights: np.ndarray, masks: np.ndarray, rate: np.ndarray, strength: float,
           rng: np.random.Generator, spec: NetSpec, structural: float = 0.0,
           evolvable_mutability: bool = False) -> np.ndarray:
    """Mutate children in place.

    Parameters
    ----------
    traits/weights/masks : (k,G) (k,P) (k,P) child genome arrays
    rate : (k,) effective per-gene mutation probability for each child
    strength : std-dev of weight perturbations
    structural : connection add/remove rate as a multiple of ``rate`` (0 disables)

    Returns the number of altered genes per child (used for lineage 'mutation history').
    """
    k = traits.shape[0]
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    r = np.clip(rate, 0.0, 1.0).astype(np.float32)[:, None]
    # --- neural weights & biases ---
    hit = rng.random(weights.shape, dtype=np.float32) < r
    nudge = rng.normal(0.0, strength, size=weights.shape).astype(np.float32)
    reset = rng.random(weights.shape, dtype=np.float32) < WEIGHT_RESET_PROB
    fresh = rng.normal(0.0, 0.8, size=weights.shape).astype(np.float32)
    delta = np.where(reset, fresh - weights, nudge)
    weights += np.where(hit, delta, 0.0).astype(np.float32)
    changed = hit.sum(axis=1)
    # --- traits ---
    thit = rng.random(traits.shape, dtype=np.float32) < np.minimum(1.0, r * 4.0)
    if not evolvable_mutability:
        thit[:, G_MUT] = False
    tdelta = rng.normal(0.0, TRAIT_SIGMA, size=traits.shape).astype(np.float32) * TRAIT_RANGE
    traits += np.where(thit, tdelta, 0.0).astype(np.float32)
    clamp_traits(traits)
    changed = changed + thit.sum(axis=1)
    # --- topology ---
    if structural > 0.0:
        changed = changed + structural_mutation(weights, masks, rate * structural, spec, rng)
    np.clip(weights, -8.0, 8.0, out=weights)
    return changed


def burst_mutation(traits: np.ndarray, weights: np.ndarray, masks: np.ndarray, rng: np.random.Generator,
                   spec: NetSpec, strength: float = 0.6, rate: float = 0.3) -> None:
    """Large-effect mutation used by the god-mode 'create mutation' tool."""
    mutate(traits, weights, masks, np.full(traits.shape[0], rate, dtype=np.float32), strength, rng, spec,
           structural=0.0, evolvable_mutability=False)
