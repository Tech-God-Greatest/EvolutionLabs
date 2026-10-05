"""Topology utilities for NEAT-style evolution.

Networks have a fixed *maximum* shape; individual connections can be switched
on/off by a per-genome mask, so topology (which connections exist) evolves along
with weights.  This is NEAT-*style* structural evolution (connection add/remove +
topology-aware compatibility distance) implemented so that it stays compatible
with fast batched inference.
"""
from __future__ import annotations

import numpy as np

from .network import NetSpec


def structural_mutation(weights: np.ndarray, masks: np.ndarray, rate: np.ndarray, spec: NetSpec,
                        rng: np.random.Generator) -> np.ndarray:
    """Toggle connections in place.  Newly enabled connections get a small random weight.

    Returns the number of structural changes per genome.
    """
    k = weights.shape[0]
    pos = spec.weight_positions
    p = np.minimum(0.5, rate)[:, None]
    flip = (rng.random(weights.shape) < p) & pos[None, :]
    newly_on = flip & (masks == 0)
    masks[flip] ^= 1
    if newly_on.any():
        weights[newly_on] = rng.normal(0.0, 0.5, size=int(newly_on.sum())).astype(np.float32)
    return flip.sum(axis=1)


def enabled_connections(masks_row: np.ndarray, spec: NetSpec) -> int:
    return int(np.count_nonzero(masks_row[spec.weight_positions]))


def topology_distance(mask_a: np.ndarray, mask_b: np.ndarray, spec: NetSpec) -> float:
    """Fraction of connection slots whose enabled-state differs (NEAT 'disjoint' analogue)."""
    pos = spec.weight_positions
    return float(np.mean(mask_a[..., pos] != mask_b[..., pos]))


def connection_list(params_row: np.ndarray, mask_row: np.ndarray, spec: NetSpec):
    """Yield (layer, src, dst, weight, enabled) – used by the network visualiser."""
    out = []
    for li, (i, o) in enumerate(spec.layers):
        w0, w1, _ = spec.offsets[li]
        W = params_row[w0:w1].reshape(i, o)
        M = mask_row[w0:w1].reshape(i, o)
        for s in range(i):
            for d in range(o):
                out.append((li, s, d, float(W[s, d]), bool(M[s, d])))
    return out
