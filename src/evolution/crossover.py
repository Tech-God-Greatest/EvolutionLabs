"""Crossover operators for sexual reproduction."""
from __future__ import annotations

import numpy as np


def crossover(tA, wA, mA, tB, wB, mB, rng: np.random.Generator, mode: str = "single_point"):
    """Combine two parents' genomes.

    The genome is the concatenation ``[traits | weights]``.  In ``single_point``
    mode the child takes parent A's genes up to a random cut point and parent B's
    after it (``AAAABBBBBB``).  ``uniform`` picks each gene from either parent
    with equal probability.  Connection masks travel with their weights.

    All arguments are 2-D arrays with one row per child.  Returns
    ``(traits, weights, masks)`` for the children.
    """
    k, G = tA.shape
    P = wA.shape[1]
    L = G + P
    if mode == "uniform":
        pick_b = rng.random((k, L)) < 0.5
    else:
        cut = rng.integers(1, L, size=k)
        pick_b = np.arange(L)[None, :] >= cut[:, None]
    tb, wb = pick_b[:, :G], pick_b[:, G:]
    t = np.where(tb, tB, tA).astype(np.float32)
    w = np.where(wb, wB, wA).astype(np.float32)
    m = np.where(wb, mB, mA).astype(np.uint8)
    return t, w, m
