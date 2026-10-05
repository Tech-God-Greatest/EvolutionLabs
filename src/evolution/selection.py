"""Selection operators.

In the continuous ecosystem there is *no* selection operator – survival and
reproduction under resource competition do the selecting.  These functions are
used by the *generational* modes (maze experiment, algorithm comparison) and by
mate choice in the genetic-algorithm strategy.
"""
from __future__ import annotations

import numpy as np


def tournament(rng: np.random.Generator, scores: np.ndarray, n: int, size: int = 3) -> np.ndarray:
    """Return ``n`` winners (indices) of independent tournaments of ``size`` contestants."""
    m = len(scores)
    cand = rng.integers(0, m, size=(n, size))
    best = np.argmax(scores[cand], axis=1)
    return cand[np.arange(n), best]


def roulette(rng: np.random.Generator, scores: np.ndarray, n: int) -> np.ndarray:
    s = scores - scores.min() + 1e-9
    p = s / s.sum()
    return rng.choice(len(scores), size=n, p=p)


def truncation(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Indices of the top ``fraction`` of the population (best first)."""
    k = max(1, int(round(len(scores) * fraction)))
    return np.argsort(-scores, kind="stable")[:k]


def rank_select(rng: np.random.Generator, scores: np.ndarray, n: int, pressure: float = 1.7) -> np.ndarray:
    m = len(scores)
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(m)
    ranks[order] = np.arange(m)
    p = (2 - pressure) / m + 2 * ranks * (pressure - 1) / (m * (m - 1) + 1e-9)
    p = p / p.sum()
    return rng.choice(m, size=n, p=p)
