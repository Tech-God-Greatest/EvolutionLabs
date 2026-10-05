"""Small shared helpers (angles, ramps, colours, procedural names)."""
from __future__ import annotations

import colorsys
import math
from typing import Set, Tuple

import numpy as np

from .rng import derive_rng

TAU = 2.0 * math.pi


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (a + np.pi) % TAU - np.pi


def ramp(x, lo: float, hi: float):
    """Linear 0..1 ramp between lo and hi."""
    return np.clip((np.asarray(x, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0)


def hsv_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return int(r * 255), int(g * 255), int(b * 255)


_C = ["v", "k", "n", "z", "r", "th", "kr", "br", "l", "m", "x", "d", "g", "h", "s", "t", "f", "q", "y", "w", "j", "p", "sy", "ny"]
_V = ["a", "e", "i", "o", "u", "y", "ae", "ia", "ei"]
_E = ["th", "x", "n", "s", "k", "a", "ra", "on", "ith", "ex", "is", "or", "un", "", "", "el", "ar", "ix"]


def species_name(seed: int, species_id: int, used: Set[str]) -> str:
    """Procedural, deterministic species name (Velora, Kryth, Nuvex, ...)."""
    rng = derive_rng(seed, 7919, species_id)
    for _ in range(40):
        syl = int(rng.integers(1, 3))
        parts = []
        for i in range(syl + 1):
            c = _C[int(rng.integers(len(_C)))]
            v = _V[int(rng.integers(len(_V)))]
            parts.append(c + v)
        name = "".join(parts[:syl + 1]) + _E[int(rng.integers(len(_E)))]
        name = name[:9].capitalize()
        if 4 <= len(name) and name not in used:
            return name
    base = name
    k = 2
    while f"{base}{k}" in used:
        k += 1
    return f"{base}{k}"


def fmt_int(n) -> str:
    return f"{int(n):,}"


def fmt_time(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 120:
        return f"{seconds:.1f} s"
    m, s = divmod(int(seconds), 60)
    if m < 120:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"
