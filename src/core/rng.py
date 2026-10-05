"""Centralised, seedable random-number system.

Every stochastic decision in the *simulation* draws from a named stream owned by
:class:`RngHub`.  Streams are statistically independent (SeedSequence.spawn) so
that, e.g., drawing extra numbers for an analysis metric can never alter the
simulated trajectory.  The full state of the hub is serialisable, which is what
makes save/load and keyframe replay bit-for-bit deterministic.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np

STREAMS = ("world", "spawn", "mutation", "noise", "events", "species", "combat", "analysis")


class RngHub:
    def __init__(self, seed: int):
        self.seed = int(seed)
        ss = np.random.SeedSequence(self.seed)
        kids = ss.spawn(len(STREAMS))
        self._g: Dict[str, np.random.Generator] = {
            name: np.random.Generator(np.random.PCG64(k)) for name, k in zip(STREAMS, kids)
        }

    # attribute-style access: hub.world, hub.mutation, ...
    def __getattr__(self, name: str) -> np.random.Generator:
        g = self.__dict__.get("_g")
        if g is not None and name in g:
            return g[name]
        raise AttributeError(name)

    def get_state(self) -> dict:
        return {"seed": self.seed, "streams": {n: g.bit_generator.state for n, g in self._g.items()}}

    def set_state(self, state: dict) -> None:
        self.seed = int(state["seed"])
        for n, st in state["streams"].items():
            if n in self._g:
                self._g[n].bit_generator.state = st

    @staticmethod
    def random_seed() -> int:
        """Fresh seed from OS entropy (used only when the user did not supply one)."""
        return int.from_bytes(os.urandom(4), "little") % 100_000_000


def derive_rng(seed: int, *keys: int) -> np.random.Generator:
    """Independent generator derived from (seed, keys) – used for names/colours."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), *[int(k) for k in keys]])))
