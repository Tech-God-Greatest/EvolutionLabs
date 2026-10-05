"""Batched feed-forward / recurrent neural networks.

Every organism owns a flat parameter vector (weights then biases, layer by
layer).  ``forward`` evaluates *all* organisms at once with batched matrix
products – this is the "batch neural-network inference" the simulation relies on
for performance.  Recurrent mode (Elman-style) feeds the first hidden layer's
previous activations back in as extra inputs, giving organisms primitive memory.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

# Outputs ----------------------------------------------------------------
OUTPUT_NAMES = ["thrust", "turn", "eat", "attack", "reproduce", "sprint", "gait"]
N_OUT = len(OUTPUT_NAMES)
O_THRUST, O_TURN, O_EAT, O_ATTACK, O_REPRO, O_SPRINT, O_GAIT = range(N_OUT)
# Genome prior for output biases (a weak starting bias, *not* a behaviour rule).
OUTPUT_BIAS_PRIOR = np.array([0.6, 0.0, 1.0, -0.8, 0.8, -1.0, 0.5], dtype=np.float32)


class NetSpec:
    """Shape description of a network.  Immutable after construction."""

    def __init__(self, n_in: int, hidden: Sequence[int], n_out: int = N_OUT, recurrent: bool = False):
        self.n_in = int(n_in)
        self.hidden = tuple(int(h) for h in hidden)
        self.n_out = int(n_out)
        self.recurrent = bool(recurrent and len(self.hidden) > 0)
        first_in = self.n_in + (self.hidden[0] if self.recurrent else 0)
        sizes = [first_in] + list(self.hidden) + [self.n_out]
        self.layers: List[Tuple[int, int]] = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.offsets: List[Tuple[int, int, int]] = []
        off = 0
        for i, o in self.layers:
            self.offsets.append((off, off + i * o, off + i * o + o))
            off += i * o + o
        self.n_params = off
        self.state_size = self.hidden[0] if self.recurrent else 0
        # True where the parameter is a connection weight (maskable); False for biases
        wm = np.zeros(off, dtype=bool)
        for (w0, w1, _b1) in self.offsets:
            wm[w0:w1] = True
        self.weight_positions = wm

    def to_dict(self) -> dict:
        return {"n_in": self.n_in, "hidden": list(self.hidden), "n_out": self.n_out, "recurrent": self.recurrent}

    @staticmethod
    def from_dict(d: dict) -> "NetSpec":
        return NetSpec(d["n_in"], d["hidden"], d.get("n_out", N_OUT), d.get("recurrent", False))

    def same_as(self, other: "NetSpec") -> bool:
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"NetSpec(in={self.n_in}, hidden={self.hidden}, out={self.n_out}, recurrent={self.recurrent})"


def init_params(spec: NetSpec, rng: np.random.Generator, n: int, scale: float = 0.9) -> np.ndarray:
    """Random parameters; weights ~ N(0, 2*scale/sqrt(fan_in)) with weakly biased output priors."""
    P = np.zeros((n, spec.n_params), dtype=np.float32)
    for li, (i, o) in enumerate(spec.layers):
        w0, w1, b1 = spec.offsets[li]
        std = scale * 2.0 / np.sqrt(max(i, 1))
        P[:, w0:w1] = rng.normal(0.0, std, size=(n, i * o)).astype(np.float32)
        P[:, w1:b1] = rng.normal(0.0, 0.1, size=(n, o)).astype(np.float32)
    # output bias prior
    w0, w1, b1 = spec.offsets[-1]
    P[:, w1:b1] = OUTPUT_BIAS_PRIOR[: spec.n_out] + rng.normal(0.0, 0.15, size=(n, spec.n_out)).astype(np.float32)
    return P


def forward(spec: NetSpec, params: np.ndarray, x: np.ndarray,
            h_prev: Optional[np.ndarray] = None, return_acts: bool = False):
    """Batched inference.

    params: (N, P) effective (already masked) parameters, x: (N, n_in).
    Returns ``(outputs (N, n_out), new_state (N, state_size))`` and optionally the
    list of per-layer activations.
    """
    n = x.shape[0]
    a = x.astype(np.float32, copy=False)
    acts = [a] if return_acts else None
    new_state = np.zeros((n, spec.state_size), dtype=np.float32)
    for li, (i, o) in enumerate(spec.layers):
        w0, w1, b1 = spec.offsets[li]
        W = params[:, w0:w1].reshape(n, i, o)
        b = params[:, w1:b1]
        if li == 0 and spec.recurrent:
            hp = h_prev if h_prev is not None else new_state
            a = np.concatenate([a, hp], axis=1)
        z = np.matmul(a[:, None, :], W)[:, 0, :] + b
        a = np.tanh(z)
        if li == 0 and spec.recurrent:
            new_state = a
        if return_acts:
            acts.append(a)
    if return_acts:
        return a, new_state, acts
    return a, new_state


def forward_one(spec: NetSpec, params_row: np.ndarray, x_row: np.ndarray, h_row: Optional[np.ndarray] = None):
    """Convenience single-network evaluation that also returns every layer's activations."""
    h = None if h_row is None else h_row[None, :]
    out, st, acts = forward(spec, params_row[None, :], x_row[None, :], h, return_acts=True)
    return out[0], st[0], [a[0] for a in acts]
