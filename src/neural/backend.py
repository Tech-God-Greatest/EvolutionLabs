"""Inference backends.  NumPy is the default and always available; PyTorch is optional.

The torch backend is auto-detected.  If torch (or a requested GPU) is missing we
fall back to NumPy with a warning instead of crashing (graceful degradation).
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .network import NetSpec, forward as np_forward


class NumpyBackend:
    name = "numpy"

    def forward(self, spec: NetSpec, params, x, h_prev):
        return np_forward(spec, params, x, h_prev)


class TorchBackend:
    name = "torch"

    def __init__(self):
        import torch  # noqa: optional dependency
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, spec: NetSpec, params, x, h_prev):
        t = self.torch
        with t.no_grad():
            P = t.from_numpy(np.ascontiguousarray(params)).to(self.device)
            a = t.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)
            n = a.shape[0]
            state = t.zeros((n, spec.state_size), device=self.device)
            for li, (i, o) in enumerate(spec.layers):
                w0, w1, b1 = spec.offsets[li]
                W = P[:, w0:w1].reshape(n, i, o)
                b = P[:, w1:b1]
                if li == 0 and spec.recurrent:
                    hp = t.from_numpy(np.ascontiguousarray(h_prev)).to(self.device) if h_prev is not None else state
                    a = t.cat([a, hp], dim=1)
                a = t.tanh(t.bmm(a.unsqueeze(1), W).squeeze(1) + b)
                if li == 0 and spec.recurrent:
                    state = a
            return a.cpu().numpy(), state.cpu().numpy()


def get_backend(name: str = "numpy"):
    if name == "torch":
        try:
            return TorchBackend()
        except Exception as e:  # ImportError, CUDA errors, ...
            warnings.warn(f"torch backend unavailable ({e}); falling back to NumPy")
    return NumpyBackend()
