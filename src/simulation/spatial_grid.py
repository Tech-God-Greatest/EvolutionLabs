"""Uniform-grid spatial partitioning.

``SpatialGrid.build`` sorts points by cell (O(n log n) once), after which a
neighbourhood query for *all* organisms at once is a handful of contiguous slice
ranges – no O(n^2) pair scan.  Query results are returned in a flat "pairs" layout
grouped by source so per-source reductions (nearest neighbour, counts, sums) are
simple vectorised segment operations.
"""
from __future__ import annotations

import numpy as np


class Pairs:
    """Candidate (src -> dst) pairs, grouped by src."""

    __slots__ = ("src", "dst", "counts", "seg_starts", "m")

    def __init__(self, src, dst, counts):
        self.src = src
        self.dst = dst
        self.counts = counts
        self.m = len(counts)
        self.seg_starts = np.cumsum(counts) - counts

    def __len__(self):
        return len(self.src)

    def segment_argmin(self, values: np.ndarray):
        """Per-source minimum of ``values`` (use +inf for invalid pairs).

        Returns ``(pair_index (m,) or -1, min_value (m,))``.
        """
        m = self.m
        out_idx = np.full(m, -1, dtype=np.int64)
        out_min = np.full(m, np.inf, dtype=np.float32)
        if len(self.src) == 0:
            return out_idx, out_min
        nz = self.counts > 0
        mins = np.minimum.reduceat(values, self.seg_starts[nz])
        out_min[nz] = mins
        per_pair = np.repeat(mins, self.counts[nz])
        hit = np.flatnonzero((values == per_pair) & np.isfinite(values))
        if hit.size:
            first_src, first = np.unique(self.src[hit], return_index=True)
            out_idx[first_src] = hit[first]
        return out_idx, out_min

    def segment_sum(self, weights: np.ndarray) -> np.ndarray:
        return np.bincount(self.src, weights=weights, minlength=self.m)


class SpatialGrid:
    def __init__(self, width: float, height: float, cell: float):
        self.width = float(width)
        self.height = float(height)
        self.cell = float(max(cell, 1.0))
        self.gw = max(1, int(np.ceil(self.width / self.cell)))
        self.gh = max(1, int(np.ceil(self.height / self.cell)))
        self.n = 0
        self.order = np.zeros(0, dtype=np.int64)
        self.starts = np.zeros(self.gw * self.gh + 1, dtype=np.int64)

    def set_cell(self, cell: float) -> None:
        if abs(cell - self.cell) > 1e-6:
            self.__init__(self.width, self.height, cell)

    def _cells(self, pos: np.ndarray):
        cx = np.clip((pos[:, 0] / self.cell).astype(np.int64), 0, self.gw - 1)
        cy = np.clip((pos[:, 1] / self.cell).astype(np.int64), 0, self.gh - 1)
        return cx, cy

    def build(self, pos: np.ndarray) -> None:
        self.n = len(pos)
        if self.n == 0:
            self.order = np.zeros(0, dtype=np.int64)
            self.starts = np.zeros(self.gw * self.gh + 1, dtype=np.int64)
            return
        cx, cy = self._cells(pos)
        cid = cy * self.gw + cx
        self.order = np.argsort(cid, kind="stable")
        counts = np.bincount(cid, minlength=self.gw * self.gh)
        self.starts = np.zeros(self.gw * self.gh + 1, dtype=np.int64)
        np.cumsum(counts, out=self.starts[1:])

    def pairs(self, qpos: np.ndarray, reach_cells: int) -> Pairs:
        """All indexed points within ``reach_cells`` cells (Chebyshev) of each query point."""
        m = len(qpos)
        R = int(max(0, reach_cells))
        if m == 0 or self.n == 0:
            return Pairs(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(m, np.int64))
        qcx, qcy = self._cells(qpos)
        xlo = np.maximum(qcx - R, 0)
        xhi = np.minimum(qcx + R, self.gw - 1)
        S, C = [], []
        for dy in range(-R, R + 1):
            row = qcy + dy
            valid = (row >= 0) & (row < self.gh)
            rr = np.clip(row, 0, self.gh - 1)
            s = self.starts[rr * self.gw + xlo]
            e = self.starts[rr * self.gw + xhi + 1]
            S.append(s)
            C.append(np.where(valid, e - s, 0))
        S = np.stack(S, axis=1)
        C = np.stack(C, axis=1)
        Sf, Cf = S.ravel(), C.ravel()
        total = int(Cf.sum())
        counts = C.sum(axis=1)
        if total == 0:
            return Pairs(np.zeros(0, np.int64), np.zeros(0, np.int64), counts)
        seg_off = np.cumsum(Cf) - Cf
        within = np.arange(total) - np.repeat(seg_off, Cf)
        dst = self.order[np.repeat(Sf, Cf) + within]
        src = np.repeat(np.repeat(np.arange(m), 2 * R + 1), Cf)
        return Pairs(src, dst, counts)

    # -- convenience (UI / tests) --------------------------------------
    def query_radius(self, pos: np.ndarray, point, radius: float) -> np.ndarray:
        """Indices of indexed points within ``radius`` of ``point``."""
        R = int(np.ceil(radius / self.cell))
        p = self.pairs(np.asarray([point], dtype=np.float32), R)
        if len(p) == 0:
            return np.zeros(0, dtype=np.int64)
        d = np.hypot(pos[p.dst, 0] - point[0], pos[p.dst, 1] - point[1])
        return np.sort(p.dst[d <= radius])
