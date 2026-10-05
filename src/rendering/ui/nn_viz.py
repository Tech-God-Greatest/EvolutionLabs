"""Interactive Neural Network Architecture and Activation Visualizer (§23)."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np
import pygame

from ...neural.network import OUTPUT_NAMES
from .theme import (
    BORDER_DEFAULT,
    BORDER_GLOW,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    Theme,
)


class NetworkVisualizer:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.pan_x = 0
        self.pan_y = 0
        self.zoom = 1.0

    def draw(
        self,
        surface: pygame.Surface,
        eco,
        idx: int,
        theme: Theme,
    ) -> None:
        s = eco.org
        if idx < 0 or idx >= s.n:
            return

        spec = eco.spec
        # Get live activations from single-organism forward pass
        try:
            out, st, acts = eco.network_trace(idx)
        except Exception:
            return

        w_row = s.w[idx]
        m_row = s.mask[idx]

        # Node coordinates layout
        # Layers: [inputs, hidden_0, ..., hidden_k, outputs]
        layer_sizes = [spec.n_in] + list(spec.hidden) + [spec.n_out]
        n_layers = len(layer_sizes)

        pad_x = 35
        pad_y = 30
        avail_w = self.rect.w - pad_x * 2
        avail_h = self.rect.h - pad_y * 2

        col_spacing = avail_w / max(1, n_layers - 1)
        node_coords: List[List[Tuple[float, float]]] = []

        for li, count in enumerate(layer_sizes):
            cx = self.rect.x + pad_x + li * col_spacing
            row_spacing = avail_h / max(1, count - 1) if count > 1 else 0
            start_y = self.rect.y + pad_y + (avail_h - (count - 1) * row_spacing) / 2 if count > 1 else self.rect.y + self.rect.h / 2
            layer_pts = []
            for ni in range(count):
                cy = start_y + ni * row_spacing
                layer_pts.append((cx, cy))
            node_coords.append(layer_pts)

        # Draw connections
        for li in range(len(spec.layers)):
            w0, w1, _ = spec.offsets[li]
            inp_dim, out_dim = spec.layers[li]
            W = w_row[w0:w1].reshape(inp_dim, out_dim)
            M = m_row[w0:w1].reshape(inp_dim, out_dim)

            # Cap drawn connections for visual clarity if too dense
            for src_i in range(inp_dim):
                for dst_i in range(out_dim):
                    if not M[src_i, dst_i]:
                        continue  # disabled synapse
                    weight = float(W[src_i, dst_i])
                    if abs(weight) < 0.05:
                        continue  # negligible

                    p1 = node_coords[li][src_i]
                    p2 = node_coords[li + 1][dst_i]

                    # Color: Green/cyan for positive weight, Crimson/orange for negative weight
                    col = (56, 189, 248) if weight > 0 else (244, 63, 94)
                    alpha = min(220, max(30, int(abs(weight) * 90)))
                    width = max(1, min(4, int(abs(weight) * 2.0)))

                    pygame.draw.line(surface, col, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), width)

        # Draw nodes & activations
        for li, pts in enumerate(node_coords):
            act_vec = acts[li] if li < len(acts) else np.zeros(len(pts))
            for ni, (nx, ny) in enumerate(pts):
                val = float(act_vec[ni]) if ni < len(act_vec) else 0.0
                # Glow brightness based on activation [-1, 1]
                norm_a = max(0.0, min(1.0, (val + 1.0) / 2.0))
                brightness = int(50 + norm_a * 205)
                br = min(255, max(0, int(brightness * 1.1)))
                bg = min(255, max(0, int(brightness * 0.9)))
                bb = min(255, max(0, int(brightness)))
                node_col = (bb, bg, br) if val > 0 else (bg, bb, bb)
                pygame.draw.circle(surface, (20, 28, 42), (int(nx), int(ny)), 5)
                pygame.draw.circle(surface, (0, 220, 255), (int(nx), int(ny)), 5, 1)
                pygame.draw.circle(surface, node_col, (int(nx), int(ny)), 3)

        # Labels for Output layer
        out_pts = node_coords[-1]
        for oi, (ox, oy) in enumerate(out_pts):
            if oi < len(OUTPUT_NAMES):
                lbl = theme.font_xs.render(OUTPUT_NAMES[oi], True, TEXT_MUTED)
                surface.blit(lbl, (int(ox) + 8, int(oy) - lbl.get_height() // 2))
