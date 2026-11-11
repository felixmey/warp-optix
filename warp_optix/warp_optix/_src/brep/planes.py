# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planar faces as hardware triangles with opacity micromaps.

A planar face is exactly the rectangle of its (u, v) range cut by its trims, so two triangles over
the rectangle are exact geometry. Each triangle's opacity micromap comes from the face's trim cell
grid (see ``trim_grid``): micro-triangles over cells entirely inside the trims are opaque, over
cells entirely outside transparent, and the RT cores resolve both without running any program.
Only micro-triangles touching a cell crossed by a trim edge are unknown and run an any-hit program
with the exact trim test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..micromaps import (
    INDEX_FULLY_OPAQUE,
    INDEX_FULLY_UNKNOWN_OPAQUE,
    OPAQUE,
    TRANSPARENT,
    UNKNOWN_OPAQUE,
    micro_triangle_barycentrics,
)
from .loader import FLAG_SWAP, FLAG_TRIMMED, PLANE, BrepPrim
from .trim_grid import build_trim_grids

MAX_LEVEL = 8  # 4^8 micro-triangles (16 KiB at two bits each) per triangle


@dataclass
class PlaneTriangles:
    points: np.ndarray  # (V, 3) float32, object space
    indices: np.ndarray  # (T, 3) uint32
    tri_face: np.ndarray  # (T,) int32: the prim's local face of each triangle
    prims: np.ndarray  # (P,) int32: local primitive indices replaced by triangles
    states: list  # one uint8 state array per micromap
    micromap_indices: np.ndarray  # (T,) int32: micromap per triangle, or a predefined index


def _rect_sums(table: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero border: sums[j1, i1] - ... gives rectangle sums."""
    sums = np.zeros((table.shape[0] + 1, table.shape[1] + 1), np.int64)
    sums[1:, 1:] = table.cumsum(0).cumsum(1)
    return sums


def _states(prim: BrepPrim, face: int, corners_uv: np.ndarray, level_bias: int):
    """Micromap states of one triangle with corners (3, 2) in face (u, v), or a predefined index."""
    grid, grid_map, cell_info, _, _ = build_trim_grids(prim)
    base, na, nc = (int(x) for x in grid[face])
    if na == 0:
        return INDEX_FULLY_UNKNOWN_OPAQUE  # no grid: every hit runs the exact (slab) trim test
    lo_a, lo_c, scale_a, scale_c = (float(x) for x in grid_map[face])
    info = cell_info[base : base + na * nc].reshape(nc, na, 2)
    crossed = (info[..., 1] >> 1) > 0
    inside = (info[..., 1] & 1) == 1
    crossed_sums, inside_sums = _rect_sums(crossed), _rect_sums(inside)
    level = int(np.clip(np.ceil(np.log2(max(na, nc))) + level_bias, 0, MAX_LEVEL))
    bary = micro_triangle_barycentrics(level)  # (n, 3, 2)
    w0 = 1.0 - bary[..., 0] - bary[..., 1]
    uv = w0[..., None] * corners_uv[0] + bary[..., 0:1] * corners_uv[1] + bary[..., 1:2] * corners_uv[2]
    a, c = (uv[..., 1], uv[..., 0]) if prim.face_flags[face] & FLAG_SWAP else (uv[..., 0], uv[..., 1])
    pad = 1e-4  # in cells
    i0 = np.clip(np.floor((a.min(1) - lo_a) * scale_a - pad), 0, na - 1).astype(np.int64)
    i1 = np.clip(np.floor((a.max(1) - lo_a) * scale_a + pad), 0, na - 1).astype(np.int64) + 1
    j0 = np.clip(np.floor((c.min(1) - lo_c) * scale_c - pad), 0, nc - 1).astype(np.int64)
    j1 = np.clip(np.floor((c.max(1) - lo_c) * scale_c + pad), 0, nc - 1).astype(np.int64) + 1

    def rect(s):
        return s[j1, i1] - s[j0, i1] - s[j1, i0] + s[j0, i0]

    area = (i1 - i0) * (j1 - j0)
    n_inside = rect(inside_sums)
    states = np.full(len(bary), UNKNOWN_OPAQUE, np.uint8)
    clear = rect(crossed_sums) == 0
    states[clear & (n_inside == area)] = OPAQUE
    states[clear & (n_inside == 0)] = TRANSPARENT
    return states


def plane_triangles(prim: BrepPrim, level_bias: int = 1) -> PlaneTriangles:
    """Two triangles per planar primitive over its face's (u, v) range, with micromap states.

    ``level_bias``: micromap subdivision beyond one micro-triangle edge per grid cell (each step
    halves the unknown band along trim edges and quadruples the micromap's memory).
    """
    prims = np.nonzero(prim.prim_kind == PLANE)[0].astype(np.int32)
    points, indices, tri_face, states, map_index = [], [], [], [], []
    for k, p in enumerate(prims):
        face = int(prim.prim_face[p])
        u0, u1, v0, v1 = (float(x) for x in prim.face_range[face])
        rect = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
        origin, x, y = prim.face_origin[face], prim.face_x[face], prim.face_y[face]
        points.append(origin + rect[:, :1] * x + rect[:, 1:] * y)
        indices += [[4 * k, 4 * k + 1, 4 * k + 2], [4 * k, 4 * k + 2, 4 * k + 3]]
        tri_face += [face, face]
        for corners in (rect[[0, 1, 2]], rect[[0, 2, 3]]):
            if not prim.face_flags[face] & FLAG_TRIMMED:
                map_index.append(INDEX_FULLY_OPAQUE)
                continue
            result = _states(prim, face, corners, level_bias)
            if isinstance(result, int):
                map_index.append(result)
            elif np.all(result == result[0]) and result[0] in (OPAQUE, TRANSPARENT):
                map_index.append(-2 if result[0] == OPAQUE else -1)  # predefined: no micromap needed
            else:
                map_index.append(len(states))
                states.append(result)
    return PlaneTriangles(
        np.concatenate(points).astype(np.float32) if points else np.zeros((0, 3), np.float32),
        np.asarray(indices, np.uint32).reshape(-1, 3),
        np.asarray(tri_face, np.int32),
        prims,
        states,
        np.asarray(map_index, np.int32),
    )
