# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triangle shells around bicubic BRep sub-patches, for hardware-traversed exact intersection.

Each bicubic rational Bezier sub-patch gets a closed shell of triangles that encloses it: the
patch evaluated on a grid, offset outward and inward along the surface normal by more than the
grid's deviation from the surface, joined by side walls along the patch edges. A ray that reaches
the surface must cross the shell first, so the RT cores find candidates with hardware triangle
tests; an any-hit program then refines the candidate with Newton from the triangle's
interpolated patch parameters (an excellent seed) and the face's trim test, and ignores it when
the ray misses the surface.

Every shell has the same topology and the same patch parameters at its vertices
(``SHELL_TRIANGLES``, ``SHELL_ST``); only the vertex positions differ, and one kernel computes
them for all patches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .device import SHELL_LAYER_TRIANGLES, eval_cubic, load_cubic
from .loader import NURBS, BrepPrim

# Shell cells along each patch direction. The shell is offset by twice the grid's deviation from the
# surface, which falls with the square of GRID: 6 keeps shells about half as thick as 4, so far fewer
# rays cross them without meeting the surface, for twice the triangles.
GRID = 6
# The shell covers the patch extended by MARGIN in parameter space (the Bezier polynomial continues
# smoothly past its edges), so adjacent patches' shells overlap instead of sharing coincident side
# walls: a ray crossing a patch boundary must cross both patches' shells.
MARGIN = 0.05
SAFETY = 2.0  # offset = SAFETY x the grid's deviation from the surface (plus a small floor)
OVERSAMPLE = 4  # deviation samples per cell edge
NODES = (GRID + 1) ** 2  # vertices per shell layer
FLOAT32_SLACK = 4.0 * 2.0**-23  # minimum offset, relative to the largest vertex coordinate


def _shell_topology() -> tuple[np.ndarray, np.ndarray]:
    """Triangles of one shell over its 2 * NODES vertices (outer layer, then inner, row-major in
    s), and the patch parameters (s, t) at each triangle corner."""
    m = GRID + 1
    i, j = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    a = (i * m + j).ravel()
    b, c, d = a + m, a + 1, a + m + 1
    cells = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    tris = [cells, cells[:, ::-1] + NODES]
    # Side walls along the four boundary edges, joining outer vertex k to inner vertex k + NODES.
    k = np.arange(m)
    for edge in (k * m, k * m + GRID, k, GRID * m + k):
        k0, k1 = edge[:-1], edge[1:]
        tris += [np.stack([k0, k1, k1 + NODES], 1), np.stack([k0, k1 + NODES, k0 + NODES], 1)]
    tris = np.concatenate(tris)
    node = np.linspace(-MARGIN, 1.0 + MARGIN, m)
    st = np.stack(np.meshgrid(node, node, indexing="ij"), -1).reshape(-1, 2)
    return tris, np.concatenate([st, st])[tris].astype(np.float32)


SHELL_TRIANGLES, SHELL_ST = _shell_topology()
assert SHELL_LAYER_TRIANGLES == 2 * GRID * GRID  # proxy_hit tells the layers apart by triangle index


@wp.func
def _node(k: int, count: int, lo: float, span: float) -> wp.vec2:
    """Parameters of node k of a count x count grid over [lo, lo + span]^2, row-major in s."""
    step = span / float(count - 1)
    return wp.vec2(lo + step * float(k // count), lo + step * float(k % count))


@wp.kernel(enable_backward=False)
def _shell_vertices(cvs: wp.array[wp.vec4], patch_cv: wp.array[wp.int32], vertices: wp.array2d[wp.vec3]):
    """Both layers of patch p's shell: vertices[p, :NODES] outer, vertices[p, NODES:] inner."""
    p = wp.tid()
    c = load_cubic(cvs, patch_cv[p])
    # Evaluate relative to the first control point: in float32, the deviation of the surface from
    # the grid then keeps its digits instead of cancelling against the patch's position.
    o = wp.vec3(c.x[0, 0], c.y[0, 0], c.z[0, 0]) / c.w[0, 0]
    c.x = c.x - o[0] * c.w
    c.y = c.y - o[1] * c.w
    c.z = c.z - o[2] * c.w
    m = GRID + 1
    span = 1.0 + 2.0 * MARGIN
    average = wp.vec3(0.0)
    longest = float(0.0)
    lo = wp.vec3(1.0e30)
    hi = wp.vec3(-1.0e30)
    for k in range(NODES):
        st = _node(k, m, -MARGIN, span)
        x, ds, dt = eval_cubic(c, st[0], st[1])
        n = wp.cross(ds, dt)
        average += n
        longest = wp.max(longest, wp.length(n))
        lo = wp.min(lo, x)
        hi = wp.max(hi, x)
        vertices[p, k] = x  # the grid, read back below
    # Deviation of the surface from the grid's triangles (diagonal as in SHELL_TRIANGLES).
    deviation = float(0.0)
    fine = GRID * OVERSAMPLE + 1
    for k in range(fine * fine):
        f = _node(k, fine, 0.0, 1.0)
        x, ds, dt = eval_cubic(c, -MARGIN + span * f[0], -MARGIN + span * f[1])
        ci = wp.min(int(f[0] * float(GRID)), GRID - 1)
        cj = wp.min(int(f[1] * float(GRID)), GRID - 1)
        a = f[0] * float(GRID) - float(ci)
        b = f[1] * float(GRID) - float(cj)
        p00 = vertices[p, ci * m + cj]
        p11 = vertices[p, ci * m + cj + m + 1]
        linear = wp.vec3(0.0)
        if a >= b:
            linear = p00 + a * (vertices[p, ci * m + cj + m] - p00) + b * (p11 - vertices[p, ci * m + cj + m])
        else:
            linear = p00 + b * (vertices[p, ci * m + cj + 1] - p00) + a * (p11 - vertices[p, ci * m + cj + 1])
        deviation = wp.max(deviation, wp.length(x - linear))
    offset = SAFETY * deviation + 1.0e-4 * wp.length(hi - lo)
    # The vertices are stored in float32: far from the origin, a thin shell would round across the
    # surface it encloses. Keep it a few float32 steps of its coordinates thick.
    reach = wp.max(wp.max(wp.abs(o + lo), wp.abs(o + hi)))
    offset = wp.max(offset, FLOAT32_SLACK * reach)
    # Degenerate normals (poles, apexes, collapsed edges) borrow the average normal.
    average = average / (wp.length(average) + 1.0e-30)
    for k in range(NODES):
        st = _node(k, m, -MARGIN, span)
        x, ds, dt = eval_cubic(c, st[0], st[1])
        n = wp.cross(ds, dt)
        length = wp.length(n)
        if length > 1.0e-12 * longest:
            n = n / length
        else:
            n = average
        vertices[p, k] = o + x + offset * n
        vertices[p, NODES + k] = o + x - offset * n


@dataclass
class Proxy:
    points: np.ndarray  # (V, 3) float32, object space
    indices: np.ndarray  # (T, 3) uint32
    tri_prim: np.ndarray  # (T,) int32: the prim's local primitive index of each triangle's patch
    patches: np.ndarray  # (P,) int32: local primitive indices that have shells


def empty_proxy() -> Proxy:
    return Proxy(
        np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32), np.zeros(0, np.int32), np.zeros(0, np.int32)
    )


def build_proxy(prim: BrepPrim, device=None) -> Proxy:
    """Shells around the prim's bicubic NURBS patches, built on ``device`` (default: Warp's).

    Other primitives keep their AABBs: an analytic face is one cheap closed-form call on its AABB,
    while a shell takes an any-hit call per layer crossed. NURBS shells win (3-11x): they replace
    several overlapping patch AABBs, each needing Newton, with one well-seeded refinement.
    """
    patches = np.nonzero((prim.prim_kind == NURBS) & np.all(prim.prim_degree == 3, axis=1))[0].astype(np.int32)
    if not len(patches):
        return empty_proxy()
    device = wp.get_device(device)
    vertices = wp.empty((len(patches), 2 * NODES), dtype=wp.vec3, device=device)
    wp.launch(
        _shell_vertices,
        dim=len(patches),
        inputs=[
            wp.array(prim.cvs.astype(np.float32), dtype=wp.vec4, device=device),
            wp.array(prim.prim_cv[patches], dtype=wp.int32, device=device),
        ],
        outputs=[vertices],
        device=device,
    )
    first = 2 * NODES * np.arange(len(patches), dtype=np.uint32)
    return Proxy(
        vertices.numpy().reshape(-1, 3),
        (SHELL_TRIANGLES.astype(np.uint32)[None] + first[:, None, None]).reshape(-1, 3),
        np.repeat(patches, len(SHELL_TRIANGLES)),
        patches,
    )
