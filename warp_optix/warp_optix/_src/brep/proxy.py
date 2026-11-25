# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triangle shells around rational Bezier BRep sub-patches, for hardware-traversed exact intersection.

Each rational Bezier sub-patch (degree <= 7) gets a closed shell of triangles that encloses it: the
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
from typing import Any

import numpy as np
import warp as wp

from .device import ANALYTIC_SHELLS, MAX_ORDER, SHELL_LAYER_TRIANGLES, Analytic, Bezier, eval_patch, load_cubic
from .loader import CONE, CYLINDER, FLAG_SWAP, FLAG_TRIMMED, NURBS, PLANE, SPHERE, TORUS, BrepPrim

# Shell cells along each patch direction. The shell's thickness follows the grid's deviation from the
# surface, which falls with the square of GRID: 8 keeps shells about a quarter as thick as 4, so far
# fewer rays cross them without meeting the surface (with signed offsets, MC1 15% faster than 6).
GRID = 8
# The shell covers the patch extended by MARGIN in parameter space (the Bezier polynomial continues
# smoothly past its edges), so adjacent patches' shells overlap instead of sharing coincident side
# walls: a ray crossing a patch boundary must cross both patches' shells.
MARGIN = 0.05
# Each layer's offset = SAFETY x the surface's largest deviation from the grid on that layer's side
# (plus a small floor).
SAFETY = 2.0
OVERSAMPLE = 4  # deviation samples per cell edge
NODES = (GRID + 1) ** 2  # vertices per shell layer
FLOAT32_SLACK = 4.0 * 2.0**-23  # minimum offset, relative to the largest vertex coordinate
# EXPERIMENT (WARP_OPTIX_BREP_ANALYTIC_SHELLS=1): analytic faces get triangle shells too, traced like
# the NURBS shells instead of as AABBs. Curved faces: one shell per piece of the face's parameter range
# spanning <= PIECE_ANGLE, over CURVED_CELLS x CURVED_CELLS cells without the ones the trim puts entirely
# outside the face. Planes: one layer of triangles on the plane itself over PLANE_CELLS x PLANE_CELLS
# trim-culled cells (a plane needs no enclosure).
ANALYTIC = (PLANE, CYLINDER, CONE, SPHERE, TORUS) if ANALYTIC_SHELLS else ()
PIECE_ANGLE = np.pi / 2
PLANE_CELLS = 16
CURVED_CELLS = 8
# Which of (u, v) are angles, per kind.
_ANGULAR = {
    PLANE: (False, False),
    CYLINDER: (True, False),
    CONE: (True, False),
    SPHERE: (True, True),
    TORUS: (True, True),
}


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


@wp.func
def _shell(
    patch: Any, cvs: wp.array[wp.vec4], o: wp.vec3, p: int, vertices: wp.array2d[wp.vec3], grid: int, slide_walls: bool
):
    """Both layers of patch p's grid x grid shell: vertices[p, :nodes] outer, vertices[p, nodes:] inner.

    The patch's control points are relative to ``o``, its first control point: in float32, the
    deviation of the surface from the grid then keeps its digits instead of cancelling against the
    patch's position. ``slide_walls`` moves the side walls out by the surface's deviation along
    itself (see below); NURBS shells skip it (no pixel changes in the benchmark scenes, MC1 5% slower).
    """
    m = grid + 1
    nodes = m * m
    span = 1.0 + 2.0 * MARGIN
    average = wp.vec3(0.0)
    longest = float(0.0)
    lo = wp.vec3(1.0e30)
    hi = wp.vec3(-1.0e30)
    for k in range(nodes):
        st = _node(k, m, -MARGIN, span)
        x, ds, dt = eval_patch(patch, cvs, st[0], st[1])
        n = wp.cross(ds, dt)
        average += n
        longest = wp.max(longest, wp.length(n))
        lo = wp.min(lo, x)
        hi = wp.max(hi, x)
        vertices[p, k] = x  # the grid, read back below
    # Deviation of the surface from the grid's triangles (diagonal as in SHELL_TRIANGLES), per side:
    # signed along the surface normal, so each layer is only as far out as the surface bulges toward
    # it (a convex patch lies almost entirely on one side of its grid). The rest of the deviation runs
    # along the surface: where the normal tilts against a chord's sag (cones, tori), the surface
    # slides sideways off its grid point, past the side walls of a narrow patch (a chamfer, a thin
    # fillet) unless the walls move out by as much.
    above = float(0.0)
    below = float(0.0)
    slide = float(0.0)
    fine = grid * OVERSAMPLE + 1
    for k in range(fine * fine):
        f = _node(k, fine, 0.0, 1.0)
        x, ds, dt = eval_patch(patch, cvs, -MARGIN + span * f[0], -MARGIN + span * f[1])
        ci = wp.min(int(f[0] * float(grid)), grid - 1)
        cj = wp.min(int(f[1] * float(grid)), grid - 1)
        a = f[0] * float(grid) - float(ci)
        b = f[1] * float(grid) - float(cj)
        p00 = vertices[p, ci * m + cj]
        p11 = vertices[p, ci * m + cj + m + 1]
        linear = wp.vec3(0.0)
        if a >= b:
            linear = p00 + a * (vertices[p, ci * m + cj + m] - p00) + b * (p11 - vertices[p, ci * m + cj + m])
        else:
            linear = p00 + b * (vertices[p, ci * m + cj + 1] - p00) + a * (p11 - vertices[p, ci * m + cj + 1])
        nf = wp.cross(ds, dt)
        lf = wp.length(nf)
        if lf > 1.0e-12 * longest:
            d = wp.dot(x - linear, nf / lf)
            above = wp.max(above, d)
            below = wp.max(below, -d)
            slide = wp.max(slide, wp.length(x - linear - d * (nf / lf)))
        else:  # degenerate normal: unsigned, on both sides and along the surface
            above = wp.max(above, wp.length(x - linear))
            below = wp.max(below, wp.length(x - linear))
            slide = wp.max(slide, wp.length(x - linear))
    floor = 1.0e-4 * wp.length(hi - lo)
    # The vertices are stored in float32: far from the origin, a thin shell would round across the
    # surface it encloses. Keep each layer a few float32 steps of its coordinates away.
    reach = wp.max(wp.max(wp.abs(o + lo), wp.abs(o + hi)))
    out_offset = wp.max(SAFETY * above + floor, FLOAT32_SLACK * reach)
    in_offset = wp.max(SAFETY * below + floor, FLOAT32_SLACK * reach)
    push = wp.where(slide_walls, SAFETY * slide, 0.0)
    # Degenerate normals (poles, apexes, collapsed edges) borrow the average normal.
    average = average / (wp.length(average) + 1.0e-30)
    for k in range(nodes):
        st = _node(k, m, -MARGIN, span)
        x, ds, dt = eval_patch(patch, cvs, st[0], st[1])
        n = wp.cross(ds, dt)
        length = wp.length(n)
        if length > 1.0e-12 * longest:
            n = n / length
        else:
            n = average
        # Boundary nodes move out along the surface, across their edge, by the sideways deviation.
        i = k // m
        j = k % m
        out = wp.vec3(0.0)
        if i == 0 or i == grid:
            across = wp.cross(dt, n)  # tangent, perpendicular to the s edge
            la = wp.length(across)
            if la > 0.0:
                across = across / la
                if wp.dot(across, ds) < 0.0:
                    across = -across
                out += wp.where(i == 0, -push, push) * across
        if j == 0 or j == grid:
            across = wp.cross(n, ds)  # tangent, perpendicular to the t edge
            la = wp.length(across)
            if la > 0.0:
                across = across / la
                if wp.dot(across, dt) < 0.0:
                    across = -across
                out += wp.where(j == 0, -push, push) * across
        vertices[p, k] = o + x + out_offset * n + out
        vertices[p, nodes + k] = o + x - in_offset * n + out


@wp.kernel(enable_backward=False)
def _shell_vertices(
    cvs: wp.array[wp.vec4],
    patch_cv: wp.array[wp.int32],
    patch_degree: wp.array[wp.vec2i],
    patch_origin: wp.array[wp.vec3],
    vertices: wp.array2d[wp.vec3],
):
    """Shell vertices of every patch; ``cvs`` are relative to each patch's ``patch_origin``."""
    p = wp.tid()
    deg = patch_degree[p]
    if deg[0] == 3 and deg[1] == 3:
        _shell(load_cubic(cvs, patch_cv[p]), cvs, patch_origin[p], p, vertices, GRID, False)
    else:
        patch = Bezier()
        patch.base = patch_cv[p]
        patch.degree = deg
        _shell(patch, cvs, patch_origin[p], p, vertices, GRID, False)


@wp.kernel(enable_backward=False)
def _analytic_shell_vertices(
    kind: wp.array[wp.int32],
    origin: wp.array[wp.vec3],
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    axis: wp.array[wp.vec3],
    params: wp.array[wp.vec2],
    rect: wp.array[wp.vec4],
    cvs: wp.array[wp.vec4],
    grid: int,
    vertices: wp.array2d[wp.vec3],
):
    """grid x grid shell vertices of every analytic piece (see device.Analytic)."""
    p = wp.tid()
    patch = Analytic()
    patch.kind = kind[p]
    patch.x = x[p]
    patch.y = y[p]
    patch.axis = axis[p]
    patch.params = params[p]
    patch.rect = rect[p]
    _shell(patch, cvs, origin[p], p, vertices, grid, True)


@dataclass
class Proxy:
    points: np.ndarray  # (V, 3) float32, object space
    indices: np.ndarray  # (T, 3) uint32
    tri_prim: np.ndarray  # (T,) int32: the prim's local primitive index of each triangle's patch
    patches: np.ndarray  # (P,) int32: local primitive indices that have shells (unique)
    shell_uv: np.ndarray = None  # (S, 4) float32: face parameter rectangle of each shell
    tri_shell: np.ndarray = None  # (T,) int32: each triangle's shell (index into shell_uv)


def empty_proxy() -> Proxy:
    return Proxy(
        np.zeros((0, 3), np.float32),
        np.zeros((0, 3), np.uint32),
        np.zeros(0, np.int32),
        np.zeros(0, np.int32),
        np.zeros((0, 4), np.float32),
        np.zeros(0, np.int32),
    )


def _analytic_pieces(prim: BrepPrim) -> tuple[np.ndarray, np.ndarray]:
    """Local primitive index and parameter rectangle of each piece of the prim's analytic faces
    that get shells (ANALYTIC): angular parameters split into equal pieces of <= PIECE_ANGLE."""
    kinds = np.asarray(prim.prim_kind)
    local = np.flatnonzero(np.isin(kinds, [k for k in ANALYTIC if not (k == PLANE and PLANE_CELLS > 0)]))
    if not len(local):
        return np.zeros(0, np.int32), np.zeros((0, 4))
    r = np.asarray(prim.prim_uv, np.float64)[local]
    ang = np.array([_ANGULAR[int(k)] for k in kinds[local]], bool).reshape(-1, 2)
    nu = np.where(ang[:, 0], np.maximum(1, np.ceil((r[:, 1] - r[:, 0]) / PIECE_ANGLE - 1e-9)), 1).astype(np.int64)
    nv = np.where(ang[:, 1], np.maximum(1, np.ceil((r[:, 3] - r[:, 2]) / PIECE_ANGLE - 1e-9)), 1).astype(np.int64)
    count = nu * nv
    owner = np.repeat(np.arange(len(local)), count)
    k = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    i, j = k // nv[owner], k % nv[owner]
    du = (r[owner, 1] - r[owner, 0]) / nu[owner]
    dv = (r[owner, 3] - r[owner, 2]) / nv[owner]
    u0 = r[owner, 0] + i * du
    v0 = r[owner, 2] + j * dv
    rects = np.stack(
        [
            u0,
            np.where(i == nu[owner] - 1, r[owner, 1], u0 + du),
            v0,
            np.where(j == nv[owner] - 1, r[owner, 3], v0 + dv),
        ],
        1,
    )
    return local[owner].astype(np.int32), rects


def _culled_topology(keep: np.ndarray) -> np.ndarray:
    """Triangles of an n x n shell (vertex layout as _shell_topology's) over the cells ``keep``
    (n, n) marks: both layers of each kept cell, and a side wall on every edge of a kept cell that
    borders a dropped cell or the shell's edge, so that the kept cells stay enclosed."""
    n = keep.shape[0]
    m = n + 1
    nodes = m * m
    i, j = np.nonzero(keep)
    a = i * m + j
    b, c, d = a + m, a + 1, a + m + 1
    cells = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    tris = [cells, cells[:, ::-1] + nodes]
    pad = np.zeros((n + 2, n + 2), bool)
    pad[1:-1, 1:-1] = keep
    # (neighbor offset, edge's two vertices) per cell side
    for (di, dj), (k0, k1) in (((-1, 0), (a, c)), ((1, 0), (b, d)), ((0, -1), (a, b)), ((0, 1), (c, d))):
        open_ = ~pad[i + 1 + di, j + 1 + dj]
        k0, k1 = k0[open_], k1[open_]
        tris += [np.stack([k0, k1, k1 + nodes], 1), np.stack([k0, k1 + nodes, k0 + nodes], 1)]
    return np.concatenate(tris)


def _curved_keep(prim: BrepPrim, local: int, rect: np.ndarray, n: int) -> np.ndarray:
    """_trim_cells over the shell's extended parameter rectangle, keeping the cells that reach past
    the face's range in a periodic direction (the trim test wraps them; the classification does not)."""
    du, dv = rect[1] - rect[0], rect[3] - rect[2]
    ext = np.array([rect[0] - MARGIN * du, rect[1] + MARGIN * du, rect[2] - MARGIN * dv, rect[3] + MARGIN * dv])
    face = int(prim.prim_face[local])
    keep = _trim_cells(prim, face, ext, n)
    if keep.all():
        return keep
    rng, period = prim.face_range[face], prim.face_period[face]
    eu = np.linspace(ext[0], ext[1], n + 1)
    ev = np.linspace(ext[2], ext[3], n + 1)
    if period[0] > 0:
        keep |= ((eu[:-1] < rng[0]) | (eu[1:] > rng[1]))[:, None]
    if period[1] > 0:
        keep |= ((ev[:-1] < rng[2]) | (ev[1:] > rng[3]))[None, :]
    return keep


def _analytic_proxy(prim: BrepPrim, device) -> Proxy:
    owners, rects = _analytic_pieces(prim)
    if not len(owners):
        return empty_proxy()
    face = np.asarray(prim.prim_face)[owners]

    def arr(values, dtype):
        return wp.array(np.ascontiguousarray(values, np.float32), dtype=dtype, device=device)

    n = CURVED_CELLS if CURVED_CELLS > 0 else GRID
    block = 2 * (n + 1) ** 2
    vertices = wp.empty((len(owners), block), dtype=wp.vec3, device=device)
    wp.launch(
        _analytic_shell_vertices,
        dim=len(owners),
        inputs=[
            wp.array(np.asarray(prim.prim_kind)[owners].astype(np.int32), dtype=wp.int32, device=device),
            arr(prim.face_origin[face], wp.vec3),
            arr(prim.face_x[face], wp.vec3),
            arr(prim.face_y[face], wp.vec3),
            arr(prim.face_axis[face], wp.vec3),
            arr(prim.face_params[face], wp.vec2),
            arr(rects, wp.vec4),
            wp.zeros(1, dtype=wp.vec4, device=device),
            n,
        ],
        outputs=[vertices],
        device=device,
    )
    if CURVED_CELLS > 0:
        cache = {}
        indices, tri_shell = [], []
        for k, (local, rect) in enumerate(zip(owners, rects)):
            keep = _curved_keep(prim, int(local), rect, n)
            key = keep.tobytes()
            if key not in cache:
                cache[key] = _culled_topology(keep).astype(np.uint32)
            tris = cache[key]
            indices.append(tris + np.uint32(k * block))
            tri_shell.append(np.full(len(tris), k, np.int32))
        tri_shell = np.concatenate(tri_shell)
        return Proxy(
            vertices.numpy().reshape(-1, 3),
            np.concatenate(indices),
            owners[tri_shell],
            np.unique(owners),
            rects.astype(np.float32),
            tri_shell,
        )
    offsets = 2 * NODES * np.arange(len(owners), dtype=np.uint32)
    return Proxy(
        vertices.numpy().reshape(-1, 3),
        (SHELL_TRIANGLES.astype(np.uint32)[None] + offsets[:, None, None]).reshape(-1, 3),
        np.repeat(owners, len(SHELL_TRIANGLES)),
        np.unique(owners),
        rects.astype(np.float32),
        np.repeat(np.arange(len(owners), dtype=np.int32), len(SHELL_TRIANGLES)),
    )


def _face_chords(prim: BrepPrim, face: int):
    """The face's trim chords (a0, c0, a1, c1) and bulges; a face's chords are contiguous."""
    base, count = (int(x) for x in prim.face_slab[face])
    lo, hi = int(prim.slab_start[base]), int(prim.slab_start[base + count])
    ids = prim.slab_segments[lo:hi]
    if not len(ids):
        return np.zeros((0, 4)), np.zeros(0)
    first, last = int(ids.min()), int(ids.max()) + 1
    return np.asarray(prim.segments[first:last], np.float64), np.asarray(prim.segment_bulge[first:last], np.float64)


def _trim_cells(prim: BrepPrim, face: int, rect: np.ndarray, n: int) -> np.ndarray:
    """(n, n) mask of the cells of ``rect`` (u0, u1, v0, v1; row-major in u) that may hold points
    the face accepts: cells that a trim chord (thickened by its bulge, which covers its curve)
    touches, and the others whose center has inside parity (the slab test's, toward +c)."""
    keep = np.ones((n, n), bool)
    if not (prim.face_flags[face] & FLAG_TRIMMED):
        return keep
    seg, bulge = _face_chords(prim, face)
    if not len(seg):
        return keep
    swap = bool(prim.face_flags[face] & FLAG_SWAP)
    edges_u = np.linspace(rect[0], rect[1], n + 1)
    edges_v = np.linspace(rect[2], rect[3], n + 1)
    edges_a, edges_c = (edges_v, edges_u) if swap else (edges_u, edges_v)
    scale = max(1.0, float(np.abs(seg).max()), *np.abs(rect))
    slack = 64.0 * np.finfo(np.float32).eps * scale
    # cells [a cell, c cell], grown by the slack
    a0 = edges_a[:-1, None, None] - slack
    a1 = edges_a[1:, None, None] + slack
    c0 = edges_c[None, :-1, None] - slack
    c1 = edges_c[None, 1:, None] + slack
    r = np.abs(bulge) + slack
    sa0, sc0, sa1, sc1 = (seg[:, k] for k in range(4))
    boxes = (
        (np.maximum(sa0, sa1) + r >= a0)
        & (np.minimum(sa0, sa1) - r <= a1)
        & (np.maximum(sc0, sc1) + r >= c0)
        & (np.minimum(sc0, sc1) - r <= c1)
    )
    # Separating axis: the chord's normal. Signed distances of the cell's corners to the chord's line.
    da, dc = sa1 - sa0, sc1 - sc0
    length = np.maximum(np.hypot(da, dc), 1e-300)
    na, nc = -dc / length, da / length
    off = na * sa0 + nc * sc0
    lo = np.minimum(na * a0, na * a1) + np.minimum(nc * c0, nc * c1) - off
    hi = np.maximum(na * a0, na * a1) + np.maximum(nc * c0, nc * c1) - off
    touched = (boxes & (lo <= r) & (hi >= -r)).any(-1)
    # The face's chords exclude the ones along c with no bulge (loader._build_trim: the slab test needs
    # no chord that no vertical ray crosses). Such a chord joins two chord endpoints at the same a, so
    # cells at that a, within the c span of the endpoints there, also touch the boundary.
    ends_a = np.concatenate([sa0, sa1])
    ends_c = np.concatenate([sc0, sc1])
    values, group, count = np.unique(ends_a, return_inverse=True, return_counts=True)
    shared = count >= 2
    if shared.any():
        c_lo = np.full(len(values), np.inf)
        c_hi = np.full(len(values), -np.inf)
        np.minimum.at(c_lo, group, ends_c)
        np.maximum.at(c_hi, group, ends_c)
        va, vlo, vhi = values[shared], c_lo[shared], c_hi[shared]
        touched |= ((a0 <= va) & (va <= a1) & (c0 <= vhi) & (vlo <= c1)).any(-1)
    # Parity at the centers of untouched cells: chords crossing the vertical line, above the center.
    ca = 0.5 * (edges_a[:-1] + edges_a[1:])[:, None, None]
    cc = 0.5 * (edges_c[:-1] + edges_c[1:])[None, :, None]
    crossing = (sa0 <= ca) != (sa1 <= ca)
    with np.errstate(divide="ignore", invalid="ignore"):
        height = sc0 + (ca - sa0) * (sc1 - sc0) / (sa1 - sa0)
    inside = (np.count_nonzero(crossing & (height > cc), axis=-1) & 1) == 1
    keep = touched | inside
    return keep.T if swap else keep


def _plane_proxy(prim: BrepPrim) -> Proxy:
    """One on-plane triangle layer per planar face (PLANE_CELLS), trim-culled cell by cell."""
    kinds = np.asarray(prim.prim_kind)
    local = np.flatnonzero(kinds == PLANE) if (PLANE in ANALYTIC and PLANE_CELLS > 0) else np.zeros(0, int)
    if not len(local):
        return empty_proxy()
    n = PLANE_CELLS
    m = n + 1
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    a = i * m + j
    quad = np.stack([np.stack([a, a + m, a + m + 1], -1), np.stack([a, a + m + 1, a + 1], -1)], -2)  # (n, n, 2, 3)
    points, indices, tri_prim, rects, tri_shell = [], [], [], [], []
    nverts = 0
    for p in local:
        face = int(prim.prim_face[p])
        r = np.asarray(prim.prim_uv[p], np.float64)
        # Grown like accept_uv's tolerance, so that rounding does not open gaps at the face's edges.
        eu = 1.0e-5 * (r[1] - r[0]) + 1.0e-7
        ev = 1.0e-5 * (r[3] - r[2]) + 1.0e-7
        g = np.array([r[0] - eu, r[1] + eu, r[2] - ev, r[3] + ev])
        keep = _trim_cells(prim, face, g, n)
        if not keep.any():
            continue
        u = np.linspace(g[0], g[1], m)
        v = np.linspace(g[2], g[3], m)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        o, x, y = prim.face_origin[face], prim.face_x[face], prim.face_y[face]
        pts = o + uu.reshape(-1, 1) * x + vv.reshape(-1, 1) * y
        tris = quad[keep].reshape(-1, 3) + nverts
        points.append(pts)
        indices.append(tris)
        tri_prim.append(np.full(len(tris), p, np.int32))
        tri_shell.append(np.full(len(tris), len(rects), np.int32))
        rects.append(r)
        nverts += len(pts)
    if not rects:
        return empty_proxy()
    return Proxy(
        np.concatenate(points).astype(np.float32),
        np.concatenate(indices).astype(np.uint32),
        np.concatenate(tri_prim),
        np.unique(local),
        np.asarray(rects, np.float32),
        np.concatenate(tri_shell),
    )


def _concat(a: Proxy, b: Proxy) -> Proxy:
    if not len(b.indices):
        return a
    if not len(a.indices):
        return b
    return Proxy(
        np.concatenate([a.points, b.points]),
        np.concatenate([a.indices, b.indices + np.uint32(len(a.points))]),
        np.concatenate([a.tri_prim, b.tri_prim]),
        np.concatenate([a.patches, b.patches]),
        np.concatenate([a.shell_uv, b.shell_uv]),
        np.concatenate([a.tri_shell, b.tri_shell + np.int32(len(a.shell_uv))]),
    )


def build_proxy(prim: BrepPrim, device=None) -> Proxy:
    """Shells around the prim's NURBS sub-patches, built on ``device`` (default: Warp's).

    Other primitives keep their AABBs: an analytic face is one cheap closed-form call on its AABB,
    while a shell takes an any-hit call per layer crossed. NURBS shells win (3-11x): they replace
    several overlapping patch AABBs, each needing Newton, with one well-seeded refinement. That
    holds for bicubic patches and for higher degrees (quintic springs 1.2x faster than on AABBs),
    but not for degree (3, 4) patches: their AABB path has a specialized evaluator and rejects
    most candidates with its oriented bound and convex hull, before Newton (nurbs_parts 14% faster).
    """
    degree = np.asarray(prim.prim_degree)
    patches = np.flatnonzero(
        (prim.prim_kind == NURBS)
        & np.all((degree >= 1) & (degree < MAX_ORDER), axis=1)
        & ~np.all(degree == (3, 4), axis=1)
    ).astype(np.int32)
    device = wp.get_device(device)
    analytic = _concat(_analytic_proxy(prim, device), _plane_proxy(prim)) if ANALYTIC else empty_proxy()
    if not len(patches):
        return analytic
    # Each patch's control points relative to its first one, shifted in float64 (see _shell).
    base = prim.prim_cv[patches].astype(np.int64)
    count = np.prod(prim.prim_degree[patches].astype(np.int64) + 1, axis=1)
    owner = np.repeat(np.arange(len(patches)), count)
    index = np.repeat(base - np.cumsum(count) + count, count) + np.arange(count.sum())
    first = prim.cvs[base].astype(np.float64)
    origin = (first[:, :3] / first[:, 3:4]).astype(np.float32).astype(np.float64)  # as the kernel adds it
    cvs = prim.cvs[index].astype(np.float64)
    cvs[:, :3] -= origin[owner] * cvs[:, 3:4]
    vertices = wp.empty((len(patches), 2 * NODES), dtype=wp.vec3, device=device)
    wp.launch(
        _shell_vertices,
        dim=len(patches),
        inputs=[
            wp.array(cvs.astype(np.float32), dtype=wp.vec4, device=device),
            wp.array((np.cumsum(count) - count).astype(np.int32), dtype=wp.int32, device=device),
            wp.array(prim.prim_degree[patches].astype(np.int32), dtype=wp.vec2i, device=device),
            wp.array(origin.astype(np.float32), dtype=wp.vec3, device=device),
        ],
        outputs=[vertices],
        device=device,
    )
    offsets = 2 * NODES * np.arange(len(patches), dtype=np.uint32)
    nurbs = Proxy(
        vertices.numpy().reshape(-1, 3),
        (SHELL_TRIANGLES.astype(np.uint32)[None] + offsets[:, None, None]).reshape(-1, 3),
        np.repeat(patches, len(SHELL_TRIANGLES)),
        patches,
        np.asarray(prim.prim_uv[patches], np.float32).reshape(-1, 4),
        np.repeat(np.arange(len(patches), dtype=np.int32), len(SHELL_TRIANGLES)),
    )
    return _concat(nurbs, analytic)


def shell_sbt_indices(prim: BrepPrim, proxy: Proxy) -> np.ndarray:
    """Per shell triangle, 0 for a bicubic patch's shell and 1 for another degree's: renderers
    trace the two with separate any-hit programs (see device.proxy_hit_bezier)."""
    degree = np.asarray(prim.prim_degree)[proxy.tri_prim]
    return np.any(degree != 3, axis=1).astype(np.uint32)
