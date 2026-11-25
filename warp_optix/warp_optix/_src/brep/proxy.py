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


def _curved_keep(prim: BrepPrim, owners: np.ndarray, rects: np.ndarray, n: int, device, tables) -> np.ndarray:
    """(K, n, n) trim cell masks of the pieces over their shells' extended parameter rectangles,
    keeping the cells that reach past the face's range in a periodic direction (the trim test
    wraps them; the classification does not)."""
    du, dv = rects[:, 1] - rects[:, 0], rects[:, 3] - rects[:, 2]
    ext = np.stack(
        [rects[:, 0] - MARGIN * du, rects[:, 1] + MARGIN * du, rects[:, 2] - MARGIN * dv, rects[:, 3] + MARGIN * dv], 1
    )
    face = np.asarray(prim.prim_face, np.int64)[owners]
    keep = _trim_cells_batch(prim, face, ext, n, device, tables)
    rng = np.asarray(prim.face_range, np.float64)[face]
    period = np.asarray(prim.face_period, np.float64)[face]
    eu = np.linspace(ext[:, 0], ext[:, 1], n + 1, axis=1)
    ev = np.linspace(ext[:, 2], ext[:, 3], n + 1, axis=1)
    wrap_u = (period[:, 0:1] > 0) & ((eu[:, :-1] < rng[:, 0:1]) | (eu[:, 1:] > rng[:, 1:2]))
    wrap_v = (period[:, 1:2] > 0) & ((ev[:, :-1] < rng[:, 2:3]) | (ev[:, 1:] > rng[:, 3:4]))
    return keep | wrap_u[:, :, None] | wrap_v[:, None, :]


def _gather_topologies(keep: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    """_culled_topology of each (n, n) mask in ``keep`` (K, n, n), offset to the item's vertex block:
    (triangles, item of each triangle). Masks repeat a lot, so each distinct one is built once."""
    k_items = len(keep)
    packed = np.packbits(keep.reshape(k_items, -1), axis=1)
    distinct, first_item, inverse = np.unique(packed, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    topologies = [_culled_topology(keep[i]).astype(np.uint32) for i in first_item]
    lengths = np.array([len(t) for t in topologies], np.int64)
    flat = np.concatenate(topologies)
    starts = np.cumsum(lengths) - lengths
    count = lengths[inverse]
    owner = np.repeat(np.arange(k_items), count)
    rows = np.arange(count.sum()) + np.repeat(starts[inverse] - np.cumsum(count) + count, count)
    return flat[rows] + (owner * block).astype(np.uint32)[:, None], owner.astype(np.int32)


def _analytic_proxy(prim: BrepPrim, device, tables: dict | None = None) -> Proxy:
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
        indices, tri_shell = _gather_topologies(_curved_keep(prim, owners, rects, n, device, tables), block)
        return Proxy(
            vertices.numpy().reshape(-1, 3),
            indices,
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


def _face_chord_ranges(prim: BrepPrim) -> tuple[np.ndarray, np.ndarray]:
    """Per face, the [first, end) range of its trim chords in ``prim.segments``: the chords its
    slabs list (a face's chords are contiguous)."""
    nfaces = len(prim.face_slab)
    first = np.full(nfaces, np.iinfo(np.int64).max)
    last = np.full(nfaces, -1, np.int64)
    base, count = prim.face_slab[:, 0].astype(np.int64), prim.face_slab[:, 1].astype(np.int64)
    starts = np.asarray(prim.slab_start, np.int64)
    lo, hi = starts[base], starts[base + count]
    face = np.repeat(np.arange(nfaces), hi - lo)
    entry = np.arange((hi - lo).sum()) + np.repeat(lo - np.cumsum(hi - lo) + (hi - lo), hi - lo)
    ids = np.asarray(prim.slab_segments, np.int64)[entry]
    np.minimum.at(first, face, ids)
    np.maximum.at(last, face, ids)
    empty = last < 0
    first[empty], last[empty] = 0, -1
    return first, last + 1


def _shared_endpoints(segments: np.ndarray, first: np.ndarray, end: np.ndarray):
    """Chord endpoints that share their a-value with another endpoint of the same face (see
    _trim_cells_kernel): (face, a, lowest c, highest c) per shared a-value, sorted by face. Face f's
    chords are segments[first[f]:end[f]]."""
    count = end - first
    seg_face = np.repeat(np.arange(len(first)), count)
    seg = np.arange(count.sum()) + np.repeat(first - np.cumsum(count) + count, count)
    s = segments[seg]
    face = np.concatenate([seg_face, seg_face])
    a = np.concatenate([s[:, 0], s[:, 2]])
    c = np.concatenate([s[:, 1], s[:, 3]])
    # = np.lexsort((a, face)): a stable integer sort (radix) after the float sort is much faster.
    order = np.argsort(a, kind="stable")
    order = order[np.argsort(face[order], kind="stable")]
    face, a, c = face[order], a[order], c[order]
    new = np.ones(len(a), bool)
    new[1:] = (face[1:] != face[:-1]) | (a[1:] != a[:-1])
    starts = np.flatnonzero(new)
    if not len(starts):
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0)
    sizes = np.diff(np.append(starts, len(a)))
    shared = sizes >= 2
    c_lo = np.minimum.reduceat(c, starts)[shared]
    c_hi = np.maximum.reduceat(c, starts)[shared]
    return face[starts][shared], a[starts][shared], c_lo, c_hi


@wp.kernel(enable_backward=False)
def _trim_cells_kernel(
    faces: wp.array[wp.int32],
    rects: wp.array2d[wp.float64],
    swap: wp.array[wp.int32],
    first: wp.array[wp.int32],
    count: wp.array[wp.int32],
    face_abs: wp.array[wp.float64],
    segments: wp.array2d[wp.float64],
    bulges: wp.array[wp.float64],
    shared_first: wp.array[wp.int32],
    shared_count: wp.array[wp.int32],
    shared: wp.array2d[wp.float64],
    n: int,
    keep: wp.array3d[wp.uint8],
):
    """Whether cell (ia, ic) of item k's n x n grid over rects[k] (u0, u1, v0, v1; keep is
    row-major in u) may hold points the face accepts: cells that a trim chord (thickened by its
    bulge, which covers its curve) touches, and the others whose center has inside parity (the
    slab test's, toward +c). Faces without chords keep every cell. Float64, with cells grown by a
    slack for the device's float32 arithmetic."""
    tid = wp.tid()
    k = tid // (n * n)
    ia = (tid // n) % n
    ic = tid % n
    face = faces[k]
    s = count[face]
    sw = swap[k]
    # the cell's a and c ranges, as np.linspace(rect[lo], rect[hi], n + 1)
    a_lo = rects[k, 0]
    a_hi = rects[k, 1]
    c_lo = rects[k, 2]
    c_hi = rects[k, 3]
    if sw != 0:
        a_lo = rects[k, 2]
        a_hi = rects[k, 3]
        c_lo = rects[k, 0]
        c_hi = rects[k, 1]
    step_a = (a_hi - a_lo) / wp.float64(n)
    step_c = (c_hi - c_lo) / wp.float64(n)
    e_a0 = wp.float64(ia) * step_a + a_lo
    e_a1 = wp.float64(ia + 1) * step_a + a_lo
    if ia + 1 == n:
        e_a1 = a_hi
    e_c0 = wp.float64(ic) * step_c + c_lo
    e_c1 = wp.float64(ic + 1) * step_c + c_lo
    if ic + 1 == n:
        e_c1 = c_hi
    scale = wp.max(
        wp.max(wp.float64(1.0), face_abs[face]),
        wp.max(wp.max(wp.abs(a_lo), wp.abs(a_hi)), wp.max(wp.abs(c_lo), wp.abs(c_hi))),
    )
    slack = wp.float64(64.0 * 1.1920928955078125e-07) * scale
    a0 = e_a0 - slack
    a1 = e_a1 + slack
    c0 = e_c0 - slack
    c1 = e_c1 + slack
    ca = wp.float64(0.5) * (e_a0 + e_a1)
    cc = wp.float64(0.5) * (e_c0 + e_c1)
    touched = int(0)
    above = int(0)
    base = first[face]
    for q in range(s):
        g = base + q
        sa0 = segments[g, 0]
        sc0 = segments[g, 1]
        sa1 = segments[g, 2]
        sc1 = segments[g, 3]
        r = wp.abs(bulges[g]) + slack
        if touched == 0:  # box test, then the chord's normal as separating axis
            if (
                wp.max(sa0, sa1) + r >= a0
                and wp.min(sa0, sa1) - r <= a1
                and wp.max(sc0, sc1) + r >= c0
                and wp.min(sc0, sc1) - r <= c1
            ):
                da = sa1 - sa0
                dc = sc1 - sc0
                length = wp.max(wp.sqrt(da * da + dc * dc), wp.float64(1.0e-300))
                na = -dc / length
                nc = da / length
                off = na * sa0 + nc * sc0
                lo = wp.min(na * a0, na * a1) + wp.min(nc * c0, nc * c1) - off
                hi = wp.max(na * a0, na * a1) + wp.max(nc * c0, nc * c1) - off
                if lo <= r and hi >= -r:
                    touched = 1
        if (sa0 <= ca) != (sa1 <= ca):  # parity: chords crossing the vertical line above the center
            height = sc0 + (ca - sa0) * (sc1 - sc0) / (sa1 - sa0)
            if height > cc:
                above += 1
    # The face's chords exclude the ones along c with no bulge (loader._build_trim: the slab test needs
    # no chord that no vertical ray crosses). Such a chord joins two chord endpoints at the same a, so
    # cells at that a, within the c span of the endpoints there, also touch the boundary.
    if touched == 0:
        sb = shared_first[face]
        for q in range(shared_count[face]):
            va = shared[sb + q, 0]
            if a0 <= va and va <= a1 and c0 <= shared[sb + q, 2] and shared[sb + q, 1] <= c1:
                touched = 1
    value = wp.uint8(0)
    if s == 0 or touched != 0 or (above & 1) == 1:  # no chords: an untrimmed face keeps every cell
        value = wp.uint8(1)
    if sw != 0:
        keep[k, ic, ia] = value
    else:
        keep[k, ia, ic] = value


def _trim_tables(prim: BrepPrim, device, faces: np.ndarray | None = None) -> dict:
    """The trim chords of the prim's ``faces`` (default: all), on ``device``, for _trim_cells_batch.

    Other faces get no chords (NURBS faces can have 100k+ each; analytic shells never need them)."""
    first, end = _face_chord_ranges(prim)
    flags = np.asarray(prim.face_flags)
    count = np.where((flags & FLAG_TRIMMED) != 0, end - first, 0)
    if faces is not None:
        wanted = np.zeros(len(first), bool)
        wanted[np.asarray(faces, np.int64)] = True
        count = np.where(wanted, count, 0)
    # Gather the wanted faces' chords into a compact array, rebasing ``first``.
    seg_ids = np.arange(count.sum()) + np.repeat(first - np.cumsum(count) + count, count)
    first = np.cumsum(count) - count
    end = first + count
    seg = np.asarray(prim.segments, np.float64).reshape(-1, 4)[seg_ids]
    bulge = np.asarray(prim.segment_bulge, np.float64)[seg_ids]
    face_abs = np.zeros(len(first))  # each face's largest |chord coordinate|: the slack's scale
    if len(seg):
        np.maximum.at(face_abs, np.repeat(np.arange(len(first)), count), np.abs(seg).max(axis=1))
    sh_face, sh_a, sh_lo, sh_hi = _shared_endpoints(seg, first, end)
    sh_first = np.searchsorted(sh_face, np.arange(len(first)))
    sh_count = np.searchsorted(sh_face, np.arange(len(first)), side="right") - sh_first

    def arr(values, dtype):
        return wp.array(np.ascontiguousarray(values), dtype=dtype, device=device)

    return {
        "flags": flags,
        "first": arr(first.astype(np.int32), wp.int32),
        "count": arr(count.astype(np.int32), wp.int32),
        "face_abs": arr(face_abs, wp.float64),
        "segments": arr(seg if len(seg) else np.zeros((1, 4)), wp.float64),
        "bulges": arr(bulge if len(seg) else np.zeros(1), wp.float64),
        "shared_first": arr(sh_first.astype(np.int32), wp.int32),
        "shared_count": arr(sh_count.astype(np.int32), wp.int32),
        "shared": arr(np.stack([sh_a, sh_lo, sh_hi], 1) if len(sh_a) else np.zeros((1, 3)), wp.float64),
    }


def _trim_cells_batch(
    prim: BrepPrim, faces: np.ndarray, rects: np.ndarray, n: int, device=None, tables: dict | None = None
) -> np.ndarray:
    """(K, n, n) masks of the cells of each (face, rect) item that may hold points the face accepts
    (_trim_cells_kernel), on ``device``: MC1 has 67k pieces and planes, 20 s as NumPy per item.
    ``tables`` (_trim_tables) can be shared between calls on one prim."""
    faces = np.asarray(faces, np.int64)
    rects = np.asarray(rects, np.float64).reshape(-1, 4)
    if not len(faces):
        return np.ones((0, n, n), bool)
    device = wp.get_device(device)
    t = tables if tables is not None else _trim_tables(prim, device, faces)
    keep = wp.empty((len(faces), n, n), dtype=wp.uint8, device=device)
    wp.launch(
        _trim_cells_kernel,
        dim=len(faces) * n * n,
        inputs=[
            wp.array(faces.astype(np.int32), dtype=wp.int32, device=device),
            wp.array(rects, dtype=wp.float64, device=device),
            wp.array(((t["flags"][faces] & FLAG_SWAP) != 0).astype(np.int32), dtype=wp.int32, device=device),
            t["first"],
            t["count"],
            t["face_abs"],
            t["segments"],
            t["bulges"],
            t["shared_first"],
            t["shared_count"],
            t["shared"],
            n,
        ],
        outputs=[keep],
        device=device,
    )
    return keep.numpy().astype(bool)


def _plane_faces(prim: BrepPrim) -> np.ndarray:
    kinds = np.asarray(prim.prim_kind)
    return np.flatnonzero(kinds == PLANE) if (PLANE in ANALYTIC and PLANE_CELLS > 0) else np.zeros(0, np.int64)


def _plane_proxy(prim: BrepPrim, device=None, tables: dict | None = None) -> Proxy:
    """One on-plane triangle layer per planar face (PLANE_CELLS), trim-culled cell by cell."""
    local = _plane_faces(prim)
    if not len(local):
        return empty_proxy()
    n = PLANE_CELLS
    m = n + 1
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    a = i * m + j
    quad = np.stack([np.stack([a, a + m, a + m + 1], -1), np.stack([a, a + m + 1, a + 1], -1)], -2)  # (n, n, 2, 3)
    face = np.asarray(prim.prim_face, np.int64)[local]
    r = np.asarray(prim.prim_uv, np.float64)[local].reshape(-1, 4)
    # Grown like accept_uv's tolerance, so that rounding does not open gaps at the face's edges.
    eu = 1.0e-5 * (r[:, 1] - r[:, 0]) + 1.0e-7
    ev = 1.0e-5 * (r[:, 3] - r[:, 2]) + 1.0e-7
    g = np.stack([r[:, 0] - eu, r[:, 1] + eu, r[:, 2] - ev, r[:, 3] + ev], 1)
    keep = _trim_cells_batch(prim, face, g, n, device, tables)
    kept = keep.reshape(len(local), -1).any(axis=1)
    if not kept.any():
        return empty_proxy()
    local, face, r, g, keep = local[kept], face[kept], r[kept], g[kept], keep[kept]
    u = np.linspace(g[:, 0], g[:, 1], m, axis=1)
    v = np.linspace(g[:, 2], g[:, 3], m, axis=1)
    o = np.asarray(prim.face_origin, np.float64)[face]
    x = np.asarray(prim.face_x, np.float64)[face]
    y = np.asarray(prim.face_y, np.float64)[face]
    # points[k, iu * m + iv] = o + u[iu] x + v[iv] y
    points = (
        o[:, None, None] + u[:, :, None, None] * x[:, None, None] + v[:, None, :, None] * y[:, None, None]
    ).reshape(-1, 3)
    cells = np.broadcast_to(quad, (len(local), n, n, 2, 3))[keep]  # (cells, 2, 3), in item order
    owner = np.repeat(np.arange(len(local)), keep.reshape(len(local), -1).sum(axis=1))
    tris = (cells + (owner * m * m)[:, None, None]).reshape(-1, 3)
    tri_shell = np.repeat(owner, 2).astype(np.int32)
    return Proxy(
        points.astype(np.float32),
        tris.astype(np.uint32),
        local[tri_shell].astype(np.int32),
        np.unique(local).astype(np.int32),
        r.astype(np.float32),
        tri_shell,
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
    analytic = empty_proxy()
    if ANALYTIC:
        owners, _ = _analytic_pieces(prim)
        faces = np.asarray(prim.prim_face, np.int64)[np.concatenate([owners, _plane_faces(prim)]).astype(np.int64)]
        tables = _trim_tables(prim, device, faces)  # shared by both
        analytic = _concat(_analytic_proxy(prim, device, tables), _plane_proxy(prim, device, tables))
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
