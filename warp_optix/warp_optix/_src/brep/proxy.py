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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loader import CONE, CYLINDER, NURBS, PLANE, SPHERE, TORUS, BrepPrim


@dataclass
class Proxy:
    points: np.ndarray  # (V, 3) float32, object space
    indices: np.ndarray  # (T, 3) uint32
    tri_prim: np.ndarray  # (T,) int32: the prim's local primitive index of each triangle's patch
    tri_st: np.ndarray  # (T, 3, 2) float32: per corner, patch (s, t) for NURBS, face (u, v) otherwise
    patches: np.ndarray  # (P,) int32: local primitive indices that have shells


def _bernstein3(x: np.ndarray):
    y = 1.0 - x
    b = np.stack([y**3, 3 * x * y**2, 3 * x**2 * y, x**3], axis=-1)
    db = 3.0 * np.stack([-(y**2), y**2 - 2 * x * y, 2 * x * y - x**2, x**2], axis=-1)
    return b, db


def _eval(pw: np.ndarray, s: np.ndarray, t: np.ndarray, ders: bool = False):
    """Bicubic rational patch (4, 4, 4) at parameter arrays s, t (same shape)."""
    bu, dbu = _bernstein3(s)
    bv, dbv = _bernstein3(t)
    h = np.einsum("...i,...j,ijk->...k", bu, bv, pw)
    point = h[..., :3] / h[..., 3:]
    if not ders:
        return point
    hs = np.einsum("...i,...j,ijk->...k", dbu, bv, pw)
    ht = np.einsum("...i,...j,ijk->...k", bu, dbv, pw)
    ds = (hs[..., :3] - hs[..., 3:] * point) / h[..., 3:]
    dt = (ht[..., :3] - ht[..., 3:] * point) / h[..., 3:]
    return point, ds, dt


def _cell_triangles(gu: int, gv: int) -> np.ndarray:
    """Two triangles per grid cell over (gu + 1) x (gv + 1) vertices, row-major in u."""
    i, j = np.meshgrid(np.arange(gu), np.arange(gv), indexing="ij")
    a = (i * (gv + 1) + j).ravel()
    b, c, d = a + (gv + 1), a + 1, a + (gv + 2)
    return np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])


def _shell_from(evaluate, u_range, v_range, gu, gv, safety=2.0, oversample=4):
    """Closed shell around a surface given by ``evaluate(u, v) -> (point, du, dv)`` arrays.

    Returns (points, triangles, corner parameters (u, v)) over the given parameter ranges.
    """
    u = np.linspace(*u_range, gu + 1)
    v = np.linspace(*v_range, gv + 1)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    point, du, dv = evaluate(uu, vv)
    normal = np.cross(du, dv)
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    # Degenerate normals (poles, apexes, collapsed edges): borrow the average normal.
    average = normal.reshape(-1, 3).sum(0)
    average = average / (np.linalg.norm(average) + 1e-300)
    normal = np.where(length > 1e-12 * (length.max() + 1e-300), normal / np.maximum(length, 1e-300), average)
    # Deviation of the surface from the piecewise-linear grid, sampled densely in each cell.
    fu = np.linspace(0.0, 1.0, gu * oversample + 1)
    fv = np.linspace(0.0, 1.0, gv * oversample + 1)
    fs, ft = np.meshgrid(fu, fv, indexing="ij")
    surface = evaluate(u_range[0] + fs * (u_range[1] - u_range[0]), v_range[0] + ft * (v_range[1] - v_range[0]))[0]
    ci = np.minimum((fs * gu).astype(int), gu - 1)
    cj = np.minimum((ft * gv).astype(int), gv - 1)
    a, b = fs * gu - ci, ft * gv - cj  # local cell coordinates
    p00, p10 = point[ci, cj], point[ci + 1, cj]
    p01, p11 = point[ci, cj + 1], point[ci + 1, cj + 1]
    upper = (a >= b)[..., None]  # matches _cell_triangles' diagonal (a, d)
    a, b = a[..., None], b[..., None]
    linear = np.where(upper, p00 + a * (p10 - p00) + b * (p11 - p10), p00 + b * (p01 - p00) + a * (p11 - p01))
    deviation = float(np.linalg.norm(surface - linear, axis=-1).max())
    size = float(np.linalg.norm(point.reshape(-1, 3).max(0) - point.reshape(-1, 3).min(0)))
    offset = safety * deviation + 1e-4 * size + 1e-9
    outer = (point + offset * normal).reshape(-1, 3)
    inner = (point - offset * normal).reshape(-1, 3)
    n = (gu + 1) * (gv + 1)
    points = np.concatenate([outer, inner])
    cells = _cell_triangles(gu, gv)
    tris = [cells, cells[:, ::-1] + n]
    # Side walls along the four boundary edges, joining outer vertex k to inner vertex k + n.
    rows, cols = np.arange(gu + 1), np.arange(gv + 1)
    for edge in (rows * (gv + 1), rows * (gv + 1) + gv, cols, gu * (gv + 1) + cols):
        k0, k1 = edge[:-1], edge[1:]
        tris += [np.stack([k0, k1, k1 + n], 1), np.stack([k0, k1 + n, k0 + n], 1)]
    tris = np.concatenate(tris)
    st = np.stack([uu, vv], -1).reshape(-1, 2)
    st = np.concatenate([st, st])
    return points, tris, st[tris]


def shell(pw: np.ndarray, grid: int = 4, safety: float = 2.0, oversample: int = 4, margin: float = 0.05):
    """Closed triangle shell around one bicubic patch: (points, triangles, corner (s, t)).

    The shell covers the patch extended by ``margin`` in parameter space (the Bezier polynomial
    continues smoothly past its edges), so adjacent patches' shells overlap instead of sharing
    coincident side walls: a ray crossing a patch boundary must cross both patches' shells.
    """
    return _shell_from(
        lambda s, t: _eval(pw, s, t, ders=True),
        (-margin, 1.0 + margin),
        (-margin, 1.0 + margin),
        grid,
        grid,
        safety,
        oversample,
    )


def _analytic_evaluator(surface):
    """``evaluate(u, v) -> (point, du, dv)`` for an analytic surface, by central differences."""

    def evaluate(u, v):
        shape = u.shape
        u, v = u.ravel(), v.ravel()
        h = 1e-6 * (1.0 + np.abs(u).max() + np.abs(v).max())
        point = surface.evaluate(u, v)
        du = (surface.evaluate(u + h, v) - surface.evaluate(u - h, v)) / (2 * h)
        dv = (surface.evaluate(u, v + h) - surface.evaluate(u, v - h)) / (2 * h)
        return point.reshape(shape + (3,)), du.reshape(shape + (3,)), dv.reshape(shape + (3,))

    return evaluate


def analytic_shell(surface, face_range, margin: float = 0.02, max_angle: float = np.pi / 12):
    """Shell around an analytic face over its (u, v) range; for a plane, the exact quad itself."""
    umin, umax, vmin, vmax = face_range
    mu, mv = margin * (umax - umin) + 1e-9, margin * (vmax - vmin) + 1e-9
    u_range, v_range = (umin - mu, umax + mu), (vmin - mv, vmax + mv)
    if surface.kind == PLANE:
        # Planar: two triangles lie exactly on the surface (no shell, exact t).
        corners = np.array([[u, v] for u in u_range for v in v_range])
        points = surface.evaluate(corners[:, 0], corners[:, 1])
        tris = np.array([[0, 2, 3], [0, 3, 1]])
        return points, tris, corners[tris]
    # Angular parameters get a segment per max_angle; linear ones stay single cells (v of
    # cylinders and cones is straight; a sphere's or torus's v is angular too).
    gu = max(1, int(np.ceil((u_range[1] - u_range[0]) / max_angle)))
    angular_v = surface.kind in (SPHERE, TORUS)
    gv = max(1, int(np.ceil((v_range[1] - v_range[0]) / max_angle))) if angular_v else 1
    return _shell_from(_analytic_evaluator(surface), u_range, v_range, gu, gv)


def _face_surface(prim: BrepPrim, face: int):
    """The face's analytic surface, rebuilt from the per-face arrays (``surfaces`` may be dropped
    by the parallel loader)."""
    from .loader import Surface  # noqa: PLC0415

    return Surface(
        int(prim.face_kind[face]),
        prim.face_origin[face],
        prim.face_x[face],
        prim.face_y[face],
        prim.face_axis[face],
        (float(prim.face_params[face][0]), float(prim.face_params[face][1])),
    )


def empty_proxy() -> Proxy:
    return Proxy(
        np.zeros((0, 3), np.float32),
        np.zeros((0, 3), np.uint32),
        np.zeros(0, np.int32),
        np.zeros((0, 3, 2), np.float32),
        np.zeros(0, np.int32),
    )


def build_proxy(prim: BrepPrim, grid: int = 4, analytic: bool = False) -> Proxy:
    """Triangle proxies: shells for bicubic NURBS patches and, with ``analytic``, exact quads for
    planar faces and shells for cylinders, cones and spheres. Other primitives keep AABBs.

    ``analytic`` is off by default: an any-hit call costs about as much as an intersection call,
    and a closed-form face is a single cheap call on its AABB, while its shell takes one any-hit
    per layer crossed (and a planar face's quad spans its whole UV range). Measured slower on
    small parts (2x) and on a dense assembly (3x). NURBS shells win (3-11x): they replace several
    overlapping patch AABBs, each needing Newton, with one well-seeded refinement.
    """
    shelled = (prim.prim_kind == NURBS) & np.all(prim.prim_degree == 3, axis=1)
    if analytic:
        shelled |= np.isin(prim.prim_kind, (PLANE, CYLINDER, CONE, SPHERE))
    patches = np.nonzero(shelled)[0].astype(np.int32)
    points, indices, tri_prim, tri_st = [], [], [], []
    base = 0
    for p in patches:
        if prim.prim_kind[p] == NURBS:
            pw = prim.cvs[prim.prim_cv[p] : prim.prim_cv[p] + 16].reshape(4, 4, 4).astype(np.float64)
            pts, tris, st = shell(pw, grid)
        else:
            face = prim.prim_face[p]
            pts, tris, st = analytic_shell(_face_surface(prim, face), prim.face_range[face])
        points.append(pts)
        indices.append(tris + base)
        tri_prim.append(np.full(len(tris), p, np.int32))
        tri_st.append(st)
        base += len(pts)
    if not len(patches):
        return empty_proxy()
    return Proxy(
        np.concatenate(points).astype(np.float32),
        np.concatenate(indices).astype(np.uint32),
        np.concatenate(tri_prim),
        np.concatenate(tri_st).astype(np.float32),
        patches,
    )
