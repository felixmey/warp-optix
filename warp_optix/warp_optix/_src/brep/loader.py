# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read OpenUSD BRep prims (``BrepArray``, omniSolid schema) into flat arrays for GPU ray tracing.

Only ``pxr`` and NumPy are used: the BRep is read straight from the schema attributes
(https://nvidia-omniverse.github.io/usd-brep/), so tracing does not depend on a modeling kernel.

What the GPU needs per face:

* the surface: an analytic frame (plane, cylinder, cone, sphere, torus), or a NURBS surface
  decomposed into rational Bezier sub-patches with conservative AABBs;
* the trim: the face's loops as closed UV polylines, bucketed into slabs for a point-in-polygon
  test. UV trim curves (``brep:curveUv``) are optional in the schema, so edges without one are
  sampled in 3D and projected onto the surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, overload

import numpy as np

PLANE, CYLINDER, CONE, SPHERE, TORUS, NURBS = range(6)
KIND_NAMES = ("plane", "cylinder", "cone", "sphere", "torus", "nurbs")
_SURFACE_KINDS = {
    "BrepSurfacePlaneAPI": PLANE,
    "BrepSurfaceCylinderAPI": CYLINDER,
    "BrepSurfaceConeAPI": CONE,
    "BrepSurfaceSphereAPI": SPHERE,
    "BrepSurfaceTorusAPI": TORUS,
    "BrepSurfaceNurbAPI": NURBS,
}
_ANALYTIC_PREFIX = {
    CYLINDER: "brep:surface:cylinder:",
    CONE: "brep:surface:cone:",
    SPHERE: "brep:surface:sphere:",
    TORUS: "brep:surface:torus:",
    PLANE: "brep:surface:plane:",
}

# face_flags bits
FLAG_FLIP = 1  # the outward (solid -> void) side is opposite the surface normal Su x Sv
FLAG_TRIMMED = 2  # the face has trim segments to test
FLAG_SWAP = 4  # the crossing test runs along +u instead of +v (loops wrap around v)
FLAG_SOLID = 8  # exactly one side of the face bounds a solid region (else a sheet face)
FLAG_CLOSED = 16  # a solid face of a prim whose solid faces close up (each edge used twice, see _solids_closed)

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------------------------
# B-spline helpers (vectorized over parameters)
# ---------------------------------------------------------------------------------------------


@overload
def _basis(
    knots: np.ndarray, p: int, n: int, t: np.ndarray, ders: Literal[False] = False
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def _basis(
    knots: np.ndarray, p: int, n: int, t: np.ndarray, ders: Literal[True]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def _basis(knots: np.ndarray, p: int, n: int, t: np.ndarray, ders: bool = False):
    """B-spline basis (and first derivatives) at ``t``: returns span (N,), N (N, p+1)[, dN]."""
    t = np.clip(np.asarray(t, dtype=np.float64), knots[p], knots[n])
    span = np.searchsorted(knots, t, side="right") - 1
    span = np.clip(span, p, n - 1)
    m = t.shape[0]
    basis = np.zeros((m, p + 1))
    basis[:, 0] = 1.0
    left = np.zeros((m, p + 1))
    right = np.zeros((m, p + 1))
    lower = None
    for j in range(1, p + 1):
        left[:, j] = t - knots[span + 1 - j]
        right[:, j] = knots[span + j] - t
        saved = np.zeros(m)
        if j == p:
            lower = basis[:, :p].copy()
        for r in range(j):
            denom = right[:, r + 1] + left[:, j - r]
            temp = np.divide(basis[:, r], denom, out=np.zeros(m), where=denom != 0.0)
            basis[:, r] = saved + right[:, r + 1] * temp
            saved = left[:, j - r] * temp
        basis[:, j] = saved
    if not ders:
        return span, basis
    deriv = np.zeros((m, p + 1))
    if p > 0:
        if lower is None:
            lower = np.ones((m, 1))
        for r in range(p + 1):
            if r > 0:
                denom = knots[span + r] - knots[span + r - p]
                deriv[:, r] += np.divide(lower[:, r - 1], denom, out=np.zeros(m), where=denom != 0.0)
            if r < p:
                denom = knots[span + r + 1] - knots[span + r + 1 - p]
                deriv[:, r] -= np.divide(lower[:, r], denom, out=np.zeros(m), where=denom != 0.0)
        deriv *= p
    return span, basis, deriv


@dataclass
class NurbsCurve:
    points: np.ndarray  # (n, dim) Euclidean
    weights: np.ndarray  # (n,)
    knots: np.ndarray
    order: int

    @property
    def domain(self) -> tuple[float, float]:
        p = self.order - 1
        return float(self.knots[p]), float(self.knots[len(self.points)])

    def spans(self) -> int:
        a, b = self.domain
        k = self.knots[(self.knots >= a) & (self.knots <= b)]
        return max(1, int(np.count_nonzero(np.diff(k) > 0.0)))

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        p, n = self.order - 1, len(self.points)
        span, basis = _basis(self.knots, p, n, t)
        idx = span[:, None] - p + np.arange(p + 1)[None, :]
        w = self.weights[idx] * basis
        return np.einsum("mk,mkd->md", w, self.points[idx]) / w.sum(axis=1, keepdims=True)


@dataclass
class NurbsSurface:
    points: np.ndarray  # (nu, nv, 3), row-major with U as rows
    weights: np.ndarray  # (nu, nv)
    uknots: np.ndarray
    vknots: np.ndarray
    uorder: int
    vorder: int

    @property
    def domain(self) -> tuple[float, float, float, float]:
        p, q = self.uorder - 1, self.vorder - 1
        nu, nv = self.weights.shape
        return float(self.uknots[p]), float(self.uknots[nu]), float(self.vknots[q]), float(self.vknots[nv])

    @overload
    def evaluate(self, u: np.ndarray, v: np.ndarray, ders: Literal[False] = False) -> np.ndarray: ...
    @overload
    def evaluate(
        self, u: np.ndarray, v: np.ndarray, ders: Literal[True]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def evaluate(self, u: np.ndarray, v: np.ndarray, ders: bool = False):
        p, q = self.uorder - 1, self.vorder - 1
        nu, nv = self.weights.shape
        su, bu, dbu = _basis(self.uknots, p, nu, u, ders=True)
        sv, bv, dbv = _basis(self.vknots, q, nv, v, ders=True)
        iu = su[:, None] - p + np.arange(p + 1)[None, :]
        iv = sv[:, None] - q + np.arange(q + 1)[None, :]
        w = self.weights[iu[:, :, None], iv[:, None, :]]  # (m, p+1, q+1)
        pw = self.points[iu[:, :, None], iv[:, None, :]] * w[..., None]
        a = np.einsum("mi,mj,mijd->md", bu, bv, pw)
        ww = np.einsum("mi,mj,mij->m", bu, bv, w)[:, None]
        point = a / ww
        if not ders:
            return point
        au = np.einsum("mi,mj,mijd->md", dbu, bv, pw)
        av = np.einsum("mi,mj,mijd->md", bu, dbv, pw)
        wu = np.einsum("mi,mj,mij->m", dbu, bv, w)[:, None]
        wv = np.einsum("mi,mj,mij->m", bu, dbv, w)[:, None]
        return point, (au - wu * point) / ww, (av - wv * point) / ww


# ---------------------------------------------------------------------------------------------
# NURBS surface -> rational Bezier patches -> flat sub-patches
# ---------------------------------------------------------------------------------------------


def _insert_knot(pw: np.ndarray, knots: np.ndarray, p: int, u: float):
    """Boehm insertion of ``u`` once along axis 0 of homogeneous points ``pw``."""
    n = pw.shape[0]
    k = int(np.searchsorted(knots, u, side="right") - 1)
    k = min(max(k, p), n - 1)
    new = np.empty((n + 1,) + pw.shape[1:])
    new[: k - p + 1] = pw[: k - p + 1]
    for i in range(k - p + 1, k + 1):
        denom = knots[i + p] - knots[i]
        a = 0.0 if denom == 0.0 else (u - knots[i]) / denom
        new[i] = (1.0 - a) * pw[i - 1] + a * pw[i]
    new[k + 1 :] = pw[k:]
    return new, np.insert(knots, k + 1, u)


def _bezier_split_axis0(pw: np.ndarray, knots: np.ndarray, p: int):
    """Split along axis 0 into Bezier segments: list of ((p+1, ...) points, (t0, t1))."""
    n = pw.shape[0]
    a, b = knots[p], knots[n]
    for value in np.unique(knots[(knots >= a) & (knots <= b)]):
        mult = int(np.count_nonzero(knots == value))
        for _ in range(max(0, p - mult)):
            pw, knots = _insert_knot(pw, knots, p, float(value))
    n = pw.shape[0]
    segments = []
    for i in range(p, n):
        if knots[i] < knots[i + 1] and knots[i] >= a and knots[i + 1] <= b:
            segments.append((pw[i - p : i + 1], (float(knots[i]), float(knots[i + 1]))))
    return segments


def _split_bezier(pw: np.ndarray, axis: int):
    """de Casteljau split at 0.5 along ``axis`` of a (p+1, q+1, 4) homogeneous patch."""
    work = np.moveaxis(pw, axis, 0).copy()
    deg = work.shape[0] - 1
    left = [work[0].copy()]
    right = [work[deg].copy()]
    for r in range(1, deg + 1):
        work[: deg - r + 1] = 0.5 * (work[: deg - r + 1] + work[1 : deg - r + 2])
        left.append(work[0].copy())
        right.append(work[deg - r].copy())
    return np.moveaxis(np.stack(left), 0, axis), np.moveaxis(np.stack(right[::-1]), 0, axis)


def _elevate(pw: np.ndarray, axis: int) -> np.ndarray:
    """Raise a Bezier patch's degree by one along ``axis``; the surface is unchanged."""
    p = np.moveaxis(pw, axis, 0)
    n = p.shape[0] - 1
    a = (np.arange(1, n + 1) / (n + 1)).reshape((-1,) + (1,) * (p.ndim - 1))
    q = np.concatenate([p[:1], a * p[:-1] + (1.0 - a) * p[1:], p[-1:]], axis=0)
    return np.moveaxis(q, 0, axis)


def to_bicubic(pw: np.ndarray) -> np.ndarray:
    """Degree-elevate a (p + 1, q + 1, 4) patch with p, q <= 3 to bicubic, for the fast device path."""
    while pw.shape[0] < 4:
        pw = _elevate(pw, 0)
    while pw.shape[1] < 4:
        pw = _elevate(pw, 1)
    return pw


def _flatness(points: np.ndarray, axis: int) -> float:
    """Max deviation of control rows along ``axis`` from the chord between their end points."""
    pts = np.moveaxis(points, axis, 0)
    deg = pts.shape[0] - 1
    if deg < 1:
        return 0.0
    s = np.linspace(0.0, 1.0, deg + 1)[:, None, None]
    chord = (1.0 - s) * pts[0][None] + s * pts[deg][None]
    return float(np.max(np.linalg.norm(pts - chord, axis=-1)))


def bezier_subpatches(surface: NurbsSurface, tol: float, max_depth: int = 6):
    """Decompose into rational Bezier patches, split until flat: yields (pw, (u0, u1, v0, v1))."""
    w = surface.weights[..., None]
    pw = np.concatenate([surface.points * w, w], axis=-1)
    p, q = surface.uorder - 1, surface.vorder - 1
    out = []
    for urow, (u0, u1) in _bezier_split_axis0(pw, surface.uknots.copy(), p):
        for vpatch, (v0, v1) in _bezier_split_axis0(np.moveaxis(urow, 1, 0), surface.vknots.copy(), q):
            stack = [(np.moveaxis(vpatch, 0, 1), (u0, u1, v0, v1), 0)]
            while stack:
                patch, rect, depth = stack.pop()
                euclid = patch[..., :3] / patch[..., 3:4]
                du, dv = _flatness(euclid, 0), _flatness(euclid, 1)
                if depth >= max_depth or max(du, dv) <= tol:
                    out.append((patch, rect))
                    continue
                a0, a1, b0, b1 = rect
                if du >= dv:
                    lo, hi = _split_bezier(patch, 0)
                    mid = 0.5 * (a0 + a1)
                    stack += [(lo, (a0, mid, b0, b1), depth + 1), (hi, (mid, a1, b0, b1), depth + 1)]
                else:
                    lo, hi = _split_bezier(patch, 1)
                    mid = 0.5 * (b0 + b1)
                    stack += [(lo, (a0, a1, b0, mid), depth + 1), (hi, (a0, a1, mid, b1), depth + 1)]
    return out


# ---------------------------------------------------------------------------------------------
# Surfaces: evaluation, inversion, periodicity, bounds
# ---------------------------------------------------------------------------------------------


@dataclass
class Surface:
    kind: int
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    x: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    y: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    params: tuple[float, float] = (0.0, 0.0)  # cylinder/sphere: (r, 0); cone: (r, tan); torus: (R, r)
    nurbs: NurbsSurface | None = None

    def periods(self) -> tuple[float, float]:
        if self.kind in (CYLINDER, CONE, SPHERE):
            return TWO_PI, 0.0
        if self.kind == TORUS:
            return TWO_PI, TWO_PI
        if self.kind == NURBS:
            return self._nurbs_closed_periods()
        return 0.0, 0.0

    def _nurbs_closed_periods(self) -> tuple[float, float]:
        s = self.nurbs
        u0, u1, v0, v1 = s.domain
        size = float(np.ptp(s.points.reshape(-1, 3), axis=0).max()) or 1.0
        t = np.linspace(0.0, 1.0, 9)
        pu = 0.0
        a = s.evaluate(np.full(9, u0), v0 + t * (v1 - v0))
        b = s.evaluate(np.full(9, u1), v0 + t * (v1 - v0))
        if np.max(np.linalg.norm(a - b, axis=1)) < 1e-6 * size:
            pu = u1 - u0
        pv = 0.0
        a = s.evaluate(u0 + t * (u1 - u0), np.full(9, v0))
        b = s.evaluate(u0 + t * (u1 - u0), np.full(9, v1))
        if np.max(np.linalg.norm(a - b, axis=1)) < 1e-6 * size:
            pv = v1 - v0
        return pu, pv

    def evaluate(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        if self.kind == NURBS:
            return self.nurbs.evaluate(u, v)
        o, x, y, a = self.origin, self.x, self.y, self.axis
        r0, r1 = self.params
        if self.kind == PLANE:
            return o + u[:, None] * x + v[:, None] * y
        if self.kind == CYLINDER:
            return o + r0 * (np.cos(u)[:, None] * x + np.sin(u)[:, None] * y) + v[:, None] * a
        if self.kind == CONE:
            rad = (r0 + v * r1)[:, None]
            return o + rad * (np.cos(u)[:, None] * x + np.sin(u)[:, None] * y) + v[:, None] * a
        if self.kind == SPHERE:
            cv = np.cos(v)[:, None]
            return o + r0 * (cv * np.cos(u)[:, None] * x + cv * np.sin(u)[:, None] * y + np.sin(v)[:, None] * a)
        rad = (r0 + r1 * np.cos(v))[:, None]
        return o + rad * (np.cos(u)[:, None] * x + np.sin(u)[:, None] * y) + r1 * np.sin(v)[:, None] * a

    def invert(self, points: np.ndarray, guess: np.ndarray | None = None) -> np.ndarray:
        """Map 3D points (on or near the surface) to (u, v); angles in (-pi, pi]."""
        if self.kind == NURBS:
            return self._invert_nurbs(points, guess)
        d = points - self.origin
        lx, ly, lz = d @ self.x, d @ self.y, d @ self.axis
        r0, r1 = self.params
        if self.kind == PLANE:
            return np.stack([lx, ly], axis=1)
        u = np.arctan2(ly, lx)
        if self.kind == CYLINDER:
            return np.stack([u, lz], axis=1)
        if self.kind == CONE:
            u = np.where(r0 + lz * r1 < 0.0, np.arctan2(-ly, -lx), u)
            # u is undefined at the apex; edge samples land there only up to the model's
            # tolerance, so treat a radius below 1e-6 of the surface size as the apex.
            apex = np.hypot(lx, ly) < 1e-6 * (abs(r0) + np.abs(lz * r1) + 1e-30)
            return np.stack([np.where(apex, np.nan, u), lz], axis=1)
        if self.kind == SPHERE:
            v = np.arcsin(np.clip(lz / r0, -1.0, 1.0))
            pole = np.hypot(lx, ly) < 1e-6 * max(r0, 1e-30)
            return np.stack([np.where(pole, np.nan, u), v], axis=1)
        return np.stack([u, np.arctan2(lz, np.hypot(lx, ly) - r0)], axis=1)

    def _invert_nurbs(self, points: np.ndarray, guess: np.ndarray | None) -> np.ndarray:
        s = self.nurbs
        u0, u1, v0, v1 = s.domain
        if guess is None:
            gu, gv = np.meshgrid(np.linspace(u0, u1, 48), np.linspace(v0, v1, 48), indexing="ij")
            grid = s.evaluate(gu.ravel(), gv.ravel())
            nearest = np.argmin(((points[:, None, :] - grid[None, :, :]) ** 2).sum(-1), axis=1)
            uv = np.stack([gu.ravel()[nearest], gv.ravel()[nearest]], axis=1)
        else:
            uv = guess.copy()
        for _ in range(30):  # Gauss-Newton on |S(u, v) - P|^2
            pt, su, sv = s.evaluate(uv[:, 0], uv[:, 1], ders=True)
            r = points - pt
            a, b, c = (su * su).sum(1), (su * sv).sum(1), (sv * sv).sum(1)
            gu, gv = (su * r).sum(1), (sv * r).sum(1)
            det = a * c - b * b
            det = np.where(np.abs(det) < 1e-30, 1e-30, det)
            uv[:, 0] = np.clip(uv[:, 0] + (c * gu - b * gv) / det, u0, u1)
            uv[:, 1] = np.clip(uv[:, 1] + (a * gv - b * gu) / det, v0, v1)
        return uv

    def bounds(self, rng: tuple[float, float, float, float], n: int = 24) -> tuple[np.ndarray, np.ndarray]:
        """Conservative AABB of the surface over the UV rectangle ``rng``."""
        u0, u1, v0, v1 = rng
        gu, gv = np.meshgrid(np.linspace(u0, u1, n + 1), np.linspace(v0, v1, n + 1), indexing="ij")
        pts = self.evaluate(gu.ravel(), gv.ravel())
        r0, r1 = self.params
        du, dv = (u1 - u0) / n, (v1 - v0) / n

        def sag(radius: float, step: float) -> float:
            return abs(radius) * (1.0 - math.cos(min(abs(step), math.pi) * 0.5))

        pad = 0.0
        if self.kind in (CYLINDER, SPHERE):
            pad = sag(r0, du) + (sag(r0, dv) if self.kind == SPHERE else 0.0)
        elif self.kind == CONE:
            pad = sag(max(abs(r0 + v0 * r1), abs(r0 + v1 * r1)), du)
        elif self.kind == TORUS:
            pad = sag(r0 + r1, du) + sag(r1, dv)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        pad += 1e-6 * float(np.max(hi - lo) + 1e-12)
        return lo - pad, hi + pad


# ---------------------------------------------------------------------------------------------
# Reading the schema
# ---------------------------------------------------------------------------------------------


def _attr(prim, name: str, dtype=np.float64, width: int = 0) -> np.ndarray:
    attribute = prim.GetAttribute(name)
    value = attribute.Get() if attribute and attribute.HasAuthoredValue() else None
    if value is None or len(value) == 0:
        return np.zeros((0, width) if width else (0,), dtype=dtype)
    if dtype is str:
        return np.array([str(v) for v in value], dtype=object)
    return np.asarray(value, dtype=dtype)


def _frame(axis: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (x, y, axis) from axis and reference direction; also for (N, 3) batches."""
    a = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    x = ref - np.sum(ref * a, axis=-1, keepdims=True) * a
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    return x, np.cross(a, x), a


class _Packed:
    """Iterate packed per-object NURBS records (curves or surfaces) in order of use."""

    def __init__(self, counts: np.ndarray, orders: np.ndarray, cvs: np.ndarray, knots: np.ndarray, weights: np.ndarray):
        self.counts, self.orders, self.cvs, self.knots, self.weights = counts, orders, cvs, knots, weights
        self.cv_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.knot_offsets = np.concatenate([[0], np.cumsum(counts + orders)]).astype(np.int64)

    def curve(self, i: int) -> NurbsCurve | None:
        n, k = int(self.counts[i]), int(self.orders[i])
        if n == 0 or k == 0:
            return None
        a = self.cv_offsets[i]
        w = self.weights[a : a + n] if len(self.weights) == len(self.cvs) else np.ones(n)
        kn = self.knots[self.knot_offsets[i] : self.knot_offsets[i] + n + k]
        return NurbsCurve(self.cvs[a : a + n], np.asarray(w, dtype=np.float64), np.asarray(kn), k)


@dataclass
class BrepPrim:
    """One BrepArray prim flattened for the GPU (object space)."""

    path: str
    world: np.ndarray  # 4x4 object-to-world, column-vector convention
    # faces
    face_kind: np.ndarray
    face_origin: np.ndarray
    face_x: np.ndarray
    face_y: np.ndarray
    face_axis: np.ndarray
    face_params: np.ndarray
    face_range: np.ndarray  # (F, 4) umin, umax, vmin, vmax
    face_period: np.ndarray  # (F, 2)
    face_flags: np.ndarray
    face_slab: np.ndarray  # (F, 2) slab base, slab count
    face_slab_map: np.ndarray  # (F, 2) lo, 1 / width  along the slab axis
    slab_start: np.ndarray
    slab_segments: np.ndarray
    segments: np.ndarray  # (S, 4) a.u a.v b.u b.v (swapped for FLAG_SWAP faces)
    segment_bulge: np.ndarray  # (S,) signed chord-to-curve midpoint distance (see _build_trim)
    # primitives
    prim_kind: np.ndarray
    prim_face: np.ndarray
    prim_bounds: np.ndarray  # (P, 6)
    prim_cv: np.ndarray
    prim_degree: np.ndarray  # (P, 2)
    prim_uv: np.ndarray  # (P, 4)
    prim_tol: np.ndarray
    cvs: np.ndarray  # (C, 4) homogeneous
    surfaces: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    # Trim cell grids (trim_grid.build_trim_grids), built on first use and cached here.
    trim_grids: tuple | None = field(default=None, repr=False, compare=False)


def _sample_edge_3d(prim_data, edge: int) -> np.ndarray:
    density = prim_data["samples_per_span"]
    kind, local, (t0, t1) = prim_data["edge_kind"][edge], prim_data["edge_local"][edge], prim_data["edge_range"][edge]
    if kind == "BrepCurve3dNurbAPI":
        curve = prim_data["edge_nurbs"].curve(local)
        n = max(8, density * curve.spans() * (curve.order - 1))
        return curve.evaluate(np.linspace(t0, t1, 2 * n + 1))
    if kind == "BrepCurve3dLineAPI":
        o, d = prim_data["line"][0][local], prim_data["line"][1][local]
        t = np.linspace(t0, t1, 33)
        return o + t[:, None] * d
    n = max(8, int(math.ceil(abs(t1 - t0) / TWO_PI * 8 * density)))
    t = np.linspace(t0, t1, 2 * n + 1)
    if kind == "BrepCurve3dCircleAPI":
        c, a, ref, r = (arr[local] for arr in prim_data["circle"])
        x, y, _ = _frame(a, ref)
        return c + r * (np.cos(t)[:, None] * x + np.sin(t)[:, None] * y)
    if kind == "BrepCurve3dEllipseAPI":
        c, a, ref, rx, ry = (arr[local] for arr in prim_data["ellipse"])
        x, y, _ = _frame(a, ref)
        return c + rx * np.cos(t)[:, None] * x + ry * np.sin(t)[:, None] * y
    raise ValueError(f"unsupported edge curve type {kind!r}")


def _unwrap(values: np.ndarray, period: float) -> np.ndarray:
    if period <= 0.0 or len(values) < 2:
        return values
    out = values.copy()
    jumps = np.diff(values)
    out[1:] = values[1:] - period * np.cumsum(np.round(jumps / period))
    return out


def _fill_nan(column: np.ndarray) -> np.ndarray:
    bad = np.isnan(column)
    if not bad.any() or bad.all():
        return column
    idx = np.arange(len(column))
    good = idx[~bad]
    nearest = good[np.clip(np.searchsorted(good, idx), 0, len(good) - 1)]
    prev = good[np.clip(np.searchsorted(good, idx) - 1, 0, len(good) - 1)]
    pick = np.where(np.abs(prev - idx) < np.abs(nearest - idx), prev, nearest)
    out = column.copy()
    out[bad] = column[pick[bad]]
    return out


def _solids_closed(kinds, solid, loop_off, eu_off, eu_edge) -> bool:
    """Whether the solid faces of a prim close up: every edge of their trim loops is used an even
    number of times (twice: once by each face it joins, or twice by the face whose seam it is).
    Data can claim a solid region yet leave it open (rods without end caps), where rays get inside
    without crossing a face. A face without loops has no listed edges; only spheres and tori close
    up then (poles, two periods), any other such face leaves the prim's solids unverified."""
    faces = np.flatnonzero(solid)
    if len(faces) == 0:
        return False
    uses = []
    for f in faces.tolist():
        first, last = int(loop_off[f]), int(loop_off[f + 1])
        if first == last:
            if kinds[f] not in (SPHERE, TORUS):
                return False
            continue
        uses.append(eu_edge[int(eu_off[first]) : int(eu_off[last])])
    if not uses:
        return True
    counts = np.bincount(np.concatenate(uses))
    used = counts[counts > 0]
    return bool(np.all(used % 2 == 0))


def _face_loops_uv(prim_data, surface: Surface, face_rng, loop_list, period) -> tuple[list, list]:
    """Return (closed polylines, open polylines) in UV for a face's loops.

    Polylines are dense: even samples are vertices, odd samples the curve between them, which
    gives each segment its bulge (``_build_trim``).
    """
    pu, pv = period
    umin, umax, vmin, vmax = face_rng
    eu_edge = prim_data["edgeuse_edge"]
    eu_same = prim_data["edgeuse_same"]
    uv_curves = prim_data["uv_curves"]
    face_edges = [int(eu_edge[e]) for loop in loop_list for e in loop]
    seams = {e for e in face_edges if face_edges.count(e) > 1}
    closed, open_ = [], []
    stats = prim_data["stats"]
    placed_seams: dict[int, float] = {}
    for loop in loop_list:
        pieces = []
        prev_end = None
        prev_degenerate = False
        for eu in loop:
            curve = uv_curves.curve(eu) if uv_curves is not None else None
            degenerate = (False, False)  # (starts, ends) at an apex or pole, where u is undefined
            if curve is not None and curve.order == 2 and not (pu or pv):
                # A degree-1 pcurve is its control polygon. Not on periodic faces: unwrapping across
                # the seam needs steps well below the period, and one segment may span all of it.
                corners = np.array(curve.points, dtype=np.float64)
                uv = np.empty((2 * len(corners) - 1, 2))
                uv[0::2] = corners
                uv[1::2] = 0.5 * (corners[:-1] + corners[1:])
                stats["pcurve_edgeuses"] = stats.get("pcurve_edgeuses", 0) + 1
            elif curve is not None:
                n = max(8, prim_data["samples_per_span"] * curve.spans() * (curve.order - 1))
                a, b = curve.domain
                uv = curve.evaluate(np.linspace(a, b, 2 * n + 1))
                stats["pcurve_edgeuses"] = stats.get("pcurve_edgeuses", 0) + 1
            else:
                pts = _sample_edge_3d(prim_data, int(eu_edge[eu]))
                if not eu_same[eu]:
                    pts = pts[::-1]
                guess = None
                if surface.kind == NURBS and prev_end is not None:
                    guess = np.repeat(prev_end[None, :], len(pts), axis=0)
                    uv = surface.invert(pts[:1], guess[:1])
                    guess = None
                uv = surface.invert(pts, guess)
                if surface.kind == NURBS:  # reject bad local minima from the continuation seed
                    err = np.linalg.norm(surface.evaluate(uv[:, 0], uv[:, 1]) - pts, axis=1)
                    if err.max() > 1e-4 * (np.ptp(pts, axis=0).max() + 1e-9):
                        uv = surface.invert(pts, None)
                degenerate = (bool(np.isnan(uv[0, 0])), bool(np.isnan(uv[-1, 0])))
                uv[:, 0] = _fill_nan(uv[:, 0])
                stats["projected_edgeuses"] = stats.get("projected_edgeuses", 0) + 1
            # Make each piece continuous across the periodic dims.
            if pu:
                uv[:, 0] = _unwrap(uv[:, 0], pu)
            if pv:
                uv[:, 1] = _unwrap(uv[:, 1], pv)
            # Chain each piece onto the previous one's end. Across an apex or pole u is arbitrary
            # (the two sides of a half cone meet there half a period apart), so there the piece is
            # placed by overlap with the face's u range instead.
            if prev_end is not None:
                if pu and (degenerate[0] or prev_degenerate):
                    center = 0.5 * (uv[:, 0].min() + uv[:, 0].max())
                    uv[:, 0] += pu * np.round((0.5 * (umin + umax) - center) / pu)
                elif pu:
                    uv[:, 0] += pu * np.round((prev_end[0] - uv[0, 0]) / pu)
                if pv:
                    uv[:, 1] += pv * np.round((prev_end[1] - uv[0, 1]) / pv)
            edge = int(eu_edge[eu])
            if edge in seams:
                # A seam is used twice by its face, once at each end of the periodic range.
                # Continuity places both uses correctly unless they follow each other (through a
                # pole), in which case they land on the same branch: move this one a period away.
                # Either direction works, as the trim test also sees the periodic copies.
                dim = 0 if (pu and np.ptp(uv[:, 0]) < 1e-6 * pu) else 1
                period_d = (pu, pv)[dim]
                other = placed_seams.get(edge)
                if period_d and other is not None and abs(uv[0, dim] - other) < 0.5 * period_d:
                    uv[:, dim] += period_d
                placed_seams[edge] = float(uv[0, dim])
            if prev_end is not None:
                gap = np.abs(uv[0] - prev_end)
                stats["max_uv_gap"] = max(stats.get("max_uv_gap", 0.0), float(np.max(gap)))
            pieces.append(uv)
            prev_end = uv[-1]
            prev_degenerate = degenerate[1]
        if not pieces:
            continue
        # Pieces are dense (odd samples are segment midpoints); join them with midpoints.
        dense = [pieces[0]]
        for prev, cur in zip(pieces, pieces[1:]):
            dense += [0.5 * (prev[-1:] + cur[:1]), cur]
        poly = np.concatenate(dense, axis=0)
        # Fill pole samples (sphere) from their neighbours along the whole loop.
        poly[:, 0] = _fill_nan(poly[:, 0])
        delta = poly[-1] - poly[0]
        is_open = (pu and abs(delta[0]) > 0.5 * pu) or (pv and abs(delta[1]) > 0.5 * pv)
        if is_open:
            # A loop that wraps only along a degenerate line (sphere pole, cone apex) is closed
            # through that line: the UV segment between its ends maps to a single 3D point.
            mid = 0.5 * (poly[0] + poly[-1])
            ends = surface.evaluate(np.array([poly[0, 0], mid[0]]), np.array([poly[0, 1], mid[1]]))
            scale = float(np.ptp(surface.evaluate(poly[:, 0], poly[:, 1]), axis=0).max()) + 1e-30
            is_open = bool(np.linalg.norm(ends[0] - ends[1]) > 1e-7 * scale)
        # Shift the loop so it overlaps the face range.
        center = 0.5 * (poly.min(axis=0) + poly.max(axis=0))
        if pu:
            poly[:, 0] += pu * np.round((0.5 * (umin + umax) - center[0]) / pu)
        if pv:
            poly[:, 1] += pv * np.round((0.5 * (vmin + vmax) - center[1]) / pv)
        (open_ if is_open else closed).append(poly)
    return closed, open_


def _trims_range(segs: np.ndarray, bulges: np.ndarray, face_rng, swap: bool) -> bool:
    """Whether every trim chord is straight and runs along one side of the face's UV range: the
    trims then bound the range itself (a rectangular face, or a full turn of a periodic one), and
    the range test alone decides, without a trim lookup per hit (on a dense assembly, 70% of the
    faces: 14% less BRep data)."""
    a_lo, a_hi, c_lo, c_hi = face_rng[[2, 3, 0, 1]] if swap else face_rng
    ea = 1.0e-6 * max(a_hi - a_lo, 1.0e-12)
    ec = 1.0e-6 * max(c_hi - c_lo, 1.0e-12)
    a, c = segs[:, [0, 2]], segs[:, [1, 3]]
    on_side = (
        (np.abs(a - a_lo) <= ea).all(axis=1)
        | (np.abs(a - a_hi) <= ea).all(axis=1)
        | (np.abs(c - c_lo) <= ec).all(axis=1)
        | (np.abs(c - c_hi) <= ec).all(axis=1)
    )
    return bool(on_side.all() and (np.abs(bulges) <= min(ea, ec)).all())


def _build_trim(polys_closed, polys_open, face_rng, period, swap: bool, num_slabs: int):
    """Segments (with periodic copies), their bulges and slab buckets for one face.

    A segment's bulge is the signed distance (positive to its left) from the chord's midpoint to
    the curve's: the trim test treats the region between a chord and its parabolic arc as on the
    other side, so trim boundaries follow curves to third order (no cracks along curved edges).
    """
    pu, pv = period
    segs, bulges = [], []
    shifts_u = (-pu, 0.0, pu) if pu else (0.0,)
    shifts_v = (-pv, 0.0, pv) if pv else (0.0,)
    for polys, close in ((polys_closed, True), (polys_open, False)):
        for poly in polys:
            ring = np.concatenate([poly, 0.5 * (poly[-1:] + poly[:1]), poly[:1]]) if close else poly
            vertices, middles = ring[0::2], ring[1::2]
            start, end = vertices[:-1], vertices[1:]
            chord = end - start
            length = np.linalg.norm(chord, axis=1)
            offset = middles - start
            bulge = np.divide(
                chord[:, 0] * offset[:, 1] - chord[:, 1] * offset[:, 0],
                length,
                out=np.zeros(len(length)),
                where=length > 0.0,
            )
            base = np.concatenate([start, end], axis=1)
            for su in shifts_u:
                for sv in shifts_v:
                    segs.append(base + np.array([su, sv, su, sv]))
                    bulges.append(bulge)
    if not segs:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(num_slabs + 1, np.int64), np.zeros(0, np.int64), (0.0, 0.0)
    segs = np.concatenate(segs)
    bulges = np.concatenate(bulges)
    if swap:
        segs = segs[:, [1, 0, 3, 2]]
        bulges = -bulges  # mirroring the coordinates flips left and right
        lo, hi = face_rng[2], face_rng[3]
    else:
        lo, hi = face_rng[0], face_rng[1]
    width = max(hi - lo, 1e-12)
    reach = np.abs(bulges)  # the arc reaches this far beyond its chord
    a = np.minimum(segs[:, 0], segs[:, 2]) - reach
    b = np.maximum(segs[:, 0], segs[:, 2]) + reach
    keep = (b >= lo - 1e-9 * width) & (a <= hi + 1e-9 * width) & (b > a)
    segs, bulges = segs[keep], bulges[keep]
    a, b = a[keep], b[keep]
    first = np.clip(np.floor((a - lo) / width * num_slabs).astype(np.int64), 0, num_slabs - 1)
    last = np.clip(np.floor((b - lo) / width * num_slabs).astype(np.int64), 0, num_slabs - 1)
    # Bucket each segment into every slab it spans, as CSR (slab starts, segment indices).
    counts = last - first + 1
    segment_ids = np.repeat(np.arange(len(segs)), counts)
    within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    slab_ids = np.repeat(first, counts) + within
    order = np.argsort(slab_ids, kind="stable")
    starts = np.searchsorted(slab_ids[order], np.arange(num_slabs + 1))
    return segs, bulges, starts, segment_ids[order], (lo, num_slabs / width)


def load_brep_prim(
    prim,
    world: np.ndarray,
    *,
    flat_tol: float = 2e-3,
    max_depth: int = 6,
    num_slabs: int = 128,
    samples_per_span: int = 6,
    faces: tuple[int, int] | None = None,
):
    """Flatten one ``BrepArray`` prim.

    ``flat_tol`` (relative to the prim's extent) bounds how far NURBS sub-patch control points
    may deviate from flat; ``num_slabs`` buckets each face's trim segments; trim curves are
    sampled with ``samples_per_span`` points per knot span and degree (8 per span for circles).
    ``faces`` limits the trim and primitive work to a range of faces, for parallel loading of
    large prims; ``_merge_partials`` joins the results.
    """
    stats: dict = {}
    kinds = [_SURFACE_KINDS[str(t)] for t in _attr(prim, "face:surfaceType", str)]
    nfaces = len(kinds)
    face_range = _attr(prim, "face:range", np.float64, 2).reshape(-1, 4)  # umin vmin umax vmax
    face_range = face_range[:, [0, 2, 1, 3]]
    trim_type = _attr(prim, "face:trimType", str)
    loop_count = _attr(prim, "face:loopCount", np.int64)
    eu_count = _attr(prim, "loop:edgeuseCount", np.int64)
    eu_edge = _attr(prim, "edgeuse:edgeIndex", np.int64)
    eu_same = np.array([str(t) == "same" for t in _attr(prim, "edgeuse:orientationType", str)], dtype=bool)
    extent = _attr(prim, "brep:extent", np.float64, 3).reshape(-1, 3)
    diag = float(np.linalg.norm(extent.max(axis=0) - extent.min(axis=0))) if len(extent) else 1.0

    # Orientation: the faceuse that bounds a solid region with orientation "same" means the
    # surface normal points into the solid, so flip it to get the outward normal.
    flip = np.zeros(nfaces, dtype=bool)
    solid_sides = np.zeros(nfaces, dtype=np.int64)
    region_type = _attr(prim, "region:type", str)
    region_shells = _attr(prim, "region:shellCount", np.int64)
    shell_faceuses = _attr(prim, "shell:faceuseCount", np.int64)
    fu_face = _attr(prim, "faceuse:faceIndex", np.int64)
    fu_same = [str(t) == "same" for t in _attr(prim, "faceuse:orientationType", str)]
    shell = fu = 0
    for region, count in zip(region_type, region_shells):
        for _ in range(int(count)):
            for _ in range(int(shell_faceuses[shell])):
                if str(region) == "solidRegion":
                    flip[fu_face[fu]] = fu_same[fu]
                    solid_sides[fu_face[fu]] += 1
                fu += 1
            shell += 1

    # Surfaces, in order of use per type.
    counters = dict.fromkeys(range(6), 0)
    nurbs_counts = (
        np.stack(
            [
                _attr(prim, "brep:surface:nurb:uVertexCount", np.int64),
                _attr(prim, "brep:surface:nurb:vVertexCount", np.int64),
            ],
            axis=1,
        )
        if NURBS in kinds
        else None
    )
    # NURBS surface arrays, read only when the prim has NURBS faces.
    n_cvs = n_w = n_uo = n_vo = n_uk = n_vk = cv_off = uk_off = vk_off = None
    if NURBS in kinds:
        n_cvs = _attr(prim, "brep:surface:nurb:controlVertices", np.float64, 3).reshape(-1, 3)
        n_w = _attr(prim, "brep:surface:nurb:weights", np.float64)
        if len(n_w) != len(n_cvs):
            n_w = np.ones(len(n_cvs))
        n_uo = _attr(prim, "brep:surface:nurb:uOrder", np.int64)
        n_vo = _attr(prim, "brep:surface:nurb:vOrder", np.int64)
        n_uk = _attr(prim, "brep:surface:nurb:uKnots", np.float64)
        n_vk = _attr(prim, "brep:surface:nurb:vKnots", np.float64)
        cv_off = np.concatenate([[0], np.cumsum(nurbs_counts[:, 0] * nurbs_counts[:, 1])])
        uk_off = np.concatenate([[0], np.cumsum(nurbs_counts[:, 0] + n_uo)])
        vk_off = np.concatenate([[0], np.cumsum(nurbs_counts[:, 1] + n_vo)])
    analytic = {}
    for kind, prefix in _ANALYTIC_PREFIX.items():
        if kind not in kinds:
            continue
        names = {
            PLANE: ("origin", "axis", "refDirection"),
            CYLINDER: ("origin", "axis", "refDirection", "radius"),
            CONE: ("origin", "axis", "refDirection", "radius", "semiAngle"),
            SPHERE: ("center", "axis", "refDirection", "radius"),
            TORUS: ("origin", "axis", "refDirection", "majorRadius", "minorRadius"),
        }[kind]
        analytic[kind] = [
            _attr(prim, prefix + n, np.float64, 3 if n in ("origin", "center", "axis", "refDirection") else 0)
            for n in names
        ]

    # Frames of all analytic surfaces at once (per face, NumPy call overhead dominates).
    frames = {kind: _frame(arr[1], arr[2]) for kind, arr in analytic.items() if len(arr[1])}
    surfaces = []
    for kind in kinds:
        i = counters[kind]
        counters[kind] += 1
        if kind == NURBS:
            nu, nv = (int(c) for c in nurbs_counts[i])
            pts = n_cvs[cv_off[i] : cv_off[i] + nu * nv].reshape(nu, nv, 3)
            w = n_w[cv_off[i] : cv_off[i] + nu * nv].reshape(nu, nv)
            s = NurbsSurface(
                pts, w, n_uk[uk_off[i] : uk_off[i + 1]], n_vk[vk_off[i] : vk_off[i + 1]], int(n_uo[i]), int(n_vo[i])
            )
            surfaces.append(Surface(NURBS, nurbs=s))
            continue
        arr = analytic[kind]
        o = arr[0][i]
        x, y, ax = (frame[i] for frame in frames[kind])
        if kind == PLANE:
            params = (0.0, 0.0)
        elif kind == CONE:
            params = (float(arr[3][i]), math.tan(float(arr[4][i])))
        elif kind == TORUS:
            params = (float(arr[3][i]), float(arr[4][i]))
        else:
            params = (float(arr[3][i]), 0.0)
        surfaces.append(Surface(kind, o, x, y, ax, params))

    # Edges.
    edge_kind = [str(t) for t in _attr(prim, "edge:curveType", str)]
    edge_range = _attr(prim, "edge:range", np.float64).reshape(-1, 2)
    edge_local, seen = [], {}
    for k in edge_kind:
        edge_local.append(seen.get(k, 0))
        seen[k] = seen.get(k, 0) + 1

    def curve3d(kind_name):
        base = f"brep:edge3d{kind_name}:curve3d:{kind_name.lower()}:"
        return base

    nb = "brep:edge3dNurb:curve3d:nurb:"
    edge_nurbs = _Packed(
        _attr(prim, nb + "vertexCount", np.int64),
        _attr(prim, nb + "order", np.int64),
        _attr(prim, nb + "controlVertices", np.float64, 3).reshape(-1, 3),
        _attr(prim, nb + "knots", np.float64),
        _attr(prim, nb + "weights", np.float64),
    )
    lb, cb, eb = curve3d("Line"), curve3d("Circle"), curve3d("Ellipse")
    v3 = lambda n: _attr(prim, n, np.float64, 3).reshape(-1, 3)  # noqa: E731
    uvb = "brep:curveUv:nurb:"
    uv_counts = _attr(prim, uvb + "vertexCount", np.int64)
    uv_curves = None
    if len(uv_counts):
        uv_curves = _Packed(
            uv_counts,
            _attr(prim, uvb + "order", np.int64),
            _attr(prim, uvb + "controlVertices", np.float64, 2).reshape(-1, 2),
            _attr(prim, uvb + "knots", np.float64),
            _attr(prim, uvb + "weights", np.float64),
        )
    prim_data = {
        "edge_kind": edge_kind,
        "edge_local": edge_local,
        "edge_range": edge_range,
        "edge_nurbs": edge_nurbs,
        "line": (v3(lb + "origin"), v3(lb + "direction")),
        "circle": (v3(cb + "center"), v3(cb + "axis"), v3(cb + "refDirection"), _attr(prim, cb + "radius")),
        "ellipse": (
            v3(eb + "center"),
            v3(eb + "axis"),
            v3(eb + "refDirection"),
            _attr(prim, eb + "xRadius"),
            _attr(prim, eb + "yRadius"),
        ),
        "edgeuse_edge": eu_edge,
        "edgeuse_same": eu_same,
        "uv_curves": uv_curves,
        "stats": stats,
        "samples_per_span": samples_per_span,
    }

    loop_off = np.concatenate([[0], np.cumsum(loop_count)]).astype(np.int64)
    eu_off = np.concatenate([[0], np.cumsum(eu_count)]).astype(np.int64)

    face_period = np.zeros((nfaces, 2))
    face_flags = np.zeros(nfaces, dtype=np.int32)
    face_slab = np.zeros((nfaces, 2), dtype=np.int32)
    face_slab_map = np.zeros((nfaces, 2))
    slab_start, slab_segments, all_segments, all_bulges = [0], [], [], []
    slab_total = 0
    seg_base = 0
    owned = range(*faces) if faces is not None else range(nfaces)
    face_flags[flip] |= FLAG_FLIP
    face_flags[solid_sides == 1] |= FLAG_SOLID
    if _solids_closed(kinds, solid_sides == 1, loop_off, eu_off, eu_edge):
        face_flags[solid_sides == 1] |= FLAG_CLOSED
    for f in owned:
        surface = surfaces[f]
        period = surface.periods()
        face_period[f] = period
        loops = [list(range(eu_off[li], eu_off[li + 1])) for li in range(loop_off[f], loop_off[f + 1])]
        loops = [lp for lp in loops if lp]
        if str(trim_type[f]) == "rectangular" and len(loops) <= 1:
            continue
        try:
            closed, open_ = _face_loops_uv(prim_data, surface, face_range[f], loops, period)
        except Exception as error:  # keep rendering the untrimmed face
            stats.setdefault("trim_failures", []).append((f, repr(error)))
            continue
        open_in_v = any(np.ptp(p[:, 1]) > 0.5 * period[1] for p in open_) if period[1] else False
        open_in_u = any(np.ptp(p[:, 0]) > 0.5 * period[0] for p in open_) if period[0] else False
        swap = open_in_v and not open_in_u
        segs, bulges, starts, members, (lo, scale) = _build_trim(closed, open_, face_range[f], period, swap, num_slabs)
        if not len(segs) or _trims_range(segs, bulges, face_range[f], swap):
            continue
        face_flags[f] |= FLAG_TRIMMED | (FLAG_SWAP if swap else 0)
        face_slab[f] = (len(slab_start) - 1, num_slabs)
        face_slab_map[f] = (lo, scale)
        slab_start.extend((starts[1:] + slab_total).tolist())
        slab_segments.append(members + seg_base)
        slab_total += len(members)
        all_segments.append(segs)
        all_bulges.append(bulges)
        seg_base += len(segs)

    # Primitives: one per analytic face, flat rational Bezier sub-patches for NURBS faces.
    prim_kind, prim_face, prim_bounds, prim_cv, prim_degree, prim_uv, prim_tol, cvs = [], [], [], [], [], [], [], []
    ncv = 0
    for f in owned:
        surface = surfaces[f]
        if surface.kind != NURBS:
            lo, hi = surface.bounds(tuple(face_range[f]))
            prim_kind.append(surface.kind)
            prim_face.append(f)
            prim_bounds.append(np.concatenate([lo, hi]))
            prim_cv.append(0)
            prim_degree.append((0, 0))
            prim_uv.append(face_range[f])
            prim_tol.append(0.0)
            continue
        urng = face_range[f]
        for patch, rect in bezier_subpatches(surface.nurbs, flat_tol * diag, max_depth):
            if rect[1] < urng[0] or rect[0] > urng[1] or rect[3] < urng[2] or rect[2] > urng[3]:
                continue  # outside the face's trimmed domain
            if max(patch.shape[:2]) <= 4:
                patch = to_bicubic(patch)  # exact; the device evaluates bicubics from registers
            if max(patch.shape[:2]) > 8:
                raise ValueError(f"face {f}: NURBS degree > 7 is not supported by the prototype")
            if np.any(patch[..., 3] <= 0.0):
                raise ValueError(f"face {f}: non-positive NURBS weights break the convex-hull bounds")
            euclid = patch[..., :3] / patch[..., 3:4]
            lo, hi = euclid.reshape(-1, 3).min(axis=0), euclid.reshape(-1, 3).max(axis=0)
            pad = 1e-6 * diag
            prim_kind.append(NURBS)
            prim_face.append(f)
            prim_bounds.append(np.concatenate([lo - pad, hi + pad]))
            prim_cv.append(ncv)
            prim_degree.append((patch.shape[0] - 1, patch.shape[1] - 1))
            prim_uv.append(rect)
            prim_tol.append(1e-5 * float(np.linalg.norm(hi - lo)) + 1e-7 * diag)
            cvs.append(patch.reshape(-1, 4))
            ncv += patch.shape[0] * patch.shape[1]

    stats.update(
        faces=nfaces,
        primitives=len(prim_kind),
        nurbs_subpatches=int(sum(1 for k in prim_kind if k == NURBS)),
        trim_segments=seg_base,
        face_kinds={KIND_NAMES[k]: kinds.count(k) for k in set(kinds)},
    )
    return BrepPrim(
        path=str(prim.GetPath()),
        world=world,
        face_kind=np.asarray(kinds, dtype=np.int32),
        face_origin=np.array([s.origin for s in surfaces]),
        face_x=np.array([s.x for s in surfaces]),
        face_y=np.array([s.y for s in surfaces]),
        face_axis=np.array([s.axis for s in surfaces]),
        face_params=np.array([s.params for s in surfaces]),
        face_range=face_range,
        face_period=face_period,
        face_flags=face_flags,
        face_slab=face_slab,
        face_slab_map=face_slab_map,
        slab_start=np.asarray(slab_start, dtype=np.int32),
        slab_segments=np.concatenate(slab_segments).astype(np.int32) if slab_segments else np.zeros(0, np.int32),
        segments=(np.concatenate(all_segments) if all_segments else np.zeros((0, 4))).astype(np.float32),
        segment_bulge=(np.concatenate(all_bulges) if all_bulges else np.zeros(0)).astype(np.float32),
        prim_kind=np.asarray(prim_kind, dtype=np.int32),
        prim_face=np.asarray(prim_face, dtype=np.int32),
        prim_bounds=np.asarray(prim_bounds),
        prim_cv=np.asarray(prim_cv, dtype=np.int32),
        prim_degree=np.asarray(prim_degree, dtype=np.int32).reshape(-1, 2),
        prim_uv=np.asarray(prim_uv).reshape(-1, 4),
        prim_tol=np.asarray(prim_tol),
        cvs=np.concatenate(cvs) if cvs else np.zeros((0, 4)),
        surfaces=surfaces if faces is None or faces[0] == 0 else [],
        stats=dict(stats, owned=(owned.start, owned.stop)),
    )


def _merge_partials(parts: list[BrepPrim]) -> BrepPrim:
    """Join loads of one prim over disjoint face ranges (see ``load_brep_prim(faces=)``)."""
    import dataclasses  # noqa: PLC0415

    if len(parts) == 1:
        return parts[0]
    parts = sorted(parts, key=lambda p: p.stats["owned"][0])
    nfaces = len(parts[0].face_kind)
    face_flags = np.zeros(nfaces, dtype=np.int32)
    face_period = np.zeros((nfaces, 2))
    face_slab = np.zeros((nfaces, 2), dtype=np.int32)
    face_slab_map = np.zeros((nfaces, 2))
    slab_start, slab_segments, segments, bulges, cvs = [], [], [], [], []
    columns = {
        k: [] for k in ("prim_kind", "prim_face", "prim_bounds", "prim_degree", "prim_uv", "prim_tol", "prim_cv")
    }
    slabs = members = segs = ncv = 0
    stats: dict = {}
    for part in parts:
        f0, f1 = part.stats["owned"]
        # Owned per-face arrays may be full-size (direct loads) or sliced (from workers).
        sl = slice(f0, f1) if len(part.face_flags) == nfaces else slice(None)
        face_flags[f0:f1] = part.face_flags[sl]
        face_slab[f0:f1] = part.face_slab[sl] + np.array([slabs, 0])
        face_slab_map[f0:f1] = part.face_slab_map[sl]
        face_period[f0:f1] = part.face_period[sl]
        slab_start.append(part.slab_start[:-1] + members)
        slab_segments.append(part.slab_segments + segs)
        segments.append(part.segments)
        bulges.append(part.segment_bulge)
        for key, values in columns.items():
            values.append(getattr(part, key) + ((part.prim_kind == NURBS) * ncv if key == "prim_cv" else 0))
        cvs.append(part.cvs)
        slabs += len(part.slab_start) - 1
        members += len(part.slab_segments)
        segs += len(part.segments)
        ncv += len(part.cvs)
        for key, value in part.stats.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                stats[key] = (
                    max(stats.get(key, 0), value) if key in ("faces", "max_uv_gap") else stats.get(key, 0) + value
                )
    stats["face_kinds"] = parts[0].stats.get("face_kinds")
    stats["primitives"] = sum(len(p.prim_kind) for p in parts)
    return dataclasses.replace(
        parts[0],
        face_flags=face_flags,
        face_period=face_period,
        face_slab=face_slab,
        face_slab_map=face_slab_map,
        slab_start=np.concatenate([*slab_start, [members]]).astype(np.int32),
        slab_segments=np.concatenate(slab_segments).astype(np.int32),
        segments=np.concatenate(segments),
        segment_bulge=np.concatenate(bulges),
        cvs=np.concatenate(cvs),
        stats=stats,
        **{k: np.concatenate(v) for k, v in columns.items()},
    )


def local_transform(prim, time=None) -> tuple[np.ndarray, bool]:
    """A prim's local transform (4x4, column vectors) and whether it resets the transform stack.

    Without the omniSolid schema plugin, USD does not know ``BrepArray`` is a ``Gprim`` and
    ``UsdGeom.Xformable`` rejects it, which would drop its own xformOps. Evaluate them directly.
    """
    from pxr import Gf, Usd, UsdGeom  # noqa: PLC0415

    time = Usd.TimeCode.Default() if time is None else time
    xformable = UsdGeom.Xformable(prim)
    if xformable:
        return np.array(xformable.GetLocalTransformation(time), dtype=np.float64).T, xformable.GetResetXformStack()
    order = prim.GetAttribute("xformOpOrder")
    names = order.Get() if order and order.HasAuthoredValue() else []
    matrix = Gf.Matrix4d(1.0)
    reset = False
    for name in names or []:
        if name == "!resetXformStack!":
            reset = True
            continue
        inverse = name.startswith("!invert!")
        attribute = prim.GetAttribute(name[len("!invert!") :] if inverse else name)
        matrix = UsdGeom.XformOp(attribute, inverse).GetOpTransform(time) * matrix
    return np.array(matrix, dtype=np.float64).T, reset


class _Attribute:
    def __init__(self, value):
        self._value = value

    def __bool__(self) -> bool:
        return self._value is not None

    def HasAuthoredValue(self) -> bool:  # noqa: N802 - mirrors Usd.Attribute
        return self._value is not None

    def Get(self):  # noqa: N802
        return self._value


class PrimSnapshot:
    """A BrepArray prim's authored attributes as plain arrays: picklable, for worker processes."""

    def __init__(self, prim):
        self.path = str(prim.GetPath())
        self.values = {}
        for attribute in prim.GetAuthoredAttributes():
            value = attribute.Get()
            if value is None:
                continue
            if "token" in str(attribute.GetTypeName()):
                self.values[attribute.GetName()] = [str(v) for v in value]
            else:
                self.values[attribute.GetName()] = np.asarray(value)

    def GetAttribute(self, name: str) -> _Attribute:  # noqa: N802 - mirrors Usd.Prim
        return _Attribute(self.values.get(name))

    def GetPath(self) -> str:  # noqa: N802
        return self.path


def _load_snapshot(snapshot: PrimSnapshot, options: dict, faces=None) -> BrepPrim:
    return load_brep_prim(snapshot, np.eye(4), faces=faces, **options)


def _worker_main() -> None:
    """Subprocess entry: pickled (snapshots, tasks, options) on stdin -> partial BrepPrims on stdout."""
    import pickle  # noqa: PLC0415
    import sys  # noqa: PLC0415

    blobs, tasks, options = pickle.load(sys.stdin.buffer)
    snapshots = [pickle.loads(blob) for blob in blobs]  # pickled once by the parent, shared
    results = [_strip_partial(_load_snapshot(snapshots[i], options, faces)) for i, faces in tasks]
    pickle.dump(results, sys.stdout.buffer, protocol=pickle.HIGHEST_PROTOCOL)


_SHARED_FACE_FIELDS = ("face_kind", "face_origin", "face_x", "face_y", "face_axis", "face_params", "face_range")
_OWNED_FACE_FIELDS = ("face_flags", "face_period", "face_slab", "face_slab_map")


def _strip_partial(part: BrepPrim) -> BrepPrim:
    """Keep only what a face-range load contributes, so workers send back little."""
    import dataclasses  # noqa: PLC0415

    f0, f1 = part.stats["owned"]
    if f0 == 0 and f1 == len(part.face_kind):
        return part
    changes = {k: getattr(part, k)[f0:f1] for k in _OWNED_FACE_FIELDS}
    if f0 != 0:  # identical in every range: the first range brings them
        changes.update({k: None for k in _SHARED_FACE_FIELDS})
    return dataclasses.replace(part, **changes)


def load_brep_prims(prims, *, workers: int | None = None, **options) -> list[BrepPrim]:
    """Load several BrepArray prims (in object space), in parallel processes when worthwhile.

    The loader is GIL-bound Python, so threads would not help. Each prim's attributes are
    snapshotted to plain arrays and loaded by worker subprocesses that import only this module:
    unlike multiprocessing's spawn, they do not re-run the caller's main script, and unlike fork
    they are safe in a process with CUDA and USD threads.
    """
    import os  # noqa: PLC0415
    import pickle  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    snapshots = [PrimSnapshot(prim) for prim in prims]
    sizes = [len(snap.values.get("face:surfaceType", ())) for snap in snapshots]
    workers = workers or os.cpu_count() or 1
    if workers <= 1 or sum(sizes) < 2000:
        return [_load_snapshot(snapshot, options) for snapshot in snapshots]
    # Tasks are whole prims or face ranges of big ones, balanced across workers by face count.
    # A face range still parses its whole prim, so a big prim gets one range per worker.
    chunk = max(1000, -(-max(sizes) // workers), sum(sizes) // (3 * workers))
    tasks = []
    for i, size in enumerate(sizes):
        if size <= chunk:
            tasks.append((size, i, None))
        else:
            tasks += [(min(chunk, size - f0), i, (f0, min(size, f0 + chunk))) for f0 in range(0, size, chunk)]
    workers = min(workers, len(tasks))
    assigned: list[list] = [[] for _ in range(workers)]
    loads = [0] * workers
    for size, i, faces in sorted(tasks, key=lambda t: -t[0]):
        k = loads.index(min(loads))
        assigned[k].append((i, faces))
        loads[k] += size + 1
    command = [sys.executable, "-c", "from warp_optix._src.brep.loader import _worker_main; _worker_main()"]
    # One math thread per worker: NumPy's BLAS/OpenMP pools would oversubscribe the cores otherwise.
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    blobs: dict[int, bytes] = {}
    jobs = []
    for work in assigned:
        needed = sorted({i for i, _ in work})
        local = {i: n for n, i in enumerate(needed)}
        for i in needed:
            if i not in blobs:
                blobs[i] = pickle.dumps(snapshots[i], protocol=pickle.HIGHEST_PROTOCOL)
        payload = pickle.dumps(([blobs[i] for i in needed], [(local[i], f) for i, f in work], options))
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        jobs.append((work, process, payload))
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    # Talk to all workers at once (I/O threads): communicate() feeds stdin and drains stdout
    # without deadlocking on full pipes, and a worker only starts once it has its input.
    with ThreadPoolExecutor(len(jobs)) as io:
        outputs = list(io.map(lambda job: job[1].communicate(job[2])[0], jobs))
    partials: list[list] = [[] for _ in snapshots]
    for (work, process, _), output in zip(jobs, outputs, strict=True):
        if process.returncode != 0:
            raise RuntimeError(f"BrepArray loader worker failed with exit code {process.returncode}")
        for (i, _), part in zip(work, pickle.loads(output), strict=True):
            partials[i].append(part)
    results = [_merge_partials(parts) for parts in partials]
    return results


def load_stage(path: str, **options) -> list[BrepPrim]:
    """Load every ``BrepArray`` prim of a USD stage with its world transform.

    Instance proxies of one prototype share the loaded arrays (only ``world`` differs), so callers
    can build one acceleration structure per prototype: group placements by ``prim.face_kind``.
    """
    import dataclasses  # noqa: PLC0415

    from pxr import Usd, UsdGeom  # noqa: PLC0415

    stage = Usd.Stage.Open(path)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    predicate = Usd.TraverseInstanceProxies(Usd.PrimIsActive & Usd.PrimIsDefined & ~Usd.PrimIsAbstract)
    unique: dict[str, int] = {}
    sources, placements = [], []
    for prim in stage.Traverse(predicate):
        if prim.GetTypeName() != "BrepArray":
            continue
        local, reset = local_transform(prim)
        parent = np.array(cache.GetLocalToWorldTransform(prim.GetParent()), dtype=np.float64).T
        world = local if reset else parent @ local
        key = str(prim.GetPrimInPrototype().GetPath()) if prim.IsInstanceProxy() else str(prim.GetPath())
        if key not in unique:
            unique[key] = len(sources)
            sources.append(prim)
        placements.append((unique[key], world, str(prim.GetPath())))
    loaded = load_brep_prims(sources, **options)
    return [dataclasses.replace(loaded[i], world=world, path=path) for i, world, path in placements]
