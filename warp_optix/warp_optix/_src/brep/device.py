# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-side ray/BRep intersection for custom primitives (see ``loader`` for the data layout).

Each analytic face (plane, cylinder, cone, sphere, torus) is one custom primitive whose AABB bounds
its UV range; each NURBS face becomes flat rational Bezier sub-patches (Martin et al., "Practical
Ray Tracing of Trimmed NURBS Surfaces", 2000). A candidate hit is mapped to the face's (u, v) and
kept only if it lies inside the face's trim loops (crossing test against UV polylines, bucketed
into slabs).

``intersect(brep, prim, origin, direction)`` runs in an intersection program on object-space rays
and reports at most one hit with six attributes: u, v, the outward object-space normal (x, y, z)
and the face index. The hit kind is 1 when the ray leaves a solid through the face, else 0.
"""

from __future__ import annotations

import os
from typing import Any, cast

import numpy as np
import warp as wp

from . import bounds, trim_grid
from .loader import CONE, FLAG_FLIP, FLAG_SOLID, FLAG_SWAP, FLAG_TRIMMED, NURBS, PLANE, SPHERE, TORUS

# Read once before kernel compilation. Disabled builds contain no diagnostic atomics.
DIAGNOSTICS = os.environ.get("WARP_OPTIX_BREP_DIAGNOSTICS") == "1"
DIAGNOSTIC_NAMES = (
    "aabb_candidates",
    "hull_rejects",
    "shell_candidates",
    "newton_calls",
    "newton_iterations",
    "newton_failures",
    "fallback_entries",
    "root_searches",
    "projection_calls",
    "projection_evaluations",
    "newton_last_iteration_failures",
    "newton_singular_exits",
    "newton_domain_exits",
    "fallback_nonconverged",
    "fallback_grazing",
    *tuple(f"newton_iterations_{i}" for i in range(1, 11)),
    "projection_stalled",
    "obb_rejects",
)
NEWTON_ITERATIONS = 10
# Below this |cos| between the ray and the surface, a ray can cross a flat sub-patch twice: there,
# the nearest crossing is isolated along the ray (first_root) instead of trusting Newton's root.
GRAZING_COS = 0.3
ATTRIBUTE_WORDS = 6
# Parameter-space margin around a sub-patch's square: first_root skips a stretch of the ray whose
# foot points pass outside the square grown by it (the crossings there are a neighbor's).
SQUARE_MARGIN = wp.constant(0.05)
# Triangles per layer of a sub-patch's shell (proxy.py): outer layer first, then inner, then walls.
SHELL_LAYER_TRIANGLES = wp.constant(72)
GRID_REF_A = wp.constant(trim_grid.REF_A)
GRID_REF_C = wp.constant(trim_grid.REF_C)


@wp.struct
class Surface:
    """A primitive's surface frame and its face's trim lookup in one 128-byte record (one cache
    line), so an intersection needs a single dependent load for all of it. The primitives of one
    face repeat its trim record. NURBS patches use origin/x/y/axis for an oriented bound
    (center and scaled axes), and params[0] for its coordinate reach; they need no analytic
    surface frame. Zero bound axes disable the optional rejection."""

    origin: wp.vec3
    x: wp.vec3
    y: wp.vec3
    axis: wp.vec3
    params: wp.vec2
    range: wp.vec4
    period: wp.vec2
    flags: wp.int32  # the face's FLAG_* bits, and the surface kind from bit KIND_SHIFT
    face: wp.int32
    # Trim cell grid (see trim_grid.py): (cell base, cells along a, cells along c) and (lo a, lo c,
    # cells per unit a, cells per unit c). A trimmed face without a grid (no cells along a) has
    # (first slab, 0, slab count) and (lo a, slabs per unit a, 0, 0) for the slab test instead.
    grid: wp.vec3i
    grid_map: wp.vec4
    # NURBS sub-patches: their control points are stored relative to this point, their first one
    # (zero for analytic faces). Patch math runs in these local coordinates, where float32 resolves
    # the Newton tolerance even for parts modeled far from their origin (assembly coordinates).
    anchor: wp.vec3


KIND_SHIFT = wp.constant(8)


@wp.struct
class BrepData:
    prims: wp.array[Surface]
    slab_start: wp.array[wp.int32]
    slab_segments: wp.array[wp.int32]
    segments: wp.array[wp.vec4]
    segment_bulge: wp.array[wp.float32]
    # Trim cell grids (see Surface): per cell (first member, member count << 1 | parity at its
    # reference point); members are chord copies.
    cell_info: wp.array[wp.vec2i]
    cell_chords: wp.array[wp.vec4]
    cell_bulges: wp.array[wp.float32]
    prim_cv: wp.array[wp.int32]
    prim_degree: wp.array[wp.vec2i]
    prim_uv: wp.array[wp.vec4]
    prim_tol: wp.array[wp.float32]
    cvs: wp.array[wp.vec4]  # NURBS control points, homogeneous, relative to their patch's anchor
    # Triangle shells (see proxy.py): each shell triangle's global primitive (its patch), and the
    # patch parameters (s, t) at the corners of one shell's triangles (the same for every shell).
    proxy_prim: wp.array[wp.int32]
    shell_st: wp.array[wp.vec2]
    diagnostic_counts: wp.array2d[wp.uint64]  # per primitive; empty when diagnostics are disabled


@wp.func
def diagnostic_add(brep: BrepData, prim: int, metric: int, count: int):
    if wp.static(DIAGNOSTICS):
        wp.atomic_add(brep.diagnostic_counts, prim, metric, wp.uint64(count))


# ---------------------------------------------------------------------------------------------
# Trimming and reporting
# ---------------------------------------------------------------------------------------------


@wp.func
def wrap(x: float, lo: float, period: float) -> float:
    if period > 0.0:
        return lo + (x - lo) - period * wp.floor((x - lo) / period)
    return x


@wp.func
def in_bulge(seg: wp.vec4, h: float, a: float, c: float) -> bool:
    """Whether (a, c) lies between a chord and its curve (a parabolic arc through the bulge),
    where the polygon's answer is wrong: flipping it there follows curved trim edges to third
    order."""
    if h == 0.0:
        return False
    dx = seg[2] - seg[0]
    dy = seg[3] - seg[1]
    length2 = dx * dx + dy * dy
    if length2 <= 0.0:
        return False
    along = ((a - seg[0]) * dx + (c - seg[1]) * dy) / length2
    if along <= 0.0 or along >= 1.0:
        return False
    side = ((c - seg[1]) * dx - (a - seg[0]) * dy) / wp.sqrt(length2)
    arc = 4.0 * h * along * (1.0 - along)
    return (h > 0.0 and side > 0.0 and side < arc) or (h < 0.0 and side < 0.0 and side > arc)


@wp.func
def inside_trim_slab(b: BrepData, surf: Surface, a: float, c: float) -> bool:
    """Crossings of a ray toward +c against the chords of the point's slab."""
    slab = surf.grid  # (first slab, 0, slab count) without a cell grid
    smap = surf.grid_map
    s = slab[0] + wp.clamp(int(wp.floor((a - smap[0]) * smap[1])), 0, slab[2] - 1)
    crossings = int(0)
    for k in range(b.slab_start[s], b.slab_start[s + 1]):
        index = b.slab_segments[k]
        seg = b.segments[index]
        if (seg[0] <= a) != (seg[2] <= a):
            t = (a - seg[0]) / (seg[2] - seg[0])
            if seg[1] + t * (seg[3] - seg[1]) > c:
                crossings += 1
        if in_bulge(seg, b.segment_bulge[index], a, c):
            crossings += 1
    return (crossings & 1) == 1


@wp.func
def inside_trim(b: BrepData, surf: Surface, u: float, v: float) -> bool:
    flags = surf.flags
    if (flags & FLAG_TRIMMED) == 0:
        return True
    a = u
    c = v
    if (flags & FLAG_SWAP) != 0:
        a = v
        c = u
    grid = surf.grid
    if grid[1] == 0:
        return inside_trim_slab(b, surf, a, c)
    m = surf.grid_map
    i = wp.clamp(int(wp.floor((a - m[0]) * m[2])), 0, grid[1] - 1)
    j = wp.clamp(int(wp.floor((c - m[1]) * m[3])), 0, grid[2] - 1)
    info = b.cell_info[grid[0] + j * grid[1] + i]
    parity = info[1] & 1
    count = info[1] >> 1
    if count == 0:
        return parity == 1  # no chord near: the whole cell is inside or outside
    # Polygon parity at P = parity at the cell's reference point R, flipped per chord that the
    # segment R -> P crosses (half-open in the side of R -> P, so a chord end on it counts once).
    ra = m[0] + (float(i) + GRID_REF_A) / m[2]
    rc = m[1] + (float(j) + GRID_REF_C) / m[3]
    pa = a - ra
    pc = c - rc
    for k in range(info[0], info[0] + count):
        seg = b.cell_chords[k]
        q0a = seg[0] - ra
        q0c = seg[1] - rc
        q1a = seg[2] - ra
        q1c = seg[3] - rc
        if (pa * q0c - pc * q0a > 0.0) != (pa * q1c - pc * q1a > 0.0):
            da = q1a - q0a
            dc = q1c - q0c
            side_r = dc * q0a - da * q0c > 0.0
            side_p = da * (pc - q0c) - dc * (pa - q0a) > 0.0
            if side_r != side_p:
                parity = parity ^ 1
        if in_bulge(seg, b.cell_bulges[k], a, c):
            parity = parity ^ 1
    return parity == 1


@wp.func
def accept_uv(b: BrepData, surf: Surface, u_in: float, v_in: float):
    """Whether (u, v) lies in the face (range and trim); returns (inside, wrapped u, wrapped v)."""
    rng = surf.range
    period = surf.period
    u = wrap(u_in, rng[0], period[0])
    v = wrap(v_in, rng[2], period[1])
    eu = 1.0e-5 * (rng[1] - rng[0]) + 1.0e-7
    ev = 1.0e-5 * (rng[3] - rng[2]) + 1.0e-7
    if u < rng[0] - eu or u > rng[1] + eu or v < rng[2] - ev or v > rng[3] + ev:
        return False, u, v
    return inside_trim(b, surf, u, v), u, v


@wp.func
def accept_uv(b: BrepData, prim: int, u_in: float, v_in: float):
    return accept_uv(b, b.prims[prim], u_in, v_in)


@wp.func
def in_ray(t: float) -> bool:
    return t >= wp.optix_get_ray_tmin() and t <= wp.optix_get_ray_tmax()


@wp.func
def try_report(b: BrepData, prim: int, t: float, u_in: float, v_in: float, normal: wp.vec3, direction: wp.vec3) -> bool:
    if not in_ray(t):
        return False
    inside, u, v = accept_uv(b, prim, u_in, v_in)
    if not inside:
        return False
    return report(b, prim, t, u, v, normal, direction)


@wp.func
def report(b: BrepData, prim: int, t: float, u: float, v: float, normal: wp.vec3, direction: wp.vec3) -> bool:
    """Report a hit inside the face (u, v wrapped into its range) with its outward normal."""
    n = wp.normalize(normal)
    if (b.prims[prim].flags & FLAG_FLIP) != 0:
        n = -n
    kind = wp.uint32(0)
    if wp.dot(n, direction) > 0.0 and (b.prims[prim].flags & FLAG_SOLID) != 0:
        kind = wp.uint32(1)  # leaving the solid: the ray started inside or a trim leaks
    return wp.optix_report_intersection(
        t,
        kind,
        wp.optix_float_as_uint32(u),
        wp.optix_float_as_uint32(v),
        wp.optix_float_as_uint32(n[0]),
        wp.optix_float_as_uint32(n[1]),
        wp.optix_float_as_uint32(n[2]),
        wp.uint32(b.prims[prim].face),
    )


@wp.func
def local_ray(b: BrepData, prim: int, origin: wp.vec3, direction: wp.vec3):
    """Ray in the face frame, re-centered near the surface origin for float32 precision."""
    return frame_ray(b.prims[prim].origin, b.prims[prim].x, b.prims[prim].y, b.prims[prim].axis, origin, direction)


@wp.func
def frame_ray(o: wp.vec3, x: wp.vec3, y: wp.vec3, a: wp.vec3, origin: wp.vec3, direction: wp.vec3):
    shift = wp.dot(o - origin, direction) / wp.dot(direction, direction)
    rel = origin + shift * direction - o
    return (
        wp.vec3(wp.dot(rel, x), wp.dot(rel, y), wp.dot(rel, a)),
        wp.vec3(wp.dot(direction, x), wp.dot(direction, y), wp.dot(direction, a)),
        shift,
    )


@wp.func
def to_object(b: BrepData, prim: int, n: wp.vec3) -> wp.vec3:
    return n[0] * b.prims[prim].x + n[1] * b.prims[prim].y + n[2] * b.prims[prim].axis


# ---------------------------------------------------------------------------------------------
# Intersection: quadrics (plane, cylinder, cone, sphere), torus, NURBS
# ---------------------------------------------------------------------------------------------


@wp.func
def solve_quadratic(a: float, half_b: float, c: float):
    """Roots of a t^2 + 2 half_b t + c in ascending order; count 0, 1 or 2."""
    if wp.abs(a) < 1.0e-20:
        if wp.abs(half_b) < 1.0e-20:
            return 0, 0.0, 0.0
        t = -c / (2.0 * half_b)
        return 1, t, t
    disc = half_b * half_b - a * c
    if disc < 0.0:
        return 0, 0.0, 0.0
    # Numerically stable form.
    q = -(half_b + wp.sign(half_b) * wp.sqrt(disc))
    if q == 0.0:
        q = -wp.sqrt(disc) - 1.0e-30
    t0 = q / a
    t1 = c / q
    return 2, wp.min(t0, t1), wp.max(t0, t1)


@wp.func
def quadric_hit(brep: BrepData, prim: int, origin: wp.vec3, direction: wp.vec3, first_root: int):
    """The nearest root from ``first_root`` (0 near, 1 far) of a plane, cylinder, cone or sphere that
    lies in the ray's range and inside the face: (hit, t, u, v, object normal, root).

    All four kinds are one quadric in the face frame, Q(p) = a (x^2 + y^2) + e z^2 + h z - (g + k z)^2,
    whose half gradient is the normal, so rays of a warp hitting mixed kinds share one path."""
    surf = brep.prims[prim]  # all of it at once: one round trip to memory
    tmin = wp.optix_get_ray_tmin()
    tmax = wp.optix_get_ray_tmax()
    kind = surf.flags >> KIND_SHIFT
    o, d, shift = frame_ray(surf.origin, surf.x, surf.y, surf.axis, origin, direction)
    fp = surf.params
    plane = kind == PLANE
    a = wp.where(plane, 0.0, 1.0)
    e = wp.where(kind == SPHERE, 1.0, 0.0)
    h = wp.where(plane, 1.0, 0.0)
    g = wp.where(plane, 0.0, fp[0])
    k = wp.where(kind == CONE, fp[1], 0.0)
    if wp.abs(k) > 1.0 and d[2] != 0.0:
        # A near-flat cone's two nappes lie close together: from the ray's offset to the face origin,
        # the quadratic's coefficients cancel (cracks along the face's edges at |k| ~ 700). Measure
        # the ray from where it crosses the apex plane (g + k z = 0) instead, at most the face's
        # size further along it.
        reach = wp.length(o) + wp.abs(g)
        step = wp.clamp(-(g + k * o[2]) / (k * d[2]), -reach, reach)
        o = o + step * d
        shift = shift + step
    ro = g + k * o[2]
    count, t0, t1 = solve_quadratic(
        a * (d[0] * d[0] + d[1] * d[1]) + (e - k * k) * d[2] * d[2],
        a * (o[0] * d[0] + o[1] * d[1]) + (e * o[2] + 0.5 * h - k * ro) * d[2],
        a * (o[0] * o[0] + o[1] * o[1]) + (e * o[2] + h) * o[2] - ro * ro,
    )
    for i in range(first_root, count):
        t = wp.where(i == 1, t1, t0)
        if t + shift >= tmin and t + shift <= tmax:
            p = o + t * d
            rho = g + k * p[2]
            u = p[0]
            v = p[1]
            if not plane:
                s = wp.where(rho < 0.0, -1.0, 1.0)  # a cone past its apex: the angle of the mirrored point
                u = wp.atan2(s * p[1], s * p[0])
                v = p[2]
                if kind == SPHERE:
                    v = wp.asin(wp.clamp(p[2] / g, -1.0, 1.0))
                if wp.abs(k) > 1.0:  # near-flat cone: its radius pins v |k| times tighter than its height
                    v = (s * wp.sqrt(p[0] * p[0] + p[1] * p[1]) - g) / k
            inside, u, v = accept_uv(brep, surf, u, v)
            if inside:
                n = wp.vec3(a * p[0], a * p[1], e * p[2] + 0.5 * h - k * rho)
                n = n[0] * surf.x + n[1] * surf.y + n[2] * surf.axis
                return True, t + shift, u, v, n, i
    return False, 0.0, 0.0, 0.0, wp.vec3(), count


@wp.func
def intersect_quadric(brep: BrepData, prim: int, origin: wp.vec3, direction: wp.vec3):
    """Report the nearest root inside the face. Both roots are tested before reporting: state live
    across optixReportIntersection would be saved to the stack on every call."""
    hit, t, u, v, n, root = quadric_hit(brep, prim, origin, direction, 0)
    if hit and not report(brep, prim, t, u, v, n, direction) and root == 0:
        # An any-hit rejected the near root (coverage): try the far one, recomputed from the ray.
        direction = wp.optix_get_object_ray_direction()
        hit, t, u, v, n, root = quadric_hit(brep, prim, wp.optix_get_object_ray_origin(), direction, 1)
        if hit:
            report(brep, prim, t, u, v, n, direction)


class vec5(wp.types.vector(length=5, dtype=wp.float32)):
    pass


@wp.func
def poly_eval(c: vec5, order: int, s: float) -> float:
    """The quartic (order 0), its first (1) or second (2) derivative at s."""
    if order == 0:
        return (((c[4] * s + c[3]) * s + c[2]) * s + c[1]) * s + c[0]
    if order == 1:
        return ((4.0 * c[4] * s + 3.0 * c[3]) * s + 2.0 * c[2]) * s + c[1]
    return (12.0 * c[4] * s + 6.0 * c[3]) * s + 2.0 * c[2]


@wp.func
def bisect(c: vec5, order: int, lo_in: float, hi_in: float) -> float:
    lo = lo_in
    hi = hi_in
    flo = poly_eval(c, order, lo)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        fm = poly_eval(c, order, mid)
        if (fm < 0.0) == (flo < 0.0):
            lo = mid
            flo = fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


@wp.func
def changes_sign(c: vec5, order: int, lo: float, hi: float) -> bool:
    return (poly_eval(c, order, lo) < 0.0) != (poly_eval(c, order, hi) < 0.0)


@wp.func
def derivative_root(c: vec5, lo: float, hi: float, previous: float) -> float:
    """The root of f' in [lo, hi] (where f' is monotone) if it changes sign there, else ``previous``."""
    if changes_sign(c, 1, lo, hi):
        return bisect(c, 1, lo, hi)
    return previous


@wp.func
def breakpoint_at(b: vec5, i: int) -> float:
    """b[i] by selects: a runtime index into a vector would move it to local memory."""
    if i == 0:
        return b[0]
    if i == 1:
        return b[1]
    if i == 2:
        return b[2]
    if i == 3:
        return b[3]
    return b[4]


@wp.func
def intersect_torus(brep: BrepData, prim: int, origin: wp.vec3, direction: wp.vec3):
    o, d, shift = local_ray(brep, prim, origin, direction)
    major = brep.prims[prim].params[0]
    minor = brep.prims[prim].params[1]
    bound = major + minor
    count, ta, tb = solve_quadratic(wp.dot(d, d), wp.dot(o, d), wp.dot(o, o) - bound * bound)
    ta = wp.max(ta, wp.optix_get_ray_tmin() - shift)
    tb = wp.min(tb, wp.optix_get_ray_tmax() - shift)
    if count < 2 or tb <= ta:
        return
    # Quartic in s on the clipped segment P(s) = A + s B, scaled so |P| <= 1.
    inv = 1.0 / bound
    A = (o + ta * d) * inv
    B = (tb - ta) * d * inv
    R = major * inv
    k = R * R - minor * minor * inv * inv
    a2 = wp.dot(B, B)
    a1 = 2.0 * wp.dot(A, B)
    g0 = wp.dot(A, A) + k
    b2 = B[0] * B[0] + B[1] * B[1]
    b1 = 2.0 * (A[0] * B[0] + A[1] * B[1])
    b0 = A[0] * A[0] + A[1] * A[1]
    fr = 4.0 * R * R
    c = vec5(g0 * g0 - fr * b0, 2.0 * a1 * g0 - fr * b1, a1 * a1 + 2.0 * a2 * g0 - fr * b2, 2.0 * a2 * a1, a2 * a2)
    # Isolate roots through the derivatives: the roots of f'' (a quadratic) split [0, 1] into
    # pieces where f' is monotone; the roots of f' split it where f is monotone. Missing roots
    # repeat the previous breakpoint, leaving empty pieces that cannot change sign.
    qc, q0, q1 = solve_quadratic(12.0 * c[4], 3.0 * c[3], 2.0 * c[2])
    p1 = float(0.0)
    if qc >= 1 and q0 > 0.0 and q0 < 1.0:
        p1 = q0
    p2 = p1
    if qc == 2 and q1 > 0.0 and q1 < 1.0:
        p2 = q1
    r1 = derivative_root(c, 0.0, p1, 0.0)
    r2 = derivative_root(c, p1, p2, r1)
    r3 = derivative_root(c, p2, 1.0, r2)
    crit = vec5(0.0, r1, r2, r3, 1.0)
    i = int(0)
    while i < 4:
        lo = breakpoint_at(crit, i)
        hi = breakpoint_at(crit, i + 1)
        if changes_sign(c, 0, lo, hi):
            s = bisect(c, 0, lo, hi)
            df = poly_eval(c, 1, s)
            if df != 0.0:
                s = wp.clamp(s - poly_eval(c, 0, s) / df, lo, hi)
            t = ta + s * (tb - ta)
            p = o + t * d
            rho = wp.sqrt(p[0] * p[0] + p[1] * p[1])
            # A spindle torus (minor > major) also has a sheet with signed radius
            # major + minor*cos(v) < 0. The quartic contains both sheets; recover the sign
            # from |p|^2 + major^2 - minor^2 = 2*major*signed_radius before mapping to UV.
            side = wp.where(wp.dot(p, p) + (major - minor) * (major + minor) < 0.0, -1.0, 1.0)
            # du x dv reverses on the negative-radius sheet.
            nrm = side * p - wp.vec3(p[0], p[1], 0.0) * (major / wp.max(rho, 1.0e-30))
            if try_report(
                brep,
                prim,
                t + shift,
                wp.atan2(side * p[1], side * p[0]),
                wp.atan2(p[2], side * rho - major),
                to_object(brep, prim, nrm),
                direction,
            ):
                return
        i += 1


# ---------------------------------------------------------------------------------------------
# NURBS sub-patches: rational Bezier patches, bicubic (in registers) or of degree <= 7 (global)
# ---------------------------------------------------------------------------------------------

MAX_ORDER = 8  # Bezier degree <= 7 per direction
vec8 = wp.types.vector(length=MAX_ORDER, dtype=wp.float32)


@wp.func
def bernstein_all(n: int, x: float):
    """All degree-n Bernstein polynomials at x and their derivatives (NURBS Book A1.3).

    The loops run to the largest degree, guarded: constant bounds unroll, so every vector index
    is static and the vectors stay in registers instead of local memory.
    """
    basis = vec8(0.0)
    basis[0] = 1.0
    deriv = vec8(0.0)
    x1 = 1.0 - x
    for j in range(1, MAX_ORDER):
        if j <= n:
            if j == n:  # basis holds degree n-1: dB_i^n = n (B_{i-1}^{n-1} - B_i^{n-1})
                prev = float(0.0)
                for k in range(j):
                    deriv[k] = float(j) * (prev - basis[k])
                    prev = basis[k]
                deriv[j] = float(j) * prev
            saved = float(0.0)
            for k in range(j):
                temp = basis[k]
                basis[k] = saved + x1 * temp
                saved = x * temp
            basis[j] = saved
    return basis, deriv


@wp.struct
class Bezier:
    """A rational Bezier patch of any degree up to 7: its control points' base in ``BrepData.cvs``."""

    base: int
    degree: wp.vec2i


@wp.struct
class CubicQuartic:
    """Degree (3, 4) patch; stream its 20 control points with fixed evaluation bounds."""

    base: int


@wp.struct
class Cubic:
    """A bicubic rational Bezier patch: homogeneous control points by component, row i = s index."""

    x: wp.mat44
    y: wp.mat44
    z: wp.mat44
    w: wp.mat44


@wp.func
def load_cubic(cvs: wp.array[wp.vec4], base: int) -> Cubic:
    # Row i's four control points as matrix columns: rows 0..3 of that matrix are row i of x..w.
    r0 = wp.matrix_from_cols(cvs[base + 0], cvs[base + 1], cvs[base + 2], cvs[base + 3])
    r1 = wp.matrix_from_cols(cvs[base + 4], cvs[base + 5], cvs[base + 6], cvs[base + 7])
    r2 = wp.matrix_from_cols(cvs[base + 8], cvs[base + 9], cvs[base + 10], cvs[base + 11])
    r3 = wp.matrix_from_cols(cvs[base + 12], cvs[base + 13], cvs[base + 14], cvs[base + 15])
    c = Cubic()
    c.x = wp.matrix_from_rows(r0[0], r1[0], r2[0], r3[0])
    c.y = wp.matrix_from_rows(r0[1], r1[1], r2[1], r3[1])
    c.z = wp.matrix_from_rows(r0[2], r1[2], r2[2], r3[2])
    c.w = wp.matrix_from_rows(r0[3], r1[3], r2[3], r3[3])
    return c


@wp.func
def bernstein3(x: float) -> wp.vec4:
    y = 1.0 - x
    return wp.vec4(y * y * y, 3.0 * x * y * y, 3.0 * x * x * y, x * x * x)


@wp.func
def bernstein3_derivative(x: float) -> wp.vec4:
    y = 1.0 - x
    return 3.0 * wp.vec4(-y * y, y * y - 2.0 * x * y, 2.0 * x * y - x * x, x * x)


@wp.func
def eval_cubic(c: Cubic, s: float, t: float):
    """Point and first partials of a bicubic rational patch, as small matrix-vector products."""
    bu = bernstein3(s)
    dbu = bernstein3_derivative(s)
    bv = bernstein3(t)
    dbv = bernstein3_derivative(t)
    xv = c.x * bv
    yv = c.y * bv
    zv = c.z * bv
    wv = c.w * bv
    xt = c.x * dbv
    yt = c.y * dbv
    zt = c.z * dbv
    wt = c.w * dbv
    inv_w = 1.0 / wp.dot(bu, wv)  # one division: a vector divided by w costs one per component
    point = wp.vec3(wp.dot(bu, xv), wp.dot(bu, yv), wp.dot(bu, zv)) * inv_w
    ds = (wp.vec3(wp.dot(dbu, xv), wp.dot(dbu, yv), wp.dot(dbu, zv)) - wp.dot(dbu, wv) * point) * inv_w
    dt = (wp.vec3(wp.dot(bu, xt), wp.dot(bu, yt), wp.dot(bu, zt)) - wp.dot(bu, wt) * point) * inv_w
    return point, ds, dt


@wp.func
def eval_patch(patch: Cubic, cvs: wp.array[wp.vec4], s: float, t: float):
    return eval_cubic(patch, s, t)


@wp.func
def eval_patch(patch: CubicQuartic, cvs: wp.array[wp.vec4], s: float, t: float):
    bu = bernstein3(s)
    dbu = bernstein3_derivative(s)
    b3 = bernstein3(t)
    omt = 1.0 - t
    bv = vec5(omt * b3[0], t * b3[0] + omt * b3[1], t * b3[1] + omt * b3[2], t * b3[2] + omt * b3[3], t * b3[3])
    dbv = 4.0 * vec5(-b3[0], b3[0] - b3[1], b3[1] - b3[2], b3[2] - b3[3], b3[3])
    acc = wp.vec4(0.0)
    acc_s = wp.vec4(0.0)
    acc_t = wp.vec4(0.0)
    for i in range(4):
        row = wp.vec4(0.0)
        row_t = wp.vec4(0.0)
        for j in range(5):
            cv = cvs[patch.base + i * 5 + j]
            row += bv[j] * cv
            row_t += dbv[j] * cv
        acc += bu[i] * row
        acc_s += dbu[i] * row
        acc_t += bu[i] * row_t
    inv_w = 1.0 / acc[3]
    point = wp.vec3(acc[0], acc[1], acc[2]) * inv_w
    ds = (wp.vec3(acc_s[0], acc_s[1], acc_s[2]) - acc_s[3] * point) * inv_w
    dt = (wp.vec3(acc_t[0], acc_t[1], acc_t[2]) - acc_t[3] * point) * inv_w
    return point, ds, dt


@wp.func
def eval_patch(patch: Bezier, cvs: wp.array[wp.vec4], s: float, t: float):
    """Rational tensor Bezier patch: point and first partial derivatives."""
    p = patch.degree[0]
    q = patch.degree[1]
    bu, dbu = bernstein_all(p, s)
    bv, dbv = bernstein_all(q, t)
    acc = wp.vec4(0.0)
    acc_s = wp.vec4(0.0)
    acc_t = wp.vec4(0.0)
    # Constant bounds, guarded by the degrees: the loops unroll, so every basis index is static.
    for i in range(MAX_ORDER):
        if i <= p:
            row = wp.vec4(0.0)
            row_t = wp.vec4(0.0)
            for j in range(MAX_ORDER):
                if j <= q:
                    cv = cvs[patch.base + i * (q + 1) + j]
                    row += bv[j] * cv
                    row_t += dbv[j] * cv
            acc += bu[i] * row
            acc_s += dbu[i] * row
            acc_t += bu[i] * row_t
    inv_w = 1.0 / acc[3]
    point = wp.vec3(acc[0], acc[1], acc[2]) * inv_w
    ds = (wp.vec3(acc_s[0], acc_s[1], acc_s[2]) - acc_s[3] * point) * inv_w
    dt = (wp.vec3(acc_t[0], acc_t[1], acc_t[2]) - acc_t[3] * point) * inv_w
    return point, ds, dt


@wp.func
def patch_degree(patch: Cubic) -> wp.vec2i:
    return wp.vec2i(3, 3)


@wp.func
def patch_degree(patch: Bezier) -> wp.vec2i:
    return patch.degree


@wp.func
def patch_degree(patch: CubicQuartic) -> wp.vec2i:
    return wp.vec2i(3, 4)


@wp.func
def first_point(patch: CubicQuartic, cvs: wp.array[wp.vec4]) -> wp.vec3:
    cv = cvs[patch.base]
    return wp.vec3(cv[0], cv[1], cv[2]) / cv[3]


@wp.func
def first_point(patch: Cubic, cvs: wp.array[wp.vec4]) -> wp.vec3:
    return wp.vec3(patch.x[0, 0], patch.y[0, 0], patch.z[0, 0]) / patch.w[0, 0]


@wp.func
def first_point(patch: Bezier, cvs: wp.array[wp.vec4]) -> wp.vec3:
    cv = cvs[patch.base]
    return wp.vec3(cv[0], cv[1], cv[2]) / cv[3]


@wp.func
def corner_points(patch: Cubic, cvs: wp.array[wp.vec4]):
    return (
        wp.vec3(patch.x[0, 0], patch.y[0, 0], patch.z[0, 0]) / patch.w[0, 0],
        wp.vec3(patch.x[3, 0], patch.y[3, 0], patch.z[3, 0]) / patch.w[3, 0],
        wp.vec3(patch.x[0, 3], patch.y[0, 3], patch.z[0, 3]) / patch.w[0, 3],
        wp.vec3(patch.x[3, 3], patch.y[3, 3], patch.z[3, 3]) / patch.w[3, 3],
    )


@wp.func
def corner_points(patch: Bezier, cvs: wp.array[wp.vec4]):
    q = patch.degree[1]
    last = patch.degree[0] * (q + 1)
    c00 = cvs[patch.base]
    c10 = cvs[patch.base + last]
    c01 = cvs[patch.base + q]
    c11 = cvs[patch.base + last + q]
    return (
        wp.vec3(c00[0], c00[1], c00[2]) / c00[3],
        wp.vec3(c10[0], c10[1], c10[2]) / c10[3],
        wp.vec3(c01[0], c01[1], c01[2]) / c01[3],
        wp.vec3(c11[0], c11[1], c11[2]) / c11[3],
    )


@wp.func
def corner_points(patch: CubicQuartic, cvs: wp.array[wp.vec4]):
    c00 = cvs[patch.base]
    c10 = cvs[patch.base + 15]
    c01 = cvs[patch.base + 4]
    c11 = cvs[patch.base + 19]
    return (
        wp.vec3(c00[0], c00[1], c00[2]) / c00[3],
        wp.vec3(c10[0], c10[1], c10[2]) / c10[3],
        wp.vec3(c01[0], c01[1], c01[2]) / c01[3],
        wp.vec3(c11[0], c11[1], c11[2]) / c11[3],
    )


@wp.struct
class PatchRay:
    """An object-space ray as the intersection of two planes through it (n1, n2 orthonormal to
    the direction), with a point ``near`` on it next to the patch."""

    direction: wp.vec3
    unit: wp.vec3
    dd: float  # direction . direction
    n1: wp.vec3
    n2: wp.vec3
    near: wp.vec3
    shift: float  # ray parameter of near


@wp.func
def patch_ray(origin: wp.vec3, direction: wp.vec3, anchor: wp.vec3) -> PatchRay:
    """Re-center the ray next to ``anchor``: with a distant origin, n . (S - o) cancels in float32
    down to the convergence tolerance and Newton fails at random pixels."""
    r = PatchRay()
    r.direction = direction
    r.unit = wp.normalize(direction)
    r.dd = wp.dot(direction, direction)
    if wp.abs(r.unit[0]) > wp.abs(r.unit[1]) and wp.abs(r.unit[0]) > wp.abs(r.unit[2]):
        r.n1 = wp.normalize(wp.vec3(r.unit[1], -r.unit[0], 0.0))
    else:
        r.n1 = wp.normalize(wp.vec3(0.0, r.unit[2], -r.unit[1]))
    r.n2 = wp.cross(r.n1, r.unit)
    r.shift = wp.dot(anchor - origin, direction) / r.dd
    r.near = origin + r.shift * direction
    return r


@wp.func
def ray_t(ray: PatchRay, point: wp.vec3) -> float:
    """Ray parameter of (the projection of) a point."""
    return ray.shift + wp.dot(point - ray.near, ray.direction) / ray.dd


@wp.struct
class Hull:
    """The control points seen from the ray: (n1, n2) . (P - near) ranges, their ranges along four
    more axes in that plane (normals of the corner-to-corner edges, as x and y components), the
    nearest and farthest point along the ray, and the control point closest to the ray."""

    lo: wp.vec2
    hi: wp.vec2
    nearest: float
    farthest: float
    closest: float
    closest_index: int
    axis_x: wp.vec4
    axis_y: wp.vec4
    side_lo: wp.vec4
    side_hi: wp.vec4


@wp.func
def hull_add(h: Hull, i: int, cv: wp.vec4, ray: PatchRay) -> Hull:
    rel = wp.vec3(cv[0], cv[1], cv[2]) * (1.0 / cv[3]) - ray.near
    d = wp.vec2(wp.dot(ray.n1, rel), wp.dot(ray.n2, rel))
    out = h
    out.lo = wp.min(h.lo, d)
    out.hi = wp.max(h.hi, d)
    side = d[0] * h.axis_x + d[1] * h.axis_y
    out.side_lo = wp.min(h.side_lo, side)
    out.side_hi = wp.max(h.side_hi, side)
    along = wp.dot(rel, ray.direction)
    out.nearest = wp.min(h.nearest, along)
    out.farthest = wp.max(h.farthest, along)
    if wp.dot(d, d) < h.closest:
        out.closest = wp.dot(d, d)
        out.closest_index = i
    return out


@wp.func
def plane_point(ray: PatchRay, p: wp.vec3) -> wp.vec2:
    return wp.vec2(wp.dot(ray.n1, p - ray.near), wp.dot(ray.n2, p - ray.near))


@wp.func
def edge_normal(a: wp.vec2, b: wp.vec2) -> wp.vec2:
    e = b - a
    length = wp.length(e)
    if length > 0.0:
        return wp.vec2(-e[1], e[0]) / length
    return wp.vec2(0.0)


@wp.func
def hull_new(patch: Any, cvs: wp.array[wp.vec4], ray: PatchRay) -> Hull:
    p00, p10, p01, p11 = corner_points(patch, cvs)
    a = plane_point(ray, p00)
    b = plane_point(ray, p10)
    c = plane_point(ray, p01)
    d = plane_point(ray, p11)
    n0 = edge_normal(a, b)
    n1 = edge_normal(b, d)
    n2 = edge_normal(d, c)
    n3 = edge_normal(c, a)
    h = Hull()
    h.lo = wp.vec2(1.0e30)
    h.hi = wp.vec2(-1.0e30)
    h.nearest = 1.0e30
    h.farthest = -1.0e30
    h.closest = 1.0e30
    h.closest_index = 0
    h.axis_x = wp.vec4(n0[0], n1[0], n2[0], n3[0])
    h.axis_y = wp.vec4(n0[1], n1[1], n2[1], n3[1])
    h.side_lo = wp.vec4(1.0e30)
    h.side_hi = wp.vec4(-1.0e30)
    return h


@wp.func
def patch_hull(patch: Cubic, cvs: wp.array[wp.vec4], ray: PatchRay) -> Hull:
    h = hull_new(patch, cvs, ray)
    for i in range(16):  # a constant count: unrolled, so the matrices stay in registers
        h = hull_add(
            h,
            i,
            wp.vec4(patch.x[i // 4, i % 4], patch.y[i // 4, i % 4], patch.z[i // 4, i % 4], patch.w[i // 4, i % 4]),
            ray,
        )
    return h


@wp.func
def patch_hull(patch: Bezier, cvs: wp.array[wp.vec4], ray: PatchRay) -> Hull:
    h = hull_new(patch, cvs, ray)
    for i in range((patch.degree[0] + 1) * (patch.degree[1] + 1)):
        h = hull_add(h, i, cvs[patch.base + i], ray)
    return h


@wp.func
def patch_hull(patch: CubicQuartic, cvs: wp.array[wp.vec4], ray: PatchRay) -> Hull:
    h = hull_new(patch, cvs, ray)
    for i in range(20):
        h = hull_add(h, i, cvs[patch.base + i], ray)
    return h


@wp.func
def hull_tmin(hull: Hull, ray: PatchRay, tol: float) -> float:
    """A lower bound of the ray parameter of any patch point (weights are positive, so the patch
    lies in the convex hull of its Euclidean control points)."""
    return ray.shift + hull.nearest / ray.dd - tol / wp.sqrt(ray.dd)


@wp.func
def hull_tmax(hull: Hull, ray: PatchRay, tol: float) -> float:
    """An upper bound of the ray parameter of any patch point."""
    return ray.shift + hull.farthest / ray.dd + tol / wp.sqrt(ray.dd)


@wp.func
def hull_misses(hull: Hull, tol: float) -> bool:
    """Convex-hull rejection (the first step of Bezier clipping): the ray lies on both planes, so
    it can only hit the patch if the control points straddle each plane, and each line through it
    along an edge between the corners (the tighter separating axes for a ray passing beside the
    patch: its control points' axis-aligned box in the ray's plane is loose)."""
    if hull.lo[0] > tol or hull.hi[0] < -tol or hull.lo[1] > tol or hull.hi[1] < -tol:
        return True
    for k in range(4):
        if hull.side_lo[k] > tol or hull.side_hi[k] < -tol:
            return True
    return False


@wp.func
def newton(patch: Any, brep: BrepData, prim: int, ray: PatchRay, tol: float, seed: wp.vec2):
    """Newton on the two-plane form of the ray (Martin et al. 2000) from ``seed``.

    Returns (converged, converged inside the patch, (s, t), S, S_s, S_t).
    """
    cvs = brep.cvs
    iterations = int(0)
    st = seed
    converged = bool(False)
    point = wp.vec3(0.0)
    ds = wp.vec3(0.0)
    dt = wp.vec3(0.0)
    for _ in range(NEWTON_ITERATIONS):
        if wp.static(DIAGNOSTICS):
            iterations += 1
        point, ds, dt = eval_patch(patch, cvs, st[0], st[1])
        rel = point - ray.near
        f0 = wp.dot(ray.n1, rel)
        f1 = wp.dot(ray.n2, rel)
        done = wp.abs(f0) + wp.abs(f1) < tol
        j00 = wp.dot(ray.n1, ds)
        j01 = wp.dot(ray.n1, dt)
        j10 = wp.dot(ray.n2, ds)
        j11 = wp.dot(ray.n2, dt)
        det = j00 * j11 - j01 * j10
        if wp.abs(det) < 1.0e-30:
            diagnostic_add(brep, prim, 11, int(not done))
            converged = done
            break
        inv_det = 1.0 / det
        step = wp.vec2((j11 * f0 - j01 * f1) * inv_det, (-j10 * f0 + j00 * f1) * inv_det)
        st = st - step
        if done:
            # One more Newton step, linearized (no evaluation): the reported point then lies on the
            # ray to second order instead of anywhere within tol of it. Rays spawned from it are
            # offset by less than tol (relative 1/65536 of the position, tol 1e-5 of the patch).
            converged = True
            point = point - step[0] * ds - step[1] * dt
            break
        if st[0] < -0.5 or st[0] > 1.5 or st[1] < -0.5 or st[1] > 1.5:
            diagnostic_add(brep, prim, 12, 1)
            break
    diagnostic_add(brep, prim, 10, int(not converged and iterations == NEWTON_ITERATIONS))
    if wp.static(DIAGNOSTICS):
        diagnostic_add(brep, prim, 14 + iterations, 1)
    diagnostic_add(brep, prim, 3, 1)
    diagnostic_add(brep, prim, 4, iterations)
    diagnostic_add(brep, prim, 5, int(not converged))
    return converged, converged and in_square(st), st, point, ds, dt


@wp.func
def in_square(st: wp.vec2) -> bool:
    return st[0] >= -1.0e-4 and st[0] <= 1.0001 and st[1] >= -1.0e-4 and st[1] <= 1.0001


@wp.struct
class Foot:
    """A ray point at ``t``, the parameters of its foot point on a sub-patch (the nearest surface
    point, the patch extended past its square), and its signed distance ``h`` from the surface along
    the unit normal with its rate ``g`` = dh/dt along the ray. (Five floats: the root search keeps
    several alive, and the shell any-hit runs under the path tracer's register budget.)"""

    t: float
    st: wp.vec2
    h: float
    g: float


@wp.func
def foot(patch: Any, brep: BrepData, prim: int, ray: PatchRay, t: float, seed: wp.vec2, steps: int) -> Foot:
    """Project the ray point at ``t`` onto the patch (Gauss-Newton from ``seed``). Unlike the
    two-plane Newton, whose Jacobian vanishes as the ray turns tangent, the projection stays well
    conditioned where a ray grazes the surface. One evaluation site (a ``while`` loop does not
    unroll): this runs inside first_root's loop, and code size costs instruction cache."""
    cvs = brep.cvs
    diagnostic_add(brep, prim, 8, 1)
    q = ray.near + (t - ray.shift) * ray.direction
    st = seed
    point = wp.vec3(0.0)
    ds = wp.vec3(0.0)
    dt = wp.vec3(0.0)
    k = int(0)
    while k <= steps:
        diagnostic_add(brep, prim, 9, 1)
        point, ds, dt = eval_patch(patch, cvs, st[0], st[1])
        if k == steps:
            break
        r = q - point
        a00 = wp.dot(ds, ds)
        a01 = wp.dot(ds, dt)
        a11 = wp.dot(dt, dt)
        det = a00 * a11 - a01 * a01
        if det <= 1.0e-12 * a00 * a11:
            break
        b0 = wp.dot(ds, r)
        b1 = wp.dot(dt, r)
        step = wp.vec2((a11 * b0 - a01 * b1) / det, (a00 * b1 - a01 * b0) / det)
        updated = wp.vec2(wp.clamp(st[0] + step[0], -0.5, 1.5), wp.clamp(st[1] + step[1], -0.5, 1.5))
        # Clamping or rounding can leave the parameters unchanged. The next evaluation and
        # update would repeat exactly, so retain the point/partials already evaluated here.
        if updated[0] == st[0] and updated[1] == st[1]:
            diagnostic_add(brep, prim, 25, 1)
            break
        st = updated
        k += 1
        if wp.abs(step[0]) + wp.abs(step[1]) < 1.0e-6:
            k = steps  # converged: evaluate at the new point and stop
    n = wp.normalize(wp.cross(ds, dt))
    f = Foot()
    f.t = t
    f.st = st
    f.h = wp.dot(q - point, n)
    f.g = wp.dot(ray.direction, n)
    return f


@wp.func
def path_meets_square(a: wp.vec2, b: wp.vec2, margin: float) -> bool:
    """Whether the segment a-b in parameter space meets the patch's square, grown by a margin for
    the curvature of the feet's path (the projections of the ray points between two probes)."""
    lo = float(0.0)
    hi = float(1.0)
    d = b - a
    for k in range(2):
        if wp.abs(d[k]) < 1.0e-12:
            if a[k] < -margin or a[k] > 1.0 + margin:
                return False
        else:
            u0 = (-margin - a[k]) / d[k]
            u1 = (1.0 + margin - a[k]) / d[k]
            lo = wp.max(lo, wp.min(u0, u1))
            hi = wp.min(hi, wp.max(u0, u1))
    return lo <= hi


@wp.func
def root_foot(ray: PatchRay, st: wp.vec2, point: wp.vec3, ds: wp.vec3, dt: wp.vec3) -> Foot:
    """A Newton root as a Foot (h = 0)."""
    f = Foot()
    f.t = ray_t(ray, point)
    f.st = st
    f.h = 0.0
    f.g = wp.dot(ray.direction, wp.normalize(wp.cross(ds, dt)))
    return f


@wp.func
def first_root(
    patch: Any, brep: BrepData, prim: int, ray: PatchRay, tol: float, t0: float, t1: float, r: Foot, known: bool
):
    """The ray's nearest crossing of the (extended) sub-patch with t in [t0, t1]: (found, Foot).

    Along the ray, the signed distance h to a flat sub-patch has at most one extremum, so crossings
    come alone (h changes sign between the ends) or as a pair around a silhouette or a fold (h has
    the same sign at both ends, heading toward zero at t0 and away from it at t1), separated by a
    secant search for the extremum. Safeguarded Newton on h then finds the crossing. A ray starting
    within ``tol`` of the surface (a bounce or shadow ray) counts as on the side it heads to.

    A ``known`` root ``r`` (Newton's, or a crossing found before) saves probes: ahead of t0, a
    nearer crossing exists exactly when h at t0 is on the side h heads to at r (one probe); behind
    t0, h heads away from zero after it, so at most one crossing follows: h at t1 tells (two).
    Without a root, ``r.st`` seeds the projections.

    One projection site in one loop: probe t0 (phase 0), t1 (1), the extremum (2), the crossing (3).
    """
    a = Foot()
    b = Foot()
    t = t0
    st = r.st
    phase = int(0)
    probes = int(0)
    ahead = known and r.t >= t0
    while probes < 14:
        probes += 1
        m = foot(patch, brep, prim, ray, t, st, 2)
        st = m.st
        if phase == 0:
            if wp.abs(m.h) < tol:
                m.h = wp.where(m.g > 0.0, tol, -tol)
            a = m
            if ahead:
                if (a.h > 0.0) == (r.g > 0.0):
                    b = r  # an odd number of crossings before r: bracket them
                    b.h = wp.where(r.g > 0.0, -tol, tol)
                    phase = 3
                else:
                    return r.t <= t1, r
            else:
                phase = 1
                t = t1
                continue
        elif phase == 1:
            b = m
            phase = 3
            if not path_meets_square(a.st, b.st, SQUARE_MARGIN):
                return False, a  # every crossing in between lies on a neighbor's part of the surface
            if (a.h > 0.0) == (b.h > 0.0):
                if known or a.h * a.g >= 0.0 or b.h * b.g <= 0.0:
                    return False, a  # h never heads back across zero
                phase = 2
        elif phase == 2:
            if (m.h > 0.0) != (a.h > 0.0):
                b = m
                phase = 3
            elif wp.abs(m.g) * (b.t - a.t) < wp.abs(m.h):
                return False, a  # near the extremum and still on the starting side
            elif (m.g > 0.0) == (a.g > 0.0):
                a = m
            else:
                b = m
        else:
            if wp.abs(m.h) < tol:
                m.t = m.t - m.h / m.g  # a final Newton step (no evaluation), as in newton()
                return True, m
            if (m.h > 0.0) == (a.h > 0.0):
                a = m
            else:
                b = m
            # The bracket closed in on a jump of the projection rather than a crossing (h changes
            # sign without passing zero, where the clamped foot point switches sides): skip it.
            if (b.t - a.t) * wp.sqrt(ray.dd) < tol:
                break
        if phase == 2:
            t = a.t - a.g * (b.t - a.t) / (b.g - a.g)  # secant on g, toward the extremum
        else:
            t = m.t - m.h / m.g  # Newton, bisecting where it leaves the bracket
            if not (t > a.t and t < b.t):
                t = 0.5 * (a.t + b.t)
        # Seed the next projection from the last foot point, moved along the bracket's parameter-
        # space chord: Gauss-Newton then starts next to the new foot point.
        rate = (b.st - a.st) / (b.t - a.t)
        st = m.st + (t - m.t) * rate
    # No crossing settled. Looking for one before a known root, that root stands. A bracketed one
    # is skipped as off the square (a long, curved interval can hold such crossings before the
    # patch's own): reported outside the square, so square_root searches on past it.
    if ahead:
        return r.t <= t1, r
    if phase == 3:
        b.st = wp.vec2(-10.0)
        if b.g == 0.0:
            b.g = 1.0
        return True, b
    return False, a


@wp.func
def square_root(
    patch: Any, brep: BrepData, prim: int, ray: PatchRay, tol: float, t0: float, t1: float, r: Foot, known: bool
):
    """The ray's nearest crossing of the sub-patch itself (its square) with t in [t0, t1]: a nearer
    crossing of its extension (a neighbor's part of the surface) is skipped."""
    t = t0
    root = r
    have = known
    searches = int(0)
    while searches < 3 and t < t1:
        searches += 1
        found, f = first_root(patch, brep, prim, ray, tol, t, t1, root, have)
        if not found:
            break
        if in_square(f.st):
            return True, f
        if f.g == 0.0:
            break
        t = f.t + 2.0 * tol / wp.abs(f.g)
        root = f
        have = True
    return False, Foot()


@wp.func
def outward_cosine(brep: BrepData, prim: int, ds: wp.vec3, dt: wp.vec3, unit: wp.vec3) -> float:
    """Cosine between the face's outward normal and the ray: positive where the ray leaves."""
    normal = wp.normalize(wp.cross(ds, dt))
    if (brep.prims[prim].flags & FLAG_FLIP) != 0:
        normal = -normal
    return wp.dot(normal, unit)


@wp.func
def report_patch_hit(
    brep: BrepData, prim: int, st: wp.vec2, point: wp.vec3, ds: wp.vec3, dt: wp.vec3, ray: PatchRay
) -> float:
    """Report a sub-patch root; returns the cosine between the outward normal and the ray there."""
    rect = brep.prim_uv[prim]
    try_report(
        brep,
        prim,
        ray_t(ray, point),
        rect[0] + st[0] * (rect[1] - rect[0]),
        rect[2] + st[1] * (rect[3] - rect[2]),
        wp.cross(ds, dt),
        ray.direction,
    )
    return outward_cosine(brep, prim, ds, dt, ray.unit)


@wp.func
def intersect_patch(brep: BrepData, prim: int, patch: Any, origin: wp.vec3, direction: wp.vec3):
    """The ray's nearest root with one sub-patch inside the face's trims (OptiX keeps the nearest)."""
    tol = brep.prim_tol[prim]
    ray = patch_ray(origin, direction, first_point(patch, brep.cvs))
    hull = patch_hull(patch, brep.cvs, ray)
    t_lo = hull_tmin(hull, ray, tol)
    if hull_misses(hull, tol) or t_lo > wp.optix_get_ray_tmax():
        diagnostic_add(brep, prim, 1, 1)
        return  # no hit, or none nearer than the closest hit so far
    # The first seed is the control point nearest the ray: control point (i, j) of a Bezier patch
    # sits at about parameter (i / p, j / q).
    deg = patch_degree(patch)
    seed = wp.vec2(
        float(hull.closest_index // (deg[1] + 1)) / float(wp.max(deg[0], 1)),
        float(hull.closest_index % (deg[1] + 1)) / float(wp.max(deg[1], 1)),
    )
    converged, ok, st, point, ds, dt = newton(patch, brep, prim, ray, tol, seed)
    # A ray meeting a flat sub-patch squarely crosses it once: Newton's root (or the neighbor's, if
    # it converged outside the square) is the only one.
    if converged and wp.abs(outward_cosine(brep, prim, ds, dt, ray.unit)) >= GRAZING_COS:
        if ok:
            report_patch_hit(brep, prim, st, point, ds, dt, ray)
        return
    # Grazing, or Newton failed: near a silhouette or a fold the ray crosses the patch twice (and a
    # bounce ray leaving a concave surface crosses it again), where Newton converges to the wrong
    # root or not at all. Isolate the nearest crossing ahead along the ray instead.
    diagnostic_add(brep, prim, 6, 1)
    diagnostic_add(brep, prim, 13, int(not converged))
    diagnostic_add(brep, prim, 14, int(converged))
    root = root_foot(ray, st, point, ds, dt)
    if not converged:
        root.st = wp.vec2(0.5, 0.5)
    t0 = wp.max(t_lo, wp.optix_get_ray_tmin())
    t1 = wp.min(hull_tmax(hull, ray, tol), wp.optix_get_ray_tmax())
    known = converged
    searches = int(0)
    while searches < 2:  # the nearest crossing, then the next if the trims cut the first away
        searches += 1
        diagnostic_add(brep, prim, 7, 1)
        found, f = square_root(patch, brep, prim, ray, tol, t0, t1, root, known)
        if not found:
            return
        rect = brep.prim_uv[prim]
        _p, fs, ft = eval_patch(patch, brep.cvs, f.st[0], f.st[1])
        if try_report(
            brep,
            prim,
            f.t,
            rect[0] + f.st[0] * (rect[1] - rect[0]),
            rect[2] + f.st[1] * (rect[3] - rect[2]),
            wp.cross(fs, ft),
            ray.direction,
        ):
            return
        t0 = f.t + 2.0 * tol / wp.abs(f.g)
        root = f
        known = True


@wp.func
def patch_obb_misses(surf: Surface, origin: wp.vec3, direction: wp.vec3, tmin: float, tmax: float) -> bool:
    """Cheap conservative rejection before fetching a patch's control net.

    At a crossing, |t * direction| is bounded componentwise by |origin| + the box's reach.
    Inflate the slabs for float32 subtraction, dot products and interval arithmetic there,
    including far-away ray origins and nearly parallel rays. Stored axes need not be exactly
    orthogonal: the host bounds their projections. Zero axes always pass.
    """
    delta = origin - surf.origin
    o = wp.vec3(wp.dot(surf.x, delta), wp.dot(surf.y, delta), wp.dot(surf.axis, delta))
    d = wp.vec3(wp.dot(surf.x, direction), wp.dot(surf.y, direction), wp.dot(surf.axis, direction))
    error = wp.static(8.0 * 2.0**-23) * (wp.max(wp.abs(origin)) + surf.params[0])
    radius = wp.vec3(1.0) + error * wp.vec3(
        wp.dot(wp.abs(surf.x), wp.vec3(1.0)),
        wp.dot(wp.abs(surf.y), wp.vec3(1.0)),
        wp.dot(wp.abs(surf.axis), wp.vec3(1.0)),
    )
    lo = tmin
    hi = tmax
    for k in range(3):
        if d[k] == 0.0:
            if wp.abs(o[k]) > radius[k]:
                return True
        else:
            a = (-radius[k] - o[k]) / d[k]
            b = (radius[k] - o[k]) / d[k]
            lo = wp.max(lo, wp.min(a, b))
            hi = wp.min(hi, wp.max(a, b))
    return lo > hi


@wp.func
def intersect_nurbs(brep: BrepData, prim: int, origin: wp.vec3, direction: wp.vec3):
    diagnostic_add(brep, prim, 0, 1)
    if patch_obb_misses(brep.prims[prim], origin, direction, wp.optix_get_ray_tmin(), wp.optix_get_ray_tmax()):
        diagnostic_add(brep, prim, 26, 1)
        return
    deg = brep.prim_degree[prim]
    base = brep.prim_cv[prim]
    local = origin - brep.prims[prim].anchor  # the patch's frame (see Surface.anchor)
    if deg[0] == 3 and deg[1] == 3:
        intersect_patch(brep, prim, load_cubic(brep.cvs, base), local, direction)
    elif deg[0] == 3 and deg[1] == 4:
        patch34 = CubicQuartic()
        patch34.base = base
        intersect_patch(brep, prim, patch34, local, direction)
    else:
        patch = Bezier()
        patch.base = base
        patch.degree = deg
        intersect_patch(brep, prim, patch, local, direction)


@wp.func
def proxy_seed(brep: BrepData, local_tri: int, bary: wp.vec2) -> wp.vec2:
    """Patch parameters interpolated at a shell triangle hit (``local_tri``: the triangle's index in
    its shell geometry)."""
    corner = 3 * (local_tri % (brep.shell_st.shape[0] // 3))
    return (
        (1.0 - bary[0] - bary[1]) * brep.shell_st[corner]
        + bary[0] * brep.shell_st[corner + 1]
        + bary[1] * brep.shell_st[corner + 2]
    )


@wp.func
def proxy_hit_cubic(brep: BrepData, tri: int, local_tri: int, bary: wp.vec2, origin: wp.vec3, direction: wp.vec3):
    """proxy_hit for a bicubic sub-patch's shell (control points evaluated from registers)."""
    prim = brep.proxy_prim[tri]
    diagnostic_add(brep, prim, 2, 1)
    return proxy_refine(
        brep,
        prim,
        load_cubic(brep.cvs, brep.prim_cv[prim]),
        local_tri,
        proxy_seed(brep, local_tri, bary),
        origin - brep.prims[prim].anchor,
        direction,
    )


@wp.func
def proxy_hit_bezier(brep: BrepData, tri: int, local_tri: int, bary: wp.vec2, origin: wp.vec3, direction: wp.vec3):
    """proxy_hit for a shell around a sub-patch of any other degree (<= 7; control points streamed).

    Renderers trace these shells with their own any-hit programs, so the bicubic programs keep
    their size (OptiX saves a program's registers on every call, and hot code exceeds the i-cache).
    """
    prim = brep.proxy_prim[tri]
    diagnostic_add(brep, prim, 2, 1)
    patch = Bezier()
    patch.base = brep.prim_cv[prim]
    patch.degree = brep.prim_degree[prim]
    return proxy_refine(
        brep, prim, patch, local_tri, proxy_seed(brep, local_tri, bary), origin - brep.prims[prim].anchor, direction
    )


@wp.func
def proxy_hit(brep: BrepData, tri: int, local_tri: int, bary: wp.vec2, origin: wp.vec3, direction: wp.vec3):
    """Exact surface hit for a ray that crossed shell triangle ``tri`` (global index; ``local_tri``:
    its index in its shell geometry), for a sub-patch of any degree.

    Newton starts from the patch parameters interpolated at the triangle hit, then the face's
    range and trim decide. Returns (hit, t, u, v, outward object-space normal, face, whether the
    hit must not end the ray's search: the any-hit records it but ignores the intersection).
    Renderers that give the two kinds of shells separate programs (proxy.shell_sbt_indices) call
    proxy_hit_cubic and proxy_hit_bezier directly.
    """
    ok = bool(False)
    hit_t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    normal = wp.vec3(0.0)
    face = int(0)
    keep_searching = bool(False)
    deg = brep.prim_degree[brep.proxy_prim[tri]]
    if deg[0] == 3 and deg[1] == 3:
        ok, hit_t, u, v, normal, face, keep_searching = proxy_hit_cubic(brep, tri, local_tri, bary, origin, direction)
    else:
        ok, hit_t, u, v, normal, face, keep_searching = proxy_hit_bezier(brep, tri, local_tri, bary, origin, direction)
    return ok, hit_t, u, v, normal, face, keep_searching


@wp.func
def proxy_refine(
    brep: BrepData, prim: int, patch: Any, local_tri: int, seed: wp.vec2, origin: wp.vec3, direction: wp.vec3
):
    """proxy_hit for one sub-patch representation (see there), with the ray in the patch's frame."""
    tol = brep.prim_tol[prim]
    ray = patch_ray(origin, direction, first_point(patch, brep.cvs))
    converged, ok, st, point, ds, dt = newton(
        patch, brep, prim, ray, tol, wp.vec2(wp.clamp(seed[0], 0.0, 1.0), wp.clamp(seed[1], 0.0, 1.0))
    )
    hit_t = ray_t(ray, point)
    # Grazing, or Newton failed: the ray may cross the patch twice (a silhouette, a fold, or a bounce
    # ray leaving a concave surface), and Newton's root need not be the nearest ahead of the ray's
    # start. Isolate that crossing along the ray (see intersect_patch), using the shell: it holds
    # all of the patch's crossings, and its triangles face out of it, the outer layer's along the
    # normal (where h > 0), the inner layer's against it. A ray entering through a layer, with
    # Newton's root ahead crossed the other way, meets that root first (h has at most one extremum
    # in between); a ray leaving through a layer crossed the patch before it, if at all.
    if not converged or wp.abs(outward_cosine(brep, prim, ds, dt, ray.unit)) < GRAZING_COS:
        diagnostic_add(brep, prim, 6, 1)
        diagnostic_add(brep, prim, 13, int(not converged))
        diagnostic_add(brep, prim, 14, int(converged))
        layer = local_tri % (brep.shell_st.shape[0] // 3)
        side = wp.where(layer < SHELL_LAYER_TRIANGLES, 1.0, wp.where(layer < 2 * SHELL_LAYER_TRIANGLES, -1.0, 0.0))
        entering = wp.optix_is_front_face_hit()
        root = root_foot(ray, st, point, ds, dt)
        if (
            converged
            and entering
            and side != 0.0
            and root.t >= wp.optix_get_ray_tmax()
            and (root.g > 0.0) != (side > 0.0)
        ):
            ok = ok and hit_t >= wp.optix_get_ray_tmin()
        else:
            if not converged:
                root.st = wp.vec2(wp.clamp(seed[0], 0.0, 1.0), wp.clamp(seed[1], 0.0, 1.0))
            hull = patch_hull(patch, brep.cvs, ray)
            t_lo = hull_tmin(hull, ray, tol)
            t1 = hull_tmax(hull, ray, tol)
            if hull_misses(hull, tol):
                t1 = t_lo  # the ray passes beside the patch (mostly side-wall crossings near a neighbor)
            if not entering and side != 0.0:
                t1 = wp.min(t1, wp.optix_get_ray_tmax())  # leaving: its crossings lie before t_tri
            diagnostic_add(brep, prim, 7, 1)
            found, f = square_root(
                patch, brep, prim, ray, tol, wp.max(t_lo, wp.optix_get_ray_tmin()), t1, root, converged
            )
            ok = found
            st = f.st
            hit_t = f.t
            if found:
                point, ds, dt = eval_patch(patch, brep.cvs, st[0], st[1])
    normal = wp.vec3(0.0)
    u = float(0.0)
    v = float(0.0)
    if ok and hit_t >= wp.optix_get_ray_tmin():
        rect = brep.prim_uv[prim]
        ok, u, v = accept_uv(brep, prim, rect[0] + st[0] * (rect[1] - rect[0]), rect[2] + st[1] * (rect[3] - rect[2]))
        normal = wp.normalize(wp.cross(ds, dt))
        if (brep.prims[prim].flags & FLAG_FLIP) != 0:
            normal = -normal
    else:
        ok = False
    # Accepting the triangle shrinks traversal to its distance, not the refined root's. Only
    # prune when that distance is at least the exact hit: otherwise a different surface between
    # the triangle and the root could be nearer. Orientation cannot establish this ordering
    # (even two front-facing sheets can have overlapping shells of different thicknesses).
    keep_searching = hit_t > wp.optix_get_ray_tmax()
    return ok, hit_t, u, v, normal, brep.prims[prim].face, keep_searching


@wp.func
def intersect_curved(brep: BrepData, prim: int, origin: wp.vec3, direction: wp.vec3):
    """Intersect a torus or NURBS sub-patch primitive (a global BRep primitive index).

    Renderers run these in an intersection program apart from intersect_quadric: their code and
    register footprint would otherwise slow every quadric call (OptiX saves the registers a
    program uses on each invocation), while warps rarely mix them with quadrics.
    """
    if (brep.prims[prim].flags >> KIND_SHIFT) == TORUS:
        intersect_torus(brep, prim, origin, direction)
    else:
        intersect_nurbs(brep, prim, origin, direction)


def _device_array(values, dtype, device):
    return wp.array(np.ascontiguousarray(values), dtype=dtype, device=device)


def _anchored(prim):
    """The prim's control points relative to each sub-patch's anchor, and the anchors (see
    Surface.anchor). Anchors are float32 values, so the device adds back exactly what the
    host (in float64) took off."""
    cvs = np.asarray(prim.cvs, np.float64).copy()
    anchor = np.zeros((len(prim.prim_kind), 3))
    nurbs = np.flatnonzero(prim.prim_kind == NURBS)
    if len(nurbs):
        base = np.asarray(prim.prim_cv[nurbs], np.int64)
        count = np.prod(np.asarray(prim.prim_degree[nurbs], np.int64) + 1, axis=1)
        first = cvs[base]
        anchor[nurbs] = (first[:, :3] / first[:, 3:4]).astype(np.float32)
        index = np.repeat(base - np.cumsum(count) + count, count) + np.arange(count.sum())
        cvs[index, :3] -= np.repeat(anchor[nurbs], count, axis=0) * cvs[index, 3:4]
    return cvs, anchor


def _slab_data(prim, has_grid: np.ndarray):
    """The slab-test arrays of the prim's trimmed faces that have no cell grid (the grid answers
    the others): (face_slab, slab starts without the end, slab members, segments, bulges)."""
    face_slab = np.zeros_like(prim.face_slab)
    starts, members, segments, bulges = [], [], [], []
    slabs = nmembers = nsegments = 0
    for f in np.nonzero(((prim.face_flags & FLAG_TRIMMED) != 0) & ~has_grid)[0]:
        base, count = (int(x) for x in prim.face_slab[f])
        lo, hi = int(prim.slab_start[base]), int(prim.slab_start[base + count])
        ids = prim.slab_segments[lo:hi]
        first = int(ids.min()) if len(ids) else 0  # a face's chords are contiguous
        last = int(ids.max()) + 1 if len(ids) else 0
        face_slab[f] = (slabs, count)
        starts.append(prim.slab_start[base : base + count] - lo + nmembers)
        members.append(ids - first + nsegments)
        segments.append(prim.segments[first:last])
        bulges.append(prim.segment_bulge[first:last])
        slabs += count
        nmembers += hi - lo
        nsegments += last - first
    if not starts:
        return face_slab, np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros((0, 4), np.float32), np.zeros(0)
    return face_slab, np.concatenate(starts), np.concatenate(members), np.concatenate(segments), np.concatenate(bulges)


def pack(prims, device, proxies=None):
    """Concatenate loaded ``BrepPrim``s into one ``BrepData``.

    Returns it and each prim's primitive base; with ``proxies`` (one ``proxy.Proxy`` per prim),
    also each prim's shell-triangle base. CUDA bound fitting is enqueued on the current stream
    after uploading the arrays; CPU packing uses the reference fitter.
    """
    face_base = prim_base = cv_base = seg_base = slab_base = slab_seg_base = tri_base = 0
    cell_base = member_base = 0
    gpu_bounds = wp.get_device(device).is_cuda
    bound_ids = []
    cat = {k: [] for k in cast(Any, BrepData).vars if k not in ("prims", "diagnostic_counts")}
    record_fields = ("origin", "x", "y", "axis", "params", "range", "period", "flags", "grid", "grid_map")
    faces = {k: [] for k in record_fields}  # per face, expanded to its primitives below
    prim_kind, prim_face, anchors = [], [], []
    bases, tri_bases = [], []
    grids = trim_grid.build_trim_grids(prims, device)  # all prims' grids at once
    for n, prim in enumerate(prims):
        if proxies is not None:
            cat["proxy_prim"].append(proxies[n].tri_prim + prim_base)
            tri_bases.append(tri_base)
            tri_base += len(proxies[n].tri_prim)
        for k in ("origin", "x", "y", "axis", "params", "range", "period", "flags"):
            faces[k].append(getattr(prim, f"face_{k}"))
        grid, grid_map, cell_info, cell_chords, cell_bulges = grids[n]
        face_slab, slab_start, slab_members, segments, bulges = _slab_data(prim, grid[:, 1] > 0)
        # Without a cell grid, the grid fields hold the slab test's lookup (see Surface).
        slab = face_slab[:, 1] > 0
        grid = grid + np.array([cell_base, 0, 0]) * (grid[:, 1:2] > 0)
        grid[slab] = np.stack([face_slab[slab, 0] + slab_base, np.zeros(slab.sum()), face_slab[slab, 1]], axis=1)
        grid_map = np.array(grid_map, np.float32)
        grid_map[slab] = np.concatenate([prim.face_slab_map[slab], np.zeros((slab.sum(), 2))], axis=1)
        cat["slab_start"].append(slab_start + slab_seg_base)
        cat["slab_segments"].append(slab_members + seg_base)
        cat["segments"].append(segments)
        cat["segment_bulge"].append(bulges)
        faces["grid"].append(grid)
        faces["grid_map"].append(grid_map)
        cat["cell_info"].append(cell_info + np.array([member_base, 0]))
        cat["cell_chords"].append(cell_chords)
        cat["cell_bulges"].append(cell_bulges)
        cell_base += len(cell_info)
        member_base += len(cell_chords)
        prim_kind.append(prim.prim_kind)
        prim_face.append(prim.prim_face + face_base)
        cat["prim_cv"].append(prim.prim_cv + cv_base)
        cat["prim_degree"].append(prim.prim_degree)
        cat["prim_uv"].append(prim.prim_uv)
        cat["prim_tol"].append(prim.prim_tol)
        cvs, anchor = _anchored(prim)
        cat["cvs"].append(cvs)
        anchors.append(anchor)
        bases.append(prim_base)
        face_base += len(prim.face_kind)
        prim_base += len(prim.prim_kind)
        cv_base += len(prim.cvs)
        seg_base += len(segments)
        slab_base += len(slab_start)
        slab_seg_base += len(slab_members)
    cat["slab_start"].append(np.array([slab_seg_base]))
    from .proxy import SHELL_ST  # noqa: PLC0415 - proxy builds on this module's patch evaluation

    cat["shell_st"].append(SHELL_ST.reshape(-1, 2))
    data = BrepData()
    data.diagnostic_counts = wp.zeros(
        (prim_base if DIAGNOSTICS else 0, len(DIAGNOSTIC_NAMES)), dtype=wp.uint64, device=device
    )
    records = np.zeros(max(prim_base, 1), dtype=cast(Any, Surface).numpy_dtype())
    if prim_base:
        face = np.concatenate(prim_face)
        for key, parts in faces.items():
            values = np.concatenate([np.asarray(p).reshape(len(p), -1) for p in parts])
            records[key][:prim_base] = values[face].reshape(records[key][:prim_base].shape)
        records["flags"][:prim_base] |= np.concatenate(prim_kind).astype(np.int32) << int(KIND_SHIFT)
        records["face"][:prim_base] = face
        records["anchor"][:prim_base] = np.concatenate(anchors)
        for n, prim in enumerate(prims):
            nurbs = np.flatnonzero(prim.prim_kind == NURBS)
            for key in ("origin", "x", "y", "axis", "params"):
                records[key][bases[n] + nurbs] = 0
            # Shell candidates already have a tighter hardware test. Only fit active AABBs.
            if proxies is not None:
                nurbs = np.setdiff1d(nurbs, proxies[n].patches, assume_unique=True)
            if gpu_bounds:
                if len(nurbs):
                    bound_ids.append(bases[n] + nurbs)
                continue
            for local in nurbs:
                p, q = prim.prim_degree[local]
                start = prim.prim_cv[local]
                net = prim.cvs[start : start + (p + 1) * (q + 1)].reshape(p + 1, q + 1, 4)
                center, axes, reach = bounds.patch_obb(net, prim.prim_tol[local])
                index = bases[n] + local
                records["origin"][index] = center
                records["x"][index], records["y"][index], records["axis"][index] = axes.T
                records["params"][index, 0] = reach
    data.prims = wp.array(records, dtype=Surface, device=device)
    for key, parts in cat.items():
        dtype = cast(Any, BrepData).vars[key].type.dtype
        width = int(getattr(dtype, "_length_", 1))
        scalar = np.int32 if getattr(dtype, "_wp_scalar_type_", dtype) is wp.int32 else np.float32
        parts = [np.asarray(p) for p in parts if len(p)]
        values = np.concatenate(parts).astype(scalar) if parts else np.zeros(0, scalar)
        if values.size == 0:  # every field must be a valid array for the launch
            values = np.zeros(width, scalar)
        setattr(data, key, _device_array(values.reshape(-1, width) if width > 1 else values, dtype, device))
    if bound_ids:
        from .bounds_gpu import build_bounds  # noqa: PLC0415 - requires the structs defined in this module

        build_bounds(data, np.concatenate(bound_ids))
    if proxies is not None:
        return data, bases, tri_bases
    return data, bases
