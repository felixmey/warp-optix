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

from typing import Any, cast

import numpy as np
import warp as wp

from . import trim_grid
from .loader import CONE, CYLINDER, FLAG_FLIP, FLAG_SOLID, FLAG_SWAP, FLAG_TRIMMED, NURBS, PLANE, SPHERE, TORUS

NEWTON_ITERATIONS = 10
# Retry a sub-patch from more seeds after finding the ray's exit only when the ray meets the
# surface this obliquely (|cos|): only a nearly tangent ray can also enter the same flat patch.
GRAZING_COS = 0.3
ATTRIBUTE_WORDS = 6
GRID_REF_A = wp.constant(trim_grid.REF_A)
GRID_REF_C = wp.constant(trim_grid.REF_C)


@wp.struct
class BrepData:
    face_origin: wp.array[wp.vec3]
    face_x: wp.array[wp.vec3]
    face_y: wp.array[wp.vec3]
    face_axis: wp.array[wp.vec3]
    face_params: wp.array[wp.vec2]
    face_range: wp.array[wp.vec4]
    face_period: wp.array[wp.vec2]
    face_flags: wp.array[wp.int32]
    face_slab: wp.array[wp.vec2i]
    face_slab_map: wp.array[wp.vec2]
    slab_start: wp.array[wp.int32]
    slab_segments: wp.array[wp.int32]
    segments: wp.array[wp.vec4]
    segment_bulge: wp.array[wp.float32]
    # Trim cell grids (see trim_grid.py): per face (cell base, cells along a, cells along c; 0
    # cells: slab test) and (lo a, lo c, cells per unit a, cells per unit c); per cell (first
    # member, member count << 1 | parity at its reference point); members are chord copies.
    face_grid: wp.array[wp.vec3i]
    face_grid_map: wp.array[wp.vec4]
    cell_info: wp.array[wp.vec2i]
    cell_chords: wp.array[wp.vec4]
    cell_bulges: wp.array[wp.float32]
    prim_kind: wp.array[wp.int32]
    prim_face: wp.array[wp.int32]
    prim_cv: wp.array[wp.int32]
    prim_degree: wp.array[wp.vec2i]
    prim_uv: wp.array[wp.vec4]
    prim_tol: wp.array[wp.float32]
    cvs: wp.array[wp.vec4]
    # Triangle shells (see proxy.py): each shell triangle's global primitive (its patch) and the
    # patch parameters (s, t) at its three corners.
    proxy_prim: wp.array[wp.int32]
    proxy_st: wp.array[wp.vec2]


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
def inside_trim_slab(b: BrepData, face: int, a: float, c: float) -> bool:
    """Crossings of a ray toward +c against the chords of the point's slab."""
    slab = b.face_slab[face]
    smap = b.face_slab_map[face]
    s = slab[0] + wp.clamp(int(wp.floor((a - smap[0]) * smap[1])), 0, slab[1] - 1)
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
def inside_trim(b: BrepData, face: int, u: float, v: float) -> bool:
    flags = b.face_flags[face]
    if (flags & FLAG_TRIMMED) == 0:
        return True
    a = u
    c = v
    if (flags & FLAG_SWAP) != 0:
        a = v
        c = u
    grid = b.face_grid[face]
    if grid[1] == 0:
        return inside_trim_slab(b, face, a, c)
    m = b.face_grid_map[face]
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
def accept_uv(b: BrepData, face: int, u_in: float, v_in: float):
    """Whether (u, v) lies in the face (range and trim); returns (inside, wrapped u, wrapped v)."""
    rng = b.face_range[face]
    period = b.face_period[face]
    u = wrap(u_in, rng[0], period[0])
    v = wrap(v_in, rng[2], period[1])
    eu = 1.0e-5 * (rng[1] - rng[0]) + 1.0e-7
    ev = 1.0e-5 * (rng[3] - rng[2]) + 1.0e-7
    if u < rng[0] - eu or u > rng[1] + eu or v < rng[2] - ev or v > rng[3] + ev:
        return False, u, v
    return inside_trim(b, face, u, v), u, v


@wp.func
def try_report(b: BrepData, face: int, t: float, u_in: float, v_in: float, normal: wp.vec3, direction: wp.vec3) -> bool:
    if t < wp.optix_get_ray_tmin() or t > wp.optix_get_ray_tmax():
        return False
    inside, u, v = accept_uv(b, face, u_in, v_in)
    if not inside:
        return False
    n = wp.normalize(normal)
    if (b.face_flags[face] & FLAG_FLIP) != 0:
        n = -n
    kind = wp.uint32(0)
    if wp.dot(n, direction) > 0.0 and (b.face_flags[face] & FLAG_SOLID) != 0:
        kind = wp.uint32(1)  # leaving the solid: the ray started inside or a trim leaks
    return wp.optix_report_intersection(
        t,
        kind,
        wp.optix_float_as_uint32(u),
        wp.optix_float_as_uint32(v),
        wp.optix_float_as_uint32(n[0]),
        wp.optix_float_as_uint32(n[1]),
        wp.optix_float_as_uint32(n[2]),
        wp.uint32(face),
    )


@wp.func
def local_ray(b: BrepData, face: int, origin: wp.vec3, direction: wp.vec3):
    """Ray in the face frame, re-centered near the surface origin for float32 precision."""
    o = b.face_origin[face]
    x = b.face_x[face]
    y = b.face_y[face]
    a = b.face_axis[face]
    shift = wp.dot(o - origin, direction) / wp.dot(direction, direction)
    rel = origin + shift * direction - o
    return (
        wp.vec3(wp.dot(rel, x), wp.dot(rel, y), wp.dot(rel, a)),
        wp.vec3(wp.dot(direction, x), wp.dot(direction, y), wp.dot(direction, a)),
        shift,
    )


@wp.func
def to_object(b: BrepData, face: int, n: wp.vec3) -> wp.vec3:
    return n[0] * b.face_x[face] + n[1] * b.face_y[face] + n[2] * b.face_axis[face]


# ---------------------------------------------------------------------------------------------
# Intersection programs, one per surface kind
# ---------------------------------------------------------------------------------------------


@wp.func
def intersect_plane(brep: BrepData, prim: int, origin: wp.vec3, direction: wp.vec3):
    face = brep.prim_face[prim]
    a = brep.face_axis[face]
    denom = wp.dot(direction, a)
    if wp.abs(denom) > 1.0e-20:
        t = wp.dot(brep.face_origin[face] - origin, a) / denom
        rel = origin + t * direction - brep.face_origin[face]
        try_report(brep, face, t, wp.dot(rel, brep.face_x[face]), wp.dot(rel, brep.face_y[face]), a, direction)


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
def intersect_cylinder(brep: BrepData, prim: int, origin: wp.vec3, direction: wp.vec3):
    face = brep.prim_face[prim]
    o, d, shift = local_ray(brep, face, origin, direction)
    r = brep.face_params[face][0]
    count, t0, t1 = solve_quadratic(
        d[0] * d[0] + d[1] * d[1], o[0] * d[0] + o[1] * d[1], o[0] * o[0] + o[1] * o[1] - r * r
    )
    for i in range(count):
        t = t0
        if i == 1:
            t = t1
        p = o + t * d
        if try_report(
            brep,
            face,
            t + shift,
            wp.atan2(p[1], p[0]),
            p[2],
            to_object(brep, face, wp.vec3(p[0], p[1], 0.0)),
            direction,
        ):
            break


@wp.func
def intersect_cone(brep: BrepData, prim: int, origin: wp.vec3, direction: wp.vec3):
    face = brep.prim_face[prim]
    o, d, shift = local_ray(brep, face, origin, direction)
    r = brep.face_params[face][0]
    k = brep.face_params[face][1]
    ro = r + k * o[2]
    count, t0, t1 = solve_quadratic(
        d[0] * d[0] + d[1] * d[1] - k * k * d[2] * d[2],
        o[0] * d[0] + o[1] * d[1] - k * ro * d[2],
        o[0] * o[0] + o[1] * o[1] - ro * ro,
    )
    for i in range(count):
        t = t0
        if i == 1:
            t = t1
        p = o + t * d
        radius = r + k * p[2]
        u = wp.atan2(p[1], p[0])
        if radius < 0.0:
            u = wp.atan2(-p[1], -p[0])
        er = wp.vec3(wp.cos(u), wp.sin(u), 0.0)
        # Su x Sv = R(v) (e_r - k e_z)
        n = radius * (er - wp.vec3(0.0, 0.0, k))
        if try_report(brep, face, t + shift, u, p[2], to_object(brep, face, n), direction):
            break


@wp.func
def intersect_sphere(brep: BrepData, prim: int, origin: wp.vec3, direction: wp.vec3):
    face = brep.prim_face[prim]
    o, d, shift = local_ray(brep, face, origin, direction)
    r = brep.face_params[face][0]
    count, t0, t1 = solve_quadratic(wp.dot(d, d), wp.dot(o, d), wp.dot(o, o) - r * r)
    for i in range(count):
        t = t0
        if i == 1:
            t = t1
        p = o + t * d
        v = wp.asin(wp.clamp(p[2] / r, -1.0, 1.0))
        if try_report(brep, face, t + shift, wp.atan2(p[1], p[0]), v, to_object(brep, face, p), direction):
            break


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
def monotone_split(c: vec5, order: int, pts: vec5, n: int):
    """Roots of poly_eval(order) on the monotone pieces [pts[i], pts[i+1]] -> new breakpoints."""
    out = vec5(0.0)
    out[0] = 0.0
    m = int(1)
    for i in range(n - 1):
        a = pts[i]
        b = pts[i + 1]
        fa = poly_eval(c, order, a)
        fb = poly_eval(c, order, b)
        if (fa < 0.0) != (fb < 0.0):
            out[m] = bisect(c, order, a, b)
            m += 1
    out[m] = 1.0
    return out, m + 1


@wp.func
def intersect_torus(brep: BrepData, prim: int, origin: wp.vec3, direction: wp.vec3):
    face = brep.prim_face[prim]
    o, d, shift = local_ray(brep, face, origin, direction)
    major = brep.face_params[face][0]
    minor = brep.face_params[face][1]
    bound = major + minor
    count, ta, tb = solve_quadratic(wp.dot(d, d), wp.dot(o, d), wp.dot(o, o) - bound * bound)
    ta = wp.max(ta, wp.optix_get_ray_tmin() - shift)
    tb = wp.min(tb, wp.optix_get_ray_tmax() - shift)
    if count == 2 and tb > ta:
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
        # Isolate roots through the derivatives: f'' (quadratic, closed form) splits [0, 1]
        # into pieces where f' is monotone; f' roots split it where f is monotone.
        pts = vec5(0.0)
        n = int(1)
        qc, q0, q1 = solve_quadratic(12.0 * c[4], 3.0 * c[3], 2.0 * c[2])
        for i in range(qc):
            r = q0
            if i == 1:
                r = q1
            if r > 0.0 and r < 1.0:
                pts[n] = r
                n += 1
        pts[n] = 1.0
        n += 1
        crit, m = monotone_split(c, 1, pts, n)
        for i in range(m - 1):
            fa = poly_eval(c, 0, crit[i])
            fb = poly_eval(c, 0, crit[i + 1])
            if (fa < 0.0) != (fb < 0.0):
                s = bisect(c, 0, crit[i], crit[i + 1])
                df = poly_eval(c, 1, s)
                if df != 0.0:
                    s = wp.clamp(s - poly_eval(c, 0, s) / df, crit[i], crit[i + 1])
                t = ta + s * (tb - ta)
                p = o + t * d
                rho = wp.sqrt(p[0] * p[0] + p[1] * p[1])
                nrm = p - wp.vec3(p[0], p[1], 0.0) * (major / wp.max(rho, 1.0e-30))
                if try_report(
                    brep,
                    face,
                    t + shift,
                    wp.atan2(p[1], p[0]),
                    wp.atan2(p[2], rho - major),
                    to_object(brep, face, nrm),
                    direction,
                ):
                    break


MAX_ORDER = 8  # Bezier degree <= 7 per direction
vec8 = wp.types.vector(length=MAX_ORDER, dtype=wp.float32)


@wp.func
def bernstein_all(n: int, x: float):
    """All degree-n Bernstein polynomials at x and their derivatives (NURBS Book A1.3)."""
    basis = vec8(0.0)
    basis[0] = 1.0
    deriv = vec8(0.0)
    x1 = 1.0 - x
    for j in range(1, n + 1):
        if j == n:  # basis holds degree n-1: dB_i^n = n (B_{i-1}^{n-1} - B_i^{n-1})
            prev = float(0.0)
            for k in range(n):
                deriv[k] = float(n) * (prev - basis[k])
                prev = basis[k]
            deriv[n] = float(n) * prev
        saved = float(0.0)
        for k in range(j):
            temp = basis[k]
            basis[k] = saved + x1 * temp
            saved = x * temp
        basis[j] = saved
    return basis, deriv


@wp.func
def eval_bezier(b: BrepData, base: int, p: int, q: int, s: float, t: float):
    """Rational tensor Bezier patch: point and first partial derivatives."""
    bu, dbu = bernstein_all(p, s)
    bv, dbv = bernstein_all(q, t)
    acc = wp.vec4(0.0)
    acc_s = wp.vec4(0.0)
    acc_t = wp.vec4(0.0)
    for i in range(p + 1):
        row = wp.vec4(0.0)
        row_t = wp.vec4(0.0)
        for j in range(q + 1):
            cv = b.cvs[base + i * (q + 1) + j]
            row += bv[j] * cv
            row_t += dbv[j] * cv
        acc += bu[i] * row
        acc_s += dbu[i] * row
        acc_t += bu[i] * row_t
    w = acc[3]
    point = wp.vec3(acc[0], acc[1], acc[2]) / w
    ds = (wp.vec3(acc_s[0], acc_s[1], acc_s[2]) - acc_s[3] * point) / w
    dt = (wp.vec3(acc_t[0], acc_t[1], acc_t[2]) - acc_t[3] * point) / w
    return point, ds, dt


@wp.struct
class Cubic:
    """A bicubic rational Bezier patch: homogeneous control points by component, row i = s index."""

    x: wp.mat44
    y: wp.mat44
    z: wp.mat44
    w: wp.mat44


@wp.func
def load_cubic(b: BrepData, base: int) -> Cubic:
    v0 = b.cvs[base + 0]
    v1 = b.cvs[base + 1]
    v2 = b.cvs[base + 2]
    v3 = b.cvs[base + 3]
    v4 = b.cvs[base + 4]
    v5 = b.cvs[base + 5]
    v6 = b.cvs[base + 6]
    v7 = b.cvs[base + 7]
    v8 = b.cvs[base + 8]
    v9 = b.cvs[base + 9]
    v10 = b.cvs[base + 10]
    v11 = b.cvs[base + 11]
    v12 = b.cvs[base + 12]
    v13 = b.cvs[base + 13]
    v14 = b.cvs[base + 14]
    v15 = b.cvs[base + 15]
    c = Cubic()
    c.x = wp.mat44(
        v0[0],
        v1[0],
        v2[0],
        v3[0],
        v4[0],
        v5[0],
        v6[0],
        v7[0],
        v8[0],
        v9[0],
        v10[0],
        v11[0],
        v12[0],
        v13[0],
        v14[0],
        v15[0],
    )
    c.y = wp.mat44(
        v0[1],
        v1[1],
        v2[1],
        v3[1],
        v4[1],
        v5[1],
        v6[1],
        v7[1],
        v8[1],
        v9[1],
        v10[1],
        v11[1],
        v12[1],
        v13[1],
        v14[1],
        v15[1],
    )
    c.z = wp.mat44(
        v0[2],
        v1[2],
        v2[2],
        v3[2],
        v4[2],
        v5[2],
        v6[2],
        v7[2],
        v8[2],
        v9[2],
        v10[2],
        v11[2],
        v12[2],
        v13[2],
        v14[2],
        v15[2],
    )
    c.w = wp.mat44(
        v0[3],
        v1[3],
        v2[3],
        v3[3],
        v4[3],
        v5[3],
        v6[3],
        v7[3],
        v8[3],
        v9[3],
        v10[3],
        v11[3],
        v12[3],
        v13[3],
        v14[3],
        v15[3],
    )
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
    w = wp.dot(bu, wv)
    point = wp.vec3(wp.dot(bu, xv), wp.dot(bu, yv), wp.dot(bu, zv)) / w
    ds = (wp.vec3(wp.dot(dbu, xv), wp.dot(dbu, yv), wp.dot(dbu, zv)) - wp.dot(dbu, wv) * point) / w
    dt = (wp.vec3(wp.dot(bu, xt), wp.dot(bu, yt), wp.dot(bu, zt)) - wp.dot(bu, wt) * point) / w
    return point, ds, dt


@wp.func
def newton_patch(
    brep: BrepData,
    base: int,
    deg: wp.vec2i,
    n1: wp.vec3,
    n2: wp.vec3,
    near: wp.vec3,
    tol: float,
    s_seed: float,
    t_seed: float,
):
    """Newton on the two-plane form of the ray from (s_seed, t_seed).

    Returns (converged, inside the patch, s, t, S, S_s, S_t).
    """
    s = s_seed
    t = t_seed
    converged = bool(False)
    point = wp.vec3(0.0)
    ds = wp.vec3(0.0)
    dt = wp.vec3(0.0)
    for _ in range(NEWTON_ITERATIONS):
        point, ds, dt = eval_bezier(brep, base, deg[0], deg[1], s, t)
        rel = point - near
        f0 = wp.dot(n1, rel)
        f1 = wp.dot(n2, rel)
        if wp.abs(f0) + wp.abs(f1) < tol:
            converged = True
            break
        j00 = wp.dot(n1, ds)
        j01 = wp.dot(n1, dt)
        j10 = wp.dot(n2, ds)
        j11 = wp.dot(n2, dt)
        det = j00 * j11 - j01 * j10
        if wp.abs(det) < 1.0e-30:
            break
        s = s - (j11 * f0 - j01 * f1) / det
        t = t - (-j10 * f0 + j00 * f1) / det
        if s < -0.5 or s > 1.5 or t < -0.5 or t > 1.5:
            break
    inside = s >= -1.0e-4 and s <= 1.0001 and t >= -1.0e-4 and t <= 1.0001
    return converged, converged and inside, s, t, point, ds, dt


@wp.func
def newton_cubic(
    c: Cubic,
    n1: wp.vec3,
    n2: wp.vec3,
    near: wp.vec3,
    tol: float,
    s_seed: float,
    t_seed: float,
):
    """Newton on the two-plane form of the ray from (s_seed, t_seed).

    Returns (converged, inside the patch, s, t, S, S_s, S_t).
    """
    s = s_seed
    t = t_seed
    converged = bool(False)
    point = wp.vec3(0.0)
    ds = wp.vec3(0.0)
    dt = wp.vec3(0.0)
    for _ in range(NEWTON_ITERATIONS):
        point, ds, dt = eval_cubic(c, s, t)
        rel = point - near
        f0 = wp.dot(n1, rel)
        f1 = wp.dot(n2, rel)
        if wp.abs(f0) + wp.abs(f1) < tol:
            converged = True
            break
        j00 = wp.dot(n1, ds)
        j01 = wp.dot(n1, dt)
        j10 = wp.dot(n2, ds)
        j11 = wp.dot(n2, dt)
        det = j00 * j11 - j01 * j10
        if wp.abs(det) < 1.0e-30:
            break
        s = s - (j11 * f0 - j01 * f1) / det
        t = t - (-j10 * f0 + j00 * f1) / det
        if s < -0.5 or s > 1.5 or t < -0.5 or t > 1.5:
            break
    inside = s >= -1.0e-4 and s <= 1.0001 and t >= -1.0e-4 and t <= 1.0001
    return converged, converged and inside, s, t, point, ds, dt


@wp.func
def report_patch_hit(
    brep: BrepData,
    prim: int,
    face: int,
    s: float,
    t: float,
    point: wp.vec3,
    ds: wp.vec3,
    dt: wp.vec3,
    near: wp.vec3,
    shift: float,
    direction: wp.vec3,
) -> float:
    """Report a sub-patch root; returns the cosine between the outward normal and the ray there."""
    rect = brep.prim_uv[prim]
    normal = wp.cross(ds, dt)
    try_report(
        brep,
        face,
        shift + wp.dot(point - near, direction) / wp.dot(direction, direction),
        rect[0] + s * (rect[1] - rect[0]),
        rect[2] + t * (rect[3] - rect[2]),
        normal,
        direction,
    )
    if (brep.face_flags[face] & FLAG_FLIP) != 0:
        normal = -normal
    return wp.dot(wp.normalize(normal), wp.normalize(direction))


@wp.func
def intersect_nurbs_generic(brep: BrepData, prim: int, origin: wp.vec3, direction: wp.vec3):
    """Rational Bezier patches of any degree up to 7, read from global memory."""
    face = brep.prim_face[prim]
    deg = brep.prim_degree[prim]
    base = brep.prim_cv[prim]
    tol = brep.prim_tol[prim]
    # The ray as the intersection of two planes through it.
    dn = wp.normalize(direction)
    n1 = wp.vec3(0.0)
    if wp.abs(dn[0]) > wp.abs(dn[1]) and wp.abs(dn[0]) > wp.abs(dn[2]):
        n1 = wp.normalize(wp.vec3(dn[1], -dn[0], 0.0))
    else:
        n1 = wp.normalize(wp.vec3(0.0, dn[2], -dn[1]))
    n2 = wp.cross(n1, dn)
    # Re-center the ray origin next to the patch: with a distant origin, n . (S - o) cancels
    # in float32 down to the convergence tolerance and Newton fails at random pixels.
    corner = brep.cvs[base]
    dd = wp.dot(direction, direction)
    shift = wp.dot(wp.vec3(corner[0], corner[1], corner[2]) / corner[3] - origin, direction) / dd
    near = origin + shift * direction
    # Convex-hull rejection (the first step of Bezier clipping): the ray lies on both planes, so
    # it can only hit the patch if the control points straddle each plane. Weights are positive,
    # so the patch lies in the hull of its Euclidean control points.
    # The same pass picks the control point nearest the ray as the Newton seed: control point
    # (i, j) of a Bezier patch sits at about parameter (i / p, j / q).
    lo1 = float(1.0e30)
    hi1 = float(-1.0e30)
    lo2 = float(1.0e30)
    hi2 = float(-1.0e30)
    best = float(1.0e30)
    best_index = int(0)
    nearest = float(1.0e30)
    for i in range((deg[0] + 1) * (deg[1] + 1)):
        cv = brep.cvs[base + i]
        rel = wp.vec3(cv[0], cv[1], cv[2]) / cv[3] - near
        d1 = wp.dot(n1, rel)
        d2 = wp.dot(n2, rel)
        lo1 = wp.min(lo1, d1)
        hi1 = wp.max(hi1, d1)
        lo2 = wp.min(lo2, d2)
        hi2 = wp.max(hi2, d2)
        nearest = wp.min(nearest, wp.dot(rel, direction))
        dist = d1 * d1 + d2 * d2
        if dist < best:
            best = dist
            best_index = i
    if lo1 > tol or hi1 < -tol or lo2 > tol or hi2 < -tol:
        return
    # The patch lies beyond the closest hit found so far.
    hull_tmin = shift + nearest / dd - tol / wp.sqrt(dd)
    if hull_tmin > wp.optix_get_ray_tmax():
        return
    # A ray starting within the patch's depth range (a bounce or shadow ray leaving this surface)
    # is not the silhouette case the retries are for: Newton finds its own origin, below tmin.
    arriving = hull_tmin > wp.optix_get_ray_tmin()
    seed_s = float(best_index // (deg[1] + 1)) / float(wp.max(deg[0], 1))
    seed_t = float(best_index % (deg[1] + 1)) / float(wp.max(deg[1], 1))
    # Newton fails or finds the wrong root only where the ray is nearly tangent to the surface:
    # a grazing ray crosses a curved sub-patch twice near a silhouette, entering and leaving.
    # Every root found inside the patch is reported (OptiX keeps the nearest). Without a root
    # where the ray enters, and only when the ray grazes the surface at Newton's last point:
    # - after divergence, retry once from the patch center;
    # - from an exit root, the entry lies upstream along the ray: step back along the ray's
    #   direction mapped into the parameter domain, and seed Newton there.
    converged, ok, s, t, point, ds, dt = newton_patch(brep, base, deg, n1, n2, near, tol, seed_s, seed_t)
    retry = arriving and ((not converged) or wp.abs(wp.dot(wp.normalize(wp.cross(ds, dt)), dn)) < GRAZING_COS)
    if ok:
        # Report first: the hit must be reported whether or not the ray grazes.
        cosine = report_patch_hit(brep, prim, face, s, t, point, ds, dt, near, shift, direction)
        retry = retry and cosine >= 0.0
    if retry and not ok:
        converged, ok, s, t, point, ds, dt = newton_patch(brep, base, deg, n1, n2, near, tol, 0.5, 0.5)
        if ok:
            retry = report_patch_hit(brep, prim, face, s, t, point, ds, dt, near, shift, direction) >= 0.0
    if retry and ok:
        # Least-squares parameter step that moves the surface point by -dn.
        a00 = wp.dot(ds, ds)
        a01 = wp.dot(ds, dt)
        a11 = wp.dot(dt, dt)
        b0 = -wp.dot(ds, dn)
        b1 = -wp.dot(dt, dn)
        det = a00 * a11 - a01 * a01
        if wp.abs(det) > 1.0e-30:
            step = wp.vec2((a11 * b0 - a01 * b1) / det, (a00 * b1 - a01 * b0) / det)
            length = wp.length(step)
            if length > 0.0:
                step = step / length
                exit_s = s
                exit_t = t
                for k in range(2):
                    reach = 0.35 * float(k + 1)
                    s0 = wp.clamp(exit_s + reach * step[0], 0.0, 1.0)
                    t0 = wp.clamp(exit_t + reach * step[1], 0.0, 1.0)
                    converged, ok, s, t, point, ds, dt = newton_patch(brep, base, deg, n1, n2, near, tol, s0, t0)
                    if ok:
                        if report_patch_hit(brep, prim, face, s, t, point, ds, dt, near, shift, direction) < 0.0:
                            retry = False
                            break
    if retry:
        # Last resort for grazing rays still without an entry: seeds near the patch corners.
        for k in range(4):
            s0 = 0.15
            t0 = 0.15
            if k == 1 or k == 3:
                s0 = 0.85
            if k >= 2:
                t0 = 0.85
            converged, ok, s, t, point, ds, dt = newton_patch(brep, base, deg, n1, n2, near, tol, s0, t0)
            if ok:
                if report_patch_hit(brep, prim, face, s, t, point, ds, dt, near, shift, direction) < 0.0:
                    break


@wp.func
def intersect_nurbs_cubic(brep: BrepData, prim: int, origin: wp.vec3, direction: wp.vec3):
    """``intersect_nurbs`` for bicubic patches, with the control points held in registers."""
    face = brep.prim_face[prim]
    deg = brep.prim_degree[prim]
    base = brep.prim_cv[prim]
    c = load_cubic(brep, base)
    tol = brep.prim_tol[prim]
    # The ray as the intersection of two planes through it.
    dn = wp.normalize(direction)
    n1 = wp.vec3(0.0)
    if wp.abs(dn[0]) > wp.abs(dn[1]) and wp.abs(dn[0]) > wp.abs(dn[2]):
        n1 = wp.normalize(wp.vec3(dn[1], -dn[0], 0.0))
    else:
        n1 = wp.normalize(wp.vec3(0.0, dn[2], -dn[1]))
    n2 = wp.cross(n1, dn)
    # Re-center the ray origin next to the patch: with a distant origin, n . (S - o) cancels
    # in float32 down to the convergence tolerance and Newton fails at random pixels.
    corner = wp.vec4(c.x[0, 0], c.y[0, 0], c.z[0, 0], c.w[0, 0])
    dd = wp.dot(direction, direction)
    shift = wp.dot(wp.vec3(corner[0], corner[1], corner[2]) / corner[3] - origin, direction) / dd
    near = origin + shift * direction
    # Convex-hull rejection (the first step of Bezier clipping): the ray lies on both planes, so
    # it can only hit the patch if the control points straddle each plane. Weights are positive,
    # so the patch lies in the hull of its Euclidean control points.
    # The same pass picks the control point nearest the ray as the Newton seed: control point
    # (i, j) of a Bezier patch sits at about parameter (i / p, j / q).
    lo1 = float(1.0e30)
    hi1 = float(-1.0e30)
    lo2 = float(1.0e30)
    hi2 = float(-1.0e30)
    best = float(1.0e30)
    best_index = int(0)
    nearest = float(1.0e30)
    for i in range((deg[0] + 1) * (deg[1] + 1)):
        cv = wp.vec4(c.x[i // 4, i % 4], c.y[i // 4, i % 4], c.z[i // 4, i % 4], c.w[i // 4, i % 4])
        rel = wp.vec3(cv[0], cv[1], cv[2]) / cv[3] - near
        d1 = wp.dot(n1, rel)
        d2 = wp.dot(n2, rel)
        lo1 = wp.min(lo1, d1)
        hi1 = wp.max(hi1, d1)
        lo2 = wp.min(lo2, d2)
        hi2 = wp.max(hi2, d2)
        nearest = wp.min(nearest, wp.dot(rel, direction))
        dist = d1 * d1 + d2 * d2
        if dist < best:
            best = dist
            best_index = i
    if lo1 > tol or hi1 < -tol or lo2 > tol or hi2 < -tol:
        return
    # The patch lies beyond the closest hit found so far.
    hull_tmin = shift + nearest / dd - tol / wp.sqrt(dd)
    if hull_tmin > wp.optix_get_ray_tmax():
        return
    # A ray starting within the patch's depth range (a bounce or shadow ray leaving this surface)
    # is not the silhouette case the retries are for: Newton finds its own origin, below tmin.
    arriving = hull_tmin > wp.optix_get_ray_tmin()
    seed_s = float(best_index // (deg[1] + 1)) / float(wp.max(deg[0], 1))
    seed_t = float(best_index % (deg[1] + 1)) / float(wp.max(deg[1], 1))
    # Newton fails or finds the wrong root only where the ray is nearly tangent to the surface:
    # a grazing ray crosses a curved sub-patch twice near a silhouette, entering and leaving.
    # Every root found inside the patch is reported (OptiX keeps the nearest). Without a root
    # where the ray enters, and only when the ray grazes the surface at Newton's last point:
    # - after divergence, retry once from the patch center;
    # - from an exit root, the entry lies upstream along the ray: step back along the ray's
    #   direction mapped into the parameter domain, and seed Newton there.
    converged, ok, s, t, point, ds, dt = newton_cubic(c, n1, n2, near, tol, seed_s, seed_t)
    retry = arriving and ((not converged) or wp.abs(wp.dot(wp.normalize(wp.cross(ds, dt)), dn)) < GRAZING_COS)
    if ok:
        # Report first: the hit must be reported whether or not the ray grazes.
        cosine = report_patch_hit(brep, prim, face, s, t, point, ds, dt, near, shift, direction)
        retry = retry and cosine >= 0.0
    if retry and not ok:
        converged, ok, s, t, point, ds, dt = newton_cubic(c, n1, n2, near, tol, 0.5, 0.5)
        if ok:
            retry = report_patch_hit(brep, prim, face, s, t, point, ds, dt, near, shift, direction) >= 0.0
    if retry and ok:
        # Least-squares parameter step that moves the surface point by -dn.
        a00 = wp.dot(ds, ds)
        a01 = wp.dot(ds, dt)
        a11 = wp.dot(dt, dt)
        b0 = -wp.dot(ds, dn)
        b1 = -wp.dot(dt, dn)
        det = a00 * a11 - a01 * a01
        if wp.abs(det) > 1.0e-30:
            step = wp.vec2((a11 * b0 - a01 * b1) / det, (a00 * b1 - a01 * b0) / det)
            length = wp.length(step)
            if length > 0.0:
                step = step / length
                exit_s = s
                exit_t = t
                for k in range(2):
                    reach = 0.35 * float(k + 1)
                    s0 = wp.clamp(exit_s + reach * step[0], 0.0, 1.0)
                    t0 = wp.clamp(exit_t + reach * step[1], 0.0, 1.0)
                    converged, ok, s, t, point, ds, dt = newton_cubic(c, n1, n2, near, tol, s0, t0)
                    if ok:
                        if report_patch_hit(brep, prim, face, s, t, point, ds, dt, near, shift, direction) < 0.0:
                            retry = False
                            break
    if retry:
        # Last resort for grazing rays still without an entry: seeds near the patch corners.
        for k in range(4):
            s0 = 0.15
            t0 = 0.15
            if k == 1 or k == 3:
                s0 = 0.85
            if k >= 2:
                t0 = 0.85
            converged, ok, s, t, point, ds, dt = newton_cubic(c, n1, n2, near, tol, s0, t0)
            if ok:
                if report_patch_hit(brep, prim, face, s, t, point, ds, dt, near, shift, direction) < 0.0:
                    break


@wp.func
def intersect_nurbs(brep: BrepData, prim: int, origin: wp.vec3, direction: wp.vec3):
    deg = brep.prim_degree[prim]
    if deg[0] == 3 and deg[1] == 3:
        intersect_nurbs_cubic(brep, prim, origin, direction)
    else:
        intersect_nurbs_generic(brep, prim, origin, direction)


@wp.func
def _proxy_hit_nurbs(brep: BrepData, prim: int, tri: int, seed: wp.vec2, origin: wp.vec3, direction: wp.vec3):
    """Exact surface hit for a ray that crossed shell triangle ``tri`` (global index).

    Newton starts from the patch parameters interpolated at the triangle hit, then the face's
    range and trim decide. Returns (hit, t, u, v, outward object-space normal, face).
    """
    face = brep.prim_face[prim]
    c = load_cubic(brep, brep.prim_cv[prim])
    tol = brep.prim_tol[prim]
    dn = wp.normalize(direction)
    n1 = wp.vec3(0.0)
    if wp.abs(dn[0]) > wp.abs(dn[1]) and wp.abs(dn[0]) > wp.abs(dn[2]):
        n1 = wp.normalize(wp.vec3(dn[1], -dn[0], 0.0))
    else:
        n1 = wp.normalize(wp.vec3(0.0, dn[2], -dn[1]))
    n2 = wp.cross(n1, dn)
    dd = wp.dot(direction, direction)
    corner = wp.vec3(c.x[0, 0], c.y[0, 0], c.z[0, 0]) / c.w[0, 0]
    shift = wp.dot(corner - origin, direction) / dd
    near = origin + shift * direction
    converged, ok, s, t, point, ds, dt = newton_cubic(
        c, n1, n2, near, tol, wp.clamp(seed[0], 0.0, 1.0), wp.clamp(seed[1], 0.0, 1.0)
    )
    # Rare (grazing rays on strongly curved patches): retry from the triangle's corners, which
    # bracket the crossing in parameter space, then from the patch center.
    for k in range(4):
        if ok:
            break
        retry = wp.vec2(0.5, 0.5)
        if k < 3:
            retry = brep.proxy_st[3 * tri + k]
        converged, ok, s, t, point, ds, dt = newton_cubic(c, n1, n2, near, tol, retry[0], retry[1])
    # A grazing ray can cross a curved patch twice; from an exit root, look upstream along the
    # ray (in parameter space) for the entry, like the custom intersector does.
    if ok:
        sn = wp.cross(ds, dt)
        if (brep.face_flags[brep.prim_face[prim]] & FLAG_FLIP) != 0:
            sn = -sn
        cosine = wp.dot(wp.normalize(sn), dn)
        if cosine > 0.0 and cosine < GRAZING_COS:
            a00 = wp.dot(ds, ds)
            a01 = wp.dot(ds, dt)
            a11 = wp.dot(dt, dt)
            b0 = -wp.dot(ds, dn)
            b1 = -wp.dot(dt, dn)
            det = a00 * a11 - a01 * a01
            if wp.abs(det) > 1.0e-30:
                step = wp.vec2((a11 * b0 - a01 * b1) / det, (a00 * b1 - a01 * b0) / det)
                length = wp.length(step)
                if length > 0.0:
                    step = step / length
                    exit_t = shift + wp.dot(point - near, direction) / dd
                    for k in range(2):
                        reach = 0.35 * float(k + 1)
                        c2, ok2, s2, t2, p2, ds2, dt2 = newton_cubic(
                            c,
                            n1,
                            n2,
                            near,
                            tol,
                            wp.clamp(s + reach * step[0], 0.0, 1.0),
                            wp.clamp(t + reach * step[1], 0.0, 1.0),
                        )
                        if ok2 and shift + wp.dot(p2 - near, direction) / dd < exit_t - tol:
                            s = s2
                            t = t2
                            point = p2
                            ds = ds2
                            dt = dt2
                            break
    hit_t = shift + wp.dot(point - near, direction) / dd
    normal = wp.vec3(0.0)
    u = float(0.0)
    v = float(0.0)
    if ok and hit_t >= wp.optix_get_ray_tmin():
        rect = brep.prim_uv[prim]
        inside, u, v = accept_uv(brep, face, rect[0] + s * (rect[1] - rect[0]), rect[2] + t * (rect[3] - rect[2]))
        ok = inside
        normal = wp.normalize(wp.cross(ds, dt))
        if (brep.face_flags[face] & FLAG_FLIP) != 0:
            normal = -normal
    else:
        ok = False
    return ok, hit_t, u, v, normal, face


@wp.func
def _analytic_root_near(brep: BrepData, prim: int, seed: wp.vec2, origin: wp.vec3, direction: wp.vec3):
    """Closed-form hit of a plane, cylinder, cone or sphere face, taking the root whose (u, v) is
    nearest the shell's seed. Returns (found, t, u, v, object-space Su x Sv normal)."""
    face = brep.prim_face[prim]
    kind = brep.prim_kind[prim]
    if kind == PLANE:
        a = brep.face_axis[face]
        denom = wp.dot(direction, a)
        if wp.abs(denom) < 1.0e-20:
            return False, 0.0, 0.0, 0.0, a
        t = wp.dot(brep.face_origin[face] - origin, a) / denom
        rel = origin + t * direction - brep.face_origin[face]
        return True, t, wp.dot(rel, brep.face_x[face]), wp.dot(rel, brep.face_y[face]), a
    o, d, shift = local_ray(brep, face, origin, direction)
    r = brep.face_params[face][0]
    k = brep.face_params[face][1]
    count = int(0)
    t0 = float(0.0)
    t1 = float(0.0)
    if kind == CYLINDER:
        count, t0, t1 = solve_quadratic(
            d[0] * d[0] + d[1] * d[1], o[0] * d[0] + o[1] * d[1], o[0] * o[0] + o[1] * o[1] - r * r
        )
    elif kind == CONE:
        ro = r + k * o[2]
        count, t0, t1 = solve_quadratic(
            d[0] * d[0] + d[1] * d[1] - k * k * d[2] * d[2],
            o[0] * d[0] + o[1] * d[1] - k * ro * d[2],
            o[0] * o[0] + o[1] * o[1] - ro * ro,
        )
    else:
        count, t0, t1 = solve_quadratic(wp.dot(d, d), wp.dot(o, d), wp.dot(o, o) - r * r)
    period = brep.face_period[face][0]
    best = float(1.0e30)
    found = bool(False)
    best_t = float(0.0)
    best_u = float(0.0)
    best_v = float(0.0)
    best_n = wp.vec3(0.0)
    for i in range(count):
        t = t0
        if i == 1:
            t = t1
        p = o + t * d
        u = wp.atan2(p[1], p[0])
        v = p[2]
        n = wp.vec3(p[0], p[1], 0.0)
        if kind == CONE:
            radius = r + k * p[2]
            if radius < 0.0:
                u = wp.atan2(-p[1], -p[0])
            er = wp.vec3(wp.cos(u), wp.sin(u), 0.0)
            n = radius * (er - wp.vec3(0.0, 0.0, k))
        elif kind == SPHERE:
            v = wp.asin(wp.clamp(p[2] / r, -1.0, 1.0))
            n = p
        du = u - seed[0]
        if period > 0.0:
            du = du - period * wp.round(du / period)
        distance = du * du + (v - seed[1]) * (v - seed[1])
        if distance < best:
            best = distance
            found = True
            best_t = t + shift
            best_u = seed[0] + du  # the root's u on the seed's branch
            best_v = v
            best_n = to_object(brep, face, n)
    return found, best_t, best_u, best_v, best_n


@wp.func
def proxy_hit(brep: BrepData, tri: int, bary: wp.vec2, origin: wp.vec3, direction: wp.vec3):
    """Exact surface hit for a ray that crossed proxy triangle ``tri`` (global index).

    NURBS shells: Newton from the patch parameters interpolated at the triangle hit. Analytic
    faces: the closed-form root nearest the interpolated face (u, v). Then the face's range and
    trim decide. Returns (hit, t, u, v, outward object-space normal, face, exact); ``exact`` means
    the triangle lies on the surface (planes), so its t is the surface's.
    """
    prim = brep.proxy_prim[tri]
    seed = (
        (1.0 - bary[0] - bary[1]) * brep.proxy_st[3 * tri]
        + bary[0] * brep.proxy_st[3 * tri + 1]
        + bary[1] * brep.proxy_st[3 * tri + 2]
    )
    kind = brep.prim_kind[prim]
    if kind == NURBS:
        ok, hit_t, u, v, normal, face = _proxy_hit_nurbs(brep, prim, tri, seed, origin, direction)
        return ok, hit_t, u, v, normal, face, False
    face = brep.prim_face[prim]
    found, hit_t, u_raw, v_raw, n = _analytic_root_near(brep, prim, seed, origin, direction)
    if not found or hit_t < wp.optix_get_ray_tmin():
        return False, hit_t, 0.0, 0.0, n, face, False
    inside, u, v = accept_uv(brep, face, u_raw, v_raw)
    normal = wp.normalize(n)
    if (brep.face_flags[face] & FLAG_FLIP) != 0:
        normal = -normal
    return inside, hit_t, u, v, normal, face, kind == PLANE


@wp.func
def intersect(brep: BrepData, prim: int, origin: wp.vec3, direction: wp.vec3):
    """Intersect primitive ``prim`` (a global BRep primitive index) by its surface kind."""
    kind = brep.prim_kind[prim]
    if kind == PLANE:
        intersect_plane(brep, prim, origin, direction)
    elif kind == CYLINDER:
        intersect_cylinder(brep, prim, origin, direction)
    elif kind == CONE:
        intersect_cone(brep, prim, origin, direction)
    elif kind == SPHERE:
        intersect_sphere(brep, prim, origin, direction)
    elif kind == TORUS:
        intersect_torus(brep, prim, origin, direction)
    elif kind == NURBS:
        intersect_nurbs(brep, prim, origin, direction)


def _device_array(values, dtype, device):
    return wp.array(np.ascontiguousarray(values), dtype=dtype, device=device)


def pack(prims, device, proxies=None):
    """Concatenate loaded ``BrepPrim``s into one ``BrepData``.

    Returns it and each prim's primitive base; with ``proxies`` (one ``proxy.Proxy`` per prim),
    also each prim's shell-triangle base.
    """
    face_base = prim_base = cv_base = seg_base = slab_base = slab_seg_base = tri_base = 0
    cell_base = member_base = 0
    cat = {k: [] for k in cast(Any, BrepData).vars}
    bases, tri_bases = [], []
    for n, prim in enumerate(prims):
        if proxies is not None:
            cat["proxy_prim"].append(proxies[n].tri_prim + prim_base)
            cat["proxy_st"].append(proxies[n].tri_st.reshape(-1, 2))
            tri_bases.append(tri_base)
            tri_base += len(proxies[n].tri_prim)
        cat["face_origin"].append(prim.face_origin)
        cat["face_x"].append(prim.face_x)
        cat["face_y"].append(prim.face_y)
        cat["face_axis"].append(prim.face_axis)
        cat["face_params"].append(prim.face_params)
        cat["face_range"].append(prim.face_range)
        cat["face_period"].append(prim.face_period)
        cat["face_flags"].append(prim.face_flags)
        cat["face_slab"].append(prim.face_slab + np.array([slab_base, 0]))
        cat["face_slab_map"].append(prim.face_slab_map)
        cat["slab_start"].append(prim.slab_start[:-1] + slab_seg_base)
        cat["slab_segments"].append(prim.slab_segments + seg_base)
        cat["segments"].append(prim.segments)
        cat["segment_bulge"].append(prim.segment_bulge)
        grid, grid_map, cell_info, cell_chords, cell_bulges = trim_grid.build_trim_grids(prim)
        cat["face_grid"].append(grid + np.array([cell_base, 0, 0]) * (grid[:, 1:2] > 0))
        cat["face_grid_map"].append(grid_map)
        cat["cell_info"].append(cell_info + np.array([member_base, 0]))
        cat["cell_chords"].append(cell_chords)
        cat["cell_bulges"].append(cell_bulges)
        cell_base += len(cell_info)
        member_base += len(cell_chords)
        cat["prim_kind"].append(prim.prim_kind)
        cat["prim_face"].append(prim.prim_face + face_base)
        cat["prim_cv"].append(prim.prim_cv + cv_base)
        cat["prim_degree"].append(prim.prim_degree)
        cat["prim_uv"].append(prim.prim_uv)
        cat["prim_tol"].append(prim.prim_tol)
        cat["cvs"].append(prim.cvs)
        bases.append(prim_base)
        face_base += len(prim.face_kind)
        prim_base += len(prim.prim_kind)
        cv_base += len(prim.cvs)
        seg_base += len(prim.segments)
        slab_base += len(prim.slab_start) - 1
        slab_seg_base += len(prim.slab_segments)
    cat["slab_start"].append(np.array([slab_seg_base]))
    data = BrepData()
    for key, parts in cat.items():
        dtype = cast(Any, BrepData).vars[key].type.dtype
        width = int(getattr(dtype, "_length_", 1))
        scalar = np.int32 if getattr(dtype, "_wp_scalar_type_", dtype) is wp.int32 else np.float32
        parts = [np.asarray(p) for p in parts if len(p)]
        values = np.concatenate(parts).astype(scalar) if parts else np.zeros(0, scalar)
        if values.size == 0:  # every field must be a valid array for the launch
            values = np.zeros(width, scalar)
        setattr(data, key, _device_array(values.reshape(-1, width) if width > 1 else values, dtype, device))
    if proxies is not None:
        return data, bases, tri_bases
    return data, bases
