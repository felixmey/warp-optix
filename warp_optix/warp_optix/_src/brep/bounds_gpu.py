# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched Warp construction of conservative patch bounds, using resident control nets.

Imported lazily by device.pack, after its structs are defined. Preprocessing uses float64;
only the final frames are rounded to the float32 representation used during tracing.
"""

import numpy as np
import warp as wp

from .bounds import _extension
from .device import BrepData, Surface

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def _extend(
    ids: wp.array[int],
    cvs: wp.array[wp.vec4],
    bases: wp.array[int],
    degrees: wp.array[wp.vec2i],
    records: wp.array[Surface],
    transforms: wp.array3d[wp.float64],
    points: wp.array2d[wp.vec4d],
):
    row, k = wp.tid()
    prim = ids[row]
    degree = degrees[prim]
    p = degree[0]
    q = degree[1]
    points[row, k] = wp.vec4d(wp.float64(0.0))
    if p < 0 or p > 7 or q < 0 or q > 7 or k >= (p + 1) * (q + 1):
        return
    a = k // (q + 1)
    b = k % (q + 1)
    h = wp.vec4d(wp.float64(0.0))
    for i in range(p + 1):
        for j in range(q + 1):
            cv = cvs[bases[prim] + i * (q + 1) + j]
            coefficient = transforms[p, a, i] * transforms[q, b, j]
            h += coefficient * wp.vec4d(cv)
    if h[3] <= wp.float64(0.0):
        return
    for j in range(4):
        if not wp.isfinite(h[j]):
            return
    point = wp.vec3d(h[0], h[1], h[2]) / h[3] + wp.vec3d(records[prim].anchor)  # cvs are anchor-relative
    # Match the host reference's safe float32 coordinate range.
    if wp.max(wp.abs(point)) > wp.float64(1.0633823332454027e37):
        return
    points[row, k] = wp.vec4d(point[0], point[1], point[2], wp.float64(1.0))


@wp.kernel
def _fit(
    ids: wp.array[int],
    degrees: wp.array[wp.vec2i],
    tolerances: wp.array[float],
    points: wp.array2d[wp.vec4d],
    records: wp.array[Surface],
):
    row = wp.tid()
    prim = ids[row]
    surf = records[prim]
    surf.origin = wp.vec3(0.0)
    surf.x = wp.vec3(0.0)
    surf.y = wp.vec3(0.0)
    surf.axis = wp.vec3(0.0)
    surf.params = wp.vec2(0.0)
    records[prim] = surf  # invalid input disables rejection, including on rebuild
    degree = degrees[prim]
    if degree[0] < 0 or degree[0] > 7 or degree[1] < 0 or degree[1] > 7:
        return
    tolerance = tolerances[prim]
    if not wp.isfinite(tolerance) or tolerance < 0.0:
        return
    count = (degree[0] + 1) * (degree[1] + 1)
    anchor = wp.vec3d(wp.float64(0.0))
    reach = wp.float64(1.0)
    for k in range(count):
        point = points[row, k]
        if point[3] == wp.float64(0.0):
            return
        xyz = wp.vec3d(point[0], point[1], point[2])
        anchor += xyz
        reach = wp.max(reach, wp.max(wp.abs(xyz)))
    anchor /= wp.float64(count)
    covariance = wp.mat33d(wp.float64(0.0))
    for k in range(count):
        point = points[row, k]
        relative = wp.vec3d(point[0], point[1], point[2]) - anchor
        covariance += wp.outer(relative, relative)
    scale = wp.max(covariance[0, 0], wp.max(covariance[1, 1], covariance[2, 2]))
    axes = wp.identity(n=3, dtype=wp.float64)
    if scale > wp.float64(0.0):
        axes, eigenvalues = wp.eig3(covariance / scale)
    # An approximate eigenbasis affects tightness, not containment: measure every extent in
    # that basis, and recompute around the rounded center before normalizing the axes.
    lo = wp.vec3d(wp.float64(1.0e300))
    hi = -lo
    for k in range(count):
        point = points[row, k]
        projected = wp.transpose(axes) * (wp.vec3d(point[0], point[1], point[2]) - anchor)
        lo = wp.min(lo, projected)
        hi = wp.max(hi, projected)
    center = wp.vec3(anchor + axes * (wp.float64(0.5) * (lo + hi)))
    half = wp.vec3d(wp.float64(0.0))
    for k in range(count):
        point = points[row, k]
        relative = wp.vec3d(point[0], point[1], point[2]) - wp.vec3d(center)
        half = wp.max(half, wp.abs(wp.transpose(axes) * relative))
    padding = wp.float64(4.0) * wp.float64(tolerance) + wp.float64(16.0 * 1.1920928955078125e-7) * reach
    half += wp.vec3d(padding)
    x = wp.vec3(wp.vec3d(axes[0, 0], axes[1, 0], axes[2, 0]) / half[0])
    y = wp.vec3(wp.vec3d(axes[0, 1], axes[1, 1], axes[2, 1]) / half[1])
    z = wp.vec3(wp.vec3d(axes[0, 2], axes[1, 2], axes[2, 2]) / half[2])
    for k in range(3):
        if not wp.isfinite(center[k]) or not wp.isfinite(x[k]) or not wp.isfinite(y[k]) or not wp.isfinite(z[k]):
            return
    surf.origin = center
    surf.x = x
    surf.y = y
    surf.axis = z
    surf.params[0] = wp.float32(reach)
    records[prim] = surf


def build_bounds(data: BrepData, primitive_ids: np.ndarray) -> None:
    """Fit only active NURBS AABBs; leave analytic and shell records untouched.

    Uses the current stream, with no device-to-host transfers or synchronization. Temporary
    storage is 2 KiB per active patch; the final records and all control nets already exist.
    """
    if len(primitive_ids) == 0:
        return
    device = data.cvs.device
    coefficients = np.zeros((8, 8, 8), np.float64)
    for n in range(8):
        coefficients[n, : n + 1, : n + 1] = _extension(n)
    ids = wp.array(primitive_ids, dtype=int, device=device)
    transforms = wp.array(coefficients, dtype=wp.float64, device=device)
    points = wp.empty((len(primitive_ids), 64), dtype=wp.vec4d, device=device)
    wp.launch(
        _extend,
        (len(primitive_ids), 64),
        [ids, data.cvs, data.prim_cv, data.prim_degree, data.prims, transforms, points],
        device=device,
    )
    wp.launch(_fit, len(primitive_ids), [ids, data.prim_degree, data.prim_tol, points, data.prims], device=device)
