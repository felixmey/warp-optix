# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side conservative oriented bounds for rational Bezier patches."""

from functools import lru_cache

import numpy as np

# Enclose in_square's [-1e-4, 1.0001], including float32 endpoint rounding.
PARAMETER_MARGIN = 1.1e-4


@lru_cache(maxsize=8)
def _extension(degree: int) -> np.ndarray:
    """Bernstein coefficients over the extended interval, by de Casteljau's blossom.

    Applying this to homogeneous control points extends the same rational surface, without
    sampling it or solving a potentially ill-conditioned interpolation system.
    """
    rows = []
    for i in range(degree + 1):
        work = np.eye(degree + 1)
        for t in [-PARAMETER_MARGIN] * (degree - i) + [1.0 + PARAMETER_MARGIN] * i:
            work = (1.0 - t) * work[:-1] + t * work[1:]
        rows.append(work[0])
    return np.asarray(rows)


def patch_obb(control: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return float32 (center, scaled axis columns, coordinate reach).

    The transformed box is [-1, 1]^3. Zero axes disable rejection for invalid weights or
    unrepresentable bounds. Fit the *uploaded float32* control net, not its float64 source.
    Positive extended weights put the extended surface inside its control points' convex hull.
    """
    disabled = (np.zeros(3, np.float32), np.zeros((3, 3), np.float32), 0.0)
    net = np.asarray(control, dtype=np.float32).astype(np.float64)
    if not np.isfinite(net).all() or not np.isfinite(tolerance) or tolerance < 0:
        return disabled
    p, q = np.asarray(net.shape[:2]) - 1
    extended = np.einsum("ai,ijc,bj->abc", _extension(int(p)), net, _extension(int(q))).reshape(-1, 4)
    if np.any(extended[:, 3] <= 0) or not np.isfinite(extended).all():
        return disabled
    points = extended[:, :3] / extended[:, 3:]
    if not np.isfinite(points).all() or np.abs(points).max() > np.finfo(np.float32).max / 32:
        return disabled
    anchor = points.mean(axis=0)
    relative = points - anchor
    try:
        _, axes = np.linalg.eigh(relative.T @ relative)
    except np.linalg.LinAlgError:
        return disabled
    projected = relative @ axes
    center = (anchor + axes @ (0.5 * (projected.min(axis=0) + projected.max(axis=0)))).astype(np.float32)
    reach = max(float(np.abs(points).max()), 1.0)
    # Cover solver residuals, quantization of the stored frame, and host arithmetic.
    # Ray-dependent arithmetic uncertainty is added by patch_obb_misses on the device.
    padding = 4.0 * tolerance + 16.0 * np.finfo(np.float32).eps * reach
    half = np.abs((points - center) @ axes).max(axis=0) + padding
    scaled = (axes / half).astype(np.float32)
    if not np.isfinite(center).all() or not np.isfinite(scaled).all():
        return disabled
    return center, scaled, reach
