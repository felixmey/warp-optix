# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the loader against the SMLib kernel (``pip install usd-brep``).

For every face, random UV samples are evaluated on the loader's surface and classified by the
UV point-in-trim test (the same crossing test the GPU runs). The kernel's ``Face.closest_point``
gives ground truth in 3D, independent of parameterization (SMLib scales plane UVs and uses
degrees internally): a point inside the trimmed face is at distance ~0 from it.

    uv run --extra usd --with usd-brep python examples/brep/validate_trims.py part.usda
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from warp_optix._src.brep import loader as brep_usd  # noqa: E402


def _crossings(segs: np.ndarray, bulge: np.ndarray, a: float, c: float) -> int:
    """Crossings of the +c ray from (a, c), plus flips inside chord-to-arc bulges (as on the device)."""
    straddle = (segs[:, 0] <= a) != (segs[:, 2] <= a)
    s = segs[straddle]
    t = (a - s[:, 0]) / (s[:, 2] - s[:, 0])
    count = np.count_nonzero(s[:, 1] + t * (s[:, 3] - s[:, 1]) > c)
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    length2 = dx * dx + dy * dy
    ok = (bulge != 0.0) & (length2 > 0.0)
    along = np.where(ok, ((a - segs[:, 0]) * dx + (c - segs[:, 1]) * dy) / np.where(ok, length2, 1.0), -1.0)
    side = ((c - segs[:, 1]) * dx - (a - segs[:, 0]) * dy) / np.sqrt(np.where(ok, length2, 1.0))
    arc = 4.0 * bulge * along * (1.0 - along)
    inside_arc = (
        ok
        & (along > 0)
        & (along < 1)
        & (((bulge > 0) & (side > 0) & (side < arc)) | ((bulge < 0) & (side < 0) & (side > arc)))
    )
    return int(count + np.count_nonzero(inside_arc))


def inside(prim: brep_usd.BrepPrim, face: int, uv: np.ndarray) -> np.ndarray:
    """NumPy mirror of the device trim test (range check + crossing count)."""
    umin, umax, vmin, vmax = prim.face_range[face]
    pu, pv = prim.face_period[face]
    u, v = uv[:, 0].copy(), uv[:, 1].copy()
    if pu:
        u = umin + np.mod(u - umin, pu)
    if pv:
        v = vmin + np.mod(v - vmin, pv)
    result = (u >= umin) & (u <= umax) & (v >= vmin) & (v <= vmax)
    flags = prim.face_flags[face]
    if not flags & brep_usd.FLAG_TRIMMED:
        return result
    a, b = (v, u) if flags & brep_usd.FLAG_SWAP else (u, v)
    base, count = prim.face_slab[face]
    lo, scale = prim.face_slab_map[face]
    slab = np.clip(np.floor((a - lo) * scale).astype(int), 0, count - 1)
    for i in range(len(uv)):
        s = base + slab[i]
        members = prim.slab_segments[prim.slab_start[s] : prim.slab_start[s + 1]]
        result[i] &= _crossings(prim.segments[members], prim.segment_bulge[members], a[i], b[i]) % 2 == 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("usd")
    parser.add_argument("--samples", type=int, default=400)
    args = parser.parse_args()

    import usd_brep  # noqa: PLC0415

    prims = brep_usd.load_stage(args.usd)
    breps = usd_brep.usd.import_breps(args.usd)
    faces = [f for b in breps for f in b.faces()]
    prim = prims[0]
    if len(faces) != len(prim.face_kind):
        print(f"face count mismatch: kernel {len(faces)} vs USD {len(prim.face_kind)}")
        return 1
    rng = np.random.default_rng(1)
    worst = 0
    for f, face in enumerate(faces):
        umin, umax, vmin, vmax = prim.face_range[f]
        uv = np.stack([rng.uniform(umin, umax, args.samples), rng.uniform(vmin, vmax, args.samples)], axis=1)
        points = prim.surfaces[f].evaluate(uv[:, 0], uv[:, 1])
        dist = np.array([face.closest_point(tuple(map(float, p)))[1] for p in points])
        diag = float(np.linalg.norm(np.ptp(points, axis=0))) + 1e-12
        truth = dist < 1e-6 * diag
        ambiguous = (dist >= 1e-6 * diag) & (dist < 1e-3 * diag)  # within the trim tessellation error
        mine = inside(prim, f, uv)
        wrong = int(np.count_nonzero((mine != truth) & ~ambiguous))
        surf_err = float(np.min(dist))  # the surface itself must touch the face
        worst = max(worst, wrong)
        kind = brep_usd.KIND_NAMES[prim.face_kind[f]]
        mark = "" if wrong == 0 and surf_err < 1e-6 * diag else "  <-- MISMATCH"
        print(
            f"face {f:3d} {kind:8s} in={np.count_nonzero(truth):4d} "
            f"wrong={wrong:4d} surface_err={surf_err:.2e} flags={prim.face_flags[f]}{mark}"
        )
    print("worst face mismatches:", worst)
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
