# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-face cell grids that answer most trim tests with one lookup.

A trimmed face's parameter rectangle (in the trim's (a, c) axes, see ``loader._build_trim``) is
covered by a uniform grid. A cell that no trim chord (or chord-to-curve bulge) touches lies
entirely inside or entirely outside the face: its answer is stored. A cell that one does stores
the chords touching it and the polygon parity at a reference point R inside it; the device then
counts crossings of the segment R -> P with those chords only, which gives P's polygon parity,
and flips it inside bulges as the slab test does.

The parity at R is the slab test's (a vertical ray toward +c), so the two agree wherever both
are exact. Moving from R to P changes it once per chord crossed only when the trim chords form
closed loops over the face's a-range (with the chords the slab buckets drop restored); a face
with an open trim polyline there keeps the slab test.
"""

from __future__ import annotations

import numpy as np

from .loader import FLAG_SWAP, FLAG_TRIMMED, BrepPrim

# The reference point's position within its cell: off-center, so that it does not land on the
# chords of holes centered in cells (common in CAD).
REF_A = 0.5 + 0.0123
REF_C = 0.5 - 0.0171
MAX_CELLS_PER_AXIS = 64
CELLS_PER_SQRT_SEGMENT = 2.0


def _restore_parallel_chords(segs: np.ndarray, lo: float, hi: float):
    """The slab buckets drop chords parallel to the c axis (a ray toward +c never crosses them);
    rebuild them from the chord ends they leave unpaired, or return None if some unpaired end
    cannot be explained that way (an open polyline over the face's a-range).

    Over closed loops, the unpaired ends on one line a = const bound the union (in XOR) of the
    dropped chords on it: pairing them in order of c restores an equivalent set.
    """
    ends = np.ascontiguousarray(np.concatenate([segs[:, :2], segs[:, 2:]]).astype(np.float32))
    keys = ends.view(np.int64).ravel()  # exact float32 pairs
    _, first, counts = np.unique(keys, return_index=True, return_counts=True)
    odd = ends[first[(counts & 1) == 1]].astype(np.float64)
    pad = 1e-6 * (hi - lo)
    odd = odd[(odd[:, 0] >= lo - pad) & (odd[:, 0] <= hi + pad)]
    if not len(odd):
        return np.zeros((0, 4))
    odd = odd[np.lexsort((odd[:, 1], odd[:, 0]))]
    lines, starts = np.unique(odd[:, 0], return_index=True)
    sizes = np.diff(np.append(starts, len(odd)))
    if np.any(sizes & 1):
        return None
    pairs = odd.reshape(-1, 2, 2)  # consecutive points on one line pair up
    return np.concatenate([pairs[:, 0], pairs[:, 1]], axis=1)


def _reference_points(lo_a, lo_c, scale_a, scale_c, na, nc):
    """Cell reference points, computed in float32 exactly as the device does."""
    i = np.arange(na, dtype=np.float32)
    j = np.arange(nc, dtype=np.float32)
    ra = np.float32(lo_a) + (i + np.float32(REF_A)) / np.float32(scale_a)
    rc = np.float32(lo_c) + (j + np.float32(REF_C)) / np.float32(scale_c)
    return ra.astype(np.float64), rc.astype(np.float64)


def _polygon_parity(segs: np.ndarray, ra: np.ndarray, rc: np.ndarray) -> np.ndarray:
    """(nc, na) parity of chords crossed by a ray toward +c from each (ra[i], rc[j])."""
    parity = np.zeros((len(rc), len(ra)), dtype=np.int64)
    a0, c0, a1, c1 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    for i, a in enumerate(ra):
        spans = (a0 <= a) != (a1 <= a)
        if not np.any(spans):
            continue
        t = (a - a0[spans]) / (a1[spans] - a0[spans])
        y = np.sort(c0[spans] + t * (c1[spans] - c0[spans]))
        # chords strictly above each reference point
        parity[:, i] = (len(y) - np.searchsorted(y, rc, side="right")) & 1
    return parity


def face_grid(segs: np.ndarray, bulges: np.ndarray, range_a, range_c):
    """Grid of one face over its (a, c) rectangle.

    Returns ``(na, nc, map, info, chords, bulges)`` with ``map = (lo_a, lo_c, cells per unit a,
    cells per unit c)``, ``info`` an (nc * na, 2) array of (first member, count << 1 | parity)
    and each cell's chords (copied, so the device reads them without an index), or None when
    the face keeps the slab test.
    """
    lo_a, hi_a = float(range_a[0]), float(range_a[1])
    lo_c, hi_c = float(range_c[0]), float(range_c[1])
    if not (hi_a > lo_a and hi_c > lo_c) or not len(segs):
        return None
    parallel = _restore_parallel_chords(segs, lo_a, hi_a)
    if parallel is None:
        return None
    segs = np.concatenate([segs, parallel])
    bulges = np.concatenate([bulges, np.zeros(len(parallel))])
    n = len(segs)
    cells = int(np.clip(np.ceil(CELLS_PER_SQRT_SEGMENT * np.sqrt(n)), 2, MAX_CELLS_PER_AXIS))
    na = nc = cells
    scale_a = np.float32(na / (hi_a - lo_a))
    scale_c = np.float32(nc / (hi_c - lo_c))
    lo_a32, lo_c32 = np.float32(lo_a), np.float32(lo_c)
    # Cells each chord's box (grown by its bulge's reach) touches; boxes outside the grid are
    # dropped, and the grid's border cells take what lies just past them.
    reach = np.abs(bulges) + 1e-6 * max(hi_a - lo_a, hi_c - lo_c)
    amin = np.minimum(segs[:, 0], segs[:, 2]) - reach
    amax = np.maximum(segs[:, 0], segs[:, 2]) + reach
    cmin = np.minimum(segs[:, 1], segs[:, 3]) - reach
    cmax = np.maximum(segs[:, 1], segs[:, 3]) + reach
    i0 = np.floor((amin - lo_a) * scale_a).astype(np.int64)
    i1 = np.floor((amax - lo_a) * scale_a).astype(np.int64)
    j0 = np.floor((cmin - lo_c) * scale_c).astype(np.int64)
    j1 = np.floor((cmax - lo_c) * scale_c).astype(np.int64)
    inside = (i1 >= 0) & (i0 < na) & (j1 >= 0) & (j0 < nc)
    ids = np.nonzero(inside)[0]
    i0, i1 = np.clip(i0[ids], 0, na - 1), np.clip(i1[ids], 0, na - 1)
    j0, j1 = np.clip(j0[ids], 0, nc - 1), np.clip(j1[ids], 0, nc - 1)
    wi, wj = i1 - i0 + 1, j1 - j0 + 1
    counts = wi * wj
    seg_ids = np.repeat(ids, counts)
    k = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    wi_r = np.repeat(wi, counts)
    cell_ids = (np.repeat(j0, counts) + k // wi_r) * na + np.repeat(i0, counts) + k % wi_r
    order = np.argsort(cell_ids, kind="stable")
    members = seg_ids[order]
    per_cell = np.bincount(cell_ids, minlength=na * nc)
    starts = np.cumsum(per_cell) - per_cell
    ra, rc = _reference_points(lo_a32, lo_c32, scale_a, scale_c, na, nc)
    parity = _polygon_parity(segs.astype(np.float64), ra, rc).ravel()
    info = np.stack([np.where(per_cell > 0, starts, 0), (per_cell << 1) | parity], axis=1)
    return na, nc, (lo_a32, lo_c32, scale_a, scale_c), info, segs[members], bulges[members]


def _face_segments(prim: BrepPrim, face: int) -> np.ndarray:
    """The face's chord indices (each face's chords are contiguous, all in its slabs)."""
    base, count = prim.face_slab[face]
    lo, hi = prim.slab_start[base], prim.slab_start[base + count]
    members = prim.slab_segments[lo:hi]
    if not len(members):
        return np.zeros(0, np.int64)
    return np.arange(members.min(), members.max() + 1)


def build_trim_grids(prim: BrepPrim):
    """Grids of every trimmed face of a prim (cached on it).

    Returns ``(face_grid (F, 3) int32: cell base, na, nc (na 0: slab test), face_grid_map (F, 4)
    float32, cell_info (C, 2) int32, cell_chords (M, 4) float32, cell_bulges (M,) float32)``.
    """
    cached = getattr(prim, "trim_grids", None)  # absent on prims pickled before the field existed
    if cached is not None:
        return cached
    nfaces = len(prim.face_flags)
    grid = np.zeros((nfaces, 3), np.int32)
    grid_map = np.zeros((nfaces, 4), np.float32)
    infos, chords, bulges = [], [], []
    cell_base = member_base = 0
    for f in np.nonzero(prim.face_flags & FLAG_TRIMMED)[0]:
        ids = _face_segments(prim, f)
        if not len(ids):
            continue
        rng = prim.face_range[f]
        swap = bool(prim.face_flags[f] & FLAG_SWAP)
        range_a, range_c = (rng[2:4], rng[0:2]) if swap else (rng[0:2], rng[2:4])
        result = face_grid(prim.segments[ids].astype(np.float64), prim.segment_bulge[ids], range_a, range_c)
        if result is None:
            continue
        na, nc, mapping, info, cell_chords, cell_bulges = result
        grid[f] = (cell_base, na, nc)
        grid_map[f] = mapping
        info = info.copy()
        info[:, 0] += member_base
        infos.append(info)
        chords.append(cell_chords)
        bulges.append(cell_bulges)
        cell_base += na * nc
        member_base += len(cell_chords)
    result = (
        grid,
        grid_map,
        np.concatenate(infos).astype(np.int32) if infos else np.zeros((0, 2), np.int32),
        np.concatenate(chords).astype(np.float32) if chords else np.zeros((0, 4), np.float32),
        np.concatenate(bulges).astype(np.float32) if bulges else np.zeros(0, np.float32),
    )
    prim.trim_grids = result
    return result
