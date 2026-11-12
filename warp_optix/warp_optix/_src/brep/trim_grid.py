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

The grids of all trimmed faces of all prims are built together with Warp kernels: endpoint and
cell lists are sorted (radix sort) and compacted (scans) instead of looping over faces.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .loader import FLAG_SWAP, FLAG_TRIMMED, BrepPrim

# The reference point's position within its cell: off-center, so that it does not land on the
# chords of holes centered in cells (common in CAD).
REF_A = 0.5 + 0.0123
REF_C = 0.5 - 0.0171
MAX_CELLS_PER_AXIS = 64
CELLS_PER_SQRT_SEGMENT = 2.0

# ---------------------------------------------------------------------------------------------
# Kernels. A "face" is a work face: a trimmed face with chords, of any prim, numbered globally.
# ---------------------------------------------------------------------------------------------


@wp.func
def _find(starts: wp.array[wp.int32], x: int) -> int:
    """The last index f with starts[f] <= x (starts ascending, starts[0] <= x)."""
    lo = int(0)
    hi = starts.shape[0] - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo


@wp.func
def _lower_bound(values: wp.array[wp.float64], first: int, count: int, x: wp.float64) -> int:
    """The number of values[first : first + count] (ascending) below x."""
    lo = int(0)
    hi = count
    while lo < hi:
        mid = (lo + hi) // 2
        if values[first + mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@wp.func
def _order_key(bits: wp.uint32) -> wp.uint32:
    """Order-preserving key of a float32 from its bits, with -0 equal to +0."""
    if bits == wp.uint32(0x80000000):
        bits = wp.uint32(0)
    if (bits & wp.uint32(0x80000000)) != wp.uint32(0):
        return ~bits
    return bits | wp.uint32(0x80000000)


@wp.func
def _endpoint(seg_start: wp.array[wp.int32], work_start: wp.array[wp.int32], e: int):
    """Endpoint e (two per chord) of the work faces' chords: (face, chord, column of its a)."""
    w = e // 2
    face = _find(work_start, w)
    return face, seg_start[face] + w - work_start[face], 2 * (e % 2)


@wp.kernel(enable_backward=False)
def _c_keys(
    bits: wp.array2d[wp.uint32],
    seg_start: wp.array[wp.int32],
    work_start: wp.array[wp.int32],
    keys: wp.array[wp.uint32],
    order: wp.array[wp.int32],
):
    e = wp.tid()
    face, k, col = _endpoint(seg_start, work_start, e)
    keys[e] = _order_key(bits[k, col + 1])
    order[e] = e


@wp.kernel(enable_backward=False)
def _face_a_keys(
    bits: wp.array2d[wp.uint32],
    seg_start: wp.array[wp.int32],
    work_start: wp.array[wp.int32],
    by_c: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
):
    i = wp.tid()
    e = by_c[i]
    face, k, col = _endpoint(seg_start, work_start, e)
    keys[i] = (wp.uint64(face) << wp.uint64(32)) | wp.uint64(_order_key(bits[k, col]))
    order[i] = e


@wp.func
def _endpoint_key(bits: wp.array2d[wp.uint32], seg_start: wp.array[wp.int32], work_start: wp.array[wp.int32], e: int):
    face, k, col = _endpoint(seg_start, work_start, e)
    return face, _order_key(bits[k, col]), _order_key(bits[k, col + 1])


@wp.kernel(enable_backward=False)
def _run_starts(
    bits: wp.array2d[wp.uint32],
    seg_start: wp.array[wp.int32],
    work_start: wp.array[wp.int32],
    order: wp.array[wp.int32],
    starts: wp.array[wp.int32],
):
    """Whether sorted endpoint i starts a run of equal points (same face, a and c)."""
    i = wp.tid()
    starts[i] = 1
    if i > 0:
        f0, a0, c0 = _endpoint_key(bits, seg_start, work_start, order[i - 1])
        f1, a1, c1 = _endpoint_key(bits, seg_start, work_start, order[i])
        if f0 == f1 and a0 == a1 and c0 == c1:
            starts[i] = 0


@wp.kernel(enable_backward=False)
def _run_bounds(
    starts: wp.array[wp.int32], ids: wp.array[wp.int32], first: wp.array[wp.int32], count: wp.array[wp.int32]
):
    """Run id (inclusive scan of starts, 1-based) -> its first sorted position and its length."""
    i = wp.tid()
    run = ids[i] - 1
    if starts[i] == 1:
        first[run] = i
    wp.atomic_add(count, run, 1)


@wp.kernel(enable_backward=False)
def _odd_in_range(
    segs: wp.array2d[wp.float32],
    seg_start: wp.array[wp.int32],
    work_start: wp.array[wp.int32],
    order: wp.array[wp.int32],
    first: wp.array[wp.int32],
    count: wp.array[wp.int32],
    range_a: wp.array[wp.vec2d],
    keep: wp.array[wp.int32],
):
    """Points used by an odd number of chord ends (unpaired), on the face's a-range."""
    run = wp.tid()
    keep[run] = 0
    if count[run] % 2 == 1:
        face, k, col = _endpoint(seg_start, work_start, order[first[run]])
        a = wp.float64(segs[k, col])
        r = range_a[face]
        pad = wp.float64(1.0e-6) * (r[1] - r[0])
        if a >= r[0] - pad and a <= r[1] + pad:
            keep[run] = 1


@wp.kernel(enable_backward=False)
def _gather_odd(
    order: wp.array[wp.int32],
    first: wp.array[wp.int32],
    keep: wp.array[wp.int32],
    position: wp.array[wp.int32],
    odd: wp.array[wp.int32],
):
    run = wp.tid()
    if keep[run] == 1:
        odd[position[run]] = order[first[run]]


@wp.kernel(enable_backward=False)
def _line_starts(
    bits: wp.array2d[wp.uint32],
    seg_start: wp.array[wp.int32],
    work_start: wp.array[wp.int32],
    odd: wp.array[wp.int32],
    starts: wp.array[wp.int32],
):
    """Whether unpaired point o starts a line (same face and a)."""
    o = wp.tid()
    starts[o] = 1
    if o > 0:
        f0, a0, c0 = _endpoint_key(bits, seg_start, work_start, odd[o - 1])
        f1, a1, c1 = _endpoint_key(bits, seg_start, work_start, odd[o])
        if f0 == f1 and a0 == a1:
            starts[o] = 0


@wp.kernel(enable_backward=False)
def _pair_flags(
    seg_start: wp.array[wp.int32],
    work_start: wp.array[wp.int32],
    odd: wp.array[wp.int32],
    line_ids: wp.array[wp.int32],
    line_first: wp.array[wp.int32],
    line_count: wp.array[wp.int32],
    open_faces: wp.array[wp.int32],
    pairs: wp.array[wp.int32],
):
    """Unpaired points on one line a = const pair up in order of c into the chords the slab
    buckets dropped (parallel to the c axis); a line with an odd count means an open polyline."""
    o = wp.tid()
    line = line_ids[o] - 1
    within = o - line_first[line]
    pairs[o] = 0
    if within % 2 == 0 and within + 1 < line_count[line]:
        pairs[o] = 1
    if line_count[line] % 2 == 1:
        face, k, col = _endpoint(seg_start, work_start, odd[o])
        open_faces[face] = 1


@wp.kernel(enable_backward=False)
def _write_pairs(
    segs: wp.array2d[wp.float32],
    seg_start: wp.array[wp.int32],
    work_start: wp.array[wp.int32],
    odd: wp.array[wp.int32],
    pairs: wp.array[wp.int32],
    position: wp.array[wp.int32],
    parallel: wp.array[wp.vec4],
    parallel_count: wp.array[wp.int32],
):
    o = wp.tid()
    if pairs[o] == 1:
        face, k0, col0 = _endpoint(seg_start, work_start, odd[o])
        _face, k1, col1 = _endpoint(seg_start, work_start, odd[o + 1])
        parallel[position[o]] = wp.vec4(segs[k0, col0], segs[k0, col0 + 1], segs[k1, col1], segs[k1, col1 + 1])
        wp.atomic_add(parallel_count, face, 1)


@wp.struct
class Grids:
    """The grids being built: per grid face (work faces that keep a grid), in order."""

    work: wp.array[wp.int32]  # its work face
    chord_start: wp.array[wp.int32]  # first of its chords (own, then restored parallel ones)
    own: wp.array[wp.int32]  # how many of them are its own (the rest are parallel)
    parallel_start: wp.array[wp.int32]
    cell_start: wp.array[wp.int32]  # first cell (global), cells row-major in (c, a)
    column_start: wp.array[wp.int32]  # first reference a (global)
    row_start: wp.array[wp.int32]  # first reference c (global)
    size: wp.array[wp.vec2i]  # cells along a, along c
    lo: wp.array[wp.vec2d]  # range lo (a, c)
    scale: wp.array[wp.vec2d]  # cells per unit (float32 values)
    eps: wp.array[wp.float64]  # 1e-6 of the larger range width
    ref_a: wp.array[wp.float64]  # reference point coordinates, exactly as the device computes them
    ref_c: wp.array[wp.float64]


@wp.kernel(enable_backward=False)
def _chord_cells(
    g: Grids,
    segs: wp.array2d[wp.float32],
    bulges: wp.array[wp.float32],
    seg_start: wp.array[wp.int32],
    parallel: wp.array[wp.vec4],
    chords: wp.array[wp.vec4],
    chord_bulges: wp.array[wp.float32],
    boxes: wp.array[wp.vec4i],
    counts: wp.array[wp.int32],
    marks: wp.array[wp.int32],
):
    """Per chord: gather it, the cells its box (grown by the bulge's reach) covers, and its
    crossings of the cell columns' vertical rays (marked at the first cell row above)."""
    k = wp.tid()
    f = _find(g.chord_start, k)
    local = k - g.chord_start[f]
    chord = wp.vec4(0.0)
    bulge = float(0.0)
    if local < g.own[f]:
        s = seg_start[g.work[f]] + local
        chord = wp.vec4(segs[s, 0], segs[s, 1], segs[s, 2], segs[s, 3])
        bulge = bulges[s]
    else:
        chord = parallel[g.parallel_start[f] + local - g.own[f]]
    chords[k] = chord
    chord_bulges[k] = bulge
    a0 = wp.float64(chord[0])
    c0 = wp.float64(chord[1])
    a1 = wp.float64(chord[2])
    c1 = wp.float64(chord[3])
    n = g.size[f]
    lo = g.lo[f]
    scale = g.scale[f]
    reach = wp.abs(wp.float64(bulge)) + g.eps[f]
    i0 = wp.floor((wp.min(a0, a1) - reach - lo[0]) * scale[0])
    i1 = wp.floor((wp.max(a0, a1) + reach - lo[0]) * scale[0])
    j0 = wp.floor((wp.min(c0, c1) - reach - lo[1]) * scale[1])
    j1 = wp.floor((wp.max(c0, c1) + reach - lo[1]) * scale[1])
    counts[k] = 0
    if i1 >= wp.float64(0.0) and i0 < wp.float64(n[0]) and j1 >= wp.float64(0.0) and j0 < wp.float64(n[1]):
        box = wp.vec4i(
            int(wp.max(i0, wp.float64(0.0))),
            int(wp.min(i1, wp.float64(n[0] - 1))),
            int(wp.max(j0, wp.float64(0.0))),
            int(wp.min(j1, wp.float64(n[1] - 1))),
        )
        boxes[k] = box
        counts[k] = (box[1] - box[0] + 1) * (box[3] - box[2] + 1)
    # Columns whose reference a lies in [min a, max a): the chord crosses their ray at height y,
    # above the reference points of rows j < J (J = rows with reference c below y).
    first = _lower_bound(g.ref_a, g.column_start[f], n[0], wp.min(a0, a1))
    last = _lower_bound(g.ref_a, g.column_start[f], n[0], wp.max(a0, a1))
    for i in range(first, last):
        a = g.ref_a[g.column_start[f] + i]
        y = c0 + (a - a0) / (a1 - a0) * (c1 - c0)
        rows = _lower_bound(g.ref_c, g.row_start[f], n[1], y)
        if rows > 0:
            wp.atomic_add(marks, g.cell_start[f] + (rows - 1) * n[0] + i, 1)


@wp.kernel(enable_backward=False)
def _cell_members(
    g: Grids,
    boxes: wp.array[wp.vec4i],
    counts: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    cells: wp.array[wp.int32],
    members: wp.array[wp.int32],
    per_cell: wp.array[wp.int32],
):
    """(cell, chord) pairs in chord order, for a stable sort by cell."""
    k = wp.tid()
    count = counts[k]
    if count == 0:
        return
    f = _find(g.chord_start, k)
    box = boxes[k]
    width = box[1] - box[0] + 1
    for m in range(count):
        cell = g.cell_start[f] + (box[2] + m // width) * g.size[f][0] + box[0] + m % width
        cells[offsets[k] + m] = cell
        members[offsets[k] + m] = k
        wp.atomic_add(per_cell, cell, 1)


@wp.kernel(enable_backward=False)
def _cell_info(
    g: Grids,
    marks: wp.array[wp.int32],
    per_cell: wp.array[wp.int32],
    cell_member_start: wp.array[wp.int32],
    info: wp.array[wp.vec2i],
):
    """Per column: polygon parity at each cell's reference point (chords crossing above it, a
    suffix sum of the marks), with the cell's first member and member count."""
    column = wp.tid()
    f = _find(g.column_start, column)
    i = column - g.column_start[f]
    n = g.size[f]
    face_first = cell_member_start[g.cell_start[f]]
    above = int(0)
    for r in range(n[1]):
        cell = g.cell_start[f] + (n[1] - 1 - r) * n[0] + i
        above += marks[cell]
        first = face_first
        if per_cell[cell] > 0:
            first = cell_member_start[cell]
        info[cell] = wp.vec2i(first, (per_cell[cell] << 1) | (above & 1))


@wp.kernel(enable_backward=False)
def _gather_members(
    members: wp.array[wp.int32],
    chords: wp.array[wp.vec4],
    chord_bulges: wp.array[wp.float32],
    out_chords: wp.array[wp.vec4],
    out_bulges: wp.array[wp.float32],
):
    m = wp.tid()
    out_chords[m] = chords[members[m]]
    out_bulges[m] = chord_bulges[members[m]]


# ---------------------------------------------------------------------------------------------
# Host driver
# ---------------------------------------------------------------------------------------------


def _scan(values: wp.array, inclusive: bool) -> tuple[wp.array, int]:
    """Prefix sums of an int32 array, and the total."""
    out = wp.empty_like(values)
    if len(values) == 0:
        return out, 0
    wp.utils.array_scan(values, out, inclusive=inclusive)
    last = out[-1:].numpy()[0] + (0 if inclusive else values[-1:].numpy()[0])
    return out, int(last)


def _sort(keys: wp.array, values: wp.array, count: int) -> None:
    """Stable radix sort of the first ``count`` values by their keys (arrays of 2 * count)."""
    if count > 1:
        wp.utils.radix_sort_pairs(keys, values, count)


def _empty_grids(nfaces: int):
    return (
        np.zeros((nfaces, 3), np.int32),
        np.zeros((nfaces, 4), np.float32),
        np.zeros((0, 2), np.int32),
        np.zeros((0, 4), np.float32),
        np.zeros(0, np.float32),
    )


def build_trim_grids(prims: BrepPrim | list[BrepPrim], device=None):
    """Grids of every trimmed face of the prims, cached on each prim (``prim.trim_grids``).

    Per prim: ``(face_grid (F, 3) int32: cell base, na, nc (na 0: slab test), face_grid_map
    (F, 4) float32, cell_info (C, 2) int32, cell_chords (M, 4) float32, cell_bulges (M,)
    float32)``. Returns that tuple for a single prim, else the list of them.
    """
    single = isinstance(prims, BrepPrim)
    prim_list = [prims] if single else list(prims)
    todo = [p for p in {id(p): p for p in prim_list}.values() if getattr(p, "trim_grids", None) is None]
    if todo:
        _build(todo, wp.get_device(device))
    return prim_list[0].trim_grids if single else [p.trim_grids for p in prim_list]


def _work_faces(prims: list[BrepPrim]):
    """Trimmed faces with chords and a proper range, of all prims: their prim, face, chord range
    (into the prims' concatenated segments) and (a, c) ranges."""
    rows = []
    seg_base = 0
    for n, prim in enumerate(prims):
        faces = np.nonzero(prim.face_flags & FLAG_TRIMMED)[0]
        base, count = prim.face_slab[faces, 0], prim.face_slab[faces, 1]
        lo, hi = prim.slab_start[base], prim.slab_start[base + count]
        faces, lo, hi = faces[hi > lo], lo[hi > lo], hi[hi > lo]
        # Each face's chords are contiguous and all in its slabs.
        first = np.minimum.reduceat(prim.slab_segments, lo) if len(lo) else lo
        last = np.maximum.reduceat(prim.slab_segments, lo) if len(lo) else lo
        rng = prim.face_range[faces]
        swap = (prim.face_flags[faces] & FLAG_SWAP) != 0
        range_a = np.where(swap[:, None], rng[:, 2:4], rng[:, 0:2])
        range_c = np.where(swap[:, None], rng[:, 0:2], rng[:, 2:4])
        proper = (range_a[:, 1] > range_a[:, 0]) & (range_c[:, 1] > range_c[:, 0])
        rows.append(
            (
                np.full(proper.sum(), n),
                faces[proper],
                first[proper] + seg_base,
                (last - first + 1)[proper],
                range_a[proper],
                range_c[proper],
            )
        )
        seg_base += len(prim.segments)
    return [np.concatenate(column) for column in zip(*rows, strict=True)]


def _build(prims: list[BrepPrim], device) -> None:
    prim_of, face_of, seg_start, seg_count, range_a, range_c = _work_faces(prims)
    results = [_empty_grids(len(p.face_flags)) for p in prims]
    if len(face_of) == 0:
        for prim, result in zip(prims, results, strict=True):
            prim.trim_grids = result
        return

    def dev(values, dtype):
        return wp.array(np.ascontiguousarray(values), dtype=dtype, device=device)

    segments = np.concatenate([p.segments for p in prims]).astype(np.float32)
    seg_bulges = np.concatenate([p.segment_bulge for p in prims]).astype(np.float32)
    with wp.ScopedDevice(device):
        d_segs = dev(segments, wp.float32)
        d_bits = dev(segments.view(np.uint32), wp.uint32)
        d_seg_start = dev(seg_start.astype(np.int32), wp.int32)
        work_start = np.concatenate([[0], np.cumsum(seg_count)[:-1]]).astype(np.int32)
        d_work_start = dev(work_start, wp.int32)
        nwork = len(face_of)
        ends = 2 * int(seg_count.sum())

        # 1. Restore the chords the slab buckets dropped: sort chord ends by (face, a, c).
        keys = wp.empty(2 * ends, dtype=wp.uint32)
        order = wp.empty(2 * ends, dtype=wp.int32)
        wp.launch(_c_keys, dim=ends, inputs=[d_bits, d_seg_start, d_work_start, keys, order])
        _sort(keys, order, ends)
        keys64 = wp.empty(2 * ends, dtype=wp.uint64)
        by_face = wp.empty(2 * ends, dtype=wp.int32)
        wp.launch(_face_a_keys, dim=ends, inputs=[d_bits, d_seg_start, d_work_start, order, keys64, by_face])
        del keys, order
        _sort(keys64, by_face, ends)
        del keys64
        run_starts = wp.empty(ends, dtype=wp.int32)
        wp.launch(_run_starts, dim=ends, inputs=[d_bits, d_seg_start, d_work_start, by_face, run_starts])
        run_ids, runs = _scan(run_starts, inclusive=True)
        run_first = wp.empty(runs, dtype=wp.int32)
        run_count = wp.zeros(runs, dtype=wp.int32)
        wp.launch(_run_bounds, dim=ends, inputs=[run_starts, run_ids, run_first, run_count])
        keep = wp.empty(runs, dtype=wp.int32)
        d_range_a = dev(range_a, wp.vec2d)
        wp.launch(
            _odd_in_range,
            dim=runs,
            inputs=[d_segs, d_seg_start, d_work_start, by_face, run_first, run_count, d_range_a, keep],
        )
        keep_pos, nodd = _scan(keep, inclusive=False)
        odd = wp.empty(max(nodd, 1), dtype=wp.int32)
        wp.launch(_gather_odd, dim=runs, inputs=[by_face, run_first, keep, keep_pos, odd])
        open_faces = wp.zeros(nwork, dtype=wp.int32)
        parallel_count = wp.zeros(nwork, dtype=wp.int32)
        parallel = wp.empty(max(nodd // 2, 1), dtype=wp.vec4)
        if nodd:
            line_starts = wp.empty(nodd, dtype=wp.int32)
            wp.launch(_line_starts, dim=nodd, inputs=[d_bits, d_seg_start, d_work_start, odd, line_starts])
            line_ids, lines = _scan(line_starts, inclusive=True)
            line_first = wp.empty(lines, dtype=wp.int32)
            line_count = wp.zeros(lines, dtype=wp.int32)
            wp.launch(_run_bounds, dim=nodd, inputs=[line_starts, line_ids, line_first, line_count])
            pairs = wp.empty(nodd, dtype=wp.int32)
            wp.launch(
                _pair_flags,
                dim=nodd,
                inputs=[d_seg_start, d_work_start, odd, line_ids, line_first, line_count, open_faces, pairs],
            )
            pair_pos, _ = _scan(pairs, inclusive=False)
            wp.launch(
                _write_pairs,
                dim=nodd,
                inputs=[d_segs, d_seg_start, d_work_start, odd, pairs, pair_pos, parallel, parallel_count],
            )
        is_open = open_faces.numpy().astype(bool)
        npar = parallel_count.numpy()
        parallel_start = np.concatenate([[0], np.cumsum(npar)[:-1]])

        # 2. Grid sizes and reference points of the faces that keep a grid (as the device computes).
        grid_faces = np.nonzero(~is_open)[0]
        own = seg_count[grid_faces]
        n = own + npar[grid_faces]
        cells = np.clip(np.ceil(CELLS_PER_SQRT_SEGMENT * np.sqrt(n)), 2, MAX_CELLS_PER_AXIS).astype(np.int64)
        ra, rc = range_a[grid_faces], range_c[grid_faces]
        scale_a = (cells / (ra[:, 1] - ra[:, 0])).astype(np.float32)
        scale_c = (cells / (rc[:, 1] - rc[:, 0])).astype(np.float32)
        lo_a32, lo_c32 = ra[:, 0].astype(np.float32), rc[:, 0].astype(np.float32)
        axis_start = np.concatenate([[0], np.cumsum(cells)[:-1]])
        local = (np.arange(int(cells.sum())) - np.repeat(axis_start, cells)).astype(np.float32)
        ref_a = np.repeat(lo_a32, cells) + (local + np.float32(REF_A)) / np.repeat(scale_a, cells)
        ref_c = np.repeat(lo_c32, cells) + (local + np.float32(REF_C)) / np.repeat(scale_c, cells)
        cell_start = np.concatenate([[0], np.cumsum(cells * cells)[:-1]])
        chord_start = np.concatenate([[0], np.cumsum(n)[:-1]])
        ncells, nchords = int((cells * cells).sum()), int(n.sum())

        g = Grids()
        g.work = dev(grid_faces.astype(np.int32), wp.int32)
        g.chord_start = dev(chord_start.astype(np.int32), wp.int32)
        g.own = dev(own.astype(np.int32), wp.int32)
        g.parallel_start = dev(parallel_start[grid_faces].astype(np.int32), wp.int32)
        g.cell_start = dev(cell_start.astype(np.int32), wp.int32)
        g.column_start = dev(axis_start.astype(np.int32), wp.int32)
        g.row_start = g.column_start  # square grids: rows and columns share their bases
        g.size = dev(np.stack([cells, cells], 1).astype(np.int32), wp.vec2i)
        g.lo = dev(np.stack([ra[:, 0], rc[:, 0]], 1), wp.vec2d)
        g.scale = dev(np.stack([scale_a, scale_c], 1).astype(np.float64), wp.vec2d)
        g.eps = dev(1e-6 * np.maximum(ra[:, 1] - ra[:, 0], rc[:, 1] - rc[:, 0]), wp.float64)
        g.ref_a = dev(ref_a.astype(np.float64), wp.float64)
        g.ref_c = dev(ref_c.astype(np.float64), wp.float64)

        # 3. Chords into cells (sorted by cell, stable: in chord order), parity per reference point.
        chords = wp.empty(nchords, dtype=wp.vec4)
        chord_bulges = wp.empty(nchords, dtype=wp.float32)
        boxes = wp.empty(nchords, dtype=wp.vec4i)
        counts = wp.empty(nchords, dtype=wp.int32)
        marks = wp.zeros(ncells, dtype=wp.int32)
        wp.launch(
            _chord_cells,
            dim=nchords,
            inputs=[
                g,
                d_segs,
                dev(seg_bulges, wp.float32),
                d_seg_start,
                parallel,
                chords,
                chord_bulges,
                boxes,
                counts,
                marks,
            ],
        )
        offsets, nmembers = _scan(counts, inclusive=False)
        cell_keys = wp.empty(2 * max(nmembers, 1), dtype=wp.int32)
        members = wp.empty(2 * max(nmembers, 1), dtype=wp.int32)
        per_cell = wp.zeros(ncells, dtype=wp.int32)
        wp.launch(_cell_members, dim=nchords, inputs=[g, boxes, counts, offsets, cell_keys, members, per_cell])
        _sort(cell_keys, members, nmembers)
        cell_member_start, _ = _scan(per_cell, inclusive=False)
        info = wp.empty(ncells, dtype=wp.vec2i)
        wp.launch(_cell_info, dim=int(cells.sum()), inputs=[g, marks, per_cell, cell_member_start, info])
        out_chords = wp.empty(nmembers, dtype=wp.vec4)
        out_bulges = wp.empty(nmembers, dtype=wp.float32)
        wp.launch(_gather_members, dim=nmembers, inputs=[members, chords, chord_bulges, out_chords, out_bulges])
        info_np, chords_np, bulges_np = info.numpy(), out_chords.numpy(), out_bulges.numpy()

    # 4. Split per prim: cell and member bases relative to each prim (both are contiguous per prim).
    owner = prim_of[grid_faces]
    for n_prim, prim in enumerate(prims):
        rows = np.nonzero(owner == n_prim)[0]
        grid, grid_map, *_ = results[n_prim]
        if len(rows):
            c0, c1 = int(cell_start[rows[0]]), int(cell_start[rows[-1]] + cells[rows[-1]] ** 2)
            prim_info = info_np[c0:c1].copy()
            m0 = int(prim_info[0, 0])  # the first cell's first member, empty or not
            m1 = m0 + int((prim_info[:, 1] >> 1).sum())
            prim_info[:, 0] -= m0
            faces = face_of[grid_faces[rows]]
            grid[faces] = np.stack([cell_start[rows] - c0, cells[rows], cells[rows]], 1)
            grid_map[faces] = np.stack([lo_a32[rows], lo_c32[rows], scale_a[rows], scale_c[rows]], 1)
            results[n_prim] = (grid, grid_map, prim_info, chords_np[m0:m1], bulges_np[m0:m1])
        prim.trim_grids = results[n_prim]
