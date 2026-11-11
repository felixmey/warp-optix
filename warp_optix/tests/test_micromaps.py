# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp
from warp_optix._src import micromaps

import warp_optix as wo

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA unavailable")]

SIZE = 64


@wp.kernel(module="unique")
def down_raygen(scene: wp.uint64, out: wp.array2d[wp.int32], counter: wp.array[wp.int32]):
    y, x = wp.tid()
    origin = wp.vec3((float(x) + 0.5) / float(SIZE), (float(y) + 0.5) / float(SIZE), 1.0)
    hit = wp.int32(0)
    wp.optix_trace(scene, origin, wp.vec3(0.0, 0.0, -1.0), 10.0, hit)
    out[y, x] = hit


@wp.kernel(module="unique")
def hit_closest_hit():
    wp.optix_set_payload(0, wp.uint32(1))


@wp.kernel(module="unique")
def count_any_hit(counter: wp.array[wp.int32]):
    wp.atomic_add(counter, 0, 1)


@wp.kernel(module="unique")
def no_hit_miss():
    wp.optix_set_payload(0, wp.uint32(0))


def _quad():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    return points, np.array([[0, 1, 2], [0, 2, 3]], np.uint32)


def _hole_states(points, triangles, level, unknown_rim=False):
    """Transparent micro-triangles in a disc around the quad's center, opaque elsewhere."""
    bary = micromaps.micro_triangle_barycentrics(level)
    states = []
    for tri in triangles:
        v = points[tri].astype(np.float64)
        w0 = 1.0 - bary[..., 0:1] - bary[..., 1:2]
        corners = w0 * v[0] + bary[..., 0:1] * v[1] + bary[..., 1:2] * v[2]
        r = np.linalg.norm(corners.mean(1)[:, :2] - 0.5, axis=1)
        s = np.where(r < 0.3, micromaps.TRANSPARENT, micromaps.OPAQUE).astype(np.uint8)
        if unknown_rim:
            s[(r > 0.25) & (r < 0.35)] = micromaps.UNKNOWN_OPAQUE
        states.append(s)
    return states


def _render(build_input, any_hit_counter=None):
    device = wp.get_device("cuda")
    gas = wo.Accel([build_input], device=device)
    group = wo.HitGroup(closest_hit=hit_closest_hit, any_hit=count_any_hit if any_hit_counter is not None else None)
    pipeline = wo.Pipeline(
        down_raygen,
        sbt=wo.SbtLayout(miss=no_hit_miss, hit_groups={"quad": group}),
        payload_words=1,
        opacity_micromaps=True,
        device=device,
    )
    out = wp.zeros((SIZE, SIZE), dtype=wp.int32, device=device)
    counter = any_hit_counter if any_hit_counter is not None else wp.zeros(1, dtype=wp.int32, device=device)
    wo.launch(pipeline, dim=out.shape, inputs=[gas.id, out, counter], device=device)
    wp.synchronize_device(device)
    return out.numpy()


def _radius():
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    return np.hypot((xx + 0.5) / SIZE - 0.5, (yy + 0.5) / SIZE - 0.5)


def test_micro_triangle_order_level_one():
    # Level 1: the corner triangles at the base vertices and the inverted middle one.
    bary = micromaps.micro_triangle_barycentrics(1)
    centroids = {tuple(np.round(b.mean(0), 4)) for b in bary}
    expected = {(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3), (1 / 3, 1 / 3)}
    assert centroids == {tuple(np.round(c, 4)) for c in expected}


def test_opacity_micromap_cuts_a_hole():
    points, triangles = _quad()
    device = wp.get_device("cuda")
    maps = wo.OpacityMicromaps(_hole_states(points, triangles, level=5), device=device)
    p = wp.array(points, dtype=wp.vec3, device=device)
    i = wp.array(triangles.ravel(), dtype=wp.uint32, device=device)
    hits = _render(wo.triangles(p, i, opacity_micromaps=maps, flags=wo.GeometryFlags.NONE))
    r = _radius()
    assert not hits[r < 0.27].any()  # the hole (micro-triangles are coarser than pixels)
    assert hits[r > 0.33].all()  # the opaque rest


def test_unknown_states_run_any_hit_and_predefined_indices():
    points, triangles = _quad()
    device = wp.get_device("cuda")
    maps = wo.OpacityMicromaps(_hole_states(points, triangles, level=5, unknown_rim=True), device=device)
    p = wp.array(points, dtype=wp.vec3, device=device)
    i = wp.array(triangles.ravel(), dtype=wp.uint32, device=device)
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    hits = _render(
        wo.triangles(p, i, opacity_micromaps=maps, micromap_indices=np.array([0, 1]), flags=wo.GeometryFlags.NONE),
        counter,
    )
    r = _radius()
    calls = int(counter.numpy()[0])
    assert 0 < calls < int(((r > 0.24) & (r < 0.36)).sum())  # only the unknown rim runs any-hit
    assert hits[r > 0.37].all() and not hits[r < 0.23].any()
    # A predefined index makes the whole second triangle transparent, without a micromap.
    hits = _render(
        wo.triangles(
            p,
            i,
            opacity_micromaps=maps,
            micromap_indices=np.array([0, micromaps.INDEX_FULLY_TRANSPARENT]),
            flags=wo.GeometryFlags.NONE,
        )
    )
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    upper_left = (yy > xx + 2) & (r > 0.37)  # inside triangle (0, 2, 3), away from the hole
    assert not hits[upper_left].any()
