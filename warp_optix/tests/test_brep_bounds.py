# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative rejection must retain boundaries, thin patches and distant/grazing rays."""

import math

import numpy as np
import pytest
import warp as wp

pytest.importorskip("pxr")
from warp_optix._src.brep import bounds
from warp_optix._src.brep import device as bd

wp.set_module_options({"enable_backward": False, "fast_math": True})
pytestmark = pytest.mark.usd
BUILDERS = ["host", pytest.param("warp", marks=pytest.mark.gpu)]


def _evaluate(net, uv):
    def basis(n, t):
        return np.stack([math.comb(n, i) * t**i * (1 - t) ** (n - i) for i in range(n + 1)], axis=-1)

    h = np.einsum("ni,nj,ijk->nk", basis(net.shape[0] - 1, uv[:, 0]), basis(net.shape[1] - 1, uv[:, 1]), net)
    return h[:, :3] / h[:, 3:]


def _cases():
    rng = np.random.default_rng(531)
    uv = np.concatenate(
        [rng.uniform(-1e-4, 1.0001, (256, 2)), [[s, t] for s in (-1e-4, 0, 1, 1.0001) for t in (-1e-4, 0, 1, 1.0001)]]
    )
    for p, q in ((0, 0), (1, 1), (3, 3), (3, 4), (7, 7)):
        for scale, translation in ((1e-4, 0), (1, 0), (1e4, 0), (1, 1e5)):
            x, y = np.meshgrid(np.linspace(-1, 1, p + 1), np.linspace(-1, 1, q + 1), indexing="ij")
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            points = scale * np.stack([x, y, 0.01 * x * y], axis=-1) @ rotation
            points += translation * np.array([1, -2, 3])
            weights = rng.uniform(0.5, 2, (p + 1, q + 1, 1))
            net = np.concatenate([points * weights, weights], axis=-1).astype(np.float32)
            yield net, scale * 1e-5, _evaluate(net.astype(float), uv)


def _fit_cases(cases, builder):
    if builder == "host":
        return [bounds.patch_obb(net, tolerance) for net, tolerance, _ in cases]
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    from warp_optix._src.brep.bounds_gpu import build_bounds  # noqa: PLC0415

    nets = [net.reshape(-1, 4) for net, _, _ in cases]
    data = bd.BrepData()
    data.cvs = wp.array(np.concatenate(nets), dtype=wp.vec4, device="cuda:0")
    data.prim_cv = wp.array(np.cumsum([0] + [len(net) for net in nets[:-1]]), dtype=int, device="cuda:0")
    data.prim_degree = wp.array([np.array(net.shape[:2]) - 1 for net, _, _ in cases], dtype=wp.vec2i, device="cuda:0")
    data.prim_tol = wp.array([tolerance for _, tolerance, _ in cases], dtype=float, device="cuda:0")
    data.prims = wp.zeros(len(cases), dtype=bd.Surface, device="cuda:0")
    build_bounds(data, np.arange(len(cases), dtype=np.int32))
    records = data.prims.numpy()
    return [(r["origin"], np.stack([r["x"], r["y"], r["axis"]], axis=1), r["params"][0]) for r in records]


@pytest.mark.parametrize("builder", BUILDERS)
def test_obb_contains_rational_surface_and_parameter_tolerance(builder):
    cases = list(_cases())
    for (_, _, points), (center, axes, reach) in zip(cases, _fit_cases(cases, builder), strict=True):
        assert np.isfinite(center).all() and np.isfinite(axes).all()
        assert np.any(axes), "These positive, well-conditioned weights must have usable bounds"
        assert np.abs((points - center) @ axes).max() <= 1.0
        assert np.abs(points).max() <= float(reach) * (1 + np.finfo(np.float32).eps)


@pytest.mark.parametrize("invalid", ["negative", "extension", "nonfinite"])
@pytest.mark.parametrize("builder", BUILDERS)
def test_obb_disables_rejection_without_a_positive_finite_hull(invalid, builder):
    net = np.ones((2, 2, 4))
    if invalid == "negative":
        net[0, 0, 3] = -1
    elif invalid == "extension":
        # Positive in [0,1], but extension through the lower edge loses positive weights.
        net[0, :, 3] = 1e-12
    else:
        net[0, 0, 0] = np.nan
    ((center, axes, reach),) = _fit_cases([(net, 1e-5, None)], builder)
    np.testing.assert_array_equal(center, 0)
    np.testing.assert_array_equal(axes, 0)
    assert reach == 0


@wp.kernel
def _reject(
    surfaces: wp.array[bd.Surface],
    ids: wp.array[int],
    origins: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    intervals: wp.array[wp.vec2],
    misses: wp.array[int],
):
    i = wp.tid()
    misses[i] = int(bd.patch_obb_misses(surfaces[ids[i]], origins[i], directions[i], intervals[i][0], intervals[i][1]))


def _gpu_reject(records, ids, origins, directions, intervals):
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    output = wp.empty(len(ids), dtype=int, device="cuda:0")
    wp.launch(
        _reject,
        len(ids),
        [
            wp.array(records, dtype=bd.Surface, device="cuda:0"),
            wp.array(ids, dtype=int, device="cuda:0"),
            wp.array(origins, dtype=wp.vec3, device="cuda:0"),
            wp.array(directions, dtype=wp.vec3, device="cuda:0"),
            wp.array(intervals, dtype=wp.vec2, device="cuda:0"),
            output,
        ],
        device="cuda:0",
    )
    return output.numpy()


@pytest.mark.gpu
@pytest.mark.parametrize("builder", BUILDERS)
def test_obb_retains_surface_crossings_with_distant_and_parallel_rays(builder):
    rng = np.random.default_rng(18)
    cases = list(_cases())
    records = np.zeros(len(cases), dtype=bd.Surface.numpy_dtype())
    ids, origins, directions, intervals = [], [], [], []
    for k, ((_, _, points), (center, axes, reach)) in enumerate(zip(cases, _fit_cases(cases, builder), strict=True)):
        records["origin"][k] = center
        records["x"][k], records["y"][k], records["axis"][k] = axes.T
        records["params"][k, 0] = reach
        d = rng.normal(size=points.shape)
        d /= np.linalg.norm(d, axis=1)[:, None]
        d[:4] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1e-20, 0]]
        for distance in (0.0, 1.0, 1e6):
            # Uploading these rays rounds their target positions, especially at distance 1e6.
            # The rejection must remain conservative about that uncertainty.
            ids.extend([k] * len(points))
            origins.extend(points - distance * d)
            directions.extend(d)
            intervals.extend([[distance, distance]] * len(points))
    np.testing.assert_array_equal(_gpu_reject(records, ids, origins, directions, intervals), 0)


@pytest.mark.gpu
def test_obb_parallel_rays_ray_intervals_and_disabled_bounds():
    records = np.zeros(2, dtype=bd.Surface.numpy_dtype())
    records["x"][0], records["y"][0], records["axis"][0] = np.eye(3)
    records["params"][0, 0] = 1
    origins = [[0, 0, -2], [2, 0, -2], [1, 0, -2], [0, 0, -2], [0, 0, -2], [2, 0, -2]]
    intervals = [[0, 10], [0, 10], [0, 10], [0, 0.5], [4, 10], [0, 10]]
    result = _gpu_reject(records, [0, 0, 0, 0, 0, 1], origins, [[0, 0, 1]] * 6, intervals)
    np.testing.assert_array_equal(result, [0, 1, 0, 1, 1, 0])


def test_warp_bounds_empty_selection_needs_no_arrays():
    from warp_optix._src.brep.bounds_gpu import build_bounds  # noqa: PLC0415

    build_bounds(bd.BrepData(), np.zeros(0, np.int32))


@pytest.mark.gpu
def test_warp_bounds_preserve_unselected_records_and_disable_invalid_rebuilds():
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    from warp_optix._src.brep.bounds_gpu import build_bounds  # noqa: PLC0415

    stream = wp.Stream("cuda:0")
    with wp.ScopedStream(stream):
        net, tolerance, _ = list(_cases())[8]
        records = np.zeros(3, dtype=bd.Surface.numpy_dtype())
        for name in records.dtype.names:
            records[name] = 7  # includes face IDs, flags, trims and intentionally stale bounds
        data = bd.BrepData()
        data.cvs = wp.array(net.reshape(-1, 4), dtype=wp.vec4, device="cuda:0")
        data.prim_cv = wp.zeros(3, dtype=int, device="cuda:0")
        data.prim_degree = wp.array([np.array(net.shape[:2]) - 1] * 3, dtype=wp.vec2i, device="cuda:0")
        data.prim_tol = wp.full(3, tolerance, dtype=float, device="cuda:0")
        data.prims = wp.array(records, dtype=bd.Surface, device="cuda:0")
        build_bounds(data, np.array([1], np.int32))
        wp.synchronize_stream(stream)
        actual = data.prims.numpy()
        for name in records.dtype.names:
            np.testing.assert_array_equal(actual[name][[0, 2]], records[name][[0, 2]])
            if name not in ("origin", "x", "y", "axis", "params"):
                np.testing.assert_array_equal(actual[name], records[name])
        assert np.any(actual["axis"][1])
        data.prim_tol.fill_(-1)
        build_bounds(data, np.array([1], np.int32))
        wp.synchronize_stream(stream)
        rebuilt = data.prims.numpy()
        for name in ("origin", "x", "y", "axis", "params"):
            np.testing.assert_array_equal(rebuilt[name][1], 0)
        for name in records.dtype.names:
            np.testing.assert_array_equal(rebuilt[name][[0, 2]], records[name][[0, 2]])
