# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for BRep candidate pruning and specialized patch evaluation."""

import math
from pathlib import Path

import numpy as np
import pytest
import warp as wp

pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd
from test_brep import _inside, _write_nurbs_dome
from warp_optix._src.brep import device as bd
from warp_optix._src.brep import loader

pytestmark = pytest.mark.usd


@wp.kernel(enable_backward=False)
def _evaluate34(cvs: wp.array[wp.vec4], uv: wp.array[wp.vec2], out: wp.array2d[wp.vec3]):
    i = wp.tid()
    patch = bd.CubicQuartic()
    patch.base = 0
    p, ds, dt = bd.eval_patch(patch, cvs, uv[i][0], uv[i][1])
    out[i, 0] = p
    out[i, 1] = ds
    out[i, 2] = dt


@pytest.mark.gpu
def test_cubic_quartic_matches_float64_reference():
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(81)
    points = rng.uniform(-2, 2, (4, 5, 3))
    weights = rng.uniform(0.5, 2, (4, 5, 1))
    cvs = np.concatenate([points * weights, weights], axis=-1).astype(np.float32)
    uv = np.concatenate([rng.uniform(-0.05, 1.05, (128, 2)), [[0, 0], [1, 1], [0, 1], [1, 0]]]).astype(np.float32)

    def basis(n, x):
        return np.stack([math.comb(n, i) * x**i * (1 - x) ** (n - i) for i in range(n + 1)], axis=-1)

    def derivative(n, x):
        lower = basis(n - 1, x)
        zero = np.zeros((len(x), 1))
        return n * (np.concatenate([zero, lower], axis=1) - np.concatenate([lower, zero], axis=1))

    s, t = uv.astype(np.float64).T
    bu, bv = basis(3, s), basis(4, t)
    c = cvs.astype(np.float64)
    h = np.einsum("ni,nj,ijk->nk", bu, bv, c)
    hs = np.einsum("ni,nj,ijk->nk", derivative(3, s), bv, c)
    ht = np.einsum("ni,nj,ijk->nk", bu, derivative(4, t), c)
    p = h[:, :3] / h[:, 3:]
    expected = np.stack([p, (hs[:, :3] - hs[:, 3:] * p) / h[:, 3:], (ht[:, :3] - ht[:, 3:] * p) / h[:, 3:]], axis=1)
    out = wp.empty((len(uv), 3), dtype=wp.vec3, device="cuda:0")
    wp.launch(
        _evaluate34,
        len(uv),
        [
            wp.array(cvs.reshape(-1, 4), dtype=wp.vec4, device="cuda:0"),
            wp.array(uv, dtype=wp.vec2, device="cuda:0"),
            out,
        ],
        device="cuda:0",
    )
    actual = out.numpy()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=4e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("back", [False, True])
def test_overlapping_shells_keep_nearest_exact_surface(tmp_path, monkeypatch, reverse, back):
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples" / "brep"))
    import brep_optix as bo
    import example_brep as ex

    prims = []
    sign = -1.0 if back else 1.0
    for i, (scale, y) in enumerate([(1.0, 0.0), (10.0, sign * 0.0001)]):
        path = tmp_path / f"plane{i}.usda"
        _write_nurbs_dome(path)
        stage = Usd.Stage.Open(str(path))
        prim = stage.GetPrimAtPath("/Dome")
        grid = np.linspace(-scale, scale, 4)
        prim.GetAttribute("brep:surface:nurb:controlVertices").Set(
            [Gf.Vec3d(float(x), y, float(z)) for x in grid for z in grid]
        )
        prim.GetAttribute("brep:extent").Set([Gf.Vec3d(-scale, y, -scale), Gf.Vec3d(scale, y, scale)])
        stage.GetRootLayer().Save()
        prims.extend(loader.load_stage(str(path)))
    if reverse:
        prims.reverse()
    cam = bo.Camera()
    cam.origin = wp.vec3(0.123, -sign, 0.234)
    cam.lower_left = wp.vec3(0.123, 0.0, 0.234)
    cam.du = wp.vec3(0.0)
    cam.dv = wp.vec3(0.0)
    out = ex.outputs(1, 1, "cuda:0")
    for proxy in (False, True):
        scene = bo.BrepScene(prims, "cuda:0", proxy=proxy)
        scene.render(cam, out)
        np.testing.assert_allclose(out[0].numpy(), [[1.0]], atol=1e-6, rtol=0)
        assert out[2].numpy()[0, 0] == int(reverse)


@pytest.mark.parametrize("swap", [False, True])
def test_trim_culling_keeps_boundaries_and_bulges(swap):
    # A square hole inside a larger square. Vertical chords are omitted, as in slab packing.
    segs = np.array([[-2, -2, 2, -2], [2, 2, -2, 2], [-1, -1, 1, -1], [1, 1, -1, 1]], dtype=float)
    bulges = np.zeros(4)

    def outside(rect, segs=segs, bulges=bulges):
        if swap:
            rect = (rect[2], rect[3], rect[0], rect[1])
        return loader._patch_outside_trim(rect, segs, bulges, swap)

    assert outside((-0.5, 0.5, -0.5, 0.5))  # inside the hole
    assert outside((-0.5, 0.5, 3, 4))  # outside the face
    assert not outside((-0.5, 0.5, 1.2, 1.8))  # inside material
    assert not outside((0.9, 1.1, -0.5, 0.5))  # crosses a dropped vertical edge
    assert not outside((-0.5, 0.5, 0.9, 1.1))  # crosses a retained edge
    assert not outside((-0.5, 0.5, 1.0 - 1e-6, 1.5))  # boundary and solver slack
    # Curved trim reaches into an otherwise outside rectangle.
    curved = np.array([0.0, 0.0, 0.0, 0.4])
    assert not outside((-0.2, 0.2, 0.65, 0.75), bulges=curved)


def test_loader_removes_only_outside_subpatches(tmp_path, monkeypatch):
    path = tmp_path / "trimmed_dome.usda"
    _write_nurbs_dome(path)
    stage = Usd.Stage.Open(str(path))
    stage.GetPrimAtPath("/Dome").GetAttribute("face:trimType").Set(["general"])
    stage.GetRootLayer().Save()
    # Supply an outer square with vertices interleaved with chord midpoints, as the edge loader
    # does. This exercises trim packing and patch pruning independently of edge projection.
    ring = np.array([[0.25, 0.25], [0.5, 0.25], [0.75, 0.25], [0.75, 0.5], [0.75, 0.75], [0.5, 0.75], [0.25, 0.75]])
    monkeypatch.setattr(loader, "_face_loops_uv", lambda *args: ([ring], []))
    reject = loader._patch_outside_trim
    monkeypatch.setattr(loader, "_patch_outside_trim", lambda *args: False)
    (reference,) = loader.load_stage(str(path))
    monkeypatch.setattr(loader, "_patch_outside_trim", reject)
    (candidate,) = loader.load_stage(str(path))
    kept = {tuple(rect) for rect in candidate.prim_uv}
    removed = [rect for rect in reference.prim_uv if tuple(rect) not in kept]
    assert removed and kept
    assert candidate.stats["culled_nurbs_subpatches"] == len(removed)
    for u0, u1, v0, v1 in removed:
        for u in np.linspace(u0, u1, 9):
            for v in np.linspace(v0, v1, 9):
                assert not _inside(reference, 0, u, v)


@pytest.mark.parametrize("major,minor", [(3.0, 1.0), (2.0, 3.0), (6224.99789464, 6500.0)])
def test_torus_inverse_recovers_both_sheets(major, minor):
    surface = loader.Surface(loader.TORUS, params=(major, minor))
    u = np.linspace(-2.9, 2.9, 37)
    v = np.linspace(0.05, 2 * np.pi - 0.05, 37)
    points = surface.evaluate(u, v)
    uv = surface.invert(points)
    np.testing.assert_allclose(surface.evaluate(uv[:, 0], uv[:, 1]), points, rtol=1e-10, atol=1e-9)
    # Compare angles modulo a period, including the negative-radius sheet's azimuth.
    error = (uv - np.stack([u, v], axis=1) + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(error, 0, atol=1e-10)


@pytest.mark.gpu
@pytest.mark.parametrize("major,minor,v", [(3.0, 1.0, 0.1), (2.0, 3.0, 3.2), (6224.99789464, 6500.0, 3.21)])
def test_trimmed_torus_sheet_hit_and_normal(tmp_path, monkeypatch, major, minor, v):
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples" / "brep"))
    import brep_optix as bo
    import example_brep as ex

    path = tmp_path / "torus.usda"
    _write_nurbs_dome(path)
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath("/Dome")
    prim.GetAttribute("face:surfaceType").Set(["BrepSurfaceTorusAPI"])
    u = 0.7
    prim.GetAttribute("face:range").Set([Gf.Vec2d(u - 0.2, v - 0.02), Gf.Vec2d(u + 0.2, v + 0.02)])
    prefix = "brep:surface:torus:"
    for name, value in [("origin", (0, 0, 0)), ("axis", (0, 0, 1)), ("refDirection", (1, 0, 0))]:
        prim.CreateAttribute(prefix + name, Sdf.ValueTypeNames.Double3Array).Set([Gf.Vec3d(*value)])
    for name, value in [("majorRadius", major), ("minorRadius", minor)]:
        prim.CreateAttribute(prefix + name, Sdf.ValueTypeNames.DoubleArray).Set([value])
    prim.GetAttribute("brep:extent").Set([Gf.Vec3d(-(major + minor)), Gf.Vec3d(major + minor)])
    stage.GetRootLayer().Save()
    (data,) = loader.load_stage(str(path))
    surface = data.surfaces[0]
    point = surface.evaluate(np.array([u]), np.array([v]))[0]
    # Reference normal from the parameterization's derivatives, independent of inversion.
    du = (major + minor * np.cos(v)) * np.array([-np.sin(u), np.cos(u), 0.0])
    dv = minor * np.array([-np.sin(v) * np.cos(u), -np.sin(v) * np.sin(u), np.cos(v)])
    normal = np.cross(du, dv)
    normal /= np.linalg.norm(normal)
    offset = 0.1 * abs(major - minor)
    cam = bo.Camera()
    cam.origin = wp.vec3(*(point + offset * normal))
    cam.lower_left = wp.vec3(*point)
    cam.du = wp.vec3(0.0)
    cam.dv = wp.vec3(0.0)
    out = ex.outputs(1, 1, "cuda:0")
    scene = bo.BrepScene([data], "cuda:0")
    scene.render(cam, out)
    np.testing.assert_allclose(out[0].numpy(), [[offset]], rtol=2e-4, atol=2e-5)
    if data.face_flags[0] & loader.FLAG_FLIP:
        normal = -normal
    np.testing.assert_allclose(out[1].numpy()[0, 0], normal, atol=2e-4)


@pytest.mark.gpu
def test_nurbs_diagnostic_accounting(tmp_path, monkeypatch):
    """Exercise both builds by also running with WARP_OPTIX_BREP_DIAGNOSTICS=1."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples" / "brep"))
    import brep_optix as bo
    import example_brep as ex

    path = tmp_path / "dome.usda"
    _write_nurbs_dome(path)
    prims = loader.load_stage(str(path))
    lo, hi = prims[0].prim_bounds[:, :3].min(0), prims[0].prim_bounds[:, 3:].max(0)
    cam, *_ = bo.orbit_camera((lo + hi) * 0.5, bo.frame_distance(lo, hi), -55, 35, 64, 64, "Y")
    for proxy in (False, True):
        scene = bo.BrepScene(prims, "cuda:0", proxy=proxy)
        out = ex.outputs(64, 64, "cuda:0")
        scene.render(cam, out)
        wp.synchronize()
        counts = scene.data.diagnostic_counts.numpy()
        if not bd.DIAGNOSTICS:
            assert counts.shape == (0, len(bd.DIAGNOSTIC_NAMES))
            continue
        totals = dict(zip(bd.DIAGNOSTIC_NAMES, map(int, counts.sum(axis=0)), strict=True))
        assert totals["newton_calls"] > 0
        assert totals["newton_calls"] <= totals["newton_iterations"] <= 10 * totals["newton_calls"]
        assert totals["newton_failures"] <= totals["fallback_entries"] <= totals["newton_calls"]
        assert totals["newton_calls"] == (
            totals["aabb_candidates"] - totals["obb_rejects"] - totals["hull_rejects"] + totals["shell_candidates"]
        )
        histogram = [totals[f"newton_iterations_{i}"] for i in range(1, 11)]
        assert sum(histogram) == totals["newton_calls"]
        assert sum((i + 1) * count for i, count in enumerate(histogram)) == totals["newton_iterations"]
        assert totals["fallback_grazing"] + totals["fallback_nonconverged"] == totals["fallback_entries"]
        assert totals["fallback_nonconverged"] == totals["newton_failures"]
        assert totals["projection_calls"] <= totals["projection_evaluations"] <= 3 * totals["projection_calls"]
        assert np.isfinite(out[0].numpy()).all()


@wp.kernel(enable_backward=False)
def _project_clamped_plane(brep: bd.BrepData, result: wp.array[wp.vec4]):
    patch = bd.load_cubic(brep.cvs, 0)
    ray = bd.patch_ray(wp.vec3(5.0, 0.2, 5.0), wp.vec3(0.0, -1.0, 0.0), wp.vec3(-1.0, 0.0, -1.0))
    f = bd.foot(patch, brep, 0, ray, 0.0, wp.vec2(1.5), 2)
    result[0] = wp.vec4(f.st[0], f.st[1], f.h, f.g)


@pytest.mark.gpu
def test_projection_stops_at_unchanged_clamped_parameters():
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    # Affine x/z plane. The unconstrained projection lies beyond the extended square;
    # starting on its clamped corner must preserve that corner and its exact plane normal.
    grid = np.linspace(-1, 1, 4)
    cvs = np.array([[x, 0, z, 1] for x in grid for z in grid], dtype=np.float32)
    brep = bd.BrepData()
    brep.cvs = wp.array(cvs, dtype=wp.vec4, device="cuda:0")
    brep.diagnostic_counts = wp.zeros((1, len(bd.DIAGNOSTIC_NAMES)), dtype=wp.uint64, device="cuda:0")
    result = wp.empty(1, dtype=wp.vec4, device="cuda:0")
    wp.launch(_project_clamped_plane, 1, [brep, result], device="cuda:0")
    np.testing.assert_allclose(result.numpy()[0], [1.5, 1.5, -0.2, 1.0], rtol=1e-6, atol=1e-6)
    if bd.DIAGNOSTICS:
        counts = dict(zip(bd.DIAGNOSTIC_NAMES, map(int, brep.diagnostic_counts.numpy()[0]), strict=True))
        assert counts["projection_calls"] == counts["projection_evaluations"] == counts["projection_stalled"] == 1
