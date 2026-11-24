# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental BRep (OpenUSD ``BrepArray``) support: loader and path tracer integration."""

import math

import numpy as np
import pytest
import warp as wp

pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402
from warp_optix._src.brep import loader  # noqa: E402

pytestmark = pytest.mark.usd


def _write_sphere_brep(path, radius=1.0, trimmed=False, emissive=False, translate=None):
    """A one-face solid: a full sphere, optionally trimmed to its upper half by a seam/pole loop.

    Written attribute by attribute, as the schema plugin is not needed to read or render it.
    """
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = stage.DefinePrim("/Ball", "BrepArray")
    if translate is not None:
        op = prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
        op.Set(Gf.Vec3d(*translate))
        prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray, variability=Sdf.VariabilityUniform).Set(
            ["xformOp:translate"]
        )

    def attr(name, type_name, value):
        prim.CreateAttribute(name, type_name, custom=False, variability=Sdf.VariabilityUniform).Set(value)

    t = Sdf.ValueTypeNames
    attr("brep:extent", t.Double3Array, [Gf.Vec3d(-radius), Gf.Vec3d(radius)])
    attr("brep:regionCount", t.UIntArray, [2])
    attr("region:type", t.TokenArray, ["voidRegion", "solidRegion"])
    attr("region:shellCount", t.UIntArray, [1, 1])
    attr("shell:faceuseCount", t.UIntArray, [1, 1])
    attr("faceuse:faceIndex", t.UIntArray, [0, 0])
    # The void side of the face is its normal side: the surface normal points out of the solid.
    attr("faceuse:orientationType", t.TokenArray, ["same", "opposite"])
    attr("face:surfaceType", t.TokenArray, ["BrepSurfaceSphereAPI"])
    attr("face:range", t.Double2Array, [Gf.Vec2d(0.0, -math.pi / 2), Gf.Vec2d(2 * math.pi, math.pi / 2)])
    attr("brep:surface:sphere:center", t.Point3dArray, [Gf.Vec3d(0.0)])
    attr("brep:surface:sphere:axis", t.Vector3dArray, [Gf.Vec3d(0.0, 1.0, 0.0)])
    attr("brep:surface:sphere:refDirection", t.Vector3dArray, [Gf.Vec3d(1.0, 0.0, 0.0)])
    attr("brep:surface:sphere:radius", t.DoubleArray, [radius])
    if not trimmed:
        attr("face:trimType", t.TokenArray, ["rectangular"])
        attr("face:loopCount", t.UIntArray, [0])
    else:
        # Upper hemisphere: the equator circle, then the seam up to the pole and back down.
        attr("face:trimType", t.TokenArray, ["general"])
        attr("face:loopCount", t.UIntArray, [1])
        attr("loop:edgeuseCount", t.UIntArray, [3])
        attr("edgeuse:edgeIndex", t.UIntArray, [0, 1, 1])
        attr("edgeuse:orientationType", t.TokenArray, ["same", "same", "opposite"])
        attr("edge:curveType", t.TokenArray, ["BrepCurve3dCircleAPI", "BrepCurve3dLineAPI"])
        attr("edge:range", t.DoubleArray, [0.0, 2 * math.pi, 0.0, radius])
        attr("brep:edge3dCircle:curve3d:circle:center", t.Point3dArray, [Gf.Vec3d(0.0)])
        attr("brep:edge3dCircle:curve3d:circle:axis", t.Vector3dArray, [Gf.Vec3d(0.0, 1.0, 0.0)])
        attr("brep:edge3dCircle:curve3d:circle:refDirection", t.Vector3dArray, [Gf.Vec3d(1.0, 0.0, 0.0)])
        attr("brep:edge3dCircle:curve3d:circle:radius", t.DoubleArray, [radius])
        # The seam is not a straight line on the sphere; a quarter circle in the xy plane is.
        attr("edge:curveType", t.TokenArray, ["BrepCurve3dCircleAPI", "BrepCurve3dCircleAPI"])
        attr("edge:range", t.DoubleArray, [0.0, 2 * math.pi, 0.0, math.pi / 2])
        attr("brep:edge3dCircle:curve3d:circle:center", t.Point3dArray, [Gf.Vec3d(0.0)] * 2)
        attr("brep:edge3dCircle:curve3d:circle:axis", t.Vector3dArray, [Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)])
        attr("brep:edge3dCircle:curve3d:circle:refDirection", t.Vector3dArray, [Gf.Vec3d(1, 0, 0), Gf.Vec3d(1, 0, 0)])
        attr("brep:edge3dCircle:curve3d:circle:radius", t.DoubleArray, [radius, radius])
    if emissive:
        material = UsdShade.Material.Define(stage, "/Looks/Glow")
        shader = UsdShade.Shader.Define(stage, "/Looks/Glow/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    stage.GetRootLayer().Save()
    return stage


def _inside(prim, face, u, v):
    """The device trim test (range check plus crossing count), evaluated on the host."""
    umin, umax, vmin, vmax = prim.face_range[face]
    pu, pv = prim.face_period[face]
    if pu:
        u = umin + np.mod(u - umin, pu)
    if not (umin <= u <= umax and vmin <= v <= vmax):
        return False
    if not prim.face_flags[face] & loader.FLAG_TRIMMED:
        return True
    a, b = (v, u) if prim.face_flags[face] & loader.FLAG_SWAP else (u, v)
    base, count = prim.face_slab[face]
    lo, scale = prim.face_slab_map[face]
    slab = base + int(np.clip(np.floor((a - lo) * scale), 0, count - 1))
    members = prim.slab_segments[prim.slab_start[slab] : prim.slab_start[slab + 1]]
    segs, bulge = prim.segments[members], prim.segment_bulge[members]
    straddle = segs[(segs[:, 0] <= a) != (segs[:, 2] <= a)]
    t = (a - straddle[:, 0]) / (straddle[:, 2] - straddle[:, 0])
    count = np.count_nonzero(straddle[:, 1] + t * (straddle[:, 3] - straddle[:, 1]) > b)
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    length2 = np.maximum(dx * dx + dy * dy, 1e-300)
    along = ((a - segs[:, 0]) * dx + (b - segs[:, 1]) * dy) / length2
    side = ((b - segs[:, 1]) * dx - (a - segs[:, 0]) * dy) / np.sqrt(length2)
    arc = 4.0 * bulge * along * (1.0 - along)
    flips = (
        (bulge != 0)
        & (along > 0)
        & (along < 1)
        & (((bulge > 0) & (side > 0) & (side < arc)) | ((bulge < 0) & (side < 0) & (side > arc)))
    )
    return (count + np.count_nonzero(flips)) % 2 == 1


def test_loader_reads_surfaces_orientation_and_seam_trims(tmp_path):
    stage = _write_sphere_brep(tmp_path / "half.usda", radius=2.0, trimmed=True)
    prim = loader.load_brep_prim(stage.GetPrimAtPath("/Ball"), np.eye(4))

    assert list(prim.face_kind) == [loader.SPHERE]
    surface = prim.surfaces[0]
    np.testing.assert_allclose(surface.evaluate(np.array([0.0]), np.array([0.0])), [[2.0, 0.0, 0.0]], atol=1e-12)
    # Outward normal: the solid region's faceuse is "opposite", so the normal is not flipped.
    assert prim.face_flags[0] & loader.FLAG_SOLID
    assert not prim.face_flags[0] & loader.FLAG_FLIP
    # The loop runs along the equator and twice along the seam through the pole, with no UV
    # trim curves: projecting it must place the two seam uses a period apart and close the loop
    # through the pole, leaving exactly the upper hemisphere.
    for u in np.linspace(0.05, 2 * math.pi - 0.05, 9):
        assert _inside(prim, 0, u, 0.6)
        assert not _inside(prim, 0, u, -0.6)
    # The bounds cover the full sphere face range (conservatively padded).
    bounds = prim.prim_bounds[0]
    assert np.all(bounds[:3] <= -2.0) and np.all(bounds[3:] >= 2.0)
    assert np.all(bounds[:3] > -2.1) and np.all(bounds[3:] < 2.1)


def test_loader_flags_closed_solids(tmp_path):
    """A solid is closed when each edge of its faces' loops is used twice; a loopless sphere face
    closes up by itself. The half sphere's equator edge bounds only one face: an open "solid"."""
    whole = _write_sphere_brep(tmp_path / "whole.usda")
    half = _write_sphere_brep(tmp_path / "half.usda", trimmed=True)
    whole_prim = loader.load_brep_prim(whole.GetPrimAtPath("/Ball"), np.eye(4))
    half_prim = loader.load_brep_prim(half.GetPrimAtPath("/Ball"), np.eye(4))
    assert whole_prim.face_flags[0] & loader.FLAG_CLOSED
    assert half_prim.face_flags[0] & loader.FLAG_SOLID
    assert not half_prim.face_flags[0] & loader.FLAG_CLOSED


def test_brep_xform_ops_apply_without_the_schema_plugin():
    stage = Usd.Stage.CreateInMemory()
    xform = UsdGeom.Xform.Define(stage, "/X")
    xform.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(10.0, 20.0, 30.0))
    xform.AddScaleOp().Set(Gf.Vec3f(1.0, 2.0, 3.0))
    xform.AddTranslateOp(opSuffix="pivot").Set(Gf.Vec3d(0.5, 0.0, 0.0))
    xform.AddTranslateOp(opSuffix="pivot", isInverseOp=True)
    brep = stage.DefinePrim("/B", "BrepArray")
    for attribute in xform.GetPrim().GetAuthoredAttributes():
        brep.CreateAttribute(attribute.GetName(), attribute.GetTypeName()).Set(attribute.Get())
    assert not UsdGeom.Xformable(brep)  # unknown type without the plugin
    local, reset = loader.local_transform(brep)
    np.testing.assert_allclose(local, np.array(xform.GetLocalTransformation()).T, atol=1e-12)
    assert not reset


def _silhouette(api):
    """Coverage and horizontal centroid (in pixels) of the bright pixels."""
    for _ in range(4):
        api.render_frame()
    image = np.asarray(api.get_frame()[..., :3], dtype=np.float64)
    luminance = image.mean(axis=-1)
    mask = luminance > 0.5 * luminance.max()
    return float(mask.mean()), float(np.nonzero(mask)[1].mean())


@pytest.mark.gpu
@pytest.mark.render
def test_path_tracer_renders_brep_like_the_tessellated_sphere(tmp_path):
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    from warp_optix.pathtracing import PathTracer  # noqa: PLC0415

    # The BrepArray's own xformOp places it, as USD does once the schema is known.
    _write_sphere_brep(tmp_path / "ball.usda", emissive=True, translate=(0.6, 0.0, 0.0))

    def new_api():
        api = PathTracer(width=96, height=96, denoiser="none", enable_set=False, backface_culling=False)
        api.initialize()
        return api

    api = new_api()
    try:
        assert api.scene.load_from_usd(str(tmp_path / "ball.usda"))
        assert api.scene.brep_count == 1
        api.build_scene()
        api.set_environment_color((0.0, 0.0, 0.0))
        api.set_camera_look_at((0.0, 0.0, 4.0), (0.0, 0.0, 0.0))
        brep_coverage, brep_x = _silhouette(api)
    finally:
        api.close()

    api = new_api()
    try:
        glow = api.create_emissive_material((1.0, 1.0, 1.0), intensity=1.0)
        api.scene.add_sphere((0.6, 0.0, 0.0), 1.0, 256, glow)
        api.build_scene()
        api.set_environment_color((0.0, 0.0, 0.0))
        api.set_camera_look_at((0.0, 0.0, 4.0), (0.0, 0.0, 0.0))
        mesh_coverage, mesh_x = _silhouette(api)
    finally:
        api.close()

    assert brep_coverage > 0.05
    assert abs(brep_coverage - mesh_coverage) < 0.03 * mesh_coverage
    assert abs(brep_x - mesh_x) < 0.5  # pixels
    assert abs(brep_x - 48.0) > 5.0  # and the translation did move it off center


def _write_nurbs_dome(path):
    """One untrimmed bicubic NURBS face: a dome over [-1, 1]^2, emissive."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = stage.DefinePrim("/Dome", "BrepArray")
    t = Sdf.ValueTypeNames

    def attr(name, type_name, value):
        prim.CreateAttribute(name, type_name, custom=False, variability=Sdf.VariabilityUniform).Set(value)

    grid = np.linspace(-1.0, 1.0, 4)
    heights = np.array([[0.0, 0.3, 0.3, 0.0], [0.3, 1.2, 1.2, 0.3], [0.3, 1.2, 1.2, 0.3], [0.0, 0.3, 0.3, 0.0]])
    cvs = [Gf.Vec3d(float(grid[i]), float(heights[i, j]), float(grid[j])) for i in range(4) for j in range(4)]
    attr("brep:extent", t.Double3Array, [Gf.Vec3d(-1.0, 0.0, -1.0), Gf.Vec3d(1.0, 1.2, 1.0)])
    attr("brep:regionCount", t.UIntArray, [1])
    attr("region:type", t.TokenArray, ["voidRegion"])
    attr("region:shellCount", t.UIntArray, [1])
    attr("shell:faceuseCount", t.UIntArray, [2])
    attr("faceuse:faceIndex", t.UIntArray, [0, 0])
    attr("faceuse:orientationType", t.TokenArray, ["same", "opposite"])
    attr("face:surfaceType", t.TokenArray, ["BrepSurfaceNurbAPI"])
    attr("face:trimType", t.TokenArray, ["rectangular"])
    attr("face:loopCount", t.UIntArray, [0])
    attr("face:range", t.Double2Array, [Gf.Vec2d(0.0, 0.0), Gf.Vec2d(1.0, 1.0)])
    attr("brep:surface:nurb:controlVertices", t.Point3dArray, cvs)
    attr("brep:surface:nurb:weights", t.DoubleArray, [1.0] * 16)
    attr("brep:surface:nurb:uVertexCount", t.UIntArray, [4])
    attr("brep:surface:nurb:vVertexCount", t.UIntArray, [4])
    attr("brep:surface:nurb:uOrder", t.UIntArray, [4])
    attr("brep:surface:nurb:vOrder", t.UIntArray, [4])
    attr("brep:surface:nurb:uKnots", t.DoubleArray, [0.0] * 4 + [1.0] * 4)
    attr("brep:surface:nurb:vKnots", t.DoubleArray, [0.0] * 4 + [1.0] * 4)
    material = UsdShade.Material.Define(stage, "/Looks/Glow")
    shader = UsdShade.Shader.Define(stage, "/Looks/Glow/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    stage.GetRootLayer().Save()


def _bicubic(pw, s, t):
    """Reference: a bicubic rational Bezier patch, (4, 4, 4) homogeneous control points, at (s, t)."""

    def bernstein(x):
        y = 1.0 - x
        return np.stack([y**3, 3 * x * y**2, 3 * x**2 * y, x**3], axis=-1)

    h = np.einsum("ni,nj,ijk->nk", bernstein(s), bernstein(t), pw)
    return h[:, :3] / h[:, 3:]


def test_shell_proxies_enclose_bicubic_patches(tmp_path):
    from warp_optix._src.brep import proxy  # noqa: PLC0415

    _write_nurbs_dome(tmp_path / "dome.usda")
    (prim,) = loader.load_stage(str(tmp_path / "dome.usda"))
    shells = proxy.build_proxy(prim)
    assert len(shells.patches) == len(prim.prim_kind) > 0
    assert len(shells.indices) == len(shells.tri_prim) == len(shells.patches) * len(proxy.SHELL_TRIANGLES)
    m = proxy.GRID + 1
    layers = shells.points.reshape(len(shells.patches), 2, m, m, 3)  # outer, inner
    rng = np.random.default_rng(0)
    for k, p in enumerate(shells.patches):
        pw = prim.cvs[prim.prim_cv[p] : prim.prim_cv[p] + 16].reshape(4, 4, 4)
        outer, inner = layers[k, 0].astype(np.float64), layers[k, 1].astype(np.float64)
        # The surface lies between the layers: on the segment joining the layers' points at its
        # parameters (each layer interpolated like the shell's triangles; the layers are offset
        # asymmetrically, by the surface's deviation on each side).
        s, t = rng.uniform(0.0, 1.0, 200), rng.uniform(0.0, 1.0, 200)
        cells = proxy.GRID / (1.0 + 2.0 * proxy.MARGIN)
        gs, gt = (s + proxy.MARGIN) * cells, (t + proxy.MARGIN) * cells
        i, j = np.minimum(gs.astype(int), proxy.GRID - 1), np.minimum(gt.astype(int), proxy.GRID - 1)
        a, b = (gs - i)[:, None], (gt - j)[:, None]

        def linear(layer, i=i, j=j, a=a, b=b):
            p00, p10, p01, p11 = layer[i, j], layer[i + 1, j], layer[i, j + 1], layer[i + 1, j + 1]
            return np.where(a >= b, p00 + a * (p10 - p00) + b * (p11 - p10), p00 + b * (p01 - p00) + a * (p11 - p01))

        lo, hi = linear(inner), linear(outer)
        span = hi - lo
        along = ((_bicubic(pw, s, t) - lo) * span).sum(axis=1) / (span * span).sum(axis=1)
        assert along.min() > 0.0 and along.max() < 1.0


@pytest.mark.gpu
@pytest.mark.render
def test_path_traced_shells_match_the_custom_nurbs_intersector(tmp_path, monkeypatch):
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    from warp_optix._src.pathtracing import scene as scene_module  # noqa: PLC0415
    from warp_optix.pathtracing import PathTracer  # noqa: PLC0415

    _write_nurbs_dome(tmp_path / "dome.usda")
    results = []
    original = scene_module.Brep.__init__
    for shells in (True, False):

        def init(self, *args, shells=shells, **kwargs):
            kwargs["shells"] = shells
            original(self, *args, **kwargs)

        monkeypatch.setattr(scene_module.Brep, "__init__", init)
        api = PathTracer(width=96, height=96, denoiser="none", enable_set=False, backface_culling=False)
        api.initialize()
        try:
            assert api.scene.load_from_usd(str(tmp_path / "dome.usda"))
            assert (api.scene.geometry_count == 2) == shells  # the BRep, plus its shells
            api.build_scene()
            api.set_environment_color((0.0, 0.0, 0.0))
            api.set_camera_look_at((0.0, 1.6, 3.2), (0.0, 0.4, 0.0))
            results.append(_silhouette(api))
        finally:
            api.close()
    (shell_coverage, shell_x), (custom_coverage, custom_x) = results
    assert shell_coverage > 0.05
    assert abs(shell_coverage - custom_coverage) < 0.01 * custom_coverage
    assert abs(shell_x - custom_x) < 0.5


def _write_holed_plate(path):
    """A planar face: the square [-1, 1]^2 in z = 0 with a square hole [-0.4, 0.4]^2, line edges."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = stage.DefinePrim("/Plate", "BrepArray")

    def attr(name, type_name, value):
        prim.CreateAttribute(name, type_name, custom=False, variability=Sdf.VariabilityUniform).Set(value)

    t = Sdf.ValueTypeNames
    outer = [(-1, -1), (1, -1), (1, 1), (-1, 1)]  # counterclockwise
    hole = [(-0.4, -0.4), (-0.4, 0.4), (0.4, 0.4), (0.4, -0.4)]  # clockwise
    starts, directions = [], []
    for loop in (outer, hole):
        for k, (x, y) in enumerate(loop):
            nx, ny = loop[(k + 1) % 4]
            starts.append(Gf.Vec3d(x, y, 0.0))
            directions.append(Gf.Vec3d(nx - x, ny - y, 0.0))
    attr("brep:extent", t.Double3Array, [Gf.Vec3d(-1, -1, 0), Gf.Vec3d(1, 1, 0)])
    attr("brep:regionCount", t.UIntArray, [2])
    attr("region:type", t.TokenArray, ["voidRegion", "solidRegion"])
    attr("region:shellCount", t.UIntArray, [1, 1])
    attr("shell:faceuseCount", t.UIntArray, [1, 1])
    attr("faceuse:faceIndex", t.UIntArray, [0, 0])
    attr("faceuse:orientationType", t.TokenArray, ["same", "opposite"])
    attr("face:surfaceType", t.TokenArray, ["BrepSurfacePlaneAPI"])
    attr("face:range", t.Double2Array, [Gf.Vec2d(-1.0, -1.0), Gf.Vec2d(1.0, 1.0)])
    attr("brep:surface:plane:origin", t.Point3dArray, [Gf.Vec3d(0.0)])
    attr("brep:surface:plane:axis", t.Vector3dArray, [Gf.Vec3d(0.0, 0.0, 1.0)])
    attr("brep:surface:plane:refDirection", t.Vector3dArray, [Gf.Vec3d(1.0, 0.0, 0.0)])
    attr("face:trimType", t.TokenArray, ["general"])
    attr("face:loopCount", t.UIntArray, [2])
    attr("loop:edgeuseCount", t.UIntArray, [4, 4])
    attr("edgeuse:edgeIndex", t.UIntArray, list(range(8)))
    attr("edgeuse:orientationType", t.TokenArray, ["same"] * 8)
    attr("edge:curveType", t.TokenArray, ["BrepCurve3dLineAPI"] * 8)
    attr("edge:range", t.DoubleArray, [0.0, 1.0] * 8)
    attr("brep:edge3dLine:curve3d:line:origin", t.Point3dArray, starts)
    attr("brep:edge3dLine:curve3d:line:direction", t.Vector3dArray, directions)
    stage.GetRootLayer().Save()


@pytest.mark.gpu
@pytest.mark.render
def test_path_traced_plate_shows_its_hole(tmp_path):
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    from warp_optix.pathtracing import PathTracer  # noqa: PLC0415

    _write_holed_plate(tmp_path / "plate.usda")
    api = PathTracer(width=96, height=96, denoiser="none", enable_set=False, backface_culling=False)
    api.initialize()
    try:
        assert api.scene.load_from_usd(str(tmp_path / "plate.usda"))
        api.build_scene()
        # Looking straight down (USD z-up becomes y-up) at the whole plate.
        api.set_camera_look_at((0.0, 3.0, 0.001), (0.0, 0.0, 0.0))
        api.render_frame()
        wp.synchronize()
        depth = api.linear_depth_output.numpy().copy()
    finally:
        api.close()
    hit = np.isfinite(depth) & (depth > 0) & (depth < 100.0)
    h, w = hit.shape
    # Across the middle row: plate, hole, plate, with the hole 0.8 / 2 of the plate's width.
    row = np.nonzero(hit[h // 2])[0]
    plate = row[-1] - row[0] + 1
    hole = plate - len(row)
    assert not hit[h // 2, w // 2] and plate > w // 2
    assert abs(hole / plate - 0.4) < 0.03


def _write_flat_cone_ring(path, slope):
    """A one-face sheet: the cone r = 1 + slope * z between radii 1 and 0.5, nearly flat for a steep slope
    (CAD booleans leave such cones; BoltMBox1's step ring has slope -687)."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = stage.DefinePrim("/Ring", "BrepArray")

    def attr(name, type_name, value):
        prim.CreateAttribute(name, type_name, custom=False, variability=Sdf.VariabilityUniform).Set(value)

    t = Sdf.ValueTypeNames
    lo, hi = sorted((0.0, -0.5 / slope))
    attr("brep:extent", t.Double3Array, [Gf.Vec3d(-1, -1, lo), Gf.Vec3d(1, 1, hi)])
    attr("brep:regionCount", t.UIntArray, [2])
    attr("region:type", t.TokenArray, ["voidRegion", "voidRegion"])
    attr("region:shellCount", t.UIntArray, [1, 1])
    attr("shell:faceuseCount", t.UIntArray, [1, 1])
    attr("faceuse:faceIndex", t.UIntArray, [0, 0])
    attr("faceuse:orientationType", t.TokenArray, ["same", "opposite"])
    attr("face:surfaceType", t.TokenArray, ["BrepSurfaceConeAPI"])
    attr("face:range", t.Double2Array, [Gf.Vec2d(0.0, lo), Gf.Vec2d(2 * math.pi, hi)])
    attr("brep:surface:cone:origin", t.Point3dArray, [Gf.Vec3d(0.0)])
    attr("brep:surface:cone:axis", t.Vector3dArray, [Gf.Vec3d(0.0, 0.0, 1.0)])
    attr("brep:surface:cone:refDirection", t.Vector3dArray, [Gf.Vec3d(1.0, 0.0, 0.0)])
    attr("brep:surface:cone:radius", t.DoubleArray, [1.0])
    attr("brep:surface:cone:semiAngle", t.DoubleArray, [math.atan(slope)])
    attr("face:trimType", t.TokenArray, ["rectangular"])
    attr("face:loopCount", t.UIntArray, [0])
    stage.GetRootLayer().Save()


@pytest.mark.gpu
@pytest.mark.render
def test_path_traced_near_flat_cone_has_no_cracks(tmp_path):
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    from warp_optix.pathtracing import PathTracer  # noqa: PLC0415

    _write_flat_cone_ring(tmp_path / "ring.usda", slope=-5000.0)
    api = PathTracer(width=192, height=192, denoiser="none", enable_set=False, backface_culling=False)
    api.initialize()
    try:
        assert api.scene.load_from_usd(str(tmp_path / "ring.usda"))
        api.build_scene()
        for eye in ((0.0, 2.0, 1.6), (1.4, 1.4, 1.4)):  # oblique views (USD z-up becomes y-up)
            api.set_camera_look_at(eye, (0.0, 0.0, 0.0))
            api.render_frame()
            wp.synchronize()
            depth = api.linear_depth_output.numpy().copy()
            hit = np.isfinite(depth) & (depth > 0) & (depth < 100.0)
            # Every image row crosses the ring at most twice: holes would split a crossing in two.
            runs = [np.count_nonzero(np.diff(np.r_[0, row.astype(np.int8), 0]) == 1) for row in hit]
            assert max(runs) == 2 and hit.mean() > 0.15
    finally:
        api.close()


@pytest.mark.gpu
@pytest.mark.render
def test_path_traced_paths_end_inside_closed_opaque_solids(tmp_path):
    """No light reaches the inside of a closed opaque solid, so camera paths starting there end at their
    first hit, while views from outside, glass solids and solids holding a light render as before."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    from warp_optix.pathtracing import PathTracer  # noqa: PLC0415

    _write_sphere_brep(tmp_path / "ball.usda")

    def render(api, eye, target, dark=True):
        api._viewer.dark_solid_interiors = dark
        api._viewer.accumulate_samples = True
        api._viewer.sample_index = api._viewer.frame_index = 0  # restart accumulation: same samples each time
        api.set_camera_look_at(eye, target)
        for _ in range(8):
            api.render_frame()
        return np.asarray(api.get_frame()[..., :3], dtype=np.float64)

    def new_api(light=False):
        api = PathTracer(width=64, height=64, denoiser="none", enable_set=False, backface_culling=False)
        api.initialize()
        assert api.scene.load_from_usd(str(tmp_path / "ball.usda"))
        if light:
            api.scene.add_light_sphere((0.0, 0.0, 0.0), 0.1, (1.0, 1.0, 1.0), 50.0)
        api.set_instance_material(0, api.create_diffuse_material((0.8, 0.2, 0.1)))
        api.build_scene()
        api.set_environment_color((0.3, 0.3, 0.3))
        return api

    inside, outside = ((0.0, 0.0, 0.3), (0.0, 0.0, 1.0)), ((0.0, 0.0, 4.0), (0.0, 0.0, 0.0))
    api = new_api()
    try:
        assert not api.scene.lights_in_solids
        assert render(api, *inside).max() == 0.0
        lit = render(api, *outside)
        assert np.abs(lit[32, 32] - lit[0, 0]).max() > 0.05  # the ball differs from the sky
        np.testing.assert_allclose(lit, render(api, *outside, dark=False), atol=1e-6)
        api.set_instance_material(0, api.scene.materials.add_glass())
        api.build_scene()
        glass = render(api, *outside)
        np.testing.assert_allclose(glass, render(api, *outside, dark=False), atol=1e-6)
        assert np.abs(glass - lit).mean() > 0.01
    finally:
        api.close()

    api = new_api(light=True)
    try:
        assert api.scene.lights_in_solids
        assert render(api, *inside).mean() > 0.01
    finally:
        api.close()


def _write_instanced_assembly(directory, instanceable):
    """An assembly placing BRep parts through references, instanceable or flattened."""

    def color_material(stage, path, color):
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def place(stage, path, asset, translate, instance=instanceable, prim_path="/Part"):
        # Typeless, so the referenced prim's type (Xform or BrepArray) composes through.
        prim = stage.DefinePrim(path)
        prim.GetReferences().AddReference(asset, prim_path)
        prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*translate))
        prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray, variability=Sdf.VariabilityUniform).Set(
            ["xformOp:translate"]
        )
        prim.SetInstanceable(instance)
        return prim

    ball = directory / "ball.usda"
    if not ball.exists():
        _write_sphere_brep(ball, radius=0.25)
    part = directory / "part.usda"
    if not part.exists():
        stage = Usd.Stage.CreateNew(str(part))
        UsdGeom.Xform.Define(stage, "/Part")
        red = color_material(stage, "/Part/Looks/Red", (1.0, 0.0, 0.0))
        bound = place(stage, "/Part/Ball", str(ball), (0, 0, 0), instance=False, prim_path="/Ball")
        UsdShade.MaterialBindingAPI.Apply(bound).Bind(red)
        place(stage, "/Part/Sub", str(ball), (0, 1, 0), instance=False, prim_path="/Ball")
        reset = UsdGeom.Xform.Define(stage, "/Part/Reset")
        reset.SetResetXformStack(True)
        reset.AddTranslateOp().Set(Gf.Vec3d(0, 0, 5))
        place(stage, "/Part/Reset/Ball", str(ball), (1, 0, 0), instance=False, prim_path="/Ball")
        stage.GetRootLayer().Save()
        stage = Usd.Stage.CreateNew(str(directory / "pair.usda"))
        UsdGeom.Xform.Define(stage, "/Part")
        place(stage, "/Part/Left", str(part), (-1, 0, 0), instance=True)
        place(stage, "/Part/Right", str(part), (1, 0, 0), instance=True)
        stage.GetRootLayer().Save()
        stage = Usd.Stage.CreateNew(str(directory / "with_mesh.usda"))
        UsdGeom.Xform.Define(stage, "/Part")
        place(stage, "/Part/Ball", str(ball), (0, 0, 0), instance=False, prim_path="/Ball")
        mesh = UsdGeom.Mesh.Define(stage, "/Part/Tri")
        mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        stage.GetRootLayer().Save()

    path = directory / ("instanced.usda" if instanceable else "flattened.usda")
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    blue = color_material(stage, "/World/Looks/Blue", (0.0, 0.0, 1.0))
    green = color_material(stage, "/World/Looks/Green", (0.0, 1.0, 0.0))
    place(stage, "/World/A", str(part), (0, 0, 0))
    place(stage, "/World/A2", str(part), (3, 0, 0))
    UsdShade.MaterialBindingAPI.Apply(place(stage, "/World/B", str(part), (6, 0, 0))).Bind(blue)
    group = UsdGeom.Xform.Define(stage, "/World/G").GetPrim()
    UsdShade.MaterialBindingAPI.Apply(group).Bind(green, UsdShade.Tokens.strongerThanDescendants)
    place(stage, "/World/G/C", str(part), (9, 0, 0))
    UsdGeom.Imageable(place(stage, "/World/H", str(part), (12, 0, 0))).MakeInvisible()
    place(stage, "/World/M", str(directory / "with_mesh.usda"), (15, 0, 0))
    place(stage, "/World/M2", str(directory / "with_mesh.usda"), (18, 0, 0))
    place(stage, "/World/N", str(directory / "pair.usda"), (21, 0, 0))
    stage.GetRootLayer().Save()
    return path


def test_instanced_usd_breps_expand_like_flattened_references(tmp_path):
    """Instances expanded from a shared prototype template match the per-prim traversal."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    from warp_optix.pathtracing import PathTracer  # noqa: PLC0415

    def load(instanceable):
        api = PathTracer(width=32, height=32, denoiser="none", enable_set=False)
        api.initialize()
        try:
            scene = api.scene
            assert scene.load_from_usd(str(_write_instanced_assembly(tmp_path, instanceable)))
            colors = []
            for instance in scene._instances:
                if instance.material_id is not None:
                    ids = [instance.material_id]
                else:
                    ids = np.unique(scene._meshes[instance.mesh_index].material_ids)
                entries = scene.materials.get_material_entries()
                colors.append(tuple(tuple(np.round(entries[int(i)]["pbrBaseColorFactor"], 4)) for i in ids))
            return (
                [handle.path for handle in scene.usd_scene.transforms],
                scene._usd_node_parents.copy(),
                scene._usd_world_transforms.copy(),
                scene._usd_instance_node_ids.copy(),
                [scene._meshes[instance.mesh_index].primitive_type for instance in scene._instances],
                colors,
                scene._instance_transform_cache[: len(scene._instances)].copy(),
            )
        finally:
            api.close()

    paths, parents, world, instance_nodes, kinds, colors, transforms = load(instanceable=True)
    flat = load(instanceable=False)
    assert paths == flat[0]
    np.testing.assert_array_equal(parents, flat[1])
    np.testing.assert_allclose(world, flat[2], atol=1e-6)
    np.testing.assert_array_equal(instance_nodes, flat[3])
    assert kinds == flat[4]
    assert colors == flat[5]
    np.testing.assert_allclose(transforms, flat[6], atol=1e-6)
    # Sanity-check the scenario itself: the hidden instance is skipped, the rest resolve as authored.
    assert not any(path.startswith("/World/H/") and node in instance_nodes for node, path in enumerate(paths))
    by_path = {paths[node]: color for node, color in zip(instance_nodes, colors, strict=True)}
    red, blue, green = (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 0.0, 1.0)
    assert by_path["/World/A/Ball"] == (red,)
    assert by_path["/World/B/Ball"] == (red,) and by_path["/World/B/Sub"] == (blue,)
    assert by_path["/World/G/C/Ball"] == (green,) and by_path["/World/G/C/Sub"] == (green,)
    assert "/World/N/Right/Reset/Ball" in by_path
    np.testing.assert_allclose(world[paths.index("/World/N/Right/Reset/Ball")][:3, 3], (1.0, 5.0, 0.0), atol=1e-6)
