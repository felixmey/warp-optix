# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a USD BRep (``BrepArray``) by tracing its exact surfaces, and compare against meshes.

    uv run --extra usd --extra pathtracing python examples/brep/example_brep.py part.usda

``--live`` opens an interactive orbit viewer instead (needs ``--extra ui`` for the OpenGL interop).

``--compare`` (needs ``--with usd-brep``) also tessellates the BRep with SMLib at several chord
tolerances and reports, per tessellation, the trace time, acceleration structure size and the
depth/normal/coverage error of the mesh relative to the exact BRep render.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).parent))
import brep_optix as bo  # noqa: E402
from warp_optix._src.brep import loader as brep_usd  # noqa: E402


def outputs(width, height, device):
    return [
        wp.empty((height, width), dtype=wp.float32, device=device),
        wp.empty((height, width), dtype=wp.vec3, device=device),
        wp.empty((height, width), dtype=wp.int32, device=device),
        wp.empty((height, width), dtype=wp.int32, device=device),
    ]


def time_launches(scene, cam, out, repeats):
    scene.render(cam, out)  # warm-up
    wp.synchronize_device(scene.device)
    start = wp.Event(scene.device, enable_timing=True)
    stop = wp.Event(scene.device, enable_timing=True)
    stream = wp.get_stream(scene.device)
    stream.record_event(start)
    for _ in range(repeats):
        scene.render(cam, out)
    stream.record_event(stop)
    wp.synchronize_device(scene.device)
    return wp.get_event_elapsed_time(start, stop) / repeats


def save(path, out, view, color_faces, device):
    from PIL import Image  # noqa: PLC0415

    h, w = out[0].shape
    image = wp.empty((h, w, 3), dtype=wp.uint8, device=device)
    wp.launch(bo.shade, dim=(h, w), inputs=[*out, wp.vec3(*view), int(color_faces), image], device=device)
    Image.fromarray(image.numpy()).save(path)


class OrbitController:
    """Left-drag orbit, right/middle-drag (or Shift+left) pan, wheel zoom.
    F reframe, C toggle face colors, P save a screenshot (brep_live.png)."""

    def __init__(self, viewer, lo, hi, azimuth, elevation, up, width, height):
        self.pyglet = viewer.pyglet
        self.lo, self.hi, self.up = lo, hi, up
        self.width, self.height = int(width), int(height)
        self.home = (float(azimuth), float(elevation))
        self.color_faces = True
        self.screenshot = False
        self.reframe()
        viewer.window.push_handlers(self)

    def reframe(self):
        self.target = 0.5 * (self.lo + self.hi)
        self.distance = bo.frame_distance(self.lo, self.hi)
        self.azimuth, self.elevation = self.home
        self._update()

    def _update(self):
        self.camera, self.forward, self.right, self.upv = bo.orbit_camera(
            self.target, self.distance, self.azimuth, self.elevation, self.width, self.height, self.up
        )

    def resize(self, width, height):
        self.width, self.height = max(int(width), 1), max(int(height), 1)
        self._update()

    def on_mouse_drag(self, _x, _y, dx, dy, buttons, modifiers):
        mouse, key = self.pyglet.window.mouse, self.pyglet.window.key
        if buttons & (mouse.RIGHT | mouse.MIDDLE) or (buttons & mouse.LEFT and modifiers & key.MOD_SHIFT):
            scale = 2.0 * self.distance * np.tan(np.radians(35.0) * 0.5) / self.height
            self.target = self.target - (dx * self.right + dy * self.upv) * scale
        elif buttons & mouse.LEFT:
            self.azimuth -= dx * 0.3
            self.elevation = float(np.clip(self.elevation - dy * 0.3, -89.0, 89.0))
        self._update()

    def on_mouse_scroll(self, _x, _y, _sx, sy):
        self.distance *= 0.9 ** float(sy)
        self._update()

    def on_key_press(self, symbol, _modifiers):
        key = self.pyglet.window.key
        if symbol == key.F:
            self.reframe()
        elif symbol == key.C:
            self.color_faces = not self.color_faces
        elif symbol == key.P:
            self.screenshot = True


def run_live(scene, lo, hi, up, args):
    from warp_optix.gl import GLInteropViewer  # noqa: PLC0415

    device = scene.device
    state = {"out": outputs(args.width, args.height, device)}
    controller = None

    def on_resize(width, height):
        state["out"] = outputs(max(int(width), 1), max(int(height), 1), device)
        controller.resize(width, height)

    viewer = GLInteropViewer(
        width=args.width,
        height=args.height,
        device=device,
        title=f"BRep: {Path(args.usd).name}",
        fps=0,
        on_resize=on_resize,
    )
    controller = OrbitController(viewer, lo, hi, args.azimuth, args.elevation, up, args.width, args.height)

    def render(mapped_image: wp.array, _frame: int, _elapsed: float):
        out = state["out"]
        scene.render(controller.camera, out)
        view = wp.vec3(*controller.forward)
        wp.launch(
            bo.shade_packed,
            dim=out[0].shape,
            inputs=[*out, view, int(controller.color_faces), mapped_image],
            device=device,
        )
        if controller.screenshot:
            controller.screenshot = False
            save("brep_live.png", out, controller.forward, controller.color_faces, device)
            print("saved brep_live.png")

    print(OrbitController.__doc__)
    viewer.run(render, max_frames=args.max_frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("usd")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--azimuth", type=float, default=-55.0)
    parser.add_argument("--elevation", type=float, default=35.0)
    parser.add_argument("--up", choices=("Z", "Y"), default=None)
    parser.add_argument("--output", default="brep.png")
    parser.add_argument("--live", action="store_true", help="open an interactive viewer")
    parser.add_argument("--proxy", action="store_true", help="trace NURBS patches through triangle shells")
    parser.add_argument("--max-frames", type=int, default=0, help="exit the viewer after N frames (0 = never)")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--flat-tol", type=float, default=2e-3, help="NURBS sub-patch flatness, relative to extent")
    parser.add_argument("--compare", action="store_true", help="compare against SMLib tessellations")
    parser.add_argument(
        "--mesh-usd", help="compare against the triangle meshes of this USD stage (e.g. step_to_usd --mesh)"
    )
    parser.add_argument(
        "--chord",
        type=float,
        nargs="*",
        default=[1e-2, 2e-3, 4e-4, 1e-4],
        help="tessellation chord tolerances, relative to each part's extent diagonal",
    )
    args = parser.parse_args()

    from pxr import Usd, UsdGeom  # noqa: PLC0415

    wp.init()
    device = wp.get_device("cuda")
    t0 = time.perf_counter()
    prims = brep_usd.load_stage(args.usd, flat_tol=args.flat_tol)
    load_s = time.perf_counter() - t0
    up = args.up or str(UsdGeom.GetStageUpAxis(Usd.Stage.Open(args.usd)))

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for p in prims:
        b = p.prim_bounds
        corners = np.array([[b[:, i][None] for i in idx] for idx in [(0, 1, 2), (3, 4, 5)]]).reshape(2, 3, -1)
        pts = np.concatenate([corners[0].T, corners[1].T])
        world = (p.world[:3, :3] @ pts.T).T + p.world[:3, 3]
        lo, hi = np.minimum(lo, world.min(0)), np.maximum(hi, world.max(0))
    diag = float(np.linalg.norm(hi - lo))
    cam, view, _, _ = bo.orbit_camera(
        0.5 * (lo + hi), bo.frame_distance(lo, hi), args.azimuth, args.elevation, args.width, args.height, up
    )

    t0 = time.perf_counter()
    scene = bo.BrepScene(prims, device, proxy=args.proxy)
    wp.synchronize_device(device)
    build_s = time.perf_counter() - t0
    if args.live:
        run_live(scene, lo, hi, up, args)
        return
    out = outputs(args.width, args.height, device)
    brep_ms = time_launches(scene, cam, out, args.repeats)
    save(args.output, out, view, True, device)
    depth = out[0].numpy()
    hit = depth >= 0
    leaks = int(np.count_nonzero(out[3].numpy()[hit]))
    stats = {}
    for p in prims:
        for k, v in p.stats.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                stats[k] = stats.get(k, 0) + v
    print(f"{Path(args.usd).name}: {stats}")
    print(f"  load {load_s * 1e3:.0f} ms (CPU), scene+pipeline {build_s * 1e3:.0f} ms (incl. compile)")
    mpix = args.width * args.height / 1e6
    print(
        f"  BRep  : {brep_ms:7.3f} ms/frame ({mpix / brep_ms * 1e3:6.0f} Mrays/s)  "
        f"accel {scene.accel_bytes() / 1024:8.1f} KiB  data {scene.data_bytes() / 1024:8.1f} KiB  "
        f"coverage {hit.mean() * 100:5.2f}%  inside-hits {leaks}  -> {args.output}"
    )

    normals = out[1].numpy()

    def compare(mesh, label, triangles, tess_s=None):
        mout = outputs(args.width, args.height, device)
        mesh_ms = time_launches(mesh, cam, mout, args.repeats)
        mdepth = mout[0].numpy()
        mhit = mdepth >= 0
        both = hit & mhit
        rel = np.abs(mdepth[both] - depth[both]) / diag
        cos = np.abs(np.sum(mout[1].numpy()[both] * normals[both], axis=-1)).clip(0, 1)
        ang = np.degrees(np.arccos(cos))
        name = Path(args.output).with_name(f"{Path(args.output).stem}_mesh_{label}.png")
        save(name, mout, view, False, device)
        tess = f"  tess {tess_s * 1e3:6.0f} ms" if tess_s is not None else ""
        print(
            f"  mesh {label:>7}: {mesh_ms:7.3f} ms/frame ({mpix / mesh_ms * 1e3:6.0f} Mrays/s)  "
            f"accel {mesh.accel_bytes() / 1024:8.1f} KiB  data {mesh.data_bytes() / 1024:8.1f} KiB  "
            f"tris {triangles:8d}{tess} | coverage xor {np.count_nonzero(hit ^ mhit) / hit.size * 100:5.2f}%  "
            f"depth err mean {rel.mean():.1e} p99 {np.percentile(rel, 99):.1e}  "
            f"normal err mean {ang.mean():5.2f} deg p99 {np.percentile(ang, 99):5.2f} deg"
        )

    if args.mesh_usd:
        placements = load_mesh_stage(args.mesh_usd)
        mesh = bo.InstancedMeshScene(placements, device)
        compare(mesh, "usd", sum(len(i) for _, i, _ in placements))
    if not args.compare:
        return
    import usd_brep  # noqa: PLC0415

    for chord in args.chord:
        pts, tris = [], []
        base = 0
        cache = {}  # referenced copies of the same BRep are tessellated once
        t0 = time.perf_counter()
        for p in prims:
            key = (p.prim_bounds.tobytes(), p.face_kind.tobytes())
            if key not in cache:
                (brep,) = (
                    usd_brep.usd.import_breps(args.usd)
                    if len(prims) == 1
                    else (usd_brep.usd.import_brep(args.usd, p.path),)
                )
                extent = np.linalg.norm(np.ptp(p.prim_bounds[:, :3], axis=0))  # chord is per-part relative
                mesh = brep.tessellate(
                    chord_height_tolerance=chord * extent,
                    curve_angle_tolerance_deg=90.0,
                    surface_angle_tolerance_deg=90.0,
                )
                arrays = mesh.to_numpy_arrays()
                faces = np.asarray(arrays["faces"]).reshape(-1, 4)  # count-prefixed: [3, i, j, k]
                assert np.all(faces[:, 0] == 3), "expected triangles"
                cache[key] = (np.asarray(arrays["points"], dtype=np.float64).reshape(-1, 3), faces[:, 1:])
            v, idx = cache[key]
            v = (p.world[:3, :3] @ v.T).T + p.world[:3, 3]
            pts.append(v)
            tris.append(idx.reshape(-1, 3) + base)
            base += len(v)
        tess_s = time.perf_counter() - t0
        mesh = bo.MeshScene(np.concatenate(pts), np.concatenate(tris), device)
        compare(mesh, f"{chord:g}", sum(len(t) for t in tris), tess_s)


def load_mesh_stage(path):
    """Triangle meshes of a USD stage as (points, indices, world); instance proxies share arrays."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    stage = Usd.Stage.Open(path)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    predicate = Usd.TraverseInstanceProxies(Usd.PrimIsActive & Usd.PrimIsDefined & ~Usd.PrimIsAbstract)
    loaded, placements = {}, []
    for prim in stage.Traverse(predicate):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        key = str(prim.GetPrimInPrototype().GetPath()) if prim.IsInstanceProxy() else str(prim.GetPath())
        if key not in loaded:
            mesh = UsdGeom.Mesh(prim)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            assert np.all(counts == 3), "expected a triangle mesh"
            loaded[key] = (
                np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32),
                np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.uint32).reshape(-1, 3),
            )
        world = np.array(cache.GetLocalToWorldTransform(prim), dtype=np.float64).T
        placements.append((*loaded[key], world))
    return placements


if __name__ == "__main__":
    main()
