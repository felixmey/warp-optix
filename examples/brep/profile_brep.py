# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect opt-in per-patch NURBS work counters for one path-traced frame.

Run in a fresh process: diagnostics are specialized at import/compile time. Atomic counters
perturb execution; use ordinary builds for timing. Counts include camera, bounce and shadow rays.
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--bounces", type=int, default=4)
    parser.add_argument(
        "--camera",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "YAW", "PITCH", "FOV"),
        help="use a camera printed by the path-tracing viewer, with viewer USD units, axes, and culling",
    )
    parser.add_argument("--output", help="save raw color and guide buffers as an NPZ for comparisons")
    parser.add_argument("--no-counters", action="store_true", help="reference render without instrumentation")
    args = parser.parse_args()
    os.environ["WARP_OPTIX_BREP_DIAGNOSTICS"] = "0" if args.no_counters else "1"

    import brep_optix as bo  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415
    from pxr import Usd, UsdGeom  # noqa: PLC0415
    from warp_optix._src.brep import device as bd  # noqa: PLC0415
    from warp_optix._src.brep import loader  # noqa: PLC0415
    from warp_optix.pathtracing import PathTracer  # noqa: PLC0415

    wp.init()
    if args.camera is None:
        prims = loader.load_stage(args.usd)
        if not prims:
            parser.error("scene contains no BReps")
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for prim in prims:
            # Transform all eight corners, including rotated and reflected instances.
            bounds = prim.prim_bounds
            corners = np.stack(
                [
                    bounds[:, [3 if x else 0, 4 if y else 1, 5 if z else 2]]
                    for x in (0, 1)
                    for y in (0, 1)
                    for z in (0, 1)
                ]
            ).reshape(-1, 3)
            world = corners @ prim.world[:3, :3].T + prim.world[:3, 3]
            lo, hi = np.minimum(lo, world.min(0)), np.maximum(hi, world.max(0))
        up = str(UsdGeom.GetStageUpAxis(Usd.Stage.Open(args.usd)))
        cam, *_ = bo.orbit_camera((lo + hi) * 0.5, bo.frame_distance(lo, hi), -55, 35, args.width, args.height, up)
    api = PathTracer(
        width=args.width,
        height=args.height,
        denoiser="none",
        enable_set=False,
        backface_culling=args.camera is not None,
        max_bounces=args.bounces,
    )
    try:
        api.initialize()
        viewer_coordinates = args.camera is not None
        if not api.scene.load_from_usd(
            args.usd, apply_stage_units=viewer_coordinates, convert_up_axis=viewer_coordinates
        ):
            raise RuntimeError("USD scene load failed")
        if api.scene.brep_count == 0:
            parser.error("scene contains no BReps")
        api.build_scene()
        api.set_environment_color((0.5, 0.5, 0.5))
        if args.camera is not None:
            x, y, z, yaw, pitch, fov = args.camera
            api.set_camera_angles((x, y, z), yaw, pitch, fov)
        else:
            api.set_camera_look_at(
                tuple(cam.origin), tuple((lo + hi) * 0.5), (0.0, 0.0, 1.0) if up == "Z" else (0.0, 1.0, 0.0)
            )
        api.render_frame()  # compile and initialize before the measured frame
        wp.synchronize()
        viewer = api._viewer
        counts = api.scene._brep_data.diagnostic_counts
        counts.zero_()
        wp.synchronize()
        viewer.sample_index = viewer.frame_index = 0
        api.render_frame()
        wp.synchronize()
        if bd.DIAGNOSTICS:
            per_patch = counts.numpy()
            totals = per_patch.sum(axis=0)
            result = dict(zip(bd.DIAGNOSTIC_NAMES, map(int, totals), strict=True))
            calls = result["newton_calls"]
            result["mean_newton_iterations"] = result["newton_iterations"] / max(calls, 1)
            result["fallback_fraction_of_newton_calls"] = result["fallback_entries"] / max(calls, 1)
            result["mean_projection_evaluations"] = result["projection_evaluations"] / max(
                result["projection_calls"], 1
            )
            result["fallback_grazing_fraction"] = result["fallback_grazing"] / max(result["fallback_entries"], 1)
            result["patches_with_candidates"] = int(np.count_nonzero(per_patch[:, 0] + per_patch[:, 2]))
            print(json.dumps(result, indent=2))
        if args.output:
            names = ("color", "normal_roughness", "depth", "motion", "diffuse", "specular", "spec_hit_dist")
            np.savez(args.output, **{name: getattr(viewer, f"_{name}_buffer").numpy() for name in names})
    finally:
        api.close()


if __name__ == "__main__":
    main()
