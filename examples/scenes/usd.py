# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial camera selection for user-supplied USD stages."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .style import camera_angles


def world_bounds(api):
    scene = api.scene
    instance_count = len(scene._instances)
    if not instance_count:
        return np.full(3, -0.5), np.full(3, 0.5)

    mesh_indices = np.fromiter((instance.mesh_index for instance in scene._instances), np.int64, instance_count)
    unique_meshes, mesh_slots = np.unique(mesh_indices, return_inverse=True)
    local_bounds = np.empty((len(unique_meshes), 2, 3))
    for slot, mesh_index in enumerate(unique_meshes):
        vertices = scene._meshes[mesh_index].vertices
        local_bounds[slot] = np.min(vertices, axis=0), np.max(vertices, axis=0)
    # The 8 corners of every instance's local box, transformed in one batch.
    corner_bits = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=bool)
    boxes = local_bounds[mesh_slots]
    corners = np.where(corner_bits[None], boxes[:, None, 1], boxes[:, None, 0])
    world = scene._instance_transform_cache[:instance_count].astype(np.float64)
    transformed = corners @ world[:, :3, :3].transpose(0, 2, 1) + world[:, None, :3, 3]
    minima, maxima = transformed.min(axis=1), transformed.max(axis=1)

    # Ignore huge flat ground planes when framing so a small subject stays visible.
    sizes = maxima - minima
    diagonals = np.linalg.norm(sizes, axis=1)
    positive = diagonals[diagonals > 1.0e-12]
    if len(positive):
        ordered = np.sort(positive)
        # The lower half keeps the outliers themselves from inflating the reference size.
        typical = float(np.median(ordered[: max(1, len(ordered) // 2)]))
    else:
        typical = 0.0
    longest = np.maximum(np.max(sizes, axis=1), 1.0e-20)
    is_flat = np.min(sizes, axis=1) <= longest * 1.0e-5
    is_extreme = diagonals > typical * 25.0 if typical > 0.0 else np.zeros(instance_count, dtype=bool)
    selected = ~(is_flat & is_extreme)
    if not selected.any():
        selected[:] = True
    return minima[selected].min(axis=0), maxima[selected].max(axis=0)


def authored_camera(api, requested_path: str | None = None):
    from pxr import UsdGeom, UsdRender

    usd_scene = api.usd_scene
    stage = usd_scene.stage
    camera_paths = []
    if requested_path:
        camera_paths.append(requested_path)
    else:
        settings = UsdRender.Settings.GetStageRenderSettings(stage)
        if settings:
            camera_paths.extend(str(path) for path in settings.GetCameraRel().GetTargets())
        custom_data = stage.GetPseudoRoot().GetMetadata("customLayerData") or {}
        bound_camera = custom_data.get("cameraSettings", {}).get("boundCamera")
        if bound_camera:
            camera_paths.append(str(bound_camera))
        unsuitable_tokens = ("follow", "velocity", "physics", "collision", "debug")
        authored = [
            str(prim.GetPath())
            for prim in stage.TraverseAll()
            if prim.IsA(UsdGeom.Camera)
            and str(UsdGeom.Imageable(prim).ComputeVisibility()) != "invisible"
            and not any(token in prim.GetName().lower() for token in unsuitable_tokens)
        ]

        def camera_score(path):
            name = Path(path).name.lower()
            return (0 if "overview" in name else 1 if "main" in name else 2, path)

        camera_paths.extend(sorted(authored, key=camera_score))

    seen = set()
    for camera_path in camera_paths:
        if camera_path in seen:
            continue
        seen.add(camera_path)
        prim = stage.GetPrimAtPath(camera_path)
        handle = usd_scene.get_transform(camera_path)
        if not prim or handle is None or not prim.IsA(UsdGeom.Camera):
            continue
        camera = UsdGeom.Camera(prim)
        if str(camera.GetProjectionAttr().Get()) != "perspective":
            continue
        focal_length = float(camera.GetFocalLengthAttr().Get() or 0.0)
        aperture = float(camera.GetVerticalApertureAttr().Get() or 0.0)
        if focal_length <= 0.0 or aperture <= 0.0:
            continue
        world = usd_scene.get_world_transform(handle)
        position, yaw, pitch = camera_angles(world[:3, 3], -world[:3, 2])
        fov = math.degrees(2.0 * math.atan(aperture / (2.0 * focal_length)))
        return position, yaw, pitch, float(np.clip(fov, 5.0, 120.0)), camera_path
    if requested_path:
        raise ValueError(f"USD camera path is missing, invisible, or unsupported: {requested_path}")
    return None


def camera_for_scene(api, requested_path=None, use_authored=True):
    bounds_min, bounds_max = world_bounds(api)
    center = (bounds_min + bounds_max) * 0.5
    radius = max(float(np.linalg.norm(bounds_max - bounds_min)) * 0.5, 1.0e-3)
    if use_authored:
        authored = authored_camera(api, requested_path)
        if authored is not None:
            position, yaw, pitch, fov, camera_path = authored
            return position, yaw, pitch, fov, radius, camera_path
    # Three-quarter view with the eye kept above the top of the bounds.
    position = center + radius * np.array((1.25, 0.65, 1.25), dtype=np.float32)
    position[1] = max(position[1], float(bounds_max[1]) + radius * 0.25)
    position, yaw, pitch = camera_angles(position, center - position)
    return position, yaw, pitch, 55.0, radius, None


def default_camera_speed(scene_radius: float) -> float:
    """Return a useful speed for centimeter- through world-scale stages."""
    return max(float(scene_radius) * 0.2, 0.05)
