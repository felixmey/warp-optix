# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retained USD transform hierarchy for dynamic path-traced scenes."""

from __future__ import annotations

import math
import weakref
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from .scene import Scene


@dataclass(frozen=True, slots=True)
class UsdTransformHandle:
    """Stable transform identifier for one composed USD prim path."""

    index: int
    path: str
    _owner_id: int = field(repr=False)

    def __int__(self) -> int:
        return self.index


class UsdScene:
    """Path-addressable USD hierarchy retained after geometry import.

    Local-to-world hierarchy composition, render-node updates, and OptiX TLAS
    updates execute on CUDA. Device batches accept caller-owned ``wp.int32``
    handle arrays and ``wp.mat44`` local-transform arrays without staging
    through NumPy.
    """

    def __init__(
        self,
        scene: Scene,
        stage,
        source_path: str,
        paths: Iterable[str],
        *,
        root_transform: np.ndarray | None = None,
        time: float | None = None,
    ):
        # Weak, so the scene (which references this object as ``usd_scene``) is freed by refcounting.
        self._scene_ref = weakref.ref(scene)
        self._stage = stage
        self._source_path = str(source_path)
        self._handles = tuple(UsdTransformHandle(index, str(path), id(self)) for index, path in enumerate(paths))
        self._path_to_handle = {handle.path: handle for handle in self._handles}
        self._local_transforms_device = None
        self._world_transforms_device = None
        # Unit/up-axis/root conversion folded into parentless nodes' local matrices (column vectors).
        self._root_transform = np.eye(4) if root_transform is None else np.asarray(root_transform, dtype=np.float64)
        self._time = None if time is None or math.isnan(time) else float(time)
        self._animation = None  # (node indices, sampled groups, per-prim fallbacks), built on first use
        self._held_interpolation = False

    @property
    def _scene(self) -> Scene:
        scene = self._scene_ref()
        if scene is None:
            raise RuntimeError("USD scene handle was invalidated because its scene was released")
        return scene

    def _attach_device_arrays(self, local_transforms, world_transforms):
        self._local_transforms_device = local_transforms
        self._world_transforms_device = world_transforms

    @property
    def transforms(self) -> tuple[UsdTransformHandle, ...]:
        """All transformable composed USD paths, in stable hierarchy order."""
        return self._handles

    @property
    def stage(self):
        """The composed ``pxr.Usd.Stage`` used by the importer."""
        return self._stage

    @property
    def source_path(self) -> str:
        return self._source_path

    def get_prim(self, path: str):
        """Return any composed USD prim, including non-transformable prims."""
        return self._stage.GetPrimAtPath(str(path))

    @property
    def transform_count(self) -> int:
        return len(self._handles)

    @property
    def local_transforms_device(self) -> wp.array | None:
        """CUDA ``wp.mat44`` array, available after the renderer scene is built."""
        return self._local_transforms_device

    @property
    def world_transforms_device(self) -> wp.array | None:
        """CUDA-composed world ``wp.mat44`` array, available after build."""
        return self._world_transforms_device

    def get_transform(self, path: str) -> UsdTransformHandle | None:
        """Return the handle for an exact composed USD prim path."""
        return self._path_to_handle.get(str(path))

    def require_transform(self, path: str) -> UsdTransformHandle:
        """Return a handle or raise ``KeyError`` when the path is not transformable."""
        handle = self.get_transform(path)
        if handle is None:
            raise KeyError(f"USD path has no transform handle: {path}")
        return handle

    def instance_ids(self, handle: UsdTransformHandle | int) -> tuple[int, ...]:
        """Return render instances attached directly to a transform node."""
        node_index = self._index(handle)
        selected = self._scene._usd_instance_ids[self._scene._usd_instance_node_ids == node_index]
        return tuple(int(value) for value in selected)

    def set_visible(self, visible: bool):
        """Show or hide every render instance imported from this USD stage."""
        self._require_current()
        instance_ids = self._scene._usd_instance_ids
        if len(instance_ids) == 0:
            return
        self._scene.set_instances_visible_batch(instance_ids, bool(visible))

    def get_local_transform(self, handle: UsdTransformHandle | int) -> np.ndarray:
        """Read one local matrix; this synchronizes when CUDA state exists."""
        index = self._index(handle)
        if self._local_transforms_device is not None:
            return self._local_transforms_device.numpy()[index].copy()
        return self._scene._usd_local_transforms[index].copy()

    def get_world_transform(self, handle: UsdTransformHandle | int) -> np.ndarray:
        """Read one composed world matrix; this synchronizes CUDA work."""
        index = self._index(handle)
        if self._world_transforms_device is not None:
            return self._world_transforms_device.numpy()[index].copy()
        return self._scene._usd_world_transforms[index].copy()

    def update_local_transform(
        self,
        handle: UsdTransformHandle | int,
        matrix: np.ndarray,
        *,
        stream=None,
        rebuild_tlas: bool = True,
    ):
        """Update one local matrix through the batched CUDA path."""
        self.update_local_transforms([handle], [matrix], stream=stream, rebuild_tlas=rebuild_tlas)

    def update_local_transforms(
        self,
        handles: Iterable[UsdTransformHandle | int],
        matrices: np.ndarray,
        *,
        stream=None,
        rebuild_tlas: bool = True,
    ):
        """Upload and apply a NumPy batch on an optional Warp CUDA stream."""
        indices = np.fromiter((self._index(handle) for handle in handles), dtype=np.int32)
        self._scene.set_usd_local_transforms(indices, matrices, stream=stream, rebuild_tlas=rebuild_tlas)

    def update_local_transforms_device(
        self,
        transform_count: wp.array,
        transform_ids: wp.array,
        local_transforms: wp.array,
        *,
        stream=None,
        rebuild_tlas: bool = True,
    ):
        """Apply a device-counted prefix of CUDA handle and local-matrix arrays."""
        self._require_current()
        self._scene.set_usd_local_transforms_device(
            transform_count,
            transform_ids,
            local_transforms,
            stream=stream,
            rebuild_tlas=rebuild_tlas,
        )

    def update_local_transform_trs_device(
        self,
        transform_count: wp.array,
        transform_ids: wp.array,
        local_poses: wp.array,
        local_scales: wp.array,
        *,
        stream=None,
        rebuild_tlas: bool = True,
    ):
        """Apply a device-counted prefix of CUDA pose and scale arrays."""
        self._require_current()
        self._scene.set_usd_local_transform_trs_device(
            transform_count,
            transform_ids,
            local_poses,
            local_scales,
            stream=stream,
            rebuild_tlas=rebuild_tlas,
        )

    @property
    def time_code_range(self) -> tuple[float, float] | None:
        """The stage's authored ``(startTimeCode, endTimeCode)``, or None for a static stage."""
        if not self._stage.HasAuthoredTimeCodeRange():
            return None
        return float(self._stage.GetStartTimeCode()), float(self._stage.GetEndTimeCode())

    @property
    def time_codes_per_second(self) -> float:
        return float(self._stage.GetTimeCodesPerSecond())

    @property
    def time(self) -> float | None:
        """The time code the transforms were last evaluated at; None for the default time."""
        return self._time

    @property
    def animated_transforms(self) -> tuple[UsdTransformHandle, ...]:
        """Handles whose local transform may change over time (time-sampled xformOps)."""
        return tuple(self._handles[index] for index in self._animated()[0])

    def set_time(self, time: float, *, stream=None, rebuild_tlas: bool = True) -> int:
        """Re-evaluate time-sampled transforms at a USD time code and apply them as one batch.

        Only rigid transform animation is played back; animated points, visibility or materials
        are not. Returns the number of transforms updated.

        The first call samples every animated transform at its authored time codes. Transforms
        whose only time-varying op is a matrix, translate or scale op are then interpolated from
        those samples in one vectorized step, which matches USD's linear (or held) interpolation
        exactly; any other animation (rotate angles, quaternions, several varying ops) is still
        evaluated through USD for each prim.
        """
        self._require_current()
        indices, groups, fallback = self._animated()
        self._time = float(time)
        if len(indices) == 0:
            return 0
        matrices = np.empty((len(indices), 4, 4), dtype=np.float32)
        held = self._held_interpolation
        for slots, times, samples in groups:
            upper = int(np.searchsorted(times, self._time, side="right"))
            if upper == 0:
                matrices[slots] = samples[0]
            elif upper == len(times) or held:
                matrices[slots] = samples[upper - 1]
            else:
                weight = (self._time - times[upper - 1]) / (times[upper] - times[upper - 1])
                matrices[slots] = samples[upper - 1] + np.float32(weight) * (samples[upper] - samples[upper - 1])
        if fallback:
            from pxr import Usd  # noqa: PLC0415

            code = Usd.TimeCode(self._time)
            for slot, evaluate, is_root in fallback:
                local = evaluate(code)
                matrices[slot] = self._root_transform @ local if is_root else local
        self._scene.set_usd_local_transforms(indices, matrices, stream=stream, rebuild_tlas=rebuild_tlas)
        return len(indices)

    def _animated(self):
        """``(node indices, sampled groups, per-prim fallbacks)`` for the time-varying transforms."""
        if self._animation is not None:
            return self._animation
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        self._held_interpolation = self._stage.GetInterpolationType() == Usd.InterpolationTypeHeld
        linear_ops = {UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeScale}
        parents = self._scene._usd_node_parents
        indices, sampled, fallback = [], {}, []
        for handle in self._handles:
            prim = self._stage.GetPrimAtPath(handle.path)
            if not prim:
                continue
            xformable = UsdGeom.Xformable(prim)
            if xformable:
                ops = xformable.GetOrderedXformOps()

                def evaluate(code, x=xformable):
                    return np.asarray(x.GetLocalTransformation(code), np.float64).T

            elif prim.GetTypeName() == "BrepArray":
                # Not an Xformable without the omniSolid schema plugin; read its xformOps directly.
                order = prim.GetAttribute("xformOpOrder")
                names = (order.Get() if order and order.HasAuthoredValue() else None) or []
                ops = [
                    UsdGeom.XformOp(prim.GetAttribute(name.removeprefix("!invert!")), name.startswith("!invert!"))
                    for name in names
                    if name != "!resetXformStack!"
                ]
                from ..brep.loader import local_transform  # noqa: PLC0415

                def evaluate(code, p=prim):
                    return local_transform(p, code)[0]

            else:
                continue
            varying = [op for op in ops if op and op.GetAttr().ValueMightBeTimeVarying()]
            if not varying:
                continue
            slot = len(indices)
            indices.append(handle.index)
            is_root = bool(parents[handle.index] < 0)
            op = varying[0]
            if len(varying) == 1 and op.GetOpType() in linear_ops and not op.IsInverseOp():
                # The local matrix is linear in this op's value, so interpolating whole matrices
                # between its samples equals USD interpolating the op and composing.
                times = tuple(float(t) for t in op.GetAttr().GetTimeSamples())
                sampled.setdefault(times, []).append((slot, evaluate, is_root))
            else:
                fallback.append((slot, evaluate, is_root))

        groups = []
        for times, members in sampled.items():
            samples = np.empty((len(times), len(members), 4, 4), dtype=np.float32)
            for column, (_, evaluate, is_root) in enumerate(members):
                for row, t in enumerate(times):
                    local = evaluate(Usd.TimeCode(t))
                    samples[row, column] = self._root_transform @ local if is_root else local
            slots = np.fromiter((slot for slot, _, _ in members), dtype=np.intp, count=len(members))
            groups.append((slots, np.asarray(times, dtype=np.float64), samples))
        self._animation = (np.asarray(indices, dtype=np.int32), groups, fallback)
        return self._animation

    def update_tlas(self, *, stream=None):
        """Update OptiX traversal after batches submitted with auto-update off."""
        self._require_current()
        if stream is None:
            self._scene.rebuild_tlas()
            return
        with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
            self._scene.rebuild_tlas()

    def _index(self, handle: UsdTransformHandle | int) -> int:
        self._require_current()
        index = int(handle)
        if index < 0 or index >= len(self._handles):
            raise IndexError(f"USD transform handle is out of range: {index}")
        if isinstance(handle, UsdTransformHandle) and handle._owner_id != id(self):
            raise ValueError("USD transform handle belongs to a different stage")
        return index

    def _require_current(self):
        if self._scene.usd_scene is not self:
            raise RuntimeError("USD scene handle was invalidated by a later scene load or clear")
