# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ray trace USD BReps directly with warp-optix custom primitives.

Each analytic face (plane, cylinder, cone, sphere, torus) is one custom primitive whose AABB bounds
its UV range; each NURBS face becomes flat rational Bezier sub-patches (Martin et al., "Practical
Ray Tracing of Trimmed NURBS Surfaces", 2000). One intersection program per surface kind selects
its hit record through the primitive's SBT index; planes, cylinders, cones and spheres share one
quadric program. A candidate hit is mapped to the face's (u, v) and kept only if it lies inside
the face's trim loops (crossing test against UV polylines, bucketed into slabs).

The intersection program reports u, v, the outward object-space normal and the face index as
attributes; the hit kind is 0 for the outside of the solid and 1 for the inside.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from warp_optix._src.brep import device as bd
from warp_optix._src.brep import proxy as brep_proxy
from warp_optix._src.brep.loader import KIND_NAMES, BrepPrim

import warp_optix as wo

BrepData = bd.BrepData


@wp.struct
class HitPayload:
    t: float
    nx: float
    ny: float
    nz: float
    face: wp.int32
    inside: wp.int32
    u: float
    v: float


@wp.func
def primitive() -> int:
    # Each GAS instance's id is the global index of its first BRep primitive.
    return int(wp.optix_get_instance_id()) + int(wp.optix_get_primitive_index())


# One intersection program per surface kind, selected through the primitive's SBT index; planes,
# cylinders, cones and spheres share the quadric one.
@wp.kernel(module="unique")
def isect_quadric(brep: BrepData):
    bd.intersect_quadric(brep, primitive(), wp.optix_get_object_ray_origin(), wp.optix_get_object_ray_direction())


@wp.kernel(module="unique")
def isect_torus(brep: BrepData):
    bd.intersect_torus(brep, primitive(), wp.optix_get_object_ray_origin(), wp.optix_get_object_ray_direction())


@wp.kernel(module="unique")
def isect_nurbs(brep: BrepData):
    bd.intersect_nurbs(brep, primitive(), wp.optix_get_object_ray_origin(), wp.optix_get_object_ray_direction())


@wp.kernel(module="unique")
def proxy_any_hit(brep: BrepData):
    """A shell triangle was crossed: find the exact hit, keep the nearest one in the payload.

    OptiX orders hits by the shell triangle's t, not the surface's, and a shell can reach in front
    of another surface (thin walls). So the payload records the nearest exact hit, and hits that
    may hide a nearer crossing behind their shell triangle (leaving a solid, or a sheet's back met
    at a grazing angle; see proxy_hit) are recorded but not accepted: traversal continues.
    """
    local = int(wp.optix_get_primitive_index())
    tri = int(wp.optix_get_instance_id()) + local
    direction = wp.optix_get_object_ray_direction()
    ok, hit_t, u, v, n, face, keep_searching = bd.proxy_hit(
        brep,
        tri,
        local,
        wp.optix_get_triangle_barycentrics(),
        wp.optix_get_object_ray_origin(),
        direction,
    )
    if not ok:
        wp.optix_ignore_intersection()
        return
    leaving = wp.dot(n, direction) > 0.0
    payload = HitPayload()
    wp.optix_load_payload(payload)
    if payload.t < 0.0 or hit_t < payload.t:
        world_n = wp.normalize(wp.optix_transform_normal_object_to_world(n))
        payload.t = hit_t
        payload.nx = world_n[0]
        payload.ny = world_n[1]
        payload.nz = world_n[2]
        payload.face = wp.int32(face)
        payload.inside = wp.int32(0)
        if leaving and (brep.prims[brep.proxy_prim[tri]].flags & bd.FLAG_SOLID) != 0:
            payload.inside = wp.int32(1)
        payload.u = u
        payload.v = v
        wp.optix_store_payload(payload)
    if keep_searching:
        wp.optix_ignore_intersection()


@wp.kernel(module="unique")
def proxy_closest_hit():
    pass  # the any-hits left the nearest exact hit in the payload


INTERSECTORS = (isect_quadric, isect_quadric, isect_quadric, isect_quadric, isect_torus, isect_nurbs)


# ---------------------------------------------------------------------------------------------
# Camera, raygen, hit and miss programs
# ---------------------------------------------------------------------------------------------


@wp.struct
class Camera:
    origin: wp.vec3
    lower_left: wp.vec3
    du: wp.vec3
    dv: wp.vec3


@wp.func
def camera_ray(cam: Camera, x: int, y: int, height: int):
    target = cam.lower_left + (float(x) + 0.5) * cam.du + (float(height - 1 - y) + 0.5) * cam.dv
    return cam.origin, wp.normalize(target - cam.origin)


@wp.kernel(module="unique")
def brep_raygen(
    scene: wp.uint64,
    brep: BrepData,
    cam: Camera,
    depth: wp.array2d(dtype=wp.float32),
    normal: wp.array2d(dtype=wp.vec3),
    face_id: wp.array2d(dtype=wp.int32),
    inside: wp.array2d(dtype=wp.int32),
):
    y, x = wp.tid()
    origin, direction = camera_ray(cam, x, y, depth.shape[0])
    payload = HitPayload()
    payload.t = -1.0
    payload.face = -1
    wp.optix_trace(scene, origin, direction, 1.0e30, payload)
    depth[y, x] = payload.t
    normal[y, x] = wp.vec3(payload.nx, payload.ny, payload.nz)
    face_id[y, x] = payload.face
    inside[y, x] = payload.inside


@wp.kernel(module="unique")
def brep_closest_hit():
    payload = HitPayload()
    wp.optix_load_payload(payload)
    if payload.t >= 0.0 and payload.t < wp.optix_get_ray_tmax():
        return  # a shell any-hit already recorded a nearer exact hit
    payload.t = wp.optix_get_ray_tmax()
    n = wp.vec3(
        wp.optix_uint32_as_float(wp.optix_get_attribute(2)),
        wp.optix_uint32_as_float(wp.optix_get_attribute(3)),
        wp.optix_uint32_as_float(wp.optix_get_attribute(4)),
    )
    n = wp.normalize(wp.optix_transform_normal_object_to_world(n))
    payload.nx = n[0]
    payload.ny = n[1]
    payload.nz = n[2]
    payload.face = wp.int32(wp.optix_get_attribute(5))
    payload.inside = wp.int32(wp.optix_get_hit_kind())
    payload.u = wp.optix_uint32_as_float(wp.optix_get_attribute(0))
    payload.v = wp.optix_uint32_as_float(wp.optix_get_attribute(1))
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def mesh_raygen(
    scene: wp.uint64,
    cam: Camera,
    depth: wp.array2d(dtype=wp.float32),
    normal: wp.array2d(dtype=wp.vec3),
    face_id: wp.array2d(dtype=wp.int32),
    inside: wp.array2d(dtype=wp.int32),
):
    y, x = wp.tid()
    origin, direction = camera_ray(cam, x, y, depth.shape[0])
    payload = HitPayload()
    payload.t = -1.0
    payload.face = -1
    wp.optix_trace(scene, origin, direction, 1.0e30, payload)
    depth[y, x] = payload.t
    normal[y, x] = wp.vec3(payload.nx, payload.ny, payload.nz)
    face_id[y, x] = payload.face
    inside[y, x] = payload.inside


@wp.kernel(module="unique")
def mesh_closest_hit():
    vertices = wp.optix_get_triangle_vertex_data()
    n = wp.normalize(wp.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
    n = wp.normalize(wp.optix_transform_normal_object_to_world(n))
    payload = HitPayload()
    payload.t = wp.optix_get_ray_tmax()
    payload.nx = n[0]
    payload.ny = n[1]
    payload.nz = n[2]
    payload.face = wp.int32(wp.optix_get_primitive_index())
    payload.inside = wp.int32(0)
    if wp.optix_is_triangle_back_face_hit():
        payload.inside = wp.int32(1)
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def miss():
    # Raygen initializes the payload to "no hit"; shell any-hits may have recorded a hit.
    pass


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------


def _array(values, dtype, device):
    return wp.array(np.ascontiguousarray(values), dtype=dtype, device=device)


class BrepScene:
    """GPU scene: one custom-primitive GAS per BrepArray prim, instanced with its USD transform."""

    def __init__(self, prims: list[BrepPrim], device=None, proxy: bool = False):
        """``proxy``: trace bicubic NURBS patches through triangle shells (hardware traversal)."""
        self.device = wp.get_device(device)
        dev = self.device
        # Placements of one prototype share its arrays: pack and build each prototype once.
        unique: dict[int, int] = {}
        for prim in prims:
            unique.setdefault(id(prim.face_kind), len(unique))
        prototypes = list({id(p.face_kind): p for p in prims}.values())
        proxies = [brep_proxy.build_proxy(p) for p in prototypes] if proxy else None
        if proxy:
            self.data, bases, tri_bases = bd.pack(prototypes, dev, proxies)
        else:
            self.data, bases = bd.pack(prototypes, dev)
            tri_bases = []
        hit_groups = {
            name: wo.HitGroup(closest_hit=brep_closest_hit, intersection=fn)
            for name, fn in zip(KIND_NAMES, INTERSECTORS)
        }
        hit_groups["proxy"] = wo.HitGroup(closest_hit=proxy_closest_hit, any_hit=proxy_any_hit)
        sbt = wo.SbtLayout(miss=miss, hit_groups=hit_groups)
        proxy_offset = sbt.get_sbt_offset("proxy")
        self.gases, self.buffers = [], []
        gas_records = []  # per prototype: (aabb gas, prim base, triangle gas or None, triangle base)
        for n, prototype in enumerate(prototypes):
            bounds = prototype.prim_bounds.astype(np.float32).reshape(-1, 6).copy()
            if proxy:
                bounds[proxies[n].patches] = np.nan  # inactive: these patches use their shells
            gas_bounds = _array(bounds, wp.float32, dev)
            kinds = _array(prototype.prim_kind.astype(np.uint32), wp.uint32, dev)
            flags = [wo.GeometryFlags.DISABLE_ANYHIT] * len(KIND_NAMES)
            aabb = wo.Accel([wo.aabbs(bounds=gas_bounds, sbt_indices=kinds, flags=flags)], compact=True, device=dev)
            self.gases.append(aabb)
            tris = None
            if proxy and len(proxies[n].indices):
                pts = _array(proxies[n].points, wp.vec3, dev)
                idx = _array(proxies[n].indices.ravel(), wp.uint32, dev)
                self.buffers += [pts, idx]
                tris = wo.Accel([wo.triangles(pts, idx, flags=wo.GeometryFlags.NONE)], compact=True, device=dev)
                self.gases.append(tris)
            gas_records.append((aabb, bases[n], tris, tri_bases[n] if proxy else 0))
        records = []  # (world, gas, instance id base, sbt offset)
        for p in prims:
            aabb, base, tris, tri_base = gas_records[unique[id(p.face_kind)]]
            records.append((p.world, aabb, base, 0))
            if tris is not None:
                records.append((p.world, tris, tri_base, proxy_offset))
        self.instances = wp.empty(len(records), dtype=wo.Instance, device=dev)
        mats = _array(np.array([r[0] for r in records], dtype=np.float32), wp.mat44, dev)
        ids = _array(np.array([r[1].id for r in records], dtype=np.uint64), wp.uint64, dev)
        bases = _array(np.array([r[2] for r in records], dtype=np.int32), wp.int32, dev)
        offsets = _array(np.array([r[3] for r in records], dtype=np.int32), wp.int32, dev)
        wp.launch(_write_instances, dim=len(records), inputs=[mats, ids, bases, offsets, self.instances], device=dev)
        self.ias = wo.Accel([wo.instances(self.instances, self.gases)], device=dev)
        self.pipeline = wo.Pipeline(
            brep_raygen,
            sbt=sbt,
            payload_words=8,
            attribute_words=bd.ATTRIBUTE_WORDS,
            graph_flags=wo.TraversableGraphFlags.ALLOW_SINGLE_LEVEL_INSTANCING,
            device=dev,
        )

    def accel_bytes(self) -> int:
        return sum(g._buffer.size for g in self.gases) + self.ias._buffer.size

    def data_bytes(self) -> int:
        return sum(getattr(self.data, k).capacity for k in BrepData.vars)

    def render(self, cam: Camera, outputs: list) -> None:
        wo.launch(self.pipeline, dim=outputs[0].shape, inputs=[self.ias, self.data, cam, *outputs], device=self.device)


@wp.kernel
def _write_instances(
    mats: wp.array(dtype=wp.mat44),
    ids: wp.array(dtype=wp.uint64),
    bases: wp.array(dtype=wp.int32),
    sbt_offsets: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wo.Instance),
):
    i = wp.tid()
    out[i] = wo.make_instance(mats[i], ids[i], bases[i], sbt_offsets[i])


class MeshScene:
    """Reference: the same geometry as triangles, traced with hardware triangle intersection."""

    def __init__(self, points: np.ndarray, indices: np.ndarray, device=None):
        self.device = wp.get_device(device)
        self.points = _array(points.astype(np.float32), wp.vec3, self.device)
        self.indices = _array(indices.astype(np.uint32).ravel(), wp.uint32, self.device)
        self.gas = wo.Accel([wo.triangles(self.points, self.indices)], compact=True, device=self.device)
        self.pipeline = wo.Pipeline(
            mesh_raygen,
            sbt=wo.SbtLayout(miss=miss, hit_groups={"mesh": wo.HitGroup(closest_hit=mesh_closest_hit)}),
            payload_words=8,
            graph_flags=wo.TraversableGraphFlags.ALLOW_SINGLE_GAS,
            max_traversable_depth=1,
            device=self.device,
        )

    def accel_bytes(self) -> int:
        return self.gas._buffer.size

    def data_bytes(self) -> int:
        return self.points.capacity + self.indices.capacity

    def render(self, cam: Camera, outputs: list) -> None:
        wo.launch(self.pipeline, dim=outputs[0].shape, inputs=[self.gas, cam, *outputs], device=self.device)


class InstancedMeshScene:
    """Reference for instanced stages: one triangle GAS per prototype mesh, placed by an IAS."""

    def __init__(self, placements, device=None):
        """``placements``: (points, indices, world 4x4); placements sharing ``points`` share a GAS."""
        self.device = wp.get_device(device)
        dev = self.device
        self.gases, self.buffers, index = [], [], {}
        for points, indices, _ in placements:
            if id(points) not in index:
                index[id(points)] = len(self.gases)
                pts = _array(np.asarray(points, dtype=np.float32), wp.vec3, dev)
                idx = _array(np.asarray(indices, dtype=np.uint32).ravel(), wp.uint32, dev)
                self.buffers += [pts, idx]
                self.gases.append(wo.Accel([wo.triangles(pts, idx)], compact=True, device=dev))
        self.triangles = sum(len(i) for p, i, _ in placements)
        records = wp.empty(len(placements), dtype=wo.Instance, device=dev)
        mats = _array(np.array([w for _, _, w in placements], dtype=np.float32), wp.mat44, dev)
        ids = _array(np.array([self.gases[index[id(p)]].id for p, _, _ in placements], dtype=np.uint64), wp.uint64, dev)
        zeros = _array(np.zeros(len(placements), dtype=np.int32), wp.int32, dev)
        wp.launch(_write_instances, dim=len(placements), inputs=[mats, ids, zeros, zeros, records], device=dev)
        self.instances = records
        self.ias = wo.Accel([wo.instances(records, self.gases)], device=dev)
        self.pipeline = wo.Pipeline(
            mesh_raygen,
            sbt=wo.SbtLayout(miss=miss, hit_groups={"mesh": wo.HitGroup(closest_hit=mesh_closest_hit)}),
            payload_words=8,
            graph_flags=wo.TraversableGraphFlags.ALLOW_SINGLE_LEVEL_INSTANCING,
            device=dev,
        )

    def accel_bytes(self) -> int:
        return sum(g._buffer.size for g in self.gases) + self.ias._buffer.size

    def data_bytes(self) -> int:
        return sum(b.capacity for b in self.buffers)

    def render(self, cam: Camera, outputs: list) -> None:
        wo.launch(self.pipeline, dim=outputs[0].shape, inputs=[self.ias, cam, *outputs], device=self.device)


def orbit_camera(target, distance, azimuth, elevation, width, height, up="Z", fov=35.0):
    """Pinhole camera orbiting ``target``; returns the camera and its unit forward vector."""
    az, el = np.radians(azimuth), np.radians(elevation)
    if up == "Z":
        view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        world_up = np.array([0.0, 0.0, 1.0])
    else:
        view = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
        world_up = np.array([0.0, 1.0, 0.0])
    eye = np.asarray(target, dtype=np.float64) + view * distance
    forward = -view
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    upv = np.cross(right, forward)
    half_h = np.tan(np.radians(fov) * 0.5)
    half_w = half_h * width / height
    cam = Camera()
    cam.origin = wp.vec3(*eye)
    cam.lower_left = wp.vec3(*(eye + forward - half_w * right - half_h * upv))
    cam.du = wp.vec3(*(2.0 * half_w * right / width))
    cam.dv = wp.vec3(*(2.0 * half_h * upv / height))
    return cam, forward, right, upv


def frame_distance(lo: np.ndarray, hi: np.ndarray, fov=35.0) -> float:
    """Orbit distance at which the bounding sphere of [lo, hi] fills the view."""
    return 0.5 * float(np.linalg.norm(hi - lo)) / np.tan(np.radians(fov) * 0.5) * 1.05


@wp.func
def shade_pixel(depth: float, n_in: wp.vec3, face: int, inside: int, view: wp.vec3, color_faces: int, row: float):
    c = wp.vec3(0.93, 0.94, 0.96) - 0.12 * row * wp.vec3(1.0, 1.0, 0.6)
    if depth >= 0.0:
        n = n_in
        if wp.dot(n, view) > 0.0:
            n = -n
        light = wp.normalize(-view + wp.vec3(0.3, 0.2, 0.6))
        lambert = 0.25 + 0.75 * wp.max(wp.dot(n, light), 0.0)
        base = wp.vec3(0.72, 0.74, 0.78)
        if color_faces != 0:
            h = wp.uint32(face) * wp.uint32(2654435761)
            base = wp.vec3(
                0.45 + 0.5 * float((h >> wp.uint32(8)) & wp.uint32(255)) / 255.0,
                0.45 + 0.5 * float((h >> wp.uint32(16)) & wp.uint32(255)) / 255.0,
                0.45 + 0.5 * float((h >> wp.uint32(24)) & wp.uint32(255)) / 255.0,
            )
        if inside != 0:
            base = wp.vec3(1.0, 0.0, 0.8)  # we see the inside of the solid: a trim leak
        c = base * lambert
    return c


@wp.kernel
def shade_packed(
    depth: wp.array2d(dtype=wp.float32),
    normal: wp.array2d(dtype=wp.vec3),
    face_id: wp.array2d(dtype=wp.int32),
    inside: wp.array2d(dtype=wp.int32),
    view: wp.vec3,
    color_faces: int,
    image: wp.array(dtype=wp.uint32),
):
    """Shade into a bottom-up RGBA8 buffer (OpenGL texture layout, as GLInteropViewer maps it)."""
    y, x = wp.tid()
    h = depth.shape[0]
    w = depth.shape[1]
    c = shade_pixel(depth[y, x], normal[y, x], face_id[y, x], inside[y, x], view, color_faces, float(y) / float(h))
    r = wp.uint32(wp.clamp(c[0] * 255.0, 0.0, 255.0))
    g = wp.uint32(wp.clamp(c[1] * 255.0, 0.0, 255.0))
    b = wp.uint32(wp.clamp(c[2] * 255.0, 0.0, 255.0))
    image[(h - 1 - y) * w + x] = wp.uint32(255) << wp.uint32(24) | b << wp.uint32(16) | g << wp.uint32(8) | r


@wp.kernel
def shade(
    depth: wp.array2d(dtype=wp.float32),
    normal: wp.array2d(dtype=wp.vec3),
    face_id: wp.array2d(dtype=wp.int32),
    inside: wp.array2d(dtype=wp.int32),
    view: wp.vec3,
    color_faces: int,
    image: wp.array3d(dtype=wp.uint8),
):
    y, x = wp.tid()
    c = shade_pixel(
        depth[y, x], normal[y, x], face_id[y, x], inside[y, x], view, color_faces, float(y) / float(depth.shape[0])
    )
    for k in range(3):
        image[y, x, k] = wp.uint8(wp.clamp(c[k] * 255.0, 0.0, 255.0))
