# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opacity micromaps: per-micro-triangle opacity states for triangle geometry.

A micromap subdivides a triangle into 4^level micro-triangles, each transparent (0), opaque (1),
unknown-transparent (2) or unknown-opaque (3). The RT cores skip transparent and accept opaque
micro-triangles without running any-hit programs; unknown ones run the any-hit program (unless
the ray forces two states). Pipelines tracing them need ``Pipeline(opacity_micromaps=True)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp

from warp_optix import native
from warp_optix._src.context import resolve

TRANSPARENT = 0
OPAQUE = 1
UNKNOWN_TRANSPARENT = 2
UNKNOWN_OPAQUE = 3

# Predefined micromap indices: a whole triangle in one state, without a micromap.
INDEX_FULLY_TRANSPARENT = -1
INDEX_FULLY_OPAQUE = -2
INDEX_FULLY_UNKNOWN_TRANSPARENT = -3
INDEX_FULLY_UNKNOWN_OPAQUE = -4

MAX_LEVEL = 12
_DESC = np.dtype([("byte_offset", np.uint32), ("level", np.uint16), ("format", np.uint16)])


def _extract_even_bits(x: np.ndarray) -> np.ndarray:
    x = x & 0x55555555
    x = (x | (x >> 1)) & 0x33333333
    x = (x | (x >> 2)) & 0x0F0F0F0F
    x = (x | (x >> 4)) & 0x00FF00FF
    x = (x | (x >> 8)) & 0x0000FFFF
    return x


def _prefix_eor(x: np.ndarray) -> np.ndarray:
    for shift in (1, 2, 4, 8):
        x = x ^ (x >> shift)
    return x


def micro_triangle_barycentrics(level: int) -> np.ndarray:
    """(4^level, 3, 2) base-triangle barycentrics (b1, b2) of each micro-triangle's corners, in
    OptiX's micro-triangle order (a port of ``optixMicromapIndexToBaseBarycentrics``)."""
    if level == 0:
        return np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    index = np.arange(4**level, dtype=np.uint64)
    b0 = _extract_even_bits(index)
    b1 = _extract_even_bits(index >> np.uint64(1))
    fx = _prefix_eor(b0)
    fy = _prefix_eor(b0 & ~b1)
    t = fy ^ b1
    iu = (fx & ~t) | (b0 & ~t) | (~b0 & ~fx & t)
    iv = fy ^ b0
    iw = (~fx & ~t) | (b0 & ~t) | (~b0 & fx & t)
    mask = np.uint64((1 << level) - 1)
    iu, iv, iw = iu & mask, iv & mask, iw & mask
    y_flipped = ((iu & np.uint64(1)) ^ (iv & np.uint64(1)) ^ (iw & np.uint64(1)) ^ np.uint64(1)).astype(np.float64)
    pattern = np.uint64(0x8888888888888888 ^ 0xF000F000F000F000 ^ 0xFFFF000000000000)
    x_flipped = (pattern >> (index & np.uint64(63))) & np.uint64(1)
    x_flipped ^= (pattern >> ((index >> np.uint64(6)) & np.uint64(63))) & np.uint64(1)
    x_flipped = x_flipped.astype(np.float64)
    scale = 1.0 / (1 << level)
    u, v = iu.astype(np.float64) * scale, iv.astype(np.float64) * scale
    bary0 = np.stack([u + x_flipped * scale, v + y_flipped * scale], -1)
    bary1 = np.stack([u + (1 - x_flipped) * scale, v + y_flipped * scale], -1)
    bary2 = np.stack([u + y_flipped * scale, v + (1 - y_flipped) * scale], -1)
    return np.stack([bary0, bary1, bary2], 1)


def _level_of(count: int) -> int:
    level = 0
    while 4**level < count:
        level += 1
    if 4**level != count or level > MAX_LEVEL:
        raise ValueError(f"a micromap needs 4^level states (level <= {MAX_LEVEL}), got {count}")
    return level


class OpacityMicromaps:
    """A built opacity micromap array (four-state).

    ``states``: one uint8 array per micromap, 4^level states in OptiX's micro-triangle order (see
    :func:`micro_triangle_barycentrics`). Attach it to triangles with
    ``triangles(..., opacity_micromaps=maps, micromap_indices=...)``.
    """

    def __init__(
        self,
        states: Sequence[np.ndarray],
        *,
        device: wp.DeviceLike = None,
        stream: wp.Stream | None = None,
    ):
        context, stream = resolve(device, stream)
        fmt = native.OPACITY_MICROMAP_FORMAT_4_STATE
        levels = np.array([_level_of(len(s)) for s in states], dtype=np.int64)
        chunks, offsets, offset = [], [], 0
        for s in states:
            s = np.asarray(s, dtype=np.uint8) & 3
            if len(s) < 4:
                s = np.concatenate([s, np.zeros(4 - len(s), np.uint8)])
            packed = (s[0::4] | (s[1::4] << 2) | (s[2::4] << 4) | (s[3::4] << 6)).astype(np.uint8)
            chunks.append(packed)
            offsets.append(offset)
            offset += len(packed)
        data = np.concatenate(chunks) if chunks else np.zeros(1, np.uint8)
        descs = np.zeros(len(states), _DESC)
        descs["byte_offset"] = offsets
        descs["level"] = levels
        descs["format"] = fmt
        self.levels = levels
        self.histogram = [(int(n), int(level), fmt) for level, n in zip(*np.unique(levels, return_counts=True))]
        dev = context.device
        with wp.ScopedDevice(dev):
            data_buffer = wp.array(data, dtype=wp.uint8, device=dev)
            desc_buffer = wp.array(descs.view(np.uint8), dtype=wp.uint8, device=dev)
            flags = native.OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE
            output_size, temp_size = native.opacityMicromapArrayComputeMemoryUsage(
                context.native, flags, data_buffer.ptr, desc_buffer.ptr, _DESC.itemsize, self.histogram
            )
            self.buffer = wp.empty(output_size, dtype=wp.uint8, device=dev)
            temp = wp.empty(max(temp_size, 1), dtype=wp.uint8, device=dev)
            native.opacityMicromapArrayBuild(
                context.native,
                stream.cuda_stream,
                flags,
                data_buffer.ptr,
                desc_buffer.ptr,
                _DESC.itemsize,
                self.histogram,
                self.buffer.ptr,
                output_size,
                temp.ptr,
                temp_size,
            )
        # The inputs and scratch are freed in stream order, after the build reads them.
        self.device = dev

    def __len__(self) -> int:
        return len(self.levels)

    def usage(self, indices: np.ndarray | None) -> list[tuple[int, int, int]]:
        """Usage counts (count, level, format) for triangles referencing micromaps by ``indices``
        (None: triangle k uses micromap k)."""
        used = self.levels if indices is None else self.levels[indices[indices >= 0]]
        fmt = native.OPACITY_MICROMAP_FORMAT_4_STATE
        return [(int(n), int(level), fmt) for level, n in zip(*np.unique(used, return_counts=True))]
