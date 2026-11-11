// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Opacity micromaps. Included by extension.cpp: reuses its checked PyOptiX bridge and OptiX table.
// PyOptiX 9.1.1 binds neither the micromap array build nor the triangle input's opacityMicromap
// field, so the build is bound here and the field is written through the mirrored triangle input.

#include <unordered_map>
#include <vector>

namespace warp_optix_ext {

inline std::vector<OptixOpacityMicromapHistogramEntry> histogram_entries(const py::list& entries) {
    // (count, subdivision level, format) per entry
    std::vector<OptixOpacityMicromapHistogramEntry> result;
    for (const auto& item : entries) {
        auto entry = item.cast<py::tuple>();
        result.push_back({entry[0].cast<unsigned int>(), entry[1].cast<unsigned int>(),
                          static_cast<OptixOpacityMicromapFormat>(entry[2].cast<unsigned int>())});
    }
    return result;
}

inline OptixOpacityMicromapArrayBuildInput
micromap_build_input(unsigned int flags, uintptr_t inputBuffer, uintptr_t perMicromapDescBuffer,
                     unsigned int perMicromapDescStrideInBytes,
                     const std::vector<OptixOpacityMicromapHistogramEntry>& histogram) {
    OptixOpacityMicromapArrayBuildInput input{};
    input.flags = flags;
    input.inputBuffer = static_cast<CUdeviceptr>(inputBuffer);
    input.perMicromapDescBuffer = static_cast<CUdeviceptr>(perMicromapDescBuffer);
    input.perMicromapDescStrideInBytes = perMicromapDescStrideInBytes;
    input.numMicromapHistogramEntries = static_cast<unsigned int>(histogram.size());
    input.micromapHistogramEntries = histogram.empty() ? nullptr : histogram.data();
    return input;
}

// The usage counts a triangle input points to must outlive its accel builds; keep them per input.
inline std::unordered_map<const void*, std::vector<OptixOpacityMicromapUsageCount>>& usage_counts() {
    static std::unordered_map<const void*, std::vector<OptixOpacityMicromapUsageCount>> counts;
    return counts;
}

} // namespace warp_optix_ext

inline void bind_micromaps(py::module_& m) {
    using namespace warp_optix_ext;
    m.def("setAllowOpacityMicromaps", [](pyoptix::PipelineCompileOptions& options, bool enabled) {
        options.options.allowOpacityMicromaps = enabled;
    });
    m.def(
        "opacityMicromapArrayComputeMemoryUsage",
        [](const pyoptix::DeviceContext& context, unsigned int flags, uintptr_t inputBuffer,
           uintptr_t perMicromapDescBuffer, unsigned int perMicromapDescStrideInBytes, const py::list& histogram) {
            ensure_optix_initialized();
            auto entries = histogram_entries(histogram);
            auto input =
                micromap_build_input(flags, inputBuffer, perMicromapDescBuffer, perMicromapDescStrideInBytes, entries);
            OptixMicromapBufferSizes sizes{};
            PYOPTIX_CHECK(optixOpacityMicromapArrayComputeMemoryUsage(context.deviceContext, &input, &sizes));
            return py::make_tuple(sizes.outputSizeInBytes, sizes.tempSizeInBytes);
        },
        py::arg("context"), py::arg("flags"), py::arg("inputBuffer"), py::arg("perMicromapDescBuffer"),
        py::arg("perMicromapDescStrideInBytes"), py::arg("histogram"));
    m.def(
        "opacityMicromapArrayBuild",
        [](const pyoptix::DeviceContext& context, uintptr_t stream, unsigned int flags, uintptr_t inputBuffer,
           uintptr_t perMicromapDescBuffer, unsigned int perMicromapDescStrideInBytes, const py::list& histogram,
           uintptr_t output, size_t outputSizeInBytes, uintptr_t temp, size_t tempSizeInBytes) {
            ensure_optix_initialized();
            auto entries = histogram_entries(histogram);
            auto input =
                micromap_build_input(flags, inputBuffer, perMicromapDescBuffer, perMicromapDescStrideInBytes, entries);
            OptixMicromapBuffers buffers{static_cast<CUdeviceptr>(output), outputSizeInBytes,
                                         static_cast<CUdeviceptr>(temp), tempSizeInBytes};
            PYOPTIX_CHECK(optixOpacityMicromapArrayBuild(context.deviceContext, reinterpret_cast<CUstream>(stream),
                                                         &input, &buffers));
        },
        py::arg("context"), py::arg("stream"), py::arg("flags"), py::arg("inputBuffer"),
        py::arg("perMicromapDescBuffer"), py::arg("perMicromapDescStrideInBytes"), py::arg("histogram"),
        py::arg("output"), py::arg("outputSizeInBytes"), py::arg("temp"), py::arg("tempSizeInBytes"));
    m.def(
        "setTriangleOpacityMicromap",
        [](pyoptix::BuildInputTriangleArray& triangles, unsigned int indexingMode, uintptr_t opacityMicromapArray,
           uintptr_t indexBuffer, unsigned int indexSizeInBytes, unsigned int indexStrideInBytes,
           unsigned int indexOffset, const py::list& usage) {
            auto& counts = usage_counts()[&triangles];
            counts.clear();
            for (const auto& entry : histogram_entries(usage))
                counts.push_back({entry.count, entry.subdivisionLevel, entry.format});
            OptixBuildInputOpacityMicromap& omm = triangles.build_input.opacityMicromap;
            omm.indexingMode = static_cast<OptixOpacityMicromapArrayIndexingMode>(indexingMode);
            omm.opacityMicromapArray = static_cast<CUdeviceptr>(opacityMicromapArray);
            omm.indexBuffer = static_cast<CUdeviceptr>(indexBuffer);
            omm.indexSizeInBytes = indexSizeInBytes;
            omm.indexStrideInBytes = indexStrideInBytes;
            omm.indexOffset = indexOffset;
            omm.numMicromapUsageCounts = static_cast<unsigned int>(counts.size());
            omm.micromapUsageCounts = counts.empty() ? nullptr : counts.data();
        },
        py::arg("triangles"), py::arg("indexingMode"), py::arg("opacityMicromapArray"), py::arg("indexBuffer"),
        py::arg("indexSizeInBytes"), py::arg("indexStrideInBytes"), py::arg("indexOffset"), py::arg("usage"));
    m.attr("OPACITY_MICROMAP_FORMAT_2_STATE") =
        py::int_(static_cast<unsigned int>(OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE));
    m.attr("OPACITY_MICROMAP_FORMAT_4_STATE") =
        py::int_(static_cast<unsigned int>(OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE));
    m.attr("OPACITY_MICROMAP_ARRAY_INDEXING_MODE_LINEAR") =
        py::int_(static_cast<unsigned int>(OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_LINEAR));
    m.attr("OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED") =
        py::int_(static_cast<unsigned int>(OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED));
    m.attr("OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE") =
        py::int_(static_cast<unsigned int>(OPTIX_OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE));
}
