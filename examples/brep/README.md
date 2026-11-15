# Prototype: ray tracing USD BReps directly

Feasibility prototype for tracing the exact surfaces of OpenUSD BRep prims
([`BrepArray`, omniSolid schema](https://nvidia-omniverse.github.io/usd-brep/latest/)) with
warp-optix custom primitives, without tessellating.

```bash
# render (reads the schema with pxr only; no modeling kernel needed)
uv run --extra usd --extra pathtracing python examples/brep/example_brep.py part.usda --up Z

# interactive viewer: left-drag orbit, right/middle-drag pan, wheel zoom,
# F reframe, C toggle face colors, P screenshot
uv run --extra usd --extra pathtracing --extra ui python examples/brep/example_brep.py part.usda --up Z --live

# compare with SMLib tessellations: trace time, AS size, depth/normal/coverage error
uv run --extra usd --extra pathtracing --with usd-brep python examples/brep/example_brep.py part.usda --compare

# check the loader's surfaces and UV trims against the SMLib kernel
uv run --extra usd --with usd-brep python examples/brep/validate_trims.py part.usda
```

Test files: `usd-brep` (`pip install usd-brep`, CPython 3.12) can build solids
(`usd_brep.primitives`, `booleans`, `fillets`) and convert its `.smb` test models
(`usd_brep.Brep.read_from_file(...)`, then `usd_brep.usd.export_brep`).

## Path tracing

`BrepArray` prims also load into the full path tracer (`warp_optix.pathtracing`), next to meshes
and curves, with their USD materials (`UsdPreviewSurface`, MDL such as `OmniGlass`, per-face
`materialBind` GeomSubsets), transforms and lights:

```bash
uv run --extra usd --extra pathtracing --extra ui python examples/example_pathtracing.py --usd scene.usda
```

Headless: `api.scene.load_from_usd(path)`, or `api.scene.add_brep(Brep(prim))` with
`warp_optix._src.pathtracing.scene.Brep` and a prim from `warp_optix._src.brep.loader`.

- Two hit groups in the path tracer's SBT, chosen per primitive by the GAS's SBT indices:
  `"brep"` (planes, cylinders, cones, spheres) and `"brep_curved"` (tori, NURBS sub-patches). OptiX
  saves the registers an intersection program uses on every call, so the common quadric program
  stays small (5% faster frames than one program for all kinds). Both share a closest-hit that
  fills the same geometry-neutral payload as triangles (face-forwarded world normal, face UVs,
  per-face material, front face from the outward normal, so glass tracks entering/leaving the
  medium). Any-hit uses the material's opacity.
- BRep GASes offset their primitive indices by their first global BRep primitive id, so
  intersection programs read it straight from `optix_get_primitive_index()`; closest-hit and
  any-hit subtract it again for per-primitive materials. A BRep geometry's `indices` hold its
  global primitive ids, so the existing render primitive and material plumbing maps a hit to its
  BRep data (`PathtraceLaunchParams.brep`). USD instance proxies of one prototype share one GAS.
- `BrepArray`'s own xformOps are evaluated even when the omniSolid schema plugin is not
  registered (USD then does not know the prim is a `Gprim`).
- Tests: `warp_optix/tests/test_brep.py` (hand-authored BReps, no `usd-brep` needed).

## How it works

| Part | File | Approach |
| --- | --- | --- |
| Schema reader | `warp_optix/_src/brep/loader.py` | pxr + NumPy; topology (regions → shells → faceuses, faces → loops → edgeuses), analytic surfaces, NURBS surfaces, 3D edge curves and optional UV pcurves |
| Analytic faces | `warp_optix/_src/brep/device.py` | one custom primitive per face; AABB from a sampled UV grid padded by the sagitta bound; planes, cylinders, cones and spheres as one closed-form quadric in the face frame (one code path, so rays of a warp hitting mixed kinds share the solve and trim test: 6–10% faster path tracing on a dense assembly); torus by a derivative-cascade quartic solver (bounded cost, no grazing blowup) |
| NURBS faces | both | knot insertion → rational Bézier patches → de Casteljau splits until flat (`flat_tol`); convex-hull AABBs; Newton on the two-plane ray form, seeded from the nearest control point; specialized bicubic and degree-(3,4) evaluators, generic evaluation through degree 7 |
| Trimming | both | face loops as UV polylines (pcurves when present, otherwise 3D edges projected onto the surface); even–odd crossing test, segments bucketed into slabs; periodic copies for seams |
| Surface records | `warp_optix/_src/brep/device.py` | per primitive, its face's frame, parameters, ranges, flags and kind, and trim lookup in one 128-byte record: one cache line and one dependent load per intersection (path tracing on a dense assembly is 10-13% faster than with per-field arrays, and 3% faster than per-face records reached through the primitive) |
| Trim cell grids | `warp_optix/_src/brep/trim_grid.py` | per face, a uniform cell grid: cells away from the trims store inside/outside, the others their chords and a reference parity; built for all faces at once with Warp kernels (radix sort, scans) |
| Shells | `warp_optix/_src/brep/proxy.py` | bicubic sub-patches enclosed by triangle shells (one Warp kernel builds all of them); hardware triangle traversal finds candidates, an any-hit refines them with Newton seeded from the triangle hit (`--proxy`, and the path tracer's default) |
| Dispatch (this example) | `brep_optix.py` | one hit group per surface kind, selected per primitive with `wo.aabbs(sbt_indices=)`; one GAS per `BrepArray`, instanced with the USD transform (non-uniform scale works) |

Attributes reported: `u, v`, outward object-space normal, face index; hit kind 1 marks a ray
leaving a solid (used to count cracks: "inside-hits").

Torus intersection and edge projection handle both sheets of spindle tori (`minorRadius >
majorRadius`). The negative signed-radius sheet needs its azimuth shifted by half a turn and
its parameter-space normal reversed; otherwise valid trimmed faces, such as the turbofan casing,
are rejected.

Nonperiodic trimmed NURBS faces omit sub-patches proven wholly outside their trims. The check
includes the solver's parameter tolerance, trim bulges, and boundaries omitted by slab packing;
boundary patches and periodic faces retain the usual device trim test. Loader statistics report
`culled_nurbs_subpatches`.

Shell any-hits keep the nearest refined surface hit in the payload. They accept a triangle only
when its distance is at least the refined hit distance: accepting an earlier triangle could hide
a nearer surface between that triangle and its refinement. Surface orientation alone is not a
safe pruning criterion for overlapping shells. This extra traversal has a cost, and the fixed
192 triangles per bicubic patch can substantially increase acceleration-structure memory; compare
shells with AABBs for the intended scene and ray distribution.

## Results (RTX PRO 6000 Blackwell, primary rays, `--repeats 100`)

Mesh = SMLib tessellation traced as triangles, same camera. `1e-4` is the chord tolerance
relative to each part's diagonal, the mesh that matches the BRep to < 0.01 % coverage.

| Scene | Faces (NURBS) | BRep ms | Mesh ms (1e-4) | Ratio | BRep AS | Mesh AS (1e-4) | Triangles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| part (analytic: plane/cyl/cone/sphere/torus), 1280×960 | 32 (0) | 0.092 | 0.050 | 1.8× | 3.8 KiB | 508 KiB | 22.9 k |
| Mouse, 1280×960 | 4 (3) | 0.474 | 0.044 | 10.8× | 9.6 KiB | 149 KiB | 6.3 k |
| bottle, 1280×960 | 16 (15) | 0.491 | 0.045 | 10.9× | 14.5 KiB | 205 KiB | 8.6 k |
| BoltMBox1, 1280×960 | 31 (6) | 0.472 | 0.053 | 8.9× | 39 KiB | 840 KiB | 36 k |
| 36 × BoltMBox1, 1920×1080 | 1116 | 0.683 | 0.077 | 8.9× | 1.3 MiB | 27.3 MiB | 1.30 M |
| 36 × Mouse, 1920×1080 | 144 | 0.672 | 0.066 | 10.2× | 287 KiB | 4.9 MiB | 228 k |

The mesh at 1e-4 still has mean normal errors of 0.4–0.6° (p99 1–3°) relative to the exact
surface; the coarser 2e-3 mesh is 1.5–2.6° mean / 5–11° p99.

## Findings

- **Feasible, and exact.** All 15 test models render correctly; loader trims agree with the
  SMLib kernel's point/face classification at 150 random UV samples per face (a handful of
  disagreements within the polyline chord error of circular trims).
- **Analytic CAD faces are cheap: ~2× a mesh.** Closed-form intersection plus a UV trim test
  costs little next to traversal.
- **NURBS faces cost ~9–11× a mesh.** Each NURBS hit pixel invokes 1.3–4.8 overlapping
  sub-patch intersection programs (~2–3 Newton steps each). The arithmetic is cheap; the cost is
  the number of custom-primitive invocations (an RT core → SM round trip each), which triangles
  avoid. Smaller sub-patches do not help (more primitives, same overlap).
- **Memory is the win:** the acceleration structure is 15–50× smaller than an equivalent
  mesh, and nothing has to be re-tessellated for a closer camera.
- **pcurves are optional in the schema and often missing** (e.g. 24 of 140 edgeuses in BoltMBox1;
  none for booleans of analytic solids), so a renderer must project 3D edges into UV, including
  seam placement on periodic surfaces and poles/apexes.
- **Watertightness:** adjacent faces' trim polylines do not coincide exactly, leaving rare
  hairline cracks along shared edges (0.004–0.05 % of pixels counted as inside-hits).
- **float32 matters:** Newton must run with the ray origin re-centered at the patch; with a
  distant camera the residual otherwise cancels to the tolerance (random speckle).

## Limitations and next steps

- Exact trims: Bézier clipping against the pcurves (or a tolerance band shared by adjacent faces)
  instead of polylines, to remove cracks.
- Share one GAS between references of the same `BrepArray` (currently duplicated per prim).
- Not handled: wire edges, point shells, NURBS degree > 7, non-positive weights, time samples.
- The torus quartic and the NURBS Newton run in float32; very large or far-from-origin parts would
  want local frames per face.

## NURBS work diagnostics

Run the path tracer's camera, bounce and shadow rays with per-patch counters:

```bash
uv run --extra usd --extra pathtracing python examples/brep/profile_brep.py part.usda
# Optional: compare all raw color/guide outputs with an uninstrumented fresh process.
uv run --extra usd --extra pathtracing python examples/brep/profile_brep.py part.usda --output counted.npz
uv run --extra usd --extra pathtracing python examples/brep/profile_brep.py part.usda --no-counters --output reference.npz
```

The script enables `WARP_OPTIX_BREP_DIAGNOSTICS=1` before importing the renderer. For other
applications, set that variable before process startup; changing it after imports does not
change compiled kernels. Read `scene._brep_data.diagnostic_counts` after synchronizing rendering;
its rows are global primitive IDs and columns follow `device.DIAGNOSTIC_NAMES`. Zero the counters
on the rendering stream (or synchronize around the reset) to delimit a sample interval.

Counters distinguish AABB candidates and hull rejects, shell candidates, Newton calls and
iterations, failed Newton solves, fallback entries, and root-search calls. Newton failures count
nonconvergence, not converged roots outside the patch or rejected by trims. Fallback entries
include grazing cases resolved by the shell orientation shortcut, so not every fallback launches
a root search. Repeated shell refinement contributes repeated calls; these are work counts, not
unique ray/patch counts. The script reports a single frame after warmup, with a fixed sample index.

Diagnostic atomics perturb execution: never use instrumented timings to judge speedups. Normal
builds compile out the updates and allocate no counter storage. The renderer groups bounce rays
by direction octant and writes primary guide buffers before the bounce loop to reduce state
carried across reordering. Benchmark hint and live-state changes on the intended scene; small
improvements can depend on compiler layout and ray distribution.

Newton diagnostics also include the 1–10 iteration histogram, singular/domain exits, failures
on the last iteration, and separate nonconverged/grazing fallback entries. Last-iteration failures
can overlap domain or singular exits; they do not necessarily mean the loop exhausted its budget.
`projection_calls` and `projection_evaluations` measure the Gauss-Newton work inside the fallback,
which is separate from the initial ray/surface Newton iterations. A projection performs up to
three surface evaluations (two updates and one final evaluation). These counters help distinguish
rare long Newton solves from repeated work in grazing-ray root searches.

The projection now stops when clamping or rounding leaves both parameters exactly unchanged:
the next evaluation would reproduce the point and derivatives already available. The
`projection_stalled` counter records these exits. This does not relax the residual tolerance or
reduce the ten-iteration Newton budget or two-update projection budget.

On the RTX PRO 6000 Blackwell (1280×960, four-bounce setting, default shells, denoising off),
this shortcut measured 0.465 ms versus 0.478–0.479 ms for Mouse's OptiX launch; NURBS parts and bottle
were effectively unchanged. All seven raw output buffers matched exactly on those three scenes.
Measurements excluded compilation, used ten warmup frames and five batches of 50 timed launches,
and checked baselines again after candidates. These small gains remain workload dependent.

A six-iteration Newton cap and a single-update projection were also tested, but not retained:
they changed some depth/normal or secondary-ray results. In a 320×240 diagnostic frame of
NURBS parts, 94% of Newton calls finished in two or three iterations, and 86% of fallback entries
were grazing cases after Newton converged. The fallback's 13,053 projection evaluations were
24% of the counted Newton-plus-projection evaluations; this is a work count, not a GPU-time share.
