#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cadquery-ocp", "usd-core", "numpy"]
# ///

"""Convert a STEP assembly to a USD stage of BrepArray prims, one per unique part, instanced.

Standalone: ``uv`` installs the dependencies on first run (no repository environment needed):

    ./step_to_usd.py model.stp model.usdc                       # or: uv run --script step_to_usd.py ...
    ./step_to_usd.py model.stp model.usdc --mesh model_mesh.usdc  # plus a triangle-mesh twin

OpenCASCADE (through ``cadquery-ocp``) reads the STEP file with its assembly structure (XCAF).
Each unique part becomes one ``BrepArray`` under the abstract ``/Parts`` class; each placement
becomes an ``instanceable`` Xform referencing it, so renderers share one acceleration structure
per part. STEP surface colors become ``UsdPreviewSurface`` materials on the placements.

Faces are written with the surface geometry, UV trim curves (pcurves) and orientation that the
renderer in this directory reads, in the omniSolid schema's conventions:

- planes, cylinders, cones, spheres and tori stay analytic; OCCT frames that are left-handed, and
  cones (OCCT measures v along the slant), are mapped by an affine change of (u, v), applied to
  the face range and pcurves;
- B-spline and Bezier faces keep their parameterization and pcurves; other surfaces (revolution,
  extrusion, offset) become NURBS without pcurves, since conversion may reparameterize them; the
  reader then projects the 3D edges instead.

This is a minimal writer for rendering experiments: edges are shared per part, but the radial
edgeuse ordering is not computed, so the result is not meant for modeling kernels.
"""

import argparse
import math
import sys
import time
from collections import defaultdict

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as TopTools_IndexedMapOfShape
from OCP.collections import Sequence_TDF_Label
from OCP.Geom import Geom_RectangularTrimmedSurface, Geom_TrimmedCurve
from OCP.Geom2d import Geom2d_TrimmedCurve
from OCP.Geom2dConvert import Geom2dConvert
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.GeomAdaptor import GeomAdaptor_Surface
from OCP.GeomConvert import GeomConvert
from OCP.IFSelect import IFSelect_RetDone
from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
from OCP.TDF import TDF_Label, TDF_Tool
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_VERTEX, TopAbs_WIRE
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_ColorType, XCAFDoc_DocumentTool
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

T = GeomAbs_SurfaceType
ANALYTIC = {
    T.GeomAbs_Plane: "BrepSurfacePlaneAPI",
    T.GeomAbs_Cylinder: "BrepSurfaceCylinderAPI",
    T.GeomAbs_Cone: "BrepSurfaceConeAPI",
    T.GeomAbs_Sphere: "BrepSurfaceSphereAPI",
    T.GeomAbs_Torus: "BrepSurfaceTorusAPI",
}


def _vec(d):
    return (d.X(), d.Y(), d.Z())


def _knots(n, knot, mult):
    out = []
    for i in range(1, n + 1):
        out += [knot(i)] * mult(i)
    return out


def _bspline_curve(curve, dim):
    """(points, weights, knots, order) of a B-spline curve (Geom or Geom2d)."""
    n = curve.NbPoles()
    if dim == 3:
        points = [_vec(curve.Pole(i)) for i in range(1, n + 1)]
    else:
        points = [(curve.Pole(i).X(), curve.Pole(i).Y()) for i in range(1, n + 1)]
    weights = [curve.Weight(i) for i in range(1, n + 1)]
    return points, weights, _knots(curve.NbKnots(), curve.Knot, curve.Multiplicity), curve.Degree() + 1


class PartWriter:
    """Collects one part's faces into BrepArray attribute arrays."""

    def __init__(self, shape):
        self.shape = shape
        self.stats = defaultdict(int)
        self.edges = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, TopAbs_EDGE, self.edges)
        self.vertices = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, TopAbs_VERTEX, self.vertices)
        self.a = defaultdict(list)
        self.written_edges = set()

    def surface(self, face):
        """Append the face's surface; returns (affine (su, sv), flips normal, pcurves usable) or None."""
        surf = BRep_Tool.Surface_s(face)  # with the face's location applied
        while isinstance(surf, Geom_RectangularTrimmedSurface):
            surf = surf.BasisSurface()
        ad = GeomAdaptor_Surface(surf)
        kind = ad.GetType()
        a = self.a
        if kind in ANALYTIC:
            gp = {
                T.GeomAbs_Plane: ad.Plane,
                T.GeomAbs_Cylinder: ad.Cylinder,
                T.GeomAbs_Cone: ad.Cone,
                T.GeomAbs_Sphere: ad.Sphere,
                T.GeomAbs_Torus: ad.Torus,
            }[kind]()
            ax = gp.Position()
            origin, axis, ref = _vec(ax.Location()), _vec(ax.Direction()), _vec(ax.XDirection())
            su = 1.0 if ax.Direct() else -1.0  # left-handed frames: the schema's u runs the other way
            sv = 1.0
            if kind == T.GeomAbs_Plane:
                # The schema's Y is axis x ref; with axis = X x Y the parameterization is OCCT's.
                axis = tuple(np.cross(ref, _vec(ax.YDirection())))
                a["plane"].append((origin, axis, ref))
                su = 1.0
            elif kind == T.GeomAbs_Cylinder:
                a["cylinder"].append((origin, axis, ref, gp.Radius()))
            elif kind == T.GeomAbs_Cone:
                a["cone"].append((origin, axis, ref, gp.RefRadius(), gp.SemiAngle()))
                sv = math.cos(gp.SemiAngle())  # OCCT's v runs along the slant
            elif kind == T.GeomAbs_Sphere:
                a["sphere"].append((origin, axis, ref, gp.Radius()))
            else:
                a["torus"].append((origin, axis, ref, gp.MajorRadius(), gp.MinorRadius()))
            a["face:surfaceType"].append(ANALYTIC[kind])
            self.stats[ANALYTIC[kind]] += 1
            return (su, sv), su < 0.0, True
        exact = kind in (T.GeomAbs_BSplineSurface, T.GeomAbs_BezierSurface)
        try:
            bs = GeomConvert.SurfaceToBSplineSurface_s(surf)
        except Exception:  # noqa: BLE001
            self.stats["skipped_faces"] += 1
            return None
        if bs.IsUPeriodic():
            bs.SetUNotPeriodic()
        if bs.IsVPeriodic():
            bs.SetVNotPeriodic()
        nu, nv = bs.NbUPoles(), bs.NbVPoles()
        a["nurb:cvs"] += [_vec(bs.Pole(i, j)) for i in range(1, nu + 1) for j in range(1, nv + 1)]
        a["nurb:weights"] += [bs.Weight(i, j) for i in range(1, nu + 1) for j in range(1, nv + 1)]
        a["nurb:uCount"].append(nu)
        a["nurb:vCount"].append(nv)
        a["nurb:uOrder"].append(bs.UDegree() + 1)
        a["nurb:vOrder"].append(bs.VDegree() + 1)
        a["nurb:uKnots"] += _knots(bs.NbUKnots(), bs.UKnot, bs.UMultiplicity)
        a["nurb:vKnots"] += _knots(bs.NbVKnots(), bs.VKnot, bs.VMultiplicity)
        a["face:surfaceType"].append("BrepSurfaceNurbAPI")
        self.stats["BrepSurfaceNurbAPI" if exact else "converted_to_nurbs"] += 1
        return (1.0, 1.0), False, exact

    def edge(self, edge):
        """Index of the edge (shared by the faces using it); writes its 3D curve once."""
        index = self.edges.FindIndex(edge) - 1
        if index in self.written_edges:
            return index
        self.written_edges.add(index)
        first, last = BRep_Tool.Range_s(edge)
        curve = None if BRep_Tool.Degenerated_s(edge) else BRep_Tool.Curve_s(edge, 0.0, 0.0)
        if curve is not None:
            bs = GeomConvert.CurveToBSplineCurve_s(Geom_TrimmedCurve(curve, first, last))
            points, weights, knots, order = _bspline_curve(bs, 3)
            first, last = bs.FirstParameter(), bs.LastParameter()
        else:  # a degenerate edge (e.g. a pole) is a single point
            p = _vec(BRep_Tool.Pnt_s(TopExp.FirstVertex_s(edge)))
            points, weights, knots, order = [p, p], [1.0, 1.0], [first, first, last, last], 2
        self.a["edge_records"].append((index, points, weights, knots, order, first, last))
        v0 = self.vertices.FindIndex(TopExp.FirstVertex_s(edge)) - 1
        v1 = self.vertices.FindIndex(TopExp.LastVertex_s(edge)) - 1
        self.a["edge_vertices"].append((index, v0, v1))
        return index

    def add_face(self, face):
        result = self.surface(face)
        if result is None:
            return
        (su, sv), native_flip, pcurves = result
        a = self.a
        umin, umax, vmin, vmax = BRepTools.UVBounds_s(face)
        u = sorted((su * umin, su * umax))
        v = sorted((sv * vmin, sv * vmax))
        a["face:range"] += [(u[0], v[0]), (u[1], v[1])]
        # Outward = the face's normal; it is the schema surface normal unless one of these flips.
        a["flip"].append((face.Orientation() == TopAbs_REVERSED) != native_flip)
        outer = BRepTools.OuterWire_s(face)
        wires = [outer]
        ex = TopExp_Explorer(face, TopAbs_WIRE)
        while ex.More():
            w = TopoDS.Wire(ex.Current())
            if not w.IsSame(outer):
                wires.append(w)
            ex.Next()
        loops = 0
        for wire in wires:
            count = 0
            wexp = BRepTools_WireExplorer(wire, face)
            while wexp.More():
                edge = wexp.Current()
                a["edgeuse:edgeIndex"].append(self.edge(edge))
                reversed_ = edge.Orientation() == TopAbs_REVERSED
                a["edgeuse:orientationType"].append("opposite" if reversed_ else "same")
                record = None
                if pcurves:
                    c2 = BRep_Tool.CurveOnSurface_s(edge, face, 0.0, 0.0)
                    if c2 is not None:
                        f2, l2 = BRep_Tool.Range_s(edge, face)
                        bs2 = Geom2dConvert.CurveToBSplineCurve_s(Geom2d_TrimmedCurve(c2, f2, l2))
                        if reversed_:
                            bs2.Reverse()  # pcurves follow the loop's traversal
                        points, weights, knots, order = _bspline_curve(bs2, 2)
                        record = ([(su * p[0], sv * p[1]) for p in points], weights, knots, order)
                a["uv_records"].append(record)
                count += 1
                wexp.Next()
            if count:
                a["loop:edgeuseCount"].append(count)
                loops += 1
        a["face:loopCount"].append(loops)

    def write(self, prim):
        """Author the BrepArray attributes on ``prim``."""
        a = self.a
        nfaces = len(a["face:surfaceType"])
        t = Sdf.ValueTypeNames

        def attr(name, type_name, value):
            prim.CreateAttribute(name, type_name, custom=False, variability=Sdf.VariabilityUniform).Set(value)

        points = [p for (_, pts, *_rest) in a["edge_records"] for p in pts] or [(0.0, 0.0, 0.0)]
        lo, hi = np.min(points, axis=0), np.max(points, axis=0)
        attr("brep:extent", t.Double3Array, Vt.Vec3dArray([Gf.Vec3d(*lo), Gf.Vec3d(*hi)]))
        attr("brep:intersectTol3d", t.DoubleArray, [1.0e-6 * float(np.linalg.norm(hi - lo) + 1.0)])
        attr("brep:regionCount", t.UIntArray, [2])
        attr("region:type", t.TokenArray, ["voidRegion", "solidRegion"])
        attr("region:shellCount", t.UIntArray, [1, 1])
        attr("shell:faceuseCount", t.UIntArray, [nfaces, nfaces])
        attr("shell:wireEdgeCount", t.UIntArray, [0, 0])
        attr("shell:pointType", t.TokenArray, ["none", "none"])
        attr("faceuse:faceIndex", t.UIntArray, list(range(nfaces)) * 2)
        # The void region's faceuse sees the outward side.
        void = ["opposite" if f else "same" for f in a["flip"]]
        solid = ["same" if f else "opposite" for f in a["flip"]]
        attr("faceuse:orientationType", t.TokenArray, void + solid)
        attr("face:surfaceType", t.TokenArray, a["face:surfaceType"])
        attr("face:trimType", t.TokenArray, ["general"] * nfaces)
        attr("face:loopCount", t.UIntArray, a["face:loopCount"])
        attr("face:range", t.Double2Array, Vt.Vec2dArray([Gf.Vec2d(*r) for r in a["face:range"]]))
        attr("loop:edgeuseCount", t.UIntArray, a["loop:edgeuseCount"])
        attr("loop:vertexIndex", t.UIntArray, [0] * len(a["loop:edgeuseCount"]))
        neu = len(a["edgeuse:edgeIndex"])
        attr("edgeuse:edgeIndex", t.UIntArray, a["edgeuse:edgeIndex"])
        attr("edgeuse:orientationType", t.TokenArray, a["edgeuse:orientationType"])
        attr("edgeuse:nextRadialEUIndex", t.UIntArray, list(range(neu)))
        attr("edgeuse:thisRadialEntryType", t.TokenArray, ["topEntry"] * neu)

        # Edges in index order.
        records = sorted(a["edge_records"])
        nedges = len(records)
        attr("edge:curveType", t.TokenArray, ["BrepCurve3dNurbAPI"] * nedges)
        attr("edge:range", t.DoubleArray, [x for r in records for x in (r[5], r[6])])
        verts = dict((i, (v0, v1)) for i, v0, v1 in a["edge_vertices"])
        attr("edge:vertexIndices", t.Int2Array, Vt.Vec2iArray([Gf.Vec2i(*verts[r[0]]) for r in records]))
        nb = "brep:edge3dNurb:curve3d:nurb:"
        attr(nb + "controlVertices", t.Point3dArray, Vt.Vec3dArray([Gf.Vec3d(*p) for r in records for p in r[1]]))
        attr(nb + "weights", t.DoubleArray, [w for r in records for w in r[2]])
        attr(nb + "knots", t.DoubleArray, [k for r in records for k in r[3]])
        attr(nb + "order", t.UIntArray, [r[4] for r in records])
        attr(nb + "vertexCount", t.UIntArray, [len(r[1]) for r in records])
        nverts = self.vertices.Extent()
        attr("vertex:pointType", t.TokenArray, ["BrepPointAPI"] * nverts)
        attr(
            "brep:vertexPoint:point:position",
            t.Point3dArray,
            Vt.Vec3dArray(
                [Gf.Vec3d(*_vec(BRep_Tool.Pnt_s(TopoDS.Vertex(self.vertices.FindKey(i + 1))))) for i in range(nverts)]
            ),
        )

        uv = a["uv_records"]
        if any(r is not None for r in uv):
            ub = "brep:curveUv:nurb:"
            attr(ub + "controlVertices", t.Double2Array, Vt.Vec2dArray([Gf.Vec2d(*p) for r in uv if r for p in r[0]]))
            attr(ub + "weights", t.DoubleArray, [w for r in uv if r for w in r[1]])
            attr(ub + "knots", t.DoubleArray, [k for r in uv if r for k in r[2]])
            attr(ub + "order", t.UIntArray, [r[3] if r else 0 for r in uv])
            attr(ub + "vertexCount", t.UIntArray, [len(r[0]) if r else 0 for r in uv])

        for key, fields in (
            ("plane", ("origin", "axis", "refDirection")),
            ("cylinder", ("origin", "axis", "refDirection", "radius")),
            ("cone", ("origin", "axis", "refDirection", "radius", "semiAngle")),
            ("sphere", ("center", "axis", "refDirection", "radius")),
            ("torus", ("origin", "axis", "refDirection", "majorRadius", "minorRadius")),
        ):
            rows = a[key]
            if not rows:
                continue
            for i, field in enumerate(fields):
                name = f"brep:surface:{key}:{field}"
                values = [r[i] for r in rows]
                if field in ("origin", "center"):
                    attr(name, t.Point3dArray, Vt.Vec3dArray([Gf.Vec3d(*v) for v in values]))
                elif field in ("axis", "refDirection"):
                    attr(name, t.Vector3dArray, Vt.Vec3dArray([Gf.Vec3d(*v) for v in values]))
                else:
                    attr(name, t.DoubleArray, values)
        if a["nurb:uCount"]:
            sb = "brep:surface:nurb:"
            attr(sb + "controlVertices", t.Point3dArray, Vt.Vec3dArray([Gf.Vec3d(*p) for p in a["nurb:cvs"]]))
            attr(sb + "weights", t.DoubleArray, a["nurb:weights"])
            attr(sb + "uVertexCount", t.UIntArray, a["nurb:uCount"])
            attr(sb + "vVertexCount", t.UIntArray, a["nurb:vCount"])
            attr(sb + "uOrder", t.UIntArray, a["nurb:uOrder"])
            attr(sb + "vOrder", t.UIntArray, a["nurb:vOrder"])
            attr(sb + "uKnots", t.DoubleArray, a["nurb:uKnots"])
            attr(sb + "vKnots", t.DoubleArray, a["nurb:vKnots"])
        return nfaces


def _matrix(location: TopLoc_Location) -> Gf.Matrix4d:
    trsf = location.Transformation()
    m = np.eye(4)
    for r in range(3):
        for c in range(4):
            m[r, c] = trsf.Value(r + 1, c + 1)
    return Gf.Matrix4d(m.T.tolist())  # USD matrices act on row vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step")
    parser.add_argument("usd")
    parser.add_argument("--mesh", help="also write the same assembly as OpenCASCADE triangle meshes to this USD file")
    parser.add_argument(
        "--deflection", type=float, default=1e-3, help="mesh chord deflection, relative to each part's size"
    )
    parser.add_argument("--angle", type=float, default=20.0, help="mesh angular deflection in degrees")
    args = parser.parse_args()

    t0 = time.perf_counter()
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    if reader.ReadFile(args.step) != IFSelect_RetDone or not reader.Transfer(doc):
        sys.exit(f"cannot read {args.step}")
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    colors = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    print(f"read STEP in {time.perf_counter() - t0:.1f} s")

    def color_of(label):
        c = Quantity_Color()
        for kind in (XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen):
            if colors.GetColor_s(label, kind, c):
                return c.Values(Quantity_TOC_RGB)
        return None

    prototypes = {}  # part label entry -> (index, shape, color)
    placements = []  # (prototype index, location, color)

    def walk(label, location, inherited):
        color = color_of(label) or inherited
        if shapes.IsAssembly_s(label):
            components = Sequence_TDF_Label()
            shapes.GetComponents_s(label, components)
            for i in range(1, components.Length() + 1):
                component = components.Value(i)
                ref = TDF_Label()
                shapes.GetReferredShape_s(component, ref)
                walk(ref, location * shapes.GetLocation_s(component), color_of(component) or color)
            return
        entry = TCollection_AsciiString()
        TDF_Tool.Entry_s(label, entry)
        key = entry.ToCString()
        if key not in prototypes:
            prototypes[key] = (len(prototypes), shapes.GetShape_s(label), color_of(label))
        placements.append((prototypes[key][0], location, color))

    free = Sequence_TDF_Label()
    shapes.GetFreeShapes(free)
    for i in range(1, free.Length() + 1):
        label = free.Value(i)
        walk(label, shapes.GetLocation_s(label), None)

    def author(path, write_part):
        stage = Usd.Stage.CreateNew(path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 0.001)  # OCCT reads STEP in millimeters
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        stage.CreateClassPrim("/Parts")
        looks = {}

        def material(rgb):
            rgb = tuple(round(float(x), 3) for x in (rgb or (0.7, 0.7, 0.72)))
            if rgb not in looks:
                path = f"/World/Looks/C_{len(looks)}"
                mat = UsdShade.Material.Define(stage, path)
                shader = UsdShade.Shader.Define(stage, path + "/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                looks[rgb] = mat
            return looks[rgb]

        t1 = time.perf_counter()
        totals = defaultdict(int)
        empty = set()
        for index, shape, _ in sorted(prototypes.values(), key=lambda p: p[0]):
            # A container per part, so a placement referencing it keeps its own (Xform) type.
            stage.DefinePrim(f"/Parts/P_{index}", "Xform")
            if not write_part(stage, f"/Parts/P_{index}", shape, totals):
                empty.add(index)
        print(f"  wrote {len(prototypes) - len(empty)} parts in {time.perf_counter() - t1:.1f} s: {dict(totals)}")
        placed = 0
        for n, (index, location, color) in enumerate(placements):
            if index in empty:
                continue
            xf = UsdGeom.Xform.Define(stage, f"/World/I_{n}")
            xf.AddTransformOp().Set(_matrix(location))
            xf.GetPrim().GetReferences().AddInternalReference(f"/Parts/P_{index}")
            xf.GetPrim().SetInstanceable(True)
            UsdShade.MaterialBindingAPI.Apply(xf.GetPrim()).Bind(material(color))
            placed += 1
        stage.GetRootLayer().Save()
        print(f"  {placed} placements, {len(looks)} materials -> {path}")

    def write_brep(stage, path, shape, totals):
        writer = PartWriter(shape)
        ex = TopExp_Explorer(shape, TopAbs_FACE)
        while ex.More():
            writer.add_face(TopoDS.Face(ex.Current()))
            ex.Next()
        if not writer.a["face:surfaceType"]:
            return False
        totals["faces"] += writer.write(stage.DefinePrim(path + "/Brep", "BrepArray"))
        for k, v in writer.stats.items():
            totals[k] += v
        return True

    def write_mesh(stage, path, shape, totals):
        from OCP.Bnd import Bnd_Box  # noqa: PLC0415
        from OCP.BRepBndLib import BRepBndLib  # noqa: PLC0415
        from OCP.BRepMesh import BRepMesh_IncrementalMesh  # noqa: PLC0415

        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        if box.IsVoid():
            return False
        size = math.dist(_vec(box.CornerMin()), _vec(box.CornerMax()))
        if not (size > 0.0 and math.isfinite(size)) or box.IsOpen():  # degenerate or unbounded part
            print(f"  skipping part {path}: bounding box size {size}, open {box.IsOpen()}")
            return False
        try:
            BRepMesh_IncrementalMesh(shape, args.deflection * size, False, math.radians(args.angle), True)
        except Exception as error:  # noqa: BLE001
            print(f"  mesher failed on {path} (size {size}, deflection {args.deflection * size}): {error}")
            return False
        points, indices = [], []
        ex = TopExp_Explorer(shape, TopAbs_FACE)
        while ex.More():
            face = TopoDS.Face(ex.Current())
            location = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, location)
            if tri is not None:
                trsf = location.Transformation()
                base = len(points)
                for i in range(1, tri.NbNodes() + 1):
                    points.append(_vec(tri.Node(i).Transformed(trsf)))
                flip = face.Orientation() == TopAbs_REVERSED
                for i in range(1, tri.NbTriangles() + 1):
                    t = tri.Triangle(i)
                    a, b, c = t.Value(1), t.Value(2), t.Value(3)
                    indices += (
                        [base + a - 1, base + c - 1, base + b - 1]
                        if flip
                        else [base + a - 1, base + b - 1, base + c - 1]
                    )
            ex.Next()
        if not indices:
            return False
        mesh = UsdGeom.Mesh.Define(stage, path + "/Mesh")
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(indices) // 3)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        totals["triangles"] += len(indices) // 3
        return True

    print("BRep stage:")
    author(args.usd, write_brep)
    if args.mesh:
        print(f"mesh stage (deflection {args.deflection:g} of part size, {args.angle:g} deg):")
        author(args.mesh, write_mesh)
    print(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
