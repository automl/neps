"""Machine-readable validation: solid integrity, dimension comparison, STEP re-import."""
from __future__ import annotations
import math, os
import cadquery as cq
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopExp import TopExp
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

TOL_DEFAULT = 0.05  # mm


def _check(id_, kind, expected, actual, tol, extra=None):
    ok = actual is not None and (abs(actual - expected) <= tol if isinstance(expected, (int, float)) else actual == expected)
    d = {"id": id_, "kind": kind, "expected": expected, "actual": None if actual is None else round(actual, 4) if isinstance(actual, float) else actual,
         "tolerance": tol, "pass": bool(ok)}
    if isinstance(expected, (int, float)) and actual is not None:
        d["delta"] = round(actual - expected, 4)
    if extra:
        d.update(extra)
    return d


def free_edge_count(shape: cq.Shape) -> int:
    """Edges that belong to exactly one face -> open shell."""
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape.wrapped, TopAbs_EDGE, TopAbs_FACE, m)
    n = 0
    for i in range(1, m.Extent() + 1):
        if m.FindFromIndex(i).Size() == 1:
            # seam / degenerated edges are shared by the same face twice -> Size 2, fine
            n += 1
    return n


def cylinders(shape: cq.Shape) -> list[dict]:
    """Unique cylindrical surfaces (merged halves) with radius, axis direction and axis point."""
    out = {}
    for f in shape.Faces():
        ad = BRepAdaptor_Surface(f.wrapped)
        if ad.GetType() != GeomAbs_Cylinder:
            continue
        cyl = ad.Cylinder()
        loc, dr = cyl.Axis().Location(), cyl.Axis().Direction()
        d = (round(abs(dr.X())), round(abs(dr.Y())), round(abs(dr.Z())))
        axis = "XYZ"[d.index(1)] if sum(d) == 1 else "?"
        p = (loc.X(), loc.Y(), loc.Z())
        perp = tuple(round(c, 3) for i, c in enumerate(p) if "XYZ"[i] != axis)
        key = (round(cyl.Radius(), 3), axis, perp)
        bb = f.BoundingBox()
        ai = "XYZ".index(axis) if axis != "?" else 0
        lo, hi = (bb.xmin, bb.ymin, bb.zmin)[ai], (bb.xmax, bb.ymax, bb.zmax)[ai]
        e = out.setdefault(key, {"diameter": 2 * key[0], "axis": axis, "perp": perp, "lo": lo, "hi": hi})
        e["lo"], e["hi"] = min(e["lo"], lo), max(e["hi"], hi)
    return list(out.values())


def integrity(shape: cq.Shape) -> dict:
    checks = []
    solids = shape.Solids()
    checks.append(_check("single_solid", "count", 1, len(solids), 0))
    ana = BRepCheck_Analyzer(shape.wrapped)
    checks.append(_check("brep_valid", "bool", True, bool(ana.IsValid()), 0))
    checks.append(_check("shell_closed", "bool", True, all(sh.Closed() for sh in shape.Shells()) and len(shape.Shells()) >= 1, 0))
    checks.append(_check("free_edges", "count", 0, free_edge_count(shape), 0))
    vol = shape.Volume()
    checks.append(_check("positive_volume", "bool", True, vol > 0, 0, {"volume_mm3": round(vol, 3)}))
    return {"checks": checks, "volume_mm3": round(vol, 3), "pass": all(c["pass"] for c in checks)}


def compare_dimensions(shape: cq.Shape, expected: dict) -> dict:
    checks = []
    bb = shape.BoundingBox()
    tol = expected.get("tolerance_mm", TOL_DEFAULT)
    if "bbox" in expected:
        for ax, size, act in zip("XYZ", expected["bbox"], (bb.xlen, bb.ylen, bb.zlen)):
            if size is not None:
                checks.append(_check(f"bbox_{ax.lower()}", "overall_size", size, act, tol))
    cyls = cylinders(shape)
    for c in expected.get("cylinders", []):
        t = c.get("tolerance_mm", tol)
        best = None
        for m in cyls:
            if m["axis"] != c["axis"]:
                continue
            ex = tuple(c["perp"])
            dist = math.dist(ex, m["perp"])
            if abs(m["diameter"] - c["diameter"]) <= t and dist <= t:
                best = m
                break
        if best is None:  # report nearest candidate by position for debugging
            cand = [m for m in cyls if m["axis"] == c["axis"]]
            cand.sort(key=lambda m: math.dist(tuple(c["perp"]), m["perp"]))
            near = cand[0] if cand else None
            checks.append(_check(c["id"], "cylinder_diameter+position", c["diameter"], near["diameter"] if near else None, t,
                                 {"expected_axis_position": c["perp"], "nearest_actual_position": list(near["perp"]) if near else None}))
        else:
            checks.append(_check(c["id"], "cylinder_diameter+position", c["diameter"], best["diameter"], t,
                                 {"axis_position": list(best["perp"])}))
            if "length" in c:
                L = best["hi"] - best["lo"]
                checks.append(_check(c["id"] + "_length", "cylinder_length", c["length"], L, c.get("length_tol_mm", t)))
    return {"checks": checks, "pass": all(c["pass"] for c in checks), "n_checks": len(checks)}


def reimport_step(path: str, ref: cq.Shape) -> dict:
    checks = []
    try:
        imp = cq.importers.importStep(path)
        sh = imp.val() if hasattr(imp, "val") else imp
        shape = cq.Compound.makeCompound(imp.vals()) if hasattr(imp, "vals") and len(imp.vals()) > 1 else sh
        checks.append(_check("step_loads", "bool", True, True, 0))
        checks.append(_check("step_solid_count", "count", 1, len(shape.Solids()), 0))
        checks.append(_check("step_valid", "bool", True, bool(BRepCheck_Analyzer(shape.wrapped).IsValid()), 0))
        rv = ref.Volume()
        checks.append(_check("step_volume_matches", "volume_mm3", rv, shape.Volume(), max(1e-6 * rv, 0.01)))
        b0, b1 = ref.BoundingBox(), shape.BoundingBox()
        for n, a, b in (("x", b0.xlen, b1.xlen), ("y", b0.ylen, b1.ylen), ("z", b0.zlen, b1.zlen)):
            checks.append(_check(f"step_bbox_{n}", "overall_size", a, b, 1e-3))
    except Exception as e:  # noqa
        checks.append({"id": "step_loads", "kind": "bool", "expected": True, "actual": False, "pass": False, "error": str(e)})
    return {"checks": checks, "pass": all(c["pass"] for c in checks), "file_bytes": os.path.getsize(path) if os.path.exists(path) else 0}
