"""Deterministic CAD feature kernel (CadQuery / OpenCascade).

The model is a *feature history* (list of JSON-serialisable dicts). The solid is always
rebuilt by replaying the history, so the model stays editable (change a parameter -> rebuild).
No tool in this module executes free-form code.
"""
from __future__ import annotations
import math
import cadquery as cq

AXES = {"X", "Y", "Z"}


# ---------------------------------------------------------------- 2D sketch helpers
def _wp_xy(z: float = 0.0) -> cq.Workplane:
    return cq.Workplane("XY", origin=(0, 0, z))


def _shape_solid(s: dict, z0: float, z1: float) -> cq.Workplane:
    """Extrude one 2D shape (on XY) from z0 to z1."""
    h = z1 - z0
    if h <= 0:
        raise ValueError(f"extrusion needs z_to > z_from (got {z0}..{z1})")
    wp = _wp_xy(z0)
    t = s["type"]
    if t == "circle":
        return wp.center(s["cx"], s["cy"]).circle(s["d"] / 2.0).extrude(h)
    if t == "rect":
        w = wp.center(s["cx"], s["cy"]).rect(s["w"], s["h"]).extrude(h)
        r = s.get("r", 0)
        if r:
            w = w.edges("|Z").fillet(r)
        return w
    if t == "slot":  # two end centres + width, stadium shape
        x1, y1, x2, y2, wd = s["x1"], s["y1"], s["x2"], s["y2"], s["w"]
        L = math.hypot(x2 - x1, y2 - y1)
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        return (wp.center((x1 + x2) / 2, (y1 + y2) / 2).transformed(rotate=(0, 0, ang))
                .slot2D(L + wd, wd).extrude(h))
    if t == "polygon":
        return wp.polyline([tuple(p) for p in s["points"]]).close().extrude(h)
    if t == "path":
        w = wp.moveTo(*s["start"])
        for seg in s["segments"]:
            if "line" in seg:
                w = w.lineTo(*seg["line"])
            elif "arc" in seg:  # three point arc: mid + end
                w = w.threePointArc(tuple(seg["arc"]["mid"]), tuple(seg["arc"]["to"]))
            else:
                raise ValueError(f"unknown path segment {seg}")
        return w.close().extrude(h)
    raise ValueError(f"unknown shape type '{t}'")


def _sketch_solid(shapes: list[dict], z0: float, z1: float) -> cq.Shape:
    """adds are unioned, cuts subtracted (in order)."""
    adds = [s for s in shapes if s.get("op", "add") == "add"]
    cuts = [s for s in shapes if s.get("op") == "cut"]
    if not adds:
        raise ValueError("sketch needs at least one 'add' shape")
    body = _shape_solid(adds[0], z0, z1).val()
    for s in adds[1:]:
        body = body.fuse(_shape_solid(s, z0, z1).val())
    for s in cuts:
        body = body.cut(_shape_solid(s, z0 - 1, z1 + 1).val())
    return body.clean()


# ---------------------------------------------------------------- axis helpers
def _rot_to_axis(shape: cq.Shape, axis: str) -> cq.Shape:
    """Shape built around Z -> re-orient so its Z axis becomes `axis`."""
    if axis == "Z":
        return shape
    if axis == "X":
        return shape.rotate((0, 0, 0), (0, 1, 0), 90)   # Z -> +X
    if axis == "Y":
        return shape.rotate((0, 0, 0), (1, 0, 0), -90)  # Z -> +Y
    raise ValueError(axis)


def _circ_edges(shape: cq.Shape, axis: str, radius: float, axial: float, tol=1e-3):
    out = []
    ai = "XYZ".index(axis)
    for e in shape.Edges():
        if e.geomType() != "CIRCLE":
            continue
        if abs(e.radius() - radius) > tol:
            continue
        c = e.Center()
        if abs((c.x, c.y, c.z)[ai] - axial) > tol:
            continue
        # circle must be perpendicular to axis
        n = e.normal()
        if abs(abs((n.x, n.y, n.z)[ai]) - 1) > 1e-6:
            continue
        out.append(e)
    return out


# ---------------------------------------------------------------- feature application
def apply_feature(solid: cq.Shape | None, f: dict) -> cq.Shape:
    k = f["kind"]
    if k == "extruded_body":
        return _sketch_solid(f["shapes"], f.get("z_from", 0.0), f["z_to"])

    if k == "revolved_body":
        axis = f.get("axis", "Z")
        pts = [tuple(p) for p in f["profile"]]  # [axial, radius]
        # build around Z: x=radius, y=axial -> revolve about Y then map
        wp = cq.Workplane("XZ").polyline([(r, a) for a, r in pts]).close()
        body = wp.revolve(360, (0, 0, 0), (0, 1, 0)).val()  # XZ plane: local y == global Z
        return _rot_to_axis(body, axis).clean()

    if solid is None:
        raise ValueError("create a base body first (create_extruded_body / create_revolved_body)")

    if k == "extrusion":
        tool = _sketch_solid(f["shapes"], f["z_from"], f["z_to"])
        return (solid.fuse(tool) if f.get("mode", "add") == "add" else solid.cut(tool)).clean()

    if k == "holes":
        axis = f.get("axis", "Z")
        pos = list(f.get("positions") or [])
        bc = f.get("bolt_circle")
        if bc:
            for i in range(int(bc["count"])):
                a = math.radians(bc.get("start_angle", 0.0) + i * 360.0 / bc["count"])
                pos.append([bc.get("cx", 0.0) + bc["pcd"] / 2 * math.cos(a),
                            bc.get("cy", 0.0) + bc["pcd"] / 2 * math.sin(a)])
        if not pos:
            raise ValueError("holes need positions or bolt_circle")
        z0 = f["start"]
        depth = f.get("depth")  # None -> through (long cutter)
        d = f["diameter"]
        direction = f.get("direction", -1)  # -1: drill toward -axis from start, +1 toward +axis
        L = depth if depth is not None else 10000.0
        zlo, zhi = (z0 - L, z0) if direction < 0 else (z0, z0 + L)
        if depth is None:  # through: extend a bit both sides of start
            zlo, zhi = (z0 - 10000, z0 + 1) if direction < 0 else (z0 - 1, z0 + 10000)
        shapes = [{"type": "circle", "cx": p[0], "cy": p[1], "d": d} for p in pos]
        cutter = _sketch_solid(shapes, zlo, zhi)
        if f.get("cbore_d"):
            cb = [{"type": "circle", "cx": p[0], "cy": p[1], "d": f["cbore_d"]} for p in pos]
            cz = f["cbore_depth"]
            lo, hi = (z0 - cz, z0 + 1) if direction < 0 else (z0 - 1, z0 + cz)
            cutter = cutter.fuse(_sketch_solid(cb, lo, hi))
        cutter = _rot_to_axis(cutter, axis)
        return solid.cut(cutter).clean()

    if k == "round_edges":
        axis = f["axis"]
        out = solid
        for item in f["edges"]:
            es = _circ_edges(out, axis, item["radius_at"], item["axial_at"])
            if not es:
                raise ValueError(f"no circular edge at radius {item['radius_at']} / axial {item['axial_at']}")
            wp = cq.Workplane(obj=out).newObject(es)
            wp = wp.fillet(item["size"]) if f.get("mode", "fillet") == "fillet" else wp.chamfer(item["size"])
            out = wp.val()
        return out

    if k == "script":
        raise RuntimeError("script features are resolved by the session (sandbox)")
    raise ValueError(f"unknown feature kind '{k}'")
