"""Drawing2CAD MCP server (stdio). Runs fully locally; free code only via the sandboxed fallback tool."""
from __future__ import annotations
from typing import Any, Optional
from mcp.server.fastmcp import FastMCP
from d2c.session import PartSession

mcp = FastMCP("drawing2cad")
S = PartSession()

SHAPE_DOC = (
    "shapes: list of 2D shapes on the XY plane, each with optional op='add' (default) or 'cut'. Types: "
    "{type:'circle',cx,cy,d} | {type:'rect',cx,cy,w,h,r?(corner radius)} | "
    "{type:'slot',x1,y1,x2,y2,w} (stadium; x1..y2 are the arc CENTRES) | {type:'polygon',points:[[x,y],..]} | "
    "{type:'path',start:[x,y],segments:[{line:[x,y]}|{arc:{mid:[x,y],to:[x,y]}}]} (auto-closed). Units mm.")


@mcp.tool()
def start_part(name: str, expected: Optional[dict[str, Any]] = None) -> dict:
    """Begin a new part (resets state). `expected` = dimensions read from the drawing, used by validate():
    {bbox:[dx,dy,dz], cylinders:[{id, diameter, axis:'X'|'Y'|'Z', perp:[a,b] (axis position in the two
    coordinates perpendicular to the axis, in model coordinates), length?}], tolerance_mm?}."""
    return S.start_part(name, expected)


@mcp.tool()
def create_extruded_body(shapes: list[dict], z_to: float, z_from: float = 0.0) -> dict:
    """Create the base body by extruding 2D shapes along +Z from z_from to z_to. """ + SHAPE_DOC
    return S.add_feature({"kind": "extruded_body", "shapes": shapes, "z_from": z_from, "z_to": z_to})


@mcp.tool()
def create_revolved_body(profile: list[list[float]], axis: str = "Z") -> dict:
    """Create a turned part by revolving a closed polyline profile 360 deg. profile = [[axial, radius], ...]
    (radius >= 0; closed automatically). axis = 'X' | 'Y' | 'Z' is the rotation axis in model space; the part
    starts at axial=0 and grows toward +axis."""
    return S.add_feature({"kind": "revolved_body", "profile": profile, "axis": axis})


@mcp.tool()
def add_extrusion(shapes: list[dict], z_from: float, z_to: float, mode: str = "add") -> dict:
    """Add (mode='add': boss/rib) or remove (mode='cut': pocket/slot/keyway) material by extruding 2D shapes
    along Z between z_from and z_to. """ + SHAPE_DOC
    return S.add_feature({"kind": "extrusion", "shapes": shapes, "z_from": z_from, "z_to": z_to, "mode": mode})


@mcp.tool()
def add_holes(diameter: float, start: float, positions: Optional[list[list[float]]] = None,
              bolt_circle: Optional[dict[str, float]] = None, depth: Optional[float] = None, axis: str = "Z",
              direction: int = -1, cbore_d: Optional[float] = None, cbore_depth: Optional[float] = None) -> dict:
    """Drill one or more holes (identical diameter). positions=[[a,b],..] are the two coordinates perpendicular to
    `axis` (Z axis -> [x,y]; X axis -> [y,z]; Y axis -> [z,x]). Or bolt_circle={pcd,count,start_angle?,cx?,cy?}.
    `start` = coordinate along the axis where drilling starts; direction -1 drills toward -axis, +1 toward +axis;
    depth=None -> through all. Optional counterbore (cbore_d, cbore_depth)."""
    return S.add_feature({"kind": "holes", "diameter": diameter, "start": start, "positions": positions,
                          "bolt_circle": bolt_circle, "depth": depth, "axis": axis, "direction": direction,
                          "cbore_d": cbore_d, "cbore_depth": cbore_depth})


@mcp.tool()
def round_circular_edges(edges: list[dict], axis: str, mode: str = "fillet") -> dict:
    """Fillet or chamfer circular edges selected by geometry: edges=[{radius_at, axial_at, size}] where radius_at is the
    edge radius, axial_at its coordinate along `axis`, size the fillet radius / chamfer length. mode='fillet'|'chamfer'."""
    return S.add_feature({"kind": "round_edges", "edges": edges, "axis": axis, "mode": mode})


@mcp.tool()
def add_review_item(item: str, assumed_value: str, reason: str, severity: str = "medium", drawing_source: str = "") -> dict:
    """Record an assumption / ambiguity for human review (missing dimension, unclear detail, inconsistent dims)."""
    return S.add_review_item(item, assumed_value, reason, severity, drawing_source)


@mcp.tool()
def run_sandboxed_script(code: str, mode: str = "replace") -> dict:
    """FALLBACK for geometry the structured tools cannot express. CadQuery python (`cq`, `math` preloaded; only
    cadquery/math/numpy imports) must assign the final Workplane/Shape to `result`. Runs in an isolated child process
    (no network, 30 s, 3 GB). mode: 'replace' | 'add' (union with current) | 'cut' (subtract from current)."""
    return S.run_script(code, mode)


@mcp.tool()
def validate_model() -> dict:
    """Check solid validity/closedness and compare the model with the expected drawing dimensions. Machine-readable."""
    return S.validate()


@mcp.tool()
def export_step(filename: Optional[str] = None) -> dict:
    """Validate, write STEP (AP214 geometry), re-import the file and confirm it loads, is valid and matches volume/bbox."""
    return S.export_step(filename)


@mcp.tool()
def get_state() -> dict:
    """Current feature history, bounding box, volume and review items."""
    return S.state()


if __name__ == "__main__":
    mcp.run()
