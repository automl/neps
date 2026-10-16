"""Authoring helper: writes the structured intermediate descriptions (drawing -> JSON) read off the PDFs.
Each spec = dimensions with provenance/confidence + expected checks + MCP tool plan + review items."""
import json, math, os

def dim(id, value, where, conf="high", note=""):
    d = {"id": id, "value": value, "found_in": where, "confidence": conf}
    if note: d["note"] = note
    return d

def rev(item, assumed, reason, sev="medium", src=""):
    return {"item": item, "assumed_value": assumed, "reason": reason, "severity": sev, "drawing_source": src}

def call(tool, **args): return {"tool": tool, "args": args}

def cyl(id, d, axis, perp, **kw): return {"id": id, "diameter": d, "axis": axis, "perp": perp, **kw}

specs = {}

# ------------------------------------------------------------------ WELLE (shaft, axis X)
prof = [[0,0],[0,16.5],[1,17.5],[35,17.5],[35,22.5],[100,22.5],[100,20],[250,20],[250,15],[277.5,15],[277.5,14],[280,14],[280,15],[299,15],[300,14],[300,0]]
specs["Welle"] = {
 "part": "Welle", "drawing": "Welle_dwg.pdf", "kind": "turned part (revolve about X)",
 "dimensions": [
  dim("length_total", 300, "front view, top dimension"),
  dim("d1", 35, "front view, left end"), dim("l1", 35, "front view, '35'"),
  dim("d2", 45, "front view"), dim("x_d2_end", 100, "front view, '100' from left end"),
  dim("d3", 40, "front view"), dim("l_right_section", 50, "front view, '50' from right end"),
  dim("d_right", 30, "front view, right end (Ø30)"), dim("groove_d", 28, "front view Ø28 / detail"),
  dim("groove_w", 2.5, "front view '2,5'"), dim("groove_to_end", 20, "front view '20' from right end to groove"),
  dim("left_chamfer", "1x45°", "front view bottom-left"), dim("fillet_d2_d3", 2, "front view 'R2'"),
  dim("keyway_len_centres", 80, "front view '80' (centre of arc to centre of arc)", "medium", "dimension arrows end at arc centres; overall slot = 92"),
  dim("keyway_end_offset", 35, "front view '35' from right slot-arc centre to the Ø40/Ø30 shoulder"),
  dim("keyway_w", 12, "front view '12' / section A-A"), dim("keyway_depth", 6, "section A-A '6' from shaft surface"),
 ],
 "expected": {"bbox": [300,45,45], "cylinders": [
   cyl("d35",35,"X",[0,0]), cyl("d45",45,"X",[0,0]), cyl("d40",40,"X",[0,0]), cyl("d30",30,"X",[0,0]), cyl("groove_d28",28,"X",[0,0])]},
 "plan": [
  call("create_revolved_body", profile=prof, axis="X"),
  call("round_circular_edges", edges=[{"radius_at":20,"axial_at":100,"size":2}], axis="X", mode="fillet"),
  call("round_circular_edges", edges=[{"radius_at":17.5,"axial_at":35,"size":2}], axis="X", mode="fillet"),
  call("round_circular_edges", edges=[{"radius_at":20,"axial_at":250,"size":1}], axis="X", mode="chamfer"),
  call("add_extrusion", shapes=[{"type":"slot","x1":135,"y1":0,"x2":215,"y2":0,"w":12,"op":"add"}], z_from=14, z_to=21, mode="cut"),
 ],
 "review": [
  rev("Groove Ø28 position", "x = 277.5 … 280 (2.5 wide, 20 mm from right end)", "'2,5' and '20' are stacked from the right end; groove is the Ø28 band, rest of right section is Ø30", "low", "front view detail"),
  rev("Fillet at Ø35/Ø45 shoulder", "R2", "round visible in the front view but not dimensioned; assumed equal to the only dimensioned fillet (R2)", "medium", "front view, x=35"),
  rev("Chamfer at Ø40 right shoulder edge (x=250)", "1x45°", "bevel visible on the outer corner of the Ø40 section, not dimensioned (assumed equal to the dimensioned 1x45° at the left end)", "medium", "front view, x=250"),
  rev("Chamfer at right end (Ø30)", "1x45°", "visible chamfer on the right end, no dimension (assumed equal to the dimensioned left chamfer)", "medium", "front view, right end"),
  rev("Fillet at Ø45/Ø40 shoulder", "R2 on the Ø40 side only", "'R2' arrow points at the inner corner at x=100; outer corner left sharp", "low", "front view"),
  rev("Keyway orientation", "on +Z side (top of the shaft in the isometric view)", "section A-A is rotated; the drawing does not state the angular position", "medium", "iso view / A-A"),
  rev("Keyway end shape", "full semicircular ends, centres 80 apart", "slot drawn with round ends; '80' read as centre-to-centre", "medium", "front view"),
  rev("Keyway depth reference", "6 mm below the Ø40 surface (flat bottom at 14 mm from axis)", "'6' is measured from the circle edge to the flat in A-A", "low", "section A-A"),
 ]}

# ------------------------------------------------------------------ FUSS (base plate + boss)
specs["Fuss"] = {
 "part": "Fuss", "drawing": "Fuss_dwg.pdf", "kind": "prismatic (extrude + boss)",
 "dimensions": [
  dim("plate_w", 115, "top view '115'"), dim("plate_h", 115, "top view, square", "medium", "only one 115 dim; plate drawn square"),
  dim("plate_t", 19, "front view '19'"), dim("corner_r", 22, "top view 'R22'"),
  dim("hole_d", 19, "top view Ø19"), dim("hole_pitch", 71, "top view '71' (centre to centre)"),
  dim("boss_d", 57, "top view Ø57"), dim("height_total", 76.2, "front view '76,2'"),
  dim("boss_bore_d", 33, "top view Ø33"), dim("base_bore_d", 19, "top view '19' above the centre"),
  dim("fillet", 9.5, "front view 'R9,5' (boss to plate)"),
 ],
 "expected": {"bbox": [115,115,76.2], "cylinders": [
   cyl("boss_d57",57,"Z",[0,0]), cyl("bore_d33",33,"Z",[0,0]), cyl("bore_d19",19,"Z",[0,0]),
   cyl("hole_1",19,"Z",[35.5,35.5]), cyl("hole_2",19,"Z",[-35.5,35.5]), cyl("hole_3",19,"Z",[35.5,-35.5]), cyl("hole_4",19,"Z",[-35.5,-35.5])]},
 "plan": [
  call("create_extruded_body", shapes=[{"type":"rect","cx":0,"cy":0,"w":115,"h":115,"r":22}], z_from=0, z_to=19),
  call("add_extrusion", shapes=[{"type":"circle","cx":0,"cy":0,"d":57}], z_from=19, z_to=76.2, mode="add"),
  call("round_circular_edges", edges=[{"radius_at":28.5,"axial_at":19,"size":9.5}], axis="Z", mode="fillet"),
  call("add_holes", diameter=19, start=19, positions=[[35.5,35.5],[-35.5,35.5],[35.5,-35.5],[-35.5,-35.5]], axis="Z", direction=-1),
  call("add_holes", diameter=33, start=76.2, positions=[[0,0]], depth=57.2, axis="Z", direction=-1),
  call("add_holes", diameter=19, start=19, positions=[[0,0]], axis="Z", direction=-1),
 ],
 "review": [
  rev("Plate height (Y) = 115", "115", "only one overall dimension given; plan view drawn square", "low", "top view"),
  rev("Hole pitch in both directions", "71 x 71, symmetric about the boss axis", "'71' given once (vertical); horizontal pitch read from the symmetric layout", "low", "top view"),
  rev("Ø33 bore depth", "57.2 (= 76.2 − 19, down to the plate top)", "no depth dimension; dashed lines in the front view show Ø33 over the boss height and Ø19 through the plate", "medium", "front view hidden lines"),
  rev("Which of Ø19 / Ø33 is the boss bore", "Ø33 upper bore, Ø19 lower through-bore", "plan view shows both circles; hidden lines in the front view disambiguate them", "medium", "front/top view"),
 ]}

# ------------------------------------------------------------------ ROTATION (Drehteil, revolve about Z)
rp = [[0,63.5],[0,108],[12.3,108],[12.3,87],[47.5,87],[47.5,100],[52.5,105],[78.5,105],[83.5,100],[83.5,78.1],[84.5,78.1],[84.5,67.5],[57.5,67.5],[57.5,0],[49.5,0],[49.5,63.5]]
pcd_pos = [(25*math.cos(math.radians(a)), 25*math.sin(math.radians(a))) for a in range(0,360,60)]
specs["Rotation"] = {
 "part": "Rotation", "drawing": "Rotation_dwg.pdf", "kind": "turned part (revolve about Z) + drilled plate",
 "dimensions": [
  dim("od_flange", 216, "section A-A"), dim("flange_t", 12.3, "section A-A (left)"),
  dim("groove_d", 174, "section A-A"), dim("groove_w", 35.2, "section A-A (left)"),
  dim("od_band", 210, "section A-A"), dim("chamfer_top_outer", "5x45°", "section A-A"),
  dim("lip_od", 156.2, "section A-A / detail B"), dim("lip_h", 1, "detail B '1'"),
  dim("cup_d", 135, "section A-A"), dim("cup_depth", 27, "section A-A (from lip top)"),
  dim("plate_t", 8, "section A-A"), dim("bore_d", 127, "section A-A"), dim("bore_depth", 49.5, "section A-A (right)"),
  dim("height_total", 84.5, "section A-A"), dim("fillet", 6, "section A-A 'R6'"),
  dim("hole_d", 12, "plan Ø12 (6x)"), dim("pcd", 50, "plan Ø50"), dim("centre_hole_d", 25, "plan Ø25", "medium"),
 ],
 "expected": {"bbox": [216,216,84.5], "cylinders": [
   cyl("bore_d127",127,"Z",[0,0]), cyl("cup_d135",135,"Z",[0,0]), cyl("groove_d174",174,"Z",[0,0]),
   cyl("band_d210",210,"Z",[0,0]), cyl("flange_d216",216,"Z",[0,0]), cyl("lip_d156_2",156.2,"Z",[0,0]),
   cyl("centre_hole_d25",25,"Z",[0,0])] +
   [cyl(f"pcd_hole_{i+1}",12,"Z",[round(x,4),round(y,4)]) for i,(x,y) in enumerate(pcd_pos)]},
 "plan": [
  call("create_revolved_body", profile=rp, axis="Z"),
  call("round_circular_edges", edges=[{"radius_at":108,"axial_at":12.3,"size":6}], axis="Z", mode="fillet"),
  call("round_circular_edges", edges=[{"radius_at":87,"axial_at":12.3,"size":5},{"radius_at":87,"axial_at":47.5,"size":5}], axis="Z", mode="fillet"),
  call("add_holes", diameter=25, start=57.5, positions=[[0,0]], axis="Z", direction=-1),
  call("add_holes", diameter=12, start=57.5, bolt_circle={"pcd":50,"count":6,"start_angle":0}, axis="Z", direction=-1),
 ],
 "review": [
  rev("Dimension inconsistency: 36 + 35.2 + 12.3 = 83.5 ≠ 84.5", "outer band top at z=83.5; inner lip (Ø135…Ø156.2) 1 mm higher at z=84.5", "detail B shows a 1 mm step; this reconciles the 84.5 total with the 36/35.2/12.3 chain", "high", "section A-A, detail B"),
  rev("Groove-top outer corner", "5x45° chamfer from (r=100, z=47.5) to (r=105, z=52.5); flat groove roof at z=47.5", "45° chamfer visible at the lower outer corner of the Ø210 band, mirroring the dimensioned top 5x45°; not dimensioned itself", "medium", "section A-A"),
  rev("Groove corner rounds", "R5 at both groove corners (bottom and roof); R6 on the flange top outer edge; flange bottom outer edge sharp", "only one 'R6' leader; groove rounds are undimensioned. R5 (rather than R6) was chosen after the QA comparison with the supplied reference STEP", "high", "section A-A"),
  rev("Bolt-circle angular position", "first Ø12 hole at 0°, six holes at 60° pitch", "plan view shows holes on the horizontal axis; count 6 read from drawing (no '6x' note)", "medium", "plan view"),
  rev("Centre hole Ø25", "Ø25 through the 8 mm plate", "'Ø25' dimension is drawn across the hole pattern; read as the centre hole", "medium", "plan view"),
  rev("Plate hole depth", "through the 8 mm plate (z 49.5…57.5)", "sectional view shows holes through the plate", "low", "section A-A"),
 ]}

# ------------------------------------------------------------------ HAKEN (flat plate with raised boss ring)
T = math.degrees(math.atan2(-36, 48))
ang = math.radians(26.565)
Tpt = [round(159+60*0.8*76/60,4), round(152-60*0.6*76/60,4)]   # tangent point of R16 on R76: centre(159,152) + 76*(0.8,-0.6)
Tpt = [159+76*0.8, 152-76*0.6]
mid_outer = [round(159+76*math.cos(ang),4), round(152+76*math.sin(ang),4)]
hook = {"type":"path","start":[0,0],"segments":[
  {"line":[76,0]},{"line":[76,41]},{"arc":{"mid":[111-35*math.cos(math.radians(45)),41+35*math.sin(math.radians(45))],"to":[111,76]}},
  {"line":[127,76]},{"line":[127,152]},{"arc":{"mid":[159,184],"to":[191,152]}},{"line":[191,116]},
  {"arc":{"mid":[207,100],"to":[round(Tpt[0],4),round(Tpt[1],4)]}},
  {"arc":{"mid":mid_outer,"to":[159,228]}},{"line":[0,228]}]}
lf = {"type":"path","start":[-76,76],"segments":[{"line":[-111,76]},{"arc":{"mid":[-111+35*math.cos(math.radians(45)),111-35*math.sin(math.radians(45))],"to":[-76,111]}}]}
specs["Haken"] = {
 "part": "Haken", "drawing": "Haken_dwg.pdf", "kind": "flat plate (extrude, symmetric about mid-plane) + raised boss",
 "dimensions": [
  dim("hole_d", 76, "plan Ø76 (3x)"), dim("outer_d", 152, "plan Ø152 (lower-left disc and upper disc)"),
  dim("dx_lowerleft_to_centre", 128, "plan '128'"), dim("dy_centre_to_top", 152, "plan '152'"),
  dim("hook_x", 159, "plan '159' from section line"), dim("slot_w", 64, "plan '64'"), dim("slot_depth", 76, "plan '76'"),
  dim("r_hook_outer", 76, "plan R76"), dim("r_tip", 16, "plan R16"), dim("r_fillet", 35, "plan R35 (2x)"),
  dim("t_boss", 30, "section A-A '30'"), dim("t_plate", 20, "section A-A '20'"),
 ],
 "expected": {"bbox": [439,304,30], "cylinders": [
   cyl("hole_ll",76,"Z",[-128,0]), cyl("hole_c",76,"Z",[0,0]), cyl("hole_top",76,"Z",[0,152]),
   cyl("disc_ll_d152",152,"Z",[-128,0]), cyl("disc_top_d152",152,"Z",[0,152]),
   cyl("slot_d64",64,"Z",[159,152]), cyl("hook_outer_d152",152,"Z",[159,152]), cyl("tip_d32",32,"Z",[207,116])]},
 "plan": [
  call("create_extruded_body", z_from=-10, z_to=10, shapes=[
    {"type":"circle","cx":-128,"cy":0,"d":152},{"type":"circle","cx":0,"cy":0,"d":152},{"type":"circle","cx":0,"cy":152,"d":152},
    {"type":"rect","cx":-64,"cy":0,"w":128,"h":152},{"type":"rect","cx":-38,"cy":114,"w":76,"h":76}, lf, hook]),
  call("add_extrusion", shapes=[{"type":"circle","cx":0,"cy":152,"d":152}], z_from=-15, z_to=15, mode="add"),
  call("add_holes", diameter=76, start=15, positions=[[-128,0],[0,0],[0,152]], axis="Z", direction=-1),
 ],
 "review": [
  rev("R16 hook-tip circle position", "centre (207, 116): tangent to the R76 outer arc and to the slot wall x=191", "R16 is dimensioned but its position is not; derived by tangency from the drawing", "medium", "plan, hook tip"),
  rev("Plate thickness is symmetric about the boss mid-plane", "plate −10…+10, boss −15…+15", "A-A shows the 20 mm plate centred on the 30 mm boss (5 mm proud each side)", "low", "section A-A"),
  rev("Extent of the 30 mm boss", "only the Ø152 upper disc is 30 mm thick", "A-A cut line passes through the upper disc; its 30 mm span ends at y = 76 (disc edge)", "medium", "section A-A"),
  rev("Convex round at (76,76)/(−76,76) corners", "R35 concave fillets, centres (111,41) and (−111,111)", "R35 drawn twice; centres derived from tangency with the adjoining edges", "low", "plan"),
  rev("Hook R76 centre", "(159,152), same centre as the slot arc", "R76 arc and U-slot look concentric; the '159' dimension refers to the slot axis", "low", "plan"),
  rev("Right edge of the centre disc", "vertical edge x=76 from y=0 to y=41", "drawn tangent to the R76 disc; start of the R35 arc derived (41 = 76 − 35)", "low", "plan"),
 ]}

os.makedirs("specs", exist_ok=True)
for k, v in specs.items():
    json.dump(v, open(f"specs/{k}.json","w"), indent=1, ensure_ascii=False)
print("wrote", list(specs))
