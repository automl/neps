"""QA only (not used while building): compare our STEP with the reference STEP shipped in the sample zip.
Tries the 3 principal axis orientations of the reference, reports the best symmetric difference."""
import os, sys, cadquery as cq
REF = os.environ.get("D2C_REF_DIR")
if not REF:
    sys.exit("Set D2C_REF_DIR to the external Drawing2CAD reference directory.")
for p in sys.argv[1:]:
    a = cq.importers.importStep(f"out/{p}.step").val()
    b0 = cq.importers.importStep(f"{REF}/{p}/{p}.stp").val()
    va, vb = a.Volume(), b0.Volume()
    best = None
    for name, rot in (("as-is", None), ("X+90", ((1,0,0),90)), ("X-90", ((1,0,0),-90)), ("Y+90", ((0,1,0),90)), ("Y-90", ((0,1,0),-90))):
        b = b0 if rot is None else b0.rotate((0,0,0), rot[0], rot[1])
        ba, bb = a.BoundingBox(), b.BoundingBox()
        sh = cq.Vector(ba.center.x - bb.center.x, ba.center.y - bb.center.y, ba.center.z - bb.center.z)
        if sh.Length > 1e-3:
            b = b.translate(sh)
        if any(abs(x - y) > 0.5 for x, y in zip((ba.xlen, ba.ylen, ba.zlen), (b.BoundingBox().xlen, b.BoundingBox().ylen, b.BoundingBox().zlen))):
            continue
        try:
            sd = va + vb - 2 * a.intersect(b).Volume()
        except Exception:
            continue
        if best is None or sd < best[0]:
            best = (sd, name)
    print(f"{p:9s} volume ours={va:12.1f} ref={vb:12.1f} ({100*(va-vb)/vb:+.4f}%)  symmetric diff={best[0]:9.1f} mm3 ({100*best[0]/vb:.4f}% of ref volume, ref orientation {best[1]})" if best else f"{p}: no matching orientation")
