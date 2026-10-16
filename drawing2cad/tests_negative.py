"""Self-check: validation must FAIL on wrong geometry and the sandbox must contain bad scripts."""
import sys, json
from d2c.session import PartSession
spec = json.load(open("specs/Fuss.json"))
s = PartSession(); s.start_part("Fuss_bad", spec["expected"])
for st in spec["plan"]:
    a = dict(st["args"])
    if st["tool"] == "add_holes" and a.get("diameter") == 19 and len(a.get("positions", [])) == 4:
        a["diameter"] = 17          # inject error: wrong hole diameter
    f = {"create_extruded_body": "extruded_body", "add_extrusion": "extrusion", "add_holes": "holes", "round_circular_edges": "round_edges"}[st["tool"]]
    if f == "round_edges": a = {"edges": a["edges"], "axis": a["axis"], "mode": a["mode"]}
    s.add_feature({"kind": f, **a})
v = s.validate()
failed = [c["id"] for c in v["failed"]]
assert not v["pass"] and any(i.startswith("hole_") for i in failed), failed
print("OK validation catches injected error ->", failed)
assert "not allowed" in s.run_script("import os\nresult=os.getcwd()")["error"]
assert "not allowed" in s.run_script("result=open('/etc/passwd').read()")["error"]
assert "resource limit" in s.add_feature({"kind": "script", "code": "while True: pass", "timeout_s": 2})["error"]
print("OK sandbox blocks os/open and kills runaway loops")
