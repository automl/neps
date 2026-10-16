"""Stateful part session shared by the MCP server and the CLI. Every public method returns a JSON-able dict."""
from __future__ import annotations
import json, os, re
import cadquery as cq
from . import kernel, validate, sandbox

OUT_DIR = os.environ.get("D2C_OUT_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "out"))


def _safe_name(n: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", n)


class PartSession:
    def __init__(self):
        self.reset("unnamed")

    # ------------------------------------------------------------ state
    def reset(self, name: str, expected: dict | None = None, units: str = "mm"):
        self.name, self.units = name, units
        self.features: list[dict] = []
        self.expected = expected or {}
        self.review: list[dict] = []
        self.shape: cq.Shape | None = None
        self.last_validation: dict | None = None
        self.last_export: dict | None = None

    def _rebuild(self, features: list[dict]) -> cq.Shape | None:
        solid = None
        for f in features:
            if f["kind"] == "script":
                shp, info = sandbox.run_script(f["code"], f.get("timeout_s", 30))
                if shp is None:
                    raise RuntimeError(f"sandbox: {info.get('error')}")
                mode = f.get("mode", "replace")
                if mode == "replace" or solid is None:
                    solid = shp
                else:
                    solid = (solid.fuse(shp) if mode == "add" else solid.cut(shp)).clean()
            else:
                solid = kernel.apply_feature(solid, f)
        return solid

    def _summary(self) -> dict:
        if self.shape is None:
            return {}
        bb = self.shape.BoundingBox()
        return {"volume_mm3": round(self.shape.Volume(), 3), "n_faces": len(self.shape.Faces()),
                "bbox_min": [round(bb.xmin, 3), round(bb.ymin, 3), round(bb.zmin, 3)],
                "bbox_size": [round(bb.xlen, 3), round(bb.ylen, 3), round(bb.zlen, 3)]}

    def add_feature(self, f: dict) -> dict:
        try:
            shape = self._rebuild(self.features + [f])
        except Exception as e:  # noqa
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "hint": "feature was NOT added; fix parameters and retry",
                    "n_features": len(self.features)}
        self.features.append(f)
        self.shape = shape
        return {"ok": True, "feature_index": len(self.features) - 1, "n_features": len(self.features), **self._summary()}

    # ------------------------------------------------------------ tools
    def start_part(self, name: str, expected: dict | None = None, units: str = "mm") -> dict:
        self.reset(name, expected, units)
        return {"ok": True, "part": name, "expected_checks": len((expected or {}).get("cylinders", [])) + len((expected or {}).get("bbox", []))}

    def add_review_item(self, item: str, assumed_value: str, reason: str, severity: str = "medium", source: str = "") -> dict:
        self.review.append({"item": item, "assumed_value": assumed_value, "reason": reason, "severity": severity, "drawing_source": source})
        return {"ok": True, "n_review_items": len(self.review)}

    def run_script(self, code: str, mode: str = "replace", timeout_s: int = 30) -> dict:
        r = self.add_feature({"kind": "script", "code": code, "mode": mode, "timeout_s": timeout_s})
        r["note"] = "fallback path: free code ran in sandbox; prefer structured tools"
        if r.get("ok"):
            self.review.append({"item": "free-form script used", "assumed_value": "see feature history", "reason": "geometry not covered by structured tools", "severity": "low", "drawing_source": ""})
        return r

    def validate(self) -> dict:
        if self.shape is None:
            return {"ok": False, "error": "no geometry yet"}
        integ = validate.integrity(self.shape)
        dims = validate.compare_dimensions(self.shape, self.expected)
        self.last_validation = {"part": self.name, "integrity": integ, "dimensions": dims, "pass": integ["pass"] and dims["pass"],
                                "failed": [c for c in integ["checks"] + dims["checks"] if not c["pass"]]}
        return self.last_validation

    def export_step(self, filename: str | None = None) -> dict:
        v = self.validate()
        if self.shape is None:
            return v
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, _safe_name(filename or self.name) + ("" if (filename or "").lower().endswith((".step", ".stp")) else ".step"))
        cq.exporters.export(cq.Workplane(obj=self.shape), path, exportType="STEP")
        rt = validate.reimport_step(path, self.shape)
        self.last_export = {"path": path, "pre_export_validation_pass": v["pass"], "reimport": rt,
                            "pass": v["pass"] and rt["pass"]}
        return self.last_export

    def state(self) -> dict:
        return {"part": self.name, "units": self.units, "n_features": len(self.features), "features": self.features,
                "review_items": self.review, **self._summary()}

    def report(self) -> dict:
        return {"part": self.name, "units": self.units, "geometry": self._summary(), "features": self.features,
                "validation": self.last_validation, "export": self.last_export, "review_list": self.review}
