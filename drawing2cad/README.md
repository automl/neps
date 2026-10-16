# Drawing2CAD – local MCP server for 2D drawing → 3D STEP

Pipeline: **drawing → structured JSON (values + where found + confidence + review flags) → MCP tools build the model → validation → STEP export + re-import check.**

## Run
Run these commands from `drawing2cad/` in a Python 3.11 environment. The repository's
`pyproject.toml` includes the CAD/MCP dependencies; `requirements.txt` is also kept
for standalone and Docker installs (without the NePS training stack).

```bash
pip install -r requirements.txt
python make_specs.py                 # writes specs/*.json (the drawing->JSON step, here authored by the LLM that read the PDFs)
python run_specs.py                  # drives server.py over real MCP stdio, writes out/*.step, *.report.json, *.review.md
D2C_REF_DIR=/external/path/Drawing2CAD python compare_reference.py Welle Fuss Rotation Haken
python tests_negative.py             # validation catches injected errors; sandbox blocks bad scripts
# Docker (runtime offline):  docker build -t drawing2cad .  &&  docker run -i --rm --network none -v $PWD/out:/app/out drawing2cad
```
Sample drawings and reference STEPs are not included. `D2C_REF_DIR` must point to
an external directory containing `<Part>/<Part>.stp`; comparison is optional QA.
Specs and `out/*.review.md` are versioned; generated STEP and JSON reports are
ignored. Regenerate them with `python run_specs.py` from this directory.

Build Docker from this directory, or from the repository root with
`docker build -t drawing2cad drawing2cad`. No repository Dockerfile was replaced.
The image build needs network access; `--network none` applies to runtime only.

MCP client config: `{"command":"python","args":["/path/server.py"]}` (or the `docker run -i --rm --network none …` command).

## Tools (11)
`start_part` (expected dimensions from the drawing) · `create_extruded_body` · `create_revolved_body` · `add_extrusion` (boss/pocket/slot/keyway) · `add_holes` (single, list, bolt circle, counterbore) · `round_circular_edges` (fillet/chamfer by geometry) · `add_review_item` · `validate_model` · `export_step` · `get_state` · fallback `run_sandboxed_script`.
Model = replayable feature history, so a parameter change is a rebuild (basis for the "modify existing models" extension).

## Validation (every export, JSON out)
solid single/valid/closed/no free edges/volume>0 · overall size, every hole/bore/shaft diameter **and position** vs drawing values · STEP re-imported: loads, valid, same volume and bbox.

## Sandbox
Scripts run in a child process: AST allow-list (cadquery/math/numpy only, no dunder, no open/exec), CPU+memory+file limits, wall-clock timeout, network namespace via `unshare -n` when available plus socket kill-switch, clean env; Docker `--network none` is the hard boundary.

## Not done / limits
STEP AP242 + semantic PMI (stretch), NX open test (no NX here; STEP re-imports cleanly in OpenCascade), automatic drawing parsing (the PDFs have no text layer; the LLM reads them), 4 of 9 samples (Trägerplatte, Befestigung, Maschinenteil, Dreistern: inclined faces / 3D bosses; Flansch: ring+web details).

## Integration verification

Verified with Python 3.11.17, CadQuery 2.8.0, and MCP 1.30.0:

- `python make_specs.py && python run_specs.py`: 4/4 validation and 4/4 STEP
  re-import PASS on macOS arm64 (65/65 integrity/dimension checks; 28/28 re-import checks).
- `docker build -t drawing2cad drawing2cad` from the repository root: PASS
  (Linux arm64). Image runtime with `--network none`: 4/4 validation and re-import PASS.
- `python tests_negative.py`: macOS FAIL after the first OK line; sandbox setup
  returns `ValueError: current limit exceeds maximum limit`. In Docker, the first
  run failed the runaway-loop assertion; a diagnostic run rejected `os`/`open`
  and killed the loop by resource limit, and a full rerun printed both OK lines.
  Sandbox checks therefore remain platform-dependent and potentially intermittent.
- Repository `python -m pytest -m ''`: 3,972 passed, 288 expected failures, no failures.
- `D2C_REF_DIR` missing/configured behavior checked. Actual reference comparison
  was not run because external sample assets are not included.

No NX open test or full Python/OS CI matrix was run. The Dockerfile was built and
its model runner tested, not merely inspected. Geometry and sandbox checks were
not changed to obtain passing results.

The Python AST restrictions and socket kill-switch are not a security boundary
for hostile Python. Use an isolated container with `--network none` for untrusted
scripts; do not expose this local prototype as a public service.
