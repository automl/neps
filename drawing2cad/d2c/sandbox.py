"""Run untrusted CadQuery scripts in an isolated child process (no network, CPU/mem/time limits)."""
from __future__ import annotations
import json, os, shutil, subprocess, sys, tempfile
import cadquery as cq

RUNNER = os.path.join(os.path.dirname(__file__), "sandbox_runner.py")


def _unshare_works() -> bool:
    if not shutil.which("unshare"):
        return False
    try:
        return subprocess.run(["unshare", "-n", "true"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False


NETNS = _unshare_works()


def run_script(code: str, timeout_s: int = 30, mem_mb: int = 3072) -> tuple[cq.Shape | None, dict]:
    tmp = tempfile.mkdtemp(prefix="d2c_sbx_")
    try:
        cpath, opath = os.path.join(tmp, "script.py"), os.path.join(tmp, "result.brep")
        open(cpath, "w").write(code)
        cmd = [sys.executable, "-I", RUNNER, cpath, opath, str(timeout_s), str(mem_mb)]
        if NETNS:
            cmd = ["unshare", "-n"] + cmd
        env = {"PATH": os.environ.get("PATH", ""), "HOME": tmp, "PYTHONHASHSEED": "0"}  # no proxies / tokens leak in
        for k in ("PYTHONPATH", "LD_LIBRARY_PATH"):
            if k in os.environ:
                env[k] = os.environ[k]
        info = {"network_isolation": "unshare -n (network namespace)" if NETNS else "in-process socket kill-switch (use docker --network none for hard isolation)",
                "timeout_s": timeout_s, "mem_limit_mb": mem_mb}
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 5, cwd=tmp, env=env)
        except subprocess.TimeoutExpired:
            return None, {**info, "ok": False, "error": f"timeout after {timeout_s}s"}
        last = (p.stdout.strip().splitlines() or ["{}"])[-1]
        try:
            res = json.loads(last)
            if "ok" not in res:
                raise ValueError
        except Exception:
            res = {"ok": False, "error": ((p.stderr or p.stdout)[-500:]) or f"terminated by resource limit (cpu {timeout_s}s / {mem_mb} MB), rc={p.returncode}"}
        if not res.get("ok"):
            return None, {**info, **res}
        shape = cq.Shape.importBrep(opath)
        return shape, {**info, "ok": True}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
