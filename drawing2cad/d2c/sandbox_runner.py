"""Executed in a child process by sandbox.py: restricted exec of a CadQuery script.
Contract: the script assigns the final CadQuery Workplane/Shape to `result`.
"""
import ast, sys, json, resource, builtins

ALLOWED_IMPORTS = {"cadquery", "math", "numpy", "typing"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals", "locals",
                   "vars", "getattr", "setattr", "delattr", "memoryview", "exit", "quit"}


def check_ast(code: str):
    tree = ast.parse(code)
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ValueError(f"import '{a.name}' not allowed")
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise ValueError(f"import from '{n.module}' not allowed")
        elif isinstance(n, ast.Attribute) and n.attr.startswith("__"):
            raise ValueError(f"dunder attribute '{n.attr}' not allowed")
        elif isinstance(n, ast.Name) and (n.id in FORBIDDEN_NAMES or (n.id.startswith("__") and n.id != "__name__")):
            raise ValueError(f"name '{n.id}' not allowed")


def main():
    code_path, out_path, cpu_s, mem_mb = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    resource.setrlimit(resource.RLIMIT_FSIZE, (200 << 20, 200 << 20))
    code = open(code_path).read()
    check_ast(code)
    import cadquery as cq, math  # noqa
    # software network kill-switch (defence in depth; the parent also tries unshare -n / docker --network none)
    import socket
    def _deny(*a, **k): raise PermissionError("network disabled in sandbox")
    socket.socket = _deny; socket.create_connection = _deny; socket.getaddrinfo = _deny
    safe = {k: getattr(builtins, k) for k in dir(builtins) if k not in FORBIDDEN_NAMES and not k.startswith("__")}
    real_import = builtins.__import__
    def guarded_import(name, *a, **k):
        if name.split(".")[0] not in ALLOWED_IMPORTS | {"OCP"} and not name.startswith("cadquery"):
            raise ImportError(f"import '{name}' not allowed")
        return real_import(name, *a, **k)
    safe["__import__"] = guarded_import
    env = {"__builtins__": safe, "cq": cq, "math": math}
    exec(compile(code, "<script>", "exec"), env)
    res = env.get("result")
    if res is None:
        raise ValueError("script must assign the final geometry to `result`")
    shape = res.val() if hasattr(res, "val") else res
    if hasattr(res, "vals") and len(res.vals()) > 1:
        shape = cq.Compound.makeCompound(res.vals())
    shape.exportBrep(out_path)
    print(json.dumps({"ok": True}))


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:  # noqa
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        sys.exit(2)
