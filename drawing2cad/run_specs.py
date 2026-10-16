"""Drive the MCP server over stdio with the JSON plans (stand-in for an LLM agent) and write STEP + reports.
usage: python run_specs.py [Part ...]"""
import asyncio, json, os, sys, glob
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def unwrap(res):
    if getattr(res, "structuredContent", None):
        sc = res.structuredContent
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    return json.loads(res.content[0].text)


async def build(session, spec):
    name = spec["part"]
    log = []
    r = unwrap(await session.call_tool("start_part", {"name": name, "expected": spec["expected"]}))
    for step in spec["plan"]:
        r = unwrap(await session.call_tool(step["tool"], step["args"]))
        log.append({"tool": step["tool"], "ok": r.get("ok"), "error": r.get("error")})
        if not r.get("ok"):
            print(f"  ! {name}: {step['tool']} failed -> {r.get('error')}")
            break
    for it in spec["review"]:
        await session.call_tool("add_review_item", {"item": it["item"], "assumed_value": it["assumed_value"], "reason": it["reason"],
                                                    "severity": it["severity"], "drawing_source": it["drawing_source"]})
    v = unwrap(await session.call_tool("validate_model", {}))
    # self-correction hook: an agent would read v["failed"] here and issue corrective tool calls
    e = unwrap(await session.call_tool("export_step", {"filename": name}))
    st = unwrap(await session.call_tool("get_state", {}))
    rep = {"part": name, "drawing": spec["drawing"], "build_log": log, "geometry": {k: st[k] for k in ("volume_mm3", "bbox_size", "n_faces") if k in st},
           "validation": v, "export": e, "review_list": st["review_items"], "dimension_provenance": spec["dimensions"]}
    json.dump(rep, open(os.path.join(OUT, f"{name}.report.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(OUT, f"{name}.review.md"), "w") as f:
        f.write(f"# Review list – {name}\n\nSource: `{spec['drawing']}`. Every item below is a value that was **not dimensioned** or **ambiguous** in the drawing; "
                f"the model uses the assumption shown.\n\n| # | Severity | Item | Assumed value | Reason | Where |\n|---|---|---|---|---|---|\n")
        for i, it in enumerate(st["review_items"], 1):
            f.write(f"| {i} | {it['severity']} | {it['item']} | {it['assumed_value']} | {it['reason']} | {it['drawing_source']} |\n")
    return rep


async def main(parts):
    os.makedirs(OUT, exist_ok=True)
    params = StdioServerParameters(command=sys.executable, args=[os.path.join(HERE, "server.py")], env={**os.environ, "D2C_OUT_DIR": OUT})
    async with stdio_client(params) as (rd, wr):
        async with ClientSession(rd, wr) as session:
            await session.initialize()
            tools = [t.name for t in (await session.list_tools()).tools]
            print("MCP tools:", tools)
            for p in parts:
                spec = json.load(open(os.path.join(HERE, "specs", f"{p}.json")))
                rep = await build(session, spec)
                v, e = rep["validation"], rep["export"]
                print(f"{p:10s} vol={rep['geometry'].get('volume_mm3')} validation={'PASS' if v.get('pass') else 'FAIL'} "
                      f"({sum(c['pass'] for c in v['integrity']['checks']+v['dimensions']['checks'])}/{len(v['integrity']['checks']+v['dimensions']['checks'])}) "
                      f"step_reimport={'PASS' if e['reimport']['pass'] else 'FAIL'} review_items={len(rep['review_list'])}")
                for c in v.get("failed", []):
                    print("    FAIL", c)

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or [os.path.basename(f)[:-5] for f in sorted(glob.glob(os.path.join(HERE, "specs", "*.json")))]))
