"""The DiT-on-Atlas flow: build, golden, run, report.

    python -m ditflow.flow build  OUT [--preset small] [--steps N] [--seed S] [--input-seed I] [--policy]
                                      [--no-sched]
    python -m ditflow.flow golden OUT                 # functional model, bit-exact with the RTL
    python -m ditflow.flow run    OUT --board [--port TTY] | --eth [IP] | --sim ATLAS_SIM
    python -m ditflow.flow report OUT

build writes OUT/program.S, OUT/bundle.json (program words and DRAM preload, the format
standalone/host/run_tests.py runs), OUT/compiled.pt and OUT/reference.pt: the sampler in FP32
on LeRobot's DiffusionTransformer, and the same sampler with Atlas's FP8 quantization emulated
in FP32 ("fake quant"). The program's basic blocks are list scheduled (sched.py) unless
--no-sched. run loads the bundle into a standalone Atlas (board over UART TSI, or the
Verilator model), reads back the action chunk and every step's velocity, and checks them
bit for bit against the golden.

--policy builds the whole policy instead: the observation encoder (vision.py, two camera crops
and the state in) and the DiT sampler in one program (policy.py). Its references run
specialist_lerobot's PooledVision and LeRobot's sampler; reports add the conditioning vector.
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import torch

from . import compiler as C
from . import model as M
from . import quant as Q

HOST = Path(__file__).resolve().parents[3] / "standalone" / "host"
TTY = "/dev/serial/by-id/usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_00810C93-if01-port0"


def assemble(source: str) -> list[int]:
    from npu_model.util.converter import input_to_program
    return [w & 0xFFFF_FFFF for w in input_to_program(io.StringIO(source)).assemble()]


def preload(blobs) -> list:
    out = []
    for _, addr, data in blobs:
        data = data + bytes(-len(data) % 4)
        out.append([addr, list(torch.frombuffer(bytearray(data), dtype=torch.int32).numpy().astype("uint32").tolist())])
    return out


def build(out: Path, preset: str, steps: int | None, seed: int, input_seed: int, trace: bool,
          schedule: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    d = M.PRESETS[preset]
    steps = steps or d.steps
    net = M.build(d, seed)
    p = M.params(net, d)
    plan = Q.QuantPlan(p, d, Q.calibrate(p, d))
    x0, cond = Q.random_inputs(d, input_seed)
    c = C.compile_dit(d, plan, x0, cond, steps=steps, trace=trace, schedule=schedule)
    words = assemble(c.source)
    assert len(words) <= 32768, f"{len(words)} instructions do not fit the 128 KiB IMEM"
    (out / "program.S").write_text(c.source)
    bundle = {"name": f"dit_{preset}_{steps}", "variant": "", "program": words, "preload": preload(c.blobs),
              "check": [], "perf": {}, "timeout": None}
    (out / "bundle.json").write_text(json.dumps(bundle))
    c.manifest.update({"preset": preset, "seed": seed, "input_seed": input_seed, "instructions": len(words)})
    (out / "manifest.json").write_text(json.dumps(c.manifest, indent=1))
    torch.save({"compiled": c, "x0": x0, "cond": cond}, out / "compiled.pt")

    with torch.no_grad():
        x_ref, v_ref = _partial(net, x0, cond, d, steps)
        x_fq, v_fq = _partial_fq(plan, x0, cond, d, steps)
        if steps == d.steps:                             # LeRobot's own sampler gives the same chunk
            from lerobot.policies.multi_task_dit.modeling_multi_task_dit import FlowMatchingObjective
            obj = FlowMatchingObjective(M.lerobot_config(d), d.action_dim, d.horizon)
            x_lr = obj._euler_integrate(net, x0, M.time_grid(d), cond)
            assert float((x_lr - x_ref).abs().max()) < 1e-5
            x_ref = x_lr
    torch.save({"x_fp32": x_ref, "v_fp32": v_ref, "x_fq": x_fq, "v_fq": v_fq, "x0": x0, "cond": cond},
               out / "reference.pt")
    print(f"built {out}: {len(words)} instructions, DRAM {c.manifest['dram_bytes'] / 2**20:.2f} MiB, "
          f"VMEM {c.manifest['vmem_bytes'] / 1024:.0f} KiB, {steps} steps")


def build_policy(out: Path, steps: int | None, seed: int, input_seed: int, trace: bool,
                 schedule: bool = True) -> None:
    from . import policy as P
    from . import vision as V
    out.mkdir(parents=True, exist_ok=True)
    d, v = M.SMALL, V.SMALL
    steps = steps or d.steps
    net, enc = M.build(d, seed), V.build(v, seed)
    dp, vp = M.params(net, d), V.params(enc, v)
    plans = P.PolicyPlans(vp, dp, v, d)
    views, state, x0 = P.random_observation(v, d, input_seed)
    prog = P.PolicyProgram(v, plans.vision, d, plans.dit, views, state, x0, steps=steps, trace=trace,
                           schedule=schedule)
    c = prog.compile()
    words = assemble(c.source)
    assert len(words) <= 32768, f"{len(words)} instructions do not fit the 128 KiB IMEM"
    (out / "program.S").write_text(c.source)
    bundle = {"name": f"policy_{steps}", "variant": "", "program": words, "preload": preload(c.blobs),
              "check": [], "perf": {}, "timeout": None}
    (out / "bundle.json").write_text(json.dumps(bundle))
    c.manifest.update({"policy": True, "seed": seed, "input_seed": input_seed, "instructions": len(words),
                       "vision_scales": plans.vision.to_dict()})
    (out / "manifest.json").write_text(json.dumps(c.manifest, indent=1))
    torch.save({"compiled": c, "x0": x0, "state": state, "views": views}, out / "compiled.pt")
    with torch.no_grad():
        cond_ref, x_ref, v_ref = P.reference(enc, net, v, d, views, state, x0, steps)
        cond_fq, x_fq, v_fq = P.fake_quant(plans, v, d, views, state, x0, steps)
    torch.save({"x_fp32": x_ref, "v_fp32": v_ref, "x_fq": x_fq, "v_fq": v_fq, "x0": x0,
                "cond_fp32": cond_ref, "cond_fq": cond_fq}, out / "reference.pt")
    print(f"built {out}: {len(words)} instructions, DRAM {c.manifest['dram_bytes'] / 2**20:.2f} MiB, "
          f"{steps} steps, encoder + DiT")


def _policy_cond(c, saved, read):
    from . import policy as P
    from . import vision as V
    if not c.manifest.get("vision"):
        return {}
    v = V.VisionDims(**{k: (tuple(tuple(x) for x in val) if k == "views" else val)
                        for k, val in c.manifest["vision"].items()})
    cond = P.read_cond(c, v, saved["state"], read)
    bits = b"".join(read(a, 2048) for a in c.manifest["outputs"]["pooled"])
    return {"cond": cond, "pooled_bits": bits}


def _partial(net, x0, cond, d, steps):
    grid, x, vs = M.time_grid(d), x0.clone(), []
    for i in range(steps):
        v = net(x, torch.full((1,), grid[i].item()), conditioning_vec=cond)
        vs.append(v)
        x = x + (grid[i + 1] - grid[i]).item() * v
    return x, vs


def _partial_fq(plan, x0, cond, d, steps):
    grid, x, vs = M.time_grid(d), x0.clone(), []
    for i in range(steps):
        v = M.forward(plan.p, x, grid[i:i + 1], cond, d, q=plan.q)
        vs.append(v)
        x = x + (grid[i + 1] - grid[i]).item() * v
    return x, vs


def golden(out: Path) -> None:
    from .funcsim import FuncSim
    saved = torch.load(out / "compiled.pt", weights_only=False)
    c = saved["compiled"]
    sim = FuncSim(c.source, c.blobs, 64 << 20)
    t0 = time.time()
    sim.run(progress=lambda n, s, st: print(f"  {n} instructions, {s:.0f} s, step {st.read_csrf(0xC11)}", flush=True))
    res = C.read_outputs(c, sim.read_dram)
    res.update(_policy_cond(c, saved, sim.read_dram))
    res.update({"instructions": sim.executed, "seconds": time.time() - t0})
    torch.save(res, out / "golden.pt")
    print(f"golden: {sim.executed} instructions in {time.time() - t0:.0f} s, DBG {sim.dbg}")
    # The golden as the bundle's DRAM checks: run_tests.py then verifies a run bit for bit.
    bundle = json.loads((out / "bundle.json").read_text())
    outs = c.manifest["outputs"]
    bundle["check"] = [[addr, list(torch.frombuffer(bytearray(res[key]), dtype=torch.int32).numpy().astype("uint32").tolist())]
                       for key, addr in (("x_hi_bits", outs["x_hi"]), ("x_lo_bits", outs["x_lo"]), ("trace_bits", outs["trace"]))
                       if addr is not None]
    for i, addr in enumerate(outs.get("pooled", []) if "pooled_bits" in res else []):
        tile = res["pooled_bits"][2048 * i:2048 * (i + 1)]
        bundle["check"].append([addr, list(torch.frombuffer(bytearray(tile), dtype=torch.int32).numpy().astype("uint32").tolist())])
    (out / "bundle_checked.json").write_text(json.dumps(bundle))


def run(out: Path, board: bool, port: str | None, sim_binary: str | None, budget: float | None,
        eth: str | None = None) -> None:
    sys.path.insert(0, str(HOST))
    import atlas_host as h
    saved = torch.load(out / "compiled.pt", weights_only=False)
    c = saved["compiled"]
    bundle = json.loads((out / "bundle.json").read_text())
    if eth is not None:
        link, target, budget = h.EthLink(eth or h.ETH_IP_DEFAULT), "board", budget or 600.0
    elif board:
        link, target, budget = h.TsiLink(port=port or TTY), "board", budget or 600.0
    else:
        link, target, budget = h.SimLink(sim_binary), "sim", budget or 2e9
    atlas = h.Atlas(link)
    t0 = time.time()
    r = atlas.run(bundle, budget)
    wall = time.time() - t0
    print(f"{target}: {r.status} {r.detail} cycles={r.cycles} instret={r.instret} DBG1={r.dbg1} ({wall:.0f} s, features {atlas.features:#x})")
    if r.status != "PASS":
        raise SystemExit(1)

    def read(addr, n):
        words = link.read_words(atlas.dram(addr), n // 4)
        return torch.tensor(words, dtype=torch.int64).to(torch.int32).numpy().tobytes()
    res = C.read_outputs(c, read)
    res.update(_policy_cond(c, saved, read))
    res.update({"cycles": r.cycles, "instret": r.instret, "wall_s": wall, "features": atlas.features,
                "run_s": r.run_s, "perf": r.perf, "util": h.utilization(r.perf)})
    link.close()
    torch.save(res, out / f"{target}.pt")
    report(out)


def metrics(a: torch.Tensor, r: torch.Tensor) -> dict:
    a, r = a.flatten().double(), r.flatten().double()
    e = a - r
    return {"rel_l2": float(e.norm() / r.norm()), "max_abs": float(e.abs().max()),
            "cos": float(torch.nn.functional.cosine_similarity(a, r, 0))}


def report(out: Path) -> dict:
    ref = torch.load(out / "reference.pt", weights_only=False)
    rep = {"fakequant_vs_fp32": {"x": metrics(ref["x_fq"], ref["x_fp32"]),
                                 "v_step0": metrics(ref["v_fq"][0], ref["v_fp32"][0])}}
    if "cond_fp32" in ref:
        rep["fakequant_vs_fp32"]["cond"] = metrics(ref["cond_fq"], ref["cond_fp32"])
    gold = torch.load(out / "golden.pt", weights_only=False) if (out / "golden.pt").exists() else None
    for name in ("golden", "sim", "board"):
        path = out / f"{name}.pt"
        if not path.exists():
            continue
        res = torch.load(path, weights_only=False)
        entry = {"x_vs_fp32": metrics(res["x"], ref["x_fp32"]), "x_vs_fakequant": metrics(res["x"], ref["x_fq"])}
        if "cond" in res and "cond_fp32" in ref:
            entry["cond_vs_fp32"] = metrics(res["cond"], ref["cond_fp32"])
            entry["cond_vs_fakequant"] = metrics(res["cond"], ref["cond_fq"])
        if "v" in res:
            per = [metrics(v, r)["rel_l2"] for v, r in zip(res["v"], ref["v_fp32"])]
            entry["v_vs_fp32_rel_l2"] = {"mean": sum(per) / len(per), "max": max(per), "first": per[0], "last": per[-1]}
        if gold is not None and name != "golden":
            same = {k: res.get(k) == gold.get(k) for k in ("x_hi_bits", "x_lo_bits", "trace_bits", "pooled_bits") if k in gold}
            entry["bit_exact_vs_golden"] = same
        for k in ("cycles", "instret", "wall_s", "run_s", "instructions", "perf", "util"):
            if k in res:
                entry[k] = res[k]
        rep[name] = entry
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out", type=Path)
    b.add_argument("--preset", default="small", choices=sorted(M.PRESETS))
    b.add_argument("--steps", type=int)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--input-seed", type=int, default=7)
    b.add_argument("--no-trace", action="store_true")
    b.add_argument("--policy", action="store_true", help="encoder + DiT (the whole policy)")
    b.add_argument("--no-sched", action="store_true", help="keep the program order (no sched.py)")
    g = sub.add_parser("golden")
    g.add_argument("out", type=Path)
    r = sub.add_parser("run")
    r.add_argument("out", type=Path)
    t = r.add_mutually_exclusive_group(required=True)
    t.add_argument("--board", action="store_true")
    t.add_argument("--eth", nargs="?", const="", metavar="IP", help="the board over TSI over UDP")
    t.add_argument("--sim", metavar="ATLAS_SIM")
    r.add_argument("--port")
    r.add_argument("--budget", type=float, help="seconds on the board, cycles in simulation")
    p = sub.add_parser("report")
    p.add_argument("out", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "build" and a.policy:
        build_policy(a.out, a.steps, a.seed, a.input_seed, not a.no_trace, not a.no_sched)
    elif a.cmd == "build":
        build(a.out, a.preset, a.steps, a.seed, a.input_seed, not a.no_trace, not a.no_sched)
    elif a.cmd == "golden":
        golden(a.out)
    elif a.cmd == "run":
        run(a.out, a.board, a.port, a.sim, a.budget, a.eth)
    else:
        report(a.out)


if __name__ == "__main__":
    main()
