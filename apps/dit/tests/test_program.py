"""The compiled sampler in the functional model against LeRobot's DiT in PyTorch."""
import torch

from ditflow import compiler as C
from ditflow import model as M
from ditflow import quant as Q
from ditflow.funcsim import FuncSim


def rel(a, r):
    return float((a - r).norm() / r.norm())


def test_two_steps_track_pytorch():
    d = M.SMALL
    net = M.build(d, seed=0)
    p = M.params(net, d)
    plan = Q.QuantPlan(p, d, Q.calibrate(p, d))
    x0, cond = Q.random_inputs(d, 7)
    c = C.compile_dit(d, plan, x0, cond, steps=2)
    sim = FuncSim(c.source, c.blobs, 4 << 20)
    sim.run()
    assert sim.dbg == (1, 2)
    out = C.read_outputs(c, sim.read_dram)
    grid = M.time_grid(d)
    x, xq = x0.clone(), x0.clone()
    with torch.no_grad():
        for i in range(2):
            v32 = net(x, grid[i:i + 1], conditioning_vec=cond)
            vfq = M.forward(plan.p, xq, grid[i:i + 1], cond, d, q=plan.q)
            # FP8 alone costs ~5% per velocity; Atlas's BF16 arithmetic may add a little.
            assert rel(vfq, v32) < 0.06
            assert rel(out["v"][i], v32) < rel(vfq, v32) + 0.02
            x, xq = x + 0.01 * v32, xq + 0.01 * vfq
    assert rel(out["x"], x) < 1e-3
