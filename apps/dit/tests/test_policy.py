"""The observation encoder and the whole policy program against specialist_lerobot / LeRobot."""
import torch

from ditflow import compiler as C
from ditflow import model as M
from ditflow import policy as P
from ditflow import vision as V
from ditflow.funcsim import FuncSim


def rel(a, r):
    return float((a - r).norm() / r.norm())


def test_vision_replica_matches_pooled_vision():
    v = V.SMALL
    enc = V.build(v, seed=0)
    p = V.params(enc, v)
    with torch.no_grad():
        for img in V.random_views(v, 5):
            assert float((enc(img) - V.forward(p, img, v)).abs().max()) < 1e-5


def test_policy_one_step():
    d, v = M.SMALL, V.SMALL
    net, enc = M.build(d, 0), V.build(v, 0)
    dp, vp = M.params(net, d), V.params(enc, v)
    plans = P.PolicyPlans(vp, dp, v, d)
    views, state, x0 = P.random_observation(v, d, 77)
    c = P.PolicyProgram(v, plans.vision, d, plans.dit, views, state, x0, steps=1).compile()
    sim = FuncSim(c.source, c.blobs, 64 << 20)
    sim.run()
    assert sim.dbg == (1, 1)
    cond = P.read_cond(c, v, state, sim.read_dram)
    cond_ref, _, v_ref = P.reference(enc, net, v, d, views, state, x0, steps=1)
    cond_fq, _, v_fq = P.fake_quant(plans, v, d, views, state, x0, steps=1)
    pooled = slice(v.state_dim, None)
    # FP8 costs ~8% on the pooled tokens and ~5% on a velocity; Atlas's BF16 may add a little.
    assert rel(cond[:, pooled], cond_ref[:, pooled]) < rel(cond_fq[:, pooled], cond_ref[:, pooled]) + 0.02
    out = C.read_outputs(c, sim.read_dram)
    assert rel(out["v"][0], v_ref[0]) < rel(v_fq[0], v_ref[0]) + 0.02
