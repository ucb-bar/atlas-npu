"""The assembly builder (asm.py): loop ends and the row count in loops."""
import pytest

from ditflow.asm import Builder
from ditflow.funcsim import FuncSim


@pytest.mark.parametrize("body, end", [(1023, ["ADDI", "BNE"]), (1024, ["ADDI", "BEQ", "JAL"])])
def test_loop_end_reaches_the_top(body, end):
    """A loop ends with ADDI; BNE while the BNE reaches the top of the body (1,024 instructions
    back), else with ADDI; BEQ over a JAL. Both run the body n times."""
    b = Builder(schedule=False)
    b.begin()
    with b.loop(3):
        for _ in range(body):
            b.op("ADDI x19, x19, 1")
    source = b.finish()
    ops = [s.split()[0] for s in source.splitlines() if s.startswith("    ") and s.split()[0] != "#"]
    tail = ops.index("CSRRW") - len(end) - 1     # the loop end, then LI x10 and the DBG0 write
    assert ops[tail:tail + len(end)] == end
    sim = FuncSim(source, b.dram.blobs, 4 << 20)
    sim.run()
    assert sim.state.read_xrf(19) == 3 * body


def test_loop_body_sets_m():
    """A loop that runs at M < 32 sets M at the top of its body, where sched.py costs at 32 rows,
    and a body that changes M sets it back before the next pass."""
    b = Builder(schedule=False)
    b.begin()
    b.vsetm(3)
    with b.loop(2):
        b.op("vexp.bf16 m0, m2")
        b.vsetm(32)
        b.op("vexp.bf16 m4, m2")
    lines = [s.strip() for s in b.prog.lines]
    top = next(i for i, s in enumerate(lines) if s.startswith("loop_"))
    assert lines[top + 1:top + 8] == ["vsetm 3", "vexp.bf16 m0, m2", "vsetm 32", "vexp.bf16 m4, m2",
                                      "ADDI x11, x11, -1", "vsetm 3", "BNE x11, x0, " + lines[top][:-1]]
