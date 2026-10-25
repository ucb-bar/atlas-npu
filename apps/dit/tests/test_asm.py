"""The assembly builder (asm.py): loop ends."""
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
