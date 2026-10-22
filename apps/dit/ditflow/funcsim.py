"""Functional Atlas: npu_model's instruction semantics, run in program order.

On the interlocked core every instruction behaves as if the ones before it had finished, so
program order is the architectural semantics, apart from DMA: a transfer completes at its
DMA.WAIT, which a correct program (asm.py) always issues before touching the data. Here a DMA
completes when it issues. Arithmetic is FastRtlNumerics, bit-exact with the RTL, so the DRAM
this produces is what the hardware must produce.
"""
import io
import time

import torch

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.hardware.arch_state import ArchState
from npu_model.util.converter import input_to_program

from .asm import CSR_DBG0, CSR_DBG1
from .numerics import FastRtlNumerics

CONTROL = {"BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU", "JAL", "JALR"}


class FuncSim:
    def __init__(self, source: str, dram_blobs, dram_size: int):
        self.instrs = list(input_to_program(io.StringIO(source)).instructions)
        cfg = DefaultHardwareConfig().arch_state_config
        cfg.numerics = "rtl"
        self.state = ArchState(cfg)
        self.state.math = FastRtlNumerics()
        self.dram_size = dram_size
        for _, addr, data in dram_blobs:
            self.state.write_dram(addr, torch.frombuffer(bytearray(data), dtype=torch.uint8))
        self.executed = 0
        self.counts: dict[str, int] = {}

    def run(self, max_instructions: int = 1 << 31, progress=None) -> None:
        s, instrs = self.state, self.instrs
        pc, t0 = 0, time.time()
        while not s.halted:
            assert 0 <= pc < len(instrs), f"pc {pc} outside the program"
            target = self._step(pc)
            if target is not None:
                slot = instrs[pc + 1]
                assert type(slot).__name__ not in CONTROL, f"control flow in the delay slot at {pc + 1}"
                self._step(pc + 1)
                pc = target
            else:
                pc += 1
            if self.executed >= max_instructions:
                raise RuntimeError(f"no ECALL after {self.executed} instructions")
            if progress and self.executed % 200000 == 0:
                progress(self.executed, time.time() - t0, s)

    def _step(self, pc: int):
        s = self.state
        ins = self.instrs[pc]
        s.execute_pc = pc
        s.redirect_requested = False
        s.begin_csr_cycle()
        ins.exec(s)
        s.finish_csr_cycle(True)
        self.executed += 1
        name = type(ins).__name__
        self.counts[name] = self.counts.get(name, 0) + 1
        return s.npc if s.redirect_requested else None

    def read_dram(self, addr: int, nbytes: int) -> bytes:
        return bytes(self.state.read_dram(addr, nbytes).tolist())

    def csr(self, addr: int) -> int:
        return self.state.read_csrf(addr)

    @property
    def dbg(self) -> tuple[int, int]:
        return self.csr(CSR_DBG0), self.csr(CSR_DBG1)
