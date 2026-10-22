"""Run one kernel in a throwaway program: inputs DMA'd from DRAM, outputs DMA'd back."""
import torch

from . import layout as L
from .asm import Builder, Tensor
from .funcsim import FuncSim
from .kernels import Kernels


class KernelRun:
    def __init__(self, stage_bytes: int = 34 * 1024):
        self.b = Builder()
        self.b.begin()
        self.k = Kernels(self.b, stage_bytes)
        self.outputs: dict[str, tuple[Tensor, int]] = {}
        self.sim = None

    def input_bf16(self, name: str, m: torch.Tensor) -> Tensor:
        t = self.b.vmem.tensor(name, m.shape[0], m.shape[1], "bf16")
        addr = self.b.dram.add(name, L.bf16_bytes(m))
        self.b.dma.load_now(t.addr, addr, t.nbytes, block=name)
        return t

    def input_fp8(self, name: str, codes: torch.Tensor) -> Tensor:
        t = self.b.vmem.tensor(name, codes.shape[0], codes.shape[1], "fp8")
        addr = self.b.dram.add(name, L.fp8_bytes(codes))
        self.b.dma.load_now(t.addr, addr, t.nbytes, block=name)
        return t

    def tensor(self, name: str, rows: int, cols: int, kind: str) -> Tensor:
        return self.b.vmem.tensor(name, rows, cols, kind)

    def output(self, t: Tensor) -> None:
        addr = self.b.dram.reserve(t.name + ".out", t.nbytes)
        self.outputs[t.name] = (t, addr)

    def run(self) -> "KernelRun":
        for t, addr in self.outputs.values():
            self.b.dma.store_now(t.addr, addr, t.nbytes)
        self.source = self.b.finish()
        self.sim = FuncSim(self.source, self.b.dram.blobs, self.b.dram.size)
        self.sim.run()
        return self

    def read(self, name: str) -> torch.Tensor:
        t, addr = self.outputs[name]
        data = self.sim.read_dram(addr, t.nbytes)
        if t.kind == "bf16":
            return L.from_bf16_bytes(data, t.rows, t.cols)
        return L.from_fp8_bytes(data, t.rows, t.cols)
