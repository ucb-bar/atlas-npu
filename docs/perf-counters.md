# Performance counters

The CSR block (`src/main/scala/diplomatic/memory/CSRFile.scala`) has 16 counters of 64 bits for
the host. They follow RISC-V's `mhpmcounter` / `mcountinhibit` scheme, but on MMIO, since the
host is the one that reads them. In Chipyard the host core reaches them on the periphery bus. In
the standalone SoC the host computer reaches them over UART TSI.

| Offset (CSR block, `0x0004_0000`) | Name | Access | |
|---|---|---|---|
| `0x100` | PERF_RESET | W | write 1 to bit i to clear counter i (other bits keep their counters); reads 0 |
| `0x104` | PERF_INHIBIT | RW | bit i = 1 stops counter i; 0 after reset |
| `0x108` | PERF_NUM | R | number of counters (16); 0 on a bitstream without them |
| `0x200 + 8i` | counter i, bits 31:0 | R | |
| `0x204 + 8i` | counter i, bits 63:32 | R | |

Every counter counts its event from EXEC_CONTROL ← 1 until the core has halted *and* every unit
is idle. A halt does not wait for running units, and a kernel may `ecall` while its last matmul
or DMA store is in flight; the counters keep counting that tail. They then stop, so the host
reads a finished kernel's counts without tearing between the two words. Dividing a busy counter
by `cycles` gives that unit's utilization over the run. `cycles` is RUN_CYCLES (sys_regs, the
cycles until the halt) plus the halting instruction's cycle and that tail.

| i | Name | Counts cycles in which |
|---|---|---|
| 0 | `cycles` | the core runs, or a unit is still busy after the halt |
| 1 | `instret` | an instruction issues from S1 |
| 2 | `stall_unit` | S1 holds an instruction because its unit cannot take it (structural hazard) |
| 3 | `stall_data` | S1 holds an instruction for a data hazard: MREG reservation or port, VMEM bank, scalar load, scale register |
| 4 | `stall_dma` | `dma.wait` waits for a busy channel |
| 5 | `stall_delay` | a `delay` counts down |
| 6 | `mxu0_busy` | MXU slot 0 (systolic array or NVFP4 MXU) has any engine active |
| 7 | `mxu0_feed` | a slot-0 matmul feeds one row of A: the MAC array is fully used |
| 8 | `mxu1_busy` | MXU1 has any engine active |
| 9 | `mxu1_feed` | an MXU1 matmul feeds one row of A |
| 10 | `vpu_busy` | the VPU has an instruction in either slot |
| 11 | `xlu_busy` | the XLU transposes |
| 12 | `vload_busy` | the LSU's vload path is busy |
| 13 | `vstore_busy` | the LSU's vstore path is busy |
| 14 | `dma_busy` | some DMA channel is busy |
| 15 | `dma_beats` | a 32-byte DMA beat completes on the memory port (an event count, not cycles) |

A stalled cycle counts in exactly one stall counter, in the order `stall_delay`, `stall_dma`,
`stall_unit`, `stall_data`. So `cycles = instret + stall_* + other`, where the other cycles are
front-end bubbles (after a taken branch, after a start) and the tail after the halt. `mxu*_feed / cycles` is the MAC utilization: a matmul feeds
32 rows in 32 cycles and does 1024 (FP8) or 2048 (NVFP4) multiply-accumulates per row.

## Host side

`standalone/host/atlas_host.py` clears all counters (PERF_RESET ← `0xFFFF_FFFF`) right before it
starts a kernel and reads them after the halt (`Result.perf`, by name). `utilization()` turns
them into % of cycles. `run_tests.py` writes every run's counters to `OUT/perf.csv`, and
`apps/dit/ditflow/flow.py run` stores them with the board or simulation results.

```python
from atlas_host import Atlas, TsiLink, read_perf, utilization, CSR_PERF_RESET
atlas = Atlas(TsiLink())
atlas.link.write_words(CSR_PERF_RESET, [1 << 9])   # clear only mxu1_feed
print(utilization(read_perf(atlas.link)))
```
