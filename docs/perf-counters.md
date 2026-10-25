# Performance counters

The CSR block (`src/main/scala/diplomatic/memory/CSRFile.scala`) has 27 counters of 64 bits. The
host reads them on MMIO, in the scheme of RISC-V's `mhpmcounter` / `mcountinhibit`. In Chipyard
the host core reaches them on the periphery bus. In the standalone SoC the host computer reaches
them over UART TSI or TSI over UDP. The scalar core reads them as read-only CSRs (see "Scalar core
CSRs" below).

| Offset (CSR block, `0x0004_0000`) | Name | Access | |
|---|---|---|---|
| `0x100` | PERF_RESET | W | write 1 to bit i to clear counter i (other bits keep their counters); reads 0 |
| `0x104` | PERF_INHIBIT | RW | bit i = 1 stops counter i; 0 after reset |
| `0x108` | PERF_NUM | R | number of counters (27); 16 on older bitstreams, 0 on a bitstream without them |
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
| 6 | `mxu0_busy` | MXU slot 0 (the NVFP4 MXU) has any engine active |
| 7 | `mxu0_feed` | a slot-0 matmul feeds one row of A: the MAC array is fully used |
| 8 | `mxu1_busy` | MXU1 has any engine active |
| 9 | `mxu1_feed` | an MXU1 matmul feeds one row of A |
| 10 | `vpu_busy` | the VPU has an instruction in either slot |
| 11 | `xlu_busy` | the XLU transposes |
| 12 | `vload_busy` | the LSU's vload path is busy |
| 13 | `vstore_busy` | the LSU's vstore path is busy |
| 14 | `dma_busy` | some DMA channel is busy |
| 15 | `dma_beats` | a 32-byte DMA beat completes on the memory port (an event count, not cycles) |
| 16 | `stall_unit_mxu0` | a `stall_unit` or `stall_data` cycle, and MXU slot 0 cannot take the instruction |
| 17 | `stall_unit_mxu1` | ... and MXU1 cannot take the instruction |
| 18 | `stall_unit_vpu` | ... and the VPU cannot take the instruction (its issue-busy bit is set) |
| 19 | `stall_unit_xlu` | ... and the XLU is busy |
| 20 | `stall_unit_lsu` | ... and the vload or vstore path is busy, or a scalar load waits for an earlier one |
| 21 | `stall_unit_dma` | ... and the DMA channel is busy, or the DMA cannot take one more command |
| 22 | `stall_data_mreg` | ... and an MREG register is reserved (RAW, WAR, WAW) or its physical port is taken |
| 23 | `stall_data_vmem` | ... and an LSU path holds the VMEM bank of a scalar access, vload or vstore |
| 24 | `stall_data_dma_vmem` | ... and a DMA transfer would overtake a vload or vstore on its VMEM banks |
| 25 | `stall_data_xreg` | ... and a scalar load has not written its register yet, or its result takes the write port |
| 26 | `stall_data_scale` | ... and a `seld` has not written its scale register yet, or `seli` meets a load result |

A stalled cycle counts in exactly one of counters 2 to 5, in the order `stall_delay`, `stall_dma`,
`stall_unit`, `stall_data`. So `cycles = instret + stall_* + other`, where the other cycles are
front-end bubbles (after a taken branch, after a start) and the tail after the halt.
`mxu*_feed / cycles` is the MAC utilization: a matmul feeds 32 rows in 32 cycles and does 1024
(FP8) or 2048 (NVFP4) multiply-accumulates per row.

The `mxu0` counters (6, 7 and 16) count MXU slot 0: the NVFP4 MXU in `ipt-nvfp4`. In `default`,
slot 0 is empty and they stay 0. The `mxu1` counters count the inner-product-tree MXU.

Counters 16 to 26 give the causes of the `stall_unit` and `stall_data` cycles. They are *not*
exclusive: a cycle counts in each cause that holds in that cycle. For example, an instruction
whose unit is busy and whose source register has a pending write counts in `stall_unit`,
`stall_unit_<unit>` and `stall_data_mreg`. Thus:

- A unit cause (`stall_unit_*`) counts only in `stall_unit` cycles, and the unit causes together
  count at least `stall_unit`.
- A data cause (`stall_data_*`) counts in `stall_data` cycles and also in `stall_unit` cycles.
  The data causes together count at least `stall_data`.
- The cycles of `stall_delay` and `stall_dma` do not count in a cause.

Branch `origin/bu-pmu` (c69cba3) has a different counter implementation: a PMU registry from
`ucb-bar/pmu` in the Chipyard tile, with DMA response-wait, saturation and in-flight-limit
counters, and access and contention counters for each MREG and VMEM bank. It branches from `main`
before the interlock, and it is not merged here.

## Scalar core CSRs

The scalar core reads the CSR block with the RISC-V CSR instructions (`ScalarISA.scala`).

| CSR | Name | Access | MMIO offset |
|---|---|---|---|
| `0xC00` | cycle | RW | `0x00` |
| `0xC01` | instructions | RW | `0x04` |
| `0xC02` | status ([0] halted, [2:1] halt reason) | R | `0x08` |
| `0xC03` | illegal-instruction PC | R | `0x0C` |
| `0xC10` | DBG0 | RW | `0x10` |
| `0xC11` | DBG1 | RW | `0x14` |
| `0xCC0` | ISA (below) | R | `0x1C` |
| `0xCC1` | PERF_NUM | R | `0x108` |
| `0xCC2 + 2i` | counter i, bits 31:0 | R | `0x200 + 8i` |
| `0xCC3 + 2i` | counter i, bits 63:32 | R | `0x204 + 8i` |

- An access to any other CSR is an illegal instruction, also a read: the core halts, status reads
  3 (halted, illegal instruction) and the illegal-instruction PC holds the word address.
- A write to a read-only CSR is an illegal instruction. `csrrw` and `csrrwi` always write.
  `csrrs` and `csrrc` with `rs1` = `x0`, and `csrrsi` and `csrrci` with `uimm` = 0, only read
  (RISC-V privileged spec). With any other `rs1` they write, also when the register holds 0. A
  read does not write the CSR back, so a read of cycle or instructions does not stop that counter.
- `0xC00` to `0xC11` are in the standard read-only range (`csr[11:10]` = 3), but programs can write
  cycle, instructions, DBG0 and DBG1. `0xCC0` to `0xCFF` is the custom read-only range.
- The counters count while the core runs. To read a 64-bit counter, read bits 63:32, then bits
  31:0, then bits 63:32 again. If the high word changed, read again.

The ISA CSR tells programs and the host which instruction set and units the core has:

| Bits | |
|---|---|
| 31:24 | ISA major version, 2 |
| 23:16 | ISA minor version, 0 |
| 15:4 | 0 |
| 3:0 | units, as sys_regs FEATURES: [0] hardware interlock, [1] 0, [2] inner-product-tree MXU, [3] NVFP4 MXU |

A change of an instruction's encoding or meaning increments the major version; a minor version only
adds instructions. The version is the highest `ver` of the ISA table,
`npu-model/npu_model/isa/atlas_isa.toml`. The RTL gets it from the generated
`Instructions.ISA_MAJOR` and `ISA_MINOR`, and npu_model has it as `npu_model.isa.ISA_VERSION`.
npu_model reads the ISA CSR with unit bits 0, 2 and 3 set. `make_bundle.py` writes the assembler's
version into each bundle (`"isa": [2, 0]`). `atlas_host.Atlas` reads the ISA CSR when it connects,
and `Atlas.run` does not run a bundle built for another version: the status is ERROR. A bundle
without `"isa"` counts as ISA 1.0, and so does a bitstream without the ISA CSR (it reads 0).

The baremetal tests `csr_isa_perf`, `csr_illegal_unmapped` and `csr_illegal_ro_write` check
these rules. The two `csr_illegal_*` tests have `# @EXPECT_ILLEGAL`: `run_tests.py --config`
expects them to stop with an illegal instruction in every configuration.

## Host side

`standalone/host/atlas_host.py` clears all counters (PERF_RESET ← `0xFFFF_FFFF`) right before it
starts a kernel and reads them after the halt (`Result.perf`, by name). `read_perf` reads the
first PERF_NUM counters of `PERF_NAMES`. `utilization()` turns them into % of cycles.
`run_tests.py` writes every run's counters to `OUT/perf.csv`, and `apps/dit/ditflow/flow.py run`
stores them with the board or simulation results.

```python
from atlas_host import Atlas, TsiLink, read_perf, utilization, CSR_PERF_RESET
atlas = Atlas(TsiLink())
atlas.link.write_words(CSR_PERF_RESET, [1 << 9])   # clear only mxu1_feed
print(utilization(read_perf(atlas.link)))
```
