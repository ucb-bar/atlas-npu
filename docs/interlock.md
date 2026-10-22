# Hardware interlock (`pipeline-interlock` branch)

On `main`, Atlas is software-scheduled: units never stall each other, and a program spaces
dependent instructions with `DELAY` (npu_spec §4, "Spacing instructions with `DELAY`"). An
instruction issued too early trips an RTL assertion in simulation. With the assertions stripped
(FPGA, silicon), it corrupts state or is dropped: MXU and VPU commands are fire-and-forget.

On this branch, the scalar core holds the instruction in S1 until it can issue without a
hazard. `DELAY` stays legal but is never required.

## What the interlock checks

Each check is one of the spec's software-scheduling rules, now evaluated in hardware against the
instruction in S1 (`ScalarCore.scala`, section "Interlock").

| Rule (npu_spec §4) | Hardware signal |
|---|---|
| Same-unit issue spacing: MXU | `cmdReady` from each MXU sequencer: its own accept condition for the decoded command |
| MXU buffer rules | the same `cmdReady`, plus five port/path rules the sequencers used to assert on (below) |
| Same-unit issue spacing: VPU, XLU | VPU `issueBusy(op)`, XLU `busy` |
| `vload`/`vstore`/scalar load, same path | LSU `vloadBusy` / `vstoreBusy`, scalar load pending |
| DMA to a busy channel | channel busy, and the next command slot free (`DmaEngine.canEnqueue`) |
| Register reservations (RAW, WAR, WAW; VLOAD WAR-exempt) | `MregBankTracker` read/write busy, as the old assertions used |
| Physical MREG ports (`mN` and `m(N+32)` share one read and one write port) | the tracker's busy bits folded onto 32 physical banks |
| LSU scalar and vector paths must not share a VMEM bank | LSU `vloadVmemBank` / `vstoreVmemBank` |
| DMA must not overtake a VLOAD/VSTORE on the banks it touches | the same bank reports against the DMA's line range |
| Scalar load result (no bypass) and register-file write port | pending load `rd`, response cycle |
| `seld` result and the scale-register write port | pending `seld`, for `vpack`/`vunpack`, `vmatpop.fp8`, `seli` |

The MXU sequencers' accept logic now also waits for these, which were only assertions
(`AccumulationBuffers`, shared weight/accumulator push paths):

- a weight push while another weight push still writes, and an accumulator push while another
  accumulator push still writes (one write path each);
- an accumulator push to `accN` while a matmul to `accN` is in flight (one write port);
- a pop or `vmatmul.acc` of `accN` in the cycle a push writes row 0 of `accN`;
- a pop of `accN` while a `vmatmul.acc` still reads its partial sums, and the reverse (one read port).

To let back-to-back commands on one register keep their 32-cycle stride, the MXUs now report a
port to the tracker only while it will still access its bank after the current cycle.

## What stays with software

`DMA.WAIT` remains the architectural fence between a DMA transfer and the instructions that read
or overwrite its VMEM data, as on `main`. A CSR read of `mcycle` is not a hazard, so it does not
wait for earlier long operations: a timed window that ends with one should end after an
instruction that depends on the measured work (for example the pop of the result).

## Validation

The standalone SoC (`standalone/`) runs the 121 baremetal tests in Verilator with every RTL
assertion enabled, from the original kernels and from kernels with all 4,525 `DELAY`s removed:

| RTL | with `DELAY` | `DELAY` removed |
|---|---|---|
| `main` | 121 / 121 | 12 / 121 (107 stop on hazard assertions, 2 return wrong data) |
| `pipeline-interlock` | 121 / 121 | 121 / 121 |
| `pipeline-interlock`, VCU118 (standalone bitstream) | 121 / 121 | 121 / 121 |

Without `DELAY`, kernels run from start to halt in 9.8% fewer cycles over the suite (110 faster,
11 equal, none slower; median 5%, up to 38% for `mxu0_single_output_tile_bf16`). MXU
utilization of the `perf_mm_*` kernels is unchanged (87-93%): the interlock issues them at the
same 32-cycle stride as the hand schedules.
