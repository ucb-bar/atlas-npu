# Hardware interlock (`pipeline-interlock` branch)

On `main`, Atlas is software-scheduled: units never stall each other, and a program spaces
dependent instructions with `DELAY` (npu_spec §4, "Spacing instructions with `DELAY`"). An
instruction issued too early trips an RTL assertion in simulation. With the assertions stripped
(FPGA, silicon), it corrupts state or is dropped: MXU and VPU commands are fire-and-forget.

On this branch, the scalar core holds the instruction in S1 until it can issue without a
hazard. `DELAY` stays legal but is never required.

## What the interlock checks

Each check is one of the spec's software-scheduling rules, now evaluated in hardware against the
instruction in S1 (`ScalarCore.scala`, section "Interlock"). They fall into two kinds:

- **Structural hazards.** No instruction issues to a unit that cannot take it. Every unit
  reports whether it can accept the decoded command this cycle. A pipelined unit accepts a new
  instruction when the stage that instruction needs is free, not when the whole unit is idle.
  For example, an MXU takes the next matmul on the last cycle of the previous activation feed,
  with up to 3 (MXU0) or 2 (MXU1) in flight. Waiting for the unit to drain instead would cut MXU0
  to one matmul per 95 cycles.
- **Data hazards.** Each MREG register is a resource with a read reservation and a write
  reservation. An instruction does not issue while a register it reads has a pending write
  (RAW), or while a register it writes has a pending read (WAR) or write (WAW). A reservation
  lasts from issue until the unit's last access to that register. Units report each register
  for as long as they will still access it after the current cycle: an instruction issued now
  touches MREG from the next cycle on. The MXUs' weight slots and accumulators get the same
  treatment inside each MXU, VMEM tiles through the bank checks, scalar registers through the
  pending load, and scale registers through the pending `seld`.

| Rule (npu_spec §4) | Hardware signal |
|---|---|
| Same-unit issue spacing: MXU | `cmdReady` from each MXU sequencer: its own accept condition for the decoded command |
| MXU buffer rules | the same `cmdReady`, plus five port/path rules the sequencers used to assert on (below) |
| Same-unit issue spacing: VPU, XLU | VPU `issueBusy(op)`, XLU `busy` |
| `vload`/`vstore`/scalar load, same path | LSU `vloadBusy` / `vstoreBusy`, scalar load pending |
| DMA to a busy channel | channel busy, and the next command slot free (`DmaEngine.canEnqueue`) |
| Register reservations (RAW, WAR, WAW), for every instruction including `vload` | `MregBankTracker` read/write busy |
| Physical MREG ports (`mN` and `m(N+32)` share one read and one write port) | the tracker's busy bits folded onto 32 physical banks |
| LSU scalar and vector paths must not share a VMEM bank | LSU `vloadVmemBank` / `vstoreVmemBank` |
| DMA must not overtake a VLOAD/VSTORE on the banks it touches | the same bank reports against the DMA's line range |
| Scalar load result (no bypass) and register-file write port | pending load `rd`, response cycle |
| `seld` result and the scale-register write port | pending `seld`, for `vpack`/`vunpack`, `vmatpop.fp8`, `seli` |

The NVFP4 MXU (docs/nvfp4-mxu.md) exports the same `cmdReady`, its 2- and 4-register transfers
report every register they will still touch, and `vload.scale` / `vstore.scale` count as VLOAD /
VSTORE.

The MXU sequencers' accept logic now also waits for these, which were only assertions
(`AccumulationBuffers`, shared weight/accumulator push paths):

- a weight push while another weight push still writes, and an accumulator push while another
  accumulator push still writes (one write path each);
- an accumulator push to `accN` while a matmul to `accN` is in flight (one write port);
- a pop or `vmatmul.acc` of `accN` in the cycle a push writes row 0 of `accN`;
- a pop of `accN` while a `vmatmul.acc` still reads its partial sums, and the reverse (one read port).

To let back-to-back commands on one register keep their 32-cycle stride, the MXUs now report a
port to the tracker only while it will still access its bank after the current cycle. The MXU
sequencers' own MREG-conflict assertions use the same rule. Before 2026-10-22 they counted an
engine on its last row as busy, and so fired on legal back-to-back push and pop pairs.

### One operand rule

A two-source VPU instruction (`vadd`, `vsub`, `vmul`, `vminimum`, `vmaximum`) streams both
source pairs in the same cycles. Two different pairs in one physical bank, such as `m4` and `m36`,
can never be read together, and no amount of waiting fixes that. Such an instruction is
illegal: the core halts with STATUS 3. Naming the same pair twice is fine, since one read serves
both operands.

### 2026-10-22: `vload` lost its WAR exemption

`vload` used to skip the WAR check, on the grounds that it writes each row after an earlier
reader has read that row. That holds for readers that stream one register in row order, but
not for the VPU, which reads a BF16 pair as all of `m2k` and then all of `m2k+1`. So
`vadd.bf16 m8, m2, m6` followed by `vload m3` overwrote `m3` before the VPU read it.
`hazard_vload_war_vpu_pair` (below) returns wrong data on the old RTL and passes now.

## What stays with software

`DMA.WAIT` remains the architectural fence between a DMA transfer and the instructions that read
or overwrite its VMEM data, as on `main`. A CSR read of `mcycle` is not a hazard, so it does not
wait for earlier long operations: a timed window that ends with one should end after an
instruction that depends on the measured work (for example the pop of the result).

## Validation

### Randomized hazard test

`apps/dit/ditflow/hazfuzz.py` generates random streams of dependent tensor instructions with no
`DELAY`: `vload`, `vstore`, VPU arithmetic and row reductions, MXU push, matmul and pop, XLU
transposes, and the NVFP4 2- and 4-register transfers. They run over 20 registers, including
ones that share a physical bank. Each next instruction is biased to read or overwrite what the
previous ones touched. The expected DRAM comes from the functional model, which runs in program
order, the architectural meaning on an interlocked core. A `--spaced` copy of each program puts
`DELAY 200` after every tensor instruction, so nothing overlaps there. A program that fails in
both copies points at the model's arithmetic, not at the interlock.

| Run (Verilator with all assertions, unless VCU118) | `default` (MXU0 + MXU1) | `ipt` | `ipt-nvfp4` (+ NVFP4) |
|---|---|---|---|
| before (d6ef8b3), 64 MXU1 programs × 400 instructions | 0 / 64: 51 port, 15 pop-WAR, 8 push-RAW assertions | | |
| now, 64 programs × 400 instructions + the directed test | 65 / 65 | 63 / 65 | 65 / 65 |
| now, the same 64 programs spaced | 64 / 64 | 62 / 64 | 64 / 64 |
| now, 256 programs × 600 instructions (seeds 2000-2255) | 251 / 256 | 254 / 256 | 253 / 256 |
| now, the same 256 programs spaced | 251 / 256 | 254 / 256 | 253 / 256 |
| now, VCU118 (DDR4 bitstream), 64 programs + directed | 65 / 65 | 63 / 65 | see standalone/README.md |

Every failing program fails the same way in its spaced copy: same addresses, same values. So no
program fails only without `DELAY`, over 963 programs and about 540,000 random tensor
instructions. On `ipt`, two programs (seeds 1000 and 1028) were bisected, and they fail identically in both copies. Bisection puts
the first difference at one instruction each: a `vsub.bf16` result one BF16 ULP from the RTL's,
and a `vmatmul.acc.mxu1` onto a large BF16 partial sum. npu_model's `RtlNumerics` and ditflow's
fast numerics agree with each other there, so the model's arithmetic is off in these two corner
cases. No program fails only without `DELAY`.

On the old RTL, the assertions came from three sources. Two-source VPU instructions named two
pairs in one bank (now illegal). The MXU1 sequencer's push and pop checks counted an engine on
its last row as busy (now aligned with the interlock). And `vload`'s WAR exemption, which the
directed test shows as wrong data rather than an assertion.

### The baremetal suite

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
