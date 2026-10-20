# Atlas ISA and scheduling companion

The pinned instruction syntax and operation reference is [`npu-model/npu_spec/06_instruction_set/README.md`](../../../npu-model/npu_spec/06_instruction_set/README.md). The pinned spec controls instruction syntax and encoding; this companion controls RTL timing and addressing where they differ. Atlas branches and jumps have two delay slots.

## Canonical assembly syntax

Use typed registers: scalar `xN`, exponent `eN`, matrix bank `mN`, accumulator `accN`, and weight slot `wN`. Memory operands use `offset(base)`, and DMA channels are named `ch0` through `ch7`.

```asm
jalr x0, x1, 0
vload m1, 0(x6)
dma.config x5, ch0
dma.load x6, x10, x12, ch0
dma.store x10, x6, x12, ch1
dma.wait ch1
csrrw x0, 0xC10, x7
csrrs x7, 0xC10, x0
vmatpop.fp8.acc.mxu0 m4, acc0, e0
```

CSR instructions use real RISC-V operand order (`rd, csr, rs1`); `csrr` and `csrw` are not accepted pseudo-operations. `vmatpop.fp8.acc.mxuN` takes the accumulator before the exponent operand.

## Atlas scheduling and hardware rules

Atlas scalar control flow uses IMEM word indices, not byte addresses. Numeric branch and JAL offsets are in words, and branches and jumps have two architectural delay slots. The scalar pipeline stalls only for `delay imm` and `dma.wait chN`; accelerator commands otherwise run asynchronously. Place no branch or jump in either delay slot.

Scalar loads, stores, and `seld` use local VMEM byte addresses. `vload`/`vstore` bases and DMA VMEM operands use local VMEM word addresses. DMA DRAM operands are byte offsets added to the configured DMA base. The DMA beat is 32 bytes, and one MREG bank is 1024 bytes. Do not substitute runtime model memory-window defaults for Atlas harness addresses.

The `vload`/`vstore` immediate is signed and scaled by 32 words: effective VMEM word address is `x[rs1] + imm12 * 32`. Adjacent 1 KiB slots therefore commonly use immediates `0, 8, 16, ...`.

DMA configuration sets a global base; hardware ignores the channel encoded by `dma.config`. Wait on a channel before reusing it. Unless a test intentionally covers DMA concurrency, wait after each DMA request. The LSU is shared: avoid back-to-back LSU operations, including vector loads/stores and scalar VMEM accesses, without enough independent work or delay.

## Atlas timing and hazard rules

Only `delay imm` and `dma.wait chN` stall the scalar pipe; other accelerator operations are asynchronous. Treat these minimum delays as floors: scalar load or `seld` 2 cycles; `vload`/`vstore` 33; `vtrpose.xlu` 64; `vmatpush`/`vmatpop` 31; `vmatmul` MXU0 95 and MXU1 34; `vunpack.fp8.bf16` 65; VPU row reductions 32–37, column reductions 128; `vli.all`/`vli.row` 63 and `vli.col`/`vli.one` 31; other VPU operations 64. DMA completion requires `dma.wait`, never a fixed delay. Use more time when operations overlap or contend.

A scalar load response cannot coincide with another scalar register writeback; `seld` cannot coincide with `seli`. Separate loads from register-writing work by at least the scalar-load floor. The VMEM LSU is shared: same-bank scalar load plus `vload`, or scalar store plus `vstore`, can assert. Do not issue another `vload` or another `vstore` while that same path is busy. Use 1 KiB-aligned vector bases. DMA shares VMEM ports and can extend LSU progress.

Pair-based BF16 VPU operations require even source and destination banks. Two reads may alias the same bank only when they request the same row. VPU commands must issue when their selected issue-busy bit is clear; do not assume all VPU operations can overlap. MXU compute, push, and pop use independently scheduled slots but assert on busy-slot or accumulator/weight conflicts; wait between commands unless intentionally testing overlap. BF16 pushes read adjacent banks and BF16 pops write adjacent banks. XLU accepts a command only while idle, so wait for a transpose to finish before reusing it or reading its destination bank.

## Golden data and pass/fail

The Atlas harness recognizes `# @DRAM_BASE`, `# @DRAM`, and `# @CHECK_DRAM` annotations. Generator JSON supplies `dram_preloads` and `dram_checks`; a JSON entry's DRAM base takes precedence over the source annotation, which takes precedence over the CLI default. Every DMA load needs data from JSON or inline `@DRAM` annotations.

The harness uploads the assembled program, starts Atlas, and polls `DBG0` with a bound. A completed run passes when `DBG0` is `1`; a nonzero failure code or a poll timeout fails the test. Preserve the standard CSR `0xC10` pass/fail epilogue and `ecall`.
