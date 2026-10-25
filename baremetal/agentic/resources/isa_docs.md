# Atlas ISA and scheduling companion

The pinned instruction syntax and operation reference is [`npu-model/npu_spec/06_instruction_set/README.md`](../../../npu-model/npu_spec/06_instruction_set/README.md). The pinned spec controls instruction syntax and encoding; this companion controls RTL timing and addressing where they differ. Atlas branches and jumps have no delay slot.

## Canonical assembly syntax

Use typed registers: scalar `xN`, exponent `eN`, matrix bank `mN`, accumulator `accN`, and weight slot `wN`. Memory operands use `offset(base)`, and DMA channels are named `ch0` through `ch7`.

```asm
jalr x0, x1, 0
vload m1, 0(x6)
dma.config x5, ch0
dma.load x6, x10, x12, ch0
dma.store x10, x6, x12, ch1
dma.wait ch1
csrrw x0, 0x800, x7
csrrs x7, 0x800, x0
vmatpop.fp8.acc.mxu1 m4, acc0, e0
```

CSR instructions use real RISC-V operand order (`rd, csr, rs1`); `csrr` and `csrw` are not accepted pseudo-operations. `vmatpop.fp8.acc.mxu1` takes the accumulator before the exponent operand.

## Atlas scheduling and hardware rules

Atlas scalar control flow follows RV32I: the PC, link values and numeric branch and JAL offsets are byte addresses. Branches and jumps have no delay slot: a taken branch or jump drops the instruction after it, and a taken target with bit 1 set is illegal. Accelerator commands run asynchronously after issue. A hardware interlock holds an instruction until it can issue without a structural or data hazard (`docs/interlock.md`): `delay imm` is legal but never required, and `dma.wait chN` is the fence for DMA data in VMEM.

Scalar loads, stores, and `seld` use local VMEM byte addresses. `vload`/`vstore` bases and DMA VMEM operands use local VMEM word addresses. DMA DRAM operands are byte offsets added to the configured DMA base. The DMA beat is 32 bytes, and one MREG bank is 1024 bytes. Do not substitute runtime model memory-window defaults for Atlas harness addresses.

The `vload`/`vstore` immediate is signed and scaled by 32 words: effective VMEM word address is `x[rs1] + imm12 * 32`. Adjacent 1 KiB slots therefore commonly use immediates `0, 8, 16, ...`.

DMA configuration sets a global base; hardware ignores the channel encoded by `dma.config`. Wait on a channel before reusing it. Unless a test intentionally covers DMA concurrency, wait after each DMA request. The LSU is shared: the interlock holds an LSU operation while its path or VMEM bank is busy.

## Atlas timing and hazard rules

The handwritten suite still spaces dependent instructions with these delays, which the interlock makes optional: scalar load or `seld` 2 cycles; `vload`/`vstore` 33; `vtrpose.xlu` 64; `vmatpush`/`vmatpop` 31; `vmatmul` 34; `vunpack.fp8.bf16` 65; VPU row reductions 32–37, column reductions 128; `vli.all`/`vli.row` 63 and `vli.col`/`vli.one` 31; other VPU operations 64. DMA completion requires `dma.wait`, never a fixed delay.

The interlock also holds an instruction that writes a scalar register in the cycle a scalar load result takes the write port, a `seli` while a `seld` is pending, an LSU access to a VMEM bank that another LSU path holds, and a `vload` or `vstore` while that path is busy. Use 1 KiB-aligned vector bases. DMA shares VMEM ports and can extend LSU progress.

Pair-based BF16 VPU operations require even source and destination banks. A two-source VPU instruction that names two different pairs in one physical bank (`m4` and `m36`) is illegal; naming the same pair twice is fine. The interlock holds a VPU command while its issue-busy bit is set, an MXU command while the MXU cannot take it (a busy path, accumulator or weight slot), and a transpose while the XLU is busy. BF16 pushes read adjacent banks and BF16 pops write adjacent banks; FP32 pushes and pops use four consecutive banks.

## Golden data and pass/fail

The Atlas harness recognizes `# @DRAM_BASE`, `# @DRAM`, and `# @CHECK_DRAM` annotations. Generator JSON supplies `dram_preloads` and `dram_checks`; a JSON entry's DRAM base takes precedence over the source annotation, which takes precedence over the CLI default. Every DMA load needs data from JSON or inline `@DRAM` annotations.

The harness uploads the assembled program, starts Atlas, and polls `DBG0` with a bound. A completed run passes when `DBG0` is `1`; a nonzero failure code or a poll timeout fails the test. Preserve the standard CSR `0x800` pass/fail epilogue and `ecall`.
