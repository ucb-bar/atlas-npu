# ISA 2.0

Atlas reports ISA version 2.0 in CSR `0xCC0` (docs/perf-counters.md). The instruction table
`npu-model/npu_model/isa/atlas_isa.toml` is the definition: npu_model's instruction classes,
`src/main/scala/atlas/scalar/Instructions.scala`, the green card
(`npu-model/npu_spec/06_instruction_set`) and the atlas-mlir encoding header are generated from
it. This page gives what changed from version 1.0, why, and what 2.0 leaves out.

## Version rule

- The major version changes when a word that decoded in the last release gives a different result
  or stops decoding.
- A minor version only adds instructions: words that were illegal get a meaning. A 2.0 binary
  runs unchanged on 2.x hardware.
- Every change of this page is in 2.0, also the instructions that only add (one release, one
  switch for the external tools). The table's `ver` key gives the version of each instruction.
- The host reads CSR `0xCC0` and refuses a bundle built for another ISA version
  (`standalone/host/atlas_host.py`).

## Encoding

Instructions are 32 bits. The 46 RV32I instructions keep their encodings. The 71 Atlas
instructions use only the four custom major opcodes, one format each:

| Opcode | Format | Instructions | Selector codes used |
|---|---|---|---|
| custom-0 `0x0B` | VR | VPU, XLU and MXU (55) | 55 of 128 funct7 |
| custom-1 `0x2B` | VI | `vli.*`, `seli`, `delay`, `vsetm` (7) | 7 of 8 funct3 |
| custom-2 `0x5B` | VLS | `vload`, `vstore`, `vload.scale`, `vstore.scale` (4) | 4 of 4 funct2 |
| custom-3 `0x7B` | R | `dma.*`, `seld` (5) | 5 of 128 funct7 |

Version 1.0 put Atlas instructions in OP-V (`0x57`), OP-VE (`0x77`), LOAD-FP (`0x07`), the
reserved `0x6B`, the 48-bit and 64-bit prefixes `0x5F` and `0x7F`, and in free funct3 values of
LOAD and JALR. These opcodes belong to standard extensions, so a standard assembler or
disassembler decoded Atlas words as other instructions. RISC-V reserves the custom opcodes for
this use.

- In the custom opcodes, every bit of the word is a selector, an operand or fixed. An operand
  bit that the operand does not use must be 0: an odd register of a BF16 pair, an e-register
  above `e31` in a 6-bit field, an accumulator above `acc1` and any other set reserved bit make
  the word illegal. A wrong word halts the core; it does not run as another instruction.
- The MXU instructions are funct7 = `0b11_s_oooo`: s is the MXU (0: the NVFP4 MXU, `.nv`;
  1: `mxu1`) and oooo is the command. One decoder serves both MXUs.
- Free space: custom-0 funct7 `0x09`, `0x0D`, `0x0F`, `0x11`, `0x15`, `0x17`-`0x20`, `0x22`,
  `0x23`, `0x25`, `0x27`-`0x3F`, `0x4E`, `0x51`-`0x5F`, `0x63`, `0x67`, `0x6A`-`0x6F`,
  `0x7A`-`0x7F` (the last three groups are in the MXU block); custom-1 funct3 7; custom-3
  funct7 `0x05`-`0x7F`. custom-2 is full: a new VMEM transfer goes into custom-3.

## From 1.0 to 2.0

| 1.0 | 1.0 hex | 2.0 hex | Change |
|---|---|---|---|
| 25 VPU instructions | `57/xx` | `0B/xx` | same funct7; new numerics of `vadd`, `vsub`, `vsquare`, `vcube`, `vpack`, `vunpack` (next table) |
| 7 `mxu0` instructions (systolic array) | `77/00`-`77/0C` (even) | none | removed with the systolic array |
| 7 `mxu1` instructions | `77/01`-`77/0D` (odd) | `0B/70`, `73`, `71`, `77`, `74`, `78`, `79` | FP32 accumulators, Blackwell arithmetic and conversions |
| 7 `.nv` instructions | `77/0E`-`77/14` | `0B/60`, `61`, `62`, `64`, `66`, `68`, `69` | the BF16 pop gives `0x7FFF` for NaN |
| `vtrpose.xlu` | `6B/00` | `0B/50` | none |
| `vload`, `vstore`, `vload.scale`, `vstore.scale` | `07/0`-`07/3` | `5B/0`-`5B/3` | none |
| `vli.all`, `.row`, `.col`, `.one` | `5F/0`-`5F/3` | `2B/0`-`2B/3` | none |
| `seli` | `03/7` (I) | `2B/4` (VI) | 8-bit unsigned immediate |
| `delay` | `67/1` (I) | `2B/5` (VI) | none |
| `seld e(rd), imm(x(rs1))` | `03/6` (I) | `7B/04` (R) | no offset; selects the byte of the address |
| `dma.load`, `dma.store` | `7B/00`, `7B/01` | `7B/00`, `7B/01` | none |
| `dma.config`, `dma.wait` | `7F/00`, `7F/01` | `7B/02`, `7B/03` | none |

Operations that give a different result in 2.0:

| Operation | 1.0 | 2.0 |
|---|---|---|
| `vadd.bf16`, `vsub.bf16` | FP32 sum truncated to BF16 | BF16 round to nearest even (RNE) of the exact sum |
| `vsquare.bf16`, `vcube.bf16` | truncated; subnormal and NaN inputs and results below 2^-126 give 0 | BF16 RNE of the exact x^2 or x^3, as `vmul` |
| `vpack.bf16.fp8` | x / G(e); flushes subnormals and NaN to +0 | E4M3 RN SATFINITE of x * G(e) |
| `vunpack.fp8.bf16` | flushes E4M3 subnormals and NaN; overflow gives +-`0x7F7F` | BF16 RNE of x * G(e) |
| e = 255 | G = 2^127 | G = 2^128 |
| the `mxu1` instructions | BF16 accumulators, one BF16 rounding per K = 32 step | FP32 accumulators and the Blackwell FP8 MMA (docs/fp8-mxu.md) |
| `vmatpop.bf16.acc.nv` of NaN | `0x7FC0` with the input's sign | `0x7FFF` |

G(e) = 2^(e - 127). Every instruction with an e-register operand multiplies by G(e); in 1.0,
`vpack` divided. The FP8 and BF16 conversions are those of the Blackwell conversion instructions
(`cvt.rn.satfinite.e4m3x2.f32`, `cvt.rn.bf16.f32`): E4M3 subnormals, NaN and signed zero are kept
as the RTX 5090 gives them. The vectors are in `baremetal/generators/fp8_silicon`.

## New instructions

| Instructions | Hex | Why |
|---|---|---|
| `vsetm M` | `2B/6` | Row count M (1-32), 32 after a start. Tensor instructions act on rows 0 to M - 1, so a short tile (the DiT's time path at M = 1) costs M cycles, not 32. |
| `v{add,sub,mul,minimum,maximum}.bf16.vx` | `0B/08`-`0B/0E` | Second operand is the BF16 in `x[rs2][31:16]`. Constants need no tile of their own. |
| `v{add,sub,mul,minimum,maximum}.bf16.vr` | `0B/10`-`0B/16` | Second operand is row 0 of the second pair, for every row. Bias and scale rows need no broadcast. |
| `vmatpop.bf16.scaled.acc.{mxu1,nv}` | `0B/75`, `0B/65` | BF16 RNE of acc * G(e), one rounding. The dequantization of an FP8 matmul needs no VPU pass. |
| `vmatpush.acc.fp32.mxu1`, `vmatpop.fp32.acc.mxu1` | `0B/72`, `0B/76` | Spill and reload of FP32 partial sums through `m[r..r+3]`, as the NVFP4 MXU has. |
| `vgelu.bf16` | `0B/4F` | GELU (tanh form) in one instruction. It gives the result of the eight 2.0 instructions vcube, vmul, vadd, vmul, vtanh, vadd, vmul, vmul, bit for bit. |

## Scalar core and CSRs

- `pc` is the byte address of the instruction, as in RV32I. Branch and `jal` offsets are byte
  offsets, and `jal` and `jalr` write `pc + 4`. There is no delay slot. A taken branch or jump to
  a target with bit 1 set is illegal. In 1.0, the PC counted words and the instruction after a
  branch always ran.
- CSRs: `cycle` `0xC00` and `instret` `0xC02` (read only, standard numbers), DBG0 and DBG1
  `0x800` and `0x801`, the ISA version and the performance counters in `0xCC0`-`0xCFF`. Another
  number is illegal (docs/perf-counters.md).

## Other legality rules

- `dma.load` and `dma.store` of a size that is 0, above 4096 or not a multiple of 32.
- A two-input `.bf16` VPU instruction with its two source pairs in one physical bank (`m[r]` and
  `m[r+32]`).
- `vpack.bf16.fp8` with `vd = vs2 + 1`: it would write rows of `m[vd]` before it reads them as
  `m[vs2+1]`.

## Not in 2.0

- `vquant.bf16.nvfp4`, an on-chip NVFP4 quantizer, is deferred. NVFP4 gives clearly more error
  than FP8 on the trained DINOv3 encoder (docs/nvfp4-accuracy.md), so this policy gets no more
  NVFP4 hardware.
- The systolic array (`mxu0`) is removed. `default` is the configuration with `mxu1` only, and
  `ipt-nvfp4` adds the NVFP4 MXU.

## Candidates for 2.1

- A `vload` and `vstore` of 32 rows at any M. At M < 32 each weight load now needs `vsetm 32`
  before it and `vsetm M` after it: about 3,500 instructions per DiT step.
