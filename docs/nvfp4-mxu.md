# Configurations and the NVFP4 MXU

## Configurations

`atlas.common.AtlasConfigs` names two configurations. MXU slot 1 is always the
inner-product-tree (IPT) MXU; slot 0 holds the NVFP4 MXU or nothing.

| Name | `AtlasParams` | MXUs | Chipyard fragment |
|---|---|---|---|
| `default` | `AtlasParams()` | IPT (`.mxu1`) | `WithAtlasTile()` |
| `ipt-nvfp4` | `nvfp4 = Some(Nvfp4MxuParams())` | IPT + NVFP4 (`.nv`) | `WithAtlasTileNvfp4` |

An instruction for a unit the configuration does not have is illegal: the core halts and
CSR STATUS reads 3 (halted, illegal instruction). The standalone build selects a configuration
with `EmitAtlasCore --config=NAME`; sys_regs FEATURES reports it ([2] IPT, [3] NVFP4; [1] is
0).

## NVFP4 format and storage

NVFP4 stores elements as E2M1 (4 bits) with one UE4M3 scale per 16 consecutive elements. An
m-register row is 256 bits, so it holds 64 elements: element k in byte k / 2, low nibble for even
k. A 32 x 64 NVFP4 tile is one m-register plus 32 x 4 scale bytes.

The `ipt-nvfp4` configuration adds:

| State | Size | Contents |
|---|---|---|
| Scale sidecar (`MregScaleFile`) | 64 registers x 32 rows x 4 B (8 KiB) | row r, byte g: scale of row r's elements 16g .. 16g + 15 |
| NVFP4 weight slots `w0`, `w1` | 2 x (32 x 64 E2M1 + scales) | row j: output column j's 64 elements and 4 scales |
| NVFP4 accumulators `acc0`, `acc1` | 2 x 32 x 32 FP32 | |

In VMEM a register's scales are 128 bytes (4 lines): line l holds rows 8l .. 8l + 7, row r at
bytes 4(r % 8) .. 4(r % 8) + 3. DMA moves them like any data. A register's data and its scale
sidecar are independent: `vload`, VPU, XLU and MXU writes leave the sidecar unchanged.

Tensor registers and their scales are undefined after reset. `vmatpush.weight.nv` and
`vmatmul(.acc).nv` read the scales of `m[vs1]` together with its data, so a program loads the
scales of new data with `vload.scale`. The lint `python -m npu_model.scale_lint PROGRAM.S` (in
`npu-model`; it runs only on request) warns when such a read can see scales that no `vload.scale`
loaded since reset, or data written after the register's last `vload.scale`.

An FP32 accumulator tile moves through four consecutive m-registers: register q of
`{m[r], m[r+1], m[r+2], m[r+3]}` holds columns 8q .. 8q + 7 of every row (little-endian words).
A BF16 tile uses a register pair like the other MXUs: columns 0-15 in `m[r]`, 16-31 in `m[r+1]`.

## Instructions

MXU instructions use opcode `0x77` (VR format: funct7, vs2, vs1, vd); the scale transfers use the
VLS format of `vload`/`vstore` (opcode `0x07`).

| Mnemonic | Encoding | Operation |
|---|---|---|
| `vmatpush.weight.nv w(vd), m(vs1)` | `77/0E` | `w[vd] = m[vs1]` and its scales |
| `vmatpush.acc.bf16.nv acc(vd), m(vs1)` | `77/0F` | `acc[vd] = {m[vs1], m[vs1+1]}` (BF16, exact in FP32) |
| `vmatpush.acc.fp32.nv acc(vd), m(vs1)` | `77/10` | `acc[vd] = m[vs1 .. vs1+3]` (FP32) |
| `vmatpop.bf16.acc.nv m(vd), acc(vs2)` | `77/11` | `{m[vd], m[vd+1]} = BF16(acc[vs2])`, round to nearest even |
| `vmatpop.fp32.acc.nv m(vd), acc(vs2)` | `77/12` | `m[vd .. vd+3] = acc[vs2]` |
| `vmatmul.nv acc(vd), m(vs1), w(vs2)` | `77/13` | `acc[vd] = A @ W^T`, A = `m[vs1]` with its scales |
| `vmatmul.acc.nv acc(vd), m(vs1), w(vs2)` | `77/14` | `acc[vd] += A @ W^T` |
| `vload.scale m(vd), imm(x(rs1))` | VLS funct2 `10` | `m[vd]`'s scales = 128 B of VMEM, 128 B aligned |
| `vstore.scale m(vd), imm(x(rs1))` | VLS funct2 `11` | 128 B of VMEM = `m[vd]`'s scales |

`npu-model` assembles these instructions and defines their semantics
(`npu_model/configs/isa_definition.py`, arithmetic in `npu_model/configs/nvfp4.py`). Its timing
simulator does not model the NVFP4 MXU.

## Arithmetic

Each output element is one `fp4_fedp` dot product (`src/main/resources/vsrc/nvfp4`, see
Provenance): the arithmetic of NVIDIA tensor cores, as measured on an RTX 5090 (`mma.sync`
m16n8k64) and Jetson Thor (`tcgen05`):

```text
acc[r][j] = acc[r][j] + sum over blocks g of (sum of 16 E2M1 products A[r][16g+i] W[j][16g+i]) x sA[r][g] x sW[j][g]
```

A UE4M3 scale byte has exponent field E (bits 6:3) and mantissa m (bits 2:0). Bit 7 is ignored,
and E = 15, m = 7 (`0x7F`, `0xFF`) is NaN. The other codes decode to (S / 8) x 2^e:

| Code | S | e | Value |
|---|---|---|---|
| E > 0 | 8 + m | E - 7 | (1 + m / 8) x 2^(E - 7), up to 448 (`0x7E`) |
| E = 0, m > 0 | m | -6 | m / 8 x 2^-6 (subnormal) |
| E = 0, m = 0 | 0 | -126 | 0 |

The 16-element block sums are exact. Block g's term is its sum x sA x sW at exponent
eA + eW, even when the sum or a scale is zero; the accumulator's exponent is
max(Ec - 127, -126) for its biased exponent Ec, and emax is the largest of the five. Every term,
the old accumulator included, is cut toward zero at 2^(emax - 35), and the sum is cut toward zero
to FP32 once (overflow gives +-inf). A NaN scale or accumulator gives `0x7FFFFFFF`, an infinite
accumulator passes through, and a row of all-zero products returns the accumulator (with -0
becoming +0). So a chain of `vmatmul.acc.nv` over K = 64n equals a chain of n NVIDIA m16n8k64
MMAs bit for bit, except for the known gap below.

Known gap: when a 16-element block has only zero products and its scales give it the largest
exponent, the model and `fp4_fedp` let it set emax (as MMA-Sim does), and silicon does not
always do so; the rule is not known. The full NVFP4 vector sets (Evidence) have 16 vectors with
such a block, all kept in `nvfp4_silicon` and counted by `check_nvfp4_silicon.py`. It changes
the result on one, `random_nvfp4_seed1` vector 15 (an all-zero block with scale 448,
C = -3e-10): model and RTL `0x42400000`, both GPUs `0x423fffff`. The model's
other documented differences from silicon, 28 subnormal and 6 normal outputs off by 1 ULP in
600,054 random vectors (<https://ucb-bar.github.io/nvfp4/> section 4, Model accuracy), are all
MXFP4 vectors (ucb-bar/nvfp4@0c611e1 `tcgen05/tcgen05_results.zip`, `summary.json`), a mode
Atlas does not use.

The 40,000 vectors keep the hard cases on purpose (`nvfp4_silicon/README.md`), so 1 in 40,000 is
not a rate: on the full sets the model is off silicon on 1 of 200,018 random vectors and on none
of the 300,025 stored vectors. The gap is also not limited to 1 ULP. The emax rule sets the bit
at which C is cut, and in a `vmatmul.acc.nv` chain C is the partial sum. In 20,000 runs of the
model (`nvfp4.dot`: block 0 all zero with both scales from `0x50` to `0x7E`, the other blocks and
C random), the result when block 0 sets emax differed by up to 0.35%, or 52,952 ULPs, from the
result when it cannot (`0xb9eb0000` against `0xb9ea3128`). As the silicon rule is not known, the
largest difference between model and silicon in a chain is not known either.

### Global scales and bias

An NVFP4 tensor also has one FP32 global scale: x = E2M1 value x block scale x global scale
(Transformer Engine,
<https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html>).
The global scales are not MMA operands, on NVIDIA or on Atlas. `tcgen05.mma` with `.block_scale`
computes (A * scale_A) * (B * scale_B) + D, and its only scale operands are the two block-scale
matrices (<https://docs.nvidia.com/cuda/parallel-thread-execution/>, PTX ISA 9.4, sections
9.7.18.10.7 and 9.7.18.10.10.1). Transformer Engine folds the product of the two global scales
into the GEMM alpha, which multiplies only the accumulated product
(NVIDIA/TransformerEngine@491d557 `transformer_engine/common/gemm/cublaslt_gemm.cu`, lines
387-390; <https://docs.nvidia.com/cuda/cublas/> section 3.1.4.4), and cuBLASLt adds a bias after
that product (section 3.1.4.1).

The VPU is BF16 only, so Atlas cannot multiply an FP32 accumulator by an FP32 alpha. Its epilogue
contract:

- Each global scale is a power of two, 2^gA for A and 2^gW for W, at least amax / (6 x 448) of
  its tensor, so that every block scale fits in UE4M3 (448 or less). Transformer Engine uses
  amax / (6 x 448) itself as the FP32 global scale.
- After `vmatpop.bf16.acc.nv`, the epilogue multiplies by a BF16 tile of 2^(gA + gW)
  (`vmul.bf16`). A power-of-two factor adds no rounding: BF16(acc) x 2^(gA + gW) equals
  BF16(acc x 2^(gA + gW)) while the result stays in the BF16 normal range.
- The epilogue adds a bias after that scaling (`vadd.bf16`), or it preloads the bias divided by
  2^(gA + gW) into the accumulator (`vmatpush.acc.bf16.nv` or `vmatpush.acc.fp32.nv`). The
  division is exact, and the bias is then the C of the first `vmatmul.acc.nv`.
- All the matmuls into one accumulator (a `vmatmul.acc.nv` chain) have the same gA + gW. Operands
  with a different gA + gW go into a separate chain, and the VPU adds the chains after it scales
  each one.

The FP8 flow uses the same epilogue: `apps/dit/ditflow/kernels.py` pops BF16, multiplies by a
power-of-two tile and then adds the bias.

### Provenance

`fp4_fedp.sv` and `fp4_fedp_acc.sv` are from ucb-bar/nvfp4@0c611e1 `fpga/pe`
(<https://github.com/ucb-bar/nvfp4/tree/0c611e1>; only the path references in `fp4_fedp.sv`'s
header comment are changed). The five block-datapath files (`dotprod_pkg.sv`,
`front_end_nvfp4.sv`, `mul_lane_nvfp4.sv`, `align_nvfp4.sv`, `exact_acc_tree.sv`) are from
sashakatne/nvfp4-dotprod-formal-dv@125db59
(<https://github.com/sashakatne/nvfp4-dotprod-formal-dv>, Apache-2.0,
`LICENSE.nvfp4-dotprod-formal-dv`); only `dotprod_pkg.sv` is changed, as its header says. The
model, `npu_model/configs/nvfp4.py`, is the NVFP4 mode of ucb-bar/nvfp4@0c611e1
`host/nvfp4_model.py` plus Atlas layout helpers. The design and its arithmetic contract are
described at <https://ucb-bar.github.io/nvfp4/> (section 4). Of the three open NVFP4 designs in
the [survey](https://github.com/ucb-bar/nvfp4/blob/0c611e1/docs/survey.md), only
nvfp4-dotprod-formal-dv agreed with the RTX 5090 on every vector (100,008 / 100,008; Allegro
disagreed on 651 / 100,009 NVFP4 vectors and Vortex on 80,855 / 100,008).

### Evidence

The first three rows were measured in Atlas with
`python3 -I baremetal/generators/check_nvfp4_silicon.py` (model) and
`baremetal/generators/nvfp4_silicon/fp4_fedp_vec.sh` (Atlas's `fp4_fedp.sv` in Verilator, mode
0); the vectors are described in `baremetal/generators/nvfp4_silicon/README.md`. The other rows
are ucb-bar/nvfp4@0c611e1 results; their sources are paths in that repo.

| Check | Result | Source |
|---|---|---|
| Model vs RTX 5090 and vs Thor, the 40,000 vectors in `nvfp4_silicon` | 39,999 / 40,000 each; the other is the known gap | `check_nvfp4_silicon.py` |
| RTL vs RTX 5090 and vs Thor, same vectors | 39,999 / 40,000 each, the same vector (RTL = model there); d after 5 cycles on every vector | `fp4_fedp_vec.sh` |
| Model and RTL on the full sets: stored NVFP4 vectors; random NVFP4 seed 1 | 300,025 / 300,025 (RTX 5090); 200,017 / 200,018 (both GPUs, the known gap) | the same tools on ucb-bar/nvfp4@0c611e1 `tcgen05/vectors/*.vec` and on the full seed-1 set (`nvfp4_silicon/README.md`), 2026-10-23 |
| `fp4_fedp` vs RTX 5090 | NVFP4 300,025 / 300,025; random, all four modes, 99,998 / 100,000 (two all-zero-block cases) | `docs/compute.md` (Measured results); `host/run_pe.py` (the random cases cover the four modes); the same cases for `fp4_fedp_dsp` in `results/fp4_fedp_dsp_gpu.log` |
| NVFP4 SoC on its board, m16n8k64 chains vs RTX 5090 | 60,416 / 60,416 outputs | `docs/compute.md` (Measured results) |
| Jetson Thor vs RTX 5090 | the same bits on 920,079 vectors (NVFP4, MXFP4) | `tcgen05/README.md` (Status) |

## Microarchitecture and timing

`Nvfp4Mxu` has 32 `fp4_fedp` lanes, one per output column, and three engines like the other
MXUs: E0 (MREG read port 0: matmul feed, acc push), E1 (read port 1: weight push, acc push) and W
(write ports 0 and 1: pops). It uses MXU slot 0's MREG ports. A matmul feeds one activation row
per cycle into all 32 lanes (2,048 multiply-accumulates per cycle) and writes each accumulator
row 6 cycles later (one input register plus the lanes' 5 cycles).

| Operation | Ports | Steps | Next operation of the same kind |
|---|---|---|---|
| matmul | read 0 | 32 | issues on the last feed cycle: one matmul per 32 cycles |
| weight push | read 1 (or 0) | 32 | one weight push at a time |
| acc push BF16 / FP32 | reads 0 + 1 | 32 / 64 | |
| pop BF16 / FP32 | writes 0 + 1 | 32 / 64 | |

The unit accepts a command (`cmdReady`) when its engine is free and:

- matmul: no weight push still writes its slot, no acc push still writes its buffer, and no pop
  still reads it;
- weight push: no other weight push still writes (one write path), and the matmul feeding now does
  not read the slot;
- acc push: no matmul feeds or writes back to the buffer, and no pop reads it;
- pop: every matmul writeback to the buffer has landed, and no acc push still writes it.

With these rules a weight push to one slot overlaps a matmul on the other, back-to-back
`vmatmul.acc.nv` on one accumulator run at the 32-cycle stride, and the scalar core's interlock
needs no `DELAY` (docs/interlock.md). The interlock also treats `vload.scale` / `vstore.scale`
as VLOAD / VSTORE (same LSU path and VMEM bank), with no WAR exemption for `vload.scale`.
