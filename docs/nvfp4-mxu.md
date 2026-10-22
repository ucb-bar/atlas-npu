# Configurations and the NVFP4 MXU

## Configurations

`atlas.common.AtlasConfigs` names three configurations. MXU slot 0 holds the systolic array or
the NVFP4 MXU; slot 1 is always the inner-product-tree (IPT) MXU.

| Name | `AtlasParams` | MXUs | Chipyard fragment |
|---|---|---|---|
| `default` | `AtlasParams()` | systolic array (`.mxu0`) + IPT (`.mxu1`) | `WithAtlasTile()` |
| `ipt` | `hasSystolicArray = false` | IPT only | `WithAtlasTileIptOnly` |
| `ipt-nvfp4` | `hasSystolicArray = false, nvfp4 = Some(Nvfp4MxuParams())` | IPT + NVFP4 (`.nv`) | `WithAtlasTileNvfp4` |

An instruction for a unit the configuration does not have is illegal: the core halts and
CSR STATUS reads 3 (halted, illegal instruction). The standalone build selects a configuration
with `EmitAtlasCore --config=NAME`; sys_regs FEATURES reports it ([1] systolic array, [2] IPT,
[3] NVFP4).

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

Each output element is one `fp4_fedp` dot product from the NVFP4 repo
(`src/main/resources/vsrc/nvfp4`): the arithmetic of NVIDIA tensor cores, as measured on an RTX
5090 (`mma.sync` m16n8k64) and Jetson Thor (`tcgen05`):

```text
acc[r][j] = acc[r][j] + sum over blocks g of (sum of 16 E2M1 products A[r][16g+i] W[j][16g+i]) x sA[r][g] x sW[j][g]
```

The 16-element block sums are exact; every term, the old accumulator included, is cut toward zero
at 2^(emax - 35), and the sum is cut toward zero to FP32 once. A NaN scale or accumulator gives
`0x7FFFFFFF`, an infinite accumulator passes through, and a row of all-zero products returns the
accumulator (with -0 becoming +0). So a chain of `vmatmul.acc.nv` over K = 64n equals a chain of
n NVIDIA m16n8k64 MMAs bit for bit. The block sums come from
sashakatne/nvfp4-dotprod-formal-dv (Apache-2.0).

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
