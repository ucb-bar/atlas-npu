# The FP8 MXU (`mxu1`)

`mxu1` is the inner-product-tree (IPT) MXU. Every configuration has it. Its arithmetic is the
arithmetic of an NVIDIA Blackwell FP8 tensor core, E4M3 x E4M3 with an FP32 accumulator, bit for
bit, in the same way as the NVFP4 MXU follows the NVFP4 tensor cores (`docs/nvfp4-mxu.md`). The
reference model is `npu-model/npu_model/configs/fp8_mma.py`. The silicon vectors are in
`baremetal/generators/fp8_silicon`.

## State and instructions

| State | Size | Contents |
|---|---|---|
| Weight slots `w0`, `w1` | 2 x 32 x 32 E4M3 | row j: output column j's 32 elements |
| Accumulators `acc0`, `acc1` | 2 x 32 x 32 FP32 (4 KiB each) | |

The 7 instructions keep their v1 encodings. A BF16 tile T is the register pair {m[r], m[r+1]}:
m[r] holds columns 0-15 of each row, m[r+1] columns 16-31. G(e) = 2^(e - 127), with e the
8-bit value of the e-register (e = 255 is 2^128).

| Instruction | Operation | Steps |
|---|---|---|
| `vmatpush.weight.mxu1 w(vd), m(vs1)` | w[vd] = m[vs1] | 32 |
| `vmatpush.acc.bf16.mxu1 acc(vd), m(vs1)` | acc[vd] = FP32(T(vs1)): BF16 bits << 16, exact | 32 |
| `vmatpush.acc.fp8.mxu1 acc(vd), m(vs1)` | acc[vd] = FP32(m[vs1]), exact, with subnormals; NaN gives `0x7FFFFFFF` | 32 |
| `vmatpop.bf16.acc.mxu1 m(vd), acc(vs2)` | T(vd) = BF16_RNE(acc[vs2]) | 32 |
| `vmatpop.fp8.acc.mxu1 m(vd), acc(vs2), e(es1)` | m[vd] = E4M3_RN_SATFINITE(acc[vs2] x G(e[es1])) | 32 |
| `vmatmul.mxu1 acc(vd), m(vs1), w(vs2)` | acc[vd][i][j] = MMA(A = row i of m[vs1], B = row j of w[vs2], C = +0) | 32 |
| `vmatmul.acc.mxu1 acc(vd), m(vs1), w(vs2)` | as `vmatmul.mxu1`, C = acc[vd][i][j] | 32 |

The conversions are those of the Blackwell conversion instructions, measured on the RTX 5090
(`fp8_silicon/README.md`, Conversions; npu_model `configs/cvt.py`; RTL `atlas.common.Cvt`):

- BF16_RNE: round to nearest even, with BF16 subnormals. Overflow gives +-Inf. Every NaN gives
  `0x7FFF`. A zero keeps its sign. This is `cvt.rn.bf16.f32`.
- E4M3_RN_SATFINITE: x x G(e) is exact, and the result rounds once to nearest even, with E4M3
  subnormals. A magnitude above 448 and +-Inf give +-448. Every NaN gives `0x7F`. A zero keeps
  its sign. This is `cvt.rn.satfinite.e4m3x2.f32` of x x G(e).
- E4M3 to FP32 is exact. `0x7F` and `0xFF` give `0x7FFFFFFF`.

## Arithmetic

Each output element of a matmul is one fused dot-add over K = 32:

```text
acc[i][j] = C + sum over k = 0 .. 31 of A[i][k] W[j][k]      (C FP32, A and W E4M3)
```

An E4M3 code has sign bit 7, exponent field E (bits 6:3) and mantissa m (bits 2:0). E = 15,
m = 7 (`0x7F`, `0xFF`) is NaN. E4M3 has no infinity. The other codes are +-(S / 8) x 2^e:

| Code | S | e | Value |
|---|---|---|---|
| E > 0 | 8 + m | E - 7 | (1 + m / 8) x 2^(E - 7), up to 448 (`0x7E`) |
| E = 0, m > 0 | m | -6 | m / 8 x 2^-6 (subnormal) |
| E = 0, m = 0 | 0 | -6 | +-0 |

C, with biased exponent Ec and fraction f, is +-Sc x 2^(ec - 23): Sc = 2^23 + f and
ec = Ec - 127 for Ec > 0, and Sc = f and ec = -126 for Ec = 0.

1. Special values. If an element of A or W is NaN, or C is NaN, d = `0x7FFFFFFF`. Else, if C is
   +-Inf, d = C.
2. Products. Each product is exact and not normalized: p_k = +-Sa Sb / 64 x 2^(ea + eb). For
   example, 1.5 x 1.5 is 2.25 x 2^0, not 1.125 x 2^1.
3. Alignment exponent. emax is the largest of ec and the exponents ea + eb of the nonzero
   products. A zero product does not set emax.
4. Cut. Let u = 2^(emax - 25). Each nonzero product and C becomes the integer
   t = sign x floor(|x| / u): a cut toward zero on the magnitude.
5. Sum. T = t_C + the sum of the 32 t_k, exactly. |T| < 2^32.
6. Output. d = T x u, cut toward zero to FP32 (24 significant bits). T = 0 gives +0.

If no product is nonzero, d = C exactly, except that -0 becomes +0. These results follow:

- FP32 overflow does not occur, and a subnormal d occurs only as such a C.
- An exact zero is +0, also for C = -0 with only -0 products.
- Error bound: the term that sets emax is not cut, so the cuts lose less than 32u in total, and
  the output cut loses less than 1 ULP of d.
- A chain of `vmatmul.acc.mxu1` over K = 32n equals a chain of n Blackwell FP8 MMAs bit for bit
  (Evidence).

Target instruction. The measured target is `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
on the RTX 5090 (sm_120), and the same instruction with `kind::f8f6f4`. Both compile to SASS
`QMMA.16832.F32.E4M3.E4M3`. MMA-Sim (arXiv 2511.10909, Table III and section IV-C) gives the
same algorithm, a fused dot-add with 25 fractional bits, for this instruction and for
`tcgen05.mma` `kind::f8f6f4` (datacenter Blackwell, Jetson Thor). Atlas did not measure tcgen05.
`mma.sync` FP8 on B200 is not this contract: it goes through FP16 HMMA with 16-element sums
(Khattak and Mikaitis, arXiv 2512.07004).

## Evidence

All comparisons are bit for bit. The vectors are described in
`baremetal/generators/fp8_silicon/README.md`.

| Check | Result | Command |
|---|---|---|
| Model vs RTX 5090, single steps (stored subsets of the full sets) | 140,129 / 140,129 | `python3 -I baremetal/generators/check_fp8_silicon.py` |
| Model vs RTX 5090, tile chains (14 tiles, K = 128, D after each step) | 57,344 / 57,344 | the same |
| Model vs RTX 5090, full sets (single steps) | 850,129 / 850,129 | `fp8_vectors.py gen`, scratch run of 2026-10-24 |
| RTL lane (`AnchorAccumulationTree`) vs RTX 5090 | 140,129 / 140,129 | `./mill atlas.test.testOnly atlas.ipt.Fp8MmaSiliconTest` |
| RTL tile chains through the FP32 accumulators vs RTX 5090 | 57,344 / 57,344 | the same |
| RTL pop and push conversions vs RTX 5090 | every vector of `pop`, `push` and both ends of every FP32 run | `./mill atlas.test.testOnly atlas.vector.Fp8CvtSiliconTest` |
| `ditflow.numerics.ipt_matmul` vs RTX 5090 | 197,473 / 197,473 | scratch run of 2026-10-25 |
| npu_model vs RTL, mxu1 port traces and the lane on 512 cases | every event and output | `npu-model/tests/test_rtl_matrix_traces.py`, `test_rtl_arithmetic.py` |

In the chain check, each tile goes through `InnerProductTreesTop`: `vmatpush.acc.bf16.mxu1` of
C (C is a BF16 value), then for each of the 4 steps a weight push and `vmatmul.acc.mxu1`. The
test reads the FP32 rows that each matmul writes into the accumulator. `fp8_tile.cu` computes the
same chains with 8 `mma.sync` m16n8k32 per step. The tiles take their rows from the generators
of the single-step sets, so they include NaN, Inf, subnormal and zero C, all-zero rows and
cancellation.

The silicon README gives, for each alternative rule (for example F = 24 or 26, normalized
subnormal inputs, zero products that set emax, floor instead of a cut toward zero, a rounded
output), the number of vectors that reject it.

## Changes from v1

| Item | v1 | ISA 2.0 |
|---|---|---|
| E4M3 subnormal input | zero | kept: S = m, e = -6 |
| E4M3 NaN input | 480 | d = `0x7FFFFFFF` |
| Products | normalized | not normalized |
| Alignment | 32-bit window from the largest normalized exponent | F = 25 below emax, 33-bit sum |
| Accumulators and C | BF16, 2 KiB each | FP32, 4 KiB each |
| Matmul output | BF16, round to nearest even; subnormal and NaN gave 0, Inf gave +-`0x7F7F` | FP32, cut toward zero; NaN `0x7FFFFFFF`, Inf C passes |
| `vmatpush.acc.fp8` | to BF16; subnormal and NaN gave a signed zero | to FP32, exact |
| `vmatpop.fp8` | from BF16; flushed below 2^-6; gave `0x7F` for a value that rounded to 480 | E4M3_RN_SATFINITE |
| `vmatpop.bf16` | copy of the BF16 accumulator | BF16_RNE of the FP32 accumulator |
| e = 255 in `vmatpop.fp8` | 2^127 | 2^128 |

## Microarchitecture and timing

Each of the 32 lanes (`AnchorAccumulationTree`) computes one output column; a matmul feeds one
row of A per cycle, so `mxu1` does 1,024 multiply-accumulates per cycle. A lane has these
stages:

| Stage | Logic |
|---|---|
| 0, decode and multiply | S = (E != 0):m; 4 x 4-bit products P = Sa Sb; exponent sums max(Ea, 1) + max(Eb, 1); zero and NaN flags |
| 1, emax | maximum of the 32 product exponents (zero products masked) and C's exponent |
| 2, align | P at bits 26..19, C's 24-bit significand at bits 25..2, both shifted right by emax minus their exponent (32 or more gives 0); the magnitude is cut, then negated |
| 3, sum | 33 terms into a 33-bit two's-complement sum |
| 4, normalize | leading-one detect on 32 bits, keep 24 bits; biased exponent = emax + 6 - leading zeros; the specials and the C bypass; output register |

`InnerProductTreeParams.pipelineCuts` places registers after stages; the default `Set(1)`
registers after stage 1. The lane latency is 2 cycles, as in v1, so the timing is unchanged: a
matmul accepted in cycle T reads A row r at T + r and writes accumulator row r at T + 3 + r.

Logic depth, from the same Chisel parameters: Yosys 0.70 `synth_xilinx -family xcup` of one
lane, longest path (`ltp`, LUT and CARRY4 nodes) of each pipeline stage:

| Lane | Inputs to the stage-1 register | Stage-1 register to output register | LUTs | CARRY4 | FFs |
|---|---|---|---|---|---|
| v1 (BF16, normalized products, HardFloat `INToRecFN`) | 45 | 61 | 8,316 | 726 | 462 |
| ISA 2.0 | 27 | 39 | 6,570 | 450 | 522 |

The new lane is shorter in both stages: the product exponent does not depend on the product, the
output has no rounding increment and no subnormal shifter, and the emax compare is on 8 bits. The
critical path of the v1 FPGA build ran from an MREG bank read address into the first stage of a
lane (`standalone/README.md`, Results); that stage now has 27 nodes instead of 45.
The pops now convert from FP32 in the sequencer (`atlas.common.Cvt`), between the accumulator
read and the MREG write: 26 nodes from register to register for `Cvt.e4m3RnSatfinite`, against 13
for v1's `bf16ToE4m3`, below the lane's 39. The BF16 pop is one adder: FP32 and BF16 have the
same exponent field, so `(x + 0x7FFF + x[16]) >> 16` rounds to nearest even with subnormals and
overflow to Inf, and only NaN needs a mux. This equals `cvt.rn.bf16.f32` on all 2^32 inputs.
No Vivado run checked the timing on the FPGA.

The unit accepts a command (`cmdReady`) under the v1 rules. For a matmul they are: the read port
is free or on its last feed cycle, the in-flight FIFO (2 entries; an entry retires when the
matmul's 32nd row lands) has room, no earlier matmul on the same accumulator still waits for its
row 0 to land, no push writes its weight slot or accumulator, and no pop reads the accumulator
for `vmatmul.acc`. With a latency of 2, the row-0 rule costs nothing at the 32-cycle stride:
back-to-back `vmatmul.acc.mxu1` on one accumulator issue every 32 cycles, and row r of the next
matmul reads C at T + 32 + r, after the previous matmul wrote it at T + 3 + r.
