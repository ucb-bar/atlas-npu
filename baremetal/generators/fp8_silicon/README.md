# FP8 silicon vectors

NVIDIA tensor-core outputs for 32-element E4M3 dot products with an FP32 C. They check the FP8
MXU arithmetic contract (`docs/fp8-mxu.md`): `../check_fp8_silicon.py` compares the model
(`npu-model/npu_model/configs/fp8_mma.py`) with them, and `atlas.ipt.Fp8MmaSiliconTest` compares
the RTL, bit for bit. The contract is the fused dot-add (FDA) of MMA-Sim (arXiv 2511.10909,
section IV-C) with F = 25 fractional bits, as the model's docstring states it.

The section [Tile chains](#tile-chains) gives K = 128 chains of 32 x 32 tiles, and the section
[Conversions](#conversions) gives the vectors of the FP8 and BF16 conversions.

## Silicon

- GPU: NVIDIA GeForce RTX 5090 (sm_120), driver 595.99.02, VBIOS 98.02.2E.80.05, Linux
  7.0.0-38. CUDA 13.2 (`nvcc` V13.2.78). Run on 2026-10-24.
- `fp8_mma.cu` runs each vector as one `mma.sync` m16n8k32 E4M3 x E4M3 + FP32. Every row of A
  and every column of B holds the vector, so all 128 outputs are the same dot product. The kernel
  writes `0x7FBADBAD` if two outputs differ. This did not occur.
- Kind 0 is `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`. Kind 1 is
  `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32`. `ptxas` accepts both
  for `sm_120a`, and both give the same SASS instruction, `QMMA.16832.F32.E4M3.E4M3`
  (`cuobjdump -sass`). This is the native FP8 tensor-core path, not HMMA. On B200,
  `mma.sync` FP8 goes through FP16 HMMA (Khattak and Mikaitis, arXiv 2512.07004), so B200
  `mma.sync` results are not this contract.
- The two kinds gave the same output on all 850,129 vectors. The files keep one column,
  `rtx5090`, from kind 0.
- No Jetson Thor or other tcgen05 GPU (sm_100, sm_110) is reachable from this workspace, so there
  is no `tcgen05.mma` `kind::f8f6f4` column. The `thor` column of `../nvfp4_silicon` came from a
  self-contained bundle (ucb-bar/nvfp4@0c611e1 `tcgen05/`), which ran on a Jetson AGX Thor
  outside this workspace and returned `tcgen05_results.zip`. MMA-Sim reports the same rule (FDA,
  F = 25) for UTCQMMA, the tcgen05 FP8 instruction. These vectors do not test it.

## Files

| File | Vectors | Full set | Content |
|---|---|---|---|
| `directed.vec.gz` | 129 | 129 | one rule each; header `names` gives the rule (`fp8_vectors.py`, `directed()`) |
| `bits_s1.vec.gz` | 15,000 | 100,000 | random bytes for a and b (NaN codes included), random 32-bit C |
| `normal_s2.vec.gz` | 20,000 | 150,000 | E4M3 exponent fields from N(7, 2); C from N(0, 8) |
| `wide_s3.vec.gz` | 20,000 | 150,000 | E4M3 exponent fields 0 to 15; C from 2^-40 to 2^34, or zero |
| `cancel_s4.vec.gz` | 25,000 | 150,000 | elements 16 to 31 cancel 0 to 15, then up to 3 change; C cancels the rest, or is small or zero |
| `cspecial_s5.vec.gz` | 30,000 | 100,000 | special, subnormal, power-of-two and extreme C; a quarter with only zero products |
| `sparse_s6.vec.gz` | 15,000 | 100,000 | 60% signed-zero elements; C small or zero |
| `subnormal_s7.vec.gz` | 15,000 | 100,000 | 60% subnormal elements; C small or zero |

`fp8_vectors.py gen` writes the full sets. The model equals the RTX 5090 on all 850,129 vectors of
the full sets and on all 140,129 stored vectors. A separate check, not stored here, ran chains of 4
dependent `mma.sync` (K = 128, D fed back as C): the chained model equals silicon on 240,000 of
240,000 outputs (60,000 records).

Each stored subset keeps these vectors of its full set:

- every vector where the model differs from silicon, or the two kinds differ (none)
- every vector that rejects one of the rare alternative rules in the table below (normalized
  subnormal inputs, zero products set emax, IEEE signs of zero)
- every subnormal and every infinite output

`numpy.random.default_rng(1).choice(rest, count - special, replace=False)` of the other vectors
(`rest`) fills each file. The header records
the rule (`selection`), the source (`subset_of`), the number of rule-kept vectors (`special`) and
the full-set index of every vector (`indices`).

Each alternative rule makes one change to the contract. The table gives the number of vectors
whose RTX 5090 output rejects the change, that is, where the changed model gives other bits:

| Alternative rule | Full sets (850,129) | Stored (140,129) |
|---|---|---|
| F = 24 | 114,982 | 16,955 |
| F = 26 | 68,943 | 10,496 |
| Subnormal inputs normalized before the product (as an FP16 conversion does) | 313 | 313 |
| Zero products set emax (exponent e_a + e_b, as for other products; no pass-through of C) | 15,312 | 15,312 |
| Terms cut toward minus infinity (floor), not toward zero | 216,123 | 31,885 |
| C not cut; only the sum is rounded | 131,076 | 20,462 |
| Two fused sums of 16 elements, the first cut to FP32 | 147,988 | 21,808 |
| FP32 output rounded to nearest even, not toward zero | 169,949 | 23,889 |
| IEEE signs of zero (C = -0 with only -0 products gives -0) | 4 | 4 |

## Format

A file is a gzip-compressed `fp8-vectors-v1` file. It has one JSON header line, then `count`
68-byte records (`a[32] b[32] c[4]`; byte k of a and b is E4M3 element k; c is FP32,
little-endian), then one little-endian uint32 column per entry of `columns`. A record is the input
format of `fp8_mma`.

## Commands

```bash
python3 -I baremetal/generators/check_fp8_silicon.py                 # model vs silicon
./mill atlas.test.testOnly atlas.ipt.Fp8MmaSiliconTest               # RTL vs silicon (Verilator)
```

The RTL test runs every vector through `AnchorAccumulationTree`, the `mxu1` lane, with its
default pipeline: 140,129 vectors, 0 differences (about 15 s).

On an sm_120 GPU, in this directory (each full set is at most 10.2 MB):

```bash
/usr/local/cuda-13.2/bin/nvcc -O2 -gencode arch=compute_120a,code=sm_120a -o fp8_mma fp8_mma.cu
cuobjdump -sass fp8_mma | grep MMA                                   # QMMA.16832.F32.E4M3.E4M3
python3 -I fp8_vectors.py gen full                                   # full/<set>.bin
./fp8_mma 0 full/wide_s3.bin wide_s3.out                             # one uint32 per vector
python3 -I fp8_vectors.py records wide_s3.vec.gz wide_s3.bin         # the records of a subset
./fp8_mma 0 full/directed.bin directed.k0                            # and kind 1 to directed.k1
python3 -I fp8_vectors.py pack-directed directed.k0 directed.k1 directed.vec.gz
```

`pack-directed` checks that the two kinds agree and writes `directed.vec.gz`. The repository has
no packer for the subsets of the random sets: their selection needs the alternative-rule models
of the table above, which are not in the repository.

CUDA 12 cannot target `sm_120a`. With CUDA 13.2, `-arch=sm_120a` also embeds `compute_120` PTX,
and `ptxas` then rejects kind 1. Use `-gencode` as above.

## Tile chains

`chain_s8.tiles.gz` holds 14 chains of 4 steps of 32 x 32 x 32 tiles (K = 128). Step s computes
D_s = D_(s-1) + A_s W_s^T with D_(-1) = C, as `vmatmul.acc.mxu1` does on an accumulator: row j of
W_s is output column j. `fp8_tile.cu` runs each step as 8 `mma.sync` m16n8k32 E4M3 x E4M3 + FP32
(SASS `QMMA.16832.F32.E4M3.E4M3`) and feeds D back as C. The GPU, the driver and the CUDA
version are those of [Silicon](#silicon); the run was on 2026-10-25.

`fp8_vectors.py tiles` makes the inputs (`tiles(numpy.random.default_rng(8), 14)`): tile t takes
the rows of A and W and its C from the generator `TILE_KINDS[t % 7]` of the single-step sets
(normal, wide, cancel, cspecial, sparse, subnormal, bits), so the tiles include NaN, Inf,
subnormal and zero C, all-zero rows and cancellation. C is cut to BF16, because
`vmatpush.acc.bf16.mxu1` loads it.

- `check_fp8_silicon.py` feeds `fp8_mma.matmul` its own output from step to step: 57,344 of
  57,344 outputs equal the RTX 5090 (5,433 NaN, 492 Inf, 221 subnormal).
- `atlas.ipt.Fp8MmaSiliconTest` runs each tile through `InnerProductTreesTop`: a BF16 push of
  C, then per step a weight push and `vmatmul.acc.mxu1`. The FP32 rows that each matmul writes
  into the accumulator equal the GPU's D on all 57,344 outputs.

Format: a gzip-compressed `fp8-tiles-v1` file. It has one JSON header line (`count` tiles,
`steps`), then `count` records of c[32][32] (FP32, little-endian), a[steps][32][32] and
w[steps][32][32] (E4M3), then one column per entry of `columns`: count x steps x 32 x 32 FP32
values, D after each step. `fp8_vectors.read_tiles` reads it.

```bash
python3 -I fp8_vectors.py tiles tiles.bin                            # on the host
/usr/local/cuda-13.2/bin/nvcc -O2 -gencode arch=compute_120a,code=sm_120a -o fp8_tile fp8_tile.cu
./fp8_tile tiles.bin tiles.out                                       # on the GPU
python3 -I fp8_vectors.py pack-tiles tiles.out chain_s8.tiles.gz     # on the host
```

## Conversions

The `*.cvt.gz` files hold RTX 5090 outputs of the conversions of the Atlas FP8 path (ISA v2,
section 5.2). `../check_cvt_silicon.py` compares the model (`npu-model/npu_model/configs/cvt.py`)
with them, bit for bit. The GPU, the driver, the CUDA version and the date are the same as in
[Silicon](#silicon).

`fp8_cvt.cu` runs each conversion through these instructions:

- `vpack.bf16.fp8` and `vmatpop.fp8.acc.mxu1`: x * G(e), then `cvt.rn.satfinite.e4m3x2.f32`
  (SASS `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C`).
- `vunpack.fp8.bf16`: `cvt.rn.f16x2.e4m3x2` (`F2FP.F16.E4M3.UNPACK_B`), `cvt.f32.f16`
  (`HADD2.F32`), x * G(e), then `cvt.rn.bf16.f32` (`F2FP.BF16.F32.PACK_AB`).
- `vmatpush.acc.fp8.mxu1`: `cvt.rn.f16x2.e4m3x2`, then `cvt.f32.f16`.
- The BF16 pops: `cvt.rn.bf16.f32`. In the `bf16` mode it compiles to `F2F.BF16.F32`.
  The `bf16x2` mode uses `cvt.rn.bf16x2.f32` (`F2FP.BF16.F32.PACK_AB`). The two SASS forms give
  the same output for all 2^32 FP32 inputs. The scaled BF16 pops round acc * G(e) once. The GPU
  has no instruction for this: FMUL, then `cvt.rn.bf16.f32`, rounds two times when the FP32
  product is below 2^-126 and not exact. For an accumulator that holds E4M3 values the product
  is exact, so the `vunpack` vectors give the scaled pop's result: `atlas.ipt.Mxu1TransferTest`
  checks the `mxu1` scaled pop with them at every e.

The harness forms x * G(e) = x * 2^(e - 127) in FP32 with FMUL (`__fmul_rn`). For e = 255 it
uses two FMULs (2^127, then 2), because 2^128 is not an FP32 value. The model applies the scale
exactly and rounds once. It gives the same result as the GPU on every vector.

The GPU gives these results. `cvt.py` states them as rules:

- E4M3 (`cvt.rn.satfinite.e4m3x2.f32`): round to nearest, ties to even, with E4M3 subnormals.
  A value above 448 after rounding and +-Inf give +-448 (`0x7E`, `0xFE`). Every NaN gives `0x7F`,
  for both signs and all payloads. A value that rounds to zero keeps its sign (`0x00`, `0x80`).
- BF16 (`cvt.rn.bf16.f32`): round to nearest, ties to even, with BF16 subnormals (no flush to
  zero). A value that rounds above the largest BF16 value gives +-Inf. Every NaN gives `0x7FFF`,
  for both signs and all payloads. A zero keeps its sign.
- E4M3 to FP32: exact, with subnormals. `0x80` gives -0 (`0x80000000`). `0x7F` and `0xFF` give
  `0x7FFFFFFF`.
- The two halves of each x2 conversion gave the same result on every input.

| File | Vectors | Content |
|---|---|---|
| `vpack.cvt.gz` | 16,777,216 | E4M3 of x * G(e) for every BF16 x at every e |
| `vunpack.cvt.gz` | 65,536 | BF16 of x * G(e) for every E4M3 x at every e |
| `push.cvt.gz` | 256 | FP32 of every E4M3 x |
| `pop.cvt.gz` | 489,344 | E4M3 of x * G(e) for FP32 x at E4M3 boundaries, special and random x |
| `e4m3_runs.cvt.gz` | 256 runs | E4M3 of every FP32 x (2^32 inputs) |
| `bf16_runs.cvt.gz` | 65,284 runs | BF16 of every FP32 x (2^32 inputs) |

`cvt_vectors.py` (`pop_inputs()`) makes the inputs of `pop.cvt.gz`:

- 385,248 vectors. At every e, each FP32 x for which x * G(e) is an E4M3 boundary: a positive
  finite E4M3 value, the midpoint of two neighbours (a tie, from 2^-10 up), or 464 (the tie above
  448). Also the FP32 values one ULP below and above each x, and the negatives of all of them.
- 4,096 vectors. The 16 values of `SPECIAL` at every e: +-0, +-Inf, 4 NaNs, and with both signs
  the smallest and the largest FP32 subnormal, the smallest FP32 normal and FLT_MAX.
- 100,000 vectors with a random e (`numpy.random.default_rng(1)`): 50,000 x with x * G(e) from
  2^-12 up to 2^10, and 50,000 random FP32 bit patterns. If the exponent field of such an x
  would be below 1, x is an FP32 subnormal.

The model equals the GPU on all 17,463,432 comparisons of the stored files: every vector of
`vpack`, `vunpack`, `push` and `pop`, and the first and the last input of each run. A separate
check, not stored here, compared the model with the two runs files at each of the 2^32 FP32
inputs: 0 differences for E4M3 and 0 for BF16.

The RTL equals the GPU too. `atlas.vector.Fp8CvtSiliconTest` runs `FP8Pack`, `FP8Unpack` and
`atlas.common.Cvt` in Verilator on every vector of `vpack`, `vunpack`, `pop` and `push`, and on
the first and the last input of each run: 0 differences (`./mill atlas.test.testOnly
atlas.vector.Fp8CvtSiliconTest`, about one minute). The BF16 runs also check `Cvt.bf16Rne(x, e)`
at e = 127, the rounder of the BF16 pops of both MXUs. Two separate checks, not stored here, cover
all 2^32 FP32 inputs: the one-adder BF16 rounder that the pops used before equals `cvt.py`, and
`Cvt.bf16Rne(x, 127)` equals that rounder (0 differences each).

### Format

A file is a gzip-compressed `cvt-vectors-v1` file. It has one JSON header line, then one column
for each entry of `columns`, in that order. A column has `count` little-endian values of the
numpy type that the entry gives. `vpack`, `vunpack` and `push` store only the output column,
`rtx5090`. The header field `inputs` gives the inputs of vector i, for example e = i >> 16 and
x = BF16 bits i & 0xFFFF for `vpack`. `pop` stores x (FP32 bits), e and `rtx5090`. A runs file
stores x and `rtx5090`: each FP32 x from x[k] up to x[k + 1] - 1 converts to `rtx5090[k]`. The
last run goes up to `0xFFFFFFFF`.

### Commands

```bash
python3 -I baremetal/generators/check_cvt_silicon.py                 # model vs silicon
```

On an sm_120 GPU, in this directory:

```bash
/usr/local/cuda-13.2/bin/nvcc -O2 -gencode arch=compute_120a,code=sm_120a -o fp8_cvt fp8_cvt.cu
cuobjdump -sass fp8_cvt | grep -E 'F2F|HADD2|FMUL'
./fp8_cvt vpack vpack.out                       # also vunpack, push, e4m3, bf16 and bf16x2
python3 -I cvt_vectors.py gen in
./fp8_cvt pop in/pop.bin pop.out
python3 -I cvt_vectors.py pack . .              # the *.cvt.gz files from the *.out files
```

Each mode takes less than one second. `vpack.out`, `vunpack.out`, `push.out` and `pop.out` are
the `rtx5090` columns. The runs modes write {uint32 x, uint32 d} records. `cvt_vectors.py pack`
checks that the `bf16` and `bf16x2` runs agree and writes the six files.
