# FP8 silicon vectors

NVIDIA tensor-core outputs for 32-element E4M3 dot products with an FP32 C. They check the FP8
MXU arithmetic contract: `../check_fp8_silicon.py` compares the model
(`npu-model/npu_model/configs/fp8_mma.py`) with them, bit for bit. The contract is the fused
dot-add (FDA) of MMA-Sim (arXiv 2511.10909, section IV-C) with F = 25 fractional bits, as the
model's docstring states it.

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
- The two kinds gave the same output on all 850,126 vectors. The files keep one column,
  `rtx5090`, from kind 0.
- No Jetson Thor or other tcgen05 GPU (sm_100, sm_110) is reachable from this workspace, so there
  is no `tcgen05.mma` `kind::f8f6f4` column. The `thor` column of `../nvfp4_silicon` came from a
  self-contained bundle (ucb-bar/nvfp4@0c611e1 `tcgen05/`), which ran on a Jetson AGX Thor
  outside this workspace and returned `tcgen05_results.zip`. MMA-Sim reports the same rule (FDA,
  F = 25) for UTCQMMA, the tcgen05 FP8 instruction. These vectors do not test it.

## Files

| File | Vectors | Full set | Content |
|---|---|---|---|
| `directed.vec.gz` | 126 | 126 | one rule each; header `names` gives the rule (`fp8_vectors.py`, `directed()`) |
| `bits_s1.vec.gz` | 15,000 | 100,000 | random bytes for a and b (NaN codes included), random 32-bit C |
| `normal_s2.vec.gz` | 20,000 | 150,000 | E4M3 exponent fields from N(7, 2); C from N(0, 8) |
| `wide_s3.vec.gz` | 20,000 | 150,000 | E4M3 exponent fields 0 to 15; C from 2^-40 to 2^34, or zero |
| `cancel_s4.vec.gz` | 25,000 | 150,000 | elements 16 to 31 cancel 0 to 15, then up to 3 change; C cancels the rest, or is small or zero |
| `cspecial_s5.vec.gz` | 30,000 | 100,000 | special, subnormal, power-of-two and extreme C; a quarter with only zero products |
| `sparse_s6.vec.gz` | 15,000 | 100,000 | 60% signed-zero elements; C small or zero |
| `subnormal_s7.vec.gz` | 15,000 | 100,000 | 60% subnormal elements; C small or zero |

`fp8_vectors.py gen` writes the full sets. The model equals the RTX 5090 on all 850,126 vectors of
the full sets and on all 140,126 stored vectors. A separate check, not stored here, ran chains of 4
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

| Alternative rule | Full sets (850,126) | Stored (140,126) |
|---|---|---|
| F = 24 | 114,982 | 16,955 |
| F = 26 | 68,943 | 10,496 |
| Subnormal inputs normalized before the product (as an FP16 conversion does) | 313 | 313 |
| Zero products set emax (exponent e_a + e_b, as for other products; no pass-through of C) | 15,312 | 15,312 |
| Terms cut toward minus infinity (floor), not toward zero | 216,123 | 31,885 |
| C not cut; only the sum is rounded | 131,076 | 20,462 |
| Two fused sums of 16 elements, the first cut to FP32 | 147,988 | 21,808 |
| FP32 output rounded to nearest even, not toward zero | 169,949 | 23,889 |
| IEEE signs of zero (C = -0 with only -0 products gives -0) | 1 | 1 |

## Format

A file is a gzip-compressed `fp8-vectors-v1` file. It has one JSON header line, then `count`
68-byte records (`a[32] b[32] c[4]`; byte k of a and b is E4M3 element k; c is FP32,
little-endian), then one little-endian uint32 column per entry of `columns`. A record is the input
format of `fp8_mma`.

## Commands

```bash
python3 -I baremetal/generators/check_fp8_silicon.py                 # model vs silicon
```

On an sm_120 GPU, in this directory (each full set is at most 10.2 MB):

```bash
/usr/local/cuda-13.2/bin/nvcc -O2 -gencode arch=compute_120a,code=sm_120a -o fp8_mma fp8_mma.cu
cuobjdump -sass fp8_mma | grep MMA                                   # QMMA.16832.F32.E4M3.E4M3
python3 -I fp8_vectors.py gen full                                   # full/<set>.bin
./fp8_mma 0 full/wide_s3.bin wide_s3.out                             # one uint32 per vector
python3 -I fp8_vectors.py records wide_s3.vec.gz wide_s3.bin         # the records of a subset
```

CUDA 12 cannot target `sm_120a`. With CUDA 13.2, `-arch=sm_120a` also embeds `compute_120` PTX,
and `ptxas` then rejects kind 1. Use `-gencode` as above.
