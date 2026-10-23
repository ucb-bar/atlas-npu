# NVFP4 silicon vectors

NVIDIA tensor-core outputs for 64-element NVFP4 dot products (E2M1 elements, one UE4M3 scale per
16, FP32 C), used to check the NVFP4 MXU arithmetic: `../check_nvfp4_silicon.py` checks the
model (`npu-model/npu_model/configs/nvfp4.py`), and `fp4_fedp_vec.sh` checks the RTL
(`src/main/resources/vsrc/nvfp4/fp4_fedp.sv`) in Verilator.

The full sets are in ucb-bar/nvfp4@0c611e1 `tcgen05/vectors`
(<https://github.com/ucb-bar/nvfp4/tree/0c611e1/tcgen05/vectors>). In that repo, their inputs
are the FPGA test cases of three third-party NVFP4 designs (`host/export_tcgen05_vectors.py`);
the `rtx5090` column is one `mma.sync` m16n8k64 per vector on an RTX 5090
(`host/gpu/nvfp4_mma.cu`, sm_120a), and the `thor` column is one `tcgen05.mma` per vector on a
Jetson AGX Thor (`tcgen05/nvfp4_tcgen05_check.py`, sm_110a), from
`tcgen05/tcgen05_results.zip`. The sources in the table are paths in that repo too.

| File | Vectors | Full set | Source |
|---|---|---|---|
| `allegro_nvfp4_s7.vec.gz` | 10,000 | 100,009 | `tcgen05/vectors/allegro_nvfp4_s7.vec` |
| `dotprod_formal_s11.vec.gz` | 10,000 | 100,008 | `tcgen05/vectors/dotprod_formal_s11.vec` (16 elements, zero-padded) |
| `vortex_fedp_s13.vec.gz` | 10,000 | 100,008 | `tcgen05/vectors/vortex_fedp_s13.vec` (16 elements, zero-padded) |
| `random_nvfp4_seed1.vec.gz` | 10,000 | 200,018 | `random_vectors(200000, 1, KIND_NVFP4)` of `tcgen05/nvfp4_tcgen05_check.py` |

`random_nvfp4_seed1` is the Thor random run. Its Thor outputs are in `tcgen05_results.zip`. Its
RTX 5090 outputs (run 2026-10-07) were never committed to ucb-bar/nvfp4; this subset keeps 10,000
of them. The two GPUs agree on all 200,018 vectors. Vector 15 is the known model gap (an all-zero
16-element block sets the alignment; Atlas `docs/nvfp4-mxu.md`).

Each subset keeps the directed vectors (the first 9, 8, 8 and 18), every vector where two stored
outputs of the full set disagree or the model differs from silicon (for Allegro: its FPGA and
MMA-Sim columns; Vortex's FPGA column, wrong on 81% of its set, is not used), every vector with
an all-zero 16-element block (64-element sets), and every subnormal output. A
`random.Random(1).sample()` of the other vectors fills each file to 10,000. The header records
the rule (`selection`), the source (`subset_of`) and the full-set index of every vector
(`indices`).

Format: a gzip-compressed `nvfp4-vectors-v1` file, the format of the full sets. One JSON header
line, then `count` 76-byte records (`a[32] b[32] sfa[4] sfb[4] c[4]`; element k is nibble k, low
nibble first; byte g of `sfa`/`sfb` scales elements 16g .. 16g + 15), then one little-endian
uint32 column per entry of `columns`. `gunzip -k` gives a file that
`tcgen05/nvfp4_tcgen05_check.py --vectors` reads. The subsets keep only the `rtx5090` and `thor`
columns.

```bash
python3 -I baremetal/generators/check_nvfp4_silicon.py      # model vs silicon
baremetal/generators/nvfp4_silicon/fp4_fedp_vec.sh          # RTL vs silicon (build/fp4_fedp_vec)
```
