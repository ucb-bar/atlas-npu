# DiT policy on Atlas

The noise predictor and flow-matching sampler of [specialist_lerobot](https://github.com/specialist-ai/specialist_lerobot)'s
`specialist_dit` policy, compiled to one Atlas program and run on the standalone VCU118 build
(`standalone/`). Every matmul is FP8 x FP8 with FP32 accumulation on the inner-product-tree MXU
(`mxu1`). The weights are random. There are two programs:

- the DiT sampler, which takes the policy's conditioning vector (below);
- the whole policy ("The whole policy"), which also runs the observation encoder. It takes the
  two camera crops and the state, and returns the action chunk.

`specialist_dit` uses LeRobot's `multi_task_dit` `DiffusionTransformer`. The reference here is
that LeRobot module itself (lerobot 0.6.1), and its own `FlowMatchingObjective._euler_integrate`.
The encoder's reference is specialist_lerobot's own `PooledVision` with a transformers 5.5.4
`DINOv3ViTModel`.

## What runs on Atlas

The program starts from the noise x0 and the conditioning vector and computes the whole action
chunk. All 100 Euler steps run on Atlas; the host only loads DRAM and reads the result.

- Prologue: SiLU(cond) @ W_c^T + b for each block's adaLN (the conditioning half; adaLN is
  linear in cat(SiLU(time), SiLU(cond)), so this half does not change between steps).
- Per step: the sinusoidal timestep embedding, the time MLP, the timestep half of adaLN, the
  input projection, the transformer blocks, and the output projection. Each block runs
  LayerNorm + adaLN modulation, the QKV projection, RoPE, attention per head, the output
  projection with its gate, and the GELU-tanh MLP with its gate. Then x += dt * v.

| | `small` (built and run) | `full` (specialist_dit defaults) |
|---|---|---|
| hidden, heads, layers | 256, 4, 2 | 512, 8, 4 |
| timestep embedding | 128 | 256 |
| conditioning (state + views x queries x dim) | 267 = 11 + 2 x 4 x 32 | 1035 = 11 + 2 x 8 x 64 |
| horizon x action, Euler steps | 50 x 10, 100 | 50 x 10, 100 |
| weights (FP8) | 2.9 MB | 28.7 MB |

`small` keeps the architecture and changes only widths, so it fits the board's 4 MiB of DRAM.

## The whole policy

`policy.py` puts specialist_dit's `ObservationEncoder` in front of the sampler, in the same
program. For each view (the 336 wide crop and the 224 zoom crop, already normalized by the
policy's processor), the encoder runs a DINOv3 ViT with patch 16 and 4 register tokens, 2-D
RoPE on patch tokens, LayerScale and a GELU MLP. Then the final norm, specialist's `QueryPool`
(learned queries cross-attending the patch tokens, then a residual MLP) and the projection to
`token_dim`. The pooled tokens stay in VMEM, and the DiT's prologue reads them as the
conditioning, after the state.

| encoder | `small` | specialist_dit default |
|---|---|---|
| backbone | 192 wide, 3 heads, 4 layers, MLP 768 | DINOv3 ViT-B/16 (768, 12, 12) |
| tokens per view | 441 + 5 (wide), 196 + 5 (zoom) | the same |
| queries x token_dim | 4 x 32 | 8 x 64 |
| weights (FP8) | 2.4 M | 86 M |

The encoder needs kernels that loop at run time (`loopk.py`), as a fully unrolled encoder would
not fit the 128 KiB IMEM:

- a linear over row pairs and output tiles, streaming weight blocks through two staging buffers;
- LayerNorm per row tile;
- multi-head attention over up to 14 key tiles: scores and probabilities are spilled to VMEM,
  softmax runs in two passes, and P @ V one head-dim tile at a time, since BF16 to FP8 needs a
  free accumulator.

Its four layers share one loop body, with weights and parameters behind pointer registers. So
each encoder tensor type has one FP8 scale for all layers (the largest calibrated range). The
CLS and register tokens follow the patch tokens; attention does not depend on token order, and
masks remove padding (and, for `QueryPool`, the prefix tokens) from the keys. The DiT's
conditioning K tiles are the views' pooled-token rows, replicated with FP8 "ones in column q"
tiles; its adaLN weight columns are permuted to that order.

The program is 27,930 instructions. Its DRAM image is 8.1 MiB, so it runs on the DDR4 build of
the standalone SoC (2 GiB, `standalone/README.md`) and in the Verilator model with a larger
DRAM (`DRAM_LINE_W=21`).

## Numerics

- FP8 E4M3 with per-tensor power-of-two scales: weight scales come from each weight's maximum.
  Activation scales come from calibration (the largest value over sampler trajectories on 4
  random inputs, with 2x headroom). Each scale is exact on Atlas, as a scale register for
  `vmatpop.fp8` and for the scaled BF16 pop.
- Matmul: `mxu1` accumulates in FP32 with NVIDIA Blackwell FP8 MMA arithmetic
  (`docs/fp8-mxu.md`). The BF16 pop applies the dequantization scale and rounds once
  (`vmatpop.bf16.scaled.acc.mxu1`). LayerNorm, softmax, RoPE, GELU, SiLU, the gates and
  residuals are BF16 VPU operations.
- BF16 to FP8 conversion pushes the tile into an accumulator and pops it as FP8. This also turns
  the VPU's column-split tile layout into the row-major layout `vmatmul` reads.
- Constants (1/n, eps, dt, the GELU and SiLU constants) are `.vx` operands: BF16 bits in
  x[31:16], set by `LI`. Rows that every token row uses (biases, LayerNorm's affine, adaLN's
  shift, scale and gate, LayerScale, attention masks) are `.vr` operands: row 0 of a tile,
  loaded with a 1-row `vload`.
- Row count M (`vsetm`): the timestep embedding, the time MLP and adaLN compute one row (M = 1);
  the timestep itself is a `.vx` operand (`lw` of its BF16 bits). The 50 action tokens fill two
  row tiles of 25 rows, and a step runs at M = 25. The encoder's views fill their row tiles
  evenly (wide: 14 x 32 rows for 446 tokens, zoom: 8 x 26 rows for 201) and QueryPool's 4
  queries are one tile at M = 4. A weight tile, K and V of attention, and V^T load or compute
  all 32 rows, as a weight push and a transpose read 32. Padding rows are not computed; RoPE's
  padding rows are 0 (and so is K there), and masks remove the padding keys.
- A conditioning row q > 0 of the encoder's tiles (the pooled token of query q) moves to row 0
  through the MXU with `E @ Z^T`, where E is the FP8 tile with ones in column q.
- GELU is `vgelu.bf16`, the tanh form; the time MLP's exact GELU uses it too.
- BF16 additions would lose most of each small Euler step over 100 steps, so x is carried as a
  compensated hi + lo pair. VPU additions round to nearest even (ISA v2), so lo is the exact
  error of each step's sum when |x| >= |dt * v| (Fast2Sum). dt = 0.01 is applied as two BF16
  factors.

## Results (2026-10-22)

Results dated before 2026-10-25 name the configurations of that time: `default` had the systolic
array (`mxu0`) too, and `ipt` was today's `default` (`standalone/README.md`).

The program is 24,713 instructions; 100 steps execute 1,961,645 of them. It uses 3.75 MiB of
DRAM (including the per-step velocity trace) and 754 KiB of VMEM.

| | final chunk vs FP32 (rel. L2 / max abs / cosine) | per-step velocity vs FP32 (rel. L2, mean) |
|---|---|---|
| FP8 fake quant (PyTorch, FP32 arithmetic) | 1.28% / 0.038 / 0.99992 | 4.9% (step 0) |
| Atlas, functional model | 1.54% / 0.055 / 0.99988 | 5.9% |
| Atlas, VCU118 (`ipt-nvfp4` bitstream) | identical to the functional model, bit for bit | identical |

A second draw (model seed 1, input seed 11) gives the same picture: the board is bit-exact with
its golden, and the chunk is 1.45% from FP32, against 1.29% for fake quant.

The first steps of the same schedule also run in the Verilator model with every RTL assertion
enabled, via `run_tests.py` on `bundle_checked.json`:

| Steps | Verilator `ipt-nvfp4` | Verilator `default` (both MXUs) | VCU118 `ipt-nvfp4` |
|---|---|---|---|
| 1 | PASS: 3,072 words equal the golden, 522,137 cycles | PASS, 522,137 cycles | PASS, 522,137 cycles |
| 2 | PASS: 4,096 words, 966,632 cycles | PASS, 966,632 cycles | PASS, 966,632 cycles |

Each simulation then took about 35 minutes, most of it loading DRAM through the simulated bus.
The model now loads DRAM through a backdoor (`standalone/README.md`): 1 step takes about 2.5
minutes, 2 steps about 4.5.

The board ran the 100 steps in 44.5 M cycles (0.89 s at 50 MHz); loading the 3.75 MiB image over
UART TSI takes most of the 49 s wall time. Its final chunk and all 100 velocities equal the
functional model's bit for bit. The functional model (`ditflow/funcsim.py`) is npu_model's
instruction semantics in program order with arithmetic bit-exact to the RTL
(`ditflow/numerics.py`, checked against npu_model's `rtl_math` on random and edge-case tiles; this
was the v1 `mxu1`, see below for ISA 2.0). The error against
FP32 is FP8 quantization (fake quant alone is 1.28%) plus about a fifth more from BF16
arithmetic.

### The whole policy (2026-10-22)

Encoder (both views) and the 100-step sampler in one program, 27,930 instructions; 2,627,547 of
them execute. The FP32 reference is specialist_lerobot's `PooledVision` + `ObservationEncoder`
arithmetic and LeRobot's sampler, with the same random weights and inputs.

| vs FP32 | conditioning vector (rel. L2 / cosine) | final chunk (rel. L2 / max abs / cosine) | velocity per step (rel. L2, mean) |
|---|---|---|---|
| FP8 fake quant | 5.42% / 0.99855 | 1.18% / 0.048 / 0.99993 | 5.0% (step 0) |
| Atlas, functional model | 5.49% / 0.99850 | 1.49% / 0.050 / 0.99989 | 5.5% |
| Atlas, VCU118 DDR4 build | bit-identical to the functional model | bit-identical | bit-identical |

- **VCU118 with DDR4** (default config, `standalone/README.md`): 100 steps in 56.2 M cycles,
  1.12 s at 50 MHz. The encoder takes about 11.6 M of them, both views. Loading the 8.1 MiB
  image over UART TSI takes most of the 110 s wall time. Every output word equals the
  functional model's: both views' pooled tokens, the chunk, and all 100 velocities.
- **Verilator** (`vl-default-64m`, every RTL assertion enabled, the encoder and 1 sampler step):
  PASS, 4,096 output words equal to the golden, 10,538,386 cycles.

The conditioning vector's error is the encoder's FP8 quantization (the fake quant is within
0.1 point of Atlas); the DiT then turns it into a 1.5% error on the chunk.

### ISA 2.0 VPU numerics (2026-10-25)

`vadd` and `vsub` now round to nearest even (they truncated), `vsquare` and `vcube` round the
exact power once, `vpack` and `vunpack` are the Blackwell FP8 conversions, and `quant.to_fp8`
keeps E4M3 subnormals. In this run `mxu1` still had the v1 numerics: it flushed subnormal FP8
inputs and rounded to BF16 per K tile (the next section gives the FP32 `mxu1`). The same sampler
(100 steps, the weights and inputs of the table above) in the functional model:

| vs FP32 | final chunk (rel. L2 / max abs / cosine) | per-step velocity (rel. L2, mean) |
|---|---|---|
| FP8 fake quant | 1.28% / 0.037 / 0.99992 | 4.9% (step 0) |
| Atlas, v1 VPU numerics | 1.54% / 0.055 / 0.99988 | 5.9% |
| Atlas, ISA 2.0 VPU numerics | 1.29% / 0.037 / 0.99992 | 5.5% |

The chunk is now 0.23% from fake quant (rel. L2), against 0.87% with the v1 VPU. The program
and its cycle counts do not change: in Verilator with every RTL assertion, 2 steps take 694,553
cycles in `default` and in `ipt-nvfp4`, and every output word equals the functional model. The
whole policy was not run again.

### ISA 2.0 `mxu1` numerics (2026-10-25)

`mxu1` now computes each K = 32 step as an NVIDIA Blackwell FP8 MMA into FP32 accumulators
(`docs/fp8-mxu.md`): E4M3 subnormals count, and the accumulator keeps FP32 between K tiles. The
FP8 pop (the BF16 to FP8 conversion above) rounds once and keeps E4M3 subnormals. The functional
model's `ditflow/numerics.py` is a torch port of npu_model's `fp8_mma.py`: `tests/test_numerics.py`
checks it against `fp8_mma` bit for bit, and it equals the RTX 5090 on all 197,473 outputs of
`baremetal/generators/fp8_silicon`. The same sampler (100 steps):

| vs FP32 | final chunk (rel. L2 / max abs / cosine) | per-step velocity (rel. L2, mean) |
|---|---|---|
| FP8 fake quant | 1.28% / 0.037 / 0.99992 | 4.9% (step 0) |
| Atlas, ISA 2.0 VPU, v1 `mxu1` | 1.286% / 0.037 / 0.99992 | 5.53% |
| Atlas, ISA 2.0 VPU and `mxu1` | 1.286% / 0.039 / 0.99992 | 5.52% |

The FP8 quantization sets the error (fake quant alone is 1.28%); the chunk moves from 0.230% to
0.224% from fake quant. The program and its cycle counts do not change: in Verilator with every
RTL assertion, 1 step takes 378,496 cycles and 2 steps 694,553, in `default` and in `ipt-nvfp4`,
and every output word equals the functional model. The whole policy (the encoder and 1 sampler
step) also passes in both configurations, in 9,901,025 cycles.

### ISA 2.0 build (2026-10-25)

One build has the ISA 2.0 encodings, the byte PC without delay slots and the numerics above. The
outputs are bit-identical to those of the previous section. The loops end with one branch and
no delay slot, so the cycle counts are those of the byte-PC program: in Verilator with every RTL
assertion, 1 step takes 378,308 cycles, 2 steps 694,270, and the whole policy (the encoder and
1 sampler step) 9,884,386, in `default` and in `ipt-nvfp4`. Every output word equals the
functional model.

### ISA 2.0 compiler (2026-10-26)

The compiler now uses the ISA 2.0 additions ("Numerics"). Each row adds one change to the row
above. Verilator `default` with every RTL assertion; in every run, every output word equals the
golden. "Same" means that the goldens are bit-identical to those of the row above.

| Change | 1 step | 2 steps | step 2 (2 - 1) | policy (encoder + 1 step) | outputs |
|---|---|---|---|---|---|
| before (the ISA 2.0 build) | 378,308 | 694,270 | 315,962 | 9,884,386 | |
| `Machine` gaps as measured on the RTL | 378,863 | 695,354 | 316,491 | 9,875,036 | same |
| constants as `.vx` operands | 348,634 | 635,955 | 287,321 | 9,552,726 | same |
| `vgelu.bf16` | 324,493 | 587,673 | 263,180 | 8,824,614 | same |
| dequantization in the BF16 pop | 308,009 | 561,041 | 253,032 | 8,250,626 | same |
| rows of every token as `.vr` operands | 305,393 | 555,713 | 250,320 | 8,058,407 | same |
| scheduling across `vsetm` | the same programs | | | | same |
| time path and adaLN at M = 1 | 275,506 | 509,271 | 233,765 | 8,029,114 | same |
| horizon as 2 x 25 rows, M = 25 | 254,254 | 463,546 | 209,292 | 8,007,722 | rounding of the softmax sums |
| 1-row loads of `.vr` operands | 252,418 | 459,874 | 207,456 | 7,947,236 | same |
| the pool's 4 queries as one tile, M = 4 | | | | 7,773,942 | same pooled tokens |
| each view's tokens spread over its row tiles | | | | 7,510,070 | rounding (zoom view) |
| the view's constants load after the patch embedding | 252,418 | 459,874 | 207,456 | 7,510,209 | same |
| input projections in two FP8 terms, now the default | 252,768 | 460,522 | 207,754 | 7,580,329 | more precise inputs |
| the same program on the RTL with the interlock fixes at M = 1 | 252,960 | 460,810 | 207,850 | 7,580,521 | same |

The rows above the last one ran on the RTL before the interlock fixes at M = 1
(docs/interlock.md): a pop of one row now holds a reader of its registers one cycle longer. The
1- and 2-step programs give the same cycle counts in `ipt-nvfp4`. 100 steps take about
20.8 M cycles (step 1 plus 99 times step 2), 0.42 s at 50 MHz; they took 31.7 M.

Accuracy against FP32 (100 steps of the DiT; the conditioning vector of the policy, mean of the
input seeds 7, 11 and 12):

| | DiT chunk (rel. L2) | DiT velocity per step (rel. L2, mean) | policy conditioning (rel. L2) | policy velocity, step 0 (rel. L2) |
|---|---|---|---|---|
| FP8 fake quant | 1.279% | | 5.29% | |
| before, up to the time path at M = 1 | 1.286% | 5.52% | 5.34% | |
| horizon as 2 x 25 rows | 1.282% | 5.52% | 5.34% | |
| views spread and later | 1.282% | 5.52% | 5.31% | 4.67% |
| input projections in two FP8 terms | 1.238% | 4.72% | 5.18% | 4.04% |

The split input gives the DiT's input projection and the encoder's patch embedding their input
as two FP8 terms of one scale, x = 2^a (hi + lo): `Kernels.quant_split` pops hi as FP8, pushes it
back, pops it as BF16 (x 2^a), subtracts it from x (exact in BF16) and pops the rest as lo. Both
terms multiply the same weights into one FP32 accumulator. The patch embedding also has its
weights in two terms (W_hi x_hi + W_hi x_lo + W_lo x_hi). On 192 patches of the wide view (3
inputs), the patch embedding's output is 3.75% from FP32 with FP8, 2.65% with the input split,
and 0.30% with both split (BF16 inputs and weights: 0.24%). The random weights here have no
outliers, so the pooled tokens gain less than the trained DINOv3 of docs/nvfp4-accuracy.md
("Patch embedding in two FP8 terms"), where the ViT-B features go from 15.8% to 8.2% from FP32.
The DiT's chunk gains 0.04 points and its velocity 0.8 points. The split costs 0.9% of the
policy's cycles and 0.1% of a step's. The split is the default; `build --no-split-input` gives
both one FP8 term.

Step 2 now spends its cycles as follows (% of 207,456): `mxu1` busy 59 (feed 39), the LSU's load
path 59, the DMA 46, the VPU 39; the core stalls on a unit for 79 and on data for 10. The VPU
was busy for 66.5% of 316,057 cycles before.

## Instruction scheduling (2026-10-24)

The core issues in program order and holds each instruction until its unit and its operands are
free (docs/interlock.md). The kernels emit their work one unit at a time, so in program order the
VPU, the LSU, `mxu1` and the DMA mostly take turns: 95% of a step's cycles were stalls.
`ditflow/sched.py` reorders each basic block when `Builder.finish()` returns the program:

- `OPS` gives the operands of every instruction that ditflow emits. From it the scheduler makes
  the dependency graph of a block: m-registers (a BF16 pair is two), accumulators, weight slots,
  scale and scalar registers, and VMEM. A VMEM address is exact when its base register has a
  known value (the segment registers, or a value set earlier in the block). Other addresses can
  alias any address. A DMA transfer touches its VMEM range when it issues and again at its
  `DMA.WAIT`, and each DMA channel keeps its order. DRAM has no address tracking: a load of DRAM
  stays behind the `DMA.WAIT` of an earlier store, and a store stays behind the `DMA.WAIT` of an
  earlier load. Labels, branches, CSR accesses, `ECALL` and instructions that are not in `OPS`
  stay in place. A `.vx` reads its x-register, a `.vr` reads row 0 of T(vs2) in its issue cycle
  only, and `vgelu.bf16` reads its source for 2M cycles and writes its result until 2M + 11
  cycles after issue.
- `vsetm` does not end a block. Each instruction keeps the row count M of the last `vsetm` before
  it in the text (32 after a label that no `vsetm` follows), and the scheduled block sets M before
  each instruction that needs another M than the instruction before it. So a kernel at M < 32
  can load its weight tiles at 32 rows and still overlap them with its other work. The builder
  sets M right after a label when it is not 32, and back to that M before each branch to it.
- `Machine` is a model of the interlock. It gives the cycle in which the core issues each
  instruction: unit occupancy, register reservations on the 64 registers and the 32 physical
  banks, the VPU's two slots, `mxu1`'s ports and paths, the LSU's bank rule, and the DMA. The
  latencies come from npu-model/docs/rtl-timing.md; the issue gaps are the ones measured on the
  RTL (`test_machine_follows_the_rtl`): a reader of a scalar load's register at +4, the next
  matmul or pop of an accumulator at +4, a matmul after an accumulator push when the push ends,
  a matmul after a weight push of its slot at +32, and a VPU source free 2M cycles after issue
  (`.vr`: 2M + 1, row reductions M, column reductions 4M).
- Of the ready instructions, the scheduler takes the one that the core can issue first. On a
  tie, it takes the one with the longest dependent path to the end of the block.

The schedule does not change the arithmetic, so the outputs stay bit-identical.
`build --no-sched` keeps the program order. `tests/test_sched.py` compares scheduled and
unscheduled programs bit for bit under five DMA timing models: the source read and the
destination write of each transfer happen at issue or at its `DMA.WAIT`. Directed tests cover the
DRAM order across `DMA.WAIT`, the operands of `.vx` and `.vr`, `vgelu.bf16`, and a block that
mixes M = 4 with 32-row weight loads.

Four compiler changes come with the scheduler:

- `ada_t` runs on one row tile again. On 2026-10-22 it moved to the looped `linear_rows` (so that
  the policy fits IMEM), which took only row pairs: a padding row tile cost 384 matmuls per step.
  `linear_rows` now also takes one row tile.
- The register pool gives the register that was freed longest ago, so consecutive tiles use
  different registers. Pairs come from m0-m31 and single registers from m32-m63 first. Thus two
  pairs never share a physical bank, which a two-source VPU instruction cannot read.
- A linear with an FP8 epilogue (`up`, `time1`, `time2`) puts the FP8 store of each tile, which
  borrows an accumulator, after the next tile's pop of that accumulator. The next tile's
  matmuls then run during the epilogue's GELU on the VPU.
- The DiT's two weight staging buffers each start a VMEM bank of their own. The LSU has priority
  on a bank, so a DMA into a staging buffer waited for each vload from the same bank. VMEM use
  goes from 754 KiB to 1,058 KiB.

Verilator `ipt-nvfp4` with every RTL assertion enabled. In every run, every output word equals
the golden, and the goldens of all builds are identical:

| Build | 1 step | 2 steps | step 2 (2 - 1) |
|---|---|---|---|
| before (`630722d`) | 576,280 | 1,045,425 | 469,145 |
| `ada_t` on one row tile | 532,057 | 986,166 | 454,109 |
| + register pool, deferred FP8 stores, banked staging (`--no-sched`) | 504,847 | 942,262 | 437,415 |
| + scheduling (the default) | 378,496 | 694,553 | 316,057 |

Performance counters of step 2 (2 steps minus 1 step), in % of its cycles:

| | cycles | stall_unit | stall_data | stall_dma | vpu_busy | vload_busy | mxu1_busy | mxu1_feed | dma_busy |
|---|---|---|---|---|---|---|---|---|---|
| before | 469,145 | 68.5 | 26.0 | 1.1 | 45.7 | 35.2 | 36.8 | 27.7 | 33.5 |
| scheduled | 316,057 | 80.6 | 13.0 | 0.0 | 66.5 | 52.2 | 52.9 | 37.3 | 28.5 |

The work of each unit stays almost the same (VPU 214 K to 210 K cycles, MXU feed 130 K to 118 K
without the padding tile). It now overlaps. So 100 steps take about 31.7 M cycles in Verilator
(step 1 plus 99 times step 2), 0.63 s at 50 MHz. With the DMA at the Verilator DRAM's rate (one
beat per cycle), `Machine` gives 378,828 cycles for the scheduled 1-step program, against 378,496
measured. The scheduler itself assumes half that DMA rate (`sched.py`, `DMA_BEAT`). With the
measured rate it leaves the DMA too little slack: 380,172 cycles for 1 step, 317,733 for step 2.

The whole policy (encoder and 1 sampler step) in the same Verilator model takes 10,771,167
cycles before, 10,647,881 with `--no-sched` and 9,901,025 scheduled (-8.1%). Every output word,
the pooled tokens too, equals the golden. The encoder gains less (about 7%) than the sampler: its
kernels loop at run time, and the scheduler does not move work across a loop iteration.

## Usage

Python 3.14 with torch, lerobot 0.6.1 (only its DiT module is imported), npu-model and
pyuartsi:

```bash
uv venv --python 3.14 build/ditenv
VIRTUAL_ENV=build/ditenv uv pip install torch torchvision numpy einops draccus huggingface-hub \
    safetensors packaging termcolor tqdm requests gymnasium opencv-python-headless pytest tabulate \
    -r standalone/host/requirements.txt
VIRTUAL_ENV=build/ditenv uv pip install --no-deps lerobot==0.6.1 -e npu-model
cd apps/dit
PY=../../build/ditenv/bin/python
$PY -m pytest                                          # numerics, kernels, one compiled step
$PY -m ditflow.flow build  ../../build/dit/small       # program, DRAM image, PyTorch references
$PY -m ditflow.flow golden ../../build/dit/small       # functional model (~5 min for 100 steps)
$PY -m ditflow.flow run    ../../build/dit/small --board        # or --sim <atlas_sim>
$PY -m ditflow.flow report ../../build/dit/small
```

`build --steps N` compiles the first N steps of the same schedule; that is how the Verilator
model runs it in minutes. `build --no-sched` keeps the program order ("Instruction
scheduling"). `build --no-split-input` gives the DiT's input projection and the encoder's patch
embedding their input as one FP8 term, and the patch embedding its weights as one FP8 term, not
two ("ISA 2.0 compiler"). `build --policy` builds the whole policy (encoder + sampler); its
image needs the DDR4 board or a Verilator model with `DRAM_LINE_W=21`. `golden` also writes
`bundle_checked.json`, which is the bundle with the golden as its DRAM checks, so
`standalone/host/run_tests.py --sim/--board` can verify it.

`flow run` also reads the performance counters (docs/perf-counters.md) after the halt and stores
them, with each unit's utilization, in `board.pt` / `sim.pt` and `report.json`.

`python -m ditflow.hazfuzz OUT --units mxu1,nv` writes randomized hazard-test bundles for
`run_tests.py`: random dependent tensor-instruction streams without `DELAY`, checked against the
functional model (docs/interlock.md, "Randomized hazard test").

## Files

- `ditflow/model.py`: the LeRobot module with random weights, the FP32 sampler, and the forward
  pass restated op by op (calibration, fake quant).
- `ditflow/quant.py`: FP8 scales, calibration and the quantized weights.
- `ditflow/layout.py`: tile layouts in DRAM and VMEM.
- `ditflow/asm.py`: assembly builder, VMEM/DRAM maps, register pools, DMA scheduling.
- `ditflow/kernels.py`: linear (with weights streamed from DRAM), LayerNorm + modulation,
  attention, GELU, SiLU.
- `ditflow/loopk.py`: the same kernels looping at run time, for the encoder's long sequences.
- `ditflow/vision.py`, `ditflow/policy.py`: the encoder's reference, and the policy program.
- `ditflow/compiler.py`: the sampler program.
- `ditflow/sched.py`: the list scheduler and its model of the interlocked core.
- `ditflow/funcsim.py`, `ditflow/numerics.py`: the functional model.
- `ditflow/flow.py`: the command line.
- `ditflow/hazfuzz.py`: the randomized hazard test of the interlock.
- `quantstudy/`: the NVFP4, MXFP4 and FP8 accuracy study on trained DINOv3 weights
  (docs/nvfp4-accuracy.md). It also needs `transformers==5.5.4` and `av`.

## Limits

- `full` does not compile yet. The adaLN tiles of 4 blocks (1.5 MiB, of which the kernels read
  row 0) exceed VMEM, and the unrolled DiT kernels would exceed the 128 KiB IMEM. The DDR4 board
  now has the DRAM (2 GiB), so what remains is moving the DiT to the looped kernels and storing
  only row 0 of the adaLN tiles.
  The full encoder (ViT-B, 86 M parameters) would also need it.
- The encoder's GELU (exact in DINOv3 and `QueryPool`) is evaluated in its tanh form, as the
  time MLP's is.
- The program targets the interlocked core (`pipeline-interlock` and later) and emits no
  `DELAY`. Upstream Atlas would also need `DELAY`s, which `sched.py` does not emit.
- A step takes 207 K cycles. `mxu1` and the LSU's load path are busy for 59% of them each:
  every weight tile goes through VMEM and a 32-row `vload`, and `down` loads its activations
  again for every output tile, as 64 FP8 tiles do not fit the registers. The scheduler stays
  inside a basic block, so the encoder's looped kernels (`loopk.py`) gain less.
- The scheduler moves only the instructions in `OPS`; others stay in place, and new
  instructions need a row there. The NVFP4 MXU (2- and 4-register transfers, the scale sidecar)
  is not in `OPS` yet.
