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
  `vmatpop.fp8` or a power-of-two multiply.
- Matmul: `mxu1` accumulates in FP32 with NVIDIA Blackwell FP8 MMA arithmetic
  (`docs/fp8-mxu.md`), and the BF16 pop rounds once. LayerNorm, softmax, RoPE, GELU, SiLU, the
  gates and residuals are BF16 VPU operations.
- BF16 to FP8 conversion pushes the tile into an accumulator and pops it as FP8. This also turns
  the VPU's column-split tile layout into the row-major layout `vmatmul` reads.
- The tensors that scale token rows (adaLN's shift, scale and gate) are kept as tiles of 32
  identical rows, so each comes out of a matmul whose left operand has 32 equal rows. The
  timestep tile is built by reducing one element over rows and columns. The conditioning row is
  replicated exactly with `E @ Z^T`, where E is the FP8 tile with ones in column 0.
- The time MLP's exact GELU is evaluated in its tanh form, the only one the VPU has.
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
keeps E4M3 subnormals. `mxu1` is unchanged: it still flushes subnormal FP8 inputs and rounds to
BF16 per K tile. The same sampler (100 steps, the weights and inputs of the table above) in the
functional model:

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
  stay in place. `OPS` also has the `.vx` and `.vr` VPU forms and `vgelu.bf16`, which no kernel
  emits yet: a `.vx` reads its x-register, a `.vr` reads row 0 of T(vs2) in its issue cycle only,
  and `vgelu.bf16` reads its source for 64 cycles and writes its result until 75 cycles after issue.
- `Machine` is a model of the interlock. It gives the cycle in which the core issues each
  instruction: unit occupancy, register reservations on the 64 registers and the 32 physical
  banks, the VPU's two slots, `mxu1`'s ports and paths, the LSU's bank rule, and the DMA. The
  latencies come from npu-model/docs/rtl-timing.md and the RTL.
- Of the ready instructions, the scheduler takes the one that the core can issue first. On a
  tie, it takes the one with the longest dependent path to the end of the block.

The schedule does not change the arithmetic, so the outputs stay bit-identical.
`build --no-sched` keeps the program order. `tests/test_sched.py` compares scheduled and
unscheduled programs bit for bit under five DMA timing models: the source read and the
destination write of each transfer happen at issue or at its `DMA.WAIT`. Directed tests cover the
DRAM order across `DMA.WAIT`, the operands of `.vx` and `.vr`, and `vgelu.bf16`, which the DiT and
the policy do not exercise.

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
scheduling"). `build --policy` builds the whole policy (encoder + sampler); its
image needs the DDR4 board or a Verilator model with `DRAM_LINE_W=21`. `golden` also writes `bundle_checked.json`, which is the bundle with
the golden as its DRAM checks, so `standalone/host/run_tests.py --sim/--board` can verify it.

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

- `full` does not compile yet. The replicated adaLN tiles of 4 blocks (1.5 MiB) exceed VMEM,
  and the unrolled DiT kernels would exceed the 128 KiB IMEM. The DDR4 board now has the DRAM
  (2 GiB), so what remains is moving the DiT to the looped kernels and streaming adaLN tiles.
  The full encoder (ViT-B, 86 M parameters) would also need it.
- The encoder's GELU (exact in DINOv3 and `QueryPool`) is evaluated in its tanh form, as the
  time MLP's is.
- The program targets the interlocked core (`pipeline-interlock` and later) and emits no
  `DELAY`. Upstream Atlas would also need `DELAY`s, which `sched.py` does not emit.
- A step takes 316 K cycles, and the VPU is busy for 66% of them (210 K). More speed needs less
  VPU work (vector-scalar operands, a fused GELU, a scale on `vmatpop.bf16`) or overlap across
  loop iterations: the scheduler stays inside a basic block, so the encoder's looped kernels
  (`loopk.py`) gain less. `down` loads its activations again for every output tile, as 64 FP8
  tiles do not fit the registers.
- The scheduler moves only the instructions in `OPS`; others stay in place. New instructions
  need a row there, and some need more of the model. An op on fewer than 32 rows (a row count)
  holds its unit and registers for fewer cycles, and a write of some rows also reads the rest
  of the register. A row-broadcast operand reads one row, and a vector-scalar op can use the
  VPU's second slot. A scaled pop reads a scale register and waits for a pending `seld`. The
  NVFP4 MXU (2- and 4-register transfers, the scale sidecar) is not in `OPS` yet.
