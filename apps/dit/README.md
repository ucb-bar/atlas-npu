# DiT policy on Atlas

The noise predictor and flow-matching sampler of [specialist_lerobot](https://github.com/specialist-ai/specialist_lerobot)'s
`specialist_dit` policy, compiled to one Atlas program and run on the standalone VCU118 build
(`standalone/`). Every matmul is FP8 x FP8 with BF16 accumulation on the inner-product-tree MXU
(`mxu1`). The weights are random. The vision backbone (DINOv3 ViT-B) stays on the host: the
program takes the policy's conditioning vector as input.

`specialist_dit` uses LeRobot's `multi_task_dit` `DiffusionTransformer`. The reference here is
that LeRobot module itself (lerobot 0.6.1), and its own `FlowMatchingObjective._euler_integrate`.

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

## Numerics

- FP8 E4M3 with per-tensor power-of-two scales: weight scales come from each weight's maximum.
  Activation scales come from calibration (the largest value over sampler trajectories on 4
  random inputs, with 2x headroom). Each scale is exact on Atlas, as a scale register for
  `vmatpop.fp8` or a power-of-two multiply.
- Matmul: `mxu1` accumulates each output once per 32-wide K tile, rounding to BF16. LayerNorm,
  softmax, RoPE, GELU, SiLU, the gates and residuals are BF16 VPU operations.
- BF16 to FP8 conversion pushes the tile into an accumulator and pops it as FP8. This also turns
  the VPU's column-split tile layout into the row-major layout `vmatmul` reads.
- The tensors that scale token rows (adaLN's shift, scale and gate) are kept as tiles of 32
  identical rows, so each comes out of a matmul whose left operand has 32 equal rows. The
  timestep tile is built by reducing one element over rows and columns. The conditioning row is
  replicated exactly with `E @ Z^T`, where E is the FP8 tile with ones in column 0.
- The time MLP's exact GELU is evaluated in its tanh form, the only one the VPU has.
- VPU additions truncate to BF16, which would bias 100 accumulated Euler steps, so x is carried
  as a compensated hi + lo pair. dt = 0.01 is applied as two BF16 factors.

## Results (2026-10-22)

The program is 24,713 instructions; 100 steps execute 1,961,645 of them. It uses 3.75 MiB of
DRAM (including the per-step velocity trace) and 754 KiB of VMEM.

| | final chunk vs FP32 (rel. L2 / max abs / cosine) | per-step velocity vs FP32 (rel. L2, mean) |
|---|---|---|
| FP8 fake quant (PyTorch, FP32 arithmetic) | 1.28% / 0.038 / 0.99992 | 4.9% (step 0) |
| Atlas, functional model | 1.54% / 0.055 / 0.99988 | 5.9% |
| Atlas, VCU118 (`ipt-nvfp4` bitstream) | identical to the functional model, bit for bit | identical |

The board ran the 100 steps in 44.5 M cycles (0.89 s at 50 MHz); loading the 3.75 MiB image over
UART TSI takes most of the 49 s wall time. Its final chunk and all 100 velocities equal the
functional model's bit for bit. The functional model (`ditflow/funcsim.py`) is npu_model's
instruction semantics in program order with arithmetic bit-exact to the RTL
(`ditflow/numerics.py`, checked against npu_model's `rtl_math`: the FP8 converters on every input, the IPT matmul on random and edge-case tiles). The error against
FP32 is FP8 quantization (fake quant alone is 1.28%) plus about a fifth more from BF16
arithmetic.

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
model runs it in minutes. `golden` also writes `bundle_checked.json`, which is the bundle with
the golden as its DRAM checks, so `standalone/host/run_tests.py --sim/--board` can verify it.

## Files

- `ditflow/model.py`: the LeRobot module with random weights, the FP32 sampler, and the forward
  pass restated op by op (calibration, fake quant).
- `ditflow/quant.py`: FP8 scales, calibration and the quantized weights.
- `ditflow/layout.py`: tile layouts in DRAM and VMEM.
- `ditflow/asm.py`: assembly builder, VMEM/DRAM maps, register pools, DMA scheduling.
- `ditflow/kernels.py`: linear (with weights streamed from DRAM), LayerNorm + modulation,
  attention, GELU, SiLU.
- `ditflow/compiler.py`: the sampler program.
- `ditflow/funcsim.py`, `ditflow/numerics.py`: the functional model.
- `ditflow/flow.py`: the command line.

## Limits

- `full` does not fit yet. The replicated adaLN tiles of 4 blocks (1.5 MiB) exceed VMEM, the
  fully unrolled step would exceed the 128 KiB IMEM, and 28.7 MB of weights need more DRAM than
  the standalone board has.
- The program targets the interlocked core (`pipeline-interlock` and later) and emits no
  `DELAY`. Upstream Atlas would need a scheduler.
- Performance is untuned. A step takes ~445 K cycles, against ~130 K of MXU work: in-order
  issue serializes the VPU (~3,000 BF16 ops per step), vload (~5,400 per step) and MXU streams.
