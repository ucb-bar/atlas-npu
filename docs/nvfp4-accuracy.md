# NVFP4 accuracy of the policy's encoder

Atlas has an NVFP4 MXU that matches NVIDIA silicon bit for bit (docs/nvfp4-mxu.md). The policy that
Atlas runs (apps/dit) does not use it: every matmul is FP8 on `mxu1`. This study measures, before
more NVFP4 hardware is built, whether NVFP4 is accurate enough for the policy. It compares NVFP4,
MXFP4 and FP8 against FP32 on trained weights and real images. The weights in apps/dit are random,
and random weights cannot rank number formats: they do not have the outliers of trained weights.

The script is `apps/dit/quantstudy` (see Reproduce).

## Result

NVFP4 W4A4 is clearly less accurate than FP8 in the encoder under post-training quantization, and
without a measure of the policy's actions there is no evidence that it is accurate enough. On
DINOv3 ViT-B/16, the backbone that specialist_dit loads, with NVIDIA's published NVFP4 recipe:

- Each linear's output error is 9.8% (FP8: 2.9%). The final features are 39.7% from FP32
  (cosine 0.917); with FP8 they are 15.8 to 17.4% from FP32 (cosine 0.984 to 0.986).
- Dense patch correspondence drops from 82.1% (FP32) to 76.9%; with FP8 it drops to 81.5 to
  81.7%. On ViT-S/16, NVFP4 drops it by 10.4 points, FP8 by 0.9 to 1.2 points.
- NVFP4 changes 2.1% of the decisions of a linear probe on Imagewoof, FP8 1.0 to 1.3%. On ViT-S,
  NVFP4 changes 7.8%, FP8 1.4 to 1.6%.

The other 4-bit options do not close the gap. W4A8 (MXFP4 weights x FP8 activations) has more
feature error than NVFP4 W4A4 (41.0%) but better correspondence, most of all on ViT-S (76.6%
against 72.2%); both stay about 4 points or more of correspondence below FP8. NVFP4 weights with
BF16 activations still have 27.2% feature error.

Of the formats that Atlas has or plans, FP8 with the arithmetic of Blackwell (E4M3 subnormals
kept, FP32 accumulation) is the most accurate. On ViT-B it gains 1.6 points of feature error over
today's `mxu1`: 1.4 points come from the subnormals, which `mxu1` flushes to zero, and 0.2 points
from FP32 accumulation. On ViT-S it gains 0.4 points: 0.1 from the subnormals and 0.3 from the
accumulation.

No trained specialist_dit checkpoint is in the workspace, so the study cannot measure the policy's
actions. See Limits.

## Recommendation

1. Do not build more NVFP4 hardware for this policy now: no NVFP4 block-quantize instruction and no
   FP4 x FP8 mixed mode. Keep the NVFP4 MXU of `ipt-nvfp4` as it is, for experiments.
2. Give `mxu1` the arithmetic of Blackwell FP8 MMA (npu_model `configs/fp8_mma.py`). Two of its
   properties matter here, and `mxu1` must have both: it keeps E4M3 subnormals, and it accumulates
   in FP32. On ViT-B, the subnormals that `mxu1` flushes today cost 1.4 points of feature error
   and the BF16 accumulation per K tile costs 0.2 points; on ViT-S they cost 0.1 and 0.3 points.
   Keep per-tensor power-of-two scales: MXFP8 block scales (E8M0 per 32) give the same error on
   this model.
3. Run the patch embedding more precisely than FP8. FP8 there causes about a third of the FP8
   feature error (ViT-B: 17.4% to 11.2% with a BF16 patch embedding, 12.4% with an exact input and
   FP8 weights). The patch embedding is less than 1% of the encoder's multiply-accumulates, so two
   FP8 passes (the input split into a high and a low FP8 part) cost little. This is a compiler
   change.
4. Decide 4-bit weights only with a trained specialist_dit checkpoint and the policy's own action
   error, and with quantization-aware fine-tuning: specialist_dit already fine-tunes the backbone
   (`vision_encoder_lr_multiplier`). The least harmful 4-bit option here is NVFP4 in the MLP only,
   FP8 elsewhere, as ModelOpt's `nvfp4_mlp_only` recipe does. It saves 29% of the weight bytes
   (ViT-B: 85 MB to 61 MB) and costs 2.0 points of correspondence (FP8: 0.4 points).
5. If NVFP4 is used, the power-of-two global scales of the epilogue contract (docs/nvfp4-mxu.md,
   "Global scales and bias") are safe: the same error per linear as FP32 global scales, final
   features within 1.0 point, and fewer saturated block scales (1 of 67 M activation blocks on
   ViT-B, against 27).

NVIDIA's default NVFP4 recipes also keep vision encoders in BF16. ModelOpt disables the quantizers
of `*vision_tower*`, `*visual*` and `*vision_model*` "by default"; its comment names export crashes
and "garbage image embeddings" on vision-language models (NVIDIA/Model-Optimizer@3c423a9
`modelopt_recipes/configs/ptq/units/default_disabled_quantizers.yaml`, lines 53-71).

## Method

### Weights

- No trained specialist_dit checkpoint is in the workspace. A search of the home directory for
  `*.safetensors`, `*.ckpt`, `*.pt` and `config.json` files that name `specialist_dit` or
  `multi_task_dit` found none. specialist_lerobot keeps its checkpoints as WandB artifacts
  (`robots/checkpoints.py`), and the local WandB cache holds no artifact.
- Encoder proxy: the public DINOv3 ViT-B/16 (`facebook/dinov3-vitb16-pretrain-lvd1689m`), which
  is specialist_dit's default `vision_encoder_name`, and DINOv3 ViT-S/16
  (`facebook/dinov3-vits16-pretrain-lvd1689m`). Both are gated on Hugging Face. specialist_dit
  fine-tunes this backbone at a reduced learning rate, so its deployed weights differ from the
  public ones. Its `QueryPool`, projection and DiT are not in the study.

### Images

The workspace has no camera frames of the specialist rig (`ext/specialist_lerobot` has no dataset,
and the local LeRobot teleoperation dataset has no video). The study uses public sets:

| Set | Use | Images |
|---|---|---|
| Imagenette, 320 px (fast.ai) | features; probe and k-NN; correspondence | val 3,925; train 6,000; calibration 130 |
| Imagewoof, 320 px (fast.ai, 10 dog breeds) | features; probe and k-NN (harder) | val 3,929; train 5,980; calibration 117 (one class has only 580 training images, all used by the probe) |
| `lerobot/umi_cup_in_the_wild`, `file-000.mp4` | robot frames: features; temporal correspondence | 1,024 frame pairs; calibration 128 |

The UMI frames come from a fisheye camera on a hand-held gripper, the public data closest to the
specialist lens: 224 x 224, 10 frames per second, 44,468 frames in the file. Images are resized to
224 on the short side and center-cropped to 224, then normalized with the ImageNet mean and
standard deviation of DINOv3's processor.

### Formats

All quantizers put the blocks along K, the reduction dimension.

| Configuration | Weights | Activations | Accumulation |
|---|---|---|---|
| (1) Atlas FP8 today | E4M3, one power-of-two scale from the maximum, subnormals flushed | E4M3, one power-of-two scale per tensor from calibration, 2x margin, subnormals flushed | BF16, rounded once per 32-wide K tile (`mxu1`) |
| (2) FP8, Blackwell | as (1), subnormals kept | as (1), subnormals kept | FP32 (Blackwell FP8 MMA, see Arithmetic) |
| (2), subnormals flushed | as (1) | as (1) | FP32 |
| MXFP8 | E4M3, E8M0 scale per 32 | E4M3, E8M0 scale per 32, dynamic | FP32 |
| (3) NVFP4 W4A4 | E2M1, E4M3 scale per 16, FP32 global scale amax / (6 x 448) | the same; block scales dynamic, global scale from calibration | FP32 (Atlas NVFP4 MXU) |
| (4) NVFP4, power-of-two global scales | as (3), global scale 2^ceil(log2(amax / (6 x 448))) | the same | FP32 |
| (5) W4A8 | MXFP4: E2M1, E8M0 scale per 32 | as (1) | FP32 |
| W4A8, MX | MXFP4 | MXFP8 | FP32 |
| (6) NVFP4 W4A16 | as (3) | BF16 | FP32 |
| BF16 | BF16 | BF16 | FP32 |

The rules follow the published recipes:

- NVFP4 is ModelOpt's `nvfp4.yaml` (E2M1, blocks of 16, dynamic, E4M3 scales) with its PTQ recipe
  `nvfp4_default-kv_fp8.yaml` (`algorithm: max`). Block scale E4M3(min(amax_block / (6 g), 448)),
  element E2M1(x / (s g)) with round to nearest even and saturation at 6, a zero scale zeroes the
  block (`modelopt/torch/kernels/quantization/common/nvfp4_quant.py`, `fp4_round_magnitude` and
  `fp8_quantize_scale`). The weights' global scale comes from the weight; the activations' global
  scale g comes from the calibrated maximum, which ModelOpt keeps after `max` calibration
  (`tensor_quantizer.py`, `_get_amax`). This is also Transformer Engine's NVFP4 arithmetic:
  s_global = amax / (448 x 6), s_block = (amax_block / 6) / s_global
  (<https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.13/user-guide/features/low_precision_training/nvfp4/nvfp4.html>).
  TE applies its random Hadamard transform to the wgrad GEMM and its stochastic rounding to
  gradients only. Its 16 x 16 weight blocks are a training default; ModelOpt's inference recipe
  uses blocks of 16 along K, as here.
- MX scales are 2^ceil(log2(amax_block / max_element)), as ModelOpt sets them
  (`mxfp4_tensor.py`, line 75; `tensor_quant_mx.cu`, `compute_scale_e8m0_NV`).
- FP8 is Atlas's: `apps/dit/ditflow/quant.py` (`exponent_for`, `to_fp8`). `quantstudy/formats.py`
  flushes subnormals in (1) as today's `mxu1` does. Blackwell keeps FP8 subnormals (`fp8_mma.py`,
  rule 2), so (2) keeps them. The row "(2), subnormals flushed" differs from (1) only in the accumulation and from (2)
  only in the subnormals.
- Atlas does q, k and v as one GEMM, so they share one weight scale (the maximum of the three) and
  the activation scale.
- Calibration takes the largest |value| of each matmul input over the calibration images, in FP32.
- The NVFP4 global scale is computed in float64, where ModelOpt uses FP32. A literal port of
  ModelOpt's fake quantizer changes dequantized values by at most 1.6e-7 relative and the final
  features by less than 0.1 point.

Extra rows change one part of a configuration: the patch embedding in BF16, in FP8 weights x exact
input, or in FP8; NVFP4 only in the MLP (`up`, `down`) with (2) in the other linears; and "+ FP8
attention", where QK^T and PV also take FP8 inputs with per-tensor power-of-two scales.

### Arithmetic

The study wraps the transformers `DINOv3ViTModel` (5.5.4). Each linear and the patch embedding (a
linear on 16 x 16 patches, K = 768, as Atlas runs it) multiply the quantized values. QK^T and PV are
FP32 unless the row says "+ FP8 attention". LayerNorm, RoPE, softmax, GELU, LayerScale and the
residuals are FP32 in every row, so the rows differ only in the matmuls.

A quantizer returns the FP32 values of the quantized tensor, so a matmul of them in FP32 applies
every scale as the hardware does, up to the rounding of the sum. TF32 is off. Row (1) adds each
32-wide K tile to a BF16 accumulator and rounds once per tile. `study.py check` runs these
emulations and the bit-exact models of the same matmuls on real inputs of four layers (the patch
embedding, layer 0 q, the middle layer's up and the last layer's down; 64 rows of 2 images, 8 x 64
outputs for NVFP4):

| Matmul | Bit-exact model | ViT-B: identical outputs | rel. L2, emulation vs model | rel. L2, format vs FP32 |
|---|---|---|---|---|
| (1) Atlas `mxu1` | `ditflow.numerics.ipt_matmul` | 100% in all 4 layers | 0 | 1.6% to 8.4% |
| (2) Blackwell FP8 MMA | `npu_model/configs/fp8_mma.py` | 11% to 95% | 5.5e-8 to 7.2e-7 | 1.5% to 8.4% |
| (3) Atlas NVFP4 MXU | `npu_model/configs/nvfp4.py` (`fp4_fedp`) | 12% to 40% | 8.1e-8 to 2.4e-7 | 2.5% to 22.7% |

The emulations are bit-exact for `mxu1` and within FP32 rounding of the other two models, at least
four orders of magnitude below the format errors. On ViT-S, `mxu1` is bit-exact in 3 layers and
identical on 99.99% of the outputs of the fourth.

### Metrics

- Linear output rel. L2: each of the 73 linears gets the input that the FP32 model gives it. The
  study compares the quantized GEMM (without bias) with the FP32 GEMM, over 32 Imagenette images,
  and reports the mean over the linears.
- Features: the final-norm output, CLS and patch tokens (the registers dropped). rel. L2 per image,
  averaged; and the mean cosine of each patch token with its FP32 value.
- Probe and k-NN: a logistic-regression probe and DINO's weighted k-NN (k = 20, T = 0.07) on
  [CLS, mean patch token], fit on FP32 features of the training images. Accuracy, and agreement:
  the fraction of validation images that get the same class as with FP32.
- Correspondence: 1,000 Imagenette validation images, two 224 crops of a 272 center crop, offset by
  3 patches in x and y. For each patch of the first crop that the second crop contains, the most
  similar patch (cosine) of the second crop must be the same image patch.
- UMI NN agree: for frame pairs 0.1 s apart, the fraction of patches whose most similar patch in the
  next frame is the same as with FP32.

## Results

### DINOv3 ViT-B/16 (specialist_dit's backbone)

FP32 reference: Imagewoof probe 96.1%, k-NN 95.7%; Imagenette 99.7% and 99.8%.

| Configuration | Linear output rel. L2 | Features rel. L2: ImageNet / UMI | Patch cosine | Imagewoof probe: acc. / agree | Imagewoof k-NN: acc. / agree | Correspondence acc. | UMI NN agree |
|---|---|---|---|---|---|---|---|
| FP32 (reference) | 0 | 0.0% / 0.0% | 1.000 | 96.1% / 100.0% | 95.7% / 100.0% | 82.1% | 100.0% |
| BF16 | 0.2% | 1.0% / 2.1% | 1.000 | 96.2% / 100.0% | 95.7% / 100.0% | 82.1% | 99.8% |
| (1) Atlas FP8 today | 2.9% | 17.4% / 22.6% | 0.984 | 95.8% / 98.7% | 95.6% / 99.5% | 81.5% | 96.9% |
| (2) FP8, Blackwell | 2.9% | 15.8% / 21.1% | 0.986 | 96.1% / 99.0% | 95.7% / 99.5% | 81.7% | 97.7% |
| (2), subnormals flushed | 2.9% | 17.2% / 22.4% | 0.984 | 95.9% / 98.7% | 95.7% / 99.5% | 81.5% | 97.2% |
| MXFP8 | 2.9% | 15.8% / 21.1% | 0.986 | 95.9% / 98.9% | 95.6% / 99.6% | 81.7% | 97.8% |
| (3) NVFP4 W4A4 | 9.8% | 39.7% / 52.3% | 0.917 | 95.2% / 97.9% | 94.8% / 97.8% | 76.9% | 95.2% |
| (4) NVFP4, power-of-two global scales | 9.8% | 40.5% / 52.4% | 0.915 | 94.7% / 97.2% | 95.0% / 98.1% | 77.4% | 93.6% |
| (5) W4A8: MXFP4 x FP8 | 10.0% | 41.0% / 54.9% | 0.913 | 94.4% / 96.8% | 94.8% / 97.8% | 77.7% | 98.1% |
| W4A8, MX: MXFP4 x MXFP8 | 10.0% | 41.0% / 54.9% | 0.913 | 94.4% / 96.9% | 94.8% / 97.7% | 77.6% | 98.1% |
| (6) NVFP4 W4A16 | 7.0% | 27.2% / 37.5% | 0.962 | 95.7% / 98.7% | 95.3% / 98.9% | 80.9% | 98.6% |
| (1), patch embedding BF16 | 2.9% | 11.2% / 18.7% | 0.994 | 96.0% / 99.2% | 95.7% / 99.6% | 81.5% | 96.9% |
| (1), patch embedding FP8 weights x exact input | 2.9% | 12.4% / 19.5% | 0.992 | 95.9% / 99.1% | 95.7% / 99.5% | 81.4% | 96.7% |
| (3), patch embedding FP8 | 9.6% | 29.4% / 37.9% | 0.955 | 95.7% / 98.4% | 95.2% / 98.8% | 77.9% | 95.6% |
| (3) in the MLP, (2) elsewhere | 5.0% | 22.7% / 30.1% | 0.973 | 95.7% / 98.8% | 95.6% / 99.2% | 80.1% | 95.6% |
| (1) + FP8 attention | 2.9% | 18.2% / 24.2% | 0.982 | 95.8% / 98.6% | 95.6% / 99.4% | 80.9% | 96.7% |
| (2) + FP8 attention | 2.9% | 17.1% / 24.0% | 0.984 | 95.9% / 99.0% | 95.6% / 99.6% | 80.8% | 97.4% |
| (3) + FP8 attention (2) | 9.8% | 40.7% / 52.3% | 0.913 | 95.0% / 97.3% | 94.8% / 97.6% | 75.6% | 95.0% |
| (5) + FP8 attention (2) | 10.0% | 53.1% / 70.3% | 0.853 | 92.7% / 94.9% | 94.1% / 96.4% | 67.4% | 96.5% |

Linear output rel. L2 by kind of linear (mean over the layers):

| Configuration | patch embedding | q, k, v | o | up | down |
|---|---|---|---|---|---|
| BF16 | 0.4% | 0.2% | 0.2% | 0.2% | 0.2% |
| (1) Atlas FP8 today | 6.7% | 2.7% | 3.5% | 3.1% | 2.7% |
| (2) FP8, Blackwell | 6.6% | 2.7% | 3.4% | 3.0% | 2.5% |
| (2), subnormals flushed | 6.6% | 2.7% | 3.4% | 3.0% | 2.5% |
| (3) NVFP4 W4A4 | 23.1% | 9.0% | 12.2% | 9.8% | 8.6% |
| (5) W4A8 | 21.1% | 9.7% | 11.2% | 10.0% | 8.9% |
| (6) NVFP4 W4A16 | 16.2% | 6.2% | 8.6% | 7.5% | 6.3% |

### DINOv3 ViT-S/16

FP32 reference: Imagewoof probe 94.3%, k-NN 93.6%; Imagenette 99.1% and 99.1%.

| Configuration | Linear output rel. L2 | Features rel. L2: ImageNet / UMI | Patch cosine | Imagewoof probe: acc. / agree | Imagewoof k-NN: acc. / agree | Correspondence acc. | UMI NN agree |
|---|---|---|---|---|---|---|---|
| FP32 (reference) | 0 | 0.0% / 0.0% | 1.000 | 94.3% / 100.0% | 93.6% / 100.0% | 82.6% | 100.0% |
| BF16 | 0.2% | 0.9% / 1.8% | 1.000 | 94.4% / 99.8% | 93.6% / 99.9% | 82.6% | 99.9% |
| (1) Atlas FP8 today | 3.3% | 16.2% / 19.5% | 0.986 | 94.1% / 98.4% | 93.4% / 99.0% | 81.4% | 97.4% |
| (2) FP8, Blackwell | 3.2% | 15.8% / 19.2% | 0.986 | 94.4% / 98.6% | 93.5% / 98.8% | 81.5% | 97.7% |
| (2), subnormals flushed | 3.3% | 15.9% / 19.2% | 0.986 | 94.4% / 98.4% | 93.4% / 98.7% | 81.7% | 97.7% |
| MXFP8 | 3.2% | 15.8% / 19.2% | 0.986 | 94.5% / 98.6% | 93.6% / 98.9% | 81.6% | 97.7% |
| (3) NVFP4 W4A4 | 11.0% | 42.8% / 49.9% | 0.904 | 89.9% / 92.2% | 91.5% / 95.4% | 72.2% | 87.2% |
| (4) NVFP4, power-of-two global scales | 11.0% | 43.8% / 50.6% | 0.900 | 90.0% / 92.2% | 91.3% / 95.1% | 72.8% | 88.0% |
| (5) W4A8: MXFP4 x FP8 | 10.8% | 46.2% / 55.9% | 0.892 | 89.5% / 92.0% | 90.8% / 94.6% | 76.6% | 95.8% |
| W4A8, MX: MXFP4 x MXFP8 | 10.8% | 46.2% / 55.9% | 0.892 | 89.4% / 91.9% | 90.8% / 94.5% | 76.6% | 95.8% |
| (6) NVFP4 W4A16 | 7.8% | 29.4% / 34.9% | 0.956 | 92.7% / 95.7% | 93.0% / 97.6% | 80.8% | 98.9% |
| (1), patch embedding BF16 | 3.2% | 10.6% / 13.6% | 0.994 | 94.2% / 98.9% | 93.6% / 99.0% | 81.6% | 98.2% |
| (1), patch embedding FP8 weights x exact input | 3.3% | 11.3% / 15.6% | 0.993 | 94.3% / 99.0% | 93.4% / 99.0% | 81.5% | 98.0% |
| (3), patch embedding FP8 | 10.8% | 33.1% / 40.1% | 0.945 | 92.5% / 95.2% | 92.5% / 96.9% | 74.0% | 90.2% |
| (3) in the MLP, (2) elsewhere | 5.8% | 25.3% / 33.4% | 0.967 | 93.6% / 97.0% | 93.3% / 97.9% | 78.4% | 93.7% |
| (1) + FP8 attention | 3.3% | 17.3% / 20.8% | 0.984 | 94.1% / 98.3% | 93.4% / 98.7% | 80.4% | 95.8% |
| (2) + FP8 attention | 3.2% | 16.9% / 20.6% | 0.985 | 94.3% / 98.4% | 93.4% / 98.8% | 80.4% | 96.3% |
| (3) + FP8 attention (2) | 11.0% | 43.5% / 50.2% | 0.901 | 89.6% / 92.0% | 91.8% / 95.3% | 70.7% | 85.5% |
| (5) + FP8 attention (2) | 10.8% | 46.7% / 56.6% | 0.890 | 88.8% / 91.5% | 90.7% / 94.5% | 74.6% | 93.1% |

### Observations

- The feature error ranks the formats the same way on both models and both image domains: BF16,
  the FP8 rows, NVFP4 W4A16, NVFP4 W4A4, W4A8. Correspondence puts W4A8 above NVFP4 W4A4 (ViT-B
  77.7% against 76.9%, ViT-S 76.6% against 72.2%), both about 4 points or more below FP8. The UMI
  temporal agreement ranks differently: on it, NVFP4 W4A16 (both models) and W4A8 (ViT-B) score
  higher than FP8.
- ViT-B is more robust to NVFP4 than ViT-S on the task metrics, but its feature error is still 2.5
  times that of FP8.
- About half of NVFP4's squared error per linear comes from the weights: 7.0% with the weights
  alone (W4A16), 9.8% with weights and activations (ViT-B).
- MXFP4 weights are worse than NVFP4 weights (E8M0 scales per 32 against E4M3 scales per 16), and
  this cancels the gain of FP8 activations in W4A8. On ViT-B, W4A8 with FP8 attention is much worse
  than W4A8 alone (53.1% against 41.0%); the error grows on every image (median 0.40 to 0.52 on 128
  images), and no FP8 attention operand saturates (its peak is at most 1.7 times the calibrated
  maximum, inside the 2x margin). A separate check (not in `study.py`, 260 and 130 ViT-B images)
  traces it to the FP8 rounding of K: FP8 on K alone gives the jump (51.6%), FP8 on q costs about
  1.3 points, on p or v nothing. Subtracting each head's mean key before the rounding, which does
  not change the softmax, brings W4A8 with FP8 K from 51.1% to 40.7% (40.3% without FP8
  attention), and (2) with FP8 K from 15.8% to 15.4% (15.0% without). A compiler that runs
  attention in FP8 can use this.
- Rows (1) and (2) differ in the accumulation and in the FP8 subnormals; the row "(2), subnormals
  flushed" separates the two. FP32 instead of BF16 accumulation changes the error per linear by
  less than 0.1 point and the features by 0.2 points (ViT-B) and 0.3 points (ViT-S). Kept
  subnormals do not change the mean error per linear, but they change the ViT-B features by 1.4
  points (ViT-S: 0.1 points). A separate check on 260 Imagenette validation images (not in
  `study.py`) finds all of this loss in the input of `down`, the GELU output. On ViT-B, 8.2% of its
  nonzero FP8 codes are subnormal (ViT-S: 3.7%), against at most 2.4% for the other activations
  and less than 0.1% for the weights. Flushed subnormals in `down` alone cost 1.4 points of
  feature error; flushed subnormals in all weights cost 0.02 points.
- The NVFP4 block scales are almost all normal E4M3 numbers. On ViT-B, of 33.6 M activation blocks
  (32 images), 16 saturate with FP32 global scales and none with power-of-two global scales; 2,869
  (3,168 with power-of-two) have an E4M3-subnormal scale and 2,237 (2,690) underflow to a zero
  scale. All 5.3 M weight block scales are normal. ViT-S has no subnormal or zero block scale.

## Limits

- The study measures the encoder only, on the public backbone. It does not measure specialist_dit's
  fine-tuned backbone, its `QueryPool` and projection, or the DiT, and it does not measure actions.
  The task metrics are proxies; the policy can be more or less sensitive than they are.
- Post-training quantization only, with the default `max` calibration. ModelOpt also has AWQ,
  GPTQ, MSE scale search and quantization-aware training, which can reduce the NVFP4 error.
- The inputs are 224 x 224. specialist_dit's wide view is 336 x 336 (441 patches); its zoom view is
  224 x 224. The UMI frames are whole fisheye frames scaled to 224, not the specialist crops.
- LayerNorm, softmax, GELU and the residuals are FP32 here. On Atlas they are BF16 VPU operations,
  which add error to every row (apps/dit/README.md: about a fifth more on the DiT).
- `check` compares the emulations with the bit-exact models on 4 layers of 2 images.

## Reproduce

The environment of apps/dit/README.md, plus `transformers==5.5.4` and PyAV (`av`) for the video.
The DINOv3 weights are gated: accept the license of both repositories on Hugging Face and log in
(`hf auth login`).

```bash
DATA=/path/to/data; mkdir -p $DATA && cd $DATA
curl -L https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz | tar xz
curl -L https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz | tar xz
$PY -c "from huggingface_hub import hf_hub_download as d; print(d('lerobot/umi_cup_in_the_wild', \
    'videos/observation.image/chunk-000/file-000.mp4', repo_type='dataset', local_dir='umi'))"
ln -s umi/videos/observation.image/chunk-000/file-000.mp4 umi.mp4
cd <atlas-npu>/apps/dit
$PY -m pytest quantstudy                                       # formats and arithmetic, 6 tests
$PY -m quantstudy.study run   OUT --model vitb16 --data $DATA  # OUT/vitb16.json and the tables
$PY -m quantstudy.study check OUT --model vitb16 --data $DATA  # emulation vs bit-exact models
```

`--model vits16` runs ViT-S. On an RTX 4090, `run` takes about 21 minutes for ViT-B and 7 for
ViT-S; the `mxu1` emulation (one BF16 rounding per K tile) takes most of it. The results above are
from 2026-10-24.
