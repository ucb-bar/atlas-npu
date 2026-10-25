# Hardware Architecture Summary

The Tapeout NPU is a single-issue, in-order accelerator with a scalar RISC-V-style control path and a tile-oriented tensor datapath. Execution timing is deterministic; there is no speculation or out-of-order execution. A hardware interlock holds each instruction until it can issue without a structural or data hazard (`docs/interlock.md`), so `delay` instructions are legal but never required. `dma.wait` stays the fence between a DMA transfer and the instructions that use its VMEM data.

## Programming Model

The machine fetches one 32-bit instruction per cycle from IMEM, decodes it, and issues it to the appropriate functional unit. Long-latency tensor operations (MXU matmuls, VPU ops, XLU transforms) execute concurrently with each other and with scalar instructions, but only one new instruction issues per cycle. Issue stalls while the targeted unit cannot take the instruction, or while a register, VMEM bank or scale register that it uses is still busy. Branches and jumps have no delay slot: a taken one drops the instruction after it. The `delay N` instruction stalls the frontend for N cycles. `dma.wait.chN` stalls the frontend until the specified DMA channel completes.

## Architectural State

- **Scalar registers:** 32 × 32-bit (`x0`–`x31`; `x0` hardwired to zero). Standard RV32I ABI.
- **Tensor registers:** 64 × 1024-byte (`m0`–`m63`). Each stores one 32×32 FP8 tile or one 32×16 BF16 half-tile. A full 32×32 BF16 tile spans two consecutive registers (e.g., `m4`/`m5`).
- **Scale registers:** 32 × 8-bit (`e0`–`e31`). Used for FP8 quantization/dequantization scaling.
- **MXU1 state:** 2 weight slots (1024 bytes each, holding 32×32 FP8 tiles) and 2 accumulators (4096 bytes each, holding 32×32 FP32 values). These are not directly addressable as `m` registers; data moves via explicit push/pop instructions. The `ipt-nvfp4` configuration adds the NVFP4 MXU (`.nv` instructions, `docs/nvfp4-mxu.md`).
- **DMA state:** Shared `dma.base` register (32-bit), 8 architected channels each with busy/idle state.
- **PC:** Byte address (increments by 4 per instruction, as in RV32I).

## Memory Hierarchy

| Region | Size | Access Path | Bus Width | Throughput |
|--------|------|-------------|-----------|------------|
| IMEM | 64 KiB | Instruction fetch only | 32-bit | 1 insn/cycle |
| VMEM | 1 MiB | Scalar load/store, vload/vstore, DMA target | 512-bit | 64 B/cycle (1 beat/cycle) |
| DRAM | 16 GiB | DMA only | 32-bit | 2 B/cycle (4 B every 2 cycles) |

**IMPORTANT: All addresses used in instructions are zero-based offsets, NOT physical SoC addresses.**
VMEM addresses range from `0x0000` to `0xFFFFF` (1 MiB). DRAM addresses range from `0x0000` upward.
For example, to access VMEM offset `0x2000`, load the value `0x2000` into a scalar register — do NOT use `0x2000_0000`.
The test harness places data at small offsets (e.g., DRAM activation at `0x0000`, weight at `0x0400`, output at `0x0800`;
VMEM staging at `0x2000`, `0x2400`, `0x2800`). These are the actual values to put in scalar registers.

**VMEM** is the sole on-chip data memory. All tensor loads/stores and scalar data accesses target VMEM. A 1024-byte tensor register load/store (`vload`/`vstore`) streams for 32 cycles. An accumulator push or pop streams for 32 cycles (one register, or a BF16 register pair) or 64 cycles (FP32, four registers).

**DRAM** is accessible only through DMA. Transfer time is `max(offchip_cycles, vmem_cycles)` where offchip_cycles = `ceil((bytes + 8) / 4) × 2` and vmem_cycles = `ceil(bytes / 64)`. DRAM bandwidth (2 B/cycle) is the dominant bottleneck for large transfers. All DMA addresses and sizes must be 32-byte aligned.

## Compute Units

**MXU1 (Inner-Product Tree):** Performs 32×32 FP8 matrix multiply into FP32 accumulators, bit-exact with an NVIDIA Blackwell FP8 MMA (`docs/fp8-mxu.md`). A matmul streams for 32 cycles; its last row lands at T+34. Peak throughput: 2048 FLOPs/cycle. Operands: activation from tensor register (FP8), weight from local weight slot (FP8), result in a local FP32 accumulator. A pop rounds the accumulator once, to BF16 (BF16_RNE) or to FP8 (E4M3_RN_SATFINITE).

**VPU (16 BF16 lanes):** A BF16 instruction names the even register of a pair and processes the 32×32 tile as two 32×16 halves. Supports elementwise add, sub, mul, min, max, ReLU, reciprocal, exp, exp2, sin, cos, tanh, log2, sqrt, and column/row reductions. A BF16 pair instruction streams for 64 cycles: the first row lands at T+2, the last at T+65 (`npu-model/npu_spec/04_functional_units`, Timing Reference).

**XLU (Transform):** Transpose (`vtrpose.xlu`). A transpose writes its result from T+34 to T+65.

**Scalar ALU:** Standard RV32I integer arithmetic, branches, scalar VMEM load/store. All scalar memory ops are blocking.

## Key Constraints for Code Optimization

1. **On-chip data transfers occupy their paths.** `vload`, `vstore`, `vmatpush.*` and `vmatpop.*` hold their LSU or MXU path for the whole transfer, and the interlock holds a later instruction that needs the same path or registers. Minimizing unnecessary data movement is critical.

2. **MXU data staging is explicit.** Weights must be pushed to weight slots (`vmatpush.weight.*`, 32 cycles) before matmul. Accumulator preload (`vmatpush.acc.*`) and spill (`vmatpop.*`) are separate operations. FP8 and BF16 pushes and pops take 32 cycles (1024 or 2048 bytes); FP32 pushes and pops take 64 cycles (4096 bytes).

3. **Overlap is the primary performance lever.** MXU1, VPU, and XLU can all be active simultaneously. Issue independent work while a matmul streams (32 cycles); the interlock waits for its result. For generated regression tests, be conservative with DMA overlap and avoid patterns that depend on multiple DMA requests being in flight at once.

4. **DRAM bandwidth is scarce (2 B/cycle vs. 64 B/cycle VMEM).** Prefetch data via DMA well ahead of use, but for robust regression tests issue DMA requests conservatively: after a `DMA.LOAD` or `DMA.STORE`, use `DMA.WAIT` before issuing another DMA request, even if the next request uses a different channel. Each channel supports one outstanding transfer; the interlock holds a DMA request to a busy channel. The current testbench path should not be assumed to tolerate multi-channel outstanding DMA traffic.

5. **BF16 tile = 2 consecutive `m` registers.** Operations producing or consuming full 32×32 BF16 tiles (`vmatpop.bf16.*`, `vmatpush.acc.bf16.*`) use register pairs; FP32 accumulator transfers use four consecutive registers. BF16 VPU instructions take pairs too: one instruction processes both halves of a tile.

6. **Hardware interlock.** The scalar core holds an instruction in S1 while it would read a register before its producer writes it (RAW), overwrite one that an earlier instruction still reads or writes (WAR, WAW), or use a unit, path or VMEM bank that is still busy. `delay` is never required for correctness. `dma.wait` is the only fence for DMA data in VMEM. A two-source VPU instruction that names two different pairs in one physical bank (`m4` and `m36`) is illegal.

7. **No branch delay slots.** A taken branch or jump drops the instruction after it, so put work before the branch. Branch and JAL offsets are byte offsets, and a taken target with bit 1 set is illegal.

8. **Immediate encoding limits.** Scalar immediates are 12-bit sign-extended. Addresses larger than ±2047 require `lui`/`addi` sequences. `vload`/`vstore` add `imm12 << 5` to a VMEM word address (128-byte steps, covering a ±256 KiB offset range).