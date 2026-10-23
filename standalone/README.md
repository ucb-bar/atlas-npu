# Standalone Atlas on the VCU118

A host-driven build of Atlas without Chipyard: no CPU, no rocket-chip. A host computer takes
the CPU's place over UART TSI (the protocol of [PyUARTSI](https://github.com/ucb-bar/pyuartsi)
and Chipyard's UART-TSI). It loads the kernel into IMEM and the data into DRAM, starts the
scalar core, waits for completion and reads the results. The structure follows the NVFP4 SoC
(`ucb-bar/nvfp4`), whose UART TSI target and TL-UL fabric are reused here.

```text
host ─ UART TSI ─ uart_tsi ─ tl_widen ─ tl_xbar ─┬─ sys_regs          0x0000_0000
                                                 ├─ Atlas IMEM        0x0002_0000  (128 KiB)
                                                 ├─ Atlas CSR         0x0004_0000
                                                 ├─ Atlas VMEM        0x2000_0000  (1.5 MiB)
                                                 └─ DRAM, port A      0x8000_0000 ─┐
                                    AtlasCore DMA ── DRAM, port B ─────────────────┴─ DDR4 bridge ─ MIG ─ 2 GiB DDR4
```

- `chisel/`: `EmitAtlasCore` writes AtlasCore's SystemVerilog with fixed TL-UL ports, and
  `tilelink/` stands in for the three rocket-chip TileLink names AtlasCore uses. The RTL under
  `src/main/scala` is shared with the Chipyard build unchanged.
- `rtl/`: the SoC. `tsi/` is NVFP4's UART TSI target and fabric (copied from nvfp4 `0c611e1`).
  The board's DRAM is 2 GiB of DDR4 through `atlas_ddr4_dram.sv` and the MIG (below).
  `atlas_dram.sv` is the Verilator model's DRAM, aliased over `0x8000_0000` up (DMA's 64-bit
  addresses fold onto it the same way); the baremetal tests use 465 KiB at `0x9000_0000`.
  (Until 2026-10-22 the board build could also use 4 MiB of UltraRAM; that option is gone.)
- `sim/`: a Verilator model of `atlas_soc` with a word-level host port (no UART), and its driver.
- `host/`: `atlas_host.py` (the host: `SimLink` for the model, `TsiLink` for the board),
  `make_bundle.py` (assembles the baremetal tests into JSON bundles), `run_tests.py`.
- `fpga/`: VCU118 constraints, the Vivado build and the JTAG programming script.

## Configurations

`--config=NAME` selects the Atlas configuration (docs/nvfp4-mxu.md): `default` (systolic array +
inner-product trees), `ipt` (inner-product trees only) or `ipt-nvfp4` (inner-product trees + the
NVFP4 MXU). `run_tests.py --config NAME` expects every test that needs a missing unit to stop with
an illegal instruction and every other test to pass.

## Build and run

Simulation (Verilator 5, Java for mill):

```bash
./mill standalone.runMain atlas.standalone.EmitAtlasCore build/rtl-sim --assertions --config=ipt-nvfp4
standalone/sim/build.sh build/rtl-sim build/vl_soc   # build/vl_soc/atlas_sim, about 10 min

cd baremetal                                  # bundles: npu_model's assembler + generators (torch)
uv run --locked --project ../npu-model --python 3.14 python ../standalone/host/make_bundle.py \
    --out ../build/bundle/orig --gen-python <python with torch>
uv run --locked --project ../npu-model --python 3.14 python ../standalone/host/make_bundle.py \
    --out ../build/bundle/nodelay --no-delay --gen-python <python with torch>
cd ..
python3 standalone/host/run_tests.py --sim build/vl_soc/atlas_sim --config ipt-nvfp4 -j 32 \
    --out build/sim-nodelay build/bundle/nodelay/*.json
```

`--no-delay` removes every `DELAY` from the kernels (one in a branch delay slot would become a
`NOP`). The model keeps all RTL assertions, so a hazard that got past the interlock fails the test.

Bitstream (Vivado 2024.1 or later with a VU9P license on `PATH`; about 2.5 hours):

```bash
standalone/fpga/build_bitstream.sh build/vcu118 50 16 ipt-nvfp4   # build/vcu118/atlas_standalone_vcu118_ipt-nvfp4_ddr4_50mhz.bit
```

Board: program over the VCU118's JTAG USB port, then run over the CP2105 UART (interface 1,
921,600 baud). The host needs `pyuartsi`.

```bash
vivado -mode batch -source standalone/fpga/program.tcl -tclargs build/vcu118/atlas_standalone_vcu118_ipt-nvfp4_ddr4_50mhz.bit
python3 standalone/host/run_tests.py --board --config ipt-nvfp4 --out build/board-nodelay build/bundle/nodelay/*.json
```

`run_tests.py` writes each run's performance counters (docs/perf-counters.md) to `OUT/perf.csv`.

## Results (2026-10-21, on-chip DRAM)

These builds predate the DDR4-only board build; the results for the current bitstreams are under
"DDR4 DRAM" below.

Bitstream `atlas_standalone_vcu118_50mhz.bit` (Vivado 2024.1, `pipeline-interlock` 18a3f1e): timing met
at 50 MHz (WNS +0.005 ns, WHS +0.010 ns), 604K LUTs (51%), 98K FFs, 501 RAMB36, 128 URAM,
112 DSP. The critical path is the one that also limits the Chipyard build: an MREG bank read
address into an MXU1 lane's anchor accumulation.

| Run | original kernels | every `DELAY` removed |
|---|---|---|
| Verilator, all RTL assertions | 121 / 121 | 121 / 121 |
| VCU118 over UART TSI | 121 / 121 | 121 / 121 |

On the board, kernel cycle counts (RUN_CYCLES) equal the simulation's for 120 of 121 kernels in
both variants. Without `DELAY` the suite takes 9.8% fewer cycles; `perf_mm_*` keep 93% MXU
utilization. A full suite takes a few minutes; the link loses about one byte per suite, which
`TsiLink` absorbs by reading the chunk again.

The exception is `jal_jalr`. It never initializes `x11`, and the scalar registers keep their
values from the previous kernel on the board, so its first subtest fails. Its fail path writes
`x28` to DBG0, which is 1 in the first subtest: the same value as a pass, so the runner reports
PASS. With `x11` cleared the kernel passes all four subtests on the board, in the same 36 cycles
as in simulation. 10 tests are judged by DBG0 alone (no DRAM check) and share this ambiguity.

### Configurations (`nvfp4-mxu`, 2026-10-22)

The suite now has 126 tests (the 121 above and 5 NVFP4 tests). Vivado 2024.1, 50 MHz, timing met
on all three; the critical path is still the MREG read into an MXU1 lane.

| Config | FEATURES | WNS / WHS (ns) | LUTs | FFs | RAMB36 | URAM | DSP |
|---|---|---|---|---|---|---|---|
| `default` (above) | `0x7` | +0.005 / +0.010 | 604K (51%) | 98K | 501 | 128 | 112 |
| `ipt` | `0x5` | +0.097 / +0.010 | 342K (29%) | 56K | 501 | 128 | 112 |
| `ipt-nvfp4` | `0xd` | +0.307 / +0.013 | 523K (44%) | 159K | 513 | 128 | 240 |

`Nvfp4Mxu` alone is 162K LUTs, 103K FFs (mostly the two 32 x 32 FP32 accumulators) and 128 DSPs;
`MregScaleFile` is 1K LUTs and 12 RAMB36.

| Config | expected | Verilator, original / no `DELAY` | VCU118, original / no `DELAY` |
|---|---|---|---|
| `default` | 121 pass, 5 illegal | 126 / 126, 126 / 126 | 121 / 121, 121 / 121 (above) |
| `ipt` | 93 pass, 33 illegal | 126 / 126, 126 / 126 | 126 / 126, 126 / 126 |
| `ipt-nvfp4` | 98 pass, 28 illegal | 126 / 126, 126 / 126 | 126 / 126, 126 / 126 |

On the board the status and RUN_CYCLES of every test equal the simulation's except `jal_jalr`
(see above), including the cycle at which each illegal instruction stops a kernel. The five
NVFP4 tests compare 320 to 1024 result words each with the NVFP4 reference model and match all
of them; `nvfp4_kchain` runs its four `vmatmul.acc.nv` in 171 cycles on the board as in
simulation.

## DDR4 DRAM

The board build's top, `vcu118_atlas_top`, puts the DRAM in 2 GiB of the board's DDR4 (C1 bank).
`sys_regs` DRAM_BYTES reads 0x8000_0000, and FLAGS bit 3 is set once the memory is calibrated.

- The controller is the Xilinx DDR4 MIG (`fpga/vcu118_ddr4_mig.tcl`, IP `vcu118mig`). Its
  settings, pins and clocking come from Chipyard's VCU118 shell (fpga-shells), which calibrates
  on this board: DDR4-1600 components, a 200 MHz user clock, and the 250 MHz `sys_clock` as
  the reference clock. The one change is a 256-bit AXI port, one 32-byte DMA line per beat.
- `atlas_ddr4_dram.sv` keeps atlas_dram's two ports (host, DMA) in the core clock and crosses
  requests and responses through gray-coded FIFOs (`async_fifo.sv`). In the user clock, a read
  issues only when no write is outstanding and the reverse, so the memory stays sequential
  although AXI orders reads and writes independently. Responses return in request order.
  CPU_RESET resets the bridge and the MIG's AXI side, but not the MIG (no recalibration).
- `sim/tb_ddr4.sh` runs the bridge against a randomized AXI memory model in Verilator. 20,000
  random host and DMA transactions match a reference memory.

On the board (default config, `bitstreams/AtlasStandaloneDDR4`, 2026-10-22): timing met at 50 MHz
(WNS +0.150 ns, MIG user clock +0.494 ns, WHS +0.010 ns), 620K LUTs (52%), 526 RAMB36, no URAM.
The DDR4 calibrates; host reads and writes across the 2 GiB match. The 126 tests give 126 / 126
as expected (121 pass, 5 illegal), with cycle counts within -1% to +8% of the on-chip build's.
apps/dit's sampler (3.75 MiB) and whole policy (8.1 MiB) run bit-exact with their goldens.

The Verilator model keeps the behavioral DRAM, 64 MiB by default (`DRAM_LINE_W=21`; 17 makes it
the 4 MiB of the old on-chip build), and its `B <addr> <file>` command (used by `SimLink` for every DRAM preload) loads data
through a backdoor in no simulated time. A 3.4 MiB image used to cost 35 minutes of simulated
bus writes.

## Driving it from Python

`run_tests.py` is one client of `host/atlas_host.py`. A kernel can also be run by hand; the
link is the same for the board (`TsiLink()`) and the model (`SimLink(binary)`):

```python
from atlas_host import Atlas, TsiLink, IMEM_BASE, CSR_EXEC_CONTROL, CSR_DBG0

atlas = Atlas(TsiLink())                         # checks the sys_regs ID
link = atlas.link
link.write_words(0x9000_0000, inputs)            # DRAM (32-bit words)
link.write_words(CSR_EXEC_CONTROL, [0])          # stop, then load the kernel
link.write_words(CSR_DBG0, [0])
link.write_words(IMEM_BASE, program)             # words from npu_model's assembler
link.write_words(CSR_EXEC_CONTROL, [1])          # start
link.poll(CSR_DBG0, 10.0)                        # the kernel writes DBG0 when it is done
outputs = link.read_words(0x9000_0800, 256)
```

## Board

- Clock: the 250 MHz system clock (E12/D12) through an MMCM to 50 MHz.
- Reset: CPU_RESET (L19); the SoC also resets when the MMCM loses lock.
- LEDs: `led[0]` heartbeat, `led[1]` TSI activity, `led[2]` link error or denied access,
  `led[3]` MMCM locked, `led[4]` Atlas halted, `led[7:5]` the sys_regs LEDS register.

## System registers (`0x0000_0000`)

| Offset | Name | |
|---|---|---|
| `0x000` | ID | `0x4154_4C53` ("ATLS") |
| `0x004` | SCRATCH | read/write |
| `0x008` | FLAGS | [0] TSI receive overflow, [1] bad TSI command, [2] denied access seen, [3] DRAM ready (DDR4 calibrated; 1 in the Verilator model) |
| `0x00C` | DENIED | denied (unmapped) accesses |
| `0x010` | CYCLES | free-running cycle counter |
| `0x014` | CLK_HZ | |
| `0x018` | BAUD | |
| `0x01C` | DRAM_BYTES | 2 GiB of DDR4 (`0x8000_0000`) on the board; the model's size in Verilator |
| `0x020` | ATLAS | [0] halted, [1] MXU0 data busy, [2] MXU0 compute busy, [3] MXU1 data busy, [4] MXU1 compute busy, [5] XLU busy, [6] LSU busy, [15:8] DMA channel busy |
| `0x024` | FEATURES | [0] hardware interlock, [1] systolic-array MXU, [2] inner-product-tree MXU, [3] NVFP4 MXU |
| `0x028` | LEDS | [2:0] |
| `0x02C` | RUN_CYCLES | cycles the scalar core has run (not halted); a write clears it |

Reads of IMEM stall while the scalar core runs (IMEM serves the fetch); stop the core first.

## Performance counters

The Atlas CSR block (`0x0004_0000`) has 16 64-bit counters: cycles, issued instructions, stall
cycles by cause, and the busy cycles of each unit. Write 1 to bit i of PERF_RESET (`+0x100`) to
clear counter i; counter i is at `+0x200 + 8i`. See docs/perf-counters.md.
