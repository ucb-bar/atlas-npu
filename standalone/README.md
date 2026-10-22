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
                                    AtlasCore DMA ── DRAM, port B ─────────────────┘
```

- `chisel/`: `EmitAtlasCore` writes AtlasCore's SystemVerilog with fixed TL-UL ports, and
  `tilelink/` stands in for the three rocket-chip TileLink names AtlasCore uses. The RTL under
  `src/main/scala` is shared with the Chipyard build unchanged.
- `rtl/`: the SoC. `tsi/` is NVFP4's UART TSI target and fabric (copied from nvfp4 `0c611e1`).
  `atlas_dram.sv` is the DRAM stand-in: 4 MiB of UltraRAM, aliased over `0x8000_0000` up (DMA's
  64-bit addresses fold onto it the same way); the baremetal tests use 465 KiB at `0x9000_0000`.
- `sim/`: a Verilator model of `atlas_soc` with a word-level host port (no UART), and its driver.
- `host/`: `atlas_host.py` (the host: `SimLink` for the model, `TsiLink` for the board),
  `make_bundle.py` (assembles the baremetal tests into JSON bundles), `run_tests.py`.
- `fpga/`: VCU118 constraints, the Vivado build and the JTAG programming script.

## Build and run

Simulation (Verilator 5, Java for mill):

```bash
./mill standalone.runMain atlas.standalone.EmitAtlasCore build/rtl-sim --assertions
standalone/sim/build.sh                       # build/vl_soc/atlas_sim, about 10 min

cd baremetal                                  # bundles: npu_model's assembler + generators (torch)
uv run --locked --project ../npu-model --python 3.14 python ../standalone/host/make_bundle.py \
    --out ../build/bundle/orig --gen-python <python with torch>
uv run --locked --project ../npu-model --python 3.14 python ../standalone/host/make_bundle.py \
    --out ../build/bundle/nodelay --no-delay --gen-python <python with torch>
cd ..
python3 standalone/host/run_tests.py --sim build/vl_soc/atlas_sim -j 32 --out build/sim-nodelay build/bundle/nodelay/*.json
```

`--no-delay` removes every `DELAY` from the kernels (one in a branch delay slot would become a
`NOP`). The model keeps all RTL assertions, so a hazard that got past the interlock fails the test.

Bitstream (Vivado 2024.1 or later with a VU9P license on `PATH`; about 2 hours):

```bash
standalone/fpga/build_bitstream.sh build/vcu118 50     # build/vcu118/atlas_standalone_vcu118_50mhz.bit
```

Board: program over the VCU118's JTAG USB port, then run over the CP2105 UART (interface 1,
921,600 baud). The host needs `pyuartsi`.

```bash
vivado -mode batch -source standalone/fpga/program.tcl -tclargs build/vcu118/atlas_standalone_vcu118_50mhz.bit
python3 standalone/host/run_tests.py --board --out build/board-nodelay build/bundle/nodelay/*.json
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
| `0x008` | FLAGS | [0] TSI receive overflow, [1] bad TSI command, [2] denied access seen |
| `0x00C` | DENIED | denied (unmapped) accesses |
| `0x010` | CYCLES | free-running cycle counter |
| `0x014` | CLK_HZ | |
| `0x018` | BAUD | |
| `0x01C` | DRAM_BYTES | |
| `0x020` | ATLAS | [0] halted, [1] MXU0 data busy, [2] MXU0 compute busy, [3] MXU1 data busy, [4] MXU1 compute busy, [5] XLU busy, [6] LSU busy, [15:8] DMA channel busy |
| `0x024` | FEATURES | [0] hardware interlock |
| `0x028` | LEDS | [2:0] |

Reads of IMEM stall while the scalar core runs (IMEM serves the fetch); stop the core first.
