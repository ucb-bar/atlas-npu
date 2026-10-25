# Standalone Atlas on the VCU118

A host-driven build of Atlas without Chipyard: no CPU, no rocket-chip. A host computer takes
the CPU's place over UART TSI (the protocol of [PyUARTSI](https://github.com/ucb-bar/pyuartsi)
and Chipyard's UART-TSI). It loads the kernel into IMEM and the data into DRAM, starts the
scalar core, waits for completion and reads the results. The structure follows the NVFP4 SoC
(`ucb-bar/nvfp4@0c611e1`), whose UART TSI target and TL-UL fabric are reused here.

On the board, the host can also send the same TSI commands over UDP on the 1000BASE-T port
(TSI over UDP, docs/tsi-over-udp.md). `tl_arbiter` lets the two links share the fabric.

```text
host ─ UART TSI ─ uart_tsi ─ tl_widen ──────┐
host ─ UDP ─ PHY ─ eth_udp_stack ─ eth_tsi ─┴─ tl_arbiter ─ tl_xbar ─┬─ sys_regs          0x0000_0000
                                                                     ├─ Atlas IMEM        0x0002_0000  (128 KiB)
                                                                     ├─ Atlas CSR         0x0004_0000
                                                                     ├─ Atlas VMEM        0x2000_0000  (1.5 MiB)
                                                                     └─ DRAM, port A      0x8000_0000 ─┐
                                                        AtlasCore DMA ── DRAM, port B ─────────────────┴─ DDR4 bridge ─ MIG ─ 2 GiB DDR4
```

- `chisel/`: `EmitAtlasCore` writes AtlasCore's SystemVerilog with fixed TL-UL ports, and
  `tilelink/` stands in for the three rocket-chip TileLink names AtlasCore uses. The RTL under
  `src/main/scala` is shared with the Chipyard build unchanged.
- `rtl/`: the SoC. `tsi/` is NVFP4's UART TSI target and fabric (copied from `ucb-bar/nvfp4@0c611e1:fpga/tsi` and `fpga/common`).
  The board's DRAM is 2 GiB of DDR4 through `atlas_ddr4_dram.sv` and the MIG (below).
  `atlas_dram.sv` is the Verilator model's DRAM, aliased over `0x8000_0000` up (DMA's 64-bit
  addresses fold onto it the same way); the baremetal tests use 465 KiB at `0x9000_0000`.
  (Until 2026-10-22 the board build could also use 4 MiB of UltraRAM; that option is gone.)
  `eth/` is TSI over UDP: `eth_tsi.sv` (the protocol), `eth_udp_stack.sv` (MAC, ARP, IPv4, UDP)
  and `vcu118_sgmii.sv` (the board's PHY). `eth/vendor/` is verilog-ethernet
  (`alexforencich/verilog-ethernet@77320a9`, MIT), unmodified.
- `sim/`: a Verilator model of `atlas_soc` with a word-level host port (no UART), and its driver.
  `MODEL=tsi` builds a model with the UART pins, `MODEL=eth` one with the Ethernet stack behind a
  UDP socket. `tb_eth_tsi.sh` and `tb_eth_udp_stack.sh` test the Ethernet modules alone.
- `host/`: `atlas_host.py` (the host: `SimLink` for the model, `TsiLink` for UART TSI, `EthLink`
  for TSI over UDP), `make_bundle.py` (assembles the baremetal tests into JSON bundles),
  `run_tests.py`, `link_bench.py` (throughput and latency of the links), `test_ethlink.py`
  (EthLink against a model of the board).
- `fpga/`: VCU118 constraints, the Vivado build (with the DDR4 MIG and the SGMII PCS/PMA IP) and
  the JTAG programming script.

## Configurations

`--config=NAME` selects the Atlas configuration (docs/nvfp4-mxu.md): `default` (inner-product
trees) or `ipt-nvfp4` (inner-product trees + the NVFP4 MXU). The results dated before 2026-10-25
name the configurations of that time: `default` had the systolic array (`mxu0`) too, and `ipt`
was today's `default`. `run_tests.py --config NAME` expects every test that needs a missing unit,
or that has the line `# @EXPECT_ILLEGAL`, to stop with an illegal instruction, and every other test
to pass. A bundle that is not a baremetal test (DiT, hazfuzz) must pass.

The ISA CSR (CSR block `+0x1C`, docs/perf-counters.md) gives the ISA version and the units. Each
bundle has the ISA version of the assembler that built it. `Atlas.run` does not run a bundle
built for another version than the core's: the status is ERROR.

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

`--no-delay` removes every `DELAY` from the kernels. The model keeps all RTL assertions, so a
hazard that got past the interlock fails the test.

The Ethernet model runs the same tests over UDP on 127.0.0.1, through `eth_udp_stack` and
`eth_tsi`. Each test gets its own model and UDP port:

```bash
MODEL=eth standalone/sim/build.sh build/rtl-sim build/vl_eth   # build/vl_eth/atlas_eth_sim
python3 standalone/host/run_tests.py --eth-sim build/vl_eth/atlas_eth_sim --config ipt-nvfp4 -j 16 \
    --out build/sim-eth build/bundle/nodelay/*.json
standalone/sim/tb_eth_tsi.sh                  # eth_tsi and tl_arbiter alone
standalone/sim/tb_eth_udp_stack.sh            # eth_udp_stack alone, on GMII frames
python3 standalone/host/test_ethlink.py       # EthLink against a model of the board
```

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

Over Ethernet (docs/tsi-over-udp.md, "Board setup"), connect the board's RJ45 port to a 1 Gb/s
NIC of the host. Give the NIC an address on 172.20.0.0/16. The FPGA is 172.20.1.18 (sys_regs
ETH_IP). After programming and after CPU_RESET, the PHY needs about 3 s to negotiate the copper
link. `EthLink` sends its first datagram (SYNC) again for at least 6 s. `arping` gets an answer,
`ping` does not (no ICMP). `EthLink` needs only the Python standard library.

```bash
sudo ip addr add 172.20.0.1/16 dev <nic>
sudo ip link set <nic> up
python3 standalone/host/run_tests.py --eth --config ipt-nvfp4 --out build/board-eth build/bundle/nodelay/*.json
python3 standalone/host/link_bench.py --eth --uart     # MB/s and latency of both links
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

### DDR4 bitstreams with the performance counters (2026-10-22, commit 50977db)

`bitstreams/AtlasStandaloneDDR4-v05/atlas_standalone_vcu118_<config>_ddr4_50mhz.bit`, Vivado 2024.1,
50 MHz. The tests are the 126-test suite (original kernels, then every `DELAY` removed) and
the randomized hazard test (`apps/dit/ditflow/hazfuzz.py`, 64 programs and the directed test).

| Config | WNS / WHS (ns) | LUTs | FFs | RAMB36 | DSP | suite, board | hazard test, board | DiT 100 steps | policy |
|---|---|---|---|---|---|---|---|---|---|
| `default` | +0.022 / +0.009 | 619K (52%) | 118K | 537.5 | 115 | 126 / 126, 126 / 126 as expected | 65 / 65 | 47,039,818 cycles, bit-exact | 57,307,319 cycles, bit-exact |
| `ipt` | +0.381 / +0.011 | 357K (30%) | 75K | 529.5 | 115 | 126 / 126, 126 / 126 as expected | 63 / 65 (2: model numerics) | 47,041,500 cycles, bit-exact | 57,307,630 cycles, bit-exact |
| `ipt-nvfp4` | +0.599 / +0.010 | 538K (45%) | 179K | 538.5 | 243 | 126 / 126, 126 / 126 as expected | 65 / 65 | 47,041,552 cycles, bit-exact | 57,307,596 cycles, bit-exact |

The `ipt-nvfp4` DDR4 build does not route with `build.tcl`'s default directives: congestion
(level 7) around MREG and MXU1 leaves 10,074 nets unrouted. `fpga/impl_retry.tcl` re-places
and re-routes `post_synth.dcp`; `SSI_SpreadLogic_high` placement with `Explore` routing closes
at the WNS above (about 75 minutes).

The performance counters cost 373 LUTs and 1.2K flip-flops in the CSR block. On tests of
1,000 cycles or more, board cycle counts are within ±5% of the Verilator model's (median 0%):
the model's DRAM has a fixed 8-cycle latency, the DDR4 does not. `run_tests.py` writes each
run's counters to `perf.csv`; the spec's §8.2 tabulates them.

The Verilator model keeps the behavioral DRAM, 64 MiB by default (`DRAM_LINE_W=21`; 17 makes it
the 4 MiB of the old on-chip build), and its `B <addr> <file>` command (used by `SimLink` for every DRAM preload) loads data
through a backdoor in no simulated time. A 3.4 MiB image used to cost 35 minutes of simulated
bus writes.

## TSI over UDP

The first board build with Ethernet (default config, 50 MHz) met timing with WNS +0.108 ns.
`eth_udp_stack`, `vcu118_sgmii` and `eth_tsi` together take 4,171 LUTs, 6,082 FFs, 14 RAMB36
and 5 RAMB18 (docs/tsi-over-udp.md, "Resources and timing").

Board results: pending.

## Driving it from Python

`run_tests.py` is one client of `host/atlas_host.py`. A kernel can also be run by hand; the
link is the same for the board (`TsiLink()` over UART, `EthLink()` over Ethernet) and the model
(`SimLink(binary)`):

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
- Reset: CPU_RESET (L19); the SoC also resets when the MMCM loses lock. CPU_RESET also resets
  the Ethernet PHY and sets ETH_IP back to its default.
- LEDs: `led[0]` heartbeat, `led[1]` TSI activity (toggles when a UART command or an Ethernet
  datagram completes), `led[2]` link error or denied access, `led[3]` MMCM locked and DDR4
  calibrated, `led[4]` Atlas halted, `led[7:5]` the sys_regs LEDS register.
- Ethernet: the DP83867 PHY on the RJ45 port, over SGMII (bank 64). IPv4 172.20.1.18, UDP port
  21587, MAC `02:00:41:54:4c:53`. Protocol, setup, link checks and limits: docs/tsi-over-udp.md.

## System registers (`0x0000_0000`)

| Offset | Name | |
|---|---|---|
| `0x000` | ID | `0x4154_4C53` ("ATLS") |
| `0x004` | SCRATCH | read/write |
| `0x008` | FLAGS | [0] UART TSI receive overflow, [1] bad UART TSI command, [2] denied access seen, [3] DRAM ready (DDR4 calibrated; 1 in the Verilator model) |
| `0x00C` | DENIED | denied accesses of both links (see below) |
| `0x010` | CYCLES | free-running cycle counter |
| `0x014` | CLK_HZ | |
| `0x018` | BAUD | |
| `0x01C` | DRAM_BYTES | 2 GiB of DDR4 (`0x8000_0000`) on the board; the model's size in Verilator |
| `0x020` | ATLAS | [0] halted, [1] 0, [2] MXU slot 0 (NVFP4 MXU) busy, [3] MXU1 data busy, [4] MXU1 compute busy, [5] XLU busy, [6] LSU busy, [15:8] DMA channel busy |
| `0x024` | FEATURES | [0] hardware interlock, [1] 0, [2] inner-product-tree MXU, [3] NVFP4 MXU |
| `0x028` | LEDS | [2:0] |
| `0x02C` | RUN_CYCLES | cycles the scalar core has run (not halted); a write clears it |
| `0x030` | ETH_IP | the FPGA's IPv4 address, read/write; 172.20.1.18 (`0xAC14_0112`) after reset |
| `0x034` | ETH_STATUS | [15:0] PCS/PMA status (0 while the GMII clock is stopped), [16] PHY bring-up done, [17] Ethernet session open, [18] Ethernet present, [19] PHY ID matched, [20] GMII clock running |
| `0x038` | ETH_RUN | datagrams run |
| `0x03C` | ETH_DUP | datagrams answered DUP (not run again) |
| `0x040` | ETH_DROP | datagrams dropped or not run |
| `0x044` | ETH_PORT | UDP port, 21587 (`0x5453`) |
| `0x048` | ETH_MAC_LO | MAC address [31:0] |
| `0x04C` | ETH_MAC_HI | MAC address [47:32] |
| `0x050` | ETH_RXERR | [15:0] frames dropped before the TSI target, [31:16] frames with a bad FCS or a GMII error |
| `0x054` | ETH_RXFRAMES | [15:0] frames at the GMII input, [31:16] frames out of the MAC |

In a build without Ethernet (the word-level Verilator model), ETH_STATUS and the Ethernet
counters read 0. docs/tsi-over-udp.md has the bit details.

The SoC answers an access that a device cannot serve now at once, with a denied response. Such
a request thus never holds the fabric, and the other link keeps working. IMEM reads while the
scalar core runs are denied and return 0, because IMEM serves the fetch: stop the core first.
IMEM writes work while the core runs. On the board, DRAM accesses before the DDR4 is calibrated
(FLAGS bit 3 is 0) are denied too.

## Performance counters

The Atlas CSR block (`0x0004_0000`) has 27 64-bit counters: cycles, issued instructions, stall
cycles by kind, the busy cycles of each unit, and the stall cycles by unit and data cause. Write
1 to bit i of PERF_RESET (`+0x100`) to clear counter i; counter i is at `+0x200 + 8i`. The scalar
core reads them as CSRs `0xCC2 + 2i` and `0xCC3 + 2i`. See docs/perf-counters.md.

Verilator, 2026-10-24: the suite has 129 tests (the 126 above, `csr_isa_perf` and the two
`csr_illegal_*` tests). Every `DELAY` removed, `default` gives 129 / 129 as expected (122 pass, 7
illegal), `ipt` 129 / 129 (94, 35) and `ipt-nvfp4` 129 / 129 (99, 30). The original kernels on
`default` give 129 / 129. On `ipt-nvfp4` the 126 earlier tests keep their status and cycle count.
`csr_isa_perf` reads the ISA CSR as `0x0100_0007`, `0x0100_0005` and `0x0100_000D`.
`perf_mm_single` measures its window as 32 cycles (100% MXU utilization, 106% before: a `csrrw`
to the unmapped CSR `0xC20` cleared the cycle counter).

Verilator, 2026-10-25, without the systolic array: the suite has 114 tests. 13 tests that used
`mxu0` with DMA, VPU, XLU or interlock traffic now use `mxu1`. The `mxu0` copies of `mxu1` tests and
the two dual-MXU tests (15 tests) are gone. With the original kernels and with every `DELAY`
removed, `default` gives 114 / 114 as expected (107 pass, 7 illegal) and `ipt-nvfp4` 114 / 114
(112, 2). The other 101 tests keep their status and cycle count on both configurations, and
`default` gives the same cycles and performance counters as `ipt` did.

Verilator, 2026-10-25, ISA 2.0 with the test additions: the suite has 125 tests. The 10 new tests
are six `csr_illegal_*` tests (writes to cycle and instret, the ISA 1.0 CSR numbers),
`illegal_word_v1`, `illegal_word_reserved_bit`, `seld_byte_offsets` and `nvfp4_mxu1_overlap`.
With the original kernels and with every `DELAY` removed, `default` gives 125 / 125 as expected
(108 pass, 17 illegal) and `ipt-nvfp4` 125 / 125 (114, 11). The other 115 tests keep their
status, cycle count and performance counters on both configurations. 16 of them load other FP8
data, because the generators now encode E4M3 as `E4M3_RN_SATFINITE` (`gen_utils.py`).

Verilator, 2026-10-25, with the scaled BF16 pops, the `mxu1` FP32 transfers, the `.vx` and `.vr`
forms and `vgelu.bf16`: the suite has 133 tests. The 8 new tests are `mxu1_scaled_pop`,
`mxu1_fp32_spill`, `mxu1_back_to_back`, `nvfp4_scaled_pop`, `nvfp4_fp32_spill`, `vpu_vx_vr`,
`interlock_xreg` and `vpu_gelu`. With the original kernels and with every `DELAY` removed,
`default` gives 133 / 133 as expected (114 pass, 19 illegal) and `ipt-nvfp4` 133 / 133 (122, 11).
The other 125 tests keep their status, cycle count and performance counters on both
configurations, with one exception: without `DELAY`, `mem_scalar` takes 105 cycles, not 107, and
`scalar_vector_concurrent` counts 3 fewer x-register stall cycles. An instruction now waits for a
pending scalar load only on the x-registers that it names (docs/interlock.md).
