#!/usr/bin/env bash
# Build a Verilator model of the standalone SoC.
#   standalone/sim/build.sh [rtl_dir] [out_dir]            word-level host port: atlas_soc_sim + atlas_sim.cpp
#   MODEL=tsi standalone/sim/build.sh [rtl_dir] [out_dir]  UART pins: atlas_tsi_soc + tsi_cosim.cpp
#   MODEL=eth standalone/sim/build.sh [rtl_dir] [out_dir]  GMII pins behind a UDP socket:
#                                                          atlas_eth_sim + eth_cosim.cpp
# rtl_dir is the Chisel output with assertions (default build/rtl-sim, from
# `./mill standalone.runMain atlas.standalone.EmitAtlasCore build/rtl-sim --assertions`).
# The UART model runs the link at 6.25 Mbaud (8 clocks per bit) instead of the board's 921,600.
# DRAM_LINE_W=n sizes the behavioral DRAM to 2^n x 32 bytes (default 21: 64 MiB; the board has
# 2 GiB of DDR4).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model="${MODEL:-soc}"
rtl="$(realpath "${1:-$root/build/rtl-sim}")"
sa="$root/standalone"
jobs="${JOBS:-$(nproc)}"
# Verilation runs single-threaded: Verilator 5.022 aborts with "attempted to destroy locked
# Thread Pool" after a multi-threaded run (as bin/verilator works around for the Chisel tests).

srcs=(
    "$sa/rtl/tsi/tl_pkg.sv" "$sa/rtl/tsi/sync_fifo.sv" "$sa/rtl/tsi/tl_xbar.sv"
    "$sa/rtl/tsi/tl_widen.sv" "$sa/rtl/tsi/tl_narrow.sv" "$sa/rtl/tsi/tl_arbiter.sv"
    "$sa/rtl/atlas_tl_port.sv" "$sa/rtl/atlas_dram.sv" "$sa/rtl/atlas_sys_regs.sv"
    "$sa/rtl/atlas_soc.sv"
)
tsi_soc=(
    "$sa/rtl/tsi/uart_rx.sv" "$sa/rtl/tsi/uart_tx.sv" "$sa/rtl/tsi/uart_tsi.sv"
    "$sa/rtl/eth/eth_tsi.sv" "$sa/rtl/atlas_tsi_soc.sv"
)
case "$model" in
    tsi)
        out="$(realpath -m "${2:-$root/build/vl_tsi}")"
        top=atlas_tsi_soc
        exe=atlas_tsi_sim
        srcs+=("${tsi_soc[@]}" "$sa/sim/tsi_cosim.cpp")
        extra=(-GCLK_HZ=50000000 -GBAUD=6250000 -CFLAGS -DCLKS_PER_BIT=8)
        ;;
    eth)
        # The verilog-ethernet modules come from the vendor directory as eth_udp_stack needs them.
        out="$(realpath -m "${2:-$root/build/vl_eth}")"
        top=atlas_eth_sim
        exe=atlas_eth_sim
        srcs+=("${tsi_soc[@]}" "$sa/rtl/eth/eth_udp_stack.sv" "$sa/sim/atlas_eth_sim.sv"
               "$sa/sim/eth_cosim.cpp")
        extra=(--timescale 1ns/1ps -y "$sa/rtl/eth/vendor" +libext+.v)
        ;;
    soc)
        out="$(realpath -m "${2:-$root/build/vl_soc}")"
        top=atlas_soc_sim
        exe=atlas_sim
        srcs+=("$sa/sim/atlas_soc_sim.sv" "$sa/sim/atlas_sim.cpp")
        extra=()
        ;;
    *)
        echo "MODEL must be soc, tsi or eth" >&2
        exit 1
        ;;
esac
extra+=(-GDRAM_LINE_W="${DRAM_LINE_W:-21}")
# SystemVerilog packages (dotprod_pkg of the NVFP4 MXU) must come before their users.
shopt -s nullglob
chisel=("$rtl"/*_pkg.sv)
for f in "$rtl"/*.sv; do [[ $f == *_pkg.sv ]] || chisel+=("$f"); done

mkdir -p "$out"
verilator --cc --exe --build -O3 --assert -Wno-fatal -Wno-lint -Wno-style \
    --top-module "$top" -Mdir "$out" -o "$exe" \
    -j 1 --build-jobs "$jobs" -CFLAGS "-O2" "${extra[@]}" \
    "${srcs[@]}" "${chisel[@]}"
echo "built $out/$exe"
