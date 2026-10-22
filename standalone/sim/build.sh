#!/usr/bin/env bash
# Build the Verilator model of the standalone SoC (atlas_soc_sim + atlas_sim.cpp).
#   standalone/sim/build.sh [rtl_dir] [out_dir]
# rtl_dir is the Chisel output with assertions (default build/rtl-sim, from
# `./mill standalone.runMain atlas.standalone.EmitAtlasCore build/rtl-sim --assertions`).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
rtl="$(realpath "${1:-$root/build/rtl-sim}")"
out="$(realpath -m "${2:-$root/build/vl_soc}")"
sa="$root/standalone"
jobs="${JOBS:-$(nproc)}"
# Verilation runs single-threaded: Verilator 5.022 aborts with "attempted to destroy locked
# Thread Pool" after a multi-threaded run (as bin/verilator works around for the Chisel tests).

srcs=(
    "$sa/rtl/tsi/tl_pkg.sv" "$sa/rtl/tsi/sync_fifo.sv" "$sa/rtl/tsi/tl_xbar.sv"
    "$sa/rtl/tsi/tl_widen.sv" "$sa/rtl/tsi/tl_narrow.sv"
    "$sa/rtl/atlas_tl_port.sv" "$sa/rtl/atlas_dram.sv" "$sa/rtl/atlas_sys_regs.sv"
    "$sa/rtl/atlas_soc.sv" "$sa/sim/atlas_soc_sim.sv"
)
mapfile -t chisel < <(ls "$rtl"/*.sv)

mkdir -p "$out"
verilator --cc --exe --build -O3 --assert -Wno-fatal -Wno-lint -Wno-style \
    --top-module atlas_soc_sim -Mdir "$out" -o atlas_sim \
    -j 1 --build-jobs "$jobs" -CFLAGS "-O2" \
    "${srcs[@]}" "${chisel[@]}" "$sa/sim/atlas_sim.cpp"
echo "built $out/atlas_sim"
