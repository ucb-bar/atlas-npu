#!/bin/bash
# Build and run tb_ddr4_dram.sv (atlas_ddr4_dram under random traffic) in Verilator 5.
set -euo pipefail
sa=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-build/tb_ddr4}
verilator --binary --timing -Wno-fatal -Wno-lint -Wno-style --Mdir "$out" -I"$sa/rtl/tsi" \
    "$sa/rtl/tsi/tl_pkg.sv" "$sa/rtl/tsi/sync_fifo.sv" "$sa/rtl/async_fifo.sv" \
    "$sa/rtl/atlas_ddr4_dram.sv" "$sa/sim/tb_ddr4_dram.sv" --top tb_ddr4_dram -j 1
"$out/Vtb_ddr4_dram"
