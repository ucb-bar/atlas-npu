#!/bin/bash
# Build and run tb_eth_udp_stack.cpp (eth_udp_stack on raw GMII frames) in Verilator 5.
#   standalone/sim/tb_eth_udp_stack.sh [out_dir] [seed] [clk_hz]
# clk_hz: the stack's CLK_HZ (default 50000000); the core clock runs within 4% of it.
set -euo pipefail
sa=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-build/tb_eth_udp_stack}
clk=${3:-50000000}
verilator --cc --exe --build -O3 -Wno-fatal -Wno-lint -Wno-style --timescale 1ns/1ps \
    --top-module eth_udp_stack -GCLK_HZ="$clk" -Mdir "$out" -o tb_eth_udp_stack -j 1 --build-jobs "$(nproc)" \
    -CFLAGS "-O2 -DCLK_HZ=$clk" "$sa/rtl/eth/eth_udp_stack.sv" "$sa"/rtl/eth/vendor/*.v "$sa/sim/tb_eth_udp_stack.cpp"
"$out/tb_eth_udp_stack" "${2:-1}"
