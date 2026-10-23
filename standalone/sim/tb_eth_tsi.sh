#!/bin/bash
# Build and run tb_eth_tsi.sv in Verilator 5: tb_eth_tsi (eth_tsi protocol and decoder) and
# tb_tl_arbiter (two random hosts).
#   standalone/sim/tb_eth_tsi.sh [out_dir] [+n=<random datagrams>] [+seed=<n>]
set -euo pipefail
sa=$(cd "$(dirname "$0")/.." && pwd)
out=build/tb_eth_tsi
if [ $# -gt 0 ] && [[ $1 != +* ]]; then out=$1; shift; fi
for top in tb_eth_tsi tb_tl_arbiter; do
    mkdir -p "$out/$top"
    verilator --binary --timing -Wno-fatal -Wno-lint -Wno-style --Mdir "$out/$top" \
        "$sa/rtl/tsi/tl_pkg.sv" "$sa/rtl/tsi/tl_arbiter.sv" "$sa/rtl/eth/eth_tsi.sv" \
        "$sa/sim/tb_eth_tsi.sv" --top "$top" -j 1 >/dev/null
    "$out/$top/V$top" "$@"
done
