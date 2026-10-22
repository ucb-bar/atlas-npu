#!/usr/bin/env bash
# Emit AtlasCore without assertions and build the VCU118 bitstream.
#   standalone/fpga/build_bitstream.sh [out_dir] [clk_mhz] [jobs] [config] [dram]
# config: default, ipt or ipt-nvfp4 (atlas.common.AtlasConfigs); dram: onchip (4 MiB UltraRAM) or
# ddr4 (2 GiB of the board's DDR4). The bitstream is
# atlas_standalone_vcu118_<config>[_ddr4]_<clk>mhz.bit.
# Needs `vivado` (2024.1 or later, with a VU9P license) on PATH and Java for mill.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="$(realpath -m "${1:-$root/build/vcu118}")"
clk="${2:-50}"
jobs="${3:-16}"
config="${4:-default}"
dram="${5:-onchip}"

cd "$root"
rm -rf "$out/rtl"
mkdir -p "$out"
./mill --no-server standalone.runMain atlas.standalone.EmitAtlasCore "$out/rtl" --config="$config"
python3 "$root/standalone/fpga/fpga_mems.py" "$out/rtl"   # VMEM banks -> block-RAM template
cd "$out"
vivado -mode batch -nojournal -log vivado.log -source "$root/standalone/fpga/build.tcl" \
    -tclargs "$out/rtl" "$out" "$clk" "$jobs" "$config" "$dram"
