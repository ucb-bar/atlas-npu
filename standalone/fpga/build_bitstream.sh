#!/usr/bin/env bash
# Emit AtlasCore without assertions and build the VCU118 bitstream.
#   standalone/fpga/build_bitstream.sh [out_dir] [clk_mhz] [jobs]
# Needs `vivado` (2024.1 or later, with a VU9P license) on PATH and Java for mill.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="$(realpath -m "${1:-$root/build/vcu118}")"
clk="${2:-50}"
jobs="${3:-16}"

cd "$root"
rm -rf "$out/rtl"
mkdir -p "$out"
./mill --no-server standalone.runMain atlas.standalone.EmitAtlasCore "$out/rtl"
python3 "$root/standalone/fpga/fpga_mems.py" "$out/rtl"   # VMEM banks -> block-RAM template
cd "$out"
vivado -mode batch -nojournal -log vivado.log -source "$root/standalone/fpga/build.tcl" \
    -tclargs "$out/rtl" "$out" "$clk" "$jobs"
