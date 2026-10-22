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
rm -rf build/rtl-fpga
./mill --no-server standalone.runMain atlas.standalone.EmitAtlasCore build/rtl-fpga
mkdir -p "$out"
cd "$out"
vivado -mode batch -nojournal -log vivado.log -source "$root/standalone/fpga/build.tcl" \
    -tclargs "$root/build/rtl-fpga" "$out" "$clk" "$jobs"
