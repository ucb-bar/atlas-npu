#!/usr/bin/env bash
# Build fp4_fedp_vec.cpp with Atlas's fp4_fedp RTL into build/fp4_fedp_vec, then run it on the
# NVFP4 silicon vectors in this directory (or on the .vec / .vec.gz files given as arguments).
#
#   baremetal/generators/nvfp4_silicon/fp4_fedp_vec.sh [FILE ...]
#
# Needs verilator (5.020 tested) and zlib. The build log is build/fp4_fedp_vec/build.log.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
vsrc=$root/src/main/resources/vsrc/nvfp4
out=$root/build/fp4_fedp_vec
mkdir -p "$out"

# The resources of Fp4Fedp in Nvfp4Mxu.scala, in the same order.
srcs=()
for f in dotprod_pkg front_end_nvfp4 mul_lane_nvfp4 align_nvfp4 exact_acc_tree fp4_fedp_acc fp4_fedp; do
    srcs+=("$vsrc/$f.sv")
done

if ! verilator --cc --exe --build -O3 -j 0 --top-module fp4_fedp \
        -Wno-fatal -Wno-lint -Wno-style -Wno-UNOPTFLAT \
        --Mdir "$out" -o fp4_fedp_vec -LDFLAGS -lz \
        "${srcs[@]}" "$here/fp4_fedp_vec.cpp" > "$out/build.log" 2>&1; then
    tail -n 40 "$out/build.log"
    exit 1
fi

[ $# -gt 0 ] || set -- "$here"/*.vec.gz
exec "$out/fp4_fedp_vec" "$@"
