#!/usr/bin/env python3
"""gen_lsu_rows.py — golden data for assembly/lsu_rows.S: vload and vstore at row counts M.

vload of M rows writes rows 0..M-1 of m[vd] from M VMEM lines, vstore writes M lines from rows
0..M-1; the other rows and lines keep their values.

DRAM beats: the tile T at 0, the start values of m8..m11 at 32 + 32 (r - 8), the start values of
the VMEM tiles R1 and R2 at 160 and 192; the results R1, R2, m8..m11 at 1024.
"""
import random

from nvfp4_common import Image

rng = random.Random(4095)


def tile():
    return [bytes(rng.randrange(256) for _ in range(32)) for _ in range(32)]


t, r1, r2 = tile(), tile(), tile()
start = {r: tile() for r in range(8, 12)}
out = {"r1": list(r1), "r2": list(r2), **{r: list(rows) for r, rows in start.items()}}
out[8][:3] = t[:3]                      # vload m8 at M = 3
out[10][:1] = t[:1]                     # vload m10 at M = 1
out["r1"][:5] = start[9][:5]            # vstore m9 at M = 5
out["r2"][:31] = start[11][:31]         # vstore m11 at M = 31

img = Image()
img.preload(0, b"".join(t) + b"".join(b"".join(start[r]) for r in range(8, 12)) + b"".join(r1) + b"".join(r2))
img.check(1024, b"".join(b"".join(out[k]) for k in ("r1", "r2", 8, 9, 10, 11)))
img.dump()
