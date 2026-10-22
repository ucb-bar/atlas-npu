#!/usr/bin/env python3
"""gen_nvfp4_scale_roundtrip.py — golden data for assembly/nvfp4_scale_roundtrip.S.

vload.scale / vstore.scale move a register's 128 B of scales unchanged, and a register's data and
its scale sidecar are independent: loading data into m6 after its scales leaves the scales, and
the scales come back out of m9 after vstore.scale m6 / vload.scale m9 / vstore.scale m9.

DRAM beats: scales 0-3, data 8-39, scales out 48-51, data out 56-87, scales via m9 96-99.
"""
import random

from nvfp4_common import Image

rng = random.Random(4005)
scales = bytes(rng.randrange(256) for _ in range(128))
data = bytes(rng.randrange(256) for _ in range(1024))

img = Image()
img.preload(0, scales); img.preload(8, data)
img.check(48, scales); img.check(56, data); img.check(96, scales)
img.dump()
