"""VPUOp enum mirroring src/main/scala/atlas/vector/VectorIO.scala.

Order must match the Scala ChiselEnum exactly — `VPUOp.add.value == 0`,
`VPUOp.sub.value == 1`, etc. ScalarISA is 1-indexed (VPU_NONE=0, VPU_ADD=1, …)
and `VectorEngineTop` subtracts 1 before casting; the Python side matches the
ChiselEnum's 0-indexed layout.

VPUOp.fp8 exists in the Scala enum but is NOT wired in the live engine —
VectorMVP.scala's fp8 references are all commented out. We keep it here so
the enum-parity canary test asserts one-to-one parity, and reject it at
VectorEngineModel.execute() time.
"""

from __future__ import annotations

from enum import IntEnum


class VPUOp(IntEnum):
    add = 0
    sub = 1
    mul = 2
    rcp = 3
    sqrt = 4
    sin = 5
    cos = 6
    tanh = 7
    log = 8
    exp = 9
    exp2 = 10
    square = 11
    cube = 12
    rsum = 13
    csum = 14
    fp8 = 15
    fp8pack = 16
    fp8unpack = 17
    relu = 18
    rmax = 19
    rmin = 20
    cmax = 21
    cmin = 22
    pairmax = 23
    pairmin = 24
    mov = 25
    vliOne = 26
    vliCol = 27
    vliRow = 28
    vliAll = 29


SCALA_ENUM_ORDER: tuple[str, ...] = (
    "add", "sub", "mul", "rcp", "sqrt", "sin", "cos", "tanh", "log", "exp", "exp2",
    "square", "cube", "rsum", "csum", "fp8", "fp8pack", "fp8unpack", "relu",
    "rmax", "rmin", "cmax", "cmin", "pairmax", "pairmin", "mov",
    "vliOne", "vliCol", "vliRow", "vliAll",
)
