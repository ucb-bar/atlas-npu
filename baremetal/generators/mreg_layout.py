"""m-register layouts of MXU operands (docs/fp8-mxu.md), as the 32-bit DRAM words of the
generators' preloads and checks. An m-register is 32 rows of 32 bytes."""
import numpy as np


def words(regs) -> list[int]:
    """The words of m-registers, one register after another."""
    return np.concatenate([np.ascontiguousarray(r).view(np.uint32).ravel() for r in regs]).tolist()


def bf16_pair(tile):
    """A 32 x 32 BF16 tile (bits) in {m[d], m[d+1]}: columns 0-15, then 16-31."""
    return [tile[:, :16], tile[:, 16:]]


def fp32_quad(tile):
    """A 32 x 32 FP32 tile (bits) in m[d..d+3]: register q holds columns 8q..8q+7."""
    return [tile[:, 8 * q:8 * q + 8] for q in range(4)]
