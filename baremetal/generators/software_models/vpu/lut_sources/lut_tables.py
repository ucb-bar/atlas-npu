"""lut_sources/lut_tables.py — bit-exact ports of the
`VecInit.tabulate(entries)` blocks in the four Chisel LUT modules.

Each generator returns a Python `list[int]` of length `1 << addr_bits`,
with values in `[0, 2^(m+n))`. The Chisel version stores the value as
a `(m + n)`-wide `UInt`; here we return the same integer width.

Sources (under `dependencies/sp26-fp-units/.../vpuLUTs/`):
  - `RcpLUT.scala:17-23`     → `gen_rcp_lut`
  - `SqrtLUT.scala:17-23`    → `gen_sqrt_lut`     (also `sqrt(max_val)`)
  - `LogLUT.scala:16-22`     → `gen_log_lut`
  - `SinCosLUT.scala:19-25`  → `gen_sin_cos_lut`
  - `ExLUT.scala:14-21`      → `gen_ex_lut`       (placeholder consumer)
  - `TanhLUT.scala:30-36`    → `gen_tanh_lut`     (used by `TanhRec`)

The four formulas are simple `i.toDouble * (max - min) / entries` linear
scans. `BigInt(math.round(v * (1 << n)))` in Scala uses HALF_UP for ties
(per Java's `Math.round`); Python's built-in `round()` uses banker's
rounding, which can differ on exact-half cases. Use the explicit
`int(math.floor(v * scale + 0.5))` formula for HALF_UP parity.
"""

from __future__ import annotations

import math
from typing import Callable

__all__ = [
    "round_half_up",
    "gen_rcp_lut",
    "gen_sqrt_lut",
    "sqrt_max_val",
    "gen_log_lut",
    "gen_sin_cos_lut",
    "gen_ex_lut",
    "gen_tanh_lut",
]


def round_half_up(value: float) -> int:
    """Java `Math.round(double)` semantics — round-half-up to nearest int.

    Differs from Python's `round()` (banker's rounding) on exact-half
    inputs. Matches `BigInt(math.round(v * (1 << n)))` in the LUT
    generators byte-for-byte.

    Negative half-cases: Java's `Math.round(x)` is `floor(x + 0.5)`,
    which for negatives rounds toward +inf (-1.5 → -1, not -2). We
    replicate that exactly.
    """
    return int(math.floor(value + 0.5))


def _gen_lut(
    addr_bits: int,
    m: int,
    n: int,
    func: Callable[[float], float],
    minimum: float,
    maximum: float,
) -> list[int]:
    """Internal helper. Mirrors the shared formula
        for i in 0 until entries:
            r = min + i * (max - min) / entries
            v = func(r)
            scaled = round(v * (1 << n))
    where `entries = 1 << addr_bits` and the scaled value is an
    unsigned `(m + n)`-wide int.
    """
    entries = 1 << addr_bits
    out_mask = (1 << (m + n)) - 1
    table: list[int] = [0] * entries
    for i in range(entries):
        r = minimum + i * (maximum - minimum) / entries
        v = func(r)
        scaled = round_half_up(v * (1 << n))
        table[i] = scaled & out_mask
    return table


def gen_rcp_lut(
    addr_bits: int = 7,
    m: int = 1,
    n: int = 16,
    minimum: float = 1.0,
    maximum: float = 2.0,
) -> list[int]:
    """Mirror of `RcpLUT.scala:17-23`.

    Default `addr_bits=7, m=1, n=16` matches the Rcp.scala wiring
    (`new RcpLUT(numLanes, sigWidth-1, BF16T.lutValM, BF16T.lutValN)`).
    """
    return _gen_lut(addr_bits, m, n, lambda r: 1.0 / r, minimum, maximum)


def sqrt_max_val(n: int = 16, maximum: float = 2.0) -> int:
    """Mirror of `SqrtLUT.scala:15`:
        val maxVal = BigInt(Math.round(Math.sqrt(max) * (1 << n)))

    Used by the SqrtLUT to compute the odd-exp shifted base."""
    return round_half_up(math.sqrt(maximum) * (1 << n))


def gen_sqrt_lut(
    addr_bits: int = 7,
    m: int = 1,
    n: int = 16,
    minimum: float = 1.0,
    maximum: float = 2.0,
) -> list[int]:
    """Mirror of `SqrtLUT.scala:17-23`. Same default args as RcpLUT."""
    return _gen_lut(addr_bits, m, n, math.sqrt, minimum, maximum)


def gen_log_lut(
    addr_bits: int = 7,
    m: int = 9,
    n: int = 16,
    minimum: float = 1.0,
    maximum: float = 2.0,
) -> list[int]:
    """Mirror of `LogLUT.scala:16-22`.

    Default `m=9` matches the **hardcoded** `9` passed by Log.scala (NOT
    `BF16T.lutValM`); `n=16` is `BF16T.lutValN`. The LUT stores
    log2(r) for r in [1, 2), so all values are in [0, 1) and trivially
    fit in any `m >= 1`. The wider `m` field is used downstream when
    `LogLUT.scala:30-33` adds the absolute exponent shifted left by `n`
    to the table value.
    """
    return _gen_lut(addr_bits, m, n,
                    lambda r: math.log(r) / math.log(2.0),
                    minimum, maximum)


def gen_sin_cos_lut(
    addr_bits: int = 5,
    m: int = 1,
    n: int = 16,
    minimum: float = 0.0,
    maximum: float = math.pi / 2,
) -> list[int]:
    """Mirror of `SinCosLUT.scala:19-25`.

    Defaults `addr_bits=5, m=1, n=16` match BF16's
    `(BF16T.lutAddrBits, BF16T.lutValM, BF16T.lutValN) = (5, 1, 16)`.
    The table stores `sin(r)` for r in `[0, pi/2)`; cosine is derived
    by reflecting the address (handled in `lane_boxes/sin_cos_vec.py`).
    """
    return _gen_lut(addr_bits, m, n, math.sin, minimum, maximum)


def gen_ex_lut(
    addr_bits: int = 5,
    m: int = 1,
    n: int = 16,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> list[int]:
    """Mirror of `ExLUT.scala:14-21`. Stores `2^r` for r in `[0, 1)`.
    Used by the Exp lane_box (deferred — placeholder consumer)."""
    return _gen_lut(addr_bits, m, n,
                    lambda r: math.pow(2.0, r),
                    minimum, maximum)


def gen_tanh_lut(
    addr_bits: int = 5,
    m: int = 1,
    n: int = 16,
    minimum: float = 0.0,
    maximum: float = 4.0,
) -> list[int]:
    """Mirror of `TanhLUT.scala:30-36`. Stores `tanh(r)` for r in
    `[0, 4)`. Used by the `TanhRec` lane box."""
    return _gen_lut(addr_bits, m, n, math.tanh, minimum, maximum)
