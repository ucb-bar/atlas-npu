"""Byte layouts of tensors in Atlas DRAM and VMEM.

All matrices are padded to 32 x 32 tiles, stored tile by tile in row-major tile order: tile
(r, c) of an R x C matrix sits at ((r * C/32) + c) * tile_bytes.

- FP8 tile, 1 KiB: 32 rows x 32 bytes, row-major. This is how an m register holds a matmul
  operand (vload'ed whole) and how a weight slot holds W rows: slot row j = output column j.
- BF16 tile, 2 KiB: the MXU/VPU "tile view" pair. The first KiB holds columns 0-15 of all 32
  rows (32 bytes per row), the second KiB columns 16-31; vload'ed into m[r] and m[r + 1].
- Weight blocks: for each 32-wide output tile nt of a Linear (N x K), its K/32 FP8 tiles of
  W[nt*32:(nt+1)*32, :] followed by its bias as a BF16 tile replicated over 32 rows. One block
  is what the kernel DMAs before computing that output tile.
"""
import torch

T = 32
FP8_TILE = 1024
BF16_TILE = 2048


def tiles(n: int) -> int:
    return -(-n // T)


def pad2(x: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    out = torch.zeros(rows, cols, dtype=x.dtype)
    out[:x.shape[0], :x.shape[1]] = x
    return out


def bf16_bytes(m: torch.Tensor) -> bytes:
    """A matrix (any float dtype) as BF16 tiles."""
    rt, ct = tiles(m.shape[0]), tiles(m.shape[1])
    p = pad2(m.to(torch.bfloat16), rt * T, ct * T)
    t = p.view(rt, T, ct, 2, 16).permute(0, 2, 3, 1, 4)       # rt, ct, half, row, col
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def from_bf16_bytes(data: bytes, rows: int, cols: int) -> torch.Tensor:
    rt, ct = tiles(rows), tiles(cols)
    t = torch.frombuffer(bytearray(data[:rt * ct * BF16_TILE]), dtype=torch.bfloat16)
    t = t.view(rt, ct, 2, T, 16).permute(0, 3, 1, 2, 4).reshape(rt * T, ct * T)
    return t[:rows, :cols]


def fp8_bytes(codes: torch.Tensor) -> bytes:
    """A matrix of FP8 codes (uint8) as FP8 tiles."""
    rt, ct = tiles(codes.shape[0]), tiles(codes.shape[1])
    p = pad2(codes.to(torch.uint8), rt * T, ct * T)
    return p.view(rt, T, ct, T).permute(0, 2, 1, 3).contiguous().numpy().tobytes()


def from_fp8_bytes(data: bytes, rows: int, cols: int) -> torch.Tensor:
    rt, ct = tiles(rows), tiles(cols)
    t = torch.frombuffer(bytearray(data[:rt * ct * FP8_TILE]), dtype=torch.uint8)
    return t.view(rt, ct, T, T).permute(0, 2, 1, 3).reshape(rt * T, ct * T)[:rows, :cols]


def replicated(v: torch.Tensor, rows: int = T) -> torch.Tensor:
    """A vector repeated over `rows` rows."""
    return v.reshape(1, -1).expand(rows, -1)


def weight_block_bytes(codes: torch.Tensor, bias: torch.Tensor | None) -> int:
    kt = tiles(codes.shape[1])
    return kt * FP8_TILE + (BF16_TILE if bias is not None else 0)


def weight_blocks(codes: torch.Tensor, bias: torch.Tensor | None) -> bytes:
    """Weight codes [N, K] (uint8) and an optional bias [N] as one block per output tile."""
    nt, kt = tiles(codes.shape[0]), tiles(codes.shape[1])
    w = pad2(codes.to(torch.uint8), nt * T, kt * T)
    b = None if bias is None else pad2(bias.reshape(1, -1).float(), 1, nt * T)[0]
    out = bytearray()
    for n in range(nt):
        out += fp8_bytes(w[n * T:(n + 1) * T])
        if b is not None:
            out += bf16_bytes(replicated(b[n * T:(n + 1) * T]))
    return bytes(out)


def bf16_bits(value: float) -> int:
    """Raw BF16 bits of a value, rounded to nearest even (for vli immediates)."""
    return int(torch.tensor([value], dtype=torch.float32).to(torch.bfloat16).view(torch.int16).item()) & 0xFFFF


def pow2_bits(k: int) -> int:
    assert -126 <= k <= 127
    return (127 + k) << 7
