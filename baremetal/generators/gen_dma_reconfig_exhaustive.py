#!/usr/bin/env python3
"""
gen_dma_reconfig_exhaustive.py — Golden data for dma_reconfig_exhaustive.S

FPGA variant: only the lower 4 GiB of DRAM is present, so we cannot
sweep DMA.CONFIG across multiple upper-32 base values.  Instead, this
generator emits goldens for an 8-window sweep across distinct lower-32
sub-regions inside the 0x9000_0000 window.

Transfer parameters (must match the assembly):
  BEAT_BYTES     = 32 (one DMA beat)
  BEATS_PER_XFER = 4  (128 bytes / 32 bytes per beat)
  WINDOWS        = 0 .. 7

For each window W:
  LOAD  source: DRAM[0x9000_0000 + W*0x10000 + 0x0000 .. + 0x007F]
  STORE dest:   DRAM[0x9000_0000 + W*0x10000 + 0x4000 .. + 0x407F]

Payload distinguishability:
  word[i] in beat[k] of window W =
      ((W & 0xF) << 28) | ((k & 0xFF) << 20) | ((i & 0xFF) << 12) | 0xABC

  This ensures every 32-bit word in DRAM is unique across all windows,
  beats, and positions.
"""

import json

BEAT_BYTES     = 32
WORDS_PER_BEAT = BEAT_BYTES // 4   # 8
BEATS_PER_XFER = 4                 # 128 bytes / 32 bytes
WINDOWS        = list(range(8))    # 0..7

# Byte offsets relative to @DRAM_BASE in the .S file (0x9000_0000 today).
WINDOW_STRIDE        = 0x10000     # 64 KiB between LOAD/STORE windows
LOAD_OFFSET_IN_WIN   = 0x0000      # LOAD lives at window-base + 0
STORE_OFFSET_IN_WIN  = 0x4000      # STORE lives at window-base + 0x4000


def payload_word(window: int, beat: int, word_in_beat: int) -> int:
    return (
        ((window & 0xF) << 28) |
        ((beat   & 0xFF) << 20) |
        ((word_in_beat & 0xFF) << 12) |
        0xABC
    )


def pack_beat(window: int, beat: int) -> str:
    val = 0
    for i in range(WORDS_PER_BEAT):
        val |= payload_word(window, beat, i) << (32 * i)
    return f"0x{val:064x}"


def beat_word_offset(dram_byte_addr: int, beat: int) -> int:
    """Convert a full 64-bit byte address + beat index into the assembler's
    `word_offset` field (number of BEAT_BYTES-sized beats from address 0)."""
    addr = dram_byte_addr + beat * BEAT_BYTES
    assert addr % BEAT_BYTES == 0
    return addr // BEAT_BYTES


def main() -> None:
    preloads: list[dict] = []
    checks:   list[dict] = []

    for window in WINDOWS:
        load_base  = window * WINDOW_STRIDE + LOAD_OFFSET_IN_WIN
        store_base = window * WINDOW_STRIDE + STORE_OFFSET_IN_WIN
        for beat in range(BEATS_PER_XFER):
            preloads.append({
                "word_offset": beat_word_offset(load_base, beat),
                "data":        pack_beat(window, beat),
            })
            checks.append({
                "word_offset": beat_word_offset(store_base, beat),
                "expected":    pack_beat(window, beat),
            })

    doc = {
        "description": (
            "FPGA exhaustive DMA.CONFIG reissue sweep. Eight reissues of "
            "DMA.CONFIG (upper-32=0) walk the lower-32 byte offset across "
            "eight 64KiB windows inside 0x9000_0000."
        ),
        "beat_bytes":    BEAT_BYTES,
        "beats_per_xfer": BEATS_PER_XFER,
        "windows":       [hex(w) for w in WINDOWS],
        "timeout":       500000,
        "dram_preloads": preloads,
        "dram_checks":   checks,
    }

    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
