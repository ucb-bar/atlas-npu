#!/usr/bin/env python3
"""
gen_dma_config_base_exhaustive.py — Golden data for dma_config_base_exhaustive.S

Transfer parameters (must match the assembly):
  BEAT_BYTES     = 32 (one DMA beat)
  BEATS_PER_XFER = 4  (128 bytes / 32 bytes per beat)
  BASES          = 0x0 .. 0x7

For each base B:
  LOAD  source:  DRAM[B : 0x80000000 .. 0x8000007F]
  STORE dest:    DRAM[B : 0x80010000 .. 0x8001007F]

Payload distinguishability:
  word[i] in beat[k] of base B =
      0xB0000000 | (B << 24) | (k << 16) | (beat_word_index << 8) | i
  where i = word index within the beat (0..7), k = beat index (0..3).

  This scheme ensures every 32-bit word in DRAM is unique across all bases,
  beats, and positions, so a misrouted write (wrong upper address) cannot
  accidentally produce a false-positive match.

DRAM layout (flat 64-bit address space):
  preloads: address = Cat(B, 0x80000000 + beat_index * 32 + word_index * 4)
  checks:   address = Cat(B, 0x80010000 + beat_index * 32 + word_index * 4)

The assembler's --golden-json path reads dram_base fields as the upper 32 bits
of the 64-bit address.
"""

import json

BEAT_BYTES     = 32
WORDS_PER_BEAT = BEAT_BYTES // 4   # 8
BEATS_PER_XFER = 4                 # 128 bytes / 32 bytes
BASES          = list(range(8))    # 0x0 .. 0x7

LOAD_DRAM_OFFSET  = 0x80000000     # lower-32 starting byte for loads
STORE_DRAM_OFFSET = 0x80010000     # lower-32 starting byte for stores


def payload_word(base: int, beat: int, word_in_beat: int) -> int:
    """
    Unique 32-bit value for word `word_in_beat` of beat `beat` for base `base`.
    Bit layout:
      [31:28] base (4 bits, covers 0-7 with room)
      [27:20] beat index
      [19:12] word-in-beat index
      [11: 0] 0xABC marker (makes words visually identifiable in hex dumps)
    """
    return (
        ((base        & 0xF) << 28) |
        ((beat        & 0xFF) << 20) |
        ((word_in_beat & 0xFF) << 12) |
        0xABC
    )


def pack_beat(base: int, beat: int) -> str:
    """Pack WORDS_PER_BEAT 32-bit words into a single wide hex string (LSW first)."""
    val = 0
    for i in range(WORDS_PER_BEAT):
        val |= payload_word(base, beat, i) << (32 * i)
    return f"0x{val:064x}"


def beat_word_offset(dram_base_offset: int, beat: int) -> int:
    """Word offset (in beats, not bytes) from dram_base_offset for beat index `beat`."""
    byte_addr = dram_base_offset + beat * BEAT_BYTES
    assert byte_addr % BEAT_BYTES == 0, f"Unaligned beat address: {byte_addr:#x}"
    return byte_addr // BEAT_BYTES


def main() -> None:
    preloads: list[dict] = []
    checks:   list[dict] = []

    for base in BASES:
        for beat in range(BEATS_PER_XFER):
            # ── Preload: host writes this to DRAM before Atlas starts ──────────
            # Address: Cat(base, LOAD_DRAM_OFFSET + beat * BEAT_BYTES)
            preloads.append({
                "dram_base":   hex(base),
                "word_offset": beat_word_offset(LOAD_DRAM_OFFSET, beat),
                "data":        pack_beat(base, beat),
            })

            # ── Check: after Atlas runs, host reads this back ─────────────────
            # The DMA.STORE for base B must write to Cat(base, STORE_DRAM_OFFSET
            # + beat * BEAT_BYTES).  The expected payload is identical to the
            # preload payload because the LOAD fills VMEM which the STORE then
            # drains — a round-trip check.
            # Address: Cat(base, STORE_DRAM_OFFSET + beat * BEAT_BYTES)
            checks.append({
                "dram_base":   hex(base),
                "word_offset": beat_word_offset(STORE_DRAM_OFFSET, beat),
                "expected":    pack_beat(base, beat),
            })

    doc = {
        "description": (
            "Exhaustive DMA.CONFIG base register coverage. "
            "Each of the 8 base values (0x0-0x7) is exercised with both a "
            "LOAD and a STORE. Preloads populate DRAM[B:0x80000000..0x8000007F]; "
            "checks verify DRAM[B:0x80010000..0x8001007F] after round-trip."
        ),
        "beat_bytes":    BEAT_BYTES,
        "beats_per_xfer": BEATS_PER_XFER,
        "bases_covered": [hex(b) for b in BASES],
        "timeout":       500000,
        "dram_preloads": preloads,
        "dram_checks":   checks,
    }

    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
