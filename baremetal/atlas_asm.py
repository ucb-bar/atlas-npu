#!/usr/bin/env python3
"""
atlas_asm.py — Atlas baremetal harness using npu_model's assembler.

Outputs hex text, raw binary or a C test harness, or disassembles a program.

Usage:
    python atlas_asm.py assembly/alu_basic.S
    python atlas_asm.py input.S --out-hex program.hex
    python atlas_asm.py input.S --out-bin program.bin
    python atlas_asm.py input.S --out-hex program.hex --out-bin program.bin
    python atlas_asm.py input.S --out-c header.c
    python atlas_asm.py --disasm program.hex      # or program.bin
"""

import argparse
import io
import os
import re
import struct
import sys



def encode(source):
    from npu_model.util.converter import input_to_program
    return input_to_program(io.StringIO(source)).assemble()

assemble = encode


def disassemble_file(path):
    """Assembly of a program in hex text (one word per line) or, for a .bin file, raw binary."""
    from npu_model.isa.disasm import disassemble
    with open(path, "rb") as f:
        data = f.read()
    if path.endswith(".bin"):
        words = list(struct.unpack(f"<{len(data) // 4}I", data))
    else:
        words = [int(token, 16) for token in data.decode().split()]
    return disassemble(words)


def _hex_to_words(hex_str):
    """Convert an arbitrarily wide hex string to a list of 32-bit words, LSW first."""
    val = int(hex_str, 16)
    if val == 0:
        return [0]
    words = []
    while val > 0:
        words.append(val & 0xFFFFFFFF)
        val >>= 32
    return words


def _parse_golden_json(path, dram_base, beat_bytes):
    """Parse golden JSON → (preload_addrs[], preload_words[], check_addrs[], check_words[]).

    `dram_base` is treated as a full 64-bit byte address. Each entry's
    `beat_addr = dram_base + beat_off * beat_bytes`.
    """
    import json
    with open(path) as f:
        data = json.load(f)

    words_per_beat = beat_bytes // 4

    def _parse_base(value):
        if value is None:
            return dram_base
        return int(value, 0) if isinstance(value, str) else int(value)

    def _expand(entries, key):
        """Convert beat-level entries to sorted per-word (byte_addr, u32) pairs."""
        pairs = []
        for e in entries:
            entry_dram_base = _parse_base(e.get("dram_base", dram_base))
            beat_off = e["word_offset"]
            beat_addr = entry_dram_base + beat_off * beat_bytes
            raw = _hex_to_words(e[key])
            while len(raw) < words_per_beat:
                raw.append(0)
            for i, word in enumerate(raw[:words_per_beat]):
                pairs.append((beat_addr + i * 4, word))
        pairs.sort(key=lambda p: p[0])
        return pairs

    preload_pairs = _expand(data.get("dram_preloads", []), "data")
    check_pairs   = _expand(data.get("dram_checks", []),   "expected")

    def _split(pairs):
        if not pairs:
            return ([], [])
        addrs = []
        words = []
        for addr, word in pairs:
            addrs.append(addr)
            words.append(word)
        return (addrs, words)

    p_addrs, p_words = _split(preload_pairs)
    c_addrs, c_words = _split(check_pairs)
    return p_addrs, p_words, c_addrs, c_words


def _parse_inline_golden(source, beat_bytes):
    """Parse inline @DRAM annotations from assembly comments.

    Supported directives:
      # @DRAM_BASE   <addr>            # full 64-bit byte address
      # @DRAM        <beat_offset> <hex_payload>
      # @CHECK_DRAM  <beat_offset> <hex_payload>

    Returns:
      (dram_base, preload_addrs, preload_words, check_addrs, check_words)
      or None if no inline preload/check directives were found.
    """
    dram_base = 0x00000000
    words_per_beat = beat_bytes // 4

    dram_base_re = re.compile(r"^\s*#\s*@DRAM_BASE\s+(\S+)\s*$", re.IGNORECASE)
    dram_re = re.compile(r"^\s*#\s*@DRAM\s+(\d+)\s+(0x[0-9a-fA-F]+)\s*$", re.IGNORECASE)
    check_re = re.compile(r"^\s*#\s*@CHECK_DRAM\s+(\d+)\s+(0x[0-9a-fA-F]+)\s*$", re.IGNORECASE)

    preload_pairs = []
    check_pairs = []

    for raw_line in source.splitlines():
        m = dram_base_re.match(raw_line)
        if m:
            dram_base = int(m.group(1), 0)
            continue

        m = dram_re.match(raw_line)
        if m:
            beat_off = int(m.group(1), 0)
            beat_addr = dram_base + beat_off * beat_bytes
            raw = _hex_to_words(m.group(2))
            while len(raw) < words_per_beat:
                raw.append(0)
            for i, word in enumerate(raw[:words_per_beat]):
                preload_pairs.append((beat_addr + i * 4, word))
            continue

        m = check_re.match(raw_line)
        if m:
            beat_off = int(m.group(1), 0)
            beat_addr = dram_base + beat_off * beat_bytes
            raw = _hex_to_words(m.group(2))
            while len(raw) < words_per_beat:
                raw.append(0)
            for i, word in enumerate(raw[:words_per_beat]):
                check_pairs.append((beat_addr + i * 4, word))

    if not preload_pairs and not check_pairs:
        return None

    preload_pairs.sort(key=lambda p: p[0])
    check_pairs.sort(key=lambda p: p[0])

    def _split(pairs):
        return ([addr for addr, _ in pairs], [word for _, word in pairs])

    p_addrs, p_words = _split(preload_pairs)
    c_addrs, c_words = _split(check_pairs)
    return dram_base, p_addrs, p_words, c_addrs, c_words


def _parse_dram_base(source):
    """Parse the default # @DRAM_BASE directive from assembly comments."""
    dram_base_re = re.compile(r"^\s*#\s*@DRAM_BASE\s+(\S+)\s*$", re.IGNORECASE)
    for raw_line in source.splitlines():
        m = dram_base_re.match(raw_line)
        if m:
            return int(m.group(1), 0)
    return None


def _fmt_array(words, name, ctype="uint32_t", indent="    "):
    lines = []
    for i in range(0, len(words), 8):
        chunk = words[i:i+8]
        if ctype == "uint64_t":
            vals = ", ".join(f"0x{w & 0xFFFFFFFFFFFFFFFF:016X}ULL" for w in chunk)
        else:
            vals = ", ".join(f"0x{w & 0xFFFFFFFF:08X}U" for w in chunk)
        lines.append(f"{indent}{vals},")
    body = "\n".join(lines)
    return f"static const {ctype} {name}[] = {{\n{body}\n}};\n"


def _parse_perf_threshold(source):
    """Parse `# @PERF_UTIL_THRESHOLD <percent>` from assembly source.

    Returns the threshold as an int (0..100), or None if absent.
    """
    pat = re.compile(r"^\s*#\s*@PERF_UTIL_THRESHOLD\s+(\d+)\s*$", re.IGNORECASE)
    for line in source.splitlines():
        m = pat.match(line)
        if m:
            return int(m.group(1))
    return None


def _parse_perf_report(source):
    """Parse `# @PERF_REPORT` (flag, no argument) from assembly source.

    Returns True if present. Triggers perf-counter printout in the C harness
    (incl. dbg1_cycles / util) without gating the test on utilization.
    """
    pat = re.compile(r"^\s*#\s*@PERF_REPORT\s*$", re.IGNORECASE)
    for line in source.splitlines():
        if pat.match(line):
            return True
    return False


def _parse_expect_illegal(source):
    """True if the source has `# @EXPECT_ILLEGAL`: the test must stop at an illegal instruction."""
    return re.search(r"^\s*#\s*@EXPECT_ILLEGAL\b", source, re.M) is not None


def _parse_perf_macs(source):
    """Parse the `# @PERF_MACS_MXU1 <n>` directive.

    Returns the MXU1 MACs, or None if the directive is absent. Each 32x32x32
    VMATMUL = 32768 MACs. Peak = 1024 MAC/cycle, so
    util_mxu1 = MACs * 100 / (1024 * dbg1_cycles).
    """
    pat = re.compile(r"^\s*#\s*@PERF_MACS_MXU1\s+(\d+)\s*$", re.IGNORECASE)
    macs = None
    for line in source.splitlines():
        m = pat.match(line)
        if m:
            macs = int(m.group(1))
    return macs


def emit_c_file(code, test_name="test", golden=None, perf_threshold=None,
                perf_report=False, perf_macs_mxu1=None, expect_illegal=False):
    """Generate a complete baremetal C test.

    The test fails if the Atlas ISA CSR has another version than the assembler. It polls DBG0 and
    stops when the core halts with a reason (ecall, ebreak or an illegal instruction).

    Args:
        code: list of 32-bit instruction words
        test_name: used in printf messages
        golden: None, or (preload_addrs, preload_words, check_addrs, check_words)
        perf_threshold: if set (0..100), gate pass/fail on the MAC-based util
            of MXU1 declared via @PERF_MACS_MXU1. Test fails if the declared
            util = MACs * 100 / (1024 * dbg1_cycles) is below it.
        perf_report: if True, emit the perf printout (dbg1_cycles + per-MXU
            MAC-based util) WITHOUT gating pass/fail on utilization. Implied
            when perf_threshold is set. Use for tests that bank a timed-window
            cycle delta into CSR_DBG1 (CSRRW ..., 0x801) but rely on golden
            DRAM comparison for pass/fail.
        perf_macs_mxu1: total MACs issued to MXU1 in the timed window, or None.
        expect_illegal: the test passes only if the core stops at an illegal instruction
            (# @EXPECT_ILLEGAL).
    """
    from npu_model.isa import ISA_VERSION
    major, minor = ISA_VERSION
    n = len(code)
    array_body = ""
    for i, word in enumerate(code):
        comma = "," if i < n - 1 else ""
        array_body += f"    0x{word & 0xFFFFFFFF:08X}{comma}\n"

    has_golden = golden is not None and (golden[1] or golden[3])

    # header & helpers & memory map
    out = f"""\
/* Auto-generated from {test_name} by assembler.py */
#include <stdio.h>
#include <stdint.h>

static inline uint32_t mmio_read32(uint64_t addr) {{ return *(volatile uint32_t *)addr; }}
static inline void mmio_write32(uint64_t addr, uint32_t val) {{ *(volatile uint32_t *)addr = val; }}

/* Atlas memory map */
#define ATLAS_IMEM_BASE   0x00020000UL
#define ATLAS_CSR_BASE    0x00040000UL

#define CSR_CYCLE         (ATLAS_CSR_BASE + 0x00)
#define CSR_INSTCNT       (ATLAS_CSR_BASE + 0x04)
#define CSR_STATUS        (ATLAS_CSR_BASE + 0x08)
#define CSR_ILLEGAL_PC    (ATLAS_CSR_BASE + 0x0C)
#define CSR_DBG0          (ATLAS_CSR_BASE + 0x10)
#define CSR_DBG1          (ATLAS_CSR_BASE + 0x14)
#define CSR_EXEC_CONTROL  (ATLAS_CSR_BASE + 0x18)
#define CSR_ISA           (ATLAS_CSR_BASE + 0x1C)

#define ATLAS_ISA_MAJOR   {major}U
#define ATLAS_ISA_MINOR   {minor}U

#define ATLAS_EXEC_STOP   0U
#define ATLAS_EXEC_START  1U

#define ATLAS_PROGRAM_LEN {n}

static const uint32_t atlas_program[ATLAS_PROGRAM_LEN] = {{
{array_body}}};
"""

    # golden model data
    if has_golden:
        p_addrs, p_words, c_addrs, c_words = golden
        if p_words:
            out += f"#define PRELOAD_WORDS {len(p_words)}\n"
            out += _fmt_array(p_addrs, "preload_addrs", ctype="uint64_t")
            out += _fmt_array(p_words, "preload_data")
        if c_words:
            out += f"#define CHECK_WORDS   {len(c_words)}\n"
            out += _fmt_array(c_addrs, "check_addrs", ctype="uint64_t")
            out += _fmt_array(c_words, "check_expected")

    # main()
    out += f"""
int main(void)
{{
    uint32_t i;
    int fail = 0;

    uint32_t isa = mmio_read32(CSR_ISA);
    if ((isa >> 24) != ATLAS_ISA_MAJOR || ((isa >> 16) & 0xFF) != ATLAS_ISA_MINOR) {{
        printf("*** FAILED *** ({test_name}: assembled for ISA %u.%u, Atlas has ISA %u.%u)\\n",
               ATLAS_ISA_MAJOR, ATLAS_ISA_MINOR, isa >> 24, (isa >> 16) & 0xFF);
        return 1;
    }}
"""

    if has_golden and golden[1]:
        out += """
    printf("Writing preload data to DRAM (%u words) ...\\n", PRELOAD_WORDS);
    for (i = 0; i < PRELOAD_WORDS; i++) {
        mmio_write32(preload_addrs[i], preload_data[i]);
    }
    asm volatile ("fence" ::: "memory");
"""

    out += f"""
    printf("Stopping Atlas core before programming IMEM ...\\n");
    mmio_write32(CSR_EXEC_CONTROL, ATLAS_EXEC_STOP);
    mmio_write32(CSR_DBG0, 0);
    mmio_write32(CSR_DBG1, 0);
    asm volatile ("fence" ::: "memory");

    printf("Writing program to ATLAS IMEM (%u words) ...\\n", ATLAS_PROGRAM_LEN);
    for (i = 0; i < ATLAS_PROGRAM_LEN; i++) {{
        mmio_write32(ATLAS_IMEM_BASE + i * 4, atlas_program[i]);
    }}
    asm volatile ("fence" ::: "memory");

    printf("Reading back IMEM and verifying ...\\n");
    for (i = 0; i < ATLAS_PROGRAM_LEN; i++) {{
        uint32_t got = mmio_read32(ATLAS_IMEM_BASE + i * 4);
        if (got != atlas_program[i]) {{
            printf("MISMATCH word[%u]: expected 0x%016x, got 0x%08x\\n",
                   i, atlas_program[i], got);
            fail++;
        }}
    }}

    if (fail) {{
        printf("*** FAILED *** (IMEM mismatch — %d words)\\n", fail);
        return 1;
    }}
    printf("IMEM readback OK\\n");

    printf("Issuing Atlas START command ...\\n");
    mmio_write32(CSR_EXEC_CONTROL, ATLAS_EXEC_START);
    asm volatile ("fence" ::: "memory");

    printf("Waiting for Atlas core to execute ...\\n");
    uint32_t dbg0 = 0;
    for (i = 0; i < 200000000U; i++) {{
        dbg0 = mmio_read32(CSR_DBG0);
        /* status [2:1]: the halt reason, cleared by the start */
        if (dbg0 != 0 || (mmio_read32(CSR_STATUS) >> 1 & 3) != 0) break;
    }}

    printf("Issuing Atlas STOP command ...\\n");
    mmio_write32(CSR_EXEC_CONTROL, ATLAS_EXEC_STOP);
    asm volatile ("fence" ::: "memory");

    dbg0 = mmio_read32(CSR_DBG0);  /* the core may set DBG0 just before it halts */
    uint32_t cycles   = mmio_read32(CSR_CYCLE);
    uint32_t instcnt  = mmio_read32(CSR_INSTCNT);
    uint32_t status   = mmio_read32(CSR_STATUS);

    printf("  DBG0    = %u\\n", dbg0);
    printf("  mcycles = %u\\n", cycles);
    printf("  minstret = %u\\n", instcnt);
    printf("  status  = 0x%08x\\n", status);
"""
    if expect_illegal:
        return out + f"""
    if (dbg0 == 0 && (status >> 1 & 3) == 1) {{
        printf("*** PASSED *** ({test_name}: illegal instruction at pc 0x%x, as expected)\\n",
               mmio_read32(CSR_ILLEGAL_PC));
        return 0;
    }}
    printf("*** FAILED *** ({test_name}: no illegal instruction, dbg0=%u)\\n", dbg0);
    return 1;
}}
"""
    out += """
    if (dbg0 == 0) {
        printf("*** FAILED *** (Atlas core did not complete: timeout, or halted with status 0x%08x)\\n", status);
        return 1;
    }
"""

    if perf_threshold is not None or perf_report:
        # MAC-based utilization. The program banks a timed-window cycle delta
        # into CSR_DBG1 (two CSRR snapshots of mcycles around the measured
        # region). MAC counts are declared statically via @PERF_MACS_MXU1.
        # Peak = 1024 MAC/cycle, so util = MACs * 100 /
        # (1024 * dbg1_cycles).
        out += """
    /* Perf metric. Denominator is the scalar-banked mcycles delta in CSR_DBG1. */
    uint32_t dbg1_cycles  = mmio_read32(CSR_DBG1);
    printf("  dbg1_cycles  = %u\\n", dbg1_cycles);
"""
        if perf_threshold is not None:
            out += f"""
    if (dbg1_cycles == 0) {{
        printf("*** FAILED *** ({test_name} — dbg1_cycles == 0)\\n");
        return 1;
    }}
"""
        if perf_macs_mxu1 is not None:
            out += f"""
    {{
        uint64_t macs = {perf_macs_mxu1}ULL;
        uint32_t util = dbg1_cycles ? (uint32_t)((macs * 100ULL) / (1024ULL * (uint64_t)dbg1_cycles)) : 0u;
        printf("  util_mxu1    = %u%% (MACs=%llu, peak=1024 MAC/cyc)\\n",
               util, (unsigned long long)macs);
"""
            if perf_threshold is not None:
                out += f"""        if (util < {perf_threshold}u) {{
            printf("*** FAILED *** ({test_name} — util_mxu1 %u%% below threshold %u%%)\\n",
                   util, (unsigned){perf_threshold});
            return 1;
        }}
"""
            out += "    }\n"

    if has_golden and golden[3]:
        out += f"""
    printf("Verifying DRAM results (%u words) ...\\n", CHECK_WORDS);
    for (i = 0; i < CHECK_WORDS; i++) {{
        uint32_t got = mmio_read32(check_addrs[i]);
        if (got != check_expected[i]) {{
            printf("  DRAM MISMATCH word[%u] @ 0x%08X%08X: expected 0x%08x, got 0x%08x\\n",
            i,
            (uint32_t)((uint64_t)check_addrs[i] >> 32),
            (uint32_t)((uint64_t)check_addrs[i] & 0xFFFFFFFFU),
            check_expected[i], got);
            fail++;
        }}
    }}

    if (fail) {{
        printf("*** FAILED *** (DRAM mismatch — %d words)\\n", fail);
        return 1;
    }}

    printf("*** PASSED *** ({test_name} — all DRAM checks passed)\\n");
    return 0;
}}
"""
    else:
        out += f"""
    if (dbg0 == 1) {{
        printf("*** PASSED *** ({test_name})\\n");
        return 0;
    }} else {{
        printf("*** FAILED *** ({test_name} dbg0=%u)\\n", dbg0);
        return 1;
    }}
}}
"""

    return out


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Atlas SALU assembler — .S to hex/binary/C"
    )
    parser.add_argument("input", help="Input .S assembly file (with --disasm: .hex or .bin program)")
    parser.add_argument("--out-hex", default=None, help="Output hex file (one 32-bit word per line)")
    parser.add_argument("--out-bin", default=None, help="Output raw binary file (little-endian 32-bit words)")
    parser.add_argument("--out-c", default=None, help="Output C file with full baremetal test")
    parser.add_argument("--golden-json", default=None,
                        help="Golden JSON from gen_*.py (dram_preloads + dram_checks)")
    parser.add_argument("--dram-base", default="0x00000000",
                        help="DRAM base address for golden data (default 0x00000000)")
    parser.add_argument("--beat-bytes", type=int, default=32,
                        help="DMA beat width in bytes (default 32)")
    parser.add_argument("--disasm", action="store_true",
                        help="Print the assembly of the input program (hex text, or raw binary if .bin)")
    args = parser.parse_args()

    if args.disasm:
        print(disassemble_file(args.input), end="")
        return

    with open(args.input, "r") as f:
        source = f.read()

    code = assemble(source)

    if args.out_hex:
        with open(args.out_hex, "w") as f:
            for word in code:
                f.write(f"{word & 0xFFFFFFFF:08x}\n")
        print(f"Wrote {len(code)} words to {args.out_hex}")

    if args.out_bin:
        with open(args.out_bin, "wb") as f:
            for word in code:
                f.write(struct.pack("<I", word & 0xFFFFFFFF))
        print(f"Wrote {len(code)} words ({len(code) * 4} bytes) to {args.out_bin}")

    if args.out_c:
        test_name = os.path.splitext(os.path.basename(args.input))[0]
        golden = None
        if args.golden_json:
            dram_base = _parse_dram_base(source)
            if dram_base is None:
                dram_base = int(args.dram_base, 0)
            golden = _parse_golden_json(args.golden_json, dram_base, args.beat_bytes)
            n_pre = len(golden[1])
            n_chk = len(golden[3])
            print(
                f"Golden data @DRAM_BASE 0x{dram_base:016X}: "
                f"{n_pre} preload words, {n_chk} check words"
            )
        else:
            inline = _parse_inline_golden(source, args.beat_bytes)
            if inline is not None:
                inline_dram_base, p_addrs, p_words, c_addrs, c_words = inline
                golden = (p_addrs, p_words, c_addrs, c_words)
                print(
                    f"Inline golden data @DRAM_BASE 0x{inline_dram_base:016X}: "
                    f"{len(p_words)} preload words, {len(c_words)} check words"
                )
        perf_threshold = _parse_perf_threshold(source)
        if perf_threshold is not None:
            print(f"Perf utilization threshold: {perf_threshold}%")
        perf_report = _parse_perf_report(source)
        if perf_report and perf_threshold is None:
            print("Perf reporting enabled (no gating)")
        perf_macs_mxu1 = _parse_perf_macs(source)
        if perf_macs_mxu1 is not None:
            print(f"Perf MACs MXU1: {perf_macs_mxu1}")
        c_source = emit_c_file(code, test_name, golden=golden,
                               perf_threshold=perf_threshold,
                               perf_report=perf_report,
                               perf_macs_mxu1=perf_macs_mxu1,
                               expect_illegal=_parse_expect_illegal(source))
        with open(args.out_c, "w") as f:
            f.write(c_source)
        print(f"Wrote C file to {args.out_c} ({len(code)} instructions)")

    if not args.out_hex and not args.out_bin and not args.out_c:
        print(f"Assembled {len(code)} instructions from {args.input}")
        print()
        for i, word in enumerate(code):
            print(f"  [{i:4d}]  0x{word & 0xFFFFFFFF:08x}")


if __name__ == "__main__":
    main()
    
