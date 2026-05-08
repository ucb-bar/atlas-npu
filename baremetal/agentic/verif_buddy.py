from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

AGENTIC_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = AGENTIC_DIR / "resources"
BAREMETAL_DIR = AGENTIC_DIR.parent
ASSEMBLY_DIR = BAREMETAL_DIR / "assembly"
GENERATORS_DIR = BAREMETAL_DIR / "generators"
TESTS_DIR = BAREMETAL_DIR / "tests"
TESTS_BUILD_DIR = TESTS_DIR / "build"
ASSEMBLER_PATH = BAREMETAL_DIR / "assembler.py"
ISA_DOCS_PATH = RESOURCES_DIR / "isa_docs.md"
BUILT_DIR = AGENTIC_DIR / ".built"
OUTPUT_DIR = BAREMETAL_DIR / "baremetal_gen"
USED_PROMPTS_DIRNAME = "used_prompts"

# Chipyard RTL sim (sims/vcs make run-binary)
VCS_DIR_RELPATH = "sims/vcs"
DEFAULT_SIM_CONFIG = "EE290SimConfig"
DEFAULT_COVERAGE_THRESHOLD = 90.0
DEFAULT_LLM_TIMEOUT_S = 30 * 60
COVERAGE_DIRNAME = "coverage"
COVERAGE_ANALYZER_PATH = BAREMETAL_DIR / "coverage_analyzer.py"
RUN_ASM_TESTS_PATH = AGENTIC_DIR / "run_asm_tests.sh"
GENERATED_SRC_DIRNAME = "generated-src"
SIM_FAIL_MARKERS = (
    "*** FAILED ***",
    "Assertion failed",
    "FAIL:",
    "DRAM MISMATCH",
    "Error:",
    "Fatal:",
)


def find_chipyard_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` until we find a Chipyard repository root."""

    cur = (start or BAREMETAL_DIR).resolve()
    while True:
        if (
            (cur / "sims").is_dir()
            and (cur / "generators").is_dir()
            and (cur / "tests").is_dir()
        ):
            return cur
        if cur.parent == cur:
            raise FileNotFoundError(
                "Could not locate Chipyard root (need sims/, generators/, tests/). "
                f"Started from {start or BAREMETAL_DIR}. Set --chipyard-root explicitly."
            )
        cur = cur.parent


# ---------------------------------------------------------------------------
# Built config loading helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuiltAgentAssets:
    """Parsed contents of a built agent config directory."""

    config_dir: Path
    architecture: str
    isa_docs_raw: str
    isa_sections: dict[str, str]
    isa_subsections: dict[str, dict[str, str]]
    generator_helper_reference: str
    optimization_menu: dict[str, object]
    rules: dict[str, object]
    baremetal_test_examples: dict[str, dict[str, str]]


def _resolve_resource_path(config_dir: Path, filename: str) -> Path:
    shared_path = RESOURCES_DIR / filename
    if shared_path.exists():
        return shared_path
    return config_dir / filename


def _load_text(config_dir: Path, filename: str) -> str:
    path = _resolve_resource_path(config_dir, filename)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _load_json(config_dir: Path, filename: str) -> dict:
    path = _resolve_resource_path(config_dir, filename)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optimization_menu(config_dir: Path) -> dict[str, object]:
    data = _load_json(config_dir, "optimization_menu.json")
    if not isinstance(data, dict):
        return {}
    return data


def _load_rules(config_dir: Path) -> dict[str, object]:
    data = _load_json(config_dir, "rules.json")
    if not isinstance(data, dict):
        return {}
    return data


def _load_baremetal_test_examples(config_dir: Path) -> dict[str, dict[str, str]]:
    data = _load_json(config_dir, "baremetal_test_examples.json")
    if not isinstance(data, dict):
        return {}
    return {
        str(name): value
        for name, value in data.items()
        if isinstance(value, dict)
    }


def _extract_top_level_function_block(text: str, func_name: str) -> str:
    pattern = re.compile(rf"^def {re.escape(func_name)}\(", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return ""

    lines = text[match.start():].splitlines()
    block: list[str] = []
    for idx, line in enumerate(lines):
        if idx > 0 and line.startswith("def "):
            break
        if idx > 0 and line and not line.startswith((" ", "\t")) and not line.startswith("#"):
            break
        block.append(line)
    return "\n".join(block).rstrip()


def _extract_function_signature(text: str, func_name: str) -> str:
    """Return the ``def`` signature (which may span multiple lines) and its
    docstring, if any. Handles both ``def f(x): ...`` and the more common
    multi-line ``def f(\n    x: T,\n) -> U:`` style by balancing parentheses
    and stopping at the line that closes the signature with ``:``.
    """

    lines = text.split("\n")
    # Match at file scope OR at any indentation; we accept both top-level
    # functions and methods inside classes.
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith(f"def {func_name}("):
            continue

        sig_lines = [line]
        depth = line.count("(") - line.count(")")
        end_idx = idx
        signature_complete = (
            depth == 0 and sig_lines[-1].rstrip().endswith(":")
        )
        while not signature_complete and end_idx + 1 < len(lines):
            end_idx += 1
            nxt = lines[end_idx]
            sig_lines.append(nxt)
            depth += nxt.count("(") - nxt.count(")")
            signature_complete = (
                depth == 0 and nxt.rstrip().endswith(":")
            )
        sig = "\n".join(sig_lines)

        # Capture the docstring (single- or multi-line) that may follow.
        doc_lines: list[str] = []
        rest = lines[end_idx + 1:]
        if rest and rest[0].strip().startswith(('"""', "'''")):
            quote = '"""' if rest[0].strip().startswith('"""') else "'''"
            first = rest[0]
            doc_lines.append(first)
            if first.strip().count(quote) < 2:
                for sub in rest[1:]:
                    doc_lines.append(sub)
                    if quote in sub:
                        break
        body = "\n".join(doc_lines).rstrip()
        return (sig + "\n" + body).rstrip() if body else sig

    return ""


def _extract_class_signature(text: str, class_name: str) -> str:
    """Return the class line plus the signatures (and docstrings) of
    ``__init__`` and ``__call__`` for a single class. Used to surface
    the software-model adapters (SARTLLinearFunction / IPTLinearRTLFunction)
    without dumping their full implementation bodies into the prompt.
    """

    class_pattern = re.compile(rf"^class {re.escape(class_name)}\b.*?:", re.MULTILINE)
    m = class_pattern.search(text)
    if not m:
        return ""

    tail = text[m.start():]
    lines = tail.splitlines()
    class_line = lines[0]

    body_lines: list[str] = []
    saw_class_docstring = False
    for idx, line in enumerate(lines[1:], start=1):
        stripped = line.strip()
        if idx == 1 and stripped.startswith(('"""', "'''")):
            saw_class_docstring = True
            quote = '"""' if stripped.startswith('"""') else "'''"
            body_lines.append(line)
            if stripped.count(quote) >= 2 and len(stripped) > 3:
                continue
            for sub in lines[idx + 1:]:
                body_lines.append(sub)
                if quote in sub:
                    break
            break
        if not saw_class_docstring:
            break

    sig_methods = ("__init__", "__call__")
    method_blocks: list[str] = []
    for method in sig_methods:
        pat = re.compile(rf"^\s+def {re.escape(method)}\(.*?\)[^:]*:", re.MULTILINE | re.DOTALL)
        found = pat.search(tail)
        if not found:
            continue
        sig = found.group(0).rstrip()
        rest = tail[found.end():]
        rest_lines = rest.splitlines()
        doc: list[str] = []
        if rest_lines and rest_lines[0].strip().startswith(('"""', "'''")):
            quote = '"""' if '"""' in rest_lines[0] else "'''"
            first = rest_lines[0]
            doc.append(first)
            if first.strip().count(quote) < 2:
                for sub in rest_lines[1:]:
                    doc.append(sub)
                    if quote in sub:
                        break
        method_text = sig + ("\n" + "\n".join(doc) if doc else "")
        method_blocks.append(method_text)

    class_body = "\n".join(body_lines).rstrip()
    methods_text = "\n\n".join(method_blocks).rstrip()
    parts = [class_line]
    if class_body:
        parts.append(class_body)
    if methods_text:
        parts.append(methods_text)
    return "\n".join(parts).rstrip()


def _extract_enum_summary(text: str, enum_name: str) -> str:
    """Return a one-block summary of an ``enum.Enum`` subclass (the class line
    and its member assignments). Useful for OutputFmtSel / AddendSel.
    """

    pattern = re.compile(rf"^class {re.escape(enum_name)}\b.*?:\s*\n", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return ""
    lines = text[m.start():].splitlines()
    block = [lines[0]]
    for line in lines[1:]:
        if line and not line.startswith((" ", "\t")) and not line.startswith("#"):
            break
        if not line.strip():
            if len(block) > 1:
                break
            continue
        if "=" in line or line.strip().startswith(('"""', "'''")):
            block.append(line)
    return "\n".join(block).rstrip()


def _load_gen_utils_reference(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    snippets: list[str] = []

    const_lines = []
    for name in ("DMA_WIDTH_BYTES", "WORDS_PER_BEAT"):
        match = re.search(rf"^{name}\s*=.*$", text, re.MULTILINE)
        if match:
            const_lines.append(match.group(0))
    if const_lines:
        snippets.append("\n".join(const_lines))

    full_body_funcs = (
        "pack_words_into_beats",
        "emit_test_data",
        "make_preload_any",
        "make_check_any",
        "preloads_from_words_packed",
        "checks_from_words_packed",
    )
    for func_name in full_body_funcs:
        block = _extract_top_level_function_block(text, func_name)
        if block:
            snippets.append(block)

    signature_only_funcs = (
        "float_to_bf16_bits",
        "bf16_bits_to_float",
        "pack_bf16_pair",
        "unpack_bf16_pair",
        "float_to_fp8_e4m3_bits",
        "fp8_e4m3_bits_to_float",
        "pack_fp8x4",
        "float_to_fp8_e5m2_bits",
        "matrix_to_bf16_words",
        "matrix_to_fp8_words",
        "matrix_to_int32_words",
        "quantize_bf16",
        "quantize_fp8",
        "bf16_matmul_reference",
        "fp8_matmul_reference",
        "rand_matrix",
        "rand_matrix_fp8_safe",
        "identity_matrix",
        "zero_matrix",
    )
    signature_snippets: list[str] = []
    for func_name in signature_only_funcs:
        sig = _extract_function_signature(text, func_name)
        if sig:
            signature_snippets.append(sig)
    if signature_snippets:
        snippets.append(
            "# High-level datatype and reference-model helpers (signatures only):\n"
            + "\n\n".join(signature_snippets)
        )

    return "\n\n".join(snippets).strip()


def _load_vpu_gen_utils_reference(path: Path) -> str:
    """Surface the VPU generator helpers. These wrap the VPU functional model
    (`VectorEngineModel`) and hand back DRAM preloads/checks in the exact
    beat layout the assembly expects; generators should reuse them instead
    of hand-encoding BF16 lanes or walking `MODEL.execute` directly.
    """

    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")

    const_lines: list[str] = []
    const_names = (
        "PARAMS",
        "MODEL",
        "BF16_PER_BEAT",
        "FP8_PER_BEAT",
        "ROWS_PER_REGISTER",
        "ROWS_PER_TENSOR",
    )
    for name in const_names:
        match = re.search(rf"^{name}\s*=.*$", text, re.MULTILINE)
        if match:
            const_lines.append(match.group(0))

    snippets: list[str] = []
    if const_lines:
        snippets.append("# Module constants (re-export these rather than redefining):\n"
                        + "\n".join(const_lines))

    full_body_funcs = (
        "float_to_bf16",
        "pack_u16_le",
        "pack_u8_le",
        "const_bf16_row",
        "constant_bf16_rows",
        "repeat_bf16_row",
        "tensor_preloads",
        "tensor_checks",
        "fp8_checks",
    )
    for func_name in full_body_funcs:
        block = _extract_top_level_function_block(text, func_name)
        if block:
            snippets.append(block)

    signature_only_funcs = (
        "run_unary_rows",
        "run_binary_rows",
        "run_row_reduce_tensor",
        "run_col_reduce_tensor",
        "run_fp8_pack_rows",
        "run_fp8_unpack_rows",
        "run_vli_rows",
        "run_vli_registers",
    )
    sigs: list[str] = []
    for func_name in signature_only_funcs:
        sig = _extract_function_signature(text, func_name)
        if sig:
            sigs.append(sig)
    if sigs:
        snippets.append(
            "# VPU functional-model wrappers (signatures only — the bodies call\n"
            "# MODEL.execute(op, ...) under the hood; every op name that the\n"
            "# assembly uses via VPU.* instructions is accepted here):\n"
            + "\n\n".join(sigs)
        )

    snippets.append(textwrap.dedent(
        """\
        # The wrappers above route through `MODEL.execute(op, ...)` where `op`
        # is one of (see software_models.vpu.vector_engine_model):
        #   pointwise binary : "add", "sub", "mul", "pairmax", "pairmin"
        #   pointwise unary  : "rcp", "sqrt", "sin", "cos", "tanh", "log",
        #                      "exp", "exp2", "square", "cube", "relu", "mov"
        #   row reductions   : "rsum", "rmax", "rmin"
        #   col reductions   : "csum", "cmax", "cmin"
        #   VLI ops          : "vliAll", "vliRow", "vliCol", "vliOne"
        #   FP8 phased       : "fp8pack", "fp8unpack"
        # Use these exact strings; they match the assembly VPU.<op> naming.
        """
    ).strip())

    return "\n\n".join(snippets).strip()


def _load_software_models_reference() -> str:
    """Signature-only surface for the software RTL models used by the checked-in
    MXU and attention generators. These are the source of truth for MXU0/MXU1
    golden numerics; generators MUST call them instead of re-implementing
    FP8 matmul, E4M3 quantization, or BF16 accumulation.
    """

    models_dir = BAREMETAL_DIR / "generators" / "software_models"
    if not models_dir.is_dir():
        return ""

    sa_path = models_dir / "mxu0_sa" / "systolic_array_rtl_linear.py"
    ipt_path = models_dir / "mxu1_ipt" / "ipt_rtl_linear.py"
    fp_fmt_path = models_dir / "mxu1_ipt" / "fp_formats.py"
    conv_path = models_dir / "mxu1_ipt" / "converters.py"
    vpu_eng_path = models_dir / "vpu" / "vector_engine_model.py"

    blocks: list[str] = []

    if sa_path.exists():
        sa_text = sa_path.read_text(encoding="utf-8", errors="replace")
        sa_sig = _extract_class_signature(sa_text, "SARTLLinearFunction")
        if sa_sig:
            blocks.append(
                "# --- software_models.mxu0_sa.systolic_array_rtl_linear ---\n"
                "# MXU0 systolic-array functional adapter. Computes\n"
                "#     y = x @ w^T + b\n"
                "# with FP32 accumulation over K-tiles and final BF16/E4M3 truncation.\n"
                "# For C = A @ B + bias, pass w = B.T so y = A @ B.\n"
                "# Accepts torch tensors (float or uint8 raw E4M3 bytes) and returns\n"
                "# a torch tensor shaped like x with last dim replaced by out_features.\n"
                + sa_sig
            )

    if ipt_path.exists():
        ipt_text = ipt_path.read_text(encoding="utf-8", errors="replace")
        ipt_sig = _extract_class_signature(ipt_text, "IPTLinearRTLFunction")
        if ipt_sig:
            blocks.append(
                "# --- software_models.mxu1_ipt.ipt_rtl_linear ---\n"
                "# MXU1 inner-product-tree functional adapter. Same y = x @ w^T + b\n"
                "# contract as SARTLLinearFunction but modeled bit-for-bit against\n"
                "# the IPT lane RTL (E4M3 inputs, BF16 or E4M3 output format).\n"
                + ipt_sig
            )

    if fp_fmt_path.exists():
        fmt_text = fp_fmt_path.read_text(encoding="utf-8", errors="replace")
        enum_blocks: list[str] = []
        for enum_name in ("OutputFmtSel", "AddendSel"):
            summary = _extract_enum_summary(fmt_text, enum_name)
            if summary:
                enum_blocks.append(summary)
        fmt_fn_blocks: list[str] = []
        for fn in (
            "f32_to_bf16_bits_rne",
            "bf16_bits_to_f32",
            "encode_e4m3_normal",
            "decode_e4m3",
            "sanitize_bf16",
            "round_right_shift4_rne",
        ):
            sig = _extract_function_signature(fmt_text, fn)
            if sig:
                fmt_fn_blocks.append(sig)
        if enum_blocks or fmt_fn_blocks:
            blocks.append(
                "# --- software_models.mxu1_ipt.fp_formats ---\n"
                "# Canonical FP-format helpers and the enums MXU adapters expect.\n"
                "# Use these enums as `out_fmt_sel=OutputFmtSel.OutBF16` (or OutE4M3).\n"
                + ("\n\n".join(enum_blocks + fmt_fn_blocks))
            )

    if conv_path.exists():
        conv_text = conv_path.read_text(encoding="utf-8", errors="replace")
        conv_fn_blocks: list[str] = []
        for fn in (
            "bf16_scale_to_e4m3",
            "output_conv_stage",
            "e4m3_mul_to_prod",
            "e4m3_prod_to_aligned_int",
            "aligned_int_to_bf16",
            "pack_e4m3_prod",
        ):
            sig = _extract_function_signature(conv_text, fn)
            if sig:
                conv_fn_blocks.append(sig)
        if conv_fn_blocks:
            blocks.append(
                "# --- software_models.mxu1_ipt.converters ---\n"
                "# BF16 <-> E4M3 / product-format conversions that mirror the\n"
                "# VMATPUSH.ACC / VMATPOP quantization semantics of the hardware.\n"
                "# For a BF16-valued tile that hardware will round-trip through FP8,\n"
                "# `bf16_scale_to_e4m3(bits, scale_exp)` is the bit-exact mirror.\n"
                + "\n\n".join(conv_fn_blocks)
            )

    if vpu_eng_path.exists():
        vpu_text = vpu_eng_path.read_text(encoding="utf-8", errors="replace")
        execute_sig = _extract_function_signature(vpu_text, "execute")
        if execute_sig:
            blocks.append(
                "# --- software_models.vpu.vector_engine_model.VectorEngineModel ---\n"
                "# The VPU functional model shared by every `gen_vpu_*.py` via\n"
                "# `vpu_gen_utils.MODEL`. Prefer the `vpu_gen_utils` wrappers; fall\n"
                "# back to `MODEL.execute(...)` only when you need bespoke per-row\n"
                "# dispatch. Signature and op names are authoritative:\n"
                + execute_sig
            )

    return "\n\n".join(blocks).strip()


def _generator_canonical_imports_block() -> str:
    """Canonical import boilerplate covering all three helper layers."""

    return textwrap.dedent(
        """\
        # Canonical imports for any Atlas baremetal generator. Pick the subset you
        # need, but prefer these spellings — they match every checked-in gen_*.py
        # and keep `sys.path` consistent:
        import os
        import sys
        sys.path.insert(0, os.path.dirname(__file__))

        # Packing / emit + BF16 / FP8 encoders + matmul references:
        from gen_utils import (
            emit_test_data,
            preloads_from_words_packed,
            checks_from_words_packed,
            pack_words_into_beats,
            make_preload_any,
            make_check_any,
            float_to_bf16_bits,
            bf16_bits_to_float,
            pack_bf16_pair,
            float_to_fp8_e4m3_bits,
            fp8_e4m3_bits_to_float,
            pack_fp8x4,
            matrix_to_bf16_words,
            matrix_to_fp8_words,
            quantize_bf16,
            quantize_fp8,
            bf16_matmul_reference,
            fp8_matmul_reference,
            rand_matrix,
            rand_matrix_fp8_safe,
        )

        # VPU functional-model wrappers (use for ANY test whose oracle depends on
        # a VPU op — unary math, pair binary, row/col reductions, VLI, FP8 pack):
        from vpu_gen_utils import (
            BF16_PER_BEAT,
            FP8_PER_BEAT,
            ROWS_PER_REGISTER,
            ROWS_PER_TENSOR,
            float_to_bf16,
            pack_u16_le,
            pack_u8_le,
            const_bf16_row,
            constant_bf16_rows,
            repeat_bf16_row,
            tensor_preloads,
            tensor_checks,
            fp8_checks,
            run_unary_rows,
            run_binary_rows,
            run_row_reduce_tensor,
            run_col_reduce_tensor,
            run_fp8_pack_rows,
            run_fp8_unpack_rows,
            run_vli_rows,
            run_vli_registers,
        )

        # Software RTL models for MXU goldens (use for any test that pushes a
        # matmul tile through MXU0 / MXU1 — these are bit-exact against the RTL):
        from software_models.mxu0_sa.systolic_array_rtl_linear import SARTLLinearFunction
        from software_models.mxu1_ipt.ipt_rtl_linear import IPTLinearRTLFunction
        from software_models.mxu1_ipt.fp_formats import OutputFmtSel, AddendSel
        from software_models.mxu1_ipt.converters import (
            bf16_scale_to_e4m3,
            output_conv_stage,
        )
        """
    ).rstrip()


def _generator_examples_block() -> str:
    """Three end-to-end example skeletons covering the common oracle paths:
    MXU+packing via gen_utils, VPU-only via vpu_gen_utils, and a full
    MXU0+VPU integration using the software RTL model.
    """

    mxu_only = textwrap.dedent(
        """\
        # Example 1: BF16 / FP8 MXU tile test (matches gen_mxu0_single_output_tile_bf16).
        # DMA→VLOAD→MXU→VSTORE→DMA; uses gen_utils helpers only — no VPU op in the chain.
        #
        # DRAM offsets in the assembly are BYTE offsets from @DRAM_BASE.
        # Generators emit `word_offset` values in DMA-width BEATS (1 beat = 32 B).
        # So: beat_offset = byte_offset // 32. Never conflate the two.
        import numpy as np
        import torch
        TILE = 32
        A = rand_matrix_fp8_safe(TILE, TILE, seed=42)
        W = rand_matrix_fp8_safe(TILE, TILE, seed=43)
        A_q = quantize_fp8(A).astype(np.float32)
        W_q = quantize_fp8(W).astype(np.float32)
        # Prefer the software RTL model over fp8_matmul_reference when the
        # hardware path uses FP8 inputs with a BF16 accumulator / output,
        # because SARTLLinearFunction matches the exact MXU0 K-chain rounding:
        sa = SARTLLinearFunction(rows=32, cols=32, out_fmt_sel=OutputFmtSel.OutBF16)
        C  = sa(torch.from_numpy(A_q), torch.from_numpy(W_q.T.copy())).numpy()
        # Preload FP8 tiles at DRAM byte offsets 0x0000 and 0x0400:
        preloads  = preloads_from_words_packed(0x0000 // 32, matrix_to_fp8_words(A_q))
        preloads += preloads_from_words_packed(0x0400 // 32, matrix_to_fp8_words(W_q))
        # Check BF16 result halves if the assembly stores 32x32 as two 32x16 banks:
        checks  = checks_from_words_packed(0x0800 // 32, matrix_to_bf16_words(C[:, :16]))
        checks += checks_from_words_packed(0x0C00 // 32, matrix_to_bf16_words(C[:, 16:]))
        emit_test_data(preloads, checks, timeout=500000)
        """
    ).rstrip()

    vpu_only = textwrap.dedent(
        """\
        # Example 2: VPU-only test (matches gen_vpu_unary_math / gen_vpu_binary).
        # The assembly VLOADs a 32-row BF16 tensor into a bank pair, issues one
        # or more VPU.<op> instructions, then VSTOREs each result tensor to
        # DRAM. Use `tensor_preloads` / `tensor_checks` so every beat_offset is
        # already aligned to BF16_PER_BEAT lanes per row.
        import json
        BEATS = ROWS_PER_TENSOR                  # 32 rows = one BF16 tensor pair
        TIMEOUT = 60000
        INPUT_ROWS = constant_bf16_rows(BEATS, float_to_bf16(1.0))
        OPS = [
            ("exp", 64), ("tanh", 128), ("sqrt", 192),   # (vpu op name, output beat base)
        ]
        preloads = tensor_preloads(0, INPUT_ROWS)
        checks: list[dict] = []
        for op, base in OPS:
            checks.extend(tensor_checks(base, run_unary_rows(op, INPUT_ROWS)))
        print(json.dumps({
            "dram_preloads": preloads,
            "dram_checks":   checks,
            "timeout":       TIMEOUT,
        }, indent=2))
        """
    ).rstrip()

    integ_example = textwrap.dedent(
        """\
        # Example 3: MXU0 + VPU integration test (matches gen_mxu_chain_vpu_reduce
        # and gen_smolvla_fused_*). The assembly stages one or more MXU tiles,
        # pops a BF16 result tensor, then feeds it to the VPU for reduce /
        # pointwise / pack. The golden chains SARTLLinearFunction for the MXU
        # stage and `run_*_rows` / `MODEL.execute` for every VPU op.
        import numpy as np
        import torch
        TILE = 32
        A = rand_matrix_fp8_safe(TILE, TILE, seed=42)
        W = rand_matrix_fp8_safe(TILE, TILE, seed=43)
        A_q = quantize_fp8(A).astype(np.float32)
        W_q = quantize_fp8(W).astype(np.float32)
        sa = SARTLLinearFunction(rows=32, cols=32, out_fmt_sel=OutputFmtSel.OutBF16)
        C_bf16 = sa(torch.from_numpy(A_q), torch.from_numpy(W_q.T.copy())).numpy()
        # Treat the BF16 result as a 32-row tensor of BF16 bits for VPU consumption.
        from gen_utils import float_to_bf16_bits  # already imported above
        mxu_bits = np.vectorize(float_to_bf16_bits)(C_bf16).astype(np.uint16)
        mxu_rows = [list(mxu_bits[r]) for r in range(TILE)]      # 32 rows of 32 lanes
        # Run the VPU stage through the functional model:
        relu_rows = run_unary_rows("relu", mxu_rows[:ROWS_PER_REGISTER])
        row_max   = run_row_reduce_tensor("rmax", mxu_rows)      # operates on a bank pair
        # Emit preloads for A/W and checks for each stage:
        preloads  = preloads_from_words_packed(0x0000 // 32, matrix_to_fp8_words(A_q))
        preloads += preloads_from_words_packed(0x0400 // 32, matrix_to_fp8_words(W_q))
        checks  = tensor_checks(0x0800 // 32, relu_rows)
        checks += tensor_checks(0x0C00 // 32, row_max)
        emit_test_data(preloads, checks, timeout=500000)
        """
    ).rstrip()

    return "\n\n".join([mxu_only, vpu_only, integ_example])


def _load_generator_helper_reference() -> str:
    gen_utils_ref = _load_gen_utils_reference(BAREMETAL_DIR / "generators" / "gen_utils.py")
    vpu_gen_utils_ref = _load_vpu_gen_utils_reference(
        BAREMETAL_DIR / "generators" / "vpu_gen_utils.py"
    )
    sw_models_ref = _load_software_models_reference()

    canonical_imports = _generator_canonical_imports_block()
    examples = _generator_examples_block()

    if not any([gen_utils_ref, vpu_gen_utils_ref, sw_models_ref]):
        return ""

    intro = textwrap.dedent(
        """\
        Use the checked-in generator helper layers below instead of inventing helper
        signatures, encoding tables, matmul references, or ad-hoc JSON layouts.
        Generators have three import-level sources of truth, and you MUST prefer the
        highest-level helper that solves the problem:

        1. `generators/gen_utils.py` — DRAM preload/check emission, BF16 / FP8 bit
           encoders, matrix packing, RNG, and `bf16_matmul_reference` /
           `fp8_matmul_reference` (use the matmul references only for loose sanity
           checks — the MXU software RTL model below is the authoritative golden).
        2. `generators/vpu_gen_utils.py` — VPU functional-model wrappers around
           `software_models.vpu.vector_engine_model.VectorEngineModel`. Use these
           for any oracle that depends on a `VPU.<op>` in the assembly — unary,
           binary, row/col reduce, VLI, fp8pack/fp8unpack. Do NOT hand-simulate
           exp / sqrt / tanh / relu / rcp; they go through the model which is the
           bit-exact mirror of the RTL lane boxes.
        3. `generators/software_models/` — bit-exact RTL models of the two MXUs and
           the VPU:
             - `software_models.mxu0_sa.systolic_array_rtl_linear.SARTLLinearFunction`
               — MXU0 golden for BF16 / FP8 matmul with optional bias.
             - `software_models.mxu1_ipt.ipt_rtl_linear.IPTLinearRTLFunction` —
               MXU1 (inner-product-tree) golden, same contract.
             - `software_models.mxu1_ipt.fp_formats` — canonical enums
               (`OutputFmtSel`, `AddendSel`) and BF16/E4M3 rounding primitives.
             - `software_models.mxu1_ipt.converters` — `bf16_scale_to_e4m3` and
               `output_conv_stage` model the VMATPUSH.ACC / VMATPOP quantization.
             - `software_models.vpu.vector_engine_model.VectorEngineModel` —
               the underlying VPU model; `vpu_gen_utils.MODEL` is a shared instance.

        Do NOT re-implement float-to-bf16 rounding, FP8 bit-packing, FP8 matmul
        accumulation, E4M3 quantization, VPU lane boxes, row/col tile shifting, or
        matmul reference numerics by hand — the helpers below already match the
        Atlas RTL behavior.
        """
    ).strip()

    sections: list[str] = [intro]

    if gen_utils_ref:
        sections.append(
            "### `generators/gen_utils.py`\n\n"
            f"```python\n{gen_utils_ref}\n```"
        )
    if vpu_gen_utils_ref:
        sections.append(
            "### `generators/vpu_gen_utils.py`\n\n"
            f"```python\n{vpu_gen_utils_ref}\n```"
        )
    if sw_models_ref:
        sections.append(
            "### `generators/software_models/` (signatures only)\n\n"
            f"```python\n{sw_models_ref}\n```"
        )

    sections.append(
        "### Canonical imports\n\n"
        f"```python\n{canonical_imports}\n```"
    )
    sections.append(
        "### End-to-end generator skeletons\n\n"
        f"```python\n{examples}\n```"
    )

    return "\n\n".join(sections).strip()


def _parse_isa_sections(isa_text: str) -> dict[str, str]:
    """Parse ISA docs markdown into sections keyed by ``##`` header."""

    sections: dict[str, str] = {}
    current_header = "General"
    current_lines: list[str] = []

    for line in isa_text.split("\n"):
        if line.startswith("## "):
            if current_lines:
                sections[current_header] = "\n".join(current_lines).strip()
            current_header = line[3:].strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_header] = "\n".join(current_lines).strip()

    return sections


def _parse_isa_subsections(sections: dict[str, str]) -> dict[str, dict[str, str]]:
    """Parse ``###`` subsections within each ``##`` section."""

    result: dict[str, dict[str, str]] = {}
    for sec_name, sec_text in sections.items():
        subs: dict[str, str] = {}
        cur_sub: str | None = None
        cur_lines: list[str] = []
        preamble_lines: list[str] = []

        for line in sec_text.split("\n"):
            if line.startswith("### "):
                if cur_sub is not None:
                    block = "\n".join(cur_lines).strip()
                    if cur_sub in subs:
                        subs[cur_sub] += "\n\n" + block
                    else:
                        subs[cur_sub] = block
                cur_sub = line[4:].strip()
                cur_lines = [line]
            elif cur_sub is not None:
                cur_lines.append(line)
            else:
                preamble_lines.append(line)

        if cur_sub is not None:
            block = "\n".join(cur_lines).strip()
            if cur_sub in subs:
                subs[cur_sub] += "\n\n" + block
            else:
                subs[cur_sub] = block

        if preamble_lines:
            preamble = "\n".join(preamble_lines).strip()
            if preamble:
                subs["_preamble"] = preamble

        result[sec_name] = subs

    return result


def resolve_built_config_dir(agent_name_or_dir: str | Path = "tapeout") -> Path:
    """
    Resolve either a built agent name like ``tapeout`` or an explicit directory.
    """

    candidate = Path(agent_name_or_dir)
    if candidate.is_absolute() and candidate.is_dir():
        return candidate

    search_candidates: list[Path] = []
    if candidate.is_dir():
        search_candidates.append(candidate.resolve())
    if not candidate.is_absolute():
        search_candidates.append((AGENTIC_DIR / candidate).resolve())
        search_candidates.append((BUILT_DIR / candidate).resolve())

    for path in search_candidates:
        if path.is_dir():
            return path

    raise FileNotFoundError(
        f"Built agent config directory not found for: {agent_name_or_dir}"
    )


def load_built_agent_assets(agent_name_or_dir: str | Path = "tapeout") -> BuiltAgentAssets:
    """Load and parse a built agent config directory into reusable variables."""

    config_dir = resolve_built_config_dir(agent_name_or_dir)
    architecture = _load_text(config_dir, "architecture.md")
    isa_docs_raw = _load_text(config_dir, "isa_docs.md")
    isa_sections = _parse_isa_sections(isa_docs_raw)
    isa_subsections = _parse_isa_subsections(isa_sections)
    generator_helper_reference = _load_generator_helper_reference()
    optimization_menu = _load_optimization_menu(config_dir)
    rules = _load_rules(config_dir)
    baremetal_test_examples = _load_baremetal_test_examples(config_dir)

    return BuiltAgentAssets(
        config_dir=config_dir,
        architecture=architecture,
        isa_docs_raw=isa_docs_raw,
        isa_sections=isa_sections,
        isa_subsections=isa_subsections,
        generator_helper_reference=generator_helper_reference,
        optimization_menu=optimization_menu,
        rules=rules,
        baremetal_test_examples=baremetal_test_examples,
    )

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


_DEFAULT_SHARED_ASSETS: BuiltAgentAssets | None = None


def _get_default_shared_assets() -> BuiltAgentAssets:
    global _DEFAULT_SHARED_ASSETS
    if _DEFAULT_SHARED_ASSETS is None:
        _DEFAULT_SHARED_ASSETS = load_built_agent_assets(AGENTIC_DIR)
    return _DEFAULT_SHARED_ASSETS


def _format_bullet_list(items: list[str]) -> str:
    if not items:
        return "- None provided."
    return "\n".join(f"- {item}" for item in items)


def _format_rule_block(rule_block: object) -> str:
    if not isinstance(rule_block, dict):
        return "No additional rules provided."

    sections: list[str] = []

    goal = rule_block.get("goal")
    if isinstance(goal, str) and goal.strip():
        sections.append("### Goal\n" + goal.strip())

    for title, key in [
        ("Instructions", "instructions"),
        ("Output Requirements", "output_requirements"),
        ("Guardrails", "guardrails"),
    ]:
        value = rule_block.get(key)
        if isinstance(value, list):
            items = [str(item) for item in value if str(item).strip()]
            if items:
                sections.append(f"### {title}\n{_format_bullet_list(items)}")

    return "\n\n".join(sections) if sections else "No additional rules provided."


def format_assembly_syntax_section(rules: dict[str, object] | None = None) -> str:
    """Markdown block for planner user prompts (body lives in ``rules.json``)."""

    resolved = rules if rules is not None else _get_default_shared_assets().rules
    if not isinstance(resolved, dict):
        return ""
    body = resolved.get("assembly_syntax_and_conventions")
    if not isinstance(body, str) or not body.strip():
        return ""
    return f"## Assembly Syntax and Conventions\n\n{body.strip()}\n"


def format_preload_oracle_contract_section(rules: dict[str, object] | None = None) -> str:
    """Markdown block for planner prompts describing DRAM preload/oracle rules."""

    resolved = rules if rules is not None else _get_default_shared_assets().rules
    if not isinstance(resolved, dict):
        return ""
    body = resolved.get("preload_oracle_contract")
    if not isinstance(body, str) or not body.strip():
        return ""
    return f"## Preload And Oracle Contract\n\n{body.strip()}\n"


def format_correctness_invariants_section(
    rules: dict[str, object] | None = None,
) -> str:
    """Render the unified test-correctness invariants as a compact markdown block.

    Each invariant in `rules.json["test_correctness_invariants"]` is a `{name, rule, why}`
    dict. The rendered section is deliberately terse — these are the hard rules every
    generated test must uphold, not a storytelling log of past bugs.
    """

    resolved = rules if rules is not None else _get_default_shared_assets().rules
    if not isinstance(resolved, dict):
        return ""
    body = resolved.get("test_correctness_invariants")
    if not isinstance(body, list) or not body:
        return ""

    lines: list[str] = [
        "## Test Correctness Invariants",
        "",
        (
            "Every generated test MUST uphold all of the following invariants. "
            "Each is a hard rule derived from real silent failures: if any is violated "
            "the harness will still run and still report a mismatch, but you will have "
            "wasted a full CI cycle. Re-check every invariant against your assembly and "
            "generator before emitting the test."
        ),
        "",
    ]
    for idx, item in enumerate(body, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", f"Invariant {idx}")).strip()
        rule = str(item.get("rule", "")).strip()
        why = str(item.get("why", "")).strip()
        lines.append(f"{idx}. **{name}.** {rule}")
        if why:
            lines.append(f"   _Why it matters:_ {why}")
    return "\n".join(lines) + "\n"



def _format_strategy_menu(optimization_menu: object) -> str:
    if not isinstance(optimization_menu, dict):
        return "No strategy menu provided."

    sections: list[str] = []

    objective = optimization_menu.get("objective")
    if isinstance(objective, str) and objective.strip():
        sections.append(objective.strip())

    categories = optimization_menu.get("strategy_categories")
    if isinstance(categories, list):
        for category in categories:
            if not isinstance(category, dict):
                continue
            name = str(category.get("name", "Unnamed Strategy Category")).strip()
            description = str(category.get("description", "")).strip()
            strategies = category.get("strategies")

            body: list[str] = []
            if description:
                body.append(description)
            if isinstance(strategies, list):
                items = [str(item) for item in strategies if str(item).strip()]
                if items:
                    body.append(_format_bullet_list(items))

            if body:
                sections.append(f"### {name}\n" + "\n\n".join(body))
            else:
                sections.append(f"### {name}")

    return "\n\n".join(sections) if sections else "No strategy menu provided."


def _format_test_suite_context(
    assembly_tests: dict[str, str] | None,
    generators: dict[str, str] | None,
    header_lines: int = 12,
) -> str:
    assembly_tests = assembly_tests or {}
    generators = generators or {}

    if not assembly_tests and not generators:
        return "No current tests or generators were provided."

    sections: list[str] = []

    if assembly_tests:
        test_list = "\n".join(f"  - {name}" for name in sorted(assembly_tests))
        summaries = []
        for name in sorted(assembly_tests):
            preview = "\n".join(assembly_tests[name].splitlines()[:header_lines])
            summaries.append(f"### {name}\n```\n{preview}\n```")
        sections.append(
            "The following assembly tests exist:\n"
            f"{test_list}\n\n"
            f"### Test Headers (first {header_lines} lines of each):\n\n"
            + "\n\n".join(summaries)
        )

    if generators:
        gen_list = "\n".join(f"  - {name}" for name in sorted(generators))
        sections.append(
            "The following Python generators exist to produce DRAM preloads/checks:\n"
            f"{gen_list}"
        )

    return "\n\n".join(sections)


def _format_example_section(
    baremetal_test_examples: dict[str, dict[str, str]] | None,
) -> str:
    examples = baremetal_test_examples or {}
    regular = examples.get("regular", {})
    with_golden = examples.get("with_golden", {})

    regular_description = regular.get("description", "(no description available)")
    regular_assembly = regular.get("assembly", "")
    golden_description = with_golden.get("description", "(no description available)")
    golden_assembly = with_golden.get("assembly", "")
    golden_generator = with_golden.get("golden_generator", "")

    return textwrap.dedent(
        f"""\
## Example: Assembly-Only Test

{regular_description}

```asm
{regular_assembly}
```

## Example: Assembly + Golden Generator Test

{golden_description}

Assembly:
```asm
{golden_assembly}
```

Generator:
```python
{golden_generator}
```"""
    )


def format_generator_helper_section(generator_helper_reference: str | None = None) -> str:
    resolved = (
        generator_helper_reference
        if generator_helper_reference is not None
        else _get_default_shared_assets().generator_helper_reference
    )
    if not isinstance(resolved, str) or not resolved.strip():
        return ""
    return (
        "## Generator Helper API "
        "(`gen_utils`, `vpu_gen_utils`, `software_models`)\n\n"
        f"{resolved.strip()}\n"
    )


def _truncate_block_lines(text: str, max_lines: int, note_label: str) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text.strip()
    omitted = len(lines) - max_lines
    kept = lines[:max_lines]
    kept.append(f"... ({omitted} additional {note_label} lines omitted for prompt budget)")
    return "\n".join(kept).strip()


def condense_coverage_report_for_prompt(
    report: str,
    *,
    max_module_summary_rows: int = 80,
    max_detail_modules: int = 12,
    max_detail_lines_per_module: int = 18,
    max_total_chars: int = 120_000,
) -> str:
    """Shrink the analyzer-facing coverage report to a prompt-safe summary.

    The full `coverage_analysis.txt` is useful as an artifact, but its detailed
    per-module gap dump can reach millions of characters and blow the LLM context
    window. For prompting, keep the high-level summary, the worst-scoring module
    rows, and a bounded slice of detailed module gaps.
    """

    text = report.strip()
    if not text:
        return text

    module_summary_marker = "MODULE SUMMARY (sorted by score, *** = below threshold)"
    detail_marker = "DETAILED GAPS FOR LOW-COVERAGE MODULES"
    end_marker = "END OF COVERAGE ANALYSIS"

    summary_start = text.find(module_summary_marker)
    detail_start = text.find(detail_marker)
    end_start = text.find(end_marker)

    if summary_start == -1:
        clipped = _truncate_block_lines(text, 500, "coverage report")
        return clipped[:max_total_chars].rstrip()

    preamble = text[:summary_start].strip()
    if detail_start != -1:
        module_summary = text[summary_start:detail_start].strip()
        detail_section = text[detail_start:end_start if end_start != -1 else None].strip()
    else:
        module_summary = text[summary_start:end_start if end_start != -1 else None].strip()
        detail_section = ""

    summary_lines = module_summary.splitlines()
    if len(summary_lines) > 3:
        header = summary_lines[:3]
        rows = summary_lines[3:]
        omitted = max(0, len(rows) - max_module_summary_rows)
        kept_rows = rows[:max_module_summary_rows]
        if omitted:
            kept_rows.append(
                f"... ({omitted} additional module summary rows omitted for prompt budget)"
            )
        module_summary = "\n".join(header + kept_rows).strip()

    detail_blocks: list[str] = []
    if detail_section:
        detail_lines = detail_section.splitlines()
        detail_header: list[str] = []
        blocks: list[list[str]] = []
        current: list[str] | None = None
        for line in detail_lines:
            if line.startswith("MODULE: "):
                if current:
                    blocks.append(current)
                current = [line]
                continue
            if current is None:
                detail_header.append(line)
            else:
                current.append(line)
        if current:
            blocks.append(current)

        for block in blocks[:max_detail_modules]:
            block_text = "\n".join(block).strip()
            detail_blocks.append(
                _truncate_block_lines(
                    block_text,
                    max_detail_lines_per_module,
                    "module-detail",
                )
            )

        detail_parts = []
        if detail_header:
            detail_parts.append("\n".join(detail_header).strip())
        detail_parts.extend(detail_blocks)
        if len(blocks) > max_detail_modules:
            detail_parts.append(
                f"... ({len(blocks) - max_detail_modules} additional low-coverage module "
                "blocks omitted for prompt budget)"
            )
        detail_section = "\n\n".join(part for part in detail_parts if part).strip()

    parts = [
        "## Prompt-Condensed Measured RTL Coverage Report",
        "The original coverage analysis artifact was condensed before prompting "
        "to keep the LLM input within budget while preserving the most actionable gaps.",
        preamble,
        module_summary,
    ]
    if detail_section:
        parts.append(detail_section)
    if end_start != -1:
        parts.append("=" * 70)
        parts.append(end_marker)

    condensed = "\n\n".join(part for part in parts if part).strip()
    if len(condensed) > max_total_chars:
        condensed = condensed[: max_total_chars - 1].rstrip() + "\n... (coverage report truncated to max prompt budget)"
    return condensed


# ---------------------------------------------------------------------------
# Per-slot coverage gating helpers
# ---------------------------------------------------------------------------

COVERAGE_METRIC_KEYS: tuple[str, ...] = (
    "score",
    "line",
    "cond",
    "toggle",
    "fsm",
    "branch",
)

# Metrics smaller than this absolute delta are treated as noise (URG itself
# only prints two decimals; a sub-0.005% wiggle is round-trip noise, not real
# coverage). Anything above the epsilon is a real, observable improvement.
COVERAGE_IMPROVEMENT_EPS: float = 1e-3


_COVERAGE_SUMMARY_BLOCKS: tuple[str, ...] = (
    "ATLAS TILE SUBTREE SUMMARY",
    "OVERALL COVERAGE SUMMARY",
)


def _empty_coverage_metrics() -> dict[str, float | None]:
    return {key: None for key in COVERAGE_METRIC_KEYS}


def parse_coverage_summary_metrics(report_text: str) -> dict[str, float | None]:
    """Extract the top-level SCORE/LINE/COND/TOGGLE/FSM/BRANCH metrics from
    a `coverage_analyzer.py` text report.

    The analyzer emits either an "ATLAS TILE SUBTREE SUMMARY" block (when the
    target Atlas tile instance is found in the URG report) or an "OVERALL
    COVERAGE SUMMARY" block (the legacy fall-back). We try each in turn and
    return whichever one populates first; a key with no measurable percentage
    in the chosen block is left as `None`.
    """

    metrics = _empty_coverage_metrics()
    if not report_text:
        return metrics

    for marker in _COVERAGE_SUMMARY_BLOCKS:
        idx = report_text.find(marker)
        if idx == -1:
            continue
        # Each summary block is at most ~12 short lines; cap the window so we
        # never accidentally pull metrics from the per-instance table below.
        block = report_text[idx : idx + 1200]
        candidate = _empty_coverage_metrics()
        for key, label in (
            ("score", "SCORE"),
            ("line", "LINE"),
            ("cond", "COND"),
            ("toggle", "TOGGLE"),
            ("fsm", "FSM"),
            ("branch", "BRANCH"),
        ):
            m = re.search(rf"^\s*{label}\s*:\s*(-?\d+(?:\.\d+)?)\s*%", block, re.MULTILINE)
            if m:
                try:
                    candidate[key] = float(m.group(1))
                except ValueError:
                    candidate[key] = None
        if any(v is not None for v in candidate.values()):
            return candidate

    return metrics


def coverage_metrics_improved(
    baseline: dict[str, float | None] | None,
    candidate: dict[str, float | None] | None,
    *,
    eps: float = COVERAGE_IMPROVEMENT_EPS,
) -> dict[str, object]:
    """Return whether `candidate` strictly improves `baseline` on any metric.

    URG aggregation is monotonic: merging a new `.vdb` into the union can
    only hold or increase each metric. So an "improvement" here is any
    metric whose value strictly exceeds the baseline by more than `eps`.
    A candidate that ties the baseline on every metric contributed nothing
    and should be rejected.

    The result is a structured dict so callers can both gate (boolean) and
    log (per-metric deltas + improved-key set).
    """

    baseline = baseline or _empty_coverage_metrics()
    candidate = candidate or _empty_coverage_metrics()

    deltas: dict[str, float | None] = {}
    improved_keys: list[str] = []
    for key in COVERAGE_METRIC_KEYS:
        b = baseline.get(key)
        c = candidate.get(key)
        if b is None and c is None:
            deltas[key] = None
            continue
        if b is None:
            delta = c if c is not None else 0.0
        elif c is None:
            delta = -b
        else:
            delta = c - b
        deltas[key] = delta
        if delta is not None and delta > eps:
            improved_keys.append(key)

    # If neither side has any populated metric, we cannot make a meaningful
    # decision — treat that as "not improved" so the caller can fall back on
    # whatever permissive policy it wants. The caller knows the full context.
    any_signal = any(
        baseline.get(k) is not None or candidate.get(k) is not None
        for k in COVERAGE_METRIC_KEYS
    )

    return {
        "improved": bool(improved_keys),
        "improved_keys": improved_keys,
        "deltas": deltas,
        "had_signal": any_signal,
    }


def format_coverage_delta_for_log(
    baseline: dict[str, float | None] | None,
    candidate: dict[str, float | None] | None,
    deltas: dict[str, float | None] | None = None,
) -> str:
    """Render a fixed-width before -> after (Δ) table for one coverage delta.

    Used both in CLI logs and inside the per-slot JSON record so a human can
    eyeball exactly which metric a test moved (and by how much) when they are
    debugging why a test was accepted or rejected by the coverage gate.
    """

    baseline = baseline or _empty_coverage_metrics()
    candidate = candidate or _empty_coverage_metrics()
    if deltas is None:
        deltas = coverage_metrics_improved(baseline, candidate)["deltas"]

    fmt_pct = lambda v: f"{v:6.2f}%" if isinstance(v, (int, float)) else "    --"
    fmt_dlt = lambda v: f"{v:+.4f}" if isinstance(v, (int, float)) else "  --"

    lines: list[str] = []
    for key, label in (
        ("score", "SCORE"),
        ("line", "LINE"),
        ("cond", "COND"),
        ("toggle", "TOGGLE"),
        ("fsm", "FSM"),
        ("branch", "BRANCH"),
    ):
        b = baseline.get(key)
        c = candidate.get(key)
        d = deltas.get(key) if isinstance(deltas, dict) else None
        lines.append(
            f"  {label:<6}: {fmt_pct(b)} -> {fmt_pct(c)}  (Δ {fmt_dlt(d)})"
        )
    return "\n".join(lines)


def build_analyzer_system_prompt(
    rules: dict[str, object] | None = None,
) -> str:
    if rules is None:
        rules = _get_default_shared_assets().rules

    analysis_rules = {}
    if isinstance(rules, dict):
        candidate = rules.get("analysis")
        if isinstance(candidate, dict):
            analysis_rules = candidate

    return textwrap.dedent(f"""\
You are an expert hardware verification engineer analyzing the Atlas baremetal verification test suite.

Your role is to study the provided artifacts and produce a rigorous coverage
analysis that downstream coworkers can act on directly.

Treat the supplied architecture summary, ISA reference, current tests, and generators as the
source of truth. Be concrete, precise, and skeptical of shallow coverage claims.

## Analysis Rules

{_format_rule_block(analysis_rules)}

Return structured Markdown only.
""")


def build_analyzer_user_prompt(
    architecture: str,
    isa_docs: str,
    assembly_tests: dict[str, str],
    generators: dict[str, str],
    measured_coverage_report: str = "",
) -> str:
    coverage_section = (
        "## Measured RTL Coverage Report\n\n"
        + measured_coverage_report.strip()
        + "\n"
        if measured_coverage_report.strip()
        else "## Measured RTL Coverage Report\n\n"
        "No measured RTL coverage report was provided.\n"
    )
    return textwrap.dedent(f"""\
Analyze the current Atlas baremetal test suite using the references and artifacts below.

## Hardware Architecture Summary

{architecture}

## Atlas ISA Reference

{isa_docs}

## Current Test Suite

{_format_test_suite_context(assembly_tests, generators, header_lines=15)}

{coverage_section}
## Your Task

Produce a thorough **coverage analysis** of the existing test suite. Use the measured RTL
coverage report to distinguish theoretical gaps from empirically under-covered logic. Format
the output as structured Markdown and make it directly actionable for a downstream
planner/implementer.
""")


def build_analyzer_prompt(
    isa_docs: str,
    assembly_tests: dict[str, str],
    generators: dict[str, str],
    measured_coverage_report: str = "",
    rules: dict[str, object] | None = None,
    architecture: str | None = None,
) -> str:
    if architecture is None:
        architecture = _get_default_shared_assets().architecture

    system_prompt = build_analyzer_system_prompt(rules=rules)
    user_prompt = build_analyzer_user_prompt(
        architecture=architecture,
        isa_docs=isa_docs,
        assembly_tests=assembly_tests,
        generators=generators,
        measured_coverage_report=measured_coverage_report,
    )
    return f"# System Prompt\n\n{system_prompt}\n\n# User Prompt\n\n{user_prompt}"


def build_planner_system_prompt(
    rules: dict[str, object] | None = None,
) -> str:
    defaults = _get_default_shared_assets()
    rules = rules or defaults.rules

    planning_rules = {}
    if isinstance(rules, dict):
        candidate = rules.get("planning")
        if isinstance(candidate, dict):
            planning_rules = candidate

    output_section = textwrap.dedent(
        """\
You are an expert hardware verification engineer writing baremetal integration tests
for the Atlas chip.

Every artifact you emit will be assembled, have its generator executed, and be compared
byte-for-byte against the generator's `dram_checks`. There is no fuzzy matching: a single
wrong `word_offset` or `expected` value fails the test silently (often with an "all zeros"
run) and wastes the full downstream CI cycle. Correctness is table stakes; only correct
tests contribute coverage.

## Core Contracts

1. **Uphold every Test Correctness Invariant.** The invariants in the user prompt are
   non-negotiable. Each one is the direct root cause of a past silent failure and covers
   a general class of bug — dataflow closure, unit discipline, MREG pair discipline,
   delay-slot semantics, scalar LSU isolation, value-domain golden compute via
   `gen_utils`, and DMA serialization. Before you emit a test, mentally check each
   invariant against it.
2. **Helper reuse is mandatory.** The generator helpers live in three layers —
   `generators/gen_utils.py` (DRAM emit + BF16/FP8 encoders + matmul references),
   `generators/vpu_gen_utils.py` (VPU functional-model wrappers: `run_unary_rows`,
   `run_binary_rows`, `run_row_reduce_tensor`, `run_col_reduce_tensor`,
   `run_fp8_pack_rows`, `run_vli_rows`, `tensor_preloads`/`tensor_checks`), and
   `generators/software_models/` (the bit-exact MXU0 `SARTLLinearFunction`, MXU1
   `IPTLinearRTLFunction`, VPU `VectorEngineModel`, and `bf16_scale_to_e4m3` /
   `OutputFmtSel` / `AddendSel`). They are the single source of truth for every
   numerical model in the chip. If you find yourself writing a custom
   `bf16_word(v)` lookup, a `fp8_byte(v)` dict, a `(v << 8)` conversion, an
   ad-hoc matmul loop, or a hand-rolled `exp` / `tanh` / `rsum` oracle, stop and
   use the matching helper instead.
3. **Units are distinct.** DMA-beats (32 B), bytes, VMEM words, and VLOAD/VSTORE `imm`
   steps (32 words = 128 B) are four different quantities. Generator `word_offset` is a
   beat index. VMEM word addresses are 4 × byte addresses. Scalar LSU uses VMEM bytes.
   Conflating them is the single most common source of wrong-offset mismatches.
4. **Correctness first, then coverage.** A test that fails silently contributes zero
   coverage. Cover more ground by writing fewer, sharper, demonstrably-correct tests
   rather than many speculative ones.

## Output Format

Return a JSON array where each element is an object with these fields:
- `"name"`: test name without extension (e.g. `"integ_dma_vpu_mxu_pipeline"`)
- `"description"`: 2-3 sentence description of what this test exercises
- `"assembly"`: the complete `.S` file contents as a string
- `"has_generator"`: boolean — true if this test needs a golden-value generator
- `"generator"`: if `has_generator` is true, the complete `gen_<name>.py` contents as a string; otherwise `null`
"""
    )

    return textwrap.dedent(f"""\
{output_section}

## Planner Rules

{_format_rule_block(planning_rules)}

Return ONLY the JSON array, no other text before or after it.
""")


def _format_session_accepted_section(
    accepted_generated_tests: list[dict] | None,
) -> str:
    """Render a compact summary of tests that have been accepted during this
    incremental generation session. The planner should read this to avoid
    duplicating its own siblings and to deliberately complement them.
    """

    items = accepted_generated_tests or []
    if not items:
        return ""

    lines = [
        "## Tests Already Accepted This Session",
        "",
        (
            "These tests were generated earlier in the current incremental session and are "
            "already part of the suite above. Do NOT duplicate their names, memory maps, or "
            "coverage intent — the new test should complement them, probe a different "
            "subsystem combination, or stress a different scheduling/hazard axis."
        ),
        "",
    ]
    for entry in items:
        name = str(entry.get("name", "")).strip() or "(unnamed)"
        desc = str(entry.get("description", "")).strip()
        desc = " ".join(desc.split())  # collapse whitespace/newlines
        if len(desc) > 240:
            desc = desc[:237].rstrip() + "..."
        lines.append(f"- **{name}** — {desc}" if desc else f"- **{name}**")
    return "\n".join(lines) + "\n\n"


def build_planner_user_prompt(
    analysis: str,
    architecture: str,
    isa_docs: str,
    num_tests: int,
    assembly_tests: dict[str, str] | None = None,
    generators: dict[str, str] | None = None,
    generator_helper_reference: str | None = None,
    optimization_menu: dict[str, object] | None = None,
    baremetal_test_examples: dict[str, dict[str, str]] | None = None,
    rules: dict[str, object] | None = None,
    accepted_generated_tests: list[dict] | None = None,
    forbidden_names: list[str] | set[str] | None = None,
) -> str:
    defaults = _get_default_shared_assets()
    generator_helper_reference = (
        generator_helper_reference or defaults.generator_helper_reference
    )
    optimization_menu = optimization_menu or defaults.optimization_menu
    baremetal_test_examples = baremetal_test_examples or defaults.baremetal_test_examples
    current_suite_section = (
        "## Current Test Suite\n\n"
        + _format_test_suite_context(assembly_tests, generators, header_lines=15)
        + "\n\n"
    )
    session_section = _format_session_accepted_section(accepted_generated_tests)

    forbidden = sorted({str(n).strip() for n in (forbidden_names or []) if str(n).strip()})
    if forbidden:
        forbidden_note = (
            "The following test names are already used by the suite and MUST NOT be reused. "
            "Pick a distinct, descriptive name for the new test:\n"
            + "\n".join(f"  - `{n}`" for n in forbidden)
            + "\n\n"
        )
    else:
        forbidden_note = ""

    if num_tests == 1:
        header_line = (
            "You are an expert hardware verification engineer writing baremetal "
            "integration tests for the Atlas chip. Generate exactly **one** new test "
            "that meaningfully advances the coverage profile of the existing suite, "
            "including any tests already accepted earlier in this session."
        )
    else:
        header_line = (
            "You are an expert hardware verification engineer writing baremetal "
            "integration tests for the Atlas chip. Generate exactly "
            f"**{num_tests}** new tests that meaningfully advance the coverage "
            "profile of the existing suite."
        )

    task_section = textwrap.dedent(
        f"""\
        {header_line}

        Each test must satisfy all four of the following:

        1. **Coverage intent.** The test exercises at least three Atlas subsystems in one
           coherent end-to-end story (data movement + compute + control or validation),
           and targets a gap from the coverage analysis — not a cosmetic variant of an
           existing test.
        2. **All Test Correctness Invariants hold.** Every rule in the invariants section
           above is upheld by construction. These are hard rules, not advisories: a single
           violation produces a silent mismatch and a wasted CI run.
        3. **Iteration-0 sanity.** For any loop or subroutine, trace iteration 0 on paper:
           the values of every pointer/counter register at the first `DMA.LOAD`, first
           `VLOAD`, first `VSTORE`, and first `DMA.STORE` must match the memory map that
           the generator preloads and checks assume.
        4. **Oracle discipline.** Pick one oracle path per test and commit to it:
           - assembly self-check with small inline `# @DRAM`/`# @CHECK_DRAM` blocks when
             the expected output is compact and deterministic,
           - a generator when the outputs are tensor-scale, randomized, or derived from
             multi-stage compute. Never mix a `DMA.LOAD` with an unspecified preload source.

        {forbidden_note}Write each test the way a senior verification engineer would sign it off for
        regression: a clear memory-map comment at the top, conservative scheduling unless
        the test is explicitly probing overlap, descriptive names, and a generator whose
        preloads and checks can be cross-read against the assembly line by line.
        """
    ).strip()

    return textwrap.dedent(f"""\
Use the materials below to produce the planner output.

## Coverage Analysis

{analysis}

## Hardware Architecture Summary

{architecture}

## Atlas ISA Reference

{isa_docs}

{format_assembly_syntax_section(rules)}
{format_preload_oracle_contract_section(rules)}
{format_correctness_invariants_section(rules)}
{format_generator_helper_section(generator_helper_reference)}
{current_suite_section}{session_section}\
## Strategy Menu

{_format_strategy_menu(optimization_menu)}

## Reference Examples

{_format_example_section(baremetal_test_examples)}

## Your Task

{task_section}
""")


def _trim_log_tail(text: str, max_chars: int = 8000) -> str:
    """Keep the last ``max_chars`` of a potentially-huge log, prepended with a
    small elided marker. This is what we feed back to the fix-prompt so it
    stays informative but bounded.
    """

    if text is None:
        return ""
    text = text.rstrip()
    if len(text) <= max_chars:
        return text
    return (
        f"... [log truncated, showing final {max_chars} of {len(text)} chars] ...\n"
        + text[-max_chars:]
    )


def build_fix_user_prompt(
    *,
    test_name: str,
    description: str,
    assembly: str,
    generator: str | None,
    failure_phase: str,
    failure_reason: str,
    failure_log_tail: str,
    attempt_idx: int,
    architecture: str,
    isa_docs: str,
    generator_helper_reference: str | None = None,
    rules: dict[str, object] | None = None,
) -> str:
    """Build the fix-me prompt for a single failed test attempt.

    The LLM must return a single-element JSON array using the same schema as
    the planner, with the same ``name`` so we can overwrite in place. Every
    Test Correctness Invariant still applies.
    """

    defaults = _get_default_shared_assets()
    generator_helper_reference = (
        generator_helper_reference or defaults.generator_helper_reference
    )

    if generator is not None and generator.strip():
        generator_block = (
            "## Failing Generator\n\n"
            "```python\n"
            f"{generator.rstrip()}\n"
            "```\n\n"
        )
    else:
        generator_block = (
            "## Failing Generator\n\n"
            "_(this test did not use a generator; oracle is inline `# @CHECK_DRAM`)_\n\n"
        )

    task_section = textwrap.dedent(
        f"""\
        A previous attempt at the test `{test_name}` failed in phase **{failure_phase}**
        with reason `{failure_reason}`. This is retry attempt #{attempt_idx} — do not
        guess; diagnose the root cause from the log tail below.

        Return a single-element JSON array with the corrected test, using the same
        schema as the planner output:

        ```json
        [
          {{
            "name": "{test_name}",
            "description": "...",
            "assembly": "...",
            "has_generator": true/false,
            "generator": "..."  // omit / null if has_generator is false
          }}
        ]
        ```

        Rules for the fix:

        1. **Keep the test name `{test_name}`.** We overwrite in place; do not invent a
           new name.
        2. **Fix the real bug, do not paper over it.** If the symptom is
           `DRAM MISMATCH ... got 0x00000000`, a consumer read uninitialized VMEM — find
           the missing `DMA.LOAD` / `VSTORE` / scalar init. If the symptom is
           `0x7e` saturation, a scale/exponent register was read before being loaded.
           If the assembler/generator crashed, re-read the failure log verbatim.
        3. **Every Test Correctness Invariant above still applies.** Trace iteration 0
           of every loop, check MREG pair usage, check delay-slot semantics, check
           scalar LSU isolation, check that generator preloads/checks match the
           assembly's memory map byte-for-byte.
        4. **You may change the memory map, operand selection, or oracle path** if that
           is what correctness requires — but keep the coverage intent (the high-level
           subsystem combination) similar to the original description so we stay on the
           coverage target this test was picked to hit.
        5. **Output format is non-negotiable:** one JSON array with exactly one object,
           followed by nothing else. No commentary outside the array.
        """
    ).strip()

    return textwrap.dedent(f"""\
A single-test fix iteration.

## Hardware Architecture Summary

{architecture}

## Atlas ISA Reference

{isa_docs}

{format_assembly_syntax_section(rules)}
{format_preload_oracle_contract_section(rules)}
{format_correctness_invariants_section(rules)}
{format_generator_helper_section(generator_helper_reference)}
## Failing Test

- **Name:** `{test_name}`
- **Description:** {description}
- **Failure phase:** `{failure_phase}`
- **Failure reason:** `{failure_reason}`

## Failing Assembly

```asm
{assembly.rstrip()}
```

{generator_block}\
## Failure Log (tail)

```
{failure_log_tail.rstrip()}
```

## Your Task

{task_section}
""")


def build_planner_prompt(
    analysis: str,
    isa_docs: str,
    num_tests: int,
    assembly_tests: dict[str, str] | None = None,
    generators: dict[str, str] | None = None,
    generator_helper_reference: str | None = None,
    optimization_menu: dict[str, object] | None = None,
    rules: dict[str, object] | None = None,
    baremetal_test_examples: dict[str, dict[str, str]] | None = None,
    architecture: str | None = None,
) -> str:
    if architecture is None:
        architecture = _get_default_shared_assets().architecture

    system_prompt = build_planner_system_prompt(rules=rules)
    user_prompt = build_planner_user_prompt(
        analysis=analysis,
        architecture=architecture,
        isa_docs=isa_docs,
        num_tests=num_tests,
        assembly_tests=assembly_tests,
        generators=generators,
        generator_helper_reference=generator_helper_reference,
        optimization_menu=optimization_menu,
        baremetal_test_examples=baremetal_test_examples,
        rules=rules,
    )
    return f"# System Prompt\n\n{system_prompt}\n\n# User Prompt\n\n{user_prompt}"


# ---------------------------------------------------------------------------
# Prompt dumping helpers
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def create_run_output_dir(base_dir: Path, run_label: str) -> Path:
    """
    Create a per-run output directory named ``{run_label}_{datetime}``.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir / f"{run_label}_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"{run_label}_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def collect_assembly_tests() -> dict[str, str]:
    """Return {filename: contents} for every .S file in assembly/."""

    tests: dict[str, str] = {}
    if not ASSEMBLY_DIR.is_dir():
        return tests

    for path in sorted(ASSEMBLY_DIR.glob("*.S")):
        tests[path.name] = _read_text(path)
    return tests


def collect_generators() -> dict[str, str]:
    """Return {filename: contents} for every top-level gen_*.py in generators/."""

    generators: dict[str, str] = {}
    if not GENERATORS_DIR.is_dir():
        return generators

    for path in sorted(GENERATORS_DIR.glob("gen_*.py")):
        generators[path.name] = _read_text(path)
    return generators


def load_isa_docs() -> str:
    if not ISA_DOCS_PATH.exists():
        raise FileNotFoundError(f"ISA docs not found at {ISA_DOCS_PATH}")
    return _read_text(ISA_DOCS_PATH)


def load_analysis_text(path: Path | None = None) -> str:
    analysis_path = path or (OUTPUT_DIR / "analysis.md")
    if analysis_path.exists():
        return _read_text(analysis_path)
    return textwrap.dedent(
        """\
        # Coverage Analysis Placeholder

        No saved analysis file was found. Replace this text with the analyzer output before
        using the planner prompts with an LLM.
        """
    )


def _extract_responses_api_text(resp: object) -> str:
    """Best-effort extraction of text from the Responses API payload."""

    output_text = getattr(resp, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text

    chunks: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
    return "".join(chunks)


def call_llm(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    timeout_s: int = DEFAULT_LLM_TIMEOUT_S,
) -> str:
    responses_exc: Exception | None = None

    if hasattr(client, "responses"):
        try:
            resp = client.responses.create(
                model=model,
                instructions=system,
                input=user,
                timeout=timeout_s,
            )
            text = _extract_responses_api_text(resp)
            if text:
                return text
        except Exception as exc:
            responses_exc = exc

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            ,
            timeout=timeout_s,
        )
        return resp.choices[0].message.content or ""
    except Exception:
        if responses_exc is not None:
            raise responses_exc
        raise


def extract_json_array(raw: str) -> list[dict]:
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not json_match:
        raise ValueError("Could not find a JSON array in the planner response.")

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Planner response did not contain valid JSON: {exc}") from exc

    if not isinstance(parsed, list):
        raise ValueError("Planner response JSON was not a list.")

    return parsed


def extract_single_test_from_json(
    raw: str,
    *,
    expected_name: str | None = None,
) -> dict:
    """Parse the LLM response and return exactly one test dict.

    Enforces schema keys we actually use downstream (``name``, ``assembly``,
    ``has_generator``, optional ``generator``). If ``expected_name`` is given
    we also verify the LLM didn't silently rename the test.
    """

    parsed = extract_json_array(raw)
    if len(parsed) == 0:
        raise ValueError("LLM returned an empty JSON array; expected exactly one test.")
    if len(parsed) > 1:
        raise ValueError(
            f"LLM returned {len(parsed)} tests; expected exactly one. "
            "Keeping the first and discarding the rest is not allowed — "
            "the per-test loop must stay in lockstep."
        )
    test = parsed[0]
    if not isinstance(test, dict):
        raise ValueError("LLM returned a non-object element in the JSON array.")

    name = _validate_generated_test_name(str(test.get("name", "")))
    assembly = str(test.get("assembly", ""))
    if not assembly.strip():
        raise ValueError(f"LLM returned empty assembly for test {name!r}.")
    has_generator = bool(test.get("has_generator", False))
    if has_generator:
        generator = test.get("generator")
        if not isinstance(generator, str) or not generator.strip():
            raise ValueError(
                f"LLM marked test {name!r} as has_generator=true but provided no generator."
            )

    if expected_name is not None and name != expected_name:
        raise ValueError(
            f"LLM fix response renamed the test from {expected_name!r} to {name!r}. "
            "Fix prompts must preserve the test name."
        )

    test["name"] = name
    return test


def unpublish_single_test(name: str) -> list[Path]:
    """Remove all on-disk artifacts for a previously-published generated test.

    This is used before every publish attempt (so retries can overwrite) and
    after a test exhausts its fix-retry budget (to avoid polluting the suite).
    Returns the list of paths that were actually deleted, for logging.
    """

    removed: list[Path] = []
    candidates = [
        ASSEMBLY_DIR / f"{name}.S",
        GENERATORS_DIR / f"gen_{name}.py",
        GENERATORS_DIR / f"{name}.json",
        TESTS_DIR / f"atlas_{name}.c",
    ]
    for path in candidates:
        if path.exists():
            try:
                path.unlink()
                removed.append(path)
            except OSError:
                # Best effort; leaving the file behind will surface as a
                # downstream error on the next publish attempt.
                pass

    # Build artifacts live under tests/build/ and are keyed off the target
    # name (e.g. CMakeFiles/atlas_<name>.dir/). Sweep common ones.
    if TESTS_BUILD_DIR.exists():
        for relpath in (
            TESTS_BUILD_DIR / f"atlas_{name}.riscv",
            TESTS_BUILD_DIR / f"atlas_{name}.dump",
        ):
            if relpath.exists():
                try:
                    relpath.unlink()
                    removed.append(relpath)
                except OSError:
                    pass
        cmake_obj_dir = TESTS_BUILD_DIR / "CMakeFiles" / f"atlas_{name}.dir"
        if cmake_obj_dir.exists():
            try:
                shutil.rmtree(cmake_obj_dir)
                removed.append(cmake_obj_dir)
            except OSError:
                pass
    return removed


def write_generation_logs(
    output_dir: Path,
    analysis: str,
    planner_raw: str,
    planner_results: list[dict],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "analysis": output_dir / "analysis.md",
        "planner_generate_raw": output_dir / "planner_generate_raw_response.md",
        "planner_generate_results": output_dir / "planner_generate_results.json",
    }
    outputs["analysis"].write_text(analysis, encoding="utf-8")
    outputs["planner_generate_raw"].write_text(planner_raw, encoding="utf-8")
    outputs["planner_generate_results"].write_text(
        json.dumps(planner_results, indent=2) + "\n",
        encoding="utf-8",
    )
    return outputs


def materialize_planner_results(
    output_dir: Path,
    planner_results: list[dict],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    asm_dir = output_dir / "assembly"
    gen_dir = output_dir / "generators"
    asm_dir.mkdir(exist_ok=True)
    gen_dir.mkdir(exist_ok=True)

    manifest: list[dict[str, object]] = []

    for test in planner_results:
        name = str(test.get("name", "unnamed_test"))
        description = str(test.get("description", ""))
        assembly = str(test.get("assembly", ""))
        has_generator = bool(test.get("has_generator", False))
        generator = test.get("generator")

        asm_path = asm_dir / f"{name}.S"
        asm_path.write_text(assembly, encoding="utf-8")

        gen_path_str: str | None = None
        if has_generator and isinstance(generator, str) and generator.strip():
            gen_path = gen_dir / f"gen_{name}.py"
            gen_path.write_text(generator, encoding="utf-8")
            gen_path_str = str(gen_path.relative_to(output_dir))

        manifest.append(
            {
                "name": name,
                "description": description,
                "assembly": str(asm_path.relative_to(output_dir)),
                "has_generator": has_generator,
                "generator": gen_path_str,
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "assembly_dir": asm_dir,
        "generators_dir": gen_dir,
        "manifest": manifest_path,
    }


def materialize_single_test_attempt(
    *,
    output_dir: Path,
    test_slot: int,
    attempt_idx: int,
    stage_label: str,
    planner_result: dict,
) -> dict[str, Path]:
    """Persist one parsed planner/fix candidate under the run directory.

    Unlike publication into the canonical baremetal source tree, this is a
    pure archival step: every successfully parsed candidate is recorded, even
    if it later fails name checks, publishing, build, or simulation.
    """

    base_dir = (
        output_dir
        / "generated_candidates"
        / f"test_{test_slot:02d}"
        / f"attempt_{attempt_idx:02d}_{stage_label}"
    )
    return materialize_planner_results(base_dir, [planner_result])


def _validate_generated_test_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Planner returned a test with an empty name.")
    if not re.fullmatch(r"[A-Za-z0-9_]+", cleaned):
        raise ValueError(
            f"Planner returned an invalid test name {name!r}. "
            "Only letters, numbers, and underscores are allowed."
        )
    return cleaned


def _write_text_if_missing_or_same(path: Path, contents: str, artifact_name: str) -> None:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != contents:
            raise FileExistsError(
                f"Refusing to overwrite existing {artifact_name}: {path}. "
                "Choose a new test name or remove the conflicting file first."
            )
        return
    path.write_text(contents, encoding="utf-8")


def publish_planner_results_to_baremetal(
    planner_results: list[dict],
) -> list[dict[str, object]]:
    """
    Publish generated tests into the canonical baremetal source directories.
    """

    published: list[dict[str, object]] = []
    seen_names: set[str] = set()

    for test in planner_results:
        name = _validate_generated_test_name(str(test.get("name", "unnamed_test")))
        if name in seen_names:
            raise ValueError(f"Planner returned duplicate test name: {name}")
        seen_names.add(name)

        description = str(test.get("description", ""))
        assembly = str(test.get("assembly", ""))
        if not assembly.strip():
            raise ValueError(f"Planner returned empty assembly for test {name}.")

        has_generator = bool(test.get("has_generator", False))
        generator = test.get("generator")

        asm_path = ASSEMBLY_DIR / f"{name}.S"
        _write_text_if_missing_or_same(asm_path, assembly, f"assembly test {name}")

        gen_path: Path | None = None
        if has_generator:
            if not isinstance(generator, str) or not generator.strip():
                raise ValueError(
                    f"Planner marked test {name} as requiring a generator but did not provide one."
                )
            gen_path = GENERATORS_DIR / f"gen_{name}.py"
            _write_text_if_missing_or_same(
                gen_path,
                generator,
                f"golden generator for {name}",
            )
        else:
            stale_generator = GENERATORS_DIR / f"gen_{name}.py"
            if stale_generator.exists():
                raise FileExistsError(
                    f"Generated test {name} does not use a generator, but "
                    f"{stale_generator} already exists and would create a stale golden flow."
                )

        published.append(
            {
                "name": name,
                "description": description,
                "assembly": asm_path,
                "generator": gen_path,
                "target": f"atlas_{name}",
                "c_source": TESTS_DIR / f"atlas_{name}.c",
                "binary": TESTS_BUILD_DIR / f"atlas_{name}.riscv",
                "golden_json": GENERATORS_DIR / f"{name}.json" if has_generator else None,
            }
        )

    return published


def build_published_baremetal_tests(
    output_dir: Path,
    published_tests: list[dict[str, object]],
) -> dict[str, Path]:
    """
    Generate goldens, assemble tests into C, and build the resulting binaries.
    """

    if not published_tests:
        raise ValueError("No published tests were provided for build.")

    for test in published_tests:
        name = str(test["name"])
        generator_path = test["generator"]
        golden_json_path = test["golden_json"]

        if isinstance(generator_path, Path) and isinstance(golden_json_path, Path):
            print(f"       Generating golden JSON for {name}...")
            with golden_json_path.open("w", encoding="utf-8") as handle:
                subprocess.run(
                    [sys.executable, str(generator_path.relative_to(BAREMETAL_DIR))],
                    cwd=BAREMETAL_DIR,
                    check=True,
                    stdout=handle,
                )

        assembler_cmd = [
            sys.executable,
            str(ASSEMBLER_PATH.relative_to(BAREMETAL_DIR)),
            str(Path(test["assembly"]).relative_to(BAREMETAL_DIR)),
            "--out-c",
            str(Path(test["c_source"]).relative_to(BAREMETAL_DIR)),
        ]
        if isinstance(golden_json_path, Path):
            assembler_cmd.extend(
                [
                    "--golden-json",
                    str(golden_json_path.relative_to(BAREMETAL_DIR)),
                ]
            )
        print(f"       Assembling {name} into C...")
        subprocess.run(assembler_cmd, cwd=BAREMETAL_DIR, check=True)

    print("       Reconfiguring baremetal CMake...")
    subprocess.run(
        ["cmake", "-S", str(TESTS_DIR.relative_to(BAREMETAL_DIR)), "-B", str(TESTS_BUILD_DIR.relative_to(BAREMETAL_DIR)), "-DCMAKE_BUILD_TYPE=Debug"],
        cwd=BAREMETAL_DIR,
        check=True,
    )

    targets = [str(test["target"]) for test in published_tests]
    print(f"       Building {len(targets)} generated target(s)...")
    subprocess.run(
        ["cmake", "--build", str(TESTS_BUILD_DIR.relative_to(BAREMETAL_DIR)), "--target", *targets],
        cwd=BAREMETAL_DIR,
        check=True,
    )

    build_manifest: list[dict[str, object]] = []
    for test in published_tests:
        c_source = Path(test["c_source"])
        binary = Path(test["binary"])
        if not c_source.exists():
            raise FileNotFoundError(f"Assembler did not emit expected C source: {c_source}")
        if not binary.exists():
            raise FileNotFoundError(f"CMake did not produce expected binary: {binary}")

        generator_path = test["generator"]
        golden_json_path = test["golden_json"]
        build_manifest.append(
            {
                "name": test["name"],
                "description": test["description"],
                "target": test["target"],
                "assembly": str(Path(test["assembly"]).relative_to(BAREMETAL_DIR)),
                "generator": (
                    str(Path(generator_path).relative_to(BAREMETAL_DIR))
                    if isinstance(generator_path, Path)
                    else None
                ),
                "golden_json": (
                    str(Path(golden_json_path).relative_to(BAREMETAL_DIR))
                    if isinstance(golden_json_path, Path)
                    else None
                ),
                "c_source": str(c_source.relative_to(BAREMETAL_DIR)),
                "binary": str(binary.relative_to(BAREMETAL_DIR)),
            }
        )

    build_manifest_path = output_dir / "built_binaries_manifest.json"
    build_manifest_path.write_text(
        json.dumps(build_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "canonical_assembly_dir": ASSEMBLY_DIR,
        "canonical_generators_dir": GENERATORS_DIR,
        "tests_dir": TESTS_DIR,
        "tests_build_dir": TESTS_BUILD_DIR,
        "build_manifest": build_manifest_path,
    }


def _classify_sim_run(rc: int, output: str) -> dict[str, object]:
    """Classify ``make run-binary`` result the same way as ``run_combo_suite.py``."""

    if rc != 0:
        return {"pass": False, "reason": f"nonzero_return_code_{rc}"}
    marker = next((m for m in SIM_FAIL_MARKERS if m in output), None)
    if marker is not None:
        return {"pass": False, "reason": f"failure_marker:{marker}"}
    if "PASS:" in output:
        return {"pass": True, "reason": "pass_marker"}
    return {"pass": True, "reason": "zero_return_code_without_failure_marker"}


def _stream_command(
    cmd: list[str],
    cwd: Path,
    *,
    timeout_s: int | None,
) -> tuple[int, str]:
    """Run ``cmd``, stream stdout/stderr to this process, return (rc, combined output)."""

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            print(line, end="", flush=True)
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        lines.append(f"\n[TIMEOUT] Exceeded {timeout_s}s; process killed.\n")
        rc = 124
    return rc, "".join(lines)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def try_build_and_run_single_test(
    *,
    test: dict[str, object],
    output_dir: Path,
    chipyard_root: Path,
    vcs_dir_relpath: str = VCS_DIR_RELPATH,
    sim_config: str = DEFAULT_SIM_CONFIG,
    run_timeout_s: int = 0,
    attempt_idx: int = 0,
    gen_coverage: bool = False,
) -> dict[str, object]:
    """Build + simulate a single published test, capturing every stage.

    ``test`` must be the single-element result of
    ``publish_planner_results_to_baremetal([...])``.

    When ``gen_coverage`` is true, the VCS invocation is run with
    ``GEN_COVERAGE=1`` so a per-test ``.vdb`` is produced as a side effect.
    The coverage gate (``evaluate_test_coverage_contribution``) consumes
    that ``.vdb`` to decide whether to keep the test.

    Returns a structured dict suitable both for logging and for feeding into
    the next fix-prompt::

        {
          "ok": bool,
          "phase": "generator" | "assembler" | "cmake_configure" | "cmake_build" | "sim",
          "reason": str,
          "log_tail": str,          # trimmed error text for the LLM
          "full_log": str,          # untrimmed, for on-disk diagnostics
          "vcs_log_path": Path | None,
        }
    """

    name = str(test["name"])
    target = str(test["target"])
    generator_path = test.get("generator")
    golden_json_path = test.get("golden_json")
    assembly_path = Path(str(test["assembly"]))
    c_source_path = Path(str(test["c_source"]))
    binary_path = Path(str(test["binary"]))

    log_dir = output_dir / "attempt_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    attempt_log_path = log_dir / f"{name}_attempt{attempt_idx:02d}.log"
    attempt_log_parts: list[str] = [
        f"=== Attempt {attempt_idx} :: {name} ===\n",
    ]

    def _record(rc: int, cmd: list[str], cwd: Path, output: str) -> None:
        attempt_log_parts.append(
            f"\n$ (cd {cwd} && {' '.join(cmd)})  [rc={rc}]\n{output}"
        )

    def _flush_log() -> None:
        attempt_log_path.write_text("".join(attempt_log_parts), encoding="utf-8")

    def _capture(cmd: list[str], cwd: Path, *, stdout_path: Path | None = None) -> tuple[int, str]:
        try:
            if stdout_path is not None:
                # Generator writes JSON to stdout; capture stderr only, tee stdout to file.
                with stdout_path.open("w", encoding="utf-8") as handle:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(cwd),
                        stdout=handle,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                return proc.returncode, proc.stderr or ""
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return proc.returncode, proc.stdout or ""
        except FileNotFoundError as exc:
            return 127, f"[missing executable] {exc}"

    # ------------------------------------------------------------------
    # Stage 1: generator (if present)
    # ------------------------------------------------------------------
    if isinstance(generator_path, Path) and isinstance(golden_json_path, Path):
        gen_cmd = [sys.executable, str(generator_path.relative_to(BAREMETAL_DIR))]
        print(f"       [attempt {attempt_idx}] Running generator for {name}...")
        rc, stderr_text = _capture(gen_cmd, BAREMETAL_DIR, stdout_path=golden_json_path)
        _record(rc, gen_cmd, BAREMETAL_DIR, stderr_text)
        if rc != 0:
            result = {
                "ok": False,
                "phase": "generator",
                "reason": f"generator_exit_{rc}",
                "log_tail": _trim_log_tail(stderr_text),
                "full_log": stderr_text,
                "vcs_log_path": None,
            }
            _flush_log()
            return result

    # ------------------------------------------------------------------
    # Stage 2: assembler
    # ------------------------------------------------------------------
    assembler_cmd = [
        sys.executable,
        str(ASSEMBLER_PATH.relative_to(BAREMETAL_DIR)),
        str(assembly_path.relative_to(BAREMETAL_DIR)),
        "--out-c",
        str(c_source_path.relative_to(BAREMETAL_DIR)),
    ]
    if isinstance(golden_json_path, Path):
        assembler_cmd.extend(
            ["--golden-json", str(golden_json_path.relative_to(BAREMETAL_DIR))]
        )
    print(f"       [attempt {attempt_idx}] Assembling {name}...")
    rc, out = _capture(assembler_cmd, BAREMETAL_DIR)
    _record(rc, assembler_cmd, BAREMETAL_DIR, out)
    if rc != 0:
        result = {
            "ok": False,
            "phase": "assembler",
            "reason": f"assembler_exit_{rc}",
            "log_tail": _trim_log_tail(out),
            "full_log": out,
            "vcs_log_path": None,
        }
        _flush_log()
        return result

    # ------------------------------------------------------------------
    # Stage 3: cmake configure (re-globs assembly/*.S and adds new target)
    # ------------------------------------------------------------------
    cmake_configure = [
        "cmake",
        "-S", str(TESTS_DIR.relative_to(BAREMETAL_DIR)),
        "-B", str(TESTS_BUILD_DIR.relative_to(BAREMETAL_DIR)),
        "-DCMAKE_BUILD_TYPE=Debug",
    ]
    print(f"       [attempt {attempt_idx}] Reconfiguring CMake...")
    rc, out = _capture(cmake_configure, BAREMETAL_DIR)
    _record(rc, cmake_configure, BAREMETAL_DIR, out)
    if rc != 0:
        result = {
            "ok": False,
            "phase": "cmake_configure",
            "reason": f"cmake_configure_exit_{rc}",
            "log_tail": _trim_log_tail(out),
            "full_log": out,
            "vcs_log_path": None,
        }
        _flush_log()
        return result

    # ------------------------------------------------------------------
    # Stage 4: cmake build of just this target
    # ------------------------------------------------------------------
    cmake_build = [
        "cmake",
        "--build", str(TESTS_BUILD_DIR.relative_to(BAREMETAL_DIR)),
        "--target", target,
    ]
    print(f"       [attempt {attempt_idx}] Building target {target}...")
    rc, out = _capture(cmake_build, BAREMETAL_DIR)
    _record(rc, cmake_build, BAREMETAL_DIR, out)
    if rc != 0:
        result = {
            "ok": False,
            "phase": "cmake_build",
            "reason": f"cmake_build_exit_{rc}",
            "log_tail": _trim_log_tail(out),
            "full_log": out,
            "vcs_log_path": None,
        }
        _flush_log()
        return result
    if not binary_path.exists():
        msg = f"Expected binary {binary_path} missing after successful build."
        result = {
            "ok": False,
            "phase": "cmake_build",
            "reason": "binary_missing_after_build",
            "log_tail": msg,
            "full_log": msg,
            "vcs_log_path": None,
        }
        _flush_log()
        return result

    # ------------------------------------------------------------------
    # Stage 5: VCS simulation. When ``gen_coverage`` is on, we add
    # ``GEN_COVERAGE=1`` so the same run that decides pass/fail also
    # produces the per-test ``.vdb`` consumed by the coverage gate.
    # ------------------------------------------------------------------
    vcs_dir = (chipyard_root / vcs_dir_relpath).resolve()
    if not vcs_dir.is_dir():
        raise FileNotFoundError(f"VCS directory does not exist: {vcs_dir}")
    make_cmd = ["make", "run-binary"]
    if gen_coverage:
        make_cmd.append("GEN_COVERAGE=1")
    make_cmd.extend(
        [
            f"BINARY={binary_path}",
            f"CONFIG={sim_config}",
            "LOADMEM=1",
        ]
    )
    cov_label = " (with GEN_COVERAGE=1)" if gen_coverage else ""
    print(f"       [attempt {attempt_idx}] Simulating {name} in VCS{cov_label}...")
    timeout = run_timeout_s if run_timeout_s > 0 else None
    rc, sim_output = _stream_command(make_cmd, vcs_dir, timeout_s=timeout)
    _record(rc, make_cmd, vcs_dir, sim_output)

    vcs_log_path = log_dir / f"{name}_attempt{attempt_idx:02d}_vcs.log"
    vcs_log_path.write_text(sim_output, encoding="utf-8")

    verdict = _classify_sim_run(rc, sim_output)
    passed = bool(verdict["pass"])
    result = {
        "ok": passed,
        "phase": "sim",
        "reason": str(verdict["reason"]),
        "log_tail": _trim_log_tail(sim_output),
        "full_log": sim_output,
        "vcs_log_path": vcs_log_path,
    }
    _flush_log()
    return result


def collect_handwritten_assembly_test_names() -> list[str]:
    """Return bare assembly test names from ``baremetal/assembly/*.S``."""

    return sorted(path.stem for path in ASSEMBLY_DIR.glob("*.S"))


def _normalize_coverage_test_name(name: str) -> str:
    return name[6:] if name.startswith("atlas_") else name


def _coverage_root_for_sim_config(vcs_dir: Path, sim_config: str) -> Path:
    """Return the VCS coverage directory for one specific simulation config."""

    return (
        vcs_dir
        / GENERATED_SRC_DIRNAME
        / f"chipyard.harness.TestHarness.{sim_config}"
        / COVERAGE_DIRNAME
    )


def _discover_coverage_vdb_dirs(coverage_root: Path) -> list[Path]:
    """Return per-test ``*.vdb`` directories from one config-specific coverage root."""

    if not coverage_root.is_dir():
        return []
    return sorted(path.resolve() for path in coverage_root.glob("*.vdb") if path.is_dir())


def _covered_test_names_from_vdb_dirs(vdb_dirs: list[Path]) -> set[str]:
    return {_normalize_coverage_test_name(path.stem) for path in vdb_dirs}


def ensure_handwritten_coverage(
    *,
    chipyard_root: Path,
    output_dir: Path,
    vcs_dir_relpath: str = VCS_DIR_RELPATH,
    sim_config: str = DEFAULT_SIM_CONFIG,
) -> dict[str, object]:
    """
    Ensure the handwritten assembly suite has a coverage DB for each test.

    Missing coverage DBs are backfilled by invoking ``./run_asm_tests.sh <missing...>``.
    """

    vcs_dir = (chipyard_root / vcs_dir_relpath).resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "baseline_run_asm_tests.log"

    expected_tests = collect_handwritten_assembly_test_names()
    coverage_root = _coverage_root_for_sim_config(vcs_dir, sim_config)
    existing_vdb_dirs = _discover_coverage_vdb_dirs(coverage_root)
    existing_covered = _covered_test_names_from_vdb_dirs(existing_vdb_dirs)
    missing_tests = [name for name in expected_tests if name not in existing_covered]

    summary: dict[str, object] = {
        "status": "failed",
        "chipyard_root": str(chipyard_root),
        "vcs_dir": str(vcs_dir.relative_to(chipyard_root)),
        "sim_config": sim_config,
        "coverage_root": str(coverage_root),
        "expected_tests": expected_tests,
        "existing_vdb_dirs": [str(path) for path in existing_vdb_dirs],
        "missing_tests_before": missing_tests,
        "log": str(log_path.relative_to(output_dir)),
    }

    if not expected_tests:
        summary["status"] = "skipped"
        summary["reason"] = "no_handwritten_assembly_tests_found"
        return summary

    if not missing_tests:
        summary["status"] = "passed"
        summary["reason"] = "baseline_coverage_already_complete"
        summary["missing_tests_after"] = []
        return summary

    if not RUN_ASM_TESTS_PATH.is_file():
        summary["reason"] = f"Missing handwritten test runner: {RUN_ASM_TESTS_PATH}"
        return summary

    cmd = ["bash", str(RUN_ASM_TESTS_PATH), *missing_tests]
    banner = (
        f"\n{'=' * 60}\n"
        f"Backfill handwritten coverage DBs\n"
        f"  (cd {BAREMETAL_DIR} && {' '.join(cmd)})\n"
        f"{'=' * 60}\n"
    )
    print(banner)
    rc, output = _stream_command(cmd, BAREMETAL_DIR, timeout_s=None)
    log_path.write_text(banner + output, encoding="utf-8")
    summary["run_asm_tests_rc"] = rc
    if rc != 0:
        summary["reason"] = f"run_asm_tests_failed_rc_{rc}"
        return summary

    final_vdb_dirs = _discover_coverage_vdb_dirs(coverage_root)
    final_covered = _covered_test_names_from_vdb_dirs(final_vdb_dirs)
    missing_after = [name for name in expected_tests if name not in final_covered]
    summary["existing_vdb_dirs_after"] = [str(path) for path in final_vdb_dirs]
    summary["missing_tests_after"] = missing_after
    if missing_after:
        summary["reason"] = "coverage_backfill_incomplete"
        return summary

    summary["status"] = "passed"
    summary["reason"] = "baseline_coverage_backfilled"
    return summary


def generate_vcs_coverage_report(
    *,
    chipyard_root: Path,
    output_dir: Path,
    sim_config: str = DEFAULT_SIM_CONFIG,
    vcs_dir_relpath: str = VCS_DIR_RELPATH,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    report_dir_name: str = "cov_overall",
    analysis_filename: str = "coverage_analysis.txt",
    log_prefix: str = "coverage",
) -> dict[str, object]:
    """
    Aggregate all ``coverage/*.vdb`` directories with ``urg`` and then run
    ``coverage_analyzer.py`` on the generated text report.
    """

    vcs_dir = (chipyard_root / vcs_dir_relpath).resolve()
    coverage_root = _coverage_root_for_sim_config(vcs_dir, sim_config)
    vdb_dirs = _discover_coverage_vdb_dirs(coverage_root)
    urg_report_dir = output_dir / report_dir_name
    analysis_path = output_dir / analysis_filename
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    urg_log_path = log_dir / f"{log_prefix}_urg.log"
    analyzer_log_path = log_dir / f"{log_prefix}_analyzer.log"

    summary: dict[str, object] = {
        "status": "failed",
        "chipyard_root": str(chipyard_root),
        "vcs_dir": str(vcs_dir.relative_to(chipyard_root)),
        "sim_config": sim_config,
        "coverage_root": str(coverage_root),
        "vdb_dirs": [str(path) for path in vdb_dirs],
        "urg_report_dir": str(urg_report_dir),
        "analysis_path": str(analysis_path),
        "urg_log": str(urg_log_path.relative_to(output_dir)),
        "analyzer_log": str(analyzer_log_path.relative_to(output_dir)),
        "threshold": coverage_threshold,
    }

    if not COVERAGE_ANALYZER_PATH.is_file():
        summary["reason"] = f"Coverage analyzer not found: {COVERAGE_ANALYZER_PATH}"
        return summary
    if shutil.which("urg") is None:
        summary["reason"] = "Could not find `urg` in PATH."
        return summary
    if not coverage_root.is_dir():
        summary["reason"] = f"Coverage root not found for sim config {sim_config}: {coverage_root}"
        return summary
    if not vdb_dirs:
        summary["reason"] = f"No .vdb coverage databases found under {coverage_root}"
        return summary

    if urg_report_dir.exists():
        shutil.rmtree(urg_report_dir)

    urg_cmd = [
        "urg",
        "-dir",
        *(str(path) for path in vdb_dirs),
        "-format",
        "text",
        "-report",
        str(urg_report_dir),
    ]
    urg_banner = (
        f"\n{'=' * 60}\n"
        f"URG coverage aggregation\n"
        f"  (cd {vcs_dir} && {' '.join(urg_cmd)})\n"
        f"{'=' * 60}\n"
    )
    print(urg_banner)
    urg_rc, urg_output = _stream_command(urg_cmd, vcs_dir, timeout_s=None)
    urg_log_path.write_text(urg_banner + urg_output, encoding="utf-8")
    summary["urg_rc"] = urg_rc
    if urg_rc != 0:
        summary["reason"] = f"urg_failed_rc_{urg_rc}"
        return summary
    if not urg_report_dir.is_dir():
        summary["reason"] = f"urg did not create expected report directory: {urg_report_dir}"
        return summary

    analyzer_cmd = [
        sys.executable,
        str(COVERAGE_ANALYZER_PATH),
        str(urg_report_dir),
        "--threshold",
        str(coverage_threshold),
        "--output",
        str(analysis_path),
    ]
    analyzer_banner = (
        f"\n{'=' * 60}\n"
        f"Coverage analyzer\n"
        f"  (cd {BAREMETAL_DIR} && {' '.join(analyzer_cmd)})\n"
        f"{'=' * 60}\n"
    )
    print(analyzer_banner)
    analyzer_rc, analyzer_output = _stream_command(analyzer_cmd, BAREMETAL_DIR, timeout_s=None)
    analyzer_log_path.write_text(analyzer_banner + analyzer_output, encoding="utf-8")
    summary["analyzer_rc"] = analyzer_rc
    if analyzer_rc != 0:
        summary["reason"] = f"coverage_analyzer_failed_rc_{analyzer_rc}"
        return summary
    if not analysis_path.is_file():
        summary["reason"] = f"Coverage analyzer did not produce expected output: {analysis_path}"
        return summary

    summary["status"] = "passed"
    summary["reason"] = "coverage_report_generated"
    return summary


def _test_vdb_path(
    *,
    chipyard_root: Path,
    sim_config: str,
    test_name: str,
    vcs_dir_relpath: str = VCS_DIR_RELPATH,
) -> Path:
    """Return the per-test ``.vdb`` directory that VCS produces under
    ``GEN_COVERAGE=1`` for binary ``atlas_<test_name>.riscv``.

    The make rule names the database after the binary stem, so this is the
    same scheme used by ``run_asm_tests.sh`` and by the coverage backfill
    helpers above. Resolving it once means every coverage gating decision in
    the loop talks about the same on-disk artifact.
    """

    vcs_dir = (chipyard_root / vcs_dir_relpath).resolve()
    coverage_root = _coverage_root_for_sim_config(vcs_dir, sim_config)
    return coverage_root / f"atlas_{test_name}.vdb"


def _remove_test_vdb(
    *,
    chipyard_root: Path,
    sim_config: str,
    test_name: str,
    vcs_dir_relpath: str = VCS_DIR_RELPATH,
) -> Path | None:
    """Best-effort delete of a single test's ``.vdb`` directory.

    Used to keep the coverage state clean when we reject a test (either for a
    failing simulation or for contributing zero coverage). Returns the path
    we removed, or ``None`` if there was nothing to remove or the delete
    failed (we never raise from cleanup).
    """

    vdb_path = _test_vdb_path(
        chipyard_root=chipyard_root,
        sim_config=sim_config,
        test_name=test_name,
        vcs_dir_relpath=vcs_dir_relpath,
    )
    if vdb_path.exists() and vdb_path.is_dir():
        try:
            shutil.rmtree(vdb_path)
            return vdb_path
        except OSError:
            return None
    return None


def evaluate_test_coverage_contribution(
    *,
    test_name: str,
    chipyard_root: Path,
    output_dir: Path,
    sim_config: str,
    vcs_dir_relpath: str,
    baseline_metrics: dict[str, float | None],
    coverage_threshold: float,
    slot: int,
    attempt_idx: int,
    label_prefix: str = "slot_cov",
) -> dict[str, object]:
    """Aggregate URG with this test's ``.vdb`` included and decide whether it
    moved any of the top-level Atlas-tile coverage metrics.

    The simulation phase (``try_build_and_run_single_test`` with
    ``gen_coverage=True``) is responsible for producing the per-test
    ``.vdb``; this function only does the post-sim aggregation + diff.

    Returns a structured record:
        {
          "ok": bool,                   # aggregation succeeded
          "improved": bool,             # at least one metric strictly above eps
          "improved_keys": [..],
          "metrics": {score,line,...} | None,
          "deltas":  {score,line,...} | None,
          "delta_pretty": str | None,
          "had_signal": bool,
          "coverage_summary": <generate_vcs_coverage_report dict>,
          "analysis_path": str | None,
        }

    The caller is expected to:
      * roll ``metrics`` forward as the new baseline if it accepts the test;
      * call ``_remove_test_vdb`` if it rejects the test, so the next slot's
        aggregation does not silently include this candidate.
    """

    label = f"slot{slot:02d}_attempt{attempt_idx:02d}_{test_name}"
    cov_summary = generate_vcs_coverage_report(
        chipyard_root=chipyard_root,
        output_dir=output_dir,
        sim_config=sim_config,
        vcs_dir_relpath=vcs_dir_relpath,
        coverage_threshold=coverage_threshold,
        report_dir_name=f"{label_prefix}/{label}/cov_overall",
        analysis_filename=f"{label_prefix}/{label}/coverage_analysis.txt",
        log_prefix=f"{label_prefix}_{label}",
    )

    if cov_summary["status"] != "passed":
        return {
            "ok": False,
            "improved": False,
            "improved_keys": [],
            "metrics": None,
            "deltas": None,
            "delta_pretty": None,
            "had_signal": False,
            "coverage_summary": cov_summary,
            "analysis_path": cov_summary.get("analysis_path"),
            "reason": (
                f"coverage_aggregation_failed:{cov_summary.get('reason', 'unknown')}"
            ),
        }

    analysis_path = Path(str(cov_summary["analysis_path"]))
    try:
        analysis_text = analysis_path.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "ok": False,
            "improved": False,
            "improved_keys": [],
            "metrics": None,
            "deltas": None,
            "delta_pretty": None,
            "had_signal": False,
            "coverage_summary": cov_summary,
            "analysis_path": str(analysis_path),
            "reason": f"coverage_analysis_unreadable:{exc}",
        }

    metrics = parse_coverage_summary_metrics(analysis_text)
    improvement = coverage_metrics_improved(baseline_metrics, metrics)
    deltas = improvement["deltas"] if isinstance(improvement, dict) else None
    delta_pretty = format_coverage_delta_for_log(baseline_metrics, metrics, deltas)

    return {
        "ok": True,
        "improved": bool(improvement.get("improved")),
        "improved_keys": list(improvement.get("improved_keys", [])),
        "metrics": metrics,
        "deltas": deltas,
        "delta_pretty": delta_pretty,
        "had_signal": bool(improvement.get("had_signal")),
        "coverage_summary": cov_summary,
        "analysis_path": str(analysis_path),
        "reason": "coverage_improved" if improvement.get("improved") else "coverage_unchanged",
    }


def run_vcs_binaries(
    *,
    chipyard_root: Path,
    tests: list[dict[str, object]],
    output_dir: Path,
    sim_config: str = DEFAULT_SIM_CONFIG,
    vcs_dir_relpath: str = VCS_DIR_RELPATH,
    gen_coverage: bool = True,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    coverage_report_dir_name: str = "cov_overall",
    coverage_analysis_filename: str = "coverage_analysis.txt",
    coverage_log_prefix: str = "coverage",
    loadmem: bool = True,
    run_timeout_s: int = 0,
    execute: bool = True,
) -> dict[str, object]:
    """
    For each test dict (must include ``name`` and ``binary`` as ``Path`` or str),
    run ``make run-binary`` from ``chipyard_root / vcs_dir_relpath``.

    Writes per-test logs under ``output_dir / "logs"`` and a JSON + Markdown summary.
    """

    vcs_dir = (chipyard_root / vcs_dir_relpath).resolve()
    if not vcs_dir.is_dir():
        raise FileNotFoundError(
            f"VCS directory does not exist: {vcs_dir} (chipyard_root={chipyard_root})"
        )

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timeout = run_timeout_s if run_timeout_s > 0 else None

    results_tests: list[dict[str, object]] = []
    for test in tests:
        name = str(test["name"])
        raw_bin = test["binary"]
        binary = Path(raw_bin) if isinstance(raw_bin, Path) else Path(str(raw_bin))
        if not binary.is_absolute():
            binary = (BAREMETAL_DIR / binary).resolve()
        if not binary.exists():
            raise FileNotFoundError(f"Simulator binary missing for {name}: {binary}")

        make_cmd = ["make", "run-binary"]
        if gen_coverage:
            make_cmd.append("GEN_COVERAGE=1")
        make_cmd.append(f"BINARY={binary}")
        make_cmd.append(f"CONFIG={sim_config}")
        if loadmem:
            make_cmd.append("LOADMEM=1")

        log_path = log_dir / f"vcs_{name}.log"
        banner = (
            f"\n{'=' * 60}\n"
            f"VCS run-binary :: {name}\n"
            f"  (cd {vcs_dir} && {' '.join(make_cmd)})\n"
            f"{'=' * 60}\n"
        )
        print(banner)

        if not execute:
            log_path.write_text(banner + "[dry-run] command not executed.\n", encoding="utf-8")
            results_tests.append(
                {
                    "name": name,
                    "binary": str(binary),
                    "run_rc": None,
                    "status": "dry_run",
                    "reason": "commands_not_executed",
                    "pass": None,
                    "log": str(log_path.relative_to(output_dir)),
                }
            )
            continue

        rc, output = _stream_command(make_cmd, vcs_dir, timeout_s=timeout)
        log_path.write_text(banner + output, encoding="utf-8")
        verdict = _classify_sim_run(rc, output)
        passed = bool(verdict["pass"])
        results_tests.append(
            {
                "name": name,
                "binary": str(binary),
                "run_rc": rc,
                "status": "passed" if passed else "failed",
                "reason": str(verdict["reason"]),
                "pass": passed,
                "log": str(log_path.relative_to(output_dir)),
            }
        )

    summary: dict[str, object] = {
        "chipyard_root": str(chipyard_root),
        "vcs_dir": str(vcs_dir.relative_to(chipyard_root)),
        "sim_config": sim_config,
        "gen_coverage": gen_coverage,
        "loadmem": loadmem,
        "execute": execute,
        "tests": results_tests,
    }

    if not execute:
        coverage_summary: dict[str, object] = {
            "status": "skipped",
            "reason": "dry_run_sim",
        }
    elif not gen_coverage:
        coverage_summary = {
            "status": "skipped",
            "reason": "gen_coverage_disabled",
        }
    else:
        coverage_summary = generate_vcs_coverage_report(
            chipyard_root=chipyard_root,
            output_dir=output_dir,
            sim_config=sim_config,
            vcs_dir_relpath=vcs_dir_relpath,
            coverage_threshold=coverage_threshold,
            report_dir_name=coverage_report_dir_name,
            analysis_filename=coverage_analysis_filename,
            log_prefix=coverage_log_prefix,
        )
    summary["coverage"] = coverage_summary

    results_path = output_dir / "vcs_run_results.json"
    _write_json(results_path, summary)

    passed_n = sum(1 for t in results_tests if t["status"] == "passed")
    failed_n = sum(1 for t in results_tests if t["status"] == "failed")
    skipped_n = sum(1 for t in results_tests if t["status"] == "dry_run")
    md_lines = [
        "# Atlas VCS run summary (verif_buddy)",
        "",
        f"- Chipyard root: `{chipyard_root}`",
        f"- Config: `{sim_config}`",
        f"- GEN_COVERAGE: `{gen_coverage}`",
        f"- Executed: `{execute}`",
        f"- Passed: `{passed_n}`  Failed: `{failed_n}`  Skipped/dry-run: `{skipped_n}`",
        "",
        "| Status | Test | RC | Reason | Log |",
        "|---|---|---:|---|---|",
    ]
    for t in results_tests:
        rc_s = "" if t["run_rc"] is None else str(t["run_rc"])
        md_lines.append(
            f"| {t['status']} | {t['name']} | {rc_s} | {t['reason']} | `{t['log']}` |"
        )
    md_lines.extend(
        [
            "",
            "## Coverage Aggregation",
            "",
            f"- Status: `{coverage_summary.get('status')}`",
            f"- Reason: `{coverage_summary.get('reason')}`",
        ]
    )
    if "urg_report_dir" in coverage_summary:
        md_lines.append(f"- URG report: `{coverage_summary['urg_report_dir']}`")
    if "analysis_path" in coverage_summary:
        md_lines.append(f"- Analyzer output: `{coverage_summary['analysis_path']}`")
    if "urg_log" in coverage_summary:
        md_lines.append(f"- URG log: `{coverage_summary['urg_log']}`")
    if "analyzer_log" in coverage_summary:
        md_lines.append(f"- Analyzer log: `{coverage_summary['analyzer_log']}`")
    summary_md = output_dir / "vcs_run_summary.md"
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    summary["results_path"] = results_path
    summary["summary_md"] = summary_md
    return summary


def load_build_manifest(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Build manifest must be a JSON array: {path}")
    out: list[dict[str, object]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        binary = entry.get("binary")
        if not name or not binary:
            raise ValueError(f"Manifest entry missing name/binary: {entry!r}")
        out.append({"name": str(name), "binary": str(binary)})
    return out


def dump_named_prompt_bundle(
    *,
    output_dir: Path,
    bundle_name: str,
    system_prompt: str,
    user_prompt: str,
    raw_response: str | None = None,
) -> dict[str, Path]:
    """Persist one named LLM prompt bundle under ``used_prompts/``."""

    prompt_dir = output_dir / USED_PROMPTS_DIRNAME / bundle_name
    prompt_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "used_prompts_dir": prompt_dir,
        "system": prompt_dir / "system_prompt.md",
        "user": prompt_dir / "user_prompt.md",
    }
    outputs["system"].write_text(system_prompt, encoding="utf-8")
    outputs["user"].write_text(user_prompt, encoding="utf-8")
    if raw_response is not None:
        outputs["raw_response"] = prompt_dir / "raw_response.md"
        outputs["raw_response"].write_text(raw_response, encoding="utf-8")
    return outputs


def _dump_per_test_prompt(
    *,
    output_dir: Path,
    test_slot: int,
    attempt_idx: int,
    system_prompt: str,
    user_prompt: str,
    raw_response: str,
) -> None:
    base = output_dir / USED_PROMPTS_DIRNAME / f"test_{test_slot:02d}"
    base.mkdir(parents=True, exist_ok=True)
    (base / f"attempt_{attempt_idx:02d}_system.md").write_text(system_prompt, encoding="utf-8")
    (base / f"attempt_{attempt_idx:02d}_user.md").write_text(user_prompt, encoding="utf-8")
    (base / f"attempt_{attempt_idx:02d}_raw_response.md").write_text(raw_response, encoding="utf-8")


def run_incremental_generation_loop(
    *,
    client: "OpenAI",
    model: str,
    analysis: str,
    architecture: str,
    isa_docs: str,
    assets: BuiltAgentAssets,
    output_dir: Path,
    chipyard_root: Path,
    vcs_dir_relpath: str,
    sim_config: str,
    num_tests: int,
    max_fix_attempts: int,
    run_timeout_s: int,
    llm_timeout_s: int,
    baseline_coverage_metrics: dict[str, float | None] | None = None,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    coverage_gating_enabled: bool = True,
) -> dict[str, object]:
    """Generate + validate tests one at a time, gated by measured coverage.

    For each of ``num_tests`` slots:

    1. Ask the planner for exactly one test, then publish/build/simulate it
       under ``GEN_COVERAGE=1`` (the same sim run produces the per-test
       ``.vdb`` we feed to the coverage gate).
    2. If the simulation fails, send a fix prompt that carries the failing
       artifacts + the error-log tail, up to ``max_fix_attempts`` times
       (initial + retries). This is the existing correctness retry loop.
    3. If the simulation passes, run URG + ``coverage_analyzer.py`` over the
       full ``.vdb`` set including this test, parse the top-level Atlas-tile
       coverage metrics, and compare against the rolling baseline. The test
       is **accepted** only if it strictly improves at least one metric —
       otherwise its on-disk artifacts and ``.vdb`` are removed and we move
       on to the next slot (no fix retry, since "test runs but adds nothing"
       is not a correctness bug for the LLM to fix).
    4. Each accepted test rolls the baseline metrics forward, so the next
       slot's gate compares against the cumulative coverage of every test
       already accepted in this session.

    ``baseline_coverage_metrics`` is the coverage state captured *before*
    this loop started (typically from the handwritten-test backfill). When
    ``coverage_gating_enabled`` is false (or the baseline is missing), the
    gate degrades to "always accept any sim-passing test" so the legacy
    behavior is preserved for callers that have not yet wired baseline
    metrics in.

    Returns a summary dict with ``accepted``, ``skipped``, ``per_slot``
    entries plus the final ``current_coverage_metrics`` after the loop.
    """

    planner_system = build_planner_system_prompt(rules=assets.rules)

    # The set of names already in the canonical suite when we start. We add to
    # this as we accept tests, and we forbid the LLM from reusing any of them.
    forbidden_names: set[str] = set(
        path.stem for path in ASSEMBLY_DIR.glob("*.S")
    )

    # Rolling coverage baseline. Each accepted test updates this so the next
    # slot's gate compares against the union of (handwritten suite) ∪ (every
    # test accepted earlier in this session). Coverage is monotonic, so this
    # is well-defined even when the analyzer cannot parse a metric (we keep
    # the previous value rather than regressing it to None).
    current_coverage_metrics: dict[str, float | None] = dict(
        baseline_coverage_metrics or _empty_coverage_metrics()
    )

    accepted: list[dict[str, object]] = []  # full published-test dicts
    accepted_for_prompt: list[dict[str, str]] = []  # {name, description}
    skipped: list[dict[str, object]] = []
    rejected_for_no_coverage: list[dict[str, object]] = []
    per_slot_records: list[dict[str, object]] = []

    for slot in range(num_tests):
        print("\n" + "#" * 70)
        print(f"# Incremental slot {slot + 1}/{num_tests}")
        print(f"#   accepted so far: {len(accepted)}, skipped so far: {len(skipped)}")
        print("#" * 70)

        # Snapshot the suite *as it stands right now* so the planner sees
        # every handwritten test + every test we've already accepted in
        # this session.
        current_assembly = collect_assembly_tests()
        current_generators = collect_generators()

        planner_user = build_planner_user_prompt(
            analysis=analysis,
            architecture=architecture,
            isa_docs=isa_docs,
            num_tests=1,
            assembly_tests=current_assembly,
            generators=current_generators,
            optimization_menu=assets.optimization_menu,
            baremetal_test_examples=assets.baremetal_test_examples,
            rules=assets.rules,
            accepted_generated_tests=accepted_for_prompt,
            forbidden_names=forbidden_names,
        )

        candidate: dict | None = None
        published_entry: dict | None = None
        last_result: dict | None = None
        attempts_log: list[dict[str, object]] = []
        success = False
        coverage_eval: dict[str, object] | None = None
        coverage_rejected = False

        for attempt_idx in range(max_fix_attempts):
            if attempt_idx == 0 or candidate is None:
                user_prompt = planner_user
                stage_label = "planner"
            else:
                generator_text = None
                if bool(candidate.get("has_generator")) and isinstance(candidate.get("generator"), str):
                    generator_text = str(candidate["generator"])
                user_prompt = build_fix_user_prompt(
                    test_name=str(candidate["name"]),
                    description=str(candidate.get("description", "")),
                    assembly=str(candidate.get("assembly", "")),
                    generator=generator_text,
                    failure_phase=str(last_result.get("phase")) if last_result else "unknown",
                    failure_reason=str(last_result.get("reason")) if last_result else "unknown",
                    failure_log_tail=str(last_result.get("log_tail")) if last_result else "",
                    attempt_idx=attempt_idx,
                    architecture=architecture,
                    isa_docs=isa_docs,
                    generator_helper_reference=assets.generator_helper_reference,
                    rules=assets.rules,
                )
                stage_label = "fix"

            print(f"\n   [slot {slot:02d}][attempt {attempt_idx:02d}] calling LLM ({stage_label})...")
            raw = call_llm(
                client=client,
                model=model,
                system=planner_system,
                user=user_prompt,
                timeout_s=llm_timeout_s,
            )
            _dump_per_test_prompt(
                output_dir=output_dir,
                test_slot=slot,
                attempt_idx=attempt_idx,
                system_prompt=planner_system,
                user_prompt=user_prompt,
                raw_response=raw,
            )

            expected_name = (
                str(candidate["name"]) if attempt_idx > 0 and candidate is not None else None
            )
            try:
                new_candidate = extract_single_test_from_json(raw, expected_name=expected_name)
            except ValueError as exc:
                print(f"       LLM response invalid: {exc}")
                attempts_log.append(
                    {
                        "attempt": attempt_idx,
                        "stage": stage_label,
                        "ok": False,
                        "phase": "llm_parse",
                        "reason": f"parse_error: {exc}",
                    }
                )
                last_result = {
                    "phase": "llm_parse",
                    "reason": f"parse_error: {exc}",
                    "log_tail": raw[-4000:],
                }
                # If this was the very first attempt and parse failed, we don't
                # have a candidate at all; on next iteration we'll re-prompt
                # with planner_user. Clear candidate to be safe.
                candidate = None
                continue

            candidate = new_candidate
            name = str(candidate["name"])

            # Archive every parsed candidate under the run folder, regardless
            # of whether it later passes publication/build/sim. This keeps a
            # complete record of what the LLM generated during the session.
            materialize_single_test_attempt(
                output_dir=output_dir,
                test_slot=slot,
                attempt_idx=attempt_idx,
                stage_label=stage_label,
                planner_result=candidate,
            )

            # Refuse to accept a test whose name collides with something
            # already in the suite (either handwritten or previously accepted).
            if name in forbidden_names:
                print(f"       LLM proposed forbidden name {name!r}; rejecting.")
                attempts_log.append(
                    {
                        "attempt": attempt_idx,
                        "stage": stage_label,
                        "ok": False,
                        "phase": "name_collision",
                        "reason": f"duplicate_name:{name}",
                    }
                )
                last_result = {
                    "phase": "name_collision",
                    "reason": f"duplicate_name:{name}",
                    "log_tail": (
                        f"The name {name!r} is already in use by an existing test. "
                        "Choose a different, descriptive name that reflects this test's "
                        "distinct coverage intent."
                    ),
                }
                continue

            # Clean any previous on-disk copy before (re-)publishing.
            unpublish_single_test(name)
            try:
                published_list = publish_planner_results_to_baremetal([candidate])
            except (ValueError, FileExistsError) as exc:
                print(f"       Publish failed: {exc}")
                attempts_log.append(
                    {
                        "attempt": attempt_idx,
                        "stage": stage_label,
                        "ok": False,
                        "phase": "publish",
                        "reason": str(exc),
                        "name": name,
                    }
                )
                last_result = {
                    "phase": "publish",
                    "reason": str(exc),
                    "log_tail": str(exc),
                }
                continue
            published_entry = published_list[0]

            result = try_build_and_run_single_test(
                test=published_entry,
                output_dir=output_dir,
                chipyard_root=chipyard_root,
                vcs_dir_relpath=vcs_dir_relpath,
                sim_config=sim_config,
                run_timeout_s=run_timeout_s,
                attempt_idx=attempt_idx,
                gen_coverage=coverage_gating_enabled,
            )
            attempt_record: dict[str, object] = {
                "attempt": attempt_idx,
                "stage": stage_label,
                "name": name,
                "ok": bool(result["ok"]),
                "phase": str(result["phase"]),
                "reason": str(result["reason"]),
            }
            last_result = result

            if not result["ok"]:
                attempts_log.append(attempt_record)
                print(
                    f"       [slot {slot:02d}] attempt {attempt_idx} failed "
                    f"in phase={result['phase']} reason={result['reason']}."
                )
                continue

            # ---------------------------------------------------------
            # Sim passed. Run the coverage gate before accepting.
            # ---------------------------------------------------------
            print(
                f"       [slot {slot:02d}] {name} passed sim on attempt "
                f"{attempt_idx}; evaluating coverage contribution..."
            )

            if not coverage_gating_enabled:
                attempt_record["coverage_gate"] = "disabled"
                attempts_log.append(attempt_record)
                print(
                    f"       [slot {slot:02d}] coverage gating disabled; "
                    f"ACCEPTED {name} on attempt {attempt_idx}."
                )
                success = True
                break

            coverage_eval = evaluate_test_coverage_contribution(
                test_name=name,
                chipyard_root=chipyard_root,
                output_dir=output_dir,
                sim_config=sim_config,
                vcs_dir_relpath=vcs_dir_relpath,
                baseline_metrics=current_coverage_metrics,
                coverage_threshold=coverage_threshold,
                slot=slot,
                attempt_idx=attempt_idx,
            )

            attempt_record["coverage_gate"] = {
                "ok": bool(coverage_eval.get("ok")),
                "improved": bool(coverage_eval.get("improved")),
                "improved_keys": list(coverage_eval.get("improved_keys") or []),
                "metrics": coverage_eval.get("metrics"),
                "deltas": coverage_eval.get("deltas"),
                "reason": coverage_eval.get("reason"),
                "analysis_path": coverage_eval.get("analysis_path"),
            }
            attempts_log.append(attempt_record)

            if not coverage_eval.get("ok"):
                # Aggregation itself failed (e.g. urg crashed). We do NOT
                # silently accept the test in this case — surface it as a
                # rejected slot so the operator sees it and re-runs. The
                # post-loop cleanup branch handles vdb + artifact teardown.
                reason = str(coverage_eval.get("reason", "coverage_gate_failed"))
                print(
                    f"       [slot {slot:02d}] coverage gate FAILED for {name}: "
                    f"{reason}. Moving to next slot."
                )
                last_result = {
                    "phase": "coverage_gate",
                    "reason": reason,
                    "log_tail": reason,
                }
                coverage_rejected = True
                break

            delta_pretty = str(coverage_eval.get("delta_pretty") or "")
            if delta_pretty:
                print(delta_pretty)

            if coverage_eval.get("improved"):
                improved_keys = ", ".join(coverage_eval.get("improved_keys") or [])
                print(
                    f"       [slot {slot:02d}] ACCEPTED {name} on attempt "
                    f"{attempt_idx}: coverage improved on [{improved_keys}]."
                )
                success = True
                break

            # Sim passed but coverage didn't move. Per design, we do NOT
            # retry — "test runs but adds zero coverage" is not a
            # correctness bug for the LLM to fix; it's a planning/coverage
            # issue. Cleanup of artifacts + vdb happens in the post-loop
            # branch below so it is identical to the "all attempts failed"
            # path and we don't double-print "removed 0 artifacts".
            print(
                f"       [slot {slot:02d}] REJECTED {name} on attempt "
                f"{attempt_idx}: sim passed but coverage did not improve. "
                "Moving to next slot."
            )
            last_result = {
                "phase": "coverage_gate",
                "reason": "coverage_unchanged",
                "log_tail": delta_pretty
                or "Test ran successfully but did not increase any top-level "
                "coverage metric over the rolling baseline.",
            }
            coverage_rejected = True
            break

        if success and candidate is not None and published_entry is not None:
            accepted.append(published_entry)
            accepted_for_prompt.append(
                {
                    "name": str(candidate["name"]),
                    "description": str(candidate.get("description", "")),
                }
            )
            forbidden_names.add(str(candidate["name"]))

            # Roll the rolling baseline forward using whatever metrics the
            # gate measured. Keep the previous value for any metric the
            # analyzer left as None so we never silently regress.
            new_metrics_record: dict[str, float | None] | None = None
            if isinstance(coverage_eval, dict) and coverage_eval.get("metrics"):
                new_metrics = coverage_eval["metrics"]
                if isinstance(new_metrics, dict):
                    for key in COVERAGE_METRIC_KEYS:
                        new_value = new_metrics.get(key)
                        if new_value is not None:
                            current_coverage_metrics[key] = new_value
                    new_metrics_record = dict(new_metrics)

            per_slot_records.append(
                {
                    "slot": slot,
                    "outcome": "accepted",
                    "name": str(candidate["name"]),
                    "description": str(candidate.get("description", "")),
                    "attempts": attempts_log,
                    "coverage_after": dict(current_coverage_metrics),
                    "coverage_measured_this_slot": new_metrics_record,
                    "coverage_improved_keys": (
                        list(coverage_eval.get("improved_keys") or [])
                        if isinstance(coverage_eval, dict)
                        else []
                    ),
                }
            )
        else:
            # Exhausted all fix attempts (or rejected by the coverage gate).
            # Revert any on-disk pollution so the next slot's planner context
            # stays clean, and drop the per-test ``.vdb`` so it does not
            # taint the next slot's coverage delta.
            revert_name = str(candidate["name"]) if candidate is not None else None
            removed_artifacts: list[Path] = []
            removed_vdb: Path | None = None
            if revert_name:
                removed_artifacts = unpublish_single_test(revert_name)
                removed_vdb = _remove_test_vdb(
                    chipyard_root=chipyard_root,
                    sim_config=sim_config,
                    test_name=revert_name,
                    vcs_dir_relpath=vcs_dir_relpath,
                )

            outcome = "rejected_no_coverage" if coverage_rejected else "skipped"
            if coverage_rejected:
                rejected_for_no_coverage.append(
                    {
                        "slot": slot,
                        "name": revert_name,
                        "attempts": attempts_log,
                        "last_phase": str(last_result.get("phase")) if last_result else None,
                        "last_reason": str(last_result.get("reason")) if last_result else None,
                        "coverage_eval": (
                            {
                                "improved": bool(coverage_eval.get("improved")),
                                "improved_keys": list(
                                    coverage_eval.get("improved_keys") or []
                                ),
                                "metrics": coverage_eval.get("metrics"),
                                "deltas": coverage_eval.get("deltas"),
                                "reason": coverage_eval.get("reason"),
                            }
                            if isinstance(coverage_eval, dict)
                            else None
                        ),
                    }
                )
                if revert_name:
                    print(
                        f"       [slot {slot:02d}] {outcome.upper()} {revert_name}; "
                        f"removed {len(removed_artifacts)} artifact(s)"
                        + (f" + .vdb at {removed_vdb}" if removed_vdb else "")
                        + "."
                    )
            else:
                skipped.append(
                    {
                        "slot": slot,
                        "name": revert_name,
                        "attempts": attempts_log,
                        "last_phase": str(last_result.get("phase")) if last_result else None,
                        "last_reason": str(last_result.get("reason")) if last_result else None,
                    }
                )
                if revert_name:
                    print(
                        f"       [slot {slot:02d}] SKIPPED {revert_name}; "
                        f"removed {len(removed_artifacts)} artifact(s)"
                        + (f" + .vdb at {removed_vdb}" if removed_vdb else "")
                        + "."
                    )

            per_slot_records.append(
                {
                    "slot": slot,
                    "outcome": outcome,
                    "name": revert_name,
                    "attempts": attempts_log,
                    "last_phase": str(last_result.get("phase")) if last_result else None,
                    "last_reason": str(last_result.get("reason")) if last_result else None,
                    "coverage_after": dict(current_coverage_metrics),
                    "coverage_eval": (
                        {
                            "improved": bool(coverage_eval.get("improved")),
                            "improved_keys": list(
                                coverage_eval.get("improved_keys") or []
                            ),
                            "metrics": coverage_eval.get("metrics"),
                            "deltas": coverage_eval.get("deltas"),
                            "reason": coverage_eval.get("reason"),
                        }
                        if isinstance(coverage_eval, dict)
                        else None
                    ),
                }
            )

    summary = {
        "requested": num_tests,
        "accepted_count": len(accepted),
        "skipped_count": len(skipped),
        "rejected_no_coverage_count": len(rejected_for_no_coverage),
        "coverage_gating_enabled": coverage_gating_enabled,
        "coverage_threshold": coverage_threshold,
        "baseline_coverage_metrics": dict(baseline_coverage_metrics or _empty_coverage_metrics()),
        "current_coverage_metrics": dict(current_coverage_metrics),
        "accepted": [
            {
                "name": str(t["name"]),
                "description": str(t.get("description", "")),
                "target": str(t["target"]),
                "binary": str(t["binary"]),
            }
            for t in accepted
        ],
        "skipped": skipped,
        "rejected_no_coverage": rejected_for_no_coverage,
        "per_slot": per_slot_records,
    }
    summary_path = output_dir / "incremental_generation_summary.json"
    _write_json(summary_path, summary)
    print(
        f"\nIncremental loop complete: {len(accepted)} accepted, "
        f"{len(rejected_for_no_coverage)} rejected for zero coverage, "
        f"{len(skipped)} skipped (other failures)."
    )
    if coverage_gating_enabled:
        baseline_in = dict(baseline_coverage_metrics or _empty_coverage_metrics())
        print("Coverage rolled forward across the session:")
        print(format_coverage_delta_for_log(baseline_in, current_coverage_metrics))
    print(f"Summary written to {summary_path}")

    return {
        "summary": summary,
        "summary_path": summary_path,
        "accepted_tests": accepted,  # list of published-test dicts ready for run_vcs_binaries
        "accepted_for_prompt": accepted_for_prompt,
        "current_coverage_metrics": dict(current_coverage_metrics),
        "rejected_for_no_coverage": rejected_for_no_coverage,
    }





def main() -> int:
    run_label = "coverage_loop"

    parser = argparse.ArgumentParser(
        description=(
            "Run the Atlas coverage-driven generation loop one test at a time: "
            "ensure handwritten coverage exists, aggregate baseline RTL coverage, "
            "then for each slot ask the planner for one test, build and simulate "
            "it, retry via fix prompts on failure, and finally aggregate post-"
            "generation coverage across every accepted test."
        )
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=8,
        help="Number of generation slots to attempt (each slot yields at most one accepted test).",
    )
    parser.add_argument(
        "--max-fix-attempts",
        type=int,
        default=3,
        help=(
            "Maximum LLM attempts per slot (initial + fix retries). "
            "If every attempt fails, the slot is reverted and skipped."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4",
        help="OpenAI model to use for analyzer and planner calls.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Base directory under which a per-run timestamped output folder will be created.",
    )
    parser.add_argument(
        "--analysis-file",
        type=Path,
        default=OUTPUT_DIR / "analysis.md",
        help="Optional precomputed analysis markdown. If present, skip the analyzer call.",
    )
    parser.add_argument(
        "--chipyard-root",
        type=Path,
        default=None,
        help="Chipyard repository root (auto-detected from baremetal path if omitted).",
    )
    parser.add_argument(
        "--vcs-dir",
        type=str,
        default=VCS_DIR_RELPATH,
        help="Directory under Chipyard root containing the VCS makefile (default: sims/vcs).",
    )
    parser.add_argument(
        "--run-timeout-s",
        type=int,
        default=0,
        help="Timeout in seconds per simulator invocation (0 = no limit).",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=DEFAULT_COVERAGE_THRESHOLD,
        help="Threshold passed to coverage_analyzer.py for low-coverage module reporting.",
    )
    parser.add_argument(
        "--llm-timeout-s",
        type=int,
        default=DEFAULT_LLM_TIMEOUT_S,
        help="Timeout in seconds per LLM request (default: 1800 = 30 minutes).",
    )
    args = parser.parse_args()

    if OpenAI is None:
        sys.exit(
            "ERROR: openai is required to run LLM calls. Install it with:\n"
            "  pip install openai"
        )

    if load_dotenv is not None:
        load_dotenv(BAREMETAL_DIR / ".env")
        load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: OPENAI_API_KEY not set. Create a .env file with OPENAI_API_KEY=sk-... "
            "in the baremetal/ directory or export it in the environment."
        )

    client = OpenAI(api_key=api_key)
    assets = _get_default_shared_assets()
    architecture = assets.architecture
    isa_docs = load_isa_docs()
    assembly_tests = collect_assembly_tests()
    generators = collect_generators()
    try:
        chipyard_root = (
            args.chipyard_root.resolve()
            if args.chipyard_root
            else find_chipyard_root()
        )
    except FileNotFoundError as exc:
        sys.exit(f"ERROR: {exc}")

    args.output_dir = create_run_output_dir(args.output_dir, run_label)
    n_steps = 6

    print("=" * 60)
    print("Atlas Verif Buddy (incremental loop)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output dir: {args.output_dir}")
    print(f"Chipyard root: {chipyard_root}")
    print(f"Found {len(assembly_tests)} assembly tests and {len(generators)} generators")
    print(
        f"Will attempt {args.num_tests} generation slot(s), "
        f"up to {args.max_fix_attempts} LLM attempt(s) per slot."
    )

    print(f"\n[1/{n_steps}] Ensuring handwritten assembly coverage databases exist...")
    baseline_status = ensure_handwritten_coverage(
        chipyard_root=chipyard_root,
        output_dir=args.output_dir,
        vcs_dir_relpath=args.vcs_dir,
        sim_config=DEFAULT_SIM_CONFIG,
    )
    baseline_status_path = args.output_dir / "baseline_coverage_status.json"
    _write_json(baseline_status_path, baseline_status)
    print(f"       Baseline coverage status: {baseline_status['status']} ({baseline_status['reason']})")
    print(f"       Status file: {baseline_status_path}")
    if baseline_status["status"] == "failed":
        return 1

    print(f"\n[2/{n_steps}] Aggregating baseline coverage with URG and coverage_analyzer.py...")
    baseline_coverage = generate_vcs_coverage_report(
        chipyard_root=chipyard_root,
        output_dir=args.output_dir,
        sim_config=DEFAULT_SIM_CONFIG,
        vcs_dir_relpath=args.vcs_dir,
        coverage_threshold=args.coverage_threshold,
        report_dir_name="baseline_cov_overall",
        analysis_filename="baseline_coverage_analysis.txt",
        log_prefix="baseline_coverage",
    )
    baseline_coverage_path = args.output_dir / "baseline_coverage_summary.json"
    _write_json(baseline_coverage_path, baseline_coverage)
    print(f"       Coverage status: {baseline_coverage['status']} ({baseline_coverage['reason']})")
    print(f"       Coverage summary: {baseline_coverage_path}")
    if baseline_coverage["status"] != "passed":
        return 1

    baseline_coverage_text_full = load_analysis_text(Path(str(baseline_coverage["analysis_path"])))
    baseline_coverage_text = condense_coverage_report_for_prompt(baseline_coverage_text_full)
    if baseline_coverage_text != baseline_coverage_text_full:
        analyzer_cov_prompt_path = args.output_dir / "baseline_coverage_analysis_for_prompt.txt"
        analyzer_cov_prompt_path.write_text(baseline_coverage_text, encoding="utf-8")
        print(
            "       Condensed measured coverage report for prompting: "
            f"{len(baseline_coverage_text_full):,} -> {len(baseline_coverage_text):,} chars"
        )
        print(f"       Prompt-safe coverage summary: {analyzer_cov_prompt_path}")

    # Parse the top-level Atlas-tile coverage metrics out of the baseline
    # report. These are the rolling thresholds the per-slot coverage gate
    # compares against; each accepted test rolls them forward.
    baseline_metrics = parse_coverage_summary_metrics(baseline_coverage_text_full)
    print("       Baseline coverage metrics (rolling gate starts here):")
    print(format_coverage_delta_for_log(_empty_coverage_metrics(), baseline_metrics))
    _write_json(
        args.output_dir / "baseline_coverage_metrics.json",
        {
            "metrics": baseline_metrics,
            "source": str(baseline_coverage["analysis_path"]),
            "improvement_eps": COVERAGE_IMPROVEMENT_EPS,
        },
    )

    analysis_path = args.analysis_file if args.analysis_file.exists() else None
    if analysis_path is not None:
        print(f"\n[3/{n_steps}] Using existing analysis file...")
        analysis = load_analysis_text(analysis_path)
        print(f"       Loaded analysis from {analysis_path} ({len(analysis)} chars)")
    else:
        print(f"\n[3/{n_steps}] Running analyzer with measured RTL coverage...")
        analyzer_system = build_analyzer_system_prompt(rules=assets.rules)
        analyzer_user = build_analyzer_user_prompt(
            architecture=architecture,
            isa_docs=isa_docs,
            assembly_tests=assembly_tests,
            generators=generators,
            measured_coverage_report=baseline_coverage_text,
        )
        analysis = call_llm(
            client=client,
            model=args.model,
            system=analyzer_system,
            user=analyzer_user,
            timeout_s=args.llm_timeout_s,
        )
        dump_named_prompt_bundle(
            output_dir=args.output_dir,
            bundle_name="analyzer",
            system_prompt=analyzer_system,
            user_prompt=analyzer_user,
            raw_response=analysis,
        )
        print(f"       Received analysis ({len(analysis)} chars)")

    # Persist the analysis we are going to drive the loop with, so each slot's
    # prompts dumped under used_prompts/test_<slot>/ can be cross-read against
    # the exact coverage analysis text they were built from.
    (args.output_dir / "analysis.md").write_text(analysis, encoding="utf-8")

    print(f"\n[4/{n_steps}] Incremental per-test generation loop...")
    loop_result = run_incremental_generation_loop(
        client=client,
        model=args.model,
        analysis=analysis,
        architecture=architecture,
        isa_docs=isa_docs,
        assets=assets,
        output_dir=args.output_dir,
        chipyard_root=chipyard_root,
        vcs_dir_relpath=args.vcs_dir,
        sim_config=DEFAULT_SIM_CONFIG,
        num_tests=args.num_tests,
        max_fix_attempts=args.max_fix_attempts,
        run_timeout_s=args.run_timeout_s,
        llm_timeout_s=args.llm_timeout_s,
        baseline_coverage_metrics=baseline_metrics,
        coverage_threshold=args.coverage_threshold,
        coverage_gating_enabled=True,
    )
    accepted_tests = loop_result["accepted_tests"]
    loop_summary = loop_result["summary"]

    if not accepted_tests:
        print(
            "\nNo tests were accepted across any slot. Skipping post-generation "
            "coverage aggregation."
        )
        print("\nCoverage outputs:")
        print(f"  baseline summary: {baseline_coverage_path}")
        print(f"  baseline analysis: {baseline_coverage['analysis_path']}")
        print(f"  incremental summary: {loop_result['summary_path']}")
        return 1

    print(
        f"\n[5/{n_steps}] Re-running all {len(accepted_tests)} accepted test(s) under "
        "GEN_COVERAGE=1 and aggregating post-generation coverage..."
    )
    sim_summary = run_vcs_binaries(
        chipyard_root=chipyard_root,
        tests=accepted_tests,
        output_dir=args.output_dir,
        sim_config=DEFAULT_SIM_CONFIG,
        vcs_dir_relpath=args.vcs_dir,
        gen_coverage=True,
        coverage_threshold=args.coverage_threshold,
        coverage_report_dir_name="post_generation_cov_overall",
        coverage_analysis_filename="post_generation_coverage_analysis.txt",
        coverage_log_prefix="post_generation_coverage",
        run_timeout_s=args.run_timeout_s,
        execute=True,
    )
    print("\nVCS outputs:")
    print(f"  results: {sim_summary['results_path']}")
    print(f"  summary: {sim_summary['summary_md']}")
    print(f"  coverage status: {sim_summary['coverage'].get('status')}")
    if "urg_report_dir" in sim_summary["coverage"]:
        print(f"  coverage report: {sim_summary['coverage']['urg_report_dir']}")
    if "analysis_path" in sim_summary["coverage"]:
        print(f"  coverage analysis: {sim_summary['coverage']['analysis_path']}")

    # A previously-accepted test re-running under GEN_COVERAGE=1 should still
    # pass; if it regresses under coverage instrumentation we want to know.
    regressed = [t for t in sim_summary["tests"] if t.get("status") == "failed"]
    if regressed:
        print(
            "\nWARNING: the following accepted test(s) regressed under "
            "GEN_COVERAGE=1 re-run: "
            + ", ".join(str(t.get("name")) for t in regressed)
        )

    print(f"\n[6/{n_steps}] Writing summary...")
    print(
        f"       Slots attempted          : {loop_summary['requested']}"
    )
    print(
        f"       Accepted                 : {loop_summary['accepted_count']}"
    )
    print(
        f"       Rejected (no coverage)   : {loop_summary.get('rejected_no_coverage_count', 0)}"
    )
    print(
        f"       Skipped (other failures) : {loop_summary['skipped_count']}"
    )
    print("\nRolling coverage progression (baseline -> after loop):")
    print(
        format_coverage_delta_for_log(
            loop_summary.get("baseline_coverage_metrics"),
            loop_summary.get("current_coverage_metrics"),
        )
    )
    print("\nCoverage outputs:")
    print(f"  baseline summary: {baseline_coverage_path}")
    print(f"  baseline analysis: {baseline_coverage['analysis_path']}")
    print(f"  incremental summary: {loop_result['summary_path']}")
    print(f"  post-generation analysis: {sim_summary['coverage'].get('analysis_path')}")

    # Return nonzero if nothing was accepted or if coverage aggregation failed;
    # regressed-under-coverage tests only warn (they already passed once).
    if loop_summary["accepted_count"] == 0:
        return 1
    if sim_summary["coverage"].get("status") == "failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
