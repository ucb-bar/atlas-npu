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
COVERAGE_DIRNAME = "coverage"
COVERAGE_ANALYZER_PATH = BAREMETAL_DIR / "coverage_analyzer.py"
RUN_ASM_TESTS_PATH = BAREMETAL_DIR / "run_asm_tests.sh"
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
    """Return the single-line ``def`` signature and its docstring (if any)."""

    pattern = re.compile(rf"^def {re.escape(func_name)}\(.*?\):", re.MULTILINE | re.DOTALL)
    match = pattern.search(text)
    if not match:
        return ""

    sig = match.group(0)
    rest = text[match.end():]
    doc_lines: list[str] = []
    lines = rest.splitlines()
    if lines and lines[0].strip().startswith(('"""', "'''")):
        quote = '"""' if '"""' in lines[0] else "'''"
        first = lines[0].strip()
        if first.count(quote) >= 2:
            doc_lines.append("    " + first)
        else:
            doc_lines.append("    " + first)
            for ln in lines[1:]:
                doc_lines.append(ln)
                if quote in ln:
                    break
    body = "\n".join(doc_lines).rstrip()
    return (sig + "\n" + body).rstrip() if body else sig


def _load_generator_helper_reference() -> str:
    path = BAREMETAL_DIR / "generators" / "gen_utils.py"
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
        snippets.append("# High-level datatype and reference-model helpers (signatures only):\n"
                        + "\n\n".join(signature_snippets))

    if not snippets:
        return ""

    body = "\n\n".join(snippets)
    canonical_imports = textwrap.dedent(
        """\
        # Canonical imports for any generator:
        import os
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from gen_utils import (
            # Packing/emit
            emit_test_data,
            preloads_from_words_packed,
            checks_from_words_packed,
            pack_words_into_beats,
            make_preload_any,
            make_check_any,
            # Datatype encoding (use these — do NOT invent lookup tables)
            float_to_bf16_bits,
            bf16_bits_to_float,
            pack_bf16_pair,
            float_to_fp8_e4m3_bits,
            fp8_e4m3_bits_to_float,
            pack_fp8x4,
            # Matrix helpers (use these — do NOT hand-roll row/col shifting)
            matrix_to_bf16_words,
            matrix_to_fp8_words,
            quantize_bf16,
            quantize_fp8,
            # Reference models (use these — do NOT simulate matmul by hand)
            bf16_matmul_reference,
            fp8_matmul_reference,
            # Random inputs
            rand_matrix,
            rand_matrix_fp8_safe,
        )
        """
    ).rstrip()
    end_to_end_example = textwrap.dedent(
        """\
        # End-to-end skeleton for a BF16 / FP8 MXU test (matches the checked-in
        # `gen_mxu0_single_output_tile_bf16.py` pattern). Follow this shape
        # exactly when the assembly stages tiles via DMA→VLOAD→MXU→VSTORE→DMA.
        #
        # DRAM offsets in the assembly are byte offsets from @DRAM_BASE.
        # Generators emit `word_offset` values in DMA-width BEATS (1 beat = 32 B).
        # So: beat_offset = byte_offset // 32. Never conflate the two.
        import numpy as np
        TILE = 32
        A = rand_matrix_fp8_safe(TILE, TILE, seed=42)         # architectural floats
        W = rand_matrix_fp8_safe(TILE, TILE, seed=43)
        A_q = quantize_fp8(A).astype(np.float32)              # round-trip through FP8
        W_q = quantize_fp8(W).astype(np.float32)
        C   = fp8_matmul_reference(A_q, W_q)                  # BF16-accumulated golden
        # Preload FP8 tiles at DRAM byte offsets 0x0000 and 0x0400.
        preloads  = preloads_from_words_packed(0x0000 // 32, matrix_to_fp8_words(A_q))
        preloads += preloads_from_words_packed(0x0400 // 32, matrix_to_fp8_words(W_q))
        # Check BF16 result halves at DRAM byte offsets 0x0800 / 0x0C00 if the
        # assembly stores the 32x32 result as two 32x16 banks (lo/hi lanes).
        checks  = checks_from_words_packed(0x0800 // 32, matrix_to_bf16_words(C[:, :16]))
        checks += checks_from_words_packed(0x0C00 // 32, matrix_to_bf16_words(C[:, 16:]))
        emit_test_data(preloads, checks, timeout=500000)
        """
    ).rstrip()
    return textwrap.dedent(
        f"""\
        Use the checked-in generator helpers from `generators/gen_utils.py` instead of inventing
        helper signatures, encoding tables, or ad hoc JSON layouts. Match these current APIs
        exactly. If a helper listed below would solve a sub-problem, USE IT. Do not re-implement
        float-to-bf16 rounding, FP8 bit-packing, row/col tile shifting, or matmul reference
        numerics by hand — the helpers below already match the Atlas RTL behavior.

        ```python
        {body}
        ```

        ```python
        {canonical_imports}
        ```

        ```python
        {end_to_end_example}
        ```
        """
    ).strip()


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
    return f"## Generator Helper API\n\n{resolved.strip()}\n"


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
2. **Helper reuse is mandatory.** `generators/gen_utils.py` is the single source of truth
   for BF16/FP8 encoding, matrix packing, and reference compute. If you find yourself
   writing a custom `bf16_word(v)` lookup, a `fp8_byte(v)` dict, or a `(v << 8)` style
   conversion, stop and use the helper instead.
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
    task_section = textwrap.dedent(
        f"""\
        You are an expert hardware verification engineer writing baremetal integration
        tests for the Atlas chip. Generate exactly **{num_tests}** new tests that meaningfully
        advance the coverage profile of the existing suite.

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

        Write each test the way a senior verification engineer would sign it off for
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
{current_suite_section}\
## Strategy Menu

{_format_strategy_menu(optimization_menu)}

## Reference Examples

{_format_example_section(baremetal_test_examples)}

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


def call_llm(client: OpenAI, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""


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


def dump_used_planner_prompts(
    output_dir: Path,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Path]:
    """
    Persist the exact planner prompts that were sent to the LLM.
    """

    prompt_dir = output_dir / USED_PROMPTS_DIRNAME / "planning"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "used_prompts_dir": prompt_dir,
        "planner_system": prompt_dir / "system_prompt.md",
        "planner_user": prompt_dir / "user_prompt.md",
    }
    outputs["planner_system"].write_text(system_prompt, encoding="utf-8")
    outputs["planner_user"].write_text(user_prompt, encoding="utf-8")
    return outputs





def main() -> int:
    run_label = "coverage_loop"

    parser = argparse.ArgumentParser(
        description=(
            "Run the full Atlas coverage-driven generation loop: ensure handwritten "
            "coverage exists, aggregate/analyze RTL coverage, generate new tests, "
            "run them, then aggregate coverage again."
        )
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=8,
        help="Number of tests requested from the planner.",
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
    n_steps = 8

    print("=" * 60)
    print("Atlas Verif Buddy")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output dir: {args.output_dir}")
    print(f"Chipyard root: {chipyard_root}")
    print(f"Found {len(assembly_tests)} assembly tests and {len(generators)} generators")

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
        )
        print(f"       Received analysis ({len(analysis)} chars)")

    print(f"\n[4/{n_steps}] Building planner prompts for test generation...")
    planner_system = build_planner_system_prompt(rules=assets.rules)
    planner_user = build_planner_user_prompt(
        analysis=analysis,
        architecture=architecture,
        isa_docs=isa_docs,
        num_tests=args.num_tests,
        assembly_tests=assembly_tests,
        generators=generators,
        optimization_menu=assets.optimization_menu,
        baremetal_test_examples=assets.baremetal_test_examples,
        rules=assets.rules,
    )
    prompt_outputs = dump_used_planner_prompts(
        output_dir=args.output_dir,
        system_prompt=planner_system,
        user_prompt=planner_user,
    )
    print("       Dumped exact prompts used for planning:")
    for name, path in prompt_outputs.items():
        print(f"       {name}: {path}")

    print(f"\n[5/{n_steps}] Running planner...")
    planner_raw = call_llm(
        client=client,
        model=args.model,
        system=planner_system,
        user=planner_user,
    )
    print(f"       Received planner response ({len(planner_raw)} chars)")

    try:
        planner_results = extract_json_array(planner_raw)
    except ValueError as exc:
        raw_path = args.output_dir / "planner_generate_raw_response.md"
        raw_path.write_text(planner_raw, encoding="utf-8")
        sys.exit(f"ERROR: {exc}\nRaw planner response saved to {raw_path}")

    log_outputs = write_generation_logs(
        output_dir=args.output_dir,
        analysis=analysis,
        planner_raw=planner_raw,
        planner_results=planner_results,
    )
    materialized_outputs = materialize_planner_results(
        output_dir=args.output_dir,
        planner_results=planner_results,
    )
    print(f"\n[6/{n_steps}] Publishing generated tests into baremetal source directories...")
    published_tests = publish_planner_results_to_baremetal(planner_results)
    print(f"       Published {len(published_tests)} test(s) into {ASSEMBLY_DIR} and {GENERATORS_DIR}")

    print(f"\n[7/{n_steps}] Generating goldens, assembling, and building binaries...")
    build_outputs = build_published_baremetal_tests(
        output_dir=args.output_dir,
        published_tests=published_tests,
    )

    print(f"\n[8/{n_steps}] Running generated tests and aggregating post-generation coverage...")
    sim_summary = run_vcs_binaries(
        chipyard_root=chipyard_root,
        tests=published_tests,
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
    if any(t.get("status") == "failed" for t in sim_summary["tests"]):
        return 1
    if sim_summary["coverage"].get("status") == "failed":
        return 1

    print("\nWrote run logs:")
    for name, path in log_outputs.items():
        print(f"  {name}: {path}")
    print("\nMaterialized planner outputs:")
    for name, path in materialized_outputs.items():
        print(f"  {name}: {path}")
    print("\nBuilt baremetal outputs:")
    for name, path in build_outputs.items():
        print(f"  {name}: {path}")
    print("\nCoverage outputs:")
    print(f"  baseline summary: {baseline_coverage_path}")
    print(f"  baseline analysis: {baseline_coverage['analysis_path']}")
    print(f"  post-generation analysis: {sim_summary['coverage'].get('analysis_path')}")
    print(f"\nGenerated {len(planner_results)} planner result entries.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
