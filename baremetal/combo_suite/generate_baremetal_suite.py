#!/usr/bin/env python3
"""
Generate Atlas baremetal combination tests in generators/sp26-atlas-acc/baremetal.

Outputs:
  - Generated assembly tests under baremetal/assembly/generated/<tier>/
  - Baremetal C tests in baremetal/tests/atlas_<short_stem>.c
  - Manifest and coverage under baremetal/results/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    while True:
        if (cur / "sims").is_dir() and (cur / "generators").is_dir() and (cur / "tests").is_dir():
            return cur
        if cur.parent == cur:
            raise RuntimeError("Could not locate chipyard repository root.")
        cur = cur.parent


THIS_FILE = Path(__file__).resolve()
COMBO_ROOT = THIS_FILE.parent
BAREMETAL_ROOT = COMBO_ROOT.parent
REPO_ROOT = _find_repo_root(THIS_FILE)

MATRIX_JSON = COMBO_ROOT / "atlas_instruction_matrix.json"
TEMPLATE_FILE = COMBO_ROOT / "templates" / "program_template.S.tpl"
ASM_OUT_ROOT = BAREMETAL_ROOT / "assembly" / "generated"
TESTS_DIR = BAREMETAL_ROOT / "tests"
RESULTS_DIR = BAREMETAL_ROOT / "results"
ASSEMBLER = BAREMETAL_ROOT / "assembler.py"
MANIFEST_JSON = RESULTS_DIR / "atlas_combo_suite_manifest.json"
COVERAGE_JSON = RESULTS_DIR / "atlas_combo_coverage_report.json"

INTER_ENGINE_DELAY = 80
DMA_BASE_ADDR = 0x90000000

FAMILY_SHORT = {
    "scalar_alu": "salu",
    "scalar_control": "sctl",
    "tensor_memory": "tmem",
    "vpu_compute": "vpu",
    "xlu": "xlu",
    "mxu": "mxu",
    "dma": "dma",
}


def _signature_for(name: str) -> int:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return int(digest[:4], 16)


def _join(lines: List[str]) -> str:
    return "\n".join(f"    {line}" if line and not line.endswith(":") else line for line in lines)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _parse_index_from_name(name: str) -> int:
    m = re.search(r"_(\d+)$", name)
    return int(m.group(1)) if m else 0


def _snippet_scalar_alu(_: str) -> List[str]:
    return [
        "ADDI  x1, x0, 21",
        "ADDI  x2, x0, 9",
        "ADD   x3, x1, x2",
        "ADDI  x4, x0, 30",
        "BNE   x3, x4, fail",
        "NOP",
        "NOP",
        "XOR   x5, x3, x2",
        "ADDI  x6, x0, 23",
        "BNE   x5, x6, fail",
        "NOP",
        "NOP",
    ]


def _snippet_scalar_control(tag: str) -> List[str]:
    return [
        "ADDI  x10, x0, 0",
        "ADDI  x11, x0, 0",
        f"BEQ   x10, x11, ctrl_taken_{tag}",
        "NOP",
        "NOP",
        "ADDI  x28, x28, 1",
        "JAL   x0, fail",
        "NOP",
        "NOP",
        f"ctrl_taken_{tag}:",
        "ADDI  x10, x10, 1",
    ]


def _snippet_tensor_memory(_: str) -> List[str]:
    return [
        "VLI.ALL  0, 0x33",
        "DELAY 40",
        "ADDI  x6, x0, 0",
        "VSTORE 0, x6, 0",
        "DELAY 40",
        "VLOAD  1, x6, 0",
        "DELAY 40",
        "VMOV   2, 1",
        "DELAY 40",
    ]


def _snippet_vpu_compute(_: str) -> List[str]:
    return [
        "VLI.ALL      3, 0x10",
        "DELAY 40",
        "VMOV         4, 3",
        "DELAY 40",
        "VADD.BF16    5, 4, 4",
        "DELAY 40",
        "VSUB.BF16    6, 5, 4",
        "DELAY 40",
        "VMAX.BF16    7, 6, 4",
        "DELAY 40",
        "VREDSUM.BF16 8, 7",
        "DELAY 40",
    ]


def _snippet_xlu(_: str) -> List[str]:
    return [
        "VTRPOSE.XLU  9, 8",
        "DELAY 80",
        "VREDMAX.XLU  10, 9",
        "DELAY 80",
        "VREDSUM.XLU  11, 9",
        "DELAY 80",
    ]


def _snippet_mxu(_: str) -> List[str]:
    return [
        "VLI.ALL                12, 0x11",
        "VLI.ALL                13, 0x22",
        "VMATPUSH.W.MXU0        0, 12",
        "DELAY                  40",
        "VMATMUL.MXU0           0, 13, 0",
        "DELAY                  120",
        "VMATPOP.BF16.MXU0      14, 0",
        "DELAY                  40",
        "VMATPUSH.W.MXU1        0, 12",
        "DELAY                  40",
        "VMATMUL.MXU1           0, 13, 0",
        "DELAY                  120",
        "VMATPOP.BF16.MXU1      16, 0",
    ]


def _snippet_dma(_: str) -> List[str]:
    return [
        f"LI    x5, 0x{DMA_BASE_ADDR:08X}",
        "DMA.CONFIG  x5, 0",
        "ADDI  x6, x0, 0",
        "ADDI  x1, x0, 0",
        "ADDI  x2, x0, 128",
        "DMA.LOAD   x6, x1, x2, 0",
        "DMA.WAIT   0",
        "LI    x3, 1024",
        "DMA.STORE  x3, x6, x2, 1",
        "DMA.WAIT   1",
    ]


FAMILY_SNIPPETS: Dict[str, Callable[[str], List[str]]] = {
    "scalar_alu": _snippet_scalar_alu,
    "scalar_control": _snippet_scalar_control,
    "tensor_memory": _snippet_tensor_memory,
    "vpu_compute": _snippet_vpu_compute,
    "xlu": _snippet_xlu,
    "mxu": _snippet_mxu,
    "dma": _snippet_dma,
}


def _common_tail() -> List[str]:
    return [
        "ADDI  x20, x0, 7",
        "ADDI  x21, x0, 7",
        "BNE   x20, x21, fail",
        "NOP",
        "NOP",
        "JAL   x0, pass",
        "NOP",
        "NOP",
    ]


def _render_program(template: str, scenario: dict, body_lines: List[str]) -> str:
    return template.format(
        long_name=scenario["long_name"],
        short_name=scenario["short_name"],
        tier=scenario["tier"],
        family_a=scenario["family_a"],
        family_b=scenario["family_b"],
        risk_class=scenario["risk_class"],
        seed=scenario["seed"],
        timeout=scenario["timeout"],
        signature=_signature_for(scenario["long_name"]),
        body=_join(body_lines),
    )


def _make_short_name(scenario: dict) -> str:
    idx = _parse_index_from_name(scenario["long_name"])
    fa = FAMILY_SHORT[scenario["family_a"]]
    fb = FAMILY_SHORT[scenario["family_b"]]

    if scenario["tier"] == "tier0":
        return f"smk_{fa}_{fb}_{idx:02d}"
    if scenario["tier"] == "tier1":
        return f"pw_{fa}_{fb}_{idx:02d}"
    if scenario["tier"] == "tier3":
        return f"str_{idx:03d}_{fa}_{fb}"

    risk_map = {
        "risk_dma_compute_store_wait": "rsk_dma_order",
        "risk_mxu0_mxu1_parity_path": "rsk_mxu_parity",
        "risk_transpose_reduce_axis": "rsk_xlu_axis",
        "risk_mxu_accumulate_chain": "rsk_mxu_acc",
        "risk_branch_delay_with_side_effects": "rsk_delay_slot",
        "risk_dma_busy_channel_probe": "rsk_dma_busy",
    }
    return risk_map.get(scenario["long_name"], f"rsk_{idx:02d}_{fa}_{fb}")


def _build_pairwise_scenarios(matrix: dict) -> List[dict]:
    scenarios: List[dict] = []
    for idx, pair in enumerate(matrix["pairwise_family_matrix"]):
        fam_a, fam_b = pair
        scenarios.append(
            {
                "tier": "tier1",
                "long_name": f"pair_{fam_a}__{fam_b}_{idx:02d}",
                "family_a": fam_a,
                "family_b": fam_b,
                "risk_class": "pairwise",
                "seed": idx,
                "timeout": 20000,
            }
        )
    return scenarios


def _build_tier0_scenarios() -> List[dict]:
    smoke_pairs = [
        ("scalar_alu", "scalar_control"),
        ("scalar_alu", "tensor_memory"),
        ("scalar_alu", "vpu_compute"),
        ("scalar_alu", "xlu"),
        ("scalar_alu", "mxu"),
        ("scalar_alu", "dma"),
        ("tensor_memory", "vpu_compute"),
        ("tensor_memory", "mxu"),
        ("vpu_compute", "xlu"),
        ("mxu", "dma"),
        ("scalar_control", "dma"),
        ("scalar_control", "mxu"),
    ]
    scenarios: List[dict] = []
    for idx, (fam_a, fam_b) in enumerate(smoke_pairs):
        scenarios.append(
            {
                "tier": "tier0",
                "long_name": f"smoke_{fam_a}__{fam_b}_{idx:02d}",
                "family_a": fam_a,
                "family_b": fam_b,
                "risk_class": "smoke",
                "seed": idx,
                "timeout": 10000,
            }
        )
    return scenarios


def _build_tier2_scenarios() -> List[dict]:
    return [
        {"tier": "tier2", "long_name": "risk_dma_compute_store_wait", "family_a": "dma", "family_b": "vpu_compute", "risk_class": "ordering", "seed": 200, "timeout": 40000},
        {"tier": "tier2", "long_name": "risk_mxu0_mxu1_parity_path", "family_a": "mxu", "family_b": "mxu", "risk_class": "cross_unit_parity", "seed": 201, "timeout": 40000},
        {"tier": "tier2", "long_name": "risk_transpose_reduce_axis", "family_a": "xlu", "family_b": "vpu_compute", "risk_class": "axis_semantics", "seed": 202, "timeout": 30000},
        {"tier": "tier2", "long_name": "risk_mxu_accumulate_chain", "family_a": "mxu", "family_b": "tensor_memory", "risk_class": "accumulator_lifecycle", "seed": 203, "timeout": 40000},
        {"tier": "tier2", "long_name": "risk_branch_delay_with_side_effects", "family_a": "scalar_control", "family_b": "tensor_memory", "risk_class": "delay_slots", "seed": 204, "timeout": 20000},
        {"tier": "tier2", "long_name": "risk_dma_busy_channel_probe", "family_a": "dma", "family_b": "scalar_control", "risk_class": "busy_channel", "seed": 205, "timeout": 30000}
    ]


def _build_tier3_scenarios(seed: int, count: int) -> List[dict]:
    families = sorted(FAMILY_SNIPPETS.keys())
    rng = random.Random(seed)
    scenarios: List[dict] = []
    for idx in range(count):
        fam_a = rng.choice(families)
        fam_b = rng.choice(families)
        scenarios.append(
            {
                "tier": "tier3",
                "long_name": f"stress_{fam_a}__{fam_b}_{idx:03d}",
                "family_a": fam_a,
                "family_b": fam_b,
                "risk_class": "stress_random",
                "seed": seed + idx,
                "timeout": 50000,
            }
        )
    return scenarios


def _scenario_body(scenario: dict) -> List[str]:
    fam_a = scenario["family_a"]
    fam_b = scenario["family_b"]
    tag = scenario["short_name"]

    if scenario["long_name"] == "risk_dma_compute_store_wait":
        return _snippet_dma(tag) + _snippet_tensor_memory(tag) + _snippet_vpu_compute(tag) + _snippet_dma(tag + "_2") + _common_tail()
    if scenario["long_name"] == "risk_mxu0_mxu1_parity_path":
        return _snippet_tensor_memory(tag) + _snippet_mxu(tag) + _common_tail()
    if scenario["long_name"] == "risk_transpose_reduce_axis":
        return _snippet_tensor_memory(tag) + _snippet_vpu_compute(tag) + _snippet_xlu(tag) + _common_tail()
    if scenario["long_name"] == "risk_mxu_accumulate_chain":
        return _snippet_tensor_memory(tag) + [
            "VMATPUSH.W.MXU0        0, 12",
            "DELAY                  40",
            "VMATMUL.ACC.MXU0       0, 13, 0",
            "DELAY                  120",
            "VMATMUL.ACC.MXU0       0, 13, 0",
            "DELAY                  120",
            "VMATPOP.BF16.MXU0      18, 0",
        ] + _common_tail()
    if scenario["long_name"] == "risk_branch_delay_with_side_effects":
        return [
            "ADDI  x1, x0, 1",
            "ADDI  x2, x0, 1",
            "BEQ   x1, x2, risk_delay_taken",
            "VLI.ALL  0, 0x55",
            "VLI.ALL  1, 0xAA",
            "ADDI  x28, x28, 1",
            "JAL   x0, fail",
            "NOP",
            "NOP",
            "risk_delay_taken:",
            "ADDI  x3, x0, 9",
            "ADDI  x4, x0, 9",
            "BNE   x3, x4, fail",
            "NOP",
            "NOP",
        ] + _common_tail()
    if scenario["long_name"] == "risk_dma_busy_channel_probe":
        return [
            f"LI    x5, 0x{DMA_BASE_ADDR:08X}",
            "DMA.CONFIG  x5, 0",
            "ADDI  x6, x0, 0",
            "ADDI  x1, x0, 0",
            "ADDI  x2, x0, 128",
            "DMA.LOAD   x6, x1, x2, 0",
            "DMA.LOAD   x6, x1, x2, 0",
            "DMA.WAIT   0",
            "DMA.WAIT   0",
        ] + _common_tail()

    body = []
    body.extend(FAMILY_SNIPPETS[fam_a](tag + "_a"))
    body.append(f"DELAY {INTER_ENGINE_DELAY}")
    body.extend(FAMILY_SNIPPETS[fam_b](tag + "_b"))
    body.extend(_common_tail())
    return body


def _clean_previous_outputs() -> None:
    for path in ASM_OUT_ROOT.glob("**/*.S"):
        path.unlink()
    for pattern in ("atlas_smk_*.c", "atlas_pw_*.c", "atlas_rsk_*.c", "atlas_str_*.c"):
        for path in TESTS_DIR.glob(pattern):
            path.unlink()


def _emit_test(template: str, scenario: dict, dry_run: bool) -> dict:
    asm_path = ASM_OUT_ROOT / scenario["tier"] / f"{scenario['short_name']}.S"
    c_name = f"atlas_{scenario['short_name']}.c"
    c_path = TESTS_DIR / c_name

    asm_text = _render_program(template, scenario, _scenario_body(scenario))
    _write_text(asm_path, asm_text)

    cmd = [sys.executable, str(ASSEMBLER), str(asm_path), "--out-c", str(c_path)]
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    return {
        "name": scenario["short_name"],
        "long_name": scenario["long_name"],
        "tier": scenario["tier"],
        "familyA": scenario["family_a"],
        "familyB": scenario["family_b"],
        "risk_class": scenario["risk_class"],
        "seed": scenario["seed"],
        "assembly": str(asm_path.relative_to(REPO_ROOT)),
        "c_file": str(c_path.relative_to(REPO_ROOT)),
        "cmake_target": f"atlas_{scenario['short_name']}",
    }


def _coverage_report(matrix: dict, tests: List[dict]) -> dict:
    planned = {tuple(pair) for pair in matrix["pairwise_family_matrix"]}
    covered = set()
    by_tier = {"tier0": 0, "tier1": 0, "tier2": 0, "tier3": 0}
    for test in tests:
        by_tier[test["tier"]] += 1
        pair = (test["familyA"], test["familyB"])
        if pair in planned:
            covered.add(pair)

    unsupported = []
    for family in matrix["families"]:
        for op in family["ops"]:
            if not op["supported_by_py_assembler"]:
                unsupported.append(
                    {
                        "family": family["name"],
                        "operation": op["name"],
                        "assembler_mnemonic": op["assembler_mnemonic"],
                    }
                )

    pct = (len(covered) / len(planned) * 100.0) if planned else 0.0
    return {
        "planned_pairwise_edges": sorted([list(x) for x in planned]),
        "covered_pairwise_edges": sorted([list(x) for x in covered]),
        "pairwise_coverage_pct": round(pct, 2),
        "total_tests": len(tests),
        "tests_per_tier": by_tier,
        "unsupported_operations": unsupported,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Atlas combo suite under baremetal/.")
    parser.add_argument("--stress-count", type=int, default=12, help="Number of tier3 stress tests.")
    parser.add_argument("--seed", type=int, default=1337, help="Seed for tier3 randomization.")
    parser.add_argument("--dry-run", action="store_true", help="Skip C emission; write assembly + metadata only.")
    args = parser.parse_args()

    matrix = json.loads(MATRIX_JSON.read_text(encoding="utf-8"))
    template = TEMPLATE_FILE.read_text(encoding="utf-8")

    _clean_previous_outputs()

    scenarios: List[dict] = []
    scenarios.extend(_build_tier0_scenarios())
    scenarios.extend(_build_pairwise_scenarios(matrix))
    scenarios.extend(_build_tier2_scenarios())
    scenarios.extend(_build_tier3_scenarios(args.seed, args.stress_count))
    for scenario in scenarios:
        scenario["short_name"] = _make_short_name(scenario)

    seen = set()
    collisions = []
    for scenario in scenarios:
        short = scenario["short_name"]
        if short in seen:
            collisions.append(short)
        seen.add(short)
    if collisions:
        raise RuntimeError(f"Generated non-unique short names: {sorted(set(collisions))}")

    tests = [_emit_test(template, scenario, args.dry_run) for scenario in scenarios]

    manifest = {
        "generator": str(Path(__file__).relative_to(REPO_ROOT)),
        "matrix": str(MATRIX_JSON.relative_to(REPO_ROOT)),
        "tiers": ["tier0", "tier1", "tier2", "tier3"],
        "tests": tests,
    }
    _write_text(MANIFEST_JSON, json.dumps(manifest, indent=2) + "\n")
    _write_text(COVERAGE_JSON, json.dumps(_coverage_report(matrix, tests), indent=2) + "\n")

    print(f"Generated {len(tests)} tests under {BAREMETAL_ROOT}")
    print(f"  manifest: {MANIFEST_JSON}")
    print(f"  coverage: {COVERAGE_JSON}")


if __name__ == "__main__":
    main()
