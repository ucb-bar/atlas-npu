#!/usr/bin/env python3
"""
Build and optionally run Atlas baremetal combo tests from baremetal/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    while True:
        if (cur / "sims").is_dir() and (cur / "generators").is_dir() and (cur / "tests").is_dir():
            return cur
        if cur.parent == cur:
            raise RuntimeError("Could not locate chipyard repository root.")
        cur = cur.parent


BAREMETAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _find_repo_root(BAREMETAL_ROOT)
RESULTS_DIR = BAREMETAL_ROOT / "results"
MANIFEST = RESULTS_DIR / "atlas_combo_suite_manifest.json"
RESULTS_JSON = RESULTS_DIR / "atlas_combo_run_results.json"
SUMMARY_MD = RESULTS_DIR / "atlas_combo_run_summary.md"
LOG_DIR = RESULTS_DIR / "logs"

FAIL_MARKERS = (
    "*** FAILED ***",
    "Assertion failed",
    "FAIL:",
    "Error:",
    "Fatal:",
)


def _run(
    cmd: List[str],
    cwd: Path,
    execute: bool,
    log_path: Optional[Path] = None,
    timeout_s: Optional[int] = None,
) -> Dict[str, object]:
    print(f"$ (cd {cwd} && {' '.join(cmd)})")
    if not execute:
        return {"executed": False, "rc": None, "output": ""}

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: List[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            output_lines.append(line)
            print(line, end="")
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        output_lines.append(f"\n[TIMEOUT] command exceeded {timeout_s}s and was killed.\n")
        rc = 124

    output = "".join(output_lines)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")

    return {"executed": True, "rc": rc, "output": output}


def _load_manifest() -> Dict:
    if not MANIFEST.exists():
        raise SystemExit(
            f"Missing {MANIFEST}. Run combo_suite/generate_baremetal_suite.py first."
        )
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _selected_tests(manifest: Dict, tiers: List[str], names: List[str]) -> List[Dict]:
    tier_set = set(tiers)
    name_set = set(names)
    selected = [test for test in manifest["tests"] if test["tier"] in tier_set]
    if name_set:
        selected = [test for test in selected if test["name"] in name_set]
    return selected


def _classify_run(rc: int, output: str) -> Dict[str, object]:
    if rc != 0:
        return {"pass": False, "reason": f"nonzero_return_code_{rc}"}
    marker = next((m for m in FAIL_MARKERS if m in output), None)
    if marker is not None:
        return {"pass": False, "reason": f"failure_marker:{marker}"}
    if "PASS:" in output:
        return {"pass": True, "reason": "pass_marker"}
    return {"pass": True, "reason": "zero_return_code_without_failure_marker"}


def _write_markdown_summary(results: Dict[str, object]) -> None:
    tests = results["tests"]
    passed = sum(1 for t in tests if t["status"] == "passed")
    failed = sum(1 for t in tests if t["status"] in ("build_failed", "failed"))
    skipped = sum(1 for t in tests if t["status"] == "dry_run")

    lines = [
        "# Atlas Combo Run Summary",
        "",
        f"- Config: `{results['config']}`",
        f"- Execute mode: `{results['execute']}`",
        f"- Total tests: `{len(tests)}`",
        f"- Passed: `{passed}`",
        f"- Failed: `{failed}`",
        f"- Skipped: `{skipped}`",
        "",
        "| Status | Tier | Test | Build RC | Run RC | Reason |",
        "|---|---|---|---:|---:|---|",
    ]
    for test in tests:
        build_rc = "" if test["build_rc"] is None else str(test["build_rc"])
        run_rc = "" if test["run_rc"] is None else str(test["run_rc"])
        lines.append(
            f"| {test['status']} | {test['tier']} | {test['name']} | {build_rc} | {run_rc} | {test['reason']} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Atlas combo baremetal suite from baremetal/.")
    parser.add_argument("--tiers", default="tier0", help="Comma-separated tiers (tier0,tier1,tier2,tier3).")
    parser.add_argument("--test", default="", help="Comma-separated short test names to run (optional).")
    parser.add_argument("--build-dir", default="generators/sp26-atlas-acc/baremetal/tests/build", help="Build directory relative to repo root.")
    parser.add_argument("--vcs-dir", default="sims/vcs", help="VCS directory relative to repo root.")
    parser.add_argument("--config", default="EE290SimConfig", help="Chipyard config used for run-binary.")
    parser.add_argument("--run-timeout-s", type=int, default=0, help="Timeout per run-binary invocation (0 disables timeout).")
    parser.add_argument("--list", action="store_true", help="List selected tests and exit.")
    parser.add_argument("--execute", action="store_true", help="Run build/sim commands (default is dry-run).")
    args = parser.parse_args()

    tiers = [x.strip() for x in args.tiers.split(",") if x.strip()]
    test_names = [x.strip() for x in args.test.split(",") if x.strip()]
    manifest = _load_manifest()
    tests = _selected_tests(manifest, tiers, test_names)
    if not tests:
        raise SystemExit("No tests selected. Check --tiers/--test values.")

    if args.list:
        for test in tests:
            print(f"{test['name']}\t{test['tier']}\t{test['long_name']}")
        return

    tests_dir = BAREMETAL_ROOT / "tests"
    build_dir = REPO_ROOT / args.build_dir
    vcs_dir = REPO_ROOT / args.vcs_dir

    results: Dict[str, object] = {
        "tiers": tiers,
        "selected_tests": test_names,
        "config": args.config,
        "execute": args.execute,
        "build_dir": str(build_dir.relative_to(REPO_ROOT)),
        "vcs_dir": str(vcs_dir.relative_to(REPO_ROOT)),
        "tests": [],
    }

    cmake_res = _run(
        ["cmake", "-S", ".", "-B", str(build_dir), "-D", "CMAKE_BUILD_TYPE=Debug"],
        tests_dir,
        args.execute,
    )
    cmake_rc = cmake_res["rc"]
    if args.execute and cmake_rc != 0:
        results["cmake_configure_rc"] = cmake_rc
        RESULTS_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(cmake_rc)

    for test in tests:
        target = test["cmake_target"]
        binary = str((build_dir / f"{target}.riscv").resolve())
        log_path = LOG_DIR / f"{target}.log"

        if not args.execute:
            results["tests"].append(
                {
                    "name": test["name"],
                    "long_name": test["long_name"],
                    "tier": test["tier"],
                    "target": target,
                    "build_rc": None,
                    "run_rc": None,
                    "status": "dry_run",
                    "reason": "commands_not_executed",
                    "pass": None,
                    "log": str(log_path.relative_to(REPO_ROOT)),
                }
            )
            continue

        build_res = _run(["cmake", "--build", str(build_dir), "--target", target], tests_dir, True)
        build_rc = build_res["rc"]
        run_rc = None
        status = "build_failed"
        reason = f"build_rc_{build_rc}"
        passed: Optional[bool] = False

        if build_rc == 0:
            run_res = _run(
                ["make", "run-binary", f"BINARY={binary}", f"CONFIG={args.config}", "LOADMEM=1"],
                vcs_dir,
                True,
                log_path=log_path,
                timeout_s=(args.run_timeout_s if args.run_timeout_s > 0 else None),
            )
            run_rc = run_res["rc"]
            run_class = _classify_run(run_rc, run_res["output"])
            passed = bool(run_class["pass"])
            status = "passed" if passed else "failed"
            reason = str(run_class["reason"])

        results["tests"].append(
            {
                "name": test["name"],
                "long_name": test["long_name"],
                "tier": test["tier"],
                "target": target,
                "build_rc": build_rc,
                "run_rc": run_rc,
                "status": status,
                "reason": reason,
                "pass": passed,
                "log": str(log_path.relative_to(REPO_ROOT)),
            }
        )

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _write_markdown_summary(results)

    print(f"Wrote run results to {RESULTS_JSON}")
    print(f"Wrote summary to {SUMMARY_MD}")
    print("\nAtlas combo test summary:")
    for test in results["tests"]:
        print(f"  [{test['status']}] {test['tier']} :: {test['name']} ({test['reason']})")

    if args.execute and any(t["status"] in ("build_failed", "failed") for t in results["tests"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
