import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _to_float_or_none(s: str) -> Optional[float]:
    s = s.strip()
    if not s or s == "--":
        return None
    try:
        return float(s)
    except ValueError:
        return None


# URG modinfo/dashboard: SCORE LINE COND TOGGLE FSM BRANCH [ASSERT] — ASSERT was
# added in newer URG; instance table rows append NAME after ASSERT.
_URG_METRIC = r"(?:--|[+-]?[\d.]+)"
_URG_SCORE_ROW_RE = re.compile(
    rf"^\s*(?P<sc>{_URG_METRIC})\s+"
    rf"(?P<ln>{_URG_METRIC})\s+"
    rf"(?P<cd>{_URG_METRIC})\s+"
    rf"(?P<tg>{_URG_METRIC})\s+"
    rf"(?P<fs>{_URG_METRIC})\s+"
    rf"(?P<br>{_URG_METRIC})"
    rf"(?:\s+(?P<as>{_URG_METRIC}))?\s*$"
)
_URG_INSTANCE_TABLE_ROW_RE = re.compile(
    rf"^\s*(?:{_URG_METRIC})\s+(?:{_URG_METRIC})\s+(?:{_URG_METRIC})\s+(?:{_URG_METRIC})\s+"
    rf"(?:{_URG_METRIC})\s+(?:{_URG_METRIC})\s+(?:{_URG_METRIC})\s+"
    rf"(?P<name>\S+(?:\.\S+)+)\s*$"
)


class CoverageAnalyzer:
    """Parse URG coverage-overall reports and produce a coverage analysis text."""

    def __init__(self, coverage_overall_dir: Path, threshold: float = 90.0):
        self.coverage_overall_dir = Path(coverage_overall_dir)
        self.threshold = threshold

    def analyze(self) -> str:
        overall = self._parse_dashboard_summary()
        module_gaps = self._parse_module_gaps()
        low_cov = {
            name: data
            for name, data in module_gaps.items()
            if (data.get("score") or 100.0) < self.threshold and name != "svsimTestbench"
        }
        return self._assemble_analysis(overall, module_gaps, low_cov)

    def _parse_dashboard_summary(self) -> Dict[str, Optional[float]]:
        dashboard_path = self.coverage_overall_dir / "dashboard.txt"
        if not dashboard_path.exists():
            return {
                "score": None,
                "line": None,
                "cond": None,
                "toggle": None,
                "fsm": None,
                "branch": None,
            }

        text = dashboard_path.read_text(errors="ignore")
        pat = re.compile(
            r"Total Coverage Summary\s*\n"
            r"(?P<header>[A-Z0-9_\s]+?)\n"
            r"(?P<row>\s*[-+]?\d+(?:\.\d+)?(?:\s+[-+]?\d+(?:\.\d+)?){5})",
            re.IGNORECASE,
        )
        m = pat.search(text)
        if not m:
            return {
                "score": None,
                "line": None,
                "cond": None,
                "toggle": None,
                "fsm": None,
                "branch": None,
            }

        header = m.group("header").strip()
        row = m.group("row").strip()
        cols = re.split(r"\s+", header.upper())
        vals = re.split(r"\s+", row)
        if len(cols) < 6 or len(vals) < 6:
            return {
                "score": None,
                "line": None,
                "cond": None,
                "toggle": None,
                "fsm": None,
                "branch": None,
            }

        mapping: Dict[str, float] = {}
        for c, v in zip(cols[:6], vals[:6]):
            try:
                mapping[c] = float(v)
            except ValueError:
                mapping[c] = None

        return {
            "score": mapping.get("SCORE"),
            "line": mapping.get("LINE"),
            "cond": mapping.get("COND"),
            "toggle": mapping.get("TOGGLE"),
            "fsm": mapping.get("FSM"),
            "branch": mapping.get("BRANCH"),
        }

    def _parse_module_gaps(self) -> Dict[str, dict]:
        modinfo_path = self.coverage_overall_dir / "modinfo.txt"
        if not modinfo_path.exists():
            return {}

        lines = modinfo_path.read_text(errors="ignore").splitlines()
        n = len(lines)

        boundaries: List[Tuple[int, str]] = []
        i = 0
        while i < n - 2:
            if (
                lines[i].startswith("=" * 20)
                and lines[i + 2].startswith("=" * 20)
                and lines[i + 1].strip()
            ):
                boundaries.append((i + 3, lines[i + 1].strip()))
                i += 3
            else:
                i += 1

        modules: Dict[str, dict] = {}
        for idx, (content_start, header) in enumerate(boundaries):
            m = re.match(r"^Module\s*:\s*(\S+)$", header)
            if not m:
                continue
            mod_name = m.group(1)
            content_end = boundaries[idx + 1][0] - 3 if idx + 1 < len(boundaries) else n
            section_text = "\n".join(lines[content_start:content_end])
            modules[mod_name] = self._parse_module_section(mod_name, section_text)

        return modules

    def _parse_module_section(self, mod_name: str, text: str) -> dict:
        data: Dict[str, Any] = {
            "name": mod_name,
            "score": None,
            "line": None,
            "cond": None,
            "toggle": None,
            "fsm": None,
            "branch": None,
            "instance_count": 0,
            "uncovered_conds": [],
            "untoggled_signals": [],
            "uncovered_branches": [],
        }

        section_lines = text.splitlines()
        for line in section_lines[:25]:
            sm = _URG_SCORE_ROW_RE.match(line)
            if sm:
                data["score"] = _to_float_or_none(sm.group("sc"))
                data["line"] = _to_float_or_none(sm.group("ln"))
                data["cond"] = _to_float_or_none(sm.group("cd"))
                data["toggle"] = _to_float_or_none(sm.group("tg"))
                data["fsm"] = _to_float_or_none(sm.group("fs"))
                data["branch"] = _to_float_or_none(sm.group("br"))
                break

        data["instance_count"] = sum(
            1 for ln in section_lines if _URG_INSTANCE_TABLE_ROW_RE.match(ln)
        )

        subsections = re.split(r"\n-{40,}\n", text)
        for sub in subsections:
            s = sub.strip()
            if f"Cond Coverage for Module : {mod_name}" in s:
                data["uncovered_conds"] = self._extract_uncovered_conditions(s)
            elif f"Toggle Coverage for Module : {mod_name}" in s:
                data["untoggled_signals"] = self._extract_untoggled_signals(s)
            elif f"Branch Coverage for Module : {mod_name}" in s:
                data["uncovered_branches"] = self._extract_uncovered_branches(s)

        return data

    @staticmethod
    def _extract_uncovered_conditions(text: str) -> List[dict]:
        uncovered: List[dict] = []
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            s = lines[i].strip()
            expr_m = re.match(r"^(?:SUB-)?EXPRESSION\s+(.+)$", s)
            if expr_m:
                expr = expr_m.group(1)
                line_no = None
                for j in range(max(0, i - 3), i):
                    lm = re.match(r"^\s*LINE\s+(\d+)", lines[j])
                    if lm:
                        line_no = int(lm.group(1))

                not_covered: List[str] = []
                for j in range(i + 1, min(i + 50, len(lines))):
                    sj = lines[j].strip()
                    if re.match(r"^LINE\s+\d+", sj):
                        break
                    if sj.startswith(("EXPRESSION", "SUB-EXPRESSION")):
                        break
                    if re.match(r"^-{40,}$", sj) or sj.startswith("==="):
                        break
                    if "Not Covered" in sj:
                        combo = sj.rsplit("Not Covered", 1)[0].strip()
                        not_covered.append(combo)

                if not_covered:
                    uncovered.append(
                        {
                            "line": line_no,
                            "expression": expr[:250],
                            "not_covered": not_covered,
                        }
                    )
            i += 1
        return uncovered

    @staticmethod
    def _extract_untoggled_signals(text: str) -> List[dict]:
        untoggled: List[dict] = []
        in_details = False
        detail_type: Optional[str] = None

        for line in text.splitlines():
            s = line.strip()
            if "Port Details" in s:
                in_details, detail_type = True, "port"
                continue
            if "Signal Details" in s:
                in_details, detail_type = True, "signal"
                continue
            if s.startswith("Toggle Toggle") or not s:
                continue
            if s.startswith(("---", "===", "Cond", "Branch", "Line Coverage")):
                in_details = False
                continue
            if not in_details:
                continue

            parts = s.split()
            if len(parts) >= 4 and parts[1] in ("Yes", "No"):
                if parts[1] == "No":
                    direction = parts[-1] if parts[-1] in ("INPUT", "OUTPUT") else None
                    untoggled.append(
                        {
                            "signal": parts[0],
                            "type": detail_type,
                            "direction": direction,
                        }
                    )
        return untoggled

    @staticmethod
    def _extract_uncovered_branches(text: str) -> List[dict]:
        uncovered: List[dict] = []
        branch_line: Optional[int] = None
        branch_type: Optional[str] = None

        for line in text.splitlines():
            s = line.strip()
            bm = re.match(r"^(TERNARY|IF|CASE)\s+(\d+)\s+", s)
            if bm:
                branch_type = bm.group(1)
                branch_line = int(bm.group(2))
            if "Not Covered" in s:
                combo = s.rsplit("Not Covered", 1)[0].strip()
                uncovered.append({"line": branch_line, "type": branch_type, "combo": combo})
        return uncovered

    @staticmethod
    def _normalize_lane_indices(s: str) -> str:
        return re.sub(r"_(\d+)", "_N", s)

    def _dedup_conditions(self, conds: List[dict]) -> List[dict]:
        seen: Dict[str, dict] = {}
        for c in conds:
            key = self._normalize_lane_indices(
                f"{c['expression']}|{'|'.join(c['not_covered'])}"
            )
            if key not in seen:
                seen[key] = {**c, "_count": 1}
            else:
                seen[key]["_count"] += 1
        return list(seen.values())

    def _dedup_toggles(self, toggles: List[dict]) -> List[dict]:
        seen: Dict[str, dict] = {}
        for t in toggles:
            key = self._normalize_lane_indices(t["signal"])
            if key not in seen:
                seen[key] = {**t, "pattern": key, "_count": 1}
            else:
                seen[key]["_count"] += 1
        return list(seen.values())

    def _dedup_branches(self, branches: List[dict]) -> List[dict]:
        seen: Dict[str, dict] = {}
        for b in branches:
            key = self._normalize_lane_indices(f"{b.get('type', '')}|{b.get('combo', '')}")
            if key not in seen:
                seen[key] = {**b, "_count": 1}
            else:
                seen[key]["_count"] += 1
        return list(seen.values())

    def _assemble_analysis(
        self,
        overall: Dict[str, Optional[float]],
        all_modules: Dict[str, dict],
        low_cov: Dict[str, dict],
    ) -> str:
        out: List[str] = []
        fmt = lambda v: f"{v:.2f}%" if v is not None else "--"

        out.append("=" * 70)
        out.append("COVERAGE ANALYSIS")
        out.append("=" * 70)
        out.append("")
        out.append("OVERALL COVERAGE SUMMARY")
        out.append("-" * 70)
        out.append(f"  SCORE : {fmt(overall.get('score'))}")
        out.append(f"  LINE  : {fmt(overall.get('line'))}")
        out.append(f"  COND  : {fmt(overall.get('cond'))}")
        out.append(f"  TOGGLE: {fmt(overall.get('toggle'))}")
        out.append(f"  FSM   : {fmt(overall.get('fsm'))}")
        out.append(f"  BRANCH: {fmt(overall.get('branch'))}")
        out.append("")

        out.append(f"LOW-COVERAGE THRESHOLD: {self.threshold:.1f}%")
        out.append(f"MODULES BELOW THRESHOLD: {len(low_cov)} / {len(all_modules)}")
        out.append("")

        out.append("MODULE SUMMARY (sorted by score, *** = below threshold)")
        out.append("-" * 70)
        sorted_mods = sorted(all_modules.values(), key=lambda m: m.get("score") or 100.0)
        for m in sorted_mods:
            flag = " ***" if m["name"] in low_cov else ""
            fmt_v = lambda v: f"{v:.1f}" if v is not None else "--"
            out.append(
                f"  {m['name']:<50} SCORE:{fmt_v(m['score']):>6}  "
                f"COND:{fmt_v(m['cond']):>6}  TOGGLE:{fmt_v(m['toggle']):>6}  "
                f"BRANCH:{fmt_v(m['branch']):>6}  ({m.get('instance_count', 0)} inst){flag}"
            )

        if low_cov:
            out.append("")
            out.append("DETAILED GAPS FOR LOW-COVERAGE MODULES")
            out.append("-" * 70)
            for name in sorted(low_cov, key=lambda n: low_cov[n].get("score") or 100.0):
                d = low_cov[name]
                out.append("")
                out.append(
                    f"MODULE: {name}  SCORE:{d.get('score')}  COND:{d.get('cond')}  "
                    f"TOGGLE:{d.get('toggle')}  BRANCH:{d.get('branch')}"
                )
                out.append(f"  Instances: {d.get('instance_count', 0)}")

                conds = self._dedup_conditions(d.get("uncovered_conds", []))
                toggles = self._dedup_toggles(d.get("untoggled_signals", []))
                branches = self._dedup_branches(d.get("uncovered_branches", []))

                if conds:
                    out.append("  UNCOVERED CONDITIONS:")
                    for c in conds:
                        cnt = f" (x{c['_count']})" if c["_count"] > 1 else ""
                        out.append(f"    Line {c.get('line', '?')}: {c['expression']}{cnt}")
                        for nc in c["not_covered"]:
                            out.append(f"      -> {{{nc}}} : Not Covered")

                if toggles:
                    out.append("  UNTOGGLED SIGNALS:")
                    for t in toggles:
                        cnt = f" (x{t['_count']})" if t["_count"] > 1 else ""
                        dir_s = f" {t['direction']}" if t.get("direction") else ""
                        sig = t.get("pattern", t["signal"]) if t["_count"] > 1 else t["signal"]
                        out.append(
                            f"    {sig:<30} {(t.get('type') or ''):<8}{dir_s}{cnt}"
                        )

                if branches:
                    out.append("  UNCOVERED BRANCHES:")
                    for b in branches:
                        cnt = f" (x{b['_count']})" if b["_count"] > 1 else ""
                        out.append(
                            f"    Line {b.get('line', '?')} {b.get('type', '')}: "
                            f"{{{b.get('combo', '')}}} -> Not Covered{cnt}"
                        )
        else:
            out.append("")
            out.append("No modules are below the configured threshold.")

        out.append("")
        out.append("=" * 70)
        out.append("END OF COVERAGE ANALYSIS")
        out.append("=" * 70)
        return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze URG coverage-overall reports and generate coverage analysis text."
    )
    parser.add_argument(
        "coverage_overall_dir",
        type=str,
        help="Path to coverage-overall directory (contains dashboard.txt/modinfo.txt).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Coverage threshold for low-coverage modules (default: 90.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="coverage_analysis.txt",
        help="Output file path for generated analysis (default: coverage_analysis.txt).",
    )
    args = parser.parse_args()

    coverage_dir = Path(args.coverage_overall_dir).resolve()
    if not coverage_dir.exists() or not coverage_dir.is_dir():
        raise FileNotFoundError(f"Coverage directory not found or not a directory: {coverage_dir}")

    analyzer = CoverageAnalyzer(coverage_dir, args.threshold)
    analysis = analyzer.analyze()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(analysis)

    print(f"Coverage analysis written to: {output_path}")
    print(f"Size: {len(analysis):,} chars, {analysis.count(chr(10)):,} lines")


if __name__ == "__main__":
    main()
