"""Which baremetal tests each configuration can run (run_tests.py --config).

A test that uses a unit the configuration lacks, or has the line `# @EXPECT_ILLEGAL`, must stop
with an illegal instruction; every other test must pass.
"""
import re
from pathlib import Path

ASSEMBLY = Path(__file__).resolve().parents[2] / "baremetal" / "assembly"
_MXU0 = re.compile(r"^\s*(?:\w+:\s*)?v\w*(?:\.\w+)*\.mxu0\b", re.I | re.M)
_NV = re.compile(r"^\s*(?:\w+:\s*)?(?:v\w*(?:\.\w+)*\.nv\b|vload\.scale\b|vstore\.scale\b)", re.I | re.M)
_ILLEGAL = re.compile(r"^\s*#\s*@EXPECT_ILLEGAL\b", re.M)
UNITS = {"default": {"mxu0"}, "ipt": set(), "ipt-nvfp4": {"nv"}}


def needs(test: str) -> set:
    src = "\n".join(line.split("#", 1)[0] for line in (ASSEMBLY / f"{test}.S").read_text().splitlines())
    return ({"mxu0"} if _MXU0.search(src) else set()) | ({"nv"} if _NV.search(src) else set())


def expected(test: str, config: str) -> str:
    if _ILLEGAL.search((ASSEMBLY / f"{test}.S").read_text()):
        return "ILLEGAL"
    return "PASS" if needs(test) <= UNITS[config] else "ILLEGAL"
