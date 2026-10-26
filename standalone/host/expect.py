"""Which baremetal tests each configuration can run (run_tests.py --config).

A test that uses a unit the configuration lacks, or has the line `# @EXPECT_ILLEGAL`, must stop
with an illegal instruction; every other test must pass. A bundle with its assembly beside it
(BUNDLE.S, as hazfuzz writes it) is judged by that assembly, a baremetal test by its source in
baremetal/assembly; any other bundle (DiT) must pass.
"""
import re
from pathlib import Path

ASSEMBLY = Path(__file__).resolve().parents[2] / "baremetal" / "assembly"
_NV = re.compile(r"^\s*(?:\w+:\s*)?(?:v\w*(?:\.\w+)*\.nv\b|vload\.scale\b|vstore\.scale\b)", re.I | re.M)
_ILLEGAL = re.compile(r"^\s*#\s*@EXPECT_ILLEGAL\b", re.M)
UNITS = {"default": set(), "ipt-nvfp4": {"nv"}}


def needs(source: str) -> set:
    src = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    return {"nv"} if _NV.search(src) else set()


def expected(test: str, config: str, bundle: Path) -> str:
    source = next((s for s in (bundle.with_suffix(".S"), ASSEMBLY / f"{test}.S") if s.exists()), None)
    if source is None:
        return "PASS"
    text = source.read_text()
    if _ILLEGAL.search(text):
        return "ILLEGAL"
    return "PASS" if needs(text) <= UNITS[config] else "ILLEGAL"
