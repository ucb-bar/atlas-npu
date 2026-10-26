"""The hazard fuzzer's command line (hazfuzz.py)."""
import pytest

from ditflow import hazfuzz


@pytest.mark.parametrize("units", ["mxu0", "nv", "mxu1,mxu0"])
def test_units_without_mxu1_or_unknown_stop_with_a_message(units, tmp_path, capsys):
    with pytest.raises(SystemExit):
        hazfuzz.main([str(tmp_path), "--count", "1", "--units", units])
    assert f"--units {units}: give mxu1" in capsys.readouterr().err


def test_programs_set_m_and_use_rows_below_it():
    """The random programs and the directed one run vsetm with small M, and the expected DRAM comes
    from the functional model."""
    t = hazfuzz.make(3, 300, ("mxu1", "nv"))
    rows = {int(line.split()[1]) for line in t["source"].splitlines() if line.strip().startswith("vsetm")}
    assert t["stats"]["vsetm"] > 0 and rows & {1, 2, 3} and 32 in rows
    d = hazfuzz.directed_rows(("mxu1", "nv"))
    assert d["name"] == "hazard_rows_nv" and "vsetm 1" in d["source"] and len(d["check"]) == 2
