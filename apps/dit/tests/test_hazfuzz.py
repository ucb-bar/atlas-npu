"""The hazard fuzzer's command line (hazfuzz.py)."""
import pytest

from ditflow import hazfuzz


@pytest.mark.parametrize("units", ["mxu0", "nv", "mxu1,mxu0"])
def test_units_without_mxu1_or_unknown_stop_with_a_message(units, tmp_path, capsys):
    with pytest.raises(SystemExit):
        hazfuzz.main([str(tmp_path), "--count", "1", "--units", units])
    assert f"--units {units}: give mxu1" in capsys.readouterr().err
