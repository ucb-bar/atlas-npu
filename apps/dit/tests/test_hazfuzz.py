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


def test_programs_emit_every_vpu_class_and_each_accumulator_push():
    """Column and row reductions, vpack and vunpack (never vd = vs2 + 1), every vli, FP8 and FP32
    pushes, and vrecip, vsqrt or vlog2 of a vexp result."""
    stats, source = {}, ""
    for seed in range(8):
        g = hazfuzz.Gen(seed, 400, ("mxu1", "nv"))
        source += g.build()[0]
        for k, v in g.stats.items():
            stats[k] = stats.get(k, 0) + v
    for op in ("vredsum.bf16", "vredmin.bf16", "vredmin.row.bf16", "vpack.bf16.fp8", "vunpack.fp8.bf16",
               "vli.row", "vli.col", "vli.one", "vmatpush.acc.fp8.mxu1", "vmatpush.acc.fp32.mxu1",
               "vmatpush.acc.fp32.nv", "vexp.bf16"):
        assert stats.get(op), op
    assert any(stats.get(op) for op in ("vrecip.bf16", "vsqrt.bf16", "vlog2.bf16"))
    for line in source.splitlines():
        if line.strip().startswith("vpack.bf16.fp8"):
            vd, vs2 = (int(r.strip(" ,m")) for r in line.split()[1:3])
            assert vd != vs2 + 1, line


def test_directed_small_rows_reads_right_after_each_write():
    d = hazfuzz.directed_small_rows(("mxu1", "nv"))
    src = d["source"]
    assert d["name"] == "hazard_small_rows_nv" and all(f"vsetm {m}\n" in src for m in (1, 2, 3, 4))
    assert "vmatpop.fp32.acc.nv m32, acc1\n    vmatmul.acc.nv acc0, m34, w0" in src
    assert src.count("vredsum.row.bf16 m8, m10\n    vredsum.row.bf16 m12, m14\n    vredsum.row.bf16 m16, m18") == 4
