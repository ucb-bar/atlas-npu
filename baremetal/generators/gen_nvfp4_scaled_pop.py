#!/usr/bin/env python3
"""gen_nvfp4_scaled_pop.py — golden data for assembly/nvfp4_scaled_pop.S: the data and goldens of
gen_mxu1_scaled_pop.py, since the NVFP4 MXU's pops follow the same rules."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("gen_mxu1_scaled_pop.py")), run_name="__main__")
