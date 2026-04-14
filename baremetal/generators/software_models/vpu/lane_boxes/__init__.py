"""Per-laneBox Python mirrors of src/main/scala/atlas/vector/laneBoxes/*.scala.

One file per Scala module. Each exposes `compute_now(req)` for synchronous
use (torch adapter, fast tests) and `step(op_name, req)` for cycle-accurate
use (cross-test against ChiselTest). Latency is per-op, not per-class.
"""
