#!/usr/bin/env python3
"""Generate goldens or assembly for pw_vpu_overlap_pointwise_pack.S."""

from vpu_overlap_pairwise_common import (
    POINTWISE_PACK_FIRST_OPS,
    main_entry,
)


if __name__ == "__main__":
    main_entry(
        test_name="pw_vpu_overlap_pointwise_pack",
        title="pairwise VPU overlap with pointwise/fp8 first-op cases",
        first_op_names=POINTWISE_PACK_FIRST_OPS,
    )
