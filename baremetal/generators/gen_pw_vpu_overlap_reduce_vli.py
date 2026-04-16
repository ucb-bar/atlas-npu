#!/usr/bin/env python3
"""Generate goldens or assembly for pw_vpu_overlap_reduce_vli.S."""

from vpu_overlap_pairwise_common import (
    REDUCE_VLI_FIRST_OPS,
    main_entry,
)


if __name__ == "__main__":
    main_entry(
        test_name="pw_vpu_overlap_reduce_vli",
        title="pairwise VPU overlap with reduce/VLI first-op cases",
        first_op_names=REDUCE_VLI_FIRST_OPS,
    )
