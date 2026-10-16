# pmu_counter.S — smoke test for the tile PMU run-cycle counter
# @TIMEOUT 500
# @PMU_TEST
#
# core.run_cycles sorts to counter ID 0. The C harness clears and enables the
# PMU before starting this program, then snapshots and checks the counter.

    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x0
    ECALL
