/* Run on EE290SimConfig or a quiescent Atlas system. No RVV is needed to test
 * the control protocol. VMEM word 0 counts completed kernel invocations. */
#include <stdio.h>
#include "atlas_dispatcher.h"
#include <test_kernel_image.h>
#ifdef ATLAS_DISPATCH_TEST_RVV
#include <riscv_vector.h>
#endif

#define POLLS 2000000U
#define COUNT (sizeof(atlas_test_kernel) / sizeof(uint32_t))
#define CHECK(expr) do { if (!(expr)) { \
    printf("*** FAILED *** line %d: %s status=%x ack=%x\n", __LINE__, #expr, \
           ATLAS_DISPATCH_READ(0x40008UL), atlas_dispatch_ack()); return 1; \
} } while (0)

int main(void) {
    atlas_dispatcher d;
    uint32_t a, b, again;
    CHECK(!atlas_dispatch_running());
    ATLAS_DISPATCH_WRITE(0x20000000UL, 0);
    CHECK(atlas_dispatch_init(&d, POLLS) == 0);
    /* No command: stay in the dispatcher, even with a partially loaded slot. */
    uintptr_t base = 0x20000UL + 4UL * atlas_dispatch_slot_word(0);
    for (size_t i = 0; i < COUNT / 2; ++i)
        ATLAS_DISPATCH_WRITE(base + 4 * i, atlas_test_kernel[i]);
    ATLAS_DISPATCH_FENCE();
    for (unsigned i = 0; i < 128; ++i) {
        CHECK(atlas_dispatch_ack() == ATLAS_DISPATCH_READY);
        CHECK(atlas_dispatch_running());
    }
    CHECK(ATLAS_DISPATCH_READ(0x20000000UL) == 0);
    CHECK(atlas_dispatch_submit(&d, 0, &a) == ATLAS_DISPATCH_INVALID);
    CHECK(atlas_dispatch_stage(&d, 2, atlas_test_kernel, COUNT) == ATLAS_DISPATCH_INVALID);
    CHECK(atlas_dispatch_stage(&d, 0, atlas_test_kernel,
                               ATLAS_DISPATCH_SLOT_WORDS + 1) == ATLAS_DISPATCH_INVALID);
    CHECK(atlas_dispatch_stage(&d, 0, atlas_test_kernel, COUNT) == 0);
    CHECK(atlas_dispatch_submit(&d, 0, &a) == 0);
    unsigned polls = POLLS;
    while (polls-- && (atlas_dispatch_ack() >> 1) != a) {}
    CHECK((atlas_dispatch_ack() >> 1) == a);
    CHECK(!(atlas_dispatch_ack() & 1U));
    CHECK(atlas_dispatch_stage(&d, 0, atlas_test_kernel, COUNT) == ATLAS_DISPATCH_BUSY);
    CHECK(atlas_dispatch_stage(&d, 1, atlas_test_kernel, COUNT) == 0);
    CHECK(atlas_dispatch_submit(&d, 1, &b) == 0);
    CHECK(atlas_dispatch_stage(&d, 1, atlas_test_kernel, COUNT) == ATLAS_DISPATCH_BUSY);
    CHECK(atlas_dispatch_submit(&d, 0, &again) == ATLAS_DISPATCH_BUSY);
    CHECK(atlas_dispatch_wait(b, POLLS) == 0);
    /* We intentionally did not observe A's transient DONE acknowledgment. */
    CHECK(atlas_dispatch_wait(a, 1) == 0);
    CHECK(ATLAS_DISPATCH_READ(0x20000000UL) == 2);
    for (unsigned i = 0; i < 128; ++i) CHECK(atlas_dispatch_ack() == ((b << 1) | 1));
    CHECK(ATLAS_DISPATCH_READ(0x20000000UL) == 2);
    uint32_t expected = 2;
#ifdef ATLAS_DISPATCH_TEST_RVV
    /* A real Saturn -> Atlas data handoff while the dispatcher stays running.
     * Transform a full vector's worth of VMEM, then let Atlas increment word 0.
     * Both submitted kernels are DONE, so the host owns this data buffer. */
    enum { LANES = 32 };
    for (unsigned i = 1; i < LANES; ++i)
        ATLAS_DISPATCH_WRITE(0x20000000UL + 4 * i, i);
    ATLAS_DISPATCH_FENCE();
    for (size_t offset = 0; offset < LANES;) {
        size_t vl = __riscv_vsetvl_e32m1(LANES - offset);
        uint32_t *p = (uint32_t *)(uintptr_t)(0x20000000UL + 4 * offset);
        vuint32m1_t v = __riscv_vle32_v_u32m1(p, vl);
        v = __riscv_vadd_vx_u32m1(v, 100, vl);
        __riscv_vse32_v_u32m1(p, v, vl);
        offset += vl;
    }
    ATLAS_DISPATCH_FENCE();
    expected += 100;
    CHECK(ATLAS_DISPATCH_READ(0x20000000UL) == expected);
    for (unsigned i = 1; i < LANES; ++i)
        CHECK(ATLAS_DISPATCH_READ(0x20000000UL + 4 * i) == i + 100);
#endif
    /* Reuse resident code, then overwrite the released slot and execute it. */
    CHECK(atlas_dispatch_submit(&d, 1, &again) == 0);
    CHECK(atlas_dispatch_wait(again, POLLS) == 0);
    CHECK(ATLAS_DISPATCH_READ(0x20000000UL) == ++expected);
    CHECK(atlas_dispatch_stage(&d, 0, atlas_test_kernel, COUNT) == 0);
    CHECK(atlas_dispatch_submit(&d, 0, &again) == 0);
    CHECK(atlas_dispatch_wait(again, POLLS) == 0);
    CHECK(ATLAS_DISPATCH_READ(0x20000000UL) == ++expected);
    CHECK(atlas_dispatch_running());
    ATLAS_DISPATCH_WRITE(0x40018UL, 0); /* all kernels have completed */
    ATLAS_DISPATCH_FENCE();
#ifdef ATLAS_DISPATCH_TEST_RVV
    printf("*** PASSED *** Saturn RVV / Atlas dispatcher handoff\n");
#else
    printf("*** PASSED *** resident dispatcher, delayed loading, A/B queue, reuse\n");
#endif
    return 0;
}
