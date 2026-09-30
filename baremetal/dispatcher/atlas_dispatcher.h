#ifndef ATLAS_DISPATCHER_H
#define ATLAS_DISPATCHER_H

#include <stddef.h>
#include <stdint.h>
#include <dispatcher_image.h>

/* Single host producer. See README.md before adapting an existing kernel. */
#define ATLAS_DISPATCH_SLOT_WORDS 0x3f00U
#define ATLAS_DISPATCH_MAX_SEQUENCE 0x3fffffffU
#define ATLAS_DISPATCH_READY 1U

enum { ATLAS_DISPATCH_OK = 0, ATLAS_DISPATCH_BUSY = -1,
       ATLAS_DISPATCH_INVALID = -2, ATLAS_DISPATCH_HALTED = -3,
       ATLAS_DISPATCH_TIMEOUT = -4 };

typedef struct {
    uint32_t sequence;
    uint32_t submitted;
    uint32_t owner[2];
    unsigned loaded[2];
} atlas_dispatcher;

/* Override these three macros together for a test or a different MMIO layer. */
#ifndef ATLAS_DISPATCH_READ
#define ATLAS_DISPATCH_READ(addr) (*(volatile uint32_t *)(uintptr_t)(addr))
#define ATLAS_DISPATCH_WRITE(addr, value) \
    (*(volatile uint32_t *)(uintptr_t)(addr) = (value))
#define ATLAS_DISPATCH_FENCE() __asm__ volatile ("fence iorw, iorw" ::: "memory")
#endif

static inline uint32_t atlas_dispatch_slot_word(unsigned slot) {
    return 0x100U + slot * 0x4000U;
}
static inline uint32_t atlas_dispatch_ack(void) {
    return ATLAS_DISPATCH_READ(0x40010UL);
}
static inline int atlas_dispatch_running(void) {
    return !(ATLAS_DISPATCH_READ(0x40008UL) & 1U);
}
/* A newer STARTED acknowledgment also proves every older command completed. */
static inline int atlas_dispatch_completed(uint32_t ack, uint32_t command) {
    return (ack >> 1) > command || ((ack >> 1) == command && (ack & 1U));
}
static inline int atlas_dispatch_slot_free(const atlas_dispatcher *d, unsigned slot) {
    return slot < 2 && (!d->owner[slot] ||
        atlas_dispatch_completed(atlas_dispatch_ack(), d->owner[slot]));
}

/* Call only on a halted, quiescent accelerator; this is not an engine drain.
 * Install once, start at PC=0, and wait until the dispatcher acknowledges boot.
 * poll_limit counts MMIO polls, not cycles. A timeout does not cancel startup. */
static inline int atlas_dispatch_init(atlas_dispatcher *d, unsigned poll_limit) {
    if (atlas_dispatch_running()) return ATLAS_DISPATCH_BUSY;
    *d = (atlas_dispatcher){0};
    ATLAS_DISPATCH_WRITE(0x40010UL, 0);
    ATLAS_DISPATCH_WRITE(0x40014UL, 0);
    for (size_t i = 0; i < sizeof(atlas_dispatcher_image) / sizeof(uint32_t); ++i)
        ATLAS_DISPATCH_WRITE(0x20000UL + 4 * i, atlas_dispatcher_image[i]);
    ATLAS_DISPATCH_FENCE();
    ATLAS_DISPATCH_WRITE(0x40018UL, 1);
    ATLAS_DISPATCH_FENCE();
    while (poll_limit--) {
        if (atlas_dispatch_ack() == ATLAS_DISPATCH_READY) return ATLAS_DISPATCH_OK;
    }
    return ATLAS_DISPATCH_TIMEOUT;
}

/* Writes only: runtime IMEM reads would block while Atlas is running. */
static inline int atlas_dispatch_stage(atlas_dispatcher *d, unsigned slot,
                                       const uint32_t *words, size_t count) {
    if (slot >= 2 || !words || !count || count > ATLAS_DISPATCH_SLOT_WORDS)
        return ATLAS_DISPATCH_INVALID;
    if (!atlas_dispatch_running()) return ATLAS_DISPATCH_HALTED;
    if (!atlas_dispatch_slot_free(d, slot)) return ATLAS_DISPATCH_BUSY;
    /* Order the observed release before writes that reuse the instruction slot. */
    ATLAS_DISPATCH_FENCE();
    uintptr_t base = 0x20000UL + 4UL * atlas_dispatch_slot_word(slot);
    for (size_t i = 0; i < count; ++i)
        ATLAS_DISPATCH_WRITE(base + 4 * i, words[i]);
    ATLAS_DISPATCH_FENCE();
    d->loaded[slot] = 1;
    return ATLAS_DISPATCH_OK;
}

/* Nonblocking. At most one command may be pending behind the running command.
 * A staged kernel can be submitted again without reloading once its slot is free.
 * The returned command token is used with atlas_dispatch_wait(). */
static inline int atlas_dispatch_submit(atlas_dispatcher *d, unsigned slot,
                                        uint32_t *command) {
    if (slot >= 2 || !command || !d->loaded[slot] ||
        d->sequence == ATLAS_DISPATCH_MAX_SEQUENCE) return ATLAS_DISPATCH_INVALID;
    if (!atlas_dispatch_running()) return ATLAS_DISPATCH_HALTED;
    uint32_t ack = atlas_dispatch_ack();
    if ((ack >> 1) < d->submitted || !atlas_dispatch_slot_free(d, slot))
        return ATLAS_DISPATCH_BUSY;
    uint32_t token = (++d->sequence << 1) | slot;
    d->submitted = d->owner[slot] = token;
    *command = token;
    /* Orders input data and IMEM stores before command publication. */
    ATLAS_DISPATCH_FENCE();
    ATLAS_DISPATCH_WRITE(0x40014UL, token);
    ATLAS_DISPATCH_FENCE();
    return ATLAS_DISPATCH_OK;
}

static inline int atlas_dispatch_wait(uint32_t command, unsigned poll_limit) {
    if (!(command >> 1) || (command >> 1) > ATLAS_DISPATCH_MAX_SEQUENCE)
        return ATLAS_DISPATCH_INVALID;
    while (poll_limit--) {
        if (atlas_dispatch_completed(atlas_dispatch_ack(), command)) {
            ATLAS_DISPATCH_FENCE();
            return ATLAS_DISPATCH_OK;
        }
        if (!atlas_dispatch_running()) return ATLAS_DISPATCH_HALTED;
    }
    return ATLAS_DISPATCH_TIMEOUT;
}
#endif
