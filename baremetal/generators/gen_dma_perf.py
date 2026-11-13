#!/usr/bin/env python3
"""Deterministic golden data for the DMA latency/throughput benchmarks.

Input/output addresses are DRAM byte offsets; VMEM operands are word addresses.
Each output is poisoned before execution and surrounded by checked guard beats.
"""
import json

CASES = {
    'dma_perf_read_32b': ('read', 32, 1, 'single'),
    'dma_perf_read_256b': ('read', 256, 1, 'single'),
    'dma_perf_read_1024b': ('read', 1024, 1, 'single'),
    'dma_perf_read_4096b': ('read', 4096, 1, 'single'),
    'dma_perf_read_serialized': ('read', 32, 8, 'serialized'),
    'dma_perf_read_fenced_32b': ('read', 32, 8, 'serialized'),
    'dma_perf_read_fenced_256b': ('read', 256, 8, 'serialized'),
    'dma_perf_read_fenced_1024b': ('read', 1024, 8, 'serialized'),
    'dma_perf_read_fenced_4096b': ('read', 4096, 8, 'serialized'),
    'dma_perf_read_multichannel': ('read', 32, 8, 'multichannel'),
    'dma_perf_read_saturated': ('read', 1024, 8, 'saturated'),
    'dma_perf_read_vmem_contention': ('read', 4096, 1, 'vmem_contention'),
    'dma_perf_read_imem_overlap': ('read', 1024, 8, 'imem_overlap'),
    'dma_perf_write_32b': ('write', 32, 1, 'single'),
    'dma_perf_write_256b': ('write', 256, 1, 'single'),
    'dma_perf_write_1024b': ('write', 1024, 1, 'single'),
    'dma_perf_write_4096b': ('write', 4096, 1, 'single'),
    'dma_perf_write_fenced_32b': ('write', 32, 8, 'serialized'),
    'dma_perf_write_fenced_256b': ('write', 256, 8, 'serialized'),
    'dma_perf_write_fenced_1024b': ('write', 1024, 8, 'serialized'),
    'dma_perf_write_fenced_4096b': ('write', 4096, 8, 'serialized'),
    'dma_perf_write_serialized': ('write', 32, 8, 'serialized'),
    'dma_perf_write_multichannel': ('write', 32, 8, 'multichannel'),
    'dma_perf_write_saturated': ('write', 1024, 8, 'saturated'),
    'dma_perf_write_vmem_contention': ('write', 4096, 1, 'vmem_contention'),
    'dma_perf_write_imem_overlap': ('write', 1024, 8, 'imem_overlap'),
}
DRAM_BASE = 0x90000000
INPUT_STRIDE = 0x2000
OUTPUT_BASE = 0x20000
VMEM_BASE_WORD = 0x1000  # Reserve VMEM[0] for scalar contention traffic.
POISON = "0x" + "a5" * 32
GUARD = "0x" + "5a" * 32


def payload(channel, beat):
    return "0x" + "".join(f"{(0xD0000000 | channel << 20 | beat << 4 | word):08x}"
                            for word in reversed(range(8)))


def golden(name):
    _, size, count, _ = CASES[name]
    preloads, checks = [], []
    for channel in range(count):
        src = channel * INPUT_STRIDE
        dst = OUTPUT_BASE + channel * INPUT_STRIDE
        for beat in range(size // 32):
            data = payload(channel, beat)
            preloads += [{"word_offset": src // 32 + beat, "data": data},
                         {"word_offset": dst // 32 + beat, "data": POISON}]
            checks.append({"word_offset": dst // 32 + beat, "expected": data})
        for address in (dst - 32, dst + size):
            preloads.append({"word_offset": address // 32, "data": GUARD})
            checks.append({"word_offset": address // 32, "expected": GUARD})
    return {"dram_preloads": preloads, "dram_checks": checks, "timeout": 2000000}


def emit(name):
    print(json.dumps(golden(name), indent=2))


if __name__ == "__main__":
    import sys
    emit(sys.argv[1])
