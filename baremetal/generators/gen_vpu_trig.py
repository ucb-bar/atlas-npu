#!/usr/bin/env python3

import json
import math

from vpu_gen_utils import BF16_PER_BEAT, float_to_bf16, run_unary_rows


def bf16_hex(bits: int) -> str:
    return f"0x{bits & 0xFFFF:04X}"


def unary_result_hex(op: str, value: float) -> str:
    lane_bits = float_to_bf16(value)
    row = [lane_bits] * BF16_PER_BEAT
    return bf16_hex(run_unary_rows(op, [row])[0][0])


inputs = {
    "two_pi": 2.0 * math.pi,
    "three_pi_over_2": 1.5 * math.pi,
    "pi": math.pi,
    "pi_over_2": 0.5 * math.pi,
    "pi_over_4": 0.25 * math.pi,
    "pi_over_8": 0.125 * math.pi,
    "pi_over_6": math.pi / 6.0,
    "pi_over_12": math.pi / 12.0,
}

results = {}

for name, value in inputs.items():
    input_bits = float_to_bf16(value)
    results[name] = {
        "input_val": {
            "input": name,
            "hex_bf16": bf16_hex(input_bits),
        },
        "vsin": {
            "hex_bf16": unary_result_hex("sin", value),
        },
        "vcos": {
            "hex_bf16": unary_result_hex("cos", value),
        },
        "vtanh": {
            "hex_bf16": unary_result_hex("tanh", value),
        },
    }

print(json.dumps(results, indent=4))
