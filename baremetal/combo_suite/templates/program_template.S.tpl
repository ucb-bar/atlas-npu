# {long_name} -- auto-generated Atlas combination test
# short_name: {short_name}
# tier: {tier}
# familyA: {family_a}
# familyB: {family_b}
# risk_class: {risk_class}
# seed: {seed}
# @TIMEOUT {timeout}

    ADDI  x28, x0, 1
    ADDI  x29, x0, 0
    ADDI  x27, x0, {signature}

{body}

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL
