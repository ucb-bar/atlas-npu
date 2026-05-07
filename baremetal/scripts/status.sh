#!/usr/bin/env bash
# Internal helper: print up-to-date status for one Atlas baremetal test's
# pipeline stages (golden JSON, C harness, binary, simv).
# Invoked by the Makefile `status-%` rule.
#
# Required env: NAME, GEN_DIR, TESTS_DIR, ASM_DIR, BUILD_DIR, SIMV
set -u

# Some tests fall back to a prefix-stripped generator name (mxu0_*/mxu1_* → mxu_*),
# matching the Makefile's atlas_%.c pattern rule.
GEN_BASE="$NAME"
if [[ ! -f "$GEN_DIR/gen_$GEN_BASE.py" ]]; then
    GEN_BASE=${NAME//mxu[01]_/mxu_}
fi

GEN_PY="$GEN_DIR/gen_$GEN_BASE.py"
JSON="$GEN_DIR/$NAME.json"
SRC="$ASM_DIR/$NAME.S"
C="$TESTS_DIR/atlas_$NAME.c"
BIN="$BUILD_DIR/atlas_$NAME.riscv"

# is_cached <target> [<source>...] — target exists and is not older than any source.
is_cached() {
    local tgt="$1"; shift
    [[ -f "$tgt" ]] || return 1
    for s in "$@"; do
        [[ -f "$s" && "$s" -nt "$tgt" ]] && return 1
    done
    return 0
}

echo "$NAME:"
if [[ -f "$GEN_PY" ]]; then
    if is_cached "$JSON" "$GEN_PY"; then
        echo "  golden JSON:  cached"
    else
        echo "  golden JSON:  stale or missing → will regenerate"
    fi
else
    echo "  golden JSON:  n/a (no generator)"
fi

if is_cached "$C" "$SRC" "$JSON"; then
    echo "  C harness:    cached"
else
    echo "  C harness:    stale or missing → will reassemble"
fi

if is_cached "$BIN" "$C"; then
    echo "  binary:       cached"
else
    echo "  binary:       stale or missing → will rebuild"
fi

if [[ -x "$SIMV" ]]; then
    echo "  simv:         cached"
else
    echo "  simv:         missing → will build"
fi
