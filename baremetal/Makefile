PYTHON ?= python3
CONFIG ?= EE290SimConfig
VCS_DIR ?= sims/vcs
RUN_TIMEOUT_S ?= 0
STRESS_COUNT ?= 12
SEED ?= 1337
TIERS_ALL := tier0,tier1,tier2,tier3

CHIPYARD_ROOT := $(abspath ../../..)
TESTS_DIR := $(CURDIR)/tests
BUILD_DIR := $(TESTS_DIR)/build
GEN_DIR := $(CURDIR)/generators
ASM_DIR   := $(CURDIR)/assembly

.DEFAULT_GOAL := cicd

TESTS := $(patsubst $(ASM_DIR)/%.S,%,$(wildcard $(ASM_DIR)/*.S))
GENERATED_C := $(patsubst %,$(TESTS_DIR)/atlas_%.c,$(TESTS))

# ── Sim infra ─────────────────────────────────────────────────────────
VCS_DIR_ABS  := $(CHIPYARD_ROOT)/sims/vcs
GEN_COVERAGE ?= 1
LOADMEM      ?= 1

# simv path. Chipyard names the binary $(sim_dir)/simv-$(MODEL_PACKAGE)-$(CONFIG)
# (sims/vcs/Makefile:25). MODEL_PACKAGE defaults to chipyard.harness and isn't
# customized for Atlas — hardcode rather than recursing into chipyard's makefile
# at parse time (which costs ~8s on every `make` invocation, including help).
SIMV           := $(VCS_DIR_ABS)/simv-chipyard.harness-$(CONFIG)
SIM_OUTPUT_DIR := $(VCS_DIR_ABS)/output/chipyard.harness.TestHarness.$(CONFIG)

# Test groups: auto-discovered from the prefix before the first underscore.
# E.g. tests perf_mm_64x64, perf_mm_128x128 → group `perf`. Lets the user run
# `make test-perf` to sweep every perf_* test, `make test-smolvla` for smolvla_*, etc.
TEST_GROUPS := $(sort $(foreach t,$(TESTS),$(firstword $(subst _, ,$(t)))))

.PHONY: cicd build-assembly clean help list-tests list-groups _sim-% test-all simv status-% status-all $(TESTS)

cicd: build-assembly
	mv "$(BUILD_DIR)"/*.riscv "$(CURDIR)/"

# MAKEFLAGS= prefix: parent `make` may be GNU Make 4.4+ which uses the `fifo:`
# jobserver protocol; cmake's internal gmake on this host is 4.3 and chokes on it.
# Clearing MAKEFLAGS keeps the cmake-generated Makefile from inheriting an
# auth string it can't parse. cmake's --parallel still works (it passes -j directly).
build-assembly: $(GENERATED_C)
	cmake -S $(TESTS_DIR) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug
	MAKEFLAGS= cmake --build $(BUILD_DIR) --parallel

$(TESTS): %: $(BUILD_DIR)/atlas_%.riscv
	@:

$(BUILD_DIR)/CMakeCache.txt:
	cmake -S $(TESTS_DIR) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug

# Lazy-reconfigure: if this test's .c is newer than CMakeCache, rerun cmake
# configure (handles "new test added since last configure" without forcing all
# 102 .c files to be generated upfront). CMakeCache is order-only — its mtime
# does not trigger a rebuild of the binary.
# MAKEFLAGS= avoids leaking the parent's (4.4) `fifo:` jobserver into cmake's
# internal gmake (4.3 on this host), which would otherwise error out under -j.
$(BUILD_DIR)/atlas_%.riscv: $(TESTS_DIR)/atlas_%.c | $(BUILD_DIR)/CMakeCache.txt
	@if [ "$<" -nt "$(BUILD_DIR)/CMakeCache.txt" ]; then \
	    echo "--> Reconfiguring CMake (atlas_$*.c is newer than CMakeCache)"; \
	    cmake -S $(TESTS_DIR) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug; \
	fi
	MAKEFLAGS= cmake --build $(BUILD_DIR) --target atlas_$*

# assembler.py is listed as a prereq so a marker-format change (or any other
# C-harness emission tweak) invalidates already-generated atlas_*.c files.
$(TESTS_DIR)/atlas_%.c: $(ASM_DIR)/%.S assembler.py
	@set -e; \
	BASE=$*; \
	if [ ! -f "$(GEN_DIR)/gen_$$BASE.py" ]; then \
		BASE=$$(echo $$BASE | sed -E 's/mxu[01]_/mxu_/'); \
	fi; \
	if [ -f "$(GEN_DIR)/gen_$$BASE.py" ]; then \
		echo "--> Using generator gen_$$BASE.py for test $*"; \
		$(PYTHON) $(GEN_DIR)/gen_$$BASE.py > $(GEN_DIR)/$*.json; \
		$(PYTHON) assembler.py $< \
			--out-c $@ \
			--golden-json $(GEN_DIR)/$*.json; \
	else \
		echo "--> No generator found for $*. Assembling standalone."; \
		$(PYTHON) assembler.py $< \
			--out-c $@; \
	fi

# Phony so chipyard's make is always consulted — it tracks Chisel/Verilog deps
# (common.mk:136 wires Scala sources → sbt jar → FIRRTL → Verilog → simv) and
# rebuilds simv only when needed. Cost: ~11s of chipyard makefile parse per
# top-level `make` invocation. Make dedupes phony prereqs within one run, so
# this fires once even under `make -j8 test-all`.
simv:
	$(MAKE) -C $(VCS_DIR_ABS) CONFIG=$(CONFIG) default

# Per-test pipeline-cache report. Prints which stages are already up-to-date.
status-%:
	@NAME=$* GEN_DIR=$(GEN_DIR) TESTS_DIR=$(TESTS_DIR) \
	 ASM_DIR=$(ASM_DIR) BUILD_DIR=$(BUILD_DIR) SIMV=$(SIMV) \
	 $(CURDIR)/scripts/status.sh

status-all: $(addprefix status-,$(TESTS))

# Internal per-test sim dispatch. Underscore prefix signals "don't invoke
# directly" — users get to it via `make test <name>...`, `make test-<group>`,
# or `make test-all`, never via `_sim-<name>`. The recipe just runs the sim;
# chipyard tees its log to $(SIM_OUTPUT_DIR)/atlas_<name>.log, which contains
# the *** PASSED *** / *** FAILED *** marker emitted by the C harness (or, in
# the future, an SV testbench). Pass/fail is determined later by the summary
# recipes on test, test-all, and test-<group>.
_sim-%: status-% $(BUILD_DIR)/atlas_%.riscv simv
	@$(MAKE) -s -C $(VCS_DIR_ABS) run-binary \
	    BINARY=$(BUILD_DIR)/atlas_$*.riscv CONFIG=$(CONFIG) \
	    LOADMEM=$(LOADMEM) GEN_COVERAGE=$(GEN_COVERAGE) \
	    BREAK_SIM_PREREQ=1

# Don't let GNU Make auto-delete .riscv binaries as "intermediate" after a
# test-% rule consumes them. Without this, `make test-mxu0` would print a
# bunch of `rm .../atlas_mxu0_*.riscv` lines at the end and the next sweep
# would have to relink them. The .c files have the same pattern-rule fate
# so cover both.
.SECONDARY: $(GENERATED_C) $(addprefix $(BUILD_DIR)/atlas_,$(addsuffix .riscv,$(TESTS)))

# Summary aggregator. Reads chipyard's per-test logs (which contain markers
# emitted by the C harness or, in the future, an SV testbench), classifies
# each test as PASSED / FAILED / DIED / INCOMPLETE, prints a table with bare
# test names, and returns nonzero if any test wasn't a PASS. Lives in a perl
# script (not inline in this Makefile) because its multi-condition logic is
# hard to read when escaped through Make's recipe syntax.
SUMMARY_INVOKE = TESTS="$(strip $(1))" SIM_OUTPUT_DIR=$(SIM_OUTPUT_DIR) \
                 perl $(CURDIR)/scripts/summary.pl

# Sweep entrypoint: depends on every per-test target, then prints summary.
test-all: $(addprefix _sim-,$(TESTS))
	@$(call SUMMARY_INVOKE,$(TESTS))

# `make test <name1> <name2> ...` — run a specific test or any subset and print
# the summary at the end. Trailing positional args also become Make goals in
# their own right; they match the existing $(TESTS) static pattern rule (recipe
# `@:`), which is a no-op once their .riscv is built as a transitive prereq of
# _sim-<name>. So no double-build, no rule-conflict warnings. Validation runs
# at parse time via $(error) so typos fail fast before any work.
ifneq ($(filter test,$(MAKECMDGOALS)),)
PICK_LIST := $(filter-out test,$(MAKECMDGOALS))
.PHONY: test
ifeq ($(strip $(PICK_LIST)),)
test:
	@echo 'Usage: make test <test1> <test2> ...'
	@echo '       make list-tests       # see available tests'
	@echo '       make list-groups      # see test groups'
else ifneq ($(filter-out $(TESTS),$(PICK_LIST)),)
$(error Unknown test name(s): $(filter-out $(TESTS),$(PICK_LIST)))
else
test: $(addprefix _sim-,$(PICK_LIST))
	@$(call SUMMARY_INVOKE,$(PICK_LIST))
endif
endif

# `summary-all` and `summary-<group>` re-print the summary from already-run logs
# without re-running any sims. Useful after a sweep, or to inspect older results.
.PHONY: summary-all
summary-all:
	@$(call SUMMARY_INVOKE,$(TESTS))

# Auto-generated group targets: `test-<group>` runs all tests in that group in parallel
# under `make -jN`, then prints the summary scoped to that group. `summary-<group>`
# does the same summary read-only (no sim).
define _atlas_group_rule
ifneq ($$(filter $(1)_%,$$(TESTS)),)
.PHONY: test-$(1) summary-$(1)
test-$(1): $$(addprefix _sim-,$$(filter $(1)_%,$$(TESTS)))
	@$$(call SUMMARY_INVOKE,$$(filter $(1)_%,$$(TESTS)))
summary-$(1):
	@$$(call SUMMARY_INVOKE,$$(filter $(1)_%,$$(TESTS)))
endif
endef
$(foreach g,$(TEST_GROUPS),$(eval $(call _atlas_group_rule,$(g))))


clean:
	@echo "--> Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -f $(TESTS_DIR)/atlas_*.c
	rm -f $(GEN_DIR)/*.json
	rm -f *.riscv
	@echo "Done."

help:
	@echo "Build targets:"
	@echo "  cicd                Build all tests and move binaries to root"
	@echo "  build-assembly      Build all tests in build directory"
	@echo "  clean               Remove all generated files and build directories"
	@echo ""
	@echo "Test targets:"
	@echo "  test <name>...      Run one or more specific tests (e.g. 'make test alu_basic vpu_binary')"
	@echo "  test-<group>        Sweep every test in a group, print summary table at end"
	@echo "  test-all            Run every test (use 'make -j8 test-all' for parallel)"
	@echo "  summary-<group>     Re-print summary table from existing logs (no re-run)"
	@echo "  summary-all         Same, scoped to every test"
	@echo "  status-<name>       Show which pipeline stages are cached for one test"
	@echo "  status-all          Cache report for every test"
	@echo "  list-tests          Show all detected tests in $(ASM_DIR)"
	@echo "  list-groups         Show detected test groups + member counts"
	@echo ""
	@echo "Sim logs (PASS or FAIL) are written by chipyard to:"
	@echo "  $(SIM_OUTPUT_DIR)/atlas_<name>.log"

list-tests:
	@echo "Detected tests:"
	@for test in $(TESTS); do echo "  - $$test"; done

list-groups:
	@echo "Detected test groups (use 'make test-<group>' to run all members):"
	@for g in $(TEST_GROUPS); do \
	    n=$$(echo "$(TESTS)" | tr ' ' '\n' | grep -cE "^$${g}_" || true); \
	    if [ "$$n" -gt 0 ]; then \
	        printf "  test-%-12s %3d tests\n" "$$g" "$$n"; \
	    fi; \
	 done
