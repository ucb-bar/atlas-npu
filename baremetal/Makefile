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

MAKEFLAGS += -j16

.DEFAULT_GOAL := cicd

TESTS := $(patsubst $(ASM_DIR)/%.S,%,$(wildcard $(ASM_DIR)/*.S))
GENERATED_C := $(patsubst %,$(TESTS_DIR)/atlas_%.c,$(TESTS))

.PHONY: cicd build-assembly clean help list-tests $(TESTS)

cicd: build-assembly
	mv "$(BUILD_DIR)"/*.riscv "$(CURDIR)/"

build-assembly: $(BUILD_DIR)/CMakeCache.txt
	cmake --build $(BUILD_DIR) --parallel

$(TESTS): %: $(BUILD_DIR)/CMakeCache.txt
	cmake --build $(BUILD_DIR) --target atlas_$*

$(BUILD_DIR)/CMakeCache.txt: $(GENERATED_C)
	cmake -S $(TESTS_DIR) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug

$(TESTS_DIR)/atlas_%.c: $(ASM_DIR)/%.S
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


clean:
	@echo "--> Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -f $(TESTS_DIR)/atlas_*.c
	rm -f $(GEN_DIR)/*.json
	rm -f *.riscv
	@echo "Done."

help:
	@echo "Available targets:"
	@echo "  cicd               Build all tests and move binaries to root"
	@echo "  build-assembly     Build all tests in build directory"
	@echo "  clean              Remove all generated files and build directories"
	@echo "  list-tests         Show all detected tests in $(ASM_DIR)"

list-tests:
	@echo "Detected tests:"
	@for test in $(TESTS); do echo "  - $$test"; done
