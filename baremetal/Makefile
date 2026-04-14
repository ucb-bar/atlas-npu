PYTHON ?= python3
CONFIG ?= EE290SimConfig
VCS_DIR ?= sims/vcs
RUN_TIMEOUT_S ?= 0
STRESS_COUNT ?= 12
SEED ?= 1337
TIERS_ALL := tier0,tier1,tier2,tier3

CHIPYARD_ROOT := $(abspath ../../..)
GEN_SCRIPT := $(CURDIR)/combo_suite/generate_baremetal_suite.py
RUN_SCRIPT := $(CURDIR)/run_combo_suite.py
TESTS_DIR := $(CURDIR)/tests
BUILD_DIR := $(TESTS_DIR)/build

.PHONY: cicd gen list-tests cmake build build-smoke build-all run run-smoke run-all run-test clean-results

cicd: gen build-all
	mv "$(BUILD_DIR)"/*.riscv "$(CURDIR)/"

gen:
	$(PYTHON) "$(GEN_SCRIPT)" --stress-count $(STRESS_COUNT) --seed $(SEED)

list-tests:
	$(PYTHON) "$(RUN_SCRIPT)" --tiers $(TIERS_ALL) --list

cmake:
	cmake -S "$(TESTS_DIR)" -B "$(BUILD_DIR)" -D CMAKE_BUILD_TYPE=Debug

build: build-smoke

build-smoke: cmake
	cmake --build "$(BUILD_DIR)" --target atlas-smoke

build-all: cmake
	cmake --build "$(BUILD_DIR)" --target atlas-combo

run: run-smoke

run-smoke:
	cd "$(CHIPYARD_ROOT)" && $(PYTHON) "$(RUN_SCRIPT)" --tiers tier0 --config "$(CONFIG)" --vcs-dir "$(VCS_DIR)" --run-timeout-s $(RUN_TIMEOUT_S) --execute

run-all:
	cd "$(CHIPYARD_ROOT)" && $(PYTHON) "$(RUN_SCRIPT)" --tiers $(TIERS_ALL) --config "$(CONFIG)" --vcs-dir "$(VCS_DIR)" --run-timeout-s $(RUN_TIMEOUT_S) --execute

run-test:
	@if [ -z "$(TEST)" ]; then echo "Usage: make run-test TEST=<short_test_name>"; exit 1; fi
	cd "$(CHIPYARD_ROOT)" && $(PYTHON) "$(RUN_SCRIPT)" --tiers $(TIERS_ALL) --test "$(TEST)" --config "$(CONFIG)" --vcs-dir "$(VCS_DIR)" --run-timeout-s $(RUN_TIMEOUT_S) --execute

clean-results:
	rm -rf "$(CURDIR)/results"
