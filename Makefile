
BUILD_DIR = ./generated-src
MILL = ./mill
PROJECT = atlas

.PHONY: verilog test help reformat checkformat clean

verilog:
	$(MILL) -i $(PROJECT).verilog $(abspath $(BUILD_DIR))

test:
	$(MILL) -i $(PROJECT).test
	$(MILL) -i $(PROJECT).pmuTest.test

help:
	$(MILL) -i $(PROJECT).verilog $(abspath $(BUILD_DIR)) --help

reformat:
	$(MILL) -i __.reformat

check-format:
	$(MILL) -i __.checkFormat

clean:
	-rm -rf $(BUILD_DIR)
