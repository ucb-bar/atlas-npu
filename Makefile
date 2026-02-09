
BUILD_DIR = ./generated-src
MILL = ./toolchains/mill
PROJECT = atlas
MAIN = atlas.Elaborate

.PHONY: verilog test help reformat checkformat clean

verilog:
	mkdir -p $(BUILD_DIR)
	$(MILL) -i $(PROJECT).runMain $(MAIN) --target-dir $(BUILD_DIR)

test:
	$(MILL) -i $(PROJECT).test

help:
	$(MILL) -i $(PROJECT).runMain $(MAIN) --help

reformat:
	$(MILL) -i __.reformat

check-format:
	$(MILL) -i __.checkFormat

clean:
	-rm -rf $(BUILD_DIR)
