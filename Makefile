CXX ?= c++
UNAME_S := $(shell uname -s)
ACCELERATE ?= 1
ACCELERATE_CXXFLAGS :=
ACCELERATE_LDLIBS :=
ifeq ($(UNAME_S),Darwin)
ifeq ($(ACCELERATE),1)
ACCELERATE_CXXFLAGS := -DCCZERO_USE_ACCELERATE -DACCELERATE_NEW_LAPACK
ACCELERATE_LDLIBS := -framework Accelerate
endif
endif
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic -Iinclude $(ACCELERATE_CXXFLAGS)
RELEASE_CXXFLAGS ?= -std=c++20 -O3 -mcpu=native -flto -DNDEBUG -DCCZERO_RELEASE_BUILD -DCCZERO_NATIVE_BUILD -Wall -Wextra -Wpedantic -Iinclude $(ACCELERATE_CXXFLAGS)
LDLIBS ?= $(ACCELERATE_LDLIBS)
BUILD_DIR := build

.PHONY: all release benchmark-efficiency test check clean

all: $(BUILD_DIR)/cczero $(BUILD_DIR)/cczero_tests

release:
	$(MAKE) clean
	$(MAKE) all CXXFLAGS="$(RELEASE_CXXFLAGS)"

benchmark-efficiency: $(BUILD_DIR)/cczero
	tools/benchmark_efficiency.py --engine ./$(BUILD_DIR)/cczero

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/cczero.o: src/cczero.cpp include/cczero/cczero.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c src/cczero.cpp -o $@

$(BUILD_DIR)/model.o: src/model.cpp include/cczero/model.h include/cczero/cczero.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c src/model.cpp -o $@

$(BUILD_DIR)/mcts.o: src/mcts.cpp include/cczero/mcts.h include/cczero/model.h include/cczero/cczero.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c src/mcts.cpp -o $@

$(BUILD_DIR)/cli_utils.o: src/cli_utils.cpp include/cczero/cli_utils.h include/cczero/cczero.h include/cczero/mcts.h include/cczero/model.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c src/cli_utils.cpp -o $@

$(BUILD_DIR)/main.o: src/main.cpp include/cczero/cczero.h include/cczero/cli.h include/cczero/cli_utils.h include/cczero/model.h include/cczero/mcts.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c src/main.cpp -o $@

$(BUILD_DIR)/cli_main.o: src/cli_main.cpp include/cczero/cli.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c src/cli_main.cpp -o $@

$(BUILD_DIR)/core_tests.o: tests/core_tests.cpp include/cczero/cczero.h include/cczero/model.h include/cczero/mcts.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c tests/core_tests.cpp -o $@

$(BUILD_DIR)/cczero: $(BUILD_DIR)/cczero.o $(BUILD_DIR)/model.o $(BUILD_DIR)/mcts.o $(BUILD_DIR)/cli_utils.o $(BUILD_DIR)/main.o $(BUILD_DIR)/cli_main.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD_DIR)/cczero_tests: $(BUILD_DIR)/cczero.o $(BUILD_DIR)/model.o $(BUILD_DIR)/mcts.o $(BUILD_DIR)/cli_utils.o $(BUILD_DIR)/core_tests.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

test: $(BUILD_DIR)/cczero_tests
	./$(BUILD_DIR)/cczero_tests

check:
	$(MAKE) clean
	$(MAKE) all
	./$(BUILD_DIR)/cczero_tests
	tools/check_all.py --skip-build

clean:
	rm -rf $(BUILD_DIR)
