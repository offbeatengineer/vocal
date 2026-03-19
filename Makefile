.PHONY: all clean ggml

BUILD_DIR = build
GGML_DIR = third_party/ggml
GGML_BUILD_DIR = $(GGML_DIR)/build
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
CMAKE_FLAGS ?= -DVOCAL_VERSION=$(VERSION) -DVOCAL_COMMIT=$(COMMIT)

# Auto-detect Metal on macOS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CMAKE_FLAGS += -DGGML_METAL=ON
endif

all: ggml
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release $(CMAKE_FLAGS)
	cmake --build $(BUILD_DIR) -j$(shell nproc 2>/dev/null || sysctl -n hw.ncpu)
	cp $(BUILD_DIR)/vocal .

ggml:
	@if [ ! -f $(GGML_BUILD_DIR)/src/libggml.a ]; then \
		echo "Building GGML..."; \
		cmake -B $(GGML_BUILD_DIR) -S $(GGML_DIR) \
			-DCMAKE_BUILD_TYPE=Release \
			-DGGML_METAL=$(if $(filter Darwin,$(UNAME_S)),ON,OFF) \
			-DGGML_CUDA=$(if $(VOCAL_CUDA),ON,OFF) \
			$(CMAKE_FLAGS); \
		cmake --build $(GGML_BUILD_DIR) -j$(shell nproc 2>/dev/null || sysctl -n hw.ncpu); \
	fi

clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(GGML_BUILD_DIR)
	rm -f vocal

debug: ggml
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug $(CMAKE_FLAGS)
	cmake --build $(BUILD_DIR) -j$(shell nproc 2>/dev/null || sysctl -n hw.ncpu)
	cp $(BUILD_DIR)/vocal .

timing: ggml
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DVOCAL_TIMING=ON $(CMAKE_FLAGS)
	cmake --build $(BUILD_DIR) -j$(shell nproc 2>/dev/null || sysctl -n hw.ncpu)
	cp $(BUILD_DIR)/vocal .
