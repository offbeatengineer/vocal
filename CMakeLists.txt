cmake_minimum_required(VERSION 3.14)
project(vocal VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()

# Options
option(VOCAL_TIMING "Enable detailed timing instrumentation" OFF)
if(VOCAL_TIMING)
    add_compile_definitions(QWEN3_ASR_TIMING)
endif()

find_package(Threads REQUIRED)

# --- GGML ---
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml")
set(GGML_BUILD_DIR "${GGML_DIR}/build")

# --- Mel spectrogram library ---
add_library(vocal_mel STATIC
    src/asr/mel_spectrogram.cpp
)
target_include_directories(vocal_mel PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asr
)
if(APPLE)
    target_link_libraries(vocal_mel PRIVATE Threads::Threads "-framework Accelerate")
else()
    target_link_libraries(vocal_mel PRIVATE Threads::Threads)
endif()

# --- Audio encoder library (GGML-based) ---
add_library(vocal_encoder STATIC
    src/asr/gguf_loader.cpp
    src/asr/audio_encoder.cpp
)
target_include_directories(vocal_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asr
    ${GGML_DIR}/include
)
target_link_directories(vocal_encoder PUBLIC ${GGML_BUILD_DIR}/src)
target_link_libraries(vocal_encoder PUBLIC ggml ggml-base ggml-cpu Threads::Threads)

# --- Text decoder library (GGML-based) ---
add_library(vocal_decoder STATIC
    src/asr/text_decoder.cpp
)
target_include_directories(vocal_decoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asr
    ${GGML_DIR}/include
)
target_link_directories(vocal_decoder PUBLIC ${GGML_BUILD_DIR}/src)
target_link_libraries(vocal_decoder PUBLIC ggml ggml-base ggml-cpu Threads::Threads)

# --- Audio injection library (pure C++) ---
add_library(vocal_injection STATIC
    src/asr/audio_injection.cpp
)
target_include_directories(vocal_injection PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asr
)

# --- ASR pipeline library ---
add_library(vocal_asr STATIC
    src/asr/qwen3_asr.cpp
    src/asr/vocal_asr_bridge.cpp
)
target_include_directories(vocal_asr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asr
    ${GGML_DIR}/include
)
target_link_libraries(vocal_asr PUBLIC
    vocal_mel
    vocal_encoder
    vocal_decoder
    vocal_injection
    Threads::Threads
)

# --- Audio I/O library (dr_wav) ---
add_library(vocal_audio STATIC
    src/audio/audio_io.c
)
target_include_directories(vocal_audio PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/audio
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# --- Model management ---
add_library(vocal_model STATIC
    src/vocal_model.c
)
target_include_directories(vocal_model PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# --- TTS common utilities ---
add_library(vocal_tts_common STATIC
    src/tts/tts_common.cpp
)
target_include_directories(vocal_tts_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tts
    ${GGML_DIR}/include
)
target_link_directories(vocal_tts_common PUBLIC ${GGML_BUILD_DIR}/src)
target_link_libraries(vocal_tts_common PUBLIC ggml ggml-base ggml-cpu)

# --- TTS tokenizer ---
add_library(vocal_tts_tokenizer STATIC
    src/tts/tts_tokenizer.cpp
)
target_include_directories(vocal_tts_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tts
)

# --- TTS model (Talker LLM) ---
add_library(vocal_tts_model STATIC
    src/tts/tts_model.cpp
)
target_include_directories(vocal_tts_model PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tts
    ${GGML_DIR}/include
)
target_link_directories(vocal_tts_model PUBLIC ${GGML_BUILD_DIR}/src)
target_link_libraries(vocal_tts_model PUBLIC ggml ggml-base ggml-cpu vocal_tts_common)

# --- ONNX Runtime ---
set(ONNXRT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
if(EXISTS "${ONNXRT_DIR}/include/onnxruntime_c_api.h")
    set(ONNXRT_FOUND TRUE)
    message(STATUS "Found ONNX Runtime at ${ONNXRT_DIR}")
else()
    set(ONNXRT_FOUND FALSE)
    message(WARNING "ONNX Runtime not found at ${ONNXRT_DIR} — TTS decoder will be disabled")
endif()

# --- TTS codec (audio decoder via ONNX Runtime) ---
add_library(vocal_tts_codec STATIC
    src/tts/tts_codec.cpp
)
target_include_directories(vocal_tts_codec PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tts
)
if(ONNXRT_FOUND)
    target_include_directories(vocal_tts_codec PUBLIC ${ONNXRT_DIR}/include)
    target_link_directories(vocal_tts_codec PUBLIC ${ONNXRT_DIR}/lib)
    target_link_libraries(vocal_tts_codec PUBLIC onnxruntime)
endif()

# --- TTS pipeline ---
add_library(vocal_tts STATIC
    src/tts/tts.cpp
    src/tts/vocal_tts_bridge.cpp
)
target_include_directories(vocal_tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tts
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${GGML_DIR}/include
)
target_link_libraries(vocal_tts PUBLIC
    vocal_tts_tokenizer
    vocal_tts_model
    vocal_tts_codec
    vocal_tts_common
    Threads::Threads
)

# --- CLI executable ---
# main.c is C but links C++ libraries, so we need CXX linkage
add_executable(vocal
    src/main.c
)
set_target_properties(vocal PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(vocal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asr
    ${GGML_DIR}/include
)
target_link_libraries(vocal PRIVATE
    vocal_asr
    vocal_tts
    vocal_audio
    vocal_model
)

# Link Metal framework on macOS for GPU
if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALKIT_FRAMEWORK MetalKit)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    if(METAL_FRAMEWORK)
        target_link_libraries(vocal PRIVATE ${METAL_FRAMEWORK} ${METALKIT_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    endif()
endif()

install(TARGETS vocal RUNTIME DESTINATION bin)
