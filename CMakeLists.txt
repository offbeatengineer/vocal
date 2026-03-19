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

# Detect Metal backend
if(APPLE)
    if(EXISTS "${GGML_BUILD_DIR}/src/ggml-metal/libggml-metal.dylib" OR
       EXISTS "${GGML_BUILD_DIR}/src/ggml-metal/libggml-metal.a")
        set(GGML_HAS_METAL ON)
        message(STATUS "Found GGML Metal backend")
    else()
        set(GGML_HAS_METAL OFF)
    endif()
endif()

# GGML link helper
set(GGML_LIBS ggml ggml-base ggml-cpu)
set(GGML_LINK_DIRS ${GGML_BUILD_DIR}/src)
if(GGML_HAS_METAL)
    list(APPEND GGML_LIBS ggml-metal)
    list(APPEND GGML_LINK_DIRS ${GGML_BUILD_DIR}/src/ggml-metal)
endif()
# Detect BLAS backend (Accelerate on macOS)
if(EXISTS "${GGML_BUILD_DIR}/src/ggml-blas/libggml-blas.dylib" OR
   EXISTS "${GGML_BUILD_DIR}/src/ggml-blas/libggml-blas.a")
    set(GGML_HAS_BLAS ON)
    list(APPEND GGML_LIBS ggml-blas)
    list(APPEND GGML_LINK_DIRS ${GGML_BUILD_DIR}/src/ggml-blas)
    message(STATUS "Found GGML BLAS backend")
endif()


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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/audio
    ${GGML_DIR}/include
)
target_link_libraries(vocal_asr PUBLIC
    vocal_mel
    vocal_encoder
    vocal_decoder
    vocal_injection
    vocal_audio
    Threads::Threads
)

# --- Audio I/O library (dr_wav, dr_mp3, dr_flac) ---
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

# --- TTS GGUF loader ---
add_library(vocal_tts_gguf_loader STATIC
    src/tts/gguf_loader.cpp
)
target_include_directories(vocal_tts_gguf_loader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tts
    ${GGML_DIR}/include
)
target_link_directories(vocal_tts_gguf_loader PUBLIC ${GGML_LINK_DIRS})
target_link_libraries(vocal_tts_gguf_loader PUBLIC ${GGML_LIBS})

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

# --- TTS codec (audio decoder via GGML) ---
add_library(vocal_tts_codec STATIC
    src/tts/tts_codec.cpp
)
target_include_directories(vocal_tts_codec PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tts
    ${GGML_DIR}/include
)
target_link_directories(vocal_tts_codec PUBLIC ${GGML_LINK_DIRS})
target_link_libraries(vocal_tts_codec PUBLIC ${GGML_LIBS} vocal_tts_gguf_loader)

# --- TTS encoder (speaker encoder + codec encoder via GGML) ---
add_library(vocal_tts_encoder STATIC
    src/tts/tts_encoder.cpp
)
target_include_directories(vocal_tts_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tts
    ${GGML_DIR}/include
)
target_link_directories(vocal_tts_encoder PUBLIC ${GGML_LINK_DIRS})
target_link_libraries(vocal_tts_encoder PUBLIC ${GGML_LIBS} vocal_tts_gguf_loader)
# --- Voice profile I/O ---
add_library(vocal_voice_profile STATIC
    src/tts/voice_profile.cpp
)
target_include_directories(vocal_voice_profile PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tts
)

# --- TTS pipeline ---
add_library(vocal_tts STATIC
    src/tts/tts.cpp
    src/tts/vocal_tts_bridge.cpp
)
target_include_directories(vocal_tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tts
    ${CMAKE_CURRENT_SOURCE_DIR}/src/audio
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${GGML_DIR}/include
)
target_link_libraries(vocal_tts PUBLIC
    vocal_tts_tokenizer
    vocal_tts_model
    vocal_tts_codec
    vocal_tts_encoder
    vocal_tts_common
    vocal_voice_profile
    vocal_audio
    Threads::Threads
)

# --- HTTP server ---
add_library(vocal_server STATIC
    src/server/vocal_server.cpp
)
target_include_directories(vocal_server PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/server
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asr
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tts
    ${CMAKE_CURRENT_SOURCE_DIR}/src/audio
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${GGML_DIR}/include
)
target_link_libraries(vocal_server PUBLIC
    vocal_asr
    vocal_tts
    vocal_audio
    vocal_model
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
    vocal_server
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
