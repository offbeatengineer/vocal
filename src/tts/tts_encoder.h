#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>
#include <string>
#include <cstdint>
#include <map>
#include <memory>

namespace vocal_tts {

// --- Speaker Encoder (GGML-based ECAPA-TDNN) ---

struct speaker_encoder_config {
    int32_t sample_rate = 24000;
    int32_t n_mels = 128;
    int32_t n_fft = 1024;
    int32_t hop_length = 256;
    int32_t win_length = 1024;
    int32_t embedding_dim = 1024;
    int32_t hidden_dim = 512;
    int32_t n_res2net_blocks = 3;
    int32_t res2net_scale = 8;
    float f_min = 0.0f;
    float f_max = 12000.0f;
};

struct res2net_block {
    struct ggml_tensor * tdnn1_w = nullptr;
    struct ggml_tensor * tdnn1_b = nullptr;
    struct ggml_tensor * res2net_w[7] = {nullptr};
    struct ggml_tensor * res2net_b[7] = {nullptr};
    struct ggml_tensor * tdnn2_w = nullptr;
    struct ggml_tensor * tdnn2_b = nullptr;
    struct ggml_tensor * se_conv1_w = nullptr;
    struct ggml_tensor * se_conv1_b = nullptr;
    struct ggml_tensor * se_conv2_w = nullptr;
    struct ggml_tensor * se_conv2_b = nullptr;
};

struct speaker_encoder_model {
    speaker_encoder_config config;

    struct ggml_tensor * conv0_w = nullptr;
    struct ggml_tensor * conv0_b = nullptr;
    res2net_block blocks[3];

    struct ggml_tensor * mfa_w = nullptr;
    struct ggml_tensor * mfa_b = nullptr;

    struct ggml_tensor * asp_conv_w = nullptr;
    struct ggml_tensor * asp_conv_b = nullptr;
    struct ggml_tensor * asp_tdnn_w = nullptr;
    struct ggml_tensor * asp_tdnn_b = nullptr;

    struct ggml_tensor * fc_w = nullptr;
    struct ggml_tensor * fc_b = nullptr;

    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct speaker_encoder_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
};

class SpeakerEncoder {
public:
    SpeakerEncoder();
    ~SpeakerEncoder();

    // Load from GGUF file (main model GGUF containing spk_enc.* tensors)
    bool load(const std::string & model_path);

    // Encode reference audio to speaker embedding
    // audio: float32 mono samples at 24kHz
    // Returns: speaker embedding vector (1024-dim)
    std::vector<float> encode(const float * audio, int n_samples);

    bool is_loaded() const { return model_.ctx != nullptr; }
    const std::string & get_error() const { return error_; }

private:
    bool compute_mel_spectrogram(const float * samples, int32_t n_samples,
                                  std::vector<float> & mel, int32_t & n_frames);

    struct ggml_cgraph * build_graph(int32_t n_frames);

    speaker_encoder_model model_;
    speaker_encoder_state state_;
    std::string error_;
    bool loaded_ = false;
};

void free_speaker_encoder_model(speaker_encoder_model & model);

// --- Codec Encoder (ONNX-based, optional) ---

#ifdef VOCAL_ONNX_CODEC_ENCODER

class CodecEncoder {
public:
    CodecEncoder();
    ~CodecEncoder();

    bool load(const std::string & model_path);

    // Encode audio to multi-codebook codes
    // audio: float32 mono samples at 24kHz
    // Returns: [16][T] codec codes
    std::vector<std::vector<int32_t>> encode(const float * audio, int n_samples);

    bool is_loaded() const { return loaded_; }
    const std::string & get_error() const { return error_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string error_;
    bool loaded_ = false;
};

#endif // VOCAL_ONNX_CODEC_ENCODER

} // namespace vocal_tts
