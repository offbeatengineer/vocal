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

// --- Codec Encoder (GGML-based SEANet + Transformer + RVQ) ---

struct codec_encoder_config {
    int32_t hidden_size = 512;
    int32_t num_filters = 64;
    int32_t num_layers = 8;         // Transformer layers
    int32_t num_heads = 8;
    int32_t head_dim = 64;
    int32_t intermediate_size = 2048;
    int32_t codebook_dim = 256;
    int32_t codebook_size = 2048;
    int32_t num_quantizers = 32;
    int32_t valid_quantizers = 16;
    int32_t sample_rate = 24000;
    int32_t sliding_window = 250;
    float rope_theta = 10000.0f;
};

struct codec_enc_resblock {
    struct ggml_tensor * conv1_w = nullptr;  // blk.1
    struct ggml_tensor * conv1_b = nullptr;
    struct ggml_tensor * conv2_w = nullptr;  // blk.3
    struct ggml_tensor * conv2_b = nullptr;
};

struct codec_enc_transformer_layer {
    struct ggml_tensor * attn_norm_w = nullptr;
    struct ggml_tensor * attn_norm_b = nullptr;
    struct ggml_tensor * attn_q_w = nullptr;
    struct ggml_tensor * attn_k_w = nullptr;
    struct ggml_tensor * attn_v_w = nullptr;
    struct ggml_tensor * attn_o_w = nullptr;
    struct ggml_tensor * attn_scale = nullptr;   // LayerScale
    struct ggml_tensor * ffn_norm_w = nullptr;
    struct ggml_tensor * ffn_norm_b = nullptr;
    struct ggml_tensor * ffn_up_w = nullptr;
    struct ggml_tensor * ffn_down_w = nullptr;
    struct ggml_tensor * ffn_scale = nullptr;     // LayerScale
};

struct codec_encoder_model {
    codec_encoder_config config;

    // SEANet convolutions (layer indices from HuggingFace)
    struct ggml_tensor * conv0_w = nullptr;   // initial conv
    struct ggml_tensor * conv0_b = nullptr;
    struct ggml_tensor * conv3_w = nullptr;   // downsample 4x
    struct ggml_tensor * conv3_b = nullptr;
    struct ggml_tensor * conv6_w = nullptr;   // downsample 5x
    struct ggml_tensor * conv6_b = nullptr;
    struct ggml_tensor * conv9_w = nullptr;   // downsample 6x
    struct ggml_tensor * conv9_b = nullptr;
    struct ggml_tensor * conv12_w = nullptr;  // downsample 8x
    struct ggml_tensor * conv12_b = nullptr;
    struct ggml_tensor * conv14_w = nullptr;  // final projection
    struct ggml_tensor * conv14_b = nullptr;

    // Residual blocks (at layer indices 1, 4, 7, 10)
    codec_enc_resblock res[4];

    // Transformer layers
    codec_enc_transformer_layer blk[8];

    // Downsample conv (no bias, replicate padding)
    struct ggml_tensor * downsample_w = nullptr;

    // VQ codebooks (extracted to CPU for RVQ)
    struct ggml_tensor * vq_semantic_input_proj = nullptr;
    struct ggml_tensor * vq_semantic_codebook = nullptr;
    struct ggml_tensor * vq_acoustic_input_proj = nullptr;
    static constexpr int MAX_ACOUSTIC_CODEBOOKS = 31;
    struct ggml_tensor * vq_acoustic_codebooks[MAX_ACOUSTIC_CODEBOOKS] = {nullptr};

    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, struct ggml_tensor *> tensors;

    // CPU-side copies for RVQ
    std::vector<float> sem_input_proj_data;          // [256 * 512]
    std::vector<float> sem_codebook_data;             // [2048 * 256]
    std::vector<float> acou_input_proj_data;          // [256 * 512]
    std::vector<std::vector<float>> acou_codebook_data; // [n_cb][2048 * 256]
};

struct codec_encoder_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
};

class CodecEncoder {
public:
    CodecEncoder();
    ~CodecEncoder();

    // Load from GGUF file (tokenizer GGUF containing tok_enc.* tensors)
    bool load(const std::string & model_path);

    // Encode audio to multi-codebook codes
    // audio: float32 mono samples at 24kHz
    // Returns: [16][T] codec codes
    std::vector<std::vector<int32_t>> encode(const float * audio, int n_samples);

    bool is_loaded() const { return model_.ctx != nullptr; }
    const std::string & get_error() const { return error_; }

private:
    struct ggml_cgraph * build_graph(int n_samples);

    void rvq_encode(const float * features, int n_frames,
                    std::vector<std::vector<int32_t>> & out_codes);

    codec_encoder_model model_;
    codec_encoder_state state_;
    std::string error_;
};

void free_codec_encoder_model(codec_encoder_model & model);

} // namespace vocal_tts
