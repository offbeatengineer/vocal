#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>
#include <string>
#include <cstdint>
#include <map>

namespace vocal_tts {

struct DecoderConfig {
    int32_t sample_rate = 24000;
    int32_t n_codebooks = 16;
    int32_t codebook_size = 2048;
    int32_t codebook_dim = 256;
    int32_t latent_dim = 1024;
    int32_t hidden_dim = 512;
    int32_t n_pre_tfm_layers = 8;
    int32_t n_heads = 16;
    int32_t ffn_dim = 1024;
    int32_t decoder_dim = 1536;
    int32_t upsample_rates[4] = {8, 5, 4, 3};
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    float frame_rate = 12.5f;
};

struct pre_tfm_layer {
    struct ggml_tensor * attn_norm_w = nullptr;
    struct ggml_tensor * attn_q_w = nullptr;
    struct ggml_tensor * attn_k_w = nullptr;
    struct ggml_tensor * attn_v_w = nullptr;
    struct ggml_tensor * attn_output_w = nullptr;
    struct ggml_tensor * attn_scale = nullptr;
    struct ggml_tensor * ffn_norm_w = nullptr;
    struct ggml_tensor * ffn_gate_w = nullptr;
    struct ggml_tensor * ffn_up_w = nullptr;
    struct ggml_tensor * ffn_down_w = nullptr;
    struct ggml_tensor * ffn_scale = nullptr;
};

struct residual_block {
    int dilation = 1;
    struct ggml_tensor * act1_alpha = nullptr;
    struct ggml_tensor * act1_beta = nullptr;
    struct ggml_tensor * conv1_w = nullptr;
    struct ggml_tensor * conv1_b = nullptr;
    struct ggml_tensor * act2_alpha = nullptr;
    struct ggml_tensor * act2_beta = nullptr;
    struct ggml_tensor * conv2_w = nullptr;
    struct ggml_tensor * conv2_b = nullptr;
};

struct decoder_block {
    struct ggml_tensor * snake_alpha = nullptr;
    struct ggml_tensor * snake_beta = nullptr;
    struct ggml_tensor * conv_t_w = nullptr;
    struct ggml_tensor * conv_t_b = nullptr;
    residual_block res[3];
};

struct upsample_block {
    struct ggml_tensor * conv_w = nullptr;
    struct ggml_tensor * conv_b = nullptr;
    struct ggml_tensor * dwconv_w = nullptr;
    struct ggml_tensor * dwconv_b = nullptr;
    struct ggml_tensor * norm_w = nullptr;
    struct ggml_tensor * norm_b = nullptr;
    struct ggml_tensor * pwconv1_w = nullptr;
    struct ggml_tensor * pwconv1_b = nullptr;
    struct ggml_tensor * pwconv2_w = nullptr;
    struct ggml_tensor * pwconv2_b = nullptr;
    struct ggml_tensor * gamma = nullptr;
};

struct audio_decoder_model {
    DecoderConfig config;

    // VQ codebooks
    struct ggml_tensor * vq_first_input_proj = nullptr;
    struct ggml_tensor * vq_first_output_proj = nullptr;
    struct ggml_tensor * vq_first_codebook = nullptr;
    struct ggml_tensor * vq_first_usage = nullptr;

    struct ggml_tensor * vq_rest_input_proj = nullptr;
    struct ggml_tensor * vq_rest_output_proj = nullptr;
    struct ggml_tensor * vq_rest_codebook[15] = {nullptr};
    struct ggml_tensor * vq_rest_usage[15] = {nullptr};

    // Upsample blocks (2 ConvNeXt-style)
    upsample_block upsample[2];

    // Pre-transformer
    struct ggml_tensor * pre_tfm_input_proj_w = nullptr;
    struct ggml_tensor * pre_tfm_input_proj_b = nullptr;
    pre_tfm_layer pre_tfm_layers[8];
    struct ggml_tensor * pre_tfm_norm_w = nullptr;
    struct ggml_tensor * pre_tfm_output_proj_w = nullptr;
    struct ggml_tensor * pre_tfm_output_proj_b = nullptr;

    // Pre-conv
    struct ggml_tensor * pre_conv_w = nullptr;
    struct ggml_tensor * pre_conv_b = nullptr;

    // Decoder blocks
    struct ggml_tensor * dec0_conv_w = nullptr;
    struct ggml_tensor * dec0_conv_b = nullptr;
    decoder_block dec_blocks[4];
    struct ggml_tensor * dec5_snake_alpha = nullptr;
    struct ggml_tensor * dec5_snake_beta = nullptr;
    struct ggml_tensor * dec6_conv_w = nullptr;
    struct ggml_tensor * dec6_conv_b = nullptr;

    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct audio_decoder_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
};

class AudioDecoder {
public:
    AudioDecoder();
    ~AudioDecoder();

    // Load model from GGUF file
    bool load(const std::string & model_path);

    // Release all resources
    void unload();

    // Decode audio codes to waveform
    // codes: [num_codebooks][seq_len] — multi-codebook audio codes
    // Returns PCM float samples at 24kHz
    std::vector<float> decode(const std::vector<std::vector<int32_t>> & codes);

    bool is_loaded() const { return model_.ctx != nullptr; }
    const std::string & get_error() const { return error_; }
    const DecoderConfig & get_config() const { return model_.config; }

private:
    struct ggml_cgraph * build_graph(int32_t n_frames);

    struct ggml_tensor * apply_snake(struct ggml_context * ctx,
                                      struct ggml_tensor * x,
                                      struct ggml_tensor * alpha,
                                      struct ggml_tensor * beta);

    struct ggml_tensor * apply_rms_norm(struct ggml_context * ctx,
                                         struct ggml_tensor * x,
                                         struct ggml_tensor * w,
                                         float eps);

    struct ggml_tensor * apply_pre_tfm_layer(struct ggml_context * ctx,
                                              struct ggml_tensor * x,
                                              const pre_tfm_layer & layer,
                                              int32_t n_frames,
                                              struct ggml_tensor * positions);

    struct ggml_tensor * apply_upsample_block(struct ggml_context * ctx,
                                               struct ggml_tensor * x,
                                               const upsample_block & block,
                                               int block_idx);

    struct ggml_tensor * apply_residual_block(struct ggml_context * ctx,
                                               struct ggml_tensor * x,
                                               const residual_block & block);

    struct ggml_tensor * apply_decoder_block(struct ggml_context * ctx,
                                              struct ggml_tensor * x,
                                              const decoder_block & block,
                                              int upsample_rate,
                                              int block_idx);

    void normalize_codebooks();

    audio_decoder_model model_;
    audio_decoder_state state_;
    std::string error_;

    std::vector<int32_t> codes_buf_;
};

void free_audio_decoder_model(audio_decoder_model & model);

} // namespace vocal_tts
