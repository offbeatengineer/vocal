#include "tts_encoder.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

#define VOCAL_SPK_MAX_NODES 16384

namespace vocal_tts {

// ============================================================
// Mel spectrogram (slaney normalization, matches librosa/PyTorch)
// ============================================================

static void compute_mel_filterbank_slaney(float * filterbank, int n_mels, int n_fft,
                                           int sample_rate, float f_min, float f_max) {
    auto hz_to_mel_slaney = [](float hz) -> float {
        const float f_sp = 200.0f / 3.0f;
        const float min_log_hz = 1000.0f;
        const float min_log_mel = min_log_hz / f_sp;
        const float logstep = logf(6.4f) / 27.0f;
        if (hz < min_log_hz) {
            return hz / f_sp;
        } else {
            return min_log_mel + logf(hz / min_log_hz) / logstep;
        }
    };

    auto mel_to_hz_slaney = [](float mel) -> float {
        const float f_sp = 200.0f / 3.0f;
        const float min_log_hz = 1000.0f;
        const float min_log_mel = min_log_hz / f_sp;
        const float logstep = logf(6.4f) / 27.0f;
        if (mel < min_log_mel) {
            return f_sp * mel;
        } else {
            return min_log_hz * expf(logstep * (mel - min_log_mel));
        }
    };

    float mel_min = hz_to_mel_slaney(f_min);
    float mel_max = hz_to_mel_slaney(f_max);
    int n_fft_bins = n_fft / 2 + 1;

    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
    }

    std::vector<float> hz_points(n_mels + 2);
    std::vector<float> fft_freqs(n_fft_bins);
    for (int i = 0; i < n_mels + 2; ++i) {
        hz_points[i] = mel_to_hz_slaney(mel_points[i]);
    }
    for (int i = 0; i < n_fft_bins; ++i) {
        fft_freqs[i] = (float)i * sample_rate / n_fft;
    }

    memset(filterbank, 0, n_mels * n_fft_bins * sizeof(float));

    for (int m = 0; m < n_mels; ++m) {
        float f_left = hz_points[m];
        float f_center = hz_points[m + 1];
        float f_right = hz_points[m + 2];
        float enorm = 2.0f / (f_right - f_left);

        for (int k = 0; k < n_fft_bins; ++k) {
            float freq = fft_freqs[k];
            if (freq >= f_left && freq <= f_center) {
                if (f_center > f_left)
                    filterbank[m * n_fft_bins + k] = enorm * (freq - f_left) / (f_center - f_left);
            } else if (freq > f_center && freq <= f_right) {
                if (f_right > f_center)
                    filterbank[m * n_fft_bins + k] = enorm * (f_right - freq) / (f_right - f_center);
            }
        }
    }
}

static void compute_dft(const float * input, float * real, float * imag, int n) {
    for (int k = 0; k < n; ++k) {
        real[k] = 0.0f;
        imag[k] = 0.0f;
        for (int t = 0; t < n; ++t) {
            float angle = -2.0f * (float)M_PI * k * t / n;
            real[k] += input[t] * cosf(angle);
            imag[k] += input[t] * sinf(angle);
        }
    }
}

static void compute_centered_window(float * window, int n_fft, int win_length) {
    memset(window, 0, n_fft * sizeof(float));
    int offset = (n_fft - win_length) / 2;
    for (int i = 0; i < win_length; ++i) {
        window[offset + i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / win_length));
    }
}

// ============================================================
// SpeakerEncoder (GGML-based ECAPA-TDNN)
// ============================================================

SpeakerEncoder::SpeakerEncoder() = default;

SpeakerEncoder::~SpeakerEncoder() {
    free_speaker_encoder_model(model_);

    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend) {
        release_preferred_backend(state_.backend);
        state_.backend = nullptr;
    }
    if (state_.backend_cpu) {
        ggml_backend_free(state_.backend_cpu);
        state_.backend_cpu = nullptr;
    }
}

bool SpeakerEncoder::load(const std::string & model_path) {
    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_ = loader.get_error();
        return false;
    }

    model_.config.sample_rate = loader.get_u32("qwen3-tts.speaker_encoder.sample_rate", 24000);
    model_.config.embedding_dim = loader.get_u32("qwen3-tts.speaker_encoder.embedding_length", 1024);

    int64_t n_tensors = loader.get_n_tensors();
    int spk_tensor_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (name && strncmp(name, "spk_enc.", 8) == 0) {
            spk_tensor_count++;
        }
    }

    if (spk_tensor_count == 0) {
        error_ = "No speaker encoder tensors (spk_enc.*) found in model";
        return false;
    }

    size_t ctx_size = ggml_tensor_overhead() * spk_tensor_count;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_ = "Failed to initialize GGML context";
        return false;
    }

    struct gguf_context * gguf_ctx = loader.get_ctx();
    struct ggml_context * meta_ctx = loader.get_meta_ctx();

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name || strncmp(name, "spk_enc.", 8) != 0) continue;

        struct ggml_tensor * meta_tensor = ggml_get_tensor(meta_ctx, name);
        if (!meta_tensor) continue;

        struct ggml_tensor * tensor = ggml_dup_tensor(model_.ctx, meta_tensor);
        ggml_set_name(tensor, name);
        model_.tensors[name] = tensor;

        std::string sname(name);

        if (sname == "spk_enc.conv0.weight") model_.conv0_w = tensor;
        else if (sname == "spk_enc.conv0.bias") model_.conv0_b = tensor;
        else if (sname == "spk_enc.mfa.weight") model_.mfa_w = tensor;
        else if (sname == "spk_enc.mfa.bias") model_.mfa_b = tensor;
        else if (sname == "spk_enc.asp.conv.weight") model_.asp_conv_w = tensor;
        else if (sname == "spk_enc.asp.conv.bias") model_.asp_conv_b = tensor;
        else if (sname == "spk_enc.asp.tdnn.weight") model_.asp_tdnn_w = tensor;
        else if (sname == "spk_enc.asp.tdnn.bias") model_.asp_tdnn_b = tensor;
        else if (sname == "spk_enc.fc.weight") model_.fc_w = tensor;
        else if (sname == "spk_enc.fc.bias") model_.fc_b = tensor;
        else {
            int blk_idx, res_idx;
            char suffix[64];

            if (sscanf(name, "spk_enc.blk.%d.tdnn1.%s", &blk_idx, suffix) == 2) {
                if (blk_idx >= 1 && blk_idx <= 3) {
                    if (strcmp(suffix, "weight") == 0) model_.blocks[blk_idx-1].tdnn1_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.blocks[blk_idx-1].tdnn1_b = tensor;
                }
            }
            else if (sscanf(name, "spk_enc.blk.%d.tdnn2.%s", &blk_idx, suffix) == 2) {
                if (blk_idx >= 1 && blk_idx <= 3) {
                    if (strcmp(suffix, "weight") == 0) model_.blocks[blk_idx-1].tdnn2_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.blocks[blk_idx-1].tdnn2_b = tensor;
                }
            }
            else if (sscanf(name, "spk_enc.blk.%d.res2net.%d.%s", &blk_idx, &res_idx, suffix) == 3) {
                if (blk_idx >= 1 && blk_idx <= 3 && res_idx >= 0 && res_idx < 7) {
                    if (strcmp(suffix, "weight") == 0) model_.blocks[blk_idx-1].res2net_w[res_idx] = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.blocks[blk_idx-1].res2net_b[res_idx] = tensor;
                }
            }
            else if (sscanf(name, "spk_enc.blk.%d.se.conv1.%s", &blk_idx, suffix) == 2) {
                if (blk_idx >= 1 && blk_idx <= 3) {
                    if (strcmp(suffix, "weight") == 0) model_.blocks[blk_idx-1].se_conv1_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.blocks[blk_idx-1].se_conv1_b = tensor;
                }
            }
            else if (sscanf(name, "spk_enc.blk.%d.se.conv2.%s", &blk_idx, suffix) == 2) {
                if (blk_idx >= 1 && blk_idx <= 3) {
                    if (strcmp(suffix, "weight") == 0) model_.blocks[blk_idx-1].se_conv2_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.blocks[blk_idx-1].se_conv2_b = tensor;
                }
            }
        }
    }

    if (!load_tensor_data_from_file(model_path, gguf_ctx, model_.ctx,
                                     model_.tensors, model_.buffer, error_)) {
        return false;
    }

    state_.backend = init_preferred_backend("SpeakerEncoder", &error_);
    if (!state_.backend) return false;

    ggml_backend_dev_t device = ggml_backend_get_device(state_.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    (void)device_name;

    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!state_.backend_cpu) {
            error_ = "Failed to initialize CPU fallback backend";
            return false;
        }
    }

    std::vector<ggml_backend_t> backends;
    backends.push_back(state_.backend);
    if (state_.backend_cpu) backends.push_back(state_.backend_cpu);
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), VOCAL_SPK_MAX_NODES, false, true);
    if (!state_.sched) {
        error_ = "Failed to create backend scheduler";
        return false;
    }

    state_.compute_meta.resize(ggml_tensor_overhead() * VOCAL_SPK_MAX_NODES + ggml_graph_overhead());

    loaded_ = true;
    return true;
}

bool SpeakerEncoder::compute_mel_spectrogram(const float * samples, int32_t n_samples,
                                              std::vector<float> & mel, int32_t & n_frames) {
    const auto & cfg = model_.config;

    // Reflect padding (matches PyTorch STFT center=True)
    int padding = (cfg.n_fft - cfg.hop_length) / 2;
    int padded_length = n_samples + 2 * padding;

    std::vector<float> padded(padded_length);
    for (int i = 0; i < padded_length; ++i) {
        int src_idx;
        if (i < padding) {
            src_idx = padding - i;
        } else if (i >= padding + n_samples) {
            src_idx = 2 * n_samples - (i - padding) - 2;
        } else {
            src_idx = i - padding;
        }
        src_idx = std::max(0, std::min(n_samples - 1, src_idx));
        padded[i] = samples[src_idx];
    }

    n_frames = (padded_length - cfg.n_fft) / cfg.hop_length + 1;
    if (n_frames <= 0) {
        error_ = "Audio too short for mel spectrogram";
        return false;
    }

    int n_fft_bins = cfg.n_fft / 2 + 1;

    std::vector<float> filterbank(cfg.n_mels * n_fft_bins);
    compute_mel_filterbank_slaney(filterbank.data(), cfg.n_mels, cfg.n_fft,
                                   cfg.sample_rate, cfg.f_min, cfg.f_max);

    std::vector<float> window(cfg.n_fft);
    compute_centered_window(window.data(), cfg.n_fft, cfg.win_length);

    // Output: [n_mels, n_frames] row-major
    mel.resize(cfg.n_mels * n_frames);

    std::vector<float> frame(cfg.n_fft);
    std::vector<float> fft_real(cfg.n_fft);
    std::vector<float> fft_imag(cfg.n_fft);
    std::vector<float> magnitude(n_fft_bins);

    for (int32_t f = 0; f < n_frames; ++f) {
        int start = f * cfg.hop_length;

        for (int i = 0; i < cfg.n_fft; ++i) {
            frame[i] = padded[start + i] * window[i];
        }

        compute_dft(frame.data(), fft_real.data(), fft_imag.data(), cfg.n_fft);

        // Magnitude (not power) — matches PyTorch
        for (int k = 0; k < n_fft_bins; ++k) {
            magnitude[k] = sqrtf(fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k] + 1e-9f);
        }

        // Apply mel filterbank and log compression
        for (int m = 0; m < cfg.n_mels; ++m) {
            float sum = 0.0f;
            for (int k = 0; k < n_fft_bins; ++k) {
                sum += filterbank[m * n_fft_bins + k] * magnitude[k];
            }
            mel[m * n_frames + f] = logf(std::max(sum, 1e-5f));
        }
    }

    return true;
}

// Helper: reflect pad 1D
static struct ggml_tensor * apply_reflect_pad_1d(struct ggml_context * ctx,
                                                  struct ggml_tensor * x,
                                                  int pad) {
    if (pad == 0) return x;

    int64_t T = x->ne[0];
    int64_t C = x->ne[1];
    int64_t B = x->ne[2];

    struct ggml_tensor * left_slices[16];
    struct ggml_tensor * right_slices[16];

    for (int i = 0; i < pad && i < 16; ++i) {
        int left_src_idx = pad - i;
        left_slices[i] = ggml_view_3d(ctx, x, 1, C, B,
                                       x->nb[1], x->nb[2],
                                       left_src_idx * x->nb[0]);
        left_slices[i] = ggml_cont(ctx, left_slices[i]);

        int right_src_idx = T - 2 - i;
        right_slices[i] = ggml_view_3d(ctx, x, 1, C, B,
                                        x->nb[1], x->nb[2],
                                        right_src_idx * x->nb[0]);
        right_slices[i] = ggml_cont(ctx, right_slices[i]);
    }

    struct ggml_tensor * left_pad = left_slices[0];
    for (int i = 1; i < pad && i < 16; ++i) {
        left_pad = ggml_concat(ctx, left_pad, left_slices[i], 0);
    }

    struct ggml_tensor * right_pad = right_slices[0];
    for (int i = 1; i < pad && i < 16; ++i) {
        right_pad = ggml_concat(ctx, right_pad, right_slices[i], 0);
    }

    struct ggml_tensor * padded_tensor = ggml_concat(ctx, left_pad, x, 0);
    padded_tensor = ggml_concat(ctx, padded_tensor, right_pad, 0);

    return padded_tensor;
}

// Helper: conv1d with optional reflect pad
static struct ggml_tensor * apply_conv1d(struct ggml_context * ctx,
                                          struct ggml_tensor * w,
                                          struct ggml_tensor * b,
                                          struct ggml_tensor * x,
                                          int stride, int pad, int dilation,
                                          bool use_reflect_pad = true) {
    struct ggml_tensor * input = x;
    int actual_pad = pad;

    if (use_reflect_pad && pad > 0) {
        input = apply_reflect_pad_1d(ctx, x, pad);
        actual_pad = 0;
    }

    struct ggml_tensor * y = ggml_conv_1d(ctx, w, input, stride, actual_pad, dilation);
    if (b) {
        int64_t oc = y->ne[1];
        y = ggml_add(ctx, y, ggml_reshape_3d(ctx, b, 1, oc, 1));
    }
    return y;
}

struct ggml_cgraph * SpeakerEncoder::build_graph(int32_t n_frames) {
    const auto & cfg = model_.config;
    const int hidden_dim = cfg.hidden_dim;
    const int scale = cfg.res2net_scale;
    const int branch_dim = hidden_dim / scale;

    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, VOCAL_SPK_MAX_NODES, false);

    // Input: mel [n_frames, n_mels] in GGML
    struct ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, cfg.n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    struct ggml_tensor * cur = ggml_reshape_3d(ctx0, mel, n_frames, cfg.n_mels, 1);

    // Initial conv: reflect pad 2, kernel 5
    struct ggml_tensor * mel_padded = apply_reflect_pad_1d(ctx0, cur, 2);
    cur = ggml_conv_1d(ctx0, model_.conv0_w, mel_padded, 1, 0, 1);
    if (model_.conv0_b) {
        int64_t oc = cur->ne[1];
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.conv0_b, 1, oc, 1));
    }
    cur = ggml_relu(ctx0, cur);

    int64_t seq_len = cur->ne[0];

    // Store block outputs for MFA
    struct ggml_tensor * block_outputs[4];
    block_outputs[0] = cur;

    int dilations[3] = {2, 3, 4};

    for (int blk = 0; blk < 3; ++blk) {
        const auto & block = model_.blocks[blk];
        int dilation = dilations[blk];
        struct ggml_tensor * residual = cur;

        // TDNN1
        cur = apply_conv1d(ctx0, block.tdnn1_w, block.tdnn1_b, cur, 1, 0, 1);
        cur = ggml_relu(ctx0, cur);

        // Res2Net branching
        struct ggml_tensor * branches[8];
        for (int b = 0; b < scale; ++b) {
            branches[b] = ggml_view_3d(ctx0, cur,
                                        seq_len, branch_dim, 1,
                                        cur->nb[1], cur->nb[2],
                                        b * branch_dim * cur->nb[1]);
            branches[b] = ggml_cont(ctx0, branches[b]);
        }

        struct ggml_tensor * outputs[8];
        outputs[0] = branches[0];

        for (int b = 1; b < scale; ++b) {
            struct ggml_tensor * input;
            if (b == 1) {
                input = branches[b];
            } else {
                input = ggml_add(ctx0, branches[b], outputs[b - 1]);
            }

            if (block.res2net_w[b - 1]) {
                outputs[b] = apply_conv1d(ctx0, block.res2net_w[b - 1], block.res2net_b[b - 1],
                                          input, 1, dilation, dilation);
                outputs[b] = ggml_relu(ctx0, outputs[b]);
            } else {
                outputs[b] = input;
            }
        }

        // Concatenate branches
        cur = outputs[0];
        for (int b = 1; b < scale; ++b) {
            cur = ggml_concat(ctx0, cur, outputs[b], 1);
        }

        // TDNN2
        cur = apply_conv1d(ctx0, block.tdnn2_w, block.tdnn2_b, cur, 1, 0, 1);
        cur = ggml_relu(ctx0, cur);

        // SE (Squeeze-Excitation)
        struct ggml_tensor * se = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
        se = ggml_reshape_3d(ctx0, se, 1, hidden_dim, 1);
        se = apply_conv1d(ctx0, block.se_conv1_w, block.se_conv1_b, se, 1, 0, 1);
        se = ggml_relu(ctx0, se);
        se = apply_conv1d(ctx0, block.se_conv2_w, block.se_conv2_b, se, 1, 0, 1);
        se = ggml_sigmoid(ctx0, se);
        cur = ggml_mul(ctx0, cur, se);

        // Skip connection
        cur = ggml_add(ctx0, cur, residual);

        block_outputs[blk + 1] = cur;
    }

    // MFA: concatenate block outputs 1-3
    struct ggml_tensor * mfa_input = ggml_concat(ctx0, block_outputs[1], block_outputs[2], 1);
    mfa_input = ggml_concat(ctx0, mfa_input, block_outputs[3], 1);
    cur = apply_conv1d(ctx0, model_.mfa_w, model_.mfa_b, mfa_input, 1, 0, 1);
    cur = ggml_relu(ctx0, cur);

    // ASP (Attentive Statistics Pooling)
    struct ggml_tensor * global_mean = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    global_mean = ggml_reshape_3d(ctx0, global_mean, 1, 1536, 1);

    struct ggml_tensor * sq = ggml_sqr(ctx0, cur);
    struct ggml_tensor * mean_sq = ggml_pool_1d(ctx0, sq, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    mean_sq = ggml_reshape_3d(ctx0, mean_sq, 1, 1536, 1);
    struct ggml_tensor * var = ggml_sub(ctx0, mean_sq, ggml_sqr(ctx0, global_mean));
    var = ggml_clamp(ctx0, var, 1e-12f, 1e10f);
    struct ggml_tensor * global_std = ggml_sqrt(ctx0, var);

    struct ggml_tensor * mean_expanded = ggml_repeat(ctx0, global_mean,
                                                      ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, seq_len, 1536, 1));
    struct ggml_tensor * std_expanded = ggml_repeat(ctx0, global_std,
                                                     ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, seq_len, 1536, 1));

    struct ggml_tensor * attention = ggml_concat(ctx0, cur, mean_expanded, 1);
    attention = ggml_concat(ctx0, attention, std_expanded, 1);

    attention = apply_conv1d(ctx0, model_.asp_tdnn_w, model_.asp_tdnn_b, attention, 1, 0, 1);
    attention = ggml_relu(ctx0, attention);
    attention = ggml_tanh(ctx0, attention);
    attention = apply_conv1d(ctx0, model_.asp_conv_w, model_.asp_conv_b, attention, 1, 0, 1);
    attention = ggml_soft_max(ctx0, attention);

    // Weighted mean
    struct ggml_tensor * weighted = ggml_mul(ctx0, attention, cur);
    struct ggml_tensor * weighted_mean = ggml_pool_1d(ctx0, weighted, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    weighted_mean = ggml_scale(ctx0, weighted_mean, (float)seq_len);
    weighted_mean = ggml_reshape_3d(ctx0, weighted_mean, 1, 1536, 1);

    // Weighted std
    struct ggml_tensor * mean_for_std = ggml_repeat(ctx0, weighted_mean,
                                                     ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, seq_len, 1536, 1));
    struct ggml_tensor * diff = ggml_sub(ctx0, cur, mean_for_std);
    struct ggml_tensor * diff_sq = ggml_sqr(ctx0, diff);
    struct ggml_tensor * weighted_var = ggml_mul(ctx0, attention, diff_sq);
    struct ggml_tensor * var_sum = ggml_pool_1d(ctx0, weighted_var, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    var_sum = ggml_scale(ctx0, var_sum, (float)seq_len);
    var_sum = ggml_reshape_3d(ctx0, var_sum, 1, 1536, 1);
    var_sum = ggml_clamp(ctx0, var_sum, 1e-12f, 1e10f);
    struct ggml_tensor * weighted_std = ggml_sqrt(ctx0, var_sum);

    // Concatenate mean and std: [1, 3072, 1]
    struct ggml_tensor * pooled = ggml_concat(ctx0, weighted_mean, weighted_std, 1);

    // FC: 3072 -> 1024
    cur = apply_conv1d(ctx0, model_.fc_w, model_.fc_b, pooled, 1, 0, 1);

    cur = ggml_reshape_1d(ctx0, cur, cfg.embedding_dim);
    ggml_set_name(cur, "embedding");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);

    return gf;
}

std::vector<float> SpeakerEncoder::encode(const float * audio, int n_samples) {
    if (!model_.ctx) {
        error_ = "Speaker encoder not loaded";
        return {};
    }

    std::vector<float> mel;
    int32_t n_frames;
    if (!compute_mel_spectrogram(audio, n_samples, mel, n_frames)) {
        return {};
    }

    struct ggml_cgraph * gf = build_graph(n_frames);

    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_ = "Failed to allocate graph";
        return {};
    }

    struct ggml_tensor * mel_tensor = ggml_graph_get_tensor(gf, "mel");
    if (!mel_tensor) {
        error_ = "Failed to find mel tensor";
        ggml_backend_sched_reset(state_.sched);
        return {};
    }

    ggml_backend_tensor_set(mel_tensor, mel.data(), 0, mel.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return {};
    }

    struct ggml_tensor * emb_tensor = ggml_graph_get_tensor(gf, "embedding");
    if (!emb_tensor) {
        error_ = "Failed to find embedding tensor";
        ggml_backend_sched_reset(state_.sched);
        return {};
    }

    std::vector<float> embedding(model_.config.embedding_dim);
    ggml_backend_tensor_get(emb_tensor, embedding.data(), 0, embedding.size() * sizeof(float));

    ggml_backend_sched_reset(state_.sched);

    return embedding;
}

void free_speaker_encoder_model(speaker_encoder_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
}

// ============================================================
// CodecEncoder (GGML-based SEANet + Transformer + RVQ)
// ============================================================

#define VOCAL_CODEC_ENC_MAX_NODES 8192

CodecEncoder::CodecEncoder() = default;

CodecEncoder::~CodecEncoder() {
    free_codec_encoder_model(model_);

    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend) {
        release_preferred_backend(state_.backend);
        state_.backend = nullptr;
    }
    if (state_.backend_cpu) {
        ggml_backend_free(state_.backend_cpu);
        state_.backend_cpu = nullptr;
    }
}

bool CodecEncoder::load(const std::string & model_path) {
    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_ = loader.get_error();
        return false;
    }

    // Read config from GGUF metadata
    auto & cfg = model_.config;
    cfg.hidden_size = loader.get_u32("qwen3-tts-tokenizer.encoder.hidden_size", 512);
    cfg.num_layers = loader.get_u32("qwen3-tts-tokenizer.encoder.num_layers", 8);
    cfg.num_heads = loader.get_u32("qwen3-tts-tokenizer.encoder.num_heads", 8);
    cfg.head_dim = cfg.hidden_size / cfg.num_heads;
    cfg.codebook_dim = loader.get_u32("qwen3-tts-tokenizer.encoder.codebook_dim", 256);
    cfg.codebook_size = loader.get_u32("qwen3-tts-tokenizer.codebook_size", 2048);
    cfg.num_quantizers = loader.get_u32("qwen3-tts-tokenizer.encoder.num_quantizers", 32);
    cfg.valid_quantizers = loader.get_u32("qwen3-tts-tokenizer.encoder.valid_quantizers", 16);

    // Count tok_enc.* tensors
    int64_t n_tensors = loader.get_n_tensors();
    int enc_tensor_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name) continue;
        if (strncmp(name, "tok_enc.", 8) == 0) enc_tensor_count++;
        else if (strncmp(name, "tok_dec.vq_first.", 17) == 0) enc_tensor_count++;
        else if (strncmp(name, "tok_dec.vq_rest.", 16) == 0) enc_tensor_count++;
    }
    if (enc_tensor_count == 0) {
        error_ = "No codec encoder tensors (tok_enc.*) found";
        return false;
    }

    size_t ctx_size = ggml_tensor_overhead() * enc_tensor_count;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    model_.ctx = ggml_init(params);
    if (!model_.ctx) { error_ = "Failed to init GGML context"; return false; }

    struct gguf_context * gguf_ctx = loader.get_ctx();
    struct ggml_context * meta_ctx = loader.get_meta_ctx();

    // Map tensor names to model struct fields
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name) continue;
        bool is_enc = strncmp(name, "tok_enc.", 8) == 0;
        bool is_dec_cb = strncmp(name, "tok_dec.vq_first.", 17) == 0 ||
                         strncmp(name, "tok_dec.vq_rest.", 16) == 0;
        if (!is_enc && !is_dec_cb) continue;

        struct ggml_tensor * meta = ggml_get_tensor(meta_ctx, name);
        if (!meta) continue;
        struct ggml_tensor * t = ggml_dup_tensor(model_.ctx, meta);
        ggml_set_name(t, name);
        model_.tensors[name] = t;

        std::string sname(name);
        int idx, sub;
        char suffix[64];

        // SEANet convolutions
        if (sscanf(name, "tok_enc.conv.%d.%s", &idx, suffix) == 2) {
            struct ggml_tensor ** w = nullptr, ** b = nullptr;
            if (idx == 0)  { w = &model_.conv0_w;  b = &model_.conv0_b; }
            if (idx == 3)  { w = &model_.conv3_w;  b = &model_.conv3_b; }
            if (idx == 6)  { w = &model_.conv6_w;  b = &model_.conv6_b; }
            if (idx == 9)  { w = &model_.conv9_w;  b = &model_.conv9_b; }
            if (idx == 12) { w = &model_.conv12_w; b = &model_.conv12_b; }
            if (idx == 14) { w = &model_.conv14_w; b = &model_.conv14_b; }
            if (w && strcmp(suffix, "weight") == 0) *w = t;
            if (b && strcmp(suffix, "bias") == 0)   *b = t;
        }
        // ResNet blocks
        else if (sscanf(name, "tok_enc.res.%d.blk.%d.%s", &idx, &sub, suffix) == 3) {
            int ri = -1;
            if (idx == 1)  ri = 0;
            if (idx == 4)  ri = 1;
            if (idx == 7)  ri = 2;
            if (idx == 10) ri = 3;
            if (ri >= 0) {
                if (sub == 1 && strcmp(suffix, "weight") == 0) model_.res[ri].conv1_w = t;
                if (sub == 1 && strcmp(suffix, "bias") == 0)   model_.res[ri].conv1_b = t;
                if (sub == 3 && strcmp(suffix, "weight") == 0) model_.res[ri].conv2_w = t;
                if (sub == 3 && strcmp(suffix, "bias") == 0)   model_.res[ri].conv2_b = t;
            }
        }
        // Transformer blocks
        else if (sscanf(name, "tok_enc.blk.%d.%s", &idx, suffix) == 2 && idx < 8) {
            auto & blk = model_.blk[idx];
            if (strcmp(suffix, "attn_norm.weight") == 0) blk.attn_norm_w = t;
            else if (strcmp(suffix, "attn_norm.bias") == 0) blk.attn_norm_b = t;
            else if (strcmp(suffix, "attn_q.weight") == 0) blk.attn_q_w = t;
            else if (strcmp(suffix, "attn_k.weight") == 0) blk.attn_k_w = t;
            else if (strcmp(suffix, "attn_v.weight") == 0) blk.attn_v_w = t;
            else if (strcmp(suffix, "attn_output.weight") == 0) blk.attn_o_w = t;
            else if (strcmp(suffix, "attn_scale") == 0) blk.attn_scale = t;
            else if (strcmp(suffix, "ffn_norm.weight") == 0) blk.ffn_norm_w = t;
            else if (strcmp(suffix, "ffn_norm.bias") == 0) blk.ffn_norm_b = t;
            else if (strcmp(suffix, "ffn_up.weight") == 0) blk.ffn_up_w = t;
            else if (strcmp(suffix, "ffn_down.weight") == 0) blk.ffn_down_w = t;
            else if (strcmp(suffix, "ffn_scale") == 0) blk.ffn_scale = t;
        }
        // Downsample
        else if (sname == "tok_enc.downsample.weight") model_.downsample_w = t;
        // VQ (use decoder codebooks for compatibility with decoder reconstruction)
        else if (sname == "tok_enc.vq_semantic.input_proj.weight") model_.vq_semantic_input_proj = t;
        else if (sname == "tok_dec.vq_first.0.codebook") model_.vq_semantic_codebook = t;
        else if (sname == "tok_enc.vq_acoustic.input_proj.weight") model_.vq_acoustic_input_proj = t;
        else if (sscanf(name, "tok_dec.vq_rest.%d.codebook", &idx) == 1) {
            if (idx >= 0 && idx < codec_encoder_model::MAX_ACOUSTIC_CODEBOOKS)
                model_.vq_acoustic_codebooks[idx] = t;
        }
    }

    // Verify essential tensors
    if (!model_.conv0_w || !model_.conv14_w || !model_.downsample_w) {
        error_ = "Missing essential SEANet encoder tensors";
        return false;
    }
    if (!model_.blk[0].attn_q_w) {
        error_ = "Missing transformer tensors";
        return false;
    }
    if (!model_.vq_semantic_codebook || !model_.vq_acoustic_codebooks[0]) {
        error_ = "Missing VQ codebook tensors";
        return false;
    }

    if (!load_tensor_data_from_file(model_path, gguf_ctx, model_.ctx,
                                     model_.tensors, model_.buffer, error_)) {
        return false;
    }

    // Extract VQ weights to CPU for RVQ computation
    auto extract_f16 = [](struct ggml_tensor * t, std::vector<float> & out) {
        int64_t n = ggml_nelements(t);
        out.resize(n);
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<uint8_t> raw(ggml_nbytes(t));
            ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
            const ggml_fp16_t * src = (const ggml_fp16_t *)raw.data();
            for (int64_t j = 0; j < n; j++) out[j] = ggml_fp16_to_fp32(src[j]);
        } else {
            // For other types, read raw and hope for the best
            ggml_backend_tensor_get(t, out.data(), 0, std::min(ggml_nbytes(t), (size_t)(n * sizeof(float))));
        }
    };

    if (model_.vq_semantic_input_proj)
        extract_f16(model_.vq_semantic_input_proj, model_.sem_input_proj_data);
    extract_f16(model_.vq_semantic_codebook, model_.sem_codebook_data);

    if (model_.vq_acoustic_input_proj)
        extract_f16(model_.vq_acoustic_input_proj, model_.acou_input_proj_data);

    int n_acoustic = cfg.valid_quantizers - 1;  // 15
    model_.acou_codebook_data.resize(n_acoustic);
    for (int i = 0; i < n_acoustic; i++) {
        if (model_.vq_acoustic_codebooks[i])
            extract_f16(model_.vq_acoustic_codebooks[i], model_.acou_codebook_data[i]);
    }

    // Init backend
    state_.backend = init_preferred_backend("CodecEncoder", &error_);
    if (!state_.backend) return false;

    ggml_backend_dev_t device = ggml_backend_get_device(state_.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    (void)device_name;

    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!state_.backend_cpu) { error_ = "Failed to init CPU backend"; return false; }
    }

    std::vector<ggml_backend_t> backends;
    backends.push_back(state_.backend);
    if (state_.backend_cpu) backends.push_back(state_.backend_cpu);
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(),
                                           VOCAL_CODEC_ENC_MAX_NODES, false, true);
    if (!state_.sched) { error_ = "Failed to create backend scheduler"; return false; }

    state_.compute_meta.resize(ggml_tensor_overhead() * VOCAL_CODEC_ENC_MAX_NODES + ggml_graph_overhead());

    return true;
}

// Helper: causal conv1d (left zero padding, no right padding)
static struct ggml_tensor * causal_conv1d(struct ggml_context * ctx,
                                           struct ggml_tensor * w,
                                           struct ggml_tensor * b,
                                           struct ggml_tensor * x,
                                           int stride, int dilation) {
    int kernel_size = (int)w->ne[0];
    int padding_total = (kernel_size - 1) * dilation;

    if (padding_total > 0) {
        // Left zero-pad using ggml_pad_ext(lp0, rp0, lp1, rp1, ...)
        x = ggml_pad_ext(ctx, x, padding_total, 0, 0, 0, 0, 0, 0, 0);
    }

    struct ggml_tensor * y = ggml_conv_1d(ctx, w, x, stride, 0, dilation);

    if (b) {
        int64_t oc = y->ne[1];
        y = ggml_add(ctx, y, ggml_reshape_3d(ctx, b, 1, oc, 1));
    }
    return y;
}

// Helper: replicate padding (for downsample conv)
static struct ggml_tensor * replicate_pad_conv1d(struct ggml_context * ctx,
                                                   struct ggml_tensor * w,
                                                   struct ggml_tensor * x,
                                                   int stride) {
    int kernel_size = (int)w->ne[0];
    int padding_total = (kernel_size - 1);  // dilation=1

    if (padding_total > 0) {
        int64_t C = x->ne[1];
        int64_t B = x->ne[2];

        // Replicate first element for left padding
        struct ggml_tensor * first = ggml_view_3d(ctx, x, 1, C, B,
                                                    x->nb[1], x->nb[2], 0);
        first = ggml_cont(ctx, first);
        struct ggml_tensor * left = ggml_repeat(ctx, first,
                                                  ggml_new_tensor_3d(ctx, x->type, padding_total, C, B));
        x = ggml_concat(ctx, left, x, 0);
    }

    return ggml_conv_1d(ctx, w, x, stride, 0, 1);
}

struct ggml_cgraph * CodecEncoder::build_graph(int n_samples) {
    const auto & cfg = model_.config;
    const int H = cfg.hidden_size;  // 512

    struct ggml_init_params params = {
        state_.compute_meta.size(), state_.compute_meta.data(), true
    };
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, VOCAL_CODEC_ENC_MAX_NODES, false);

    // Input: raw audio [n_samples, 1, 1]
    struct ggml_tensor * audio = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_samples, 1, 1);
    ggml_set_name(audio, "audio");
    ggml_set_input(audio);

    struct ggml_tensor * cur = audio;

    // ===== SEANet Encoder (layers 0-14) =====

    // Layer 0: Conv1d(1, 64, k=7, s=1)
    cur = causal_conv1d(ctx0, model_.conv0_w, model_.conv0_b, cur, 1, 1);

    // Layer 1: ResBlock(64) - ELU, Conv(64,32,k=3), ELU, Conv(32,64,k=1) + skip
    {
        struct ggml_tensor * residual = cur;
        cur = ggml_elu(ctx0, cur);
        cur = causal_conv1d(ctx0, model_.res[0].conv1_w, model_.res[0].conv1_b, cur, 1, 1);
        cur = ggml_elu(ctx0, cur);
        cur = causal_conv1d(ctx0, model_.res[0].conv2_w, model_.res[0].conv2_b, cur, 1, 1);
        cur = ggml_add(ctx0, cur, residual);
    }

    // Layer 2: ELU
    cur = ggml_elu(ctx0, cur);

    // Layer 3: Conv1d(64, 128, k=8, s=4) - downsample
    cur = causal_conv1d(ctx0, model_.conv3_w, model_.conv3_b, cur, 4, 1);

    // Layer 4: ResBlock(128)
    {
        struct ggml_tensor * residual = cur;
        cur = ggml_elu(ctx0, cur);
        cur = causal_conv1d(ctx0, model_.res[1].conv1_w, model_.res[1].conv1_b, cur, 1, 1);
        cur = ggml_elu(ctx0, cur);
        cur = causal_conv1d(ctx0, model_.res[1].conv2_w, model_.res[1].conv2_b, cur, 1, 1);
        cur = ggml_add(ctx0, cur, residual);
    }

    // Layer 5: ELU
    cur = ggml_elu(ctx0, cur);

    // Layer 6: Conv1d(128, 256, k=10, s=5) - downsample
    cur = causal_conv1d(ctx0, model_.conv6_w, model_.conv6_b, cur, 5, 1);

    // Layer 7: ResBlock(256)
    {
        struct ggml_tensor * residual = cur;
        cur = ggml_elu(ctx0, cur);
        cur = causal_conv1d(ctx0, model_.res[2].conv1_w, model_.res[2].conv1_b, cur, 1, 1);
        cur = ggml_elu(ctx0, cur);
        cur = causal_conv1d(ctx0, model_.res[2].conv2_w, model_.res[2].conv2_b, cur, 1, 1);
        cur = ggml_add(ctx0, cur, residual);
    }

    // Layer 8: ELU
    cur = ggml_elu(ctx0, cur);

    // Layer 9: Conv1d(256, 512, k=12, s=6) - downsample
    cur = causal_conv1d(ctx0, model_.conv9_w, model_.conv9_b, cur, 6, 1);

    // Layer 10: ResBlock(512)
    {
        struct ggml_tensor * residual = cur;
        cur = ggml_elu(ctx0, cur);
        cur = causal_conv1d(ctx0, model_.res[3].conv1_w, model_.res[3].conv1_b, cur, 1, 1);
        cur = ggml_elu(ctx0, cur);
        cur = causal_conv1d(ctx0, model_.res[3].conv2_w, model_.res[3].conv2_b, cur, 1, 1);
        cur = ggml_add(ctx0, cur, residual);
    }

    // Layer 11: ELU
    cur = ggml_elu(ctx0, cur);

    // Layer 12: Conv1d(512, 1024, k=16, s=8) - downsample
    cur = causal_conv1d(ctx0, model_.conv12_w, model_.conv12_b, cur, 8, 1);

    // Layer 13: ELU
    cur = ggml_elu(ctx0, cur);

    // Layer 14: Conv1d(1024, 512, k=3, s=1) - project to hidden_size
    cur = causal_conv1d(ctx0, model_.conv14_w, model_.conv14_b, cur, 1, 1);

    // cur shape: [T_enc, 512, 1] where T_enc ≈ n_samples / 960

    // ===== Encoder Transformer (8 layers) =====
    // Transpose: [T_enc, 512, 1] → [512, T_enc, 1] for transformer
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    // Now cur: [512, T_enc, 1]

    int T_enc = -1;  // Will be determined at runtime, but needed for RoPE/mask
    // We can't know T_enc at graph-build time without computing... but we can derive it.
    // T_enc ≈ n_samples / 960. Let's compute exactly.
    {
        // After layer 0 (k=7, s=1, pad=6): T0 = n_samples
        // After layer 3 (k=8, s=4, pad=7): T1 = floor((T0 + 7 - 8) / 4 + 1) = floor((T0-1)/4 + 1)
        // After layer 6 (k=10, s=5, pad=9): T2 = floor((T1 + 9 - 10) / 5 + 1) = floor((T1-1)/5 + 1)
        // After layer 9 (k=12, s=6, pad=11): T3 = floor((T2 + 11 - 12) / 6 + 1) = floor((T2-1)/6 + 1)
        // After layer 12 (k=16, s=8, pad=15): T4 = floor((T3 + 15 - 16) / 8 + 1) = floor((T3-1)/8 + 1)
        // After layer 14 (k=3, s=1, pad=2): T_enc = T4
        int T0 = n_samples;
        int T1 = (T0 - 1) / 4 + 1;
        int T2 = (T1 - 1) / 5 + 1;
        int T3 = (T2 - 1) / 6 + 1;
        int T4 = (T3 - 1) / 8 + 1;
        T_enc = T4;
    }

    // Position IDs for RoPE
    struct ggml_tensor * pos_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_enc);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    // Causal + sliding window attention mask
    struct ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T_enc, T_enc);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    for (int layer = 0; layer < cfg.num_layers; layer++) {
        auto & blk = model_.blk[layer];

        struct ggml_tensor * residual = cur;

        // LayerNorm (pre-attention)
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul(ctx0, cur, ggml_reshape_2d(ctx0, blk.attn_norm_w, H, 1));
        if (blk.attn_norm_b)
            cur = ggml_add(ctx0, cur, ggml_reshape_2d(ctx0, blk.attn_norm_b, H, 1));

        // Q, K, V projections: [H, T] = W[H, H] @ cur[H, T]
        struct ggml_tensor * q = ggml_mul_mat(ctx0, blk.attn_q_w, cur);
        struct ggml_tensor * k = ggml_mul_mat(ctx0, blk.attn_k_w, cur);
        struct ggml_tensor * v = ggml_mul_mat(ctx0, blk.attn_v_w, cur);

        // Reshape for multi-head: [head_dim, n_heads, T]
        q = ggml_reshape_3d(ctx0, q, cfg.head_dim, cfg.num_heads, T_enc);
        k = ggml_reshape_3d(ctx0, k, cfg.head_dim, cfg.num_heads, T_enc);
        v = ggml_reshape_3d(ctx0, v, cfg.head_dim, cfg.num_heads, T_enc);

        // RoPE (NeoX-style, matching the codec decoder)
        q = ggml_rope_ext(ctx0, q, pos_ids, nullptr,
                          cfg.head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx0, k, pos_ids, nullptr,
                          cfg.head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Flash attention with mask
        // ggml_flash_attn_ext expects: q[head_dim, T, n_heads], k[head_dim, T, n_heads], v[head_dim, T, n_heads]
        // Our shapes: [head_dim, n_heads, T] — need to permute to [head_dim, T, n_heads]
        q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));  // [head_dim, T, n_heads]
        k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));
        v = ggml_cont(ctx0, ggml_permute(ctx0, v, 0, 2, 1, 3));

        float scale = 1.0f / sqrtf((float)cfg.head_dim);
        struct ggml_tensor * attn_out = ggml_flash_attn_ext(ctx0, q, k, v, attn_mask, scale, 0.0f, 0.0f);
        // Output: [head_dim, T, n_heads]

        // Reshape back: [H, T]
        attn_out = ggml_reshape_2d(ctx0, attn_out, H, T_enc);

        // Output projection
        attn_out = ggml_mul_mat(ctx0, blk.attn_o_w, attn_out);

        // LayerScale + residual
        if (blk.attn_scale) {
            attn_out = ggml_mul(ctx0, attn_out, ggml_reshape_2d(ctx0, blk.attn_scale, H, 1));
        }
        cur = ggml_add(ctx0, residual, attn_out);

        // FFN
        residual = cur;
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul(ctx0, cur, ggml_reshape_2d(ctx0, blk.ffn_norm_w, H, 1));
        if (blk.ffn_norm_b)
            cur = ggml_add(ctx0, cur, ggml_reshape_2d(ctx0, blk.ffn_norm_b, H, 1));

        cur = ggml_mul_mat(ctx0, blk.ffn_up_w, cur);    // [2048, T]
        cur = ggml_gelu(ctx0, cur);
        cur = ggml_mul_mat(ctx0, blk.ffn_down_w, cur);  // [512, T]

        if (blk.ffn_scale) {
            cur = ggml_mul(ctx0, cur, ggml_reshape_2d(ctx0, blk.ffn_scale, H, 1));
        }
        cur = ggml_add(ctx0, residual, cur);
    }

    // cur: [512, T_enc, 1]

    // ===== Transpose back for downsample conv =====
    // After transformer, cur is [512, T_enc, 1] — transpose back to [T_enc, 512, 1]
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    // cur: [T_enc, 512, 1]

    // ===== Downsample: Conv1d(512, 512, k=4, s=2, no bias, replicate pad) =====
    cur = replicate_pad_conv1d(ctx0, model_.downsample_w, cur, 2);
    // cur: [T_out, 512, 1] where T_out ≈ T_enc / 2

    // Transpose so features for each time step are contiguous: [512, T_out]
    // This matches rvq_encode's access pattern: features[t * H + h]
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    // cur: [512, T_out, 1] — ne[0]=512, ne[1]=T_out
    cur = ggml_reshape_2d(ctx0, cur, H, cur->ne[1]);
    ggml_set_name(cur, "encoder_out");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

void CodecEncoder::rvq_encode(const float * features, int n_frames,
                               std::vector<std::vector<int32_t>> & out_codes) {
    const auto & cfg = model_.config;
    const int cb_dim = cfg.codebook_dim;    // 256
    const int cb_size = cfg.codebook_size;  // 2048
    const int H = cfg.hidden_size;          // 512
    int n_acoustic = cfg.valid_quantizers - 1;  // 15

    out_codes.resize(cfg.valid_quantizers);
    for (auto & v : out_codes) v.resize(n_frames);

    // Project features and find nearest neighbors
    auto project_and_quantize = [&](const std::vector<float> & proj_w,
                                     const std::vector<std::vector<float>> & codebooks,
                                     int n_cb, bool residual,
                                     int code_offset) {
        // Input projection: proj_w is [cb_dim, H] stored as Conv1d(H, cb_dim, k=1)
        // proj_w shape in GGUF: [1, 512, 256] → treated as [cb_dim, H] matrix
        std::vector<float> projected(n_frames * cb_dim);

        for (int t = 0; t < n_frames; t++) {
            for (int d = 0; d < cb_dim; d++) {
                float sum = 0.0f;
                for (int h = 0; h < H; h++) {
                    sum += proj_w[d * H + h] * features[t * H + h];
                }
                projected[t * cb_dim + d] = sum;
            }
        }

        std::vector<float> residual_buf;
        if (residual && n_cb > 1) {
            residual_buf = projected;
        }

        // Pre-compute codebook norms
        for (int cb = 0; cb < n_cb; cb++) {
            const float * codebook = codebooks[cb].data();

            // Precompute ||c_j||^2
            std::vector<float> cb_norms(cb_size);
            for (int j = 0; j < cb_size; j++) {
                float norm = 0.0f;
                for (int d = 0; d < cb_dim; d++) {
                    float val = codebook[j * cb_dim + d];
                    norm += val * val;
                }
                cb_norms[j] = norm;
            }

            const float * query = (cb == 0 || !residual) ? projected.data() :
                                   residual_buf.data();

            for (int t = 0; t < n_frames; t++) {
                const float * q = &query[t * cb_dim];

                // Find nearest: ||q - c||^2 = ||q||^2 + ||c||^2 - 2*q·c
                float best_dist = 1e30f;
                int32_t best_idx = 0;
                for (int j = 0; j < cb_size; j++) {
                    const float * c = &codebook[j * cb_dim];
                    float dot = 0.0f;
                    for (int d = 0; d < cb_dim; d++) dot += q[d] * c[d];
                    float dist = cb_norms[j] - 2.0f * dot;
                    if (dist < best_dist) { best_dist = dist; best_idx = j; }
                }
                out_codes[code_offset + cb][t] = best_idx;

                // Subtract quantized vector from residual
                if (residual && cb < n_cb - 1) {
                    const float * c = &codebook[best_idx * cb_dim];
                    for (int d = 0; d < cb_dim; d++) {
                        residual_buf[t * cb_dim + d] -= c[d];
                    }
                }
            }
        }
    };

    // Semantic: 1 codebook, no residual
    {
        std::vector<std::vector<float>> cbs = { model_.sem_codebook_data };
        project_and_quantize(model_.sem_input_proj_data, cbs, 1, false, 0);
    }

    // Acoustic: 15 codebooks, residual
    project_and_quantize(model_.acou_input_proj_data, model_.acou_codebook_data,
                          n_acoustic, true, 1);
}

std::vector<std::vector<int32_t>> CodecEncoder::encode(const float * audio, int n_samples) {
    if (!model_.ctx) {
        error_ = "Codec encoder not loaded";
        return {};
    }

    const auto & cfg = model_.config;

    struct ggml_cgraph * gf = build_graph(n_samples);

    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_ = "Failed to allocate graph";
        return {};
    }

    // Set audio input
    struct ggml_tensor * audio_tensor = ggml_graph_get_tensor(gf, "audio");
    ggml_backend_tensor_set(audio_tensor, audio, 0, n_samples * sizeof(float));

    // Compute T_enc for position IDs and mask
    int T0 = n_samples;
    int T1 = (T0 - 1) / 4 + 1;
    int T2 = (T1 - 1) / 5 + 1;
    int T3 = (T2 - 1) / 6 + 1;
    int T4 = (T3 - 1) / 8 + 1;
    int T_enc = T4;

    // Set position IDs
    struct ggml_tensor * pos_tensor = ggml_graph_get_tensor(gf, "pos_ids");
    if (pos_tensor) {
        std::vector<int32_t> pos(T_enc);
        for (int i = 0; i < T_enc; i++) pos[i] = i;
        ggml_backend_tensor_set(pos_tensor, pos.data(), 0, T_enc * sizeof(int32_t));
    }

    // Set attention mask (causal + sliding window) — must be F16 for flash_attn_ext
    struct ggml_tensor * mask_tensor = ggml_graph_get_tensor(gf, "attn_mask");
    if (mask_tensor) {
        std::vector<ggml_fp16_t> mask(T_enc * T_enc);
        for (int i = 0; i < T_enc; i++) {
            for (int j = 0; j < T_enc; j++) {
                bool causal_ok = j <= i;
                bool window_ok = (i - j) < cfg.sliding_window;
                float val = (causal_ok && window_ok) ? 0.0f : -INFINITY;
                mask[i * T_enc + j] = ggml_fp32_to_fp16(val);
            }
        }
        ggml_backend_tensor_set(mask_tensor, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    // Compute
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_ = "Failed to compute codec encoder graph";
        ggml_backend_sched_reset(state_.sched);
        return {};
    }

    // Extract encoder output
    struct ggml_tensor * out_tensor = ggml_graph_get_tensor(gf, "encoder_out");
    if (!out_tensor) {
        error_ = "Failed to find encoder output tensor";
        ggml_backend_sched_reset(state_.sched);
        return {};
    }

    // Output shape: ne[0]=H=512, ne[1]=T_out
    // In memory, ne[0] varies fastest: data[t * H + h] — matches rvq_encode's access pattern
    int out_dim = (int)out_tensor->ne[0];  // H = 512
    int T_out = (int)out_tensor->ne[1];

    std::vector<float> features(T_out * out_dim);
    ggml_backend_tensor_get(out_tensor, features.data(), 0, features.size() * sizeof(float));

    ggml_backend_sched_reset(state_.sched);

    // RVQ quantization on CPU
    std::vector<std::vector<int32_t>> codes;
    rvq_encode(features.data(), T_out, codes);

    return codes;
}

void free_codec_encoder_model(codec_encoder_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
}

} // namespace vocal_tts
