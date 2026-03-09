#include "tts_codec.h"
#include "onnxruntime_c_api.h"

#include <cstdio>
#include <cstring>
#include <numeric>

namespace vocal_tts {

// ONNX Runtime helpers
#define ORT_CHECK(expr) do { \
    OrtStatus * _s = (expr); \
    if (_s) { \
        const char * msg = ort_api->GetErrorMessage(_s); \
        error_ = std::string("ORT error: ") + msg; \
        ort_api->ReleaseStatus(_s); \
        return false; \
    } \
} while(0)

static const OrtApi * ort_api = nullptr;

struct AudioDecoder::Impl {
    OrtEnv * env = nullptr;
    OrtSession * session = nullptr;
    OrtSessionOptions * session_opts = nullptr;
    OrtMemoryInfo * mem_info = nullptr;

    ~Impl() {
        if (session) ort_api->ReleaseSession(session);
        if (session_opts) ort_api->ReleaseSessionOptions(session_opts);
        if (mem_info) ort_api->ReleaseMemoryInfo(mem_info);
        if (env) ort_api->ReleaseEnv(env);
    }
};

AudioDecoder::AudioDecoder() : impl_(std::make_unique<Impl>()) {
    ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
}

AudioDecoder::~AudioDecoder() = default;

bool AudioDecoder::load(const std::string & model_path) {
    fprintf(stderr, "Loading audio decoder from %s...\n", model_path.c_str());

    ORT_CHECK(ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "vocal_decoder", &impl_->env));
    ORT_CHECK(ort_api->CreateSessionOptions(&impl_->session_opts));

    // Enable CoreML on macOS for GPU acceleration
#ifdef __APPLE__
    // Try CoreML provider (optional — falls back to CPU if unavailable)
    // Note: CoreML EP may not support all ops; CPU is fine for decoder
#endif

    ORT_CHECK(ort_api->SetIntraOpNumThreads(impl_->session_opts, 4));
    ORT_CHECK(ort_api->SetSessionGraphOptimizationLevel(impl_->session_opts, ORT_ENABLE_ALL));

    ORT_CHECK(ort_api->CreateSession(impl_->env, model_path.c_str(),
                                      impl_->session_opts, &impl_->session));

    ORT_CHECK(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                            &impl_->mem_info));

    loaded_ = true;
    fprintf(stderr, "Audio decoder loaded successfully\n");
    return true;
}

std::vector<float> AudioDecoder::decode(const std::vector<std::vector<int32_t>> & codes) {
    if (!loaded_ || !impl_->session) {
        error_ = "Decoder not loaded";
        return {};
    }

    const int n_codebooks = (int)codes.size();
    if (n_codebooks == 0 || codes[0].empty()) {
        error_ = "Empty codes";
        return {};
    }

    const int seq_len = (int)codes[0].size();

    // Pad to 16 codebooks if needed (fill missing with zeros)
    const int target_codebooks = config_.num_codebooks; // 16

    // Build audio_codes tensor: [1, seq_len, 16] as int64
    std::vector<int64_t> audio_codes(seq_len * target_codebooks, 0);
    for (int t = 0; t < seq_len; t++) {
        for (int c = 0; c < target_codebooks; c++) {
            if (c < n_codebooks && t < (int)codes[c].size()) {
                audio_codes[t * target_codebooks + c] = (int64_t)codes[c][t];
            }
        }
    }

    // --- Create ORT tensors ---
    // Input names (must match ONNX model)
    const char * input_names[] = {
        "audio_codes", "is_last",
        "pre_conv_history", "latent_buffer", "conv_history",
        "past_key_0", "past_key_1", "past_key_2", "past_key_3",
        "past_key_4", "past_key_5", "past_key_6", "past_key_7",
        "past_value_0", "past_value_1", "past_value_2", "past_value_3",
        "past_value_4", "past_value_5", "past_value_6", "past_value_7",
    };
    const int n_inputs = 21;

    const char * output_names[] = { "final_wav", "valid_samples" };
    const int n_outputs = 2;

    OrtValue * input_tensors[21] = {};
    OrtValue * output_tensors[2] = {};

    auto cleanup = [&]() {
        for (int i = 0; i < n_inputs; i++) {
            if (input_tensors[i]) ort_api->ReleaseValue(input_tensors[i]);
        }
        for (int i = 0; i < n_outputs; i++) {
            if (output_tensors[i]) ort_api->ReleaseValue(output_tensors[i]);
        }
    };

    // audio_codes: [1, seq_len, 16] int64
    {
        int64_t shape[] = {1, (int64_t)seq_len, (int64_t)target_codebooks};
        OrtStatus * s = ort_api->CreateTensorWithDataAsOrtValue(
            impl_->mem_info, audio_codes.data(), audio_codes.size() * sizeof(int64_t),
            shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensors[0]);
        if (s) {
            error_ = std::string("Failed to create audio_codes tensor: ") + ort_api->GetErrorMessage(s);
            ort_api->ReleaseStatus(s);
            cleanup();
            return {};
        }
    }

    // is_last: [1] float32 = 1.0 (process entire sequence)
    float is_last_val = 1.0f;
    {
        int64_t shape[] = {1};
        OrtStatus * s = ort_api->CreateTensorWithDataAsOrtValue(
            impl_->mem_info, &is_last_val, sizeof(float),
            shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensors[1]);
        if (s) { error_ = ort_api->GetErrorMessage(s); ort_api->ReleaseStatus(s); cleanup(); return {}; }
    }

    // Empty history tensors (non-streaming: all zero-length on time dimension)
    // pre_conv_history: [1, 512, 0]
    {
        int64_t shape[] = {1, 512, 0};
        float dummy = 0;
        OrtStatus * s = ort_api->CreateTensorWithDataAsOrtValue(
            impl_->mem_info, &dummy, 0,
            shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensors[2]);
        if (s) { error_ = ort_api->GetErrorMessage(s); ort_api->ReleaseStatus(s); cleanup(); return {}; }
    }

    // latent_buffer: [1, 1024, 0]
    {
        int64_t shape[] = {1, 1024, 0};
        float dummy = 0;
        OrtStatus * s = ort_api->CreateTensorWithDataAsOrtValue(
            impl_->mem_info, &dummy, 0,
            shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensors[3]);
        if (s) { error_ = ort_api->GetErrorMessage(s); ort_api->ReleaseStatus(s); cleanup(); return {}; }
    }

    // conv_history: [1, 1024, 0]
    {
        int64_t shape[] = {1, 1024, 0};
        float dummy = 0;
        OrtStatus * s = ort_api->CreateTensorWithDataAsOrtValue(
            impl_->mem_info, &dummy, 0,
            shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensors[4]);
        if (s) { error_ = ort_api->GetErrorMessage(s); ort_api->ReleaseStatus(s); cleanup(); return {}; }
    }

    // past_key_0..7 and past_value_0..7: [1, 16, 0, 64]
    for (int i = 0; i < 16; i++) {
        int64_t shape[] = {1, 16, 0, 64};
        float dummy = 0;
        OrtStatus * s = ort_api->CreateTensorWithDataAsOrtValue(
            impl_->mem_info, &dummy, 0,
            shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensors[5 + i]);
        if (s) { error_ = ort_api->GetErrorMessage(s); ort_api->ReleaseStatus(s); cleanup(); return {}; }
    }

    // --- Run inference ---
    {
        OrtStatus * s = ort_api->Run(impl_->session, nullptr,
                                      input_names, (const OrtValue * const *)input_tensors, n_inputs,
                                      output_names, n_outputs, output_tensors);
        if (s) {
            error_ = std::string("Decoder inference failed: ") + ort_api->GetErrorMessage(s);
            ort_api->ReleaseStatus(s);
            cleanup();
            return {};
        }
    }

    // --- Extract results ---
    // valid_samples: int64 scalar
    int64_t valid_samples = 0;
    {
        int64_t * data = nullptr;
        (void)ort_api->GetTensorMutableData(output_tensors[1], (void **)&data);
        if (data) valid_samples = data[0];
    }

    // final_wav: [1, wav_len] float32
    std::vector<float> wav;
    {
        float * data = nullptr;
        (void)ort_api->GetTensorMutableData(output_tensors[0], (void **)&data);
        if (data && valid_samples > 0) {
            wav.assign(data, data + valid_samples);
        }
    }

    cleanup();

    fprintf(stderr, "Decoded %d codes → %d samples (%.1f sec at %d Hz)\n",
            seq_len, (int)wav.size(),
            (float)wav.size() / config_.sample_rate, config_.sample_rate);

    return wav;
}

} // namespace vocal_tts
