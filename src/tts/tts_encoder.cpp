#include "tts_encoder.h"
#include "onnxruntime_c_api.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace vocal_tts {

#define ORT_CHECK_ENC(expr) do { \
    OrtStatus * _s = (expr); \
    if (_s) { \
        const char * msg = ort_api->GetErrorMessage(_s); \
        error_ = std::string("ORT error: ") + msg; \
        ort_api->ReleaseStatus(_s); \
        return false; \
    } \
} while(0)

static const OrtApi * ort_api = nullptr;

static void init_ort_api() {
    if (!ort_api) {
        ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    }
}

// ============================================================
// SpeakerEncoder
// ============================================================

struct SpeakerEncoder::Impl {
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

SpeakerEncoder::SpeakerEncoder() : impl_(std::make_unique<Impl>()) {
    init_ort_api();
}

SpeakerEncoder::~SpeakerEncoder() = default;

bool SpeakerEncoder::load(const std::string & model_path) {
    fprintf(stderr, "Loading speaker encoder from %s...\n", model_path.c_str());

    ORT_CHECK_ENC(ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "vocal_spk_enc", &impl_->env));
    ORT_CHECK_ENC(ort_api->CreateSessionOptions(&impl_->session_opts));
    ORT_CHECK_ENC(ort_api->SetIntraOpNumThreads(impl_->session_opts, 4));
    ORT_CHECK_ENC(ort_api->SetSessionGraphOptimizationLevel(impl_->session_opts, ORT_ENABLE_ALL));
    ORT_CHECK_ENC(ort_api->CreateSession(impl_->env, model_path.c_str(),
                                          impl_->session_opts, &impl_->session));
    ORT_CHECK_ENC(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                                &impl_->mem_info));

    loaded_ = true;
    fprintf(stderr, "Speaker encoder loaded successfully\n");
    return true;
}

// Compute mel spectrogram for speaker encoder
// Parameters: n_fft=1024, num_mels=128, hop_size=256, win_size=1024, sample_rate=24000
static std::vector<float> compute_mel_spectrogram(const float * audio, int n_samples) {
    const int n_fft = 1024;
    const int n_mels = 128;
    const int hop_size = 256;
    const int win_size = 1024;
    const float sample_rate = 24000.0f;
    const float fmin = 0.0f;
    const float fmax = 12000.0f;

    int n_frames = (n_samples - win_size) / hop_size + 1;
    if (n_frames <= 0) n_frames = 1;

    // Build Hann window
    std::vector<float> window(win_size);
    for (int i = 0; i < win_size; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / win_size));
    }

    // Build mel filterbank [n_mels x (n_fft/2+1)]
    int n_freqs = n_fft / 2 + 1;
    std::vector<float> mel_basis(n_mels * n_freqs, 0.0f);
    {
        auto hz_to_mel = [](float hz) -> float { return 2595.0f * log10f(1.0f + hz / 700.0f); };
        auto mel_to_hz = [](float mel) -> float { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); };

        float mel_low = hz_to_mel(fmin);
        float mel_high = hz_to_mel(fmax);
        std::vector<float> mel_points(n_mels + 2);
        for (int i = 0; i < n_mels + 2; i++) {
            mel_points[i] = mel_to_hz(mel_low + (mel_high - mel_low) * i / (n_mels + 1));
        }

        std::vector<float> fft_freqs(n_freqs);
        for (int i = 0; i < n_freqs; i++) {
            fft_freqs[i] = sample_rate * i / n_fft;
        }

        for (int m = 0; m < n_mels; m++) {
            float left = mel_points[m];
            float center = mel_points[m + 1];
            float right = mel_points[m + 2];
            for (int k = 0; k < n_freqs; k++) {
                float f = fft_freqs[k];
                if (f >= left && f <= center) {
                    mel_basis[m * n_freqs + k] = (f - left) / (center - left);
                } else if (f > center && f <= right) {
                    mel_basis[m * n_freqs + k] = (right - f) / (right - center);
                }
            }
        }
    }

    // Compute STFT → power spectrum → mel → log
    std::vector<float> mel_out(n_frames * n_mels, 0.0f);

    // Simple DFT (no FFT library needed for the sizes we use)
    for (int frame = 0; frame < n_frames; frame++) {
        int start = frame * hop_size;

        // Compute power spectrum via DFT
        std::vector<float> power(n_freqs, 0.0f);
        for (int k = 0; k < n_freqs; k++) {
            float re = 0.0f, im = 0.0f;
            for (int n = 0; n < win_size; n++) {
                int idx = start + n;
                float sample = (idx < n_samples) ? audio[idx] : 0.0f;
                float windowed = sample * window[n];
                float angle = -2.0f * (float)M_PI * k * n / n_fft;
                re += windowed * cosf(angle);
                im += windowed * sinf(angle);
            }
            power[k] = re * re + im * im;
        }

        // Apply mel filterbank
        for (int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            for (int k = 0; k < n_freqs; k++) {
                sum += mel_basis[m * n_freqs + k] * power[k];
            }
            // Log mel spectrogram (add small epsilon for numerical stability)
            mel_out[frame * n_mels + m] = log10f(sum + 1e-10f);
        }
    }

    fprintf(stderr, "Mel spectrogram: %d frames × %d mels\n", n_frames, n_mels);
    return mel_out;
}

std::vector<float> SpeakerEncoder::encode(const float * audio, int n_samples) {
    if (!loaded_ || !impl_->session) {
        error_ = "Speaker encoder not loaded";
        return {};
    }

    // Compute mel spectrogram: [n_frames, 128]
    std::vector<float> mel = compute_mel_spectrogram(audio, n_samples);
    if (mel.empty()) {
        error_ = "Failed to compute mel spectrogram";
        return {};
    }

    int n_mels = 128;
    int n_frames = (int)mel.size() / n_mels;

    // Input: "mels" tensor [1, n_frames, 128] float32
    int64_t mel_shape[] = {1, (int64_t)n_frames, (int64_t)n_mels};

    OrtValue * input_tensor = nullptr;
    OrtValue * output_tensor = nullptr;

    auto cleanup = [&]() {
        if (input_tensor) ort_api->ReleaseValue(input_tensor);
        if (output_tensor) ort_api->ReleaseValue(output_tensor);
    };

    OrtStatus * s = ort_api->CreateTensorWithDataAsOrtValue(
        impl_->mem_info, mel.data(), mel.size() * sizeof(float),
        mel_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    if (s) {
        error_ = std::string("Failed to create mel tensor: ") + ort_api->GetErrorMessage(s);
        ort_api->ReleaseStatus(s);
        cleanup();
        return {};
    }

    const char * input_names[] = {"mels"};
    const char * output_names[] = {"spk_emb"};

    s = ort_api->Run(impl_->session, nullptr,
                      input_names, (const OrtValue * const *)&input_tensor, 1,
                      output_names, 1, &output_tensor);
    if (s) {
        error_ = std::string("Speaker encoder inference failed: ") + ort_api->GetErrorMessage(s);
        ort_api->ReleaseStatus(s);
        cleanup();
        return {};
    }

    // Get output shape and data
    OrtTensorTypeAndShapeInfo * info = nullptr;
    (void)ort_api->GetTensorTypeAndShape(output_tensor, &info);
    size_t n_dims = 0;
    (void)ort_api->GetDimensionsCount(info, &n_dims);
    std::vector<int64_t> dims(n_dims);
    (void)ort_api->GetDimensions(info, dims.data(), n_dims);
    ort_api->ReleaseTensorTypeAndShapeInfo(info);

    int64_t total_elements = 1;
    for (size_t i = 0; i < n_dims; i++) total_elements *= dims[i];

    float * data = nullptr;
    (void)ort_api->GetTensorMutableData(output_tensor, (void **)&data);

    std::vector<float> embedding;
    if (data && total_elements > 0) {
        embedding.assign(data, data + total_elements);
    }

    cleanup();

    fprintf(stderr, "Speaker embedding: %zu dimensions\n", embedding.size());
    return embedding;
}

// ============================================================
// CodecEncoder
// ============================================================

struct CodecEncoder::Impl {
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

CodecEncoder::CodecEncoder() : impl_(std::make_unique<Impl>()) {
    init_ort_api();
}

CodecEncoder::~CodecEncoder() = default;

bool CodecEncoder::load(const std::string & model_path) {
    fprintf(stderr, "Loading codec encoder from %s...\n", model_path.c_str());

    ORT_CHECK_ENC(ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "vocal_codec_enc", &impl_->env));
    ORT_CHECK_ENC(ort_api->CreateSessionOptions(&impl_->session_opts));
    ORT_CHECK_ENC(ort_api->SetIntraOpNumThreads(impl_->session_opts, 4));
    ORT_CHECK_ENC(ort_api->SetSessionGraphOptimizationLevel(impl_->session_opts, ORT_ENABLE_ALL));
    ORT_CHECK_ENC(ort_api->CreateSession(impl_->env, model_path.c_str(),
                                          impl_->session_opts, &impl_->session));
    ORT_CHECK_ENC(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                                &impl_->mem_info));

    loaded_ = true;
    fprintf(stderr, "Codec encoder loaded successfully\n");
    return true;
}

std::vector<std::vector<int32_t>> CodecEncoder::encode(const float * audio, int n_samples) {
    if (!loaded_ || !impl_->session) {
        error_ = "Codec encoder not loaded";
        return {};
    }

    // Input: "input_values" tensor [1, n_samples] float32
    std::vector<float> audio_buf(audio, audio + n_samples);
    int64_t audio_shape[] = {1, (int64_t)n_samples};

    OrtValue * input_tensor = nullptr;
    OrtValue * output_tensor = nullptr;

    auto cleanup = [&]() {
        if (input_tensor) ort_api->ReleaseValue(input_tensor);
        if (output_tensor) ort_api->ReleaseValue(output_tensor);
    };

    // Create audio tensor
    OrtStatus * s = ort_api->CreateTensorWithDataAsOrtValue(
        impl_->mem_info, audio_buf.data(), audio_buf.size() * sizeof(float),
        audio_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    if (s) {
        error_ = std::string("Failed to create audio tensor: ") + ort_api->GetErrorMessage(s);
        ort_api->ReleaseStatus(s);
        cleanup();
        return {};
    }

    const char * input_names[] = {"input_values"};
    const char * output_names[] = {"audio_codes"};

    s = ort_api->Run(impl_->session, nullptr,
                      input_names, (const OrtValue * const *)&input_tensor, 1,
                      output_names, 1, &output_tensor);
    if (s) {
        error_ = std::string("Codec encoder inference failed: ") + ort_api->GetErrorMessage(s);
        ort_api->ReleaseStatus(s);
        cleanup();
        return {};
    }

    // Output: "audio_codes" [1, num_codebooks, T] int64
    OrtTensorTypeAndShapeInfo * info = nullptr;
    (void)ort_api->GetTensorTypeAndShape(output_tensor, &info);
    size_t n_dims = 0;
    (void)ort_api->GetDimensionsCount(info, &n_dims);
    std::vector<int64_t> dims(n_dims);
    (void)ort_api->GetDimensions(info, dims.data(), n_dims);
    ort_api->ReleaseTensorTypeAndShapeInfo(info);

    if (n_dims < 2) {
        error_ = "Unexpected output shape from codec encoder";
        cleanup();
        return {};
    }

    // Output shape: [batch, T, num_codebooks] (e.g., [1, 39, 16])
    int T, num_codebooks;
    if (n_dims == 3) {
        T = (int)dims[1];
        num_codebooks = (int)dims[2];
    } else {
        T = (int)dims[0];
        num_codebooks = (int)dims[1];
    }

    // Only use first 16 codebooks
    if (num_codebooks > 16) num_codebooks = 16;

    int64_t * data = nullptr;
    (void)ort_api->GetTensorMutableData(output_tensor, (void **)&data);

    // Data layout: [batch, T, num_codebooks] — row-major
    // For each timestep t, codebooks are contiguous: data[t * num_codebooks + c]
    std::vector<std::vector<int32_t>> codes(num_codebooks, std::vector<int32_t>(T));
    if (data) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < num_codebooks; c++) {
                codes[c][t] = (int32_t)data[t * num_codebooks + c];
            }
        }
    }

    cleanup();

    fprintf(stderr, "Codec encoder: %d codebooks × %d frames (%.1f sec at 12.5 Hz)\n",
            num_codebooks, T, (float)T / 12.5f);
    return codes;
}

} // namespace vocal_tts
