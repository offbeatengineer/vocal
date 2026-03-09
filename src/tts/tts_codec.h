#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

namespace vocal_tts {

struct DecoderConfig {
    int num_codebooks = 16;
    int codebook_size = 2048;
    int sample_rate = 24000;
    float frame_rate = 12.5f;  // 24000 / 1920
};

class AudioDecoder {
public:
    AudioDecoder();
    ~AudioDecoder();

    // Load ONNX decoder model
    bool load(const std::string & model_path);

    // Decode audio codes to waveform
    // codes: [num_codebooks][seq_len] — multi-codebook audio codes
    // Returns PCM float samples at 24kHz
    std::vector<float> decode(const std::vector<std::vector<int32_t>> & codes);

    bool is_loaded() const { return loaded_; }
    const std::string & get_error() const { return error_; }
    const DecoderConfig & get_config() const { return config_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    DecoderConfig config_;
    bool loaded_ = false;
    std::string error_;
};

} // namespace vocal_tts
