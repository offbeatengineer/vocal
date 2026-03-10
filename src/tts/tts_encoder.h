#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

namespace vocal_tts {

// Extracts speaker embedding from reference audio via ECAPA-TDNN
class SpeakerEncoder {
public:
    SpeakerEncoder();
    ~SpeakerEncoder();

    bool load(const std::string & model_path);

    // Encode reference audio to speaker embedding
    // audio: float32 mono samples at 24kHz
    // Returns: speaker embedding vector (1024-dim for 0.6B model)
    std::vector<float> encode(const float * audio, int n_samples);

    bool is_loaded() const { return loaded_; }
    const std::string & get_error() const { return error_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string error_;
    bool loaded_ = false;
};

// Encodes audio waveform to codec codes (16 codebooks) via Mimi encoder
class CodecEncoder {
public:
    CodecEncoder();
    ~CodecEncoder();

    bool load(const std::string & model_path);

    // Encode audio to multi-codebook codes
    // audio: float32 mono samples at 24kHz
    // Returns: [16][T] codec codes (16 codebooks, T = ceil(n_samples / 1920))
    std::vector<std::vector<int32_t>> encode(const float * audio, int n_samples);

    bool is_loaded() const { return loaded_; }
    const std::string & get_error() const { return error_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string error_;
    bool loaded_ = false;
};

} // namespace vocal_tts
