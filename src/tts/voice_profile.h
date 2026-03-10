#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace vocal_tts {

// Voice profile: pre-encoded reference data for voice cloning.
// Stores codec codes + reference text so we skip audio encoding on reuse.
struct VoiceProfile {
    std::string ref_text;                              // Reference transcript (UTF-8)
    std::vector<std::vector<int32_t>> codec_codes;     // [n_codebooks][n_frames]
    std::vector<float> speaker_embed;                  // Speaker embedding (may be empty)

    int n_codebooks() const { return (int)codec_codes.size(); }
    int n_frames() const { return codec_codes.empty() ? 0 : (int)codec_codes[0].size(); }
    int speaker_dim() const { return (int)speaker_embed.size(); }

    // Save to binary file. Returns true on success.
    bool save(const std::string & path, std::string & error) const;

    // Load from binary file. Returns true on success.
    bool load(const std::string & path, std::string & error);
};

} // namespace vocal_tts
