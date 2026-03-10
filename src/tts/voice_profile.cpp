#include "voice_profile.h"

#include <cstdio>
#include <cstring>

namespace vocal_tts {

// Binary format:
//   magic:            4 bytes "VOCL"
//   version:          uint32 (1)
//   ref_text_len:     uint32
//   n_codebooks:      uint32
//   n_frames:         uint32
//   speaker_dim:      uint32
//   ref_text:         ref_text_len bytes (UTF-8, no null terminator)
//   codec_codes:      n_codebooks × n_frames × int32 (row-major: codebook-first)
//   speaker_embed:    speaker_dim × float32

static const char MAGIC[4] = {'V', 'O', 'C', 'L'};
static const uint32_t VERSION = 1;

bool VoiceProfile::save(const std::string & path, std::string & error) const {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        error = "Failed to open file for writing: " + path;
        return false;
    }

    auto write_u32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };

    fwrite(MAGIC, 1, 4, f);
    write_u32(VERSION);
    write_u32((uint32_t)ref_text.size());
    write_u32((uint32_t)codec_codes.size());
    write_u32((uint32_t)(codec_codes.empty() ? 0 : codec_codes[0].size()));
    write_u32((uint32_t)speaker_embed.size());

    // Reference text
    if (!ref_text.empty()) {
        fwrite(ref_text.data(), 1, ref_text.size(), f);
    }

    // Codec codes: codebook-major (all frames for cb0, then cb1, etc.)
    for (const auto & cb : codec_codes) {
        fwrite(cb.data(), sizeof(int32_t), cb.size(), f);
    }

    // Speaker embedding
    if (!speaker_embed.empty()) {
        fwrite(speaker_embed.data(), sizeof(float), speaker_embed.size(), f);
    }

    fclose(f);
    return true;
}

bool VoiceProfile::load(const std::string & path, std::string & error) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        error = "Failed to open voice profile: " + path;
        return false;
    }

    auto read_u32 = [&](uint32_t * v) -> bool {
        return fread(v, 4, 1, f) == 1;
    };

    // Magic
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, MAGIC, 4) != 0) {
        error = "Invalid voice profile (bad magic): " + path;
        fclose(f);
        return false;
    }

    // Version
    uint32_t version;
    if (!read_u32(&version) || version != VERSION) {
        error = "Unsupported voice profile version: " + std::to_string(version);
        fclose(f);
        return false;
    }

    uint32_t text_len, n_cb, n_fr, spk_dim;
    if (!read_u32(&text_len) || !read_u32(&n_cb) || !read_u32(&n_fr) || !read_u32(&spk_dim)) {
        error = "Truncated voice profile header";
        fclose(f);
        return false;
    }

    // Sanity checks
    if (n_cb > 32 || n_fr > 100000 || text_len > 1000000 || spk_dim > 10000) {
        error = "Voice profile has invalid dimensions";
        fclose(f);
        return false;
    }

    // Reference text
    ref_text.resize(text_len);
    if (text_len > 0 && fread(&ref_text[0], 1, text_len, f) != text_len) {
        error = "Truncated voice profile (ref_text)";
        fclose(f);
        return false;
    }

    // Codec codes
    codec_codes.resize(n_cb);
    for (uint32_t cb = 0; cb < n_cb; cb++) {
        codec_codes[cb].resize(n_fr);
        if (n_fr > 0 && fread(codec_codes[cb].data(), sizeof(int32_t), n_fr, f) != n_fr) {
            error = "Truncated voice profile (codec_codes)";
            fclose(f);
            return false;
        }
    }

    // Speaker embedding
    speaker_embed.resize(spk_dim);
    if (spk_dim > 0 && fread(speaker_embed.data(), sizeof(float), spk_dim, f) != spk_dim) {
        error = "Truncated voice profile (speaker_embed)";
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

} // namespace vocal_tts
