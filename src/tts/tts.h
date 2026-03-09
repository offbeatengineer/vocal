#pragma once

#include "tts_model.h"
#include "tts_tokenizer.h"
#include "tts_codec.h"
#include "tts_types.h"

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace vocal_tts {

struct tts_params {
    int32_t max_tokens = 2048;
    int32_t n_threads = 4;
    float speed = 1.0f;
    bool print_timing = true;
    std::string speaker = "Vivian";  // Default speaker
    std::string language = "Auto";   // Auto-detect
};

struct tts_result {
    std::vector<float> audio;    // Output waveform samples
    int32_t sample_rate = 24000; // Output sample rate
    bool success = false;
    std::string error_msg;

    int64_t t_tokenize_ms = 0;
    int64_t t_generate_ms = 0;
    int64_t t_decode_ms = 0;
    int64_t t_total_ms = 0;
    int32_t n_tokens_generated = 0;
};

class TTS {
public:
    TTS();
    ~TTS();

    // Load model from GGUF file + tokenizer from tokenizer.json + decoder ONNX
    bool load(const std::string & model_path,
              const std::string & tokenizer_path,
              const std::string & decoder_path);

    // Synthesize speech from text
    tts_result synthesize(const std::string & text, const tts_params & params = tts_params());

    const std::string & get_error() const { return error_; }
    bool is_loaded() const { return loaded_; }

private:
    qwen3::Qwen3TalkerLLM talker_;
    TTSTokenizer tokenizer_;
    std::unique_ptr<AudioDecoder> decoder_;

    bool loaded_ = false;
    std::string error_;

    // Codec special token IDs
    static constexpr int32_t CODEC_PAD  = 2148;
    static constexpr int32_t CODEC_BOS  = 2149;
    static constexpr int32_t CODEC_EOS  = 2150;
    static constexpr int32_t CODEC_THINK = 2154;
    static constexpr int32_t CODEC_NOTHINK = 2155;
    static constexpr int32_t CODEC_THINK_BOS = 2156;
    static constexpr int32_t CODEC_THINK_EOS = 2157;

    // Language codec IDs
    static constexpr int32_t LANG_ENGLISH = 2050;
    static constexpr int32_t LANG_CHINESE = 2055;

    // Speaker codec IDs (from CustomVoice config)
    int32_t get_speaker_id(const std::string & name) const;
    int32_t get_language_id(const std::string & name) const;

    // Build the prompt embedding: text+codec dual embedding at each position
    void build_prompt_embeds(const std::vector<int32_t> & text_tokens,
                             int32_t speaker_id, int32_t language_id,
                             std::vector<float> & out_embeds,
                             std::vector<float> & tts_pad_embed);

    // Autoregressive generation with correct dual-embedding approach
    void generate_codes_v2(const std::vector<float> & prompt_embeds,
                           int32_t n_prompt_tokens,
                           const std::vector<float> & tts_pad_embed,
                           const tts_params & params,
                           std::vector<std::vector<int32_t>> & out_multi_codes);
};

} // namespace vocal_tts
