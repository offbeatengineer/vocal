#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace vocal_tts {

// Special token IDs for Qwen3-TTS
struct tts_special_tokens {
    // Text-level special tokens
    int32_t im_start = 151644;   // <|im_start|>
    int32_t im_end   = 151645;   // <|im_end|>
    int32_t tts_bos  = 151672;   // Begin of TTS
    int32_t tts_eos  = 151673;   // End of TTS
    int32_t tts_pad  = 151671;   // TTS padding
    int32_t pad      = 151643;   // General padding
    int32_t newline  = 198;      // \n
    int32_t assistant = 77091;   // "assistant"

    // Codec-level special tokens (in codec vocabulary space, size 3072)
    int32_t codec_pad = 2148;    // Codec padding
    int32_t codec_bos = 2149;    // Codec begin of sequence
    int32_t codec_eos = 2150;    // Codec end of sequence
};

class TTSTokenizer {
public:
    TTSTokenizer();
    ~TTSTokenizer();

    // Load tokenizer from tokenizer.json file
    bool load(const std::string & path);

    // Encode text to token IDs
    std::vector<int32_t> encode(const std::string & text) const;

    // Build the full TTS prompt: <|im_start|>assistant\n{text}<|im_end|>\n
    std::vector<int32_t> build_tts_prompt(const std::string & text) const;

    // Get special token IDs
    const tts_special_tokens & special() const { return special_; }

    bool is_loaded() const { return loaded_; }
    const std::string & get_error() const { return error_; }

private:
    // BPE vocabulary: token bytes → token ID
    std::map<std::vector<uint8_t>, int32_t> vocab_;

    // Reverse vocab for decoding: ID → token bytes
    std::map<int32_t, std::vector<uint8_t>> id_to_token_;

    // Merge rules in priority order: pair → merge rank
    std::map<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>, int> merge_ranks_;

    // All merges in order
    std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> merges_;

    tts_special_tokens special_;
    bool loaded_ = false;
    std::string error_;

    // Pre-tokenize: split text into chunks
    std::vector<std::string> pre_tokenize(const std::string & text) const;

    // BPE encode a single chunk
    std::vector<int32_t> bpe_encode(const std::string & chunk) const;

    // Parse tokenizer.json
    bool parse_json(const std::string & json_str);
};

} // namespace vocal_tts
