#include "qwen3_asr.h"
#include "audio_io.h"
#include "timing.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <fstream>

namespace qwen3_asr {

static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

Qwen3ASR::Qwen3ASR() = default;
Qwen3ASR::~Qwen3ASR() = default;

bool Qwen3ASR::load_model(const std::string & model_path) {
    int64_t t_start = get_time_ms();
    
    if (!encoder_.load_model(model_path)) {
        error_msg_ = "Failed to load audio encoder: " + encoder_.get_error();
        return false;
    }
    
    if (!decoder_.load_model(model_path)) {
        error_msg_ = "Failed to load text decoder: " + decoder_.get_error();
        return false;
    }
    
    generate_mel_filters(mel_filters_, QWEN_N_MELS, QWEN_N_FFT, QWEN_SAMPLE_RATE);
    
    model_loaded_ = true;
    
    int64_t t_end = get_time_ms();
    fprintf(stderr, "Model loaded in %lld ms\n", (long long)(t_end - t_start));
    
    return true;
}

transcribe_result Qwen3ASR::transcribe(const std::string & audio_path,
                                        const transcribe_params & params) {
    transcribe_result result;
    
    if (!model_loaded_) {
        result.error_msg = "Model not loaded";
        return result;
    }
    
    float * raw_samples = nullptr;
    int n_samples = 0;
    int sample_rate = 0;

    if (!vocal_audio_read(audio_path.c_str(), &raw_samples, &n_samples,
                          &sample_rate, QWEN_SAMPLE_RATE)) {
        result.error_msg = "Failed to load audio file: " + audio_path;
        return result;
    }

    transcribe_result res = transcribe_internal(raw_samples, n_samples, params);
    free(raw_samples);
    return res;
}

transcribe_result Qwen3ASR::transcribe(const float * samples, int n_samples,
                                        const transcribe_params & params) {
    transcribe_result result;
    
    if (!model_loaded_) {
        result.error_msg = "Model not loaded";
        return result;
    }
    
    return transcribe_internal(samples, n_samples, params);
}

transcribe_result Qwen3ASR::transcribe_internal(const float * samples, int n_samples,
                                                 const transcribe_params & params) {
    transcribe_result result;
    int64_t t_total_start = get_time_ms();
    
    int64_t t_mel_start = get_time_ms();
    MelSpectrogram mel;
    {
        QWEN3_TIMER("mel_spectrogram");
        if (!log_mel_spectrogram(samples, n_samples, mel_filters_, mel, params.n_threads)) {
            result.error_msg = "Failed to compute mel spectrogram";
            return result;
        }
    }
    result.t_mel_ms = get_time_ms() - t_mel_start;
    
    if (params.print_progress) {
        fprintf(stderr, "Mel spectrogram: [%d, %d]\n", mel.n_mel, mel.n_len);
    }
    
    int64_t t_encode_start = get_time_ms();
    std::vector<float> audio_features;
    {
        QWEN3_TIMER("audio_encoding");
        if (!encoder_.encode(mel.data.data(), mel.n_mel, mel.n_len, audio_features)) {
            result.error_msg = "Failed to encode audio: " + encoder_.get_error();
            return result;
        }
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;
    
    const auto & text_hparams = encoder_.get_text_hparams();
    int32_t n_audio_frames = audio_features.size() / text_hparams.hidden_size;
    
    if (params.print_progress) {
        fprintf(stderr, "Audio features: [%d, %d]\n", n_audio_frames, text_hparams.hidden_size);
    }
    
    std::vector<int32_t> input_tokens = build_input_tokens(n_audio_frames, params.language);
    
    if (params.print_progress) {
        fprintf(stderr, "Input tokens: %zu\n", input_tokens.size());
    }
    
    int64_t t_decode_start = get_time_ms();
    std::vector<int32_t> output_tokens;
    if (!decode_greedy(input_tokens, audio_features, n_audio_frames, params, output_tokens)) {
        result.error_msg = "Decoding failed: " + error_msg_;
        return result;
    }
    result.t_decode_ms = get_time_ms() - t_decode_start;
    
    result.tokens = output_tokens;
    result.text = decoder_.decode_tokens(output_tokens);
    result.success = true;
    
    result.t_total_ms = get_time_ms() - t_total_start;
    
    if (params.print_timing) {
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Mel spectrogram: %lld ms\n", (long long)result.t_mel_ms);
        fprintf(stderr, "  Audio encoding:  %lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Text decoding:   %lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:           %lld ms\n", (long long)result.t_total_ms);
        fprintf(stderr, "  Tokens generated: %zu\n", output_tokens.size());
    }
    
    return result;
}

std::vector<int32_t> Qwen3ASR::build_input_tokens(int32_t n_audio_frames,
                                                   const std::string & language) {
    const auto & cfg = decoder_.get_config();
    
    std::vector<int32_t> tokens;
    tokens.reserve(n_audio_frames + 20);
    
    // Chat template format:
    // <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>...<|audio_end|><|im_end|>\n<|im_start|>assistant\n
    
    // Token IDs from Qwen3 tokenizer:
    // <|im_start|> = 151644
    // <|im_end|> = 151645
    // system = 8948
    // user = 872
    // assistant = 77091
    // \n = 198
    
    const int32_t im_start = 151644;
    const int32_t im_end = 151645;
    const int32_t system_token = 8948;
    const int32_t user_token = 872;
    const int32_t assistant_token = 77091;
    const int32_t newline = 198;
    
    // <|im_start|>system\n<|im_end|>\n
    tokens.push_back(im_start);
    tokens.push_back(system_token);
    tokens.push_back(newline);
    tokens.push_back(im_end);
    tokens.push_back(newline);
    
    // <|im_start|>user\n
    tokens.push_back(im_start);
    tokens.push_back(user_token);
    tokens.push_back(newline);
    
    // <|audio_start|><|audio_pad|>...<|audio_end|>
    tokens.push_back(cfg.audio_start_token_id);
    for (int32_t i = 0; i < n_audio_frames; ++i) {
        tokens.push_back(cfg.audio_pad_token_id);
    }
    tokens.push_back(cfg.audio_end_token_id);
    
    // <|im_end|>\n<|im_start|>assistant\n
    tokens.push_back(im_end);
    tokens.push_back(newline);
    tokens.push_back(im_start);
    tokens.push_back(assistant_token);
    tokens.push_back(newline);
    
    (void)language;
    
    return tokens;
}

bool Qwen3ASR::decode_greedy(const std::vector<int32_t> & input_tokens,
                              const std::vector<float> & audio_features,
                              int32_t n_audio_frames,
                              const transcribe_params & params,
                              std::vector<int32_t> & output_tokens) {
    const auto & cfg = decoder_.get_config();
    
    int32_t n_ctx_needed = input_tokens.size() + params.max_tokens;
    if (!decoder_.init_kv_cache(n_ctx_needed)) {
        error_msg_ = "Failed to initialize KV cache: " + decoder_.get_error();
        return false;
    }
    
    std::vector<float> logits;
    
    // Audio pad tokens start after: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
    // That's 8 tokens before the first audio_pad
    int32_t audio_start_pos = 9;
    
    {
        QWEN3_TIMER("decode.initial_forward");
        if (!decoder_.forward_with_audio(
                input_tokens.data(), input_tokens.size(),
                audio_features.data(), n_audio_frames,
                audio_start_pos, 0, logits)) {
            error_msg_ = "Initial forward pass failed: " + decoder_.get_error();
            return false;
        }
    }
    
    int32_t vocab_size = cfg.vocab_size;
    int32_t n_input = input_tokens.size();
    
    int32_t next_token = sample_greedy(logits.data(), vocab_size);
    
    output_tokens.clear();
    output_tokens.push_back(next_token);
    
    if (progress_callback_) {
        progress_callback_(1, params.max_tokens);
    }
    
    int32_t n_past = n_input;
    
    while (next_token != cfg.eos_token_id && 
           (int32_t)output_tokens.size() < params.max_tokens) {
        
        std::vector<int32_t> single_token = {next_token};
        
        {
            QWEN3_TIMER("decode.token");
            if (!decoder_.forward(single_token.data(), 1, n_past, logits)) {
                error_msg_ = "Forward pass failed at token " + 
                             std::to_string(output_tokens.size()) + ": " + decoder_.get_error();
                return false;
            }
        }
        
        next_token = sample_greedy(logits.data(), vocab_size);
        output_tokens.push_back(next_token);
        
        n_past += 1;
        
        if (progress_callback_) {
            progress_callback_(output_tokens.size(), params.max_tokens);
        }
        
        if (params.print_progress && output_tokens.size() % 10 == 0) {
            fprintf(stderr, "Generated %zu tokens...\n", output_tokens.size());
        }
    }
    
    if (output_tokens.back() == cfg.eos_token_id) {
        output_tokens.pop_back();
    }
    
    return true;
}

int32_t Qwen3ASR::sample_greedy(const float * logits, int32_t vocab_size) {
    int32_t max_idx = 0;
    float max_val = logits[0];
    
    for (int32_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

void Qwen3ASR::set_progress_callback(progress_callback_t callback) {
    progress_callback_ = std::move(callback);
}

// --- Forced Alignment ---

bool Qwen3ASR::load_aligner(const std::string & aligner_path) {
    int64_t t_start = get_time_ms();

    if (!aligner_encoder_.load_model(aligner_path)) {
        error_msg_ = "Failed to load aligner audio encoder: " + aligner_encoder_.get_error();
        return false;
    }

    if (!aligner_decoder_.load_model(aligner_path)) {
        error_msg_ = "Failed to load aligner text decoder: " + aligner_decoder_.get_error();
        return false;
    }

    if (!aligner_decoder_.is_aligner()) {
        error_msg_ = "Model is not a ForcedAligner (no classify_num in metadata)";
        return false;
    }

    aligner_loaded_ = true;

    int64_t t_end = get_time_ms();
    fprintf(stderr, "Aligner loaded in %lld ms\n", (long long)(t_end - t_start));

    return true;
}

std::vector<int32_t> Qwen3ASR::build_align_tokens(
    int32_t n_audio_frames,
    const std::string & transcript,
    std::vector<std::string> & out_words) {

    const auto & cfg = aligner_decoder_.get_config();

    // Split transcript into words: space-delimited for Latin scripts,
    // character-level for CJK (Chinese/Japanese/Korean)
    out_words.clear();
    std::string word;
    for (size_t i = 0; i < transcript.size(); ) {
        unsigned char c = (unsigned char)transcript[i];
        if (c == ' ' || c == '\t') {
            if (!word.empty()) { out_words.push_back(word); word.clear(); }
            i++;
            continue;
        }
        // Decode UTF-8 codepoint
        uint32_t cp = 0;
        int len = 1;
        if (c < 0x80)                  { cp = c;           len = 1; }
        else if ((c & 0xE0) == 0xC0)  { cp = c & 0x1F;    len = 2; }
        else if ((c & 0xF0) == 0xE0)  { cp = c & 0x0F;    len = 3; }
        else if ((c & 0xF8) == 0xF0)  { cp = c & 0x07;    len = 4; }
        for (int j = 1; j < len && i + j < transcript.size(); ++j) {
            cp = (cp << 6) | ((unsigned char)transcript[i + j] & 0x3F);
        }
        // CJK punctuation — skip (don't add as alignment word)
        bool is_cjk_punct = (cp >= 0x3000 && cp <= 0x303F)   // CJK symbols & punctuation
                          || (cp >= 0xFF01 && cp <= 0xFF0F)   // Fullwidth ! through /
                          || (cp >= 0xFF1A && cp <= 0xFF20)   // Fullwidth : through @
                          || cp == 0xFF0C || cp == 0xFF0E     // Fullwidth comma, period
                          || cp == 0x3001 || cp == 0x3002;    // Ideographic comma, period
        if (is_cjk_punct) {
            if (!word.empty()) { out_words.push_back(word); word.clear(); }
            // Skip — don't add punctuation as alignment word
        } else {
            // CJK character ranges — split each as individual word
            bool is_cjk = (cp >= 0x4E00 && cp <= 0x9FFF)   // CJK Unified
                        || (cp >= 0x3400 && cp <= 0x4DBF)   // CJK Extension A
                        || (cp >= 0x3040 && cp <= 0x309F)   // Hiragana
                        || (cp >= 0x30A0 && cp <= 0x30FF)   // Katakana
                        || (cp >= 0xAC00 && cp <= 0xD7AF)   // Hangul
                        || (cp >= 0xF900 && cp <= 0xFAFF)   // CJK Compat
                        || (cp >= 0x20000 && cp <= 0x2A6DF); // CJK Extension B
            if (is_cjk) {
                if (!word.empty()) { out_words.push_back(word); word.clear(); }
                out_words.push_back(transcript.substr(i, len));
            } else {
                word.append(transcript, i, len);
            }
        }
        i += len;
    }
    if (!word.empty()) out_words.push_back(word);

    // Build token sequence:
    // <|im_start|>system\n<|im_end|>\n<|im_start|>user\n
    // <|audio_start|><|audio_pad|>×N<|audio_end|>
    // [word1_bpe_tokens] <timestamp> <timestamp> [word2_bpe_tokens] <timestamp> <timestamp> ...
    // <|im_end|>\n<|im_start|>assistant\n

    const int32_t im_start = 151644;
    const int32_t im_end = 151645;
    const int32_t system_token = 8948;
    const int32_t user_token = 872;
    const int32_t assistant_token = 77091;
    const int32_t newline = 198;
    const int32_t timestamp_id = cfg.timestamp_token_id;

    std::vector<int32_t> tokens;
    tokens.reserve(n_audio_frames + out_words.size() * 4 + 20);

    // <|im_start|>system\n<|im_end|>\n
    tokens.push_back(im_start);
    tokens.push_back(system_token);
    tokens.push_back(newline);
    tokens.push_back(im_end);
    tokens.push_back(newline);

    // <|im_start|>user\n
    tokens.push_back(im_start);
    tokens.push_back(user_token);
    tokens.push_back(newline);

    // <|audio_start|><|audio_pad|>×N<|audio_end|>
    tokens.push_back(cfg.audio_start_token_id);
    for (int32_t i = 0; i < n_audio_frames; ++i) {
        tokens.push_back(cfg.audio_pad_token_id);
    }
    tokens.push_back(cfg.audio_end_token_id);

    // Encode each word and append timestamp pairs
    for (const auto & w : out_words) {
        // BPE encode the word (with leading space for GPT-2 convention)
        std::string word_with_space = " " + w;
        auto word_ids = aligner_decoder_.encode_text(word_with_space);
        tokens.insert(tokens.end(), word_ids.begin(), word_ids.end());
        tokens.push_back(timestamp_id);
        tokens.push_back(timestamp_id);
    }

    // <|im_end|>\n<|im_start|>assistant\n
    tokens.push_back(im_end);
    tokens.push_back(newline);
    tokens.push_back(im_start);
    tokens.push_back(assistant_token);
    tokens.push_back(newline);

    return tokens;
}

// Longest Increasing Subsequence — returns indices of LIS elements
static std::vector<int> lis_indices(const std::vector<int32_t> & vals) {
    int n = (int)vals.size();
    if (n == 0) return {};

    std::vector<int> dp(n, 1);
    std::vector<int> parent(n, -1);

    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (vals[j] <= vals[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
    }

    int max_len = 0, max_idx = 0;
    for (int i = 0; i < n; ++i) {
        if (dp[i] > max_len) { max_len = dp[i]; max_idx = i; }
    }

    std::vector<int> lis;
    for (int i = max_idx; i >= 0; i = parent[i]) {
        lis.push_back(i);
    }
    std::reverse(lis.begin(), lis.end());
    return lis;
}

// Fix non-monotonic timestamps via LIS + linear interpolation
static void fix_timestamps(std::vector<int32_t> & ts_values) {
    if (ts_values.size() < 2) return;

    auto lis = lis_indices(ts_values);

    // Mark non-LIS positions for interpolation
    std::vector<bool> is_lis(ts_values.size(), false);
    for (int idx : lis) is_lis[idx] = true;

    // Interpolate non-LIS values
    // Find previous and next LIS anchor for each non-LIS position
    for (size_t i = 0; i < ts_values.size(); ++i) {
        if (is_lis[i]) continue;

        // Find prev anchor
        int prev_idx = -1;
        int32_t prev_val = 0;
        for (int j = (int)i - 1; j >= 0; --j) {
            if (is_lis[j]) { prev_idx = j; prev_val = ts_values[j]; break; }
        }

        // Find next anchor
        int next_idx = (int)ts_values.size();
        int32_t next_val = ts_values.back();
        for (size_t j = i + 1; j < ts_values.size(); ++j) {
            if (is_lis[j]) { next_idx = (int)j; next_val = ts_values[j]; break; }
        }

        // Linear interpolation
        if (next_idx > prev_idx + 1) {
            float frac = (float)(i - (prev_idx + 1)) / (float)(next_idx - prev_idx - 1);
            ts_values[i] = prev_val + (int32_t)(frac * (next_val - prev_val));
        }
    }
}

align_result Qwen3ASR::align(const float * samples, int n_samples,
                              const std::string & transcript,
                              const transcribe_params & params) {
    align_result result;
    int64_t t_total_start = get_time_ms();

    if (!aligner_loaded_) {
        result.error_msg = "Aligner model not loaded";
        return result;
    }

    // 1. Compute mel spectrogram
    int64_t t_mel_start = get_time_ms();
    MelSpectrogram mel;
    if (!log_mel_spectrogram(samples, n_samples, mel_filters_, mel, params.n_threads)) {
        result.error_msg = "Failed to compute mel spectrogram";
        return result;
    }
    result.t_mel_ms = get_time_ms() - t_mel_start;

    // 2. Audio encoding
    int64_t t_encode_start = get_time_ms();
    std::vector<float> audio_features;
    if (!aligner_encoder_.encode(mel.data.data(), mel.n_mel, mel.n_len, audio_features)) {
        result.error_msg = "Failed to encode audio: " + aligner_encoder_.get_error();
        return result;
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;

    const auto & text_hparams = aligner_encoder_.get_text_hparams();
    int32_t n_audio_frames = audio_features.size() / text_hparams.hidden_size;

    // 3. Build input tokens with timestamp slots
    std::vector<std::string> words;
    std::vector<int32_t> input_tokens = build_align_tokens(n_audio_frames, transcript, words);

    if (words.empty()) {
        result.error_msg = "No words found in transcript";
        return result;
    }

    const auto & cfg = aligner_decoder_.get_config();

    // Audio pad tokens start after: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
    int32_t audio_start_pos = 9;

    fprintf(stderr, "  Align: %zu words, %zu input tokens, %d audio frames\n",
            words.size(), input_tokens.size(), n_audio_frames);

    // 4. Single forward pass through classify head
    int64_t t_align_start = get_time_ms();
    std::vector<float> logits;
    if (!aligner_decoder_.forward_classify(
            input_tokens.data(), (int32_t)input_tokens.size(),
            audio_features.data(), n_audio_frames,
            audio_start_pos, logits)) {
        result.error_msg = "Classify forward failed: " + aligner_decoder_.get_error();
        return result;
    }
    result.t_align_ms = get_time_ms() - t_align_start;

    // 5. Extract timestamps from positions where input_id == timestamp_token_id
    int32_t classify_num = cfg.classify_num;
    int32_t timestamp_id = cfg.timestamp_token_id;
    int32_t segment_time = cfg.timestamp_segment_time; // 80ms

    std::vector<int32_t> ts_values;
    for (size_t i = 0; i < input_tokens.size(); ++i) {
        if (input_tokens[i] == timestamp_id) {
            // argmax over classify_num classes
            const float * row = logits.data() + i * classify_num;
            int32_t max_idx = 0;
            float max_val = row[0];
            for (int32_t j = 1; j < classify_num; ++j) {
                if (row[j] > max_val) { max_val = row[j]; max_idx = j; }
            }
            ts_values.push_back(max_idx);
        }
    }

    // Should have 2 timestamps per word
    if ((int)ts_values.size() != (int)words.size() * 2) {
        result.error_msg = "Timestamp count mismatch: got " +
                           std::to_string(ts_values.size()) + " expected " +
                           std::to_string(words.size() * 2);
        return result;
    }

    // 6. Fix monotonicity
    fix_timestamps(ts_values);

    // 7. Convert to seconds and build result
    result.words.resize(words.size());
    for (size_t i = 0; i < words.size(); ++i) {
        result.words[i].word = words[i];
        result.words[i].start_s = (float)(ts_values[i * 2] * segment_time) / 1000.0f;
        result.words[i].end_s = (float)(ts_values[i * 2 + 1] * segment_time) / 1000.0f;
    }

    result.success = true;
    result.t_total_ms = get_time_ms() - t_total_start;

    if (params.print_timing) {
        fprintf(stderr, "\nAlignment Timing:\n");
        fprintf(stderr, "  Mel spectrogram: %lld ms\n", (long long)result.t_mel_ms);
        fprintf(stderr, "  Audio encoding:  %lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Alignment:       %lld ms\n", (long long)result.t_align_ms);
        fprintf(stderr, "  Total:           %lld ms\n", (long long)result.t_total_ms);
    }

    return result;
}

bool load_audio_file(const std::string & path, std::vector<float> & samples, int & sample_rate) {
    float * raw = nullptr;
    int n = 0;
    int sr = 0;
    if (!vocal_audio_read(path.c_str(), &raw, &n, &sr, 0)) {
        return false;
    }
    samples.assign(raw, raw + n);
    sample_rate = sr;
    free(raw);
    return true;
}

} // namespace qwen3_asr
