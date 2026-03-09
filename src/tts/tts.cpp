#include "tts.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>

namespace vocal_tts {

static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

TTS::TTS() = default;

TTS::~TTS() = default;

int32_t TTS::get_speaker_id(const std::string & name) const {
    // CustomVoice speaker IDs (from model config)
    std::string lower = name;
    for (auto & c : lower) c = tolower(c);
    if (lower == "serena")   return 3066;
    if (lower == "vivian")   return 3065;
    if (lower == "uncle_fu") return 3010;
    if (lower == "ryan")     return 3061;
    if (lower == "aiden")    return 2861;
    if (lower == "ono_anna") return 2873;
    if (lower == "sohee")    return 2864;
    if (lower == "eric")     return 2875;
    if (lower == "dylan")    return 2878;
    return -1;
}

int32_t TTS::get_language_id(const std::string & name) const {
    std::string lower = name;
    for (auto & c : lower) c = tolower(c);
    if (lower == "english")    return 2050;
    if (lower == "chinese")    return 2055;
    if (lower == "german")     return 2053;
    if (lower == "italian")    return 2070;
    if (lower == "portuguese") return 2071;
    if (lower == "spanish")    return 2054;
    if (lower == "japanese")   return 2058;
    if (lower == "korean")     return 2064;
    if (lower == "french")     return 2061;
    if (lower == "russian")    return 2069;
    if (lower == "auto")       return -1; // Auto-detect
    return -1;
}

bool TTS::load(const std::string & model_path,
               const std::string & tokenizer_path,
               const std::string & decoder_path) {
    if (!tokenizer_.load(tokenizer_path)) {
        error_ = "Failed to load tokenizer: " + tokenizer_.get_error();
        return false;
    }
    if (!talker_.load_model(model_path)) {
        error_ = "Failed to load talker model: " + talker_.get_error();
        return false;
    }
    decoder_ = std::make_unique<AudioDecoder>();
    if (!decoder_->load(decoder_path)) {
        error_ = "Failed to load audio decoder: " + decoder_->get_error();
        return false;
    }
    loaded_ = true;
    fprintf(stderr, "TTS model loaded successfully\n");
    return true;
}

void TTS::build_prompt_embeds(const std::vector<int32_t> & text_tokens,
                               int32_t speaker_id, int32_t language_id,
                               std::vector<float> & out_embeds,
                               std::vector<float> & tts_pad_embed) {
    const auto & cfg = talker_.get_config();
    const int H = cfg.hidden_size;  // 1024
    const auto & sp = tokenizer_.special();

    // Pre-compute frequently used embeddings
    tts_pad_embed.resize(H);
    talker_.compute_text_embedding(sp.tts_pad, tts_pad_embed.data());

    std::vector<float> tts_bos_embed(H);
    talker_.compute_text_embedding(sp.tts_bos, tts_bos_embed.data());

    std::vector<float> tts_eos_embed(H);
    talker_.compute_text_embedding(sp.tts_eos, tts_eos_embed.data());

    // Build codec prefill list
    // With language: [codec_think, codec_think_bos, language_id, codec_think_eos]
    // Without language (auto): [codec_nothink, codec_think_bos, codec_think_eos]
    std::vector<int32_t> codec_prefill;
    if (language_id >= 0) {
        codec_prefill = {CODEC_THINK, CODEC_THINK_BOS, language_id, CODEC_THINK_EOS};
    } else {
        codec_prefill = {CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS};
    }

    // With speaker: add speaker_id between think_eos and codec_pad
    std::vector<int32_t> codec_sequence;  // Full codec token sequence
    codec_sequence.insert(codec_sequence.end(), codec_prefill.begin(), codec_prefill.end());
    if (speaker_id >= 0) {
        codec_sequence.push_back(speaker_id);
    }
    codec_sequence.push_back(CODEC_PAD);
    codec_sequence.push_back(CODEC_BOS);
    // codec_sequence now: [think_stuff..., (speaker), codec_pad, codec_bos]

    int n_codec = (int)codec_sequence.size();

    // Prompt structure (non-streaming mode):
    // Pos 0-2:  text=[im_start, assistant, \n], codec=none (pure text)
    // Pos 3..3+n_codec-3: text=tts_pad, codec=codec_sequence[:-1]
    // Pos 3+n_codec-2:    text=tts_bos, codec=codec_sequence[-2] (codec_pad)
    // [then for non-streaming: all text tokens + tts_eos paired with codec_pad, then tts_pad+codec_bos]
    //
    // Wait, let me re-derive from the Python code:
    //   _talker_input_embed_role = text_proj(text_embd([im_start, assistant, \n])) → 3 positions
    //   n_codec_minus_1 = n_codec - 1  (drop the last codec_bos from paired section)
    //   text side: [tts_pad * (n_codec - 2), tts_bos]  → n_codec-1 positions
    //   codec side: codec_sequence[:-1]                  → n_codec-1 positions
    //   Non-streaming: text content paired with codec_pad, then tts_eos, then tts_pad+codec_bos

    int n_text_content = (int)text_tokens.size();
    // text_tokens from build_tts_prompt_v2: just the encoded text content (no special tokens)

    // Total prompt positions:
    // 3 (role) + (n_codec-1) (control) + n_text_content (text paired with codec_pad)
    //   + 1 (tts_eos paired with codec_pad) + 1 (tts_pad paired with codec_bos)
    int n_prompt = 3 + (n_codec - 1) + n_text_content + 1 + 1;
    out_embeds.resize(n_prompt * H);

    int pos = 0;

    // === Positions 0-2: Role (pure text, no codec) ===
    int32_t role_tokens[3] = {sp.im_start, sp.assistant, sp.newline};
    for (int i = 0; i < 3; i++) {
        talker_.compute_text_embedding(role_tokens[i], &out_embeds[pos * H]);
        pos++;
    }

    // === Control positions: text=tts_pad/tts_bos + codec ===
    std::vector<float> text_emb(H), codec_emb(H);
    for (int i = 0; i < n_codec - 1; i++) {
        // Text side: tts_pad for all except the last, which is tts_bos
        if (i < n_codec - 2) {
            memcpy(text_emb.data(), tts_pad_embed.data(), H * sizeof(float));
        } else {
            memcpy(text_emb.data(), tts_bos_embed.data(), H * sizeof(float));
        }
        // Codec side
        talker_.compute_codec_embedding(codec_sequence[i], codec_emb.data());
        // Add: text + codec
        for (int d = 0; d < H; d++) {
            out_embeds[pos * H + d] = text_emb[d] + codec_emb[d];
        }
        pos++;
    }

    // === Text content positions: text=content_token + codec=codec_pad ===
    std::vector<float> codec_pad_emb(H);
    talker_.compute_codec_embedding(CODEC_PAD, codec_pad_emb.data());

    for (int i = 0; i < n_text_content; i++) {
        talker_.compute_text_embedding(text_tokens[i], text_emb.data());
        for (int d = 0; d < H; d++) {
            out_embeds[pos * H + d] = text_emb[d] + codec_pad_emb[d];
        }
        pos++;
    }

    // === tts_eos + codec_pad ===
    for (int d = 0; d < H; d++) {
        out_embeds[pos * H + d] = tts_eos_embed[d] + codec_pad_emb[d];
    }
    pos++;

    // === tts_pad + codec_bos (generation start marker) ===
    std::vector<float> codec_bos_emb(H);
    talker_.compute_codec_embedding(CODEC_BOS, codec_bos_emb.data());
    for (int d = 0; d < H; d++) {
        out_embeds[pos * H + d] = tts_pad_embed[d] + codec_bos_emb[d];
    }
    pos++;

    fprintf(stderr, "Prompt: %d positions (3 role + %d control + %d text + 1 eos + 1 bos)\n",
            n_prompt, n_codec - 1, n_text_content);

}

void TTS::generate_codes_v2(const std::vector<float> & prompt_embeds,
                             int32_t n_prompt_tokens,
                             const std::vector<float> & tts_pad_embed,
                             const tts_params & params,
                             std::vector<std::vector<int32_t>> & out_multi_codes) {
    const auto & cfg = talker_.get_config();
    const auto & cp_cfg = talker_.get_code_pred_config();
    const int H = cfg.hidden_size;
    const int n_codebooks = cp_cfg.num_code_groups;  // 16

    talker_.clear_kv_cache();

    // 1. Prefill: process prompt with pre-computed embeddings
    // MRoPE position IDs: [n_tokens * 4] = [temporal, height, width, extra]
    // For TTS (text-only), all dimensions use the same positions
    std::vector<int32_t> pos_ids(4 * n_prompt_tokens);
    for (int j = 0; j < n_prompt_tokens; j++) {
        pos_ids[0 * n_prompt_tokens + j] = j;  // temporal
        pos_ids[1 * n_prompt_tokens + j] = j;  // height
        pos_ids[2 * n_prompt_tokens + j] = j;  // width
        pos_ids[3 * n_prompt_tokens + j] = j;  // extra
    }

    std::vector<float> logits;
    std::vector<float> hidden;

    if (!talker_.forward_embeds(prompt_embeds.data(), pos_ids.data(), n_prompt_tokens, 0,
                                logits, &hidden)) {
        fprintf(stderr, "error: prompt forward failed\n");
        return;
    }

    // Sampling with temperature=0.9, top_k=50, repetition_penalty=1.05
    const float temperature = 0.9f;
    const int top_k = 50;
    const float rep_penalty = 1.05f;
    std::mt19937 rng(42);  // Fixed seed for reproducibility

    std::vector<int32_t> generated_codes;  // For repetition penalty

    auto sample_token = [&](float * lgt, int vsize) -> int32_t {
        // Suppress special tokens (>= 2048) except EOS
        for (int v = 2048; v < vsize; v++) {
            if (v != CODEC_EOS) {
                lgt[v] = -1e30f;
            }
        }

        // Repetition penalty
        for (int32_t prev : generated_codes) {
            if (prev >= 0 && prev < vsize) {
                if (lgt[prev] > 0) {
                    lgt[prev] /= rep_penalty;
                } else {
                    lgt[prev] *= rep_penalty;
                }
            }
        }

        // Temperature
        for (int v = 0; v < vsize; v++) {
            lgt[v] /= temperature;
        }

        // Top-k: find k-th largest value
        std::vector<std::pair<float, int>> scored(vsize);
        for (int v = 0; v < vsize; v++) {
            scored[v] = {lgt[v], v};
        }
        std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                          [](const auto & a, const auto & b) { return a.first > b.first; });

        // Softmax over top-k
        float max_val = scored[0].first;
        float sum = 0.0f;
        std::vector<float> probs(top_k);
        for (int i = 0; i < top_k; i++) {
            probs[i] = expf(scored[i].first - max_val);
            sum += probs[i];
        }
        for (int i = 0; i < top_k; i++) {
            probs[i] /= sum;
        }

        // Sample from distribution
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        int idx = dist(rng);
        int32_t token = scored[idx].second;

        generated_codes.push_back(token);
        return token;
    };

    // Get last token's logits → first code_0
    int vocab_size = cfg.codec_vocab_size;
    float * last_logits = logits.data() + (n_prompt_tokens - 1) * vocab_size;
    int32_t best_id = sample_token(last_logits, vocab_size);

    // Save first hidden state (for code predictor)
    std::vector<float> last_hidden(hidden.begin() + (n_prompt_tokens - 1) * H,
                                    hidden.begin() + n_prompt_tokens * H);

    // Collect all codebook codes
    std::vector<std::vector<int32_t>> all_codes(n_codebooks);

    int32_t n_past = n_prompt_tokens;

    // 2. Autoregressive generation
    for (int32_t step = 0; step < params.max_tokens; step++) {
        if (best_id == CODEC_EOS) {
            fprintf(stderr, "  EOS at step %d\n", step);
            break;
        }

        // Store code_0
        all_codes[0].push_back(best_id);

        // Run code predictor to generate codes 1-15
        std::vector<int32_t> pred_codes;
        if (!talker_.predict_codes(last_hidden.data(), best_id, pred_codes)) {
            fprintf(stderr, "warning: code predictor failed at step %d, using zeros\n", step);
            pred_codes.assign(n_codebooks - 1, 0);
        }
        for (int cb = 1; cb < n_codebooks; cb++) {
            all_codes[cb].push_back(pred_codes[cb - 1]);
        }

        // Build next talker input: sum of ALL codebook embeddings + tts_pad_embed
        // codec_hiddens = talker.codec_embd(code_0)
        //               + code_pred.codec_embd[0](code_1)
        //               + code_pred.codec_embd[1](code_2)
        //               + ... + code_pred.codec_embd[14](code_15)
        std::vector<float> next_embed(H, 0.0f);
        std::vector<float> emb_tmp(H);

        // code_0 embedding from talker's codec_embd
        talker_.compute_codec_embedding(best_id, emb_tmp.data());
        for (int d = 0; d < H; d++) next_embed[d] += emb_tmp[d];

        // codes 1-15 embeddings from code predictor's per-codebook embeddings
        for (int cb = 0; cb < n_codebooks - 1; cb++) {
            talker_.compute_code_pred_embedding(cb, pred_codes[cb], emb_tmp.data());
            for (int d = 0; d < H; d++) next_embed[d] += emb_tmp[d];
        }

        // Add tts_pad_embed (non-streaming: trailing_text_hidden = tts_pad_embed always)
        for (int d = 0; d < H; d++) next_embed[d] += tts_pad_embed[d];

        // Position IDs for single token (4 dims for MRoPE)
        int32_t step_pos[4] = {n_past, n_past, n_past, n_past};

        if (!talker_.forward_embeds(next_embed.data(), step_pos, 1, n_past,
                                     logits, &hidden)) {
            fprintf(stderr, "error: generation forward failed at step %d\n", step);
            break;
        }

        n_past++;

        best_id = sample_token(logits.data(), vocab_size);
        last_hidden.assign(hidden.begin(), hidden.begin() + H);

        if (step > 0 && step % 100 == 0) {
            fprintf(stderr, "  Generated %d steps...\n", step);
        }
    }

    // 3. Filter special tokens from code_0 and build multi-codebook output
    int raw_len = (int)all_codes[0].size();
    std::vector<int> valid_indices;
    for (int i = 0; i < raw_len; i++) {
        if (all_codes[0][i] >= 0 && all_codes[0][i] < 2048) {
            valid_indices.push_back(i);
        }
    }

    int seq_len = (int)valid_indices.size();
    fprintf(stderr, "Generated %d steps, %d valid codes across %d codebooks\n",
            raw_len, seq_len, n_codebooks);

    out_multi_codes.resize(n_codebooks, std::vector<int32_t>(seq_len, 0));
    for (int cb = 0; cb < n_codebooks; cb++) {
        for (int j = 0; j < seq_len; j++) {
            out_multi_codes[cb][j] = all_codes[cb][valid_indices[j]];
        }
    }
}

tts_result TTS::synthesize(const std::string & text, const tts_params & params) {
    tts_result result;
    int64_t t_total_start = get_time_ms();

    if (!loaded_) {
        result.error_msg = "Model not loaded";
        return result;
    }

    // 1. Tokenize text content
    int64_t t_tok_start = get_time_ms();
    auto text_tokens = tokenizer_.encode(text);
    result.t_tokenize_ms = get_time_ms() - t_tok_start;

    fprintf(stderr, "Text tokens: %zu\n", text_tokens.size());

    // 2. Resolve speaker and language IDs
    int32_t speaker_id = get_speaker_id(params.speaker);
    int32_t language_id = get_language_id(params.language);

    if (speaker_id < 0) {
        fprintf(stderr, "Warning: unknown speaker '%s', using Vivian\n", params.speaker.c_str());
        speaker_id = 3065;  // Vivian
    }
    fprintf(stderr, "Speaker: %s (id=%d), Language: %s (id=%d)\n",
            params.speaker.c_str(), speaker_id,
            params.language.c_str(), language_id);

    // 3. Build prompt embeddings (text+codec dual embedding)
    std::vector<float> prompt_embeds;
    std::vector<float> tts_pad_embed;
    build_prompt_embeds(text_tokens, speaker_id, language_id, prompt_embeds, tts_pad_embed);

    int n_prompt = (int)(prompt_embeds.size() / talker_.get_config().hidden_size);

    // 4. Generate audio codes
    int64_t t_gen_start = get_time_ms();
    std::vector<std::vector<int32_t>> multi_codes;
    generate_codes_v2(prompt_embeds, n_prompt, tts_pad_embed, params, multi_codes);
    result.t_generate_ms = get_time_ms() - t_gen_start;

    if (multi_codes.empty() || multi_codes[0].empty()) {
        result.error_msg = "No audio codes generated";
        return result;
    }

    result.n_tokens_generated = (int32_t)multi_codes[0].size();

    // 5. Decode to waveform
    int64_t t_dec_start = get_time_ms();
    result.audio = decoder_->decode(multi_codes);
    result.t_decode_ms = get_time_ms() - t_dec_start;

    if (result.audio.empty()) {
        result.error_msg = "Decoder failed: " + decoder_->get_error();
        return result;
    }

    result.sample_rate = decoder_->get_config().sample_rate;
    result.success = true;
    result.t_total_ms = get_time_ms() - t_total_start;

    if (params.print_timing) {
        int n_samples = (int)result.audio.size();
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Tokenize:   %lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Generate:   %lld ms (%d codes)\n", (long long)result.t_generate_ms, result.n_tokens_generated);
        fprintf(stderr, "  Decode:     %lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:      %lld ms\n", (long long)result.t_total_ms);
        fprintf(stderr, "  Output:     %d samples (%.1f sec at %d Hz)\n",
                n_samples, (float)n_samples / result.sample_rate, result.sample_rate);
    }

    return result;
}

} // namespace vocal_tts
