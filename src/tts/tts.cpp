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

#include "audio_io.h"

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
    if (lower == "auto")             return -1; // Auto-detect
    if (lower == "sichuan_dialect")  return 2062;
    if (lower == "beijing_dialect")  return 2074;
    return -1;
}

bool TTS::load(const std::string & model_path,
               const std::string & tokenizer_path,
               const std::string & decoder_path) {
    int64_t t_start = get_time_ms();
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
    load_time_ms_ += get_time_ms() - t_start;
    return true;
}

bool TTS::load_encoders(const std::string & codec_encoder_path,
                         const std::string & speaker_encoder_path) {
    int64_t t_start = get_time_ms();
    if (!codec_encoder_path.empty()) {
        codec_encoder_ = std::make_unique<CodecEncoder>();
        if (!codec_encoder_->load(codec_encoder_path)) {
            error_ = "Failed to load codec encoder: " + codec_encoder_->get_error();
            return false;
        }
    }

    // Speaker encoder loads from GGUF (main model file or separate GGUF)
    speaker_encoder_ = std::make_unique<SpeakerEncoder>();
    if (!speaker_encoder_->load(speaker_encoder_path)) {
        fprintf(stderr, "Warning: Speaker encoder not available: %s\n",
                speaker_encoder_->get_error().c_str());
        fprintf(stderr, "  Voice cloning will use default speaker + reference audio codes\n");
        speaker_encoder_.reset();
    }

    encoders_loaded_ = true;
    load_time_ms_ += get_time_ms() - t_start;
    return true;
}

bool TTS::load_reference_audio(const std::string & path, std::vector<float> & out_audio) {
    float * samples = nullptr;
    int n_samples = 0;
    int sample_rate = 0;

    if (!vocal_audio_read(path.c_str(), &samples, &n_samples, &sample_rate, 24000)) {
        error_ = "Failed to read reference audio: " + path;
        return false;
    }

    out_audio.assign(samples, samples + n_samples);
    free(samples);
    return true;
}

void TTS::build_instruct_embeds(const std::string & instruct_text,
                                 std::vector<float> & out_embeds,
                                 int & n_instruct_tokens) {
    const auto & cfg = talker_.get_config();
    const int H = cfg.hidden_size;
    const auto & sp = tokenizer_.special();

    // Tokenize: <|im_start|>user\n{instruct}<|im_end|>\n
    auto instruct_tokens = tokenizer_.encode(instruct_text);

    // Build full sequence: [im_start, "user", newline, ...instruct_tokens..., im_end, newline]
    // The token for "user" needs to be looked up — encode "user" to get token IDs
    auto user_tokens = tokenizer_.encode("user");

    std::vector<int32_t> full_tokens;
    full_tokens.push_back(sp.im_start);
    full_tokens.insert(full_tokens.end(), user_tokens.begin(), user_tokens.end());
    full_tokens.push_back(sp.newline);
    full_tokens.insert(full_tokens.end(), instruct_tokens.begin(), instruct_tokens.end());
    full_tokens.push_back(sp.im_end);
    full_tokens.push_back(sp.newline);

    n_instruct_tokens = (int)full_tokens.size();
    out_embeds.resize(n_instruct_tokens * H);

    // Compute text-only embeddings (no codec pairing) via text_embedding + text_projection
    for (int i = 0; i < n_instruct_tokens; i++) {
        talker_.compute_text_embedding(full_tokens[i], &out_embeds[i * H]);
    }
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
    std::vector<int32_t> codec_prefill;
    if (language_id >= 0) {
        codec_prefill = {CODEC_THINK, CODEC_THINK_BOS, language_id, CODEC_THINK_EOS};
    } else {
        codec_prefill = {CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS};
    }

    std::vector<int32_t> codec_sequence;
    codec_sequence.insert(codec_sequence.end(), codec_prefill.begin(), codec_prefill.end());
    if (speaker_id >= 0) {
        codec_sequence.push_back(speaker_id);
    }
    codec_sequence.push_back(CODEC_PAD);
    codec_sequence.push_back(CODEC_BOS);

    int n_codec = (int)codec_sequence.size();
    int n_text_content = (int)text_tokens.size();
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
        if (i < n_codec - 2) {
            memcpy(text_emb.data(), tts_pad_embed.data(), H * sizeof(float));
        } else {
            memcpy(text_emb.data(), tts_bos_embed.data(), H * sizeof(float));
        }
        talker_.compute_codec_embedding(codec_sequence[i], codec_emb.data());
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

    (void)n_prompt;
}

void TTS::build_prompt_embeds_xvec(const std::vector<int32_t> & text_tokens,
                                    const std::vector<float> & speaker_embed,
                                    int32_t language_id,
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

    // Build codec prefill (without speaker_id — we use speaker_embed directly)
    std::vector<int32_t> codec_prefill;
    if (language_id >= 0) {
        codec_prefill = {CODEC_THINK, CODEC_THINK_BOS, language_id, CODEC_THINK_EOS};
    } else {
        codec_prefill = {CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS};
    }

    // Codec sequence: [prefill..., CODEC_PAD, CODEC_BOS]
    // Speaker embedding is inserted as a separate position between prefill and CODEC_PAD
    int n_prefill = (int)codec_prefill.size();
    int n_text_content = (int)text_tokens.size();

    // Prompt structure:
    // 3 role + n_prefill control + 1 speaker_embed + 2 (codec_pad, codec_bos via tts_bos) + n_text + 1 eos + 1 bos
    // Wait — let me match the Python structure exactly:
    // codec_input_embedding = cat([codec_emb_0, speaker_embed, codec_emb_1])
    // where codec_emb_0 = [think, think_bos, lang, think_eos] (prefill)
    //       codec_emb_1 = [codec_pad, codec_bos]
    // So the full codec sequence has n_prefill + 1 (speaker) + 2 = n_prefill + 3 tokens
    // But the paired section excludes the last CODEC_BOS (it becomes the generation start)

    // Full codec token sequence (logical):
    // [prefill..., <speaker_embed>, CODEC_PAD, CODEC_BOS]
    int n_codec_total = n_prefill + 1 + 2;  // +1 speaker, +2 (pad, bos)

    // Paired section: n_codec_total - 1 positions (exclude last CODEC_BOS)
    // Then: n_text content + 1 eos + 1 bos
    int n_prompt = 3 + (n_codec_total - 1) + n_text_content + 1 + 1;
    out_embeds.resize(n_prompt * H);

    int pos = 0;

    // === Positions 0-2: Role (pure text, no codec) ===
    int32_t role_tokens[3] = {sp.im_start, sp.assistant, sp.newline};
    for (int i = 0; i < 3; i++) {
        talker_.compute_text_embedding(role_tokens[i], &out_embeds[pos * H]);
        pos++;
    }

    std::vector<float> text_emb(H), codec_emb(H);
    std::vector<float> codec_pad_emb(H);
    talker_.compute_codec_embedding(CODEC_PAD, codec_pad_emb.data());

    // === Control: prefill tokens (text=tts_pad, codec=prefill_token) ===
    for (int i = 0; i < n_prefill; i++) {
        memcpy(text_emb.data(), tts_pad_embed.data(), H * sizeof(float));
        talker_.compute_codec_embedding(codec_prefill[i], codec_emb.data());
        for (int d = 0; d < H; d++) {
            out_embeds[pos * H + d] = text_emb[d] + codec_emb[d];
        }
        pos++;
    }

    // === Speaker embedding position (text=tts_pad, codec=speaker_embed directly) ===
    {
        // The speaker embedding is used directly as the codec embedding
        // (no lookup — it's the raw ECAPA-TDNN output, same dimensionality as hidden_size)
        for (int d = 0; d < H; d++) {
            out_embeds[pos * H + d] = tts_pad_embed[d] + speaker_embed[d];
        }
        pos++;
    }

    // === CODEC_PAD position (text=tts_pad, codec=codec_pad) ===
    for (int d = 0; d < H; d++) {
        out_embeds[pos * H + d] = tts_pad_embed[d] + codec_pad_emb[d];
    }
    pos++;

    // === tts_bos + last codec before content (this was CODEC_PAD in paired, but actually
    //     looking at the code again: the last paired position is tts_bos + last_codec_before_bos)
    //     Actually, let me re-derive. In the regular path:
    //     codec_sequence = [prefill..., speaker_id, CODEC_PAD, CODEC_BOS]
    //     paired section iterates 0..n_codec-2 (skipping last CODEC_BOS)
    //     last paired position (i == n_codec-2): text=tts_bos
    //     So for xvec: codec_sequence = [prefill..., <spk>, CODEC_PAD, CODEC_BOS]
    //     paired 0..n_codec_total-2:
    //       0..n_prefill-1: prefill tokens with tts_pad
    //       n_prefill: speaker_embed with tts_pad
    //       n_prefill+1: CODEC_PAD with tts_bos  (this is the last paired position)
    //     Then content, eos, bos follow ===
    // Hmm wait, I already placed the CODEC_PAD with tts_pad above. Let me redo this.
    // The issue is that the last paired position should use tts_bos, not tts_pad.
    // Let me restructure.

    // Actually, let me just use the same approach as the regular build_prompt_embeds.
    // Let me back up and redo.

    // I'll build a combined structure that handles the speaker embed position specially.
    // Reset to pos = 3 (after role tokens)
    pos = 3;
    out_embeds.resize(n_prompt * H);

    // Build the "codec_sequence" but mark the speaker position
    // Codec sequence: [prefill_0, ..., prefill_n-1, <SPK>, CODEC_PAD, CODEC_BOS]
    // The paired section is all except the last (CODEC_BOS)
    // In paired: last position gets tts_bos, rest get tts_pad
    int n_paired = n_codec_total - 1;  // all except CODEC_BOS

    for (int i = 0; i < n_paired; i++) {
        // Text side
        if (i < n_paired - 1) {
            memcpy(text_emb.data(), tts_pad_embed.data(), H * sizeof(float));
        } else {
            memcpy(text_emb.data(), tts_bos_embed.data(), H * sizeof(float));
        }

        // Codec side
        if (i < n_prefill) {
            // Prefill token
            talker_.compute_codec_embedding(codec_prefill[i], codec_emb.data());
        } else if (i == n_prefill) {
            // Speaker embedding (raw, not from codec table)
            memcpy(codec_emb.data(), speaker_embed.data(), H * sizeof(float));
        } else {
            // CODEC_PAD
            memcpy(codec_emb.data(), codec_pad_emb.data(), H * sizeof(float));
        }

        for (int d = 0; d < H; d++) {
            out_embeds[pos * H + d] = text_emb[d] + codec_emb[d];
        }
        pos++;
    }

    // === Text content: text_token + codec_pad ===
    for (int i = 0; i < n_text_content; i++) {
        talker_.compute_text_embedding(text_tokens[i], text_emb.data());
        for (int d = 0; d < H; d++) {
            out_embeds[pos * H + d] = text_emb[d] + codec_pad_emb[d];
        }
        pos++;
    }

    // === tts_eos + codec_pad ===
    talker_.compute_text_embedding(sp.tts_eos, text_emb.data());
    for (int d = 0; d < H; d++) {
        out_embeds[pos * H + d] = text_emb[d] + codec_pad_emb[d];
    }
    pos++;

    // === tts_pad + codec_bos (generation start) ===
    std::vector<float> codec_bos_emb(H);
    talker_.compute_codec_embedding(CODEC_BOS, codec_bos_emb.data());
    for (int d = 0; d < H; d++) {
        out_embeds[pos * H + d] = tts_pad_embed[d] + codec_bos_emb[d];
    }
    pos++;

    (void)n_prompt; (void)n_paired; (void)n_text_content;
}

void TTS::build_prompt_embeds_icl(const std::vector<int32_t> & text_tokens,
                                   const std::vector<int32_t> & ref_text_tokens,
                                   const std::vector<std::vector<int32_t>> & ref_codes,
                                   const std::vector<float> & speaker_embed,
                                   int32_t language_id,
                                   std::vector<float> & out_embeds,
                                   std::vector<float> & tts_pad_embed,
                                   std::vector<float> & trailing_text_hidden,
                                   int & n_trailing) {
    const auto & cfg = talker_.get_config();
    const auto & cp_cfg = talker_.get_code_pred_config();
    const int H = cfg.hidden_size;  // 1024
    const int n_codebooks = cp_cfg.num_code_groups;  // 16
    const auto & sp = tokenizer_.special();

    tts_pad_embed.resize(H);
    talker_.compute_text_embedding(sp.tts_pad, tts_pad_embed.data());

    std::vector<float> tts_bos_embed(H);
    talker_.compute_text_embedding(sp.tts_bos, tts_bos_embed.data());

    std::vector<float> tts_eos_embed(H);
    talker_.compute_text_embedding(sp.tts_eos, tts_eos_embed.data());

    std::vector<float> codec_pad_emb(H);
    talker_.compute_codec_embedding(CODEC_PAD, codec_pad_emb.data());

    std::vector<float> codec_bos_emb(H);
    talker_.compute_codec_embedding(CODEC_BOS, codec_bos_emb.data());

    // Build codec prefill
    std::vector<int32_t> codec_prefill;
    if (language_id >= 0) {
        codec_prefill = {CODEC_THINK, CODEC_THINK_BOS, language_id, CODEC_THINK_EOS};
    } else {
        codec_prefill = {CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS};
    }
    int n_prefill = (int)codec_prefill.size();

    // ICL prompt structure (matches Python generate_icl_prompt, streaming mode):
    //
    // Header: role(3) + control(prefill + speaker) + transition(tts_bos + codec_pad)
    //
    // ICL section: two tracks positionally overlaid
    //   Text track: [ref_text, target_text, tts_eos, tts_pad...] (padded to n_paired)
    //   Codec track: [codec_bos, ref_code_sum_0, ..., ref_code_sum_{n-1}, codec_pad...] (padded)
    //
    //   text_lens  = len(ref_text) + len(target_text) + 1 (eos)
    //   codec_lens = 1 (bos) + n_ref_codes
    //   n_paired   = max(text_lens, codec_lens)
    //
    //   If text > codec: excess text goes to trailing_text_hidden for generation
    //   If codec >= text: text padded with tts_pad, trailing = tts_pad (n_trailing=0)

    int n_ref_codes = (ref_codes.empty() || ref_codes[0].empty()) ? 0 : (int)ref_codes[0].size();

    // Concatenate all text: [ref_text, target_text]
    std::vector<int32_t> all_text;
    all_text.insert(all_text.end(), ref_text_tokens.begin(), ref_text_tokens.end());
    all_text.insert(all_text.end(), text_tokens.begin(), text_tokens.end());
    int n_all_text = (int)all_text.size();

    int text_lens = n_all_text + 1;           // +1 for tts_eos
    int codec_lens = 1 + n_ref_codes;         // codec_bos + ref_codes
    int n_paired = std::max(text_lens, codec_lens);

    // Header: 3 (role) + n_prefill + 1 (speaker) + 1 (transition)
    int n_header = 3 + n_prefill + 1 + 1;

    // Trailing text: excess text beyond codec coverage
    if (text_lens > codec_lens) {
        n_trailing = text_lens - codec_lens;
        n_paired = codec_lens;
    } else {
        n_trailing = 0;
    }

    trailing_text_hidden.clear();

    int n_prompt = n_header + n_paired;

    out_embeds.resize(n_prompt * H);
    int pos = 0;

    // === Role tokens (pure text) ===
    int32_t role_tokens[3] = {sp.im_start, sp.assistant, sp.newline};
    for (int i = 0; i < 3; i++) {
        talker_.compute_text_embedding(role_tokens[i], &out_embeds[pos * H]);
        pos++;
    }

    std::vector<float> text_emb(H), codec_emb(H);

    // === Prefill control tokens (tts_pad + codec_prefill) ===
    for (int i = 0; i < n_prefill; i++) {
        talker_.compute_codec_embedding(codec_prefill[i], codec_emb.data());
        for (int d = 0; d < H; d++) {
            out_embeds[pos * H + d] = tts_pad_embed[d] + codec_emb[d];
        }
        pos++;
    }

    // === Speaker embedding position (tts_pad + speaker_embed) ===
    for (int d = 0; d < H; d++) {
        out_embeds[pos * H + d] = tts_pad_embed[d] + speaker_embed[d];
    }
    pos++;

    // === Transition: tts_bos + codec_pad ===
    for (int d = 0; d < H; d++) {
        out_embeds[pos * H + d] = tts_bos_embed[d] + codec_pad_emb[d];
    }
    pos++;

    // === ICL paired section ===
    std::vector<float> tmp(H);
    for (int i = 0; i < n_paired; i++) {
        // Text side: all_text[i], then tts_eos, then tts_pad
        if (i < n_all_text) {
            talker_.compute_text_embedding(all_text[i], text_emb.data());
        } else if (i == n_all_text) {
            memcpy(text_emb.data(), tts_eos_embed.data(), H * sizeof(float));
        } else {
            memcpy(text_emb.data(), tts_pad_embed.data(), H * sizeof(float));
        }

        // Codec side: codec_bos at i=0, ref_code_sums at i=1..n_ref_codes, then codec_pad
        if (i == 0) {
            memcpy(codec_emb.data(), codec_bos_emb.data(), H * sizeof(float));
        } else if (i - 1 < n_ref_codes) {
            int t = i - 1;
            memset(codec_emb.data(), 0, H * sizeof(float));
            talker_.compute_codec_embedding(ref_codes[0][t], tmp.data());
            for (int d = 0; d < H; d++) codec_emb[d] += tmp[d];
            for (int cb = 1; cb < n_codebooks && cb < (int)ref_codes.size(); cb++) {
                if (t < (int)ref_codes[cb].size()) {
                    talker_.compute_code_pred_embedding(cb - 1, ref_codes[cb][t], tmp.data());
                    for (int d = 0; d < H; d++) codec_emb[d] += tmp[d];
                }
            }
        } else {
            memcpy(codec_emb.data(), codec_pad_emb.data(), H * sizeof(float));
        }

        for (int d = 0; d < H; d++) {
            out_embeds[pos * H + d] = text_emb[d] + codec_emb[d];
        }
        pos++;
    }

    // === Build trailing text hidden (text positions beyond the paired section) ===
    if (n_trailing > 0) {
        trailing_text_hidden.resize(n_trailing * H);
        int text_offset = n_paired;
        for (int i = 0; i < n_trailing; i++) {
            int text_idx = text_offset + i;
            if (text_idx < n_all_text) {
                talker_.compute_text_embedding(all_text[text_idx], &trailing_text_hidden[i * H]);
            } else if (text_idx == n_all_text) {
                memcpy(&trailing_text_hidden[i * H], tts_eos_embed.data(), H * sizeof(float));
            } else {
                memcpy(&trailing_text_hidden[i * H], tts_pad_embed.data(), H * sizeof(float));
            }
        }
    }

    (void)n_header; (void)text_lens; (void)codec_lens;
}

void TTS::generate_codes_v2(const std::vector<float> & prompt_embeds,
                             int32_t n_prompt_tokens,
                             const std::vector<float> & tts_pad_embed,
                             const tts_params & params,
                             std::vector<std::vector<int32_t>> & out_multi_codes,
                             const std::vector<float> & trailing_text_hidden,
                             int n_trailing) {
    const auto & cfg = talker_.get_config();
    const auto & cp_cfg = talker_.get_code_pred_config();
    const int H = cfg.hidden_size;
    const int n_codebooks = cp_cfg.num_code_groups;  // 16

    talker_.clear_kv_cache();

    // 1. Prefill: process prompt with pre-computed embeddings
    std::vector<int32_t> pos_ids(4 * n_prompt_tokens);
    for (int j = 0; j < n_prompt_tokens; j++) {
        pos_ids[0 * n_prompt_tokens + j] = j;
        pos_ids[1 * n_prompt_tokens + j] = j;
        pos_ids[2 * n_prompt_tokens + j] = j;
        pos_ids[3 * n_prompt_tokens + j] = j;
    }

    std::vector<float> logits;
    std::vector<float> hidden;

    if (!talker_.forward_embeds(prompt_embeds.data(), pos_ids.data(), n_prompt_tokens, 0,
                                logits, &hidden)) {
        fprintf(stderr, "error: prompt forward failed\n");
        return;
    }

    // Sampling parameters
    const float temperature = params.temperature;
    const int top_k = params.top_k;
    const float rep_penalty = params.rep_penalty;
    std::mt19937 rng(params.seed);

    std::vector<int32_t> generated_codes;

    auto sample_token = [&](float * lgt, int vsize) -> int32_t {
        for (int v = 2048; v < vsize; v++) {
            if (v != CODEC_EOS) {
                lgt[v] = -1e30f;
            }
        }

        for (int32_t prev : generated_codes) {
            if (prev >= 0 && prev < vsize) {
                if (lgt[prev] > 0) {
                    lgt[prev] /= rep_penalty;
                } else {
                    lgt[prev] *= rep_penalty;
                }
            }
        }

        for (int v = 0; v < vsize; v++) {
            lgt[v] /= temperature;
        }

        std::vector<std::pair<float, int>> scored(vsize);
        for (int v = 0; v < vsize; v++) {
            scored[v] = {lgt[v], v};
        }
        std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                          [](const auto & a, const auto & b) { return a.first > b.first; });

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

        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        int idx = dist(rng);
        int32_t token = scored[idx].second;

        generated_codes.push_back(token);
        return token;
    };

    int vocab_size = cfg.codec_vocab_size;
    float * last_logits = logits.data() + (n_prompt_tokens - 1) * vocab_size;
    int32_t best_id = sample_token(last_logits, vocab_size);

    std::vector<float> last_hidden(hidden.begin() + (n_prompt_tokens - 1) * H,
                                    hidden.begin() + n_prompt_tokens * H);

    std::vector<std::vector<int32_t>> all_codes(n_codebooks);
    int32_t n_past = n_prompt_tokens;

    // 2. Autoregressive generation
    for (int32_t step = 0; step < params.max_tokens; step++) {
        if (best_id == CODEC_EOS) {
            break;
        }

        all_codes[0].push_back(best_id);

        std::vector<int32_t> pred_codes;
        if (!talker_.predict_codes(last_hidden.data(), best_id, pred_codes)) {
            fprintf(stderr, "warning: code predictor failed at step %d, using zeros\n", step);
            pred_codes.assign(n_codebooks - 1, 0);
        }
        for (int cb = 1; cb < n_codebooks; cb++) {
            all_codes[cb].push_back(pred_codes[cb - 1]);
        }

        // Build next talker input: sum of ALL codebook embeddings + text_hidden
        std::vector<float> next_embed(H, 0.0f);
        std::vector<float> emb_tmp(H);

        talker_.compute_codec_embedding(best_id, emb_tmp.data());
        for (int d = 0; d < H; d++) next_embed[d] += emb_tmp[d];

        for (int cb = 0; cb < n_codebooks - 1; cb++) {
            talker_.compute_code_pred_embedding(cb, pred_codes[cb], emb_tmp.data());
            for (int d = 0; d < H; d++) next_embed[d] += emb_tmp[d];
        }

        // Add text hidden: trailing_text_hidden if available, else tts_pad_embed
        if (n_trailing > 0 && step < n_trailing) {
            for (int d = 0; d < H; d++) {
                next_embed[d] += trailing_text_hidden[step * H + d];
            }
        } else {
            for (int d = 0; d < H; d++) {
                next_embed[d] += tts_pad_embed[d];
            }
        }

        int32_t step_pos[4] = {n_past, n_past, n_past, n_past};

        if (!talker_.forward_embeds(next_embed.data(), step_pos, 1, n_past,
                                     logits, &hidden)) {
            fprintf(stderr, "error: generation forward failed at step %d\n", step);
            break;
        }

        n_past++;

        best_id = sample_token(logits.data(), vocab_size);
        last_hidden.assign(hidden.begin(), hidden.begin() + H);

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
    out_multi_codes.resize(n_codebooks, std::vector<int32_t>(seq_len, 0));
    for (int cb = 0; cb < n_codebooks; cb++) {
        for (int j = 0; j < seq_len; j++) {
            out_multi_codes[cb][j] = all_codes[cb][valid_indices[j]];
        }
    }
}

bool TTS::encode_voice_profile(const std::string & ref_audio_path,
                               const std::string & ref_text,
                               VoiceProfile & out_profile) {
    if (!loaded_) {
        error_ = "Model not loaded";
        return false;
    }
    if (!encoders_loaded_) {
        error_ = "Encoder models not loaded";
        return false;
    }

    // Load reference audio
    std::vector<float> ref_audio;
    if (!load_reference_audio(ref_audio_path, ref_audio)) {
        return false;
    }
    // Encode speaker embedding
    if (speaker_encoder_) {
        out_profile.speaker_embed = speaker_encoder_->encode(ref_audio.data(), (int)ref_audio.size());
        if (out_profile.speaker_embed.empty()) {
            fprintf(stderr, "Warning: Speaker encoder failed: %s\n",
                    speaker_encoder_->get_error().c_str());
        }
    }

    // Encode codec codes
    if (!codec_encoder_) {
        error_ = "Codec encoder not loaded";
        return false;
    }
    out_profile.codec_codes = codec_encoder_->encode(ref_audio.data(), (int)ref_audio.size());
    if (out_profile.codec_codes.empty()) {
        error_ = "Codec encoder failed: " + codec_encoder_->get_error();
        return false;
    }

    out_profile.ref_text = ref_text;
    return true;
}

tts_result TTS::synthesize(const std::string & text, const tts_params & params) {
    tts_result result;
    int64_t t_total_start = get_time_ms();

    if (!loaded_) {
        result.error_msg = "Model not loaded";
        return result;
    }

    bool has_voice_profile = !params.voice_profile.empty();
    bool is_clone = !params.ref_audio_path.empty() || has_voice_profile;
    bool is_icl = is_clone && (!params.ref_text.empty() || has_voice_profile);

    if (!has_voice_profile && is_clone && !encoders_loaded_) {
        result.error_msg = "Encoder models not loaded. Call load_encoders() first.";
        return result;
    }

    // 1. Tokenize text content
    int64_t t_tok_start = get_time_ms();
    auto text_tokens = tokenizer_.encode(text);
    result.t_tokenize_ms = get_time_ms() - t_tok_start;

    // 2. Language ID (with dialect override for CustomVoice speakers)
    int32_t language_id = get_language_id(params.language);
    if (language_id < 0) {
        // Auto mode: check for dialect speakers
        std::string spk_lower = params.speaker;
        for (auto & c : spk_lower) c = tolower(c);
        if (spk_lower == "eric")  language_id = get_language_id("sichuan_dialect");
        if (spk_lower == "dylan") language_id = get_language_id("beijing_dialect");
    }

    // 3. Load reference data (from voice profile or live encoding)
    std::vector<float> speaker_embed;
    std::vector<std::vector<int32_t>> ref_codes;
    std::vector<int32_t> ref_text_tokens;

    if (has_voice_profile) {
        // Load pre-encoded voice profile
        int64_t t_enc_start = get_time_ms();

        VoiceProfile profile;
        std::string profile_error;
        if (!profile.load(params.voice_profile, profile_error)) {
            result.error_msg = profile_error;
            return result;
        }

        ref_codes = std::move(profile.codec_codes);
        speaker_embed = std::move(profile.speaker_embed);
        ref_text_tokens = tokenizer_.encode(profile.ref_text);

        // Check speaker embedding dimension
        const int H_check = talker_.get_config().hidden_size;
        if (!speaker_embed.empty() && (int)speaker_embed.size() != H_check) {
            speaker_embed.clear();
        }

        result.t_encode_ms = get_time_ms() - t_enc_start;
    } else if (is_clone) {
        int64_t t_enc_start = get_time_ms();

        // Load reference audio
        std::vector<float> ref_audio;
        if (!load_reference_audio(params.ref_audio_path, ref_audio)) {
            result.error_msg = error_;
            return result;
        }
        // Extract speaker embedding (if speaker encoder is available)
        if (speaker_encoder_) {
            speaker_embed = speaker_encoder_->encode(ref_audio.data(), (int)ref_audio.size());
            if (speaker_embed.empty()) {
                fprintf(stderr, "Warning: Speaker encoder failed: %s\n",
                        speaker_encoder_->get_error().c_str());
            }

            // Check dimension matches hidden_size — if not, discard
            const int H_check = talker_.get_config().hidden_size;
            if (!speaker_embed.empty() && (int)speaker_embed.size() != H_check) {
                fprintf(stderr, "Warning: Speaker embedding dim %zu != hidden_size %d, using default speaker\n",
                        speaker_embed.size(), H_check);
                speaker_embed.clear();
            }
        }

        // For ICL mode: also encode to codec codes and tokenize ref text
        if (is_icl) {
            if (!codec_encoder_) {
                result.error_msg = "Codec encoder not loaded for ICL mode";
                return result;
            }
            ref_codes = codec_encoder_->encode(ref_audio.data(), (int)ref_audio.size());
            if (ref_codes.empty()) {
                result.error_msg = "Codec encoder failed: " + codec_encoder_->get_error();
                return result;
            }
            ref_text_tokens = tokenizer_.encode(params.ref_text);
        }

        result.t_encode_ms = get_time_ms() - t_enc_start;
    }

    // 4. Build prompt embeddings
    std::vector<float> prompt_embeds;
    std::vector<float> tts_pad_embed;
    std::vector<float> trailing_text_hidden;
    int n_trailing = 0;

    if (is_icl) {
        if (speaker_embed.empty()) {
            // No valid speaker embedding — use default speaker ID in ICL prompt
            int32_t fallback_speaker_id = get_speaker_id(params.speaker);
            if (fallback_speaker_id < 0) fallback_speaker_id = 3065;  // Vivian
            std::vector<float> fallback_emb(talker_.get_config().hidden_size);
            talker_.compute_codec_embedding(fallback_speaker_id, fallback_emb.data());
            build_prompt_embeds_icl(text_tokens, ref_text_tokens, ref_codes, fallback_emb,
                                    language_id, prompt_embeds, tts_pad_embed,
                                    trailing_text_hidden, n_trailing);
        } else {
            build_prompt_embeds_icl(text_tokens, ref_text_tokens, ref_codes, speaker_embed,
                                    language_id, prompt_embeds, tts_pad_embed,
                                    trailing_text_hidden, n_trailing);
        }
    } else if (is_clone && !speaker_embed.empty()) {
        build_prompt_embeds_xvec(text_tokens, speaker_embed, language_id,
                                 prompt_embeds, tts_pad_embed);
    } else {
        // Regular TTS with named speaker (or VoiceDesign with no speaker)
        int32_t speaker_id = -1;
        if (!params.no_speaker) {
            speaker_id = get_speaker_id(params.speaker);
            if (speaker_id < 0) {
                fprintf(stderr, "Warning: unknown speaker '%s', using Vivian\n", params.speaker.c_str());
                speaker_id = 3065;
            }
        }
        build_prompt_embeds(text_tokens, speaker_id, language_id, prompt_embeds, tts_pad_embed);
    }

    // Prepend instruct embeddings if provided
    if (!params.instruct.empty()) {
        std::vector<float> instruct_embeds;
        int n_instruct = 0;
        build_instruct_embeds(params.instruct, instruct_embeds, n_instruct);

        // Prepend: instruct_embeds + prompt_embeds
        std::vector<float> combined;
        combined.reserve(instruct_embeds.size() + prompt_embeds.size());
        combined.insert(combined.end(), instruct_embeds.begin(), instruct_embeds.end());
        combined.insert(combined.end(), prompt_embeds.begin(), prompt_embeds.end());
        prompt_embeds = std::move(combined);
    }

    int n_prompt = (int)(prompt_embeds.size() / talker_.get_config().hidden_size);

    // 5. Resolve seed and generate audio codes
    tts_params gen_params = params;
    if (gen_params.seed == 0) {
        std::random_device rd;
        gen_params.seed = rd();
    }
    result.seed_used = gen_params.seed;

    int64_t t_gen_start = get_time_ms();
    std::vector<std::vector<int32_t>> multi_codes;
    generate_codes_v2(prompt_embeds, n_prompt, tts_pad_embed, gen_params, multi_codes,
                       trailing_text_hidden, n_trailing);
    result.t_generate_ms = get_time_ms() - t_gen_start;

    if (multi_codes.empty() || multi_codes[0].empty()) {
        result.error_msg = "No audio codes generated";
        return result;
    }

    result.n_tokens_generated = (int32_t)multi_codes[0].size();

    // 6. Decode to waveform
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
    result.t_load_ms = load_time_ms_;

    if (params.print_timing) {
        float audio_duration_s = (float)result.audio.size() / (float)result.sample_rate;
        int64_t t_all = result.t_load_ms + result.t_total_ms;
        float tok_s = result.t_generate_ms > 0
            ? (float)result.n_tokens_generated / ((float)result.t_generate_ms / 1000.0f) : 0.0f;
        float rtf = result.t_total_ms > 0
            ? audio_duration_s / ((float)result.t_total_ms / 1000.0f) : 0.0f;

        fprintf(stderr, "\nPerformance:\n");
        fprintf(stderr, "  Seed:            %u\n", result.seed_used);
        fprintf(stderr, "  Load model:      %4lld ms\n", (long long)result.t_load_ms);
        fprintf(stderr, "  Tokenize:        %4lld ms\n", (long long)result.t_tokenize_ms);
        if (is_clone) {
            fprintf(stderr, "  Encode ref:      %4lld ms\n", (long long)result.t_encode_ms);
        }
        fprintf(stderr, "  Generate:        %4lld ms  (%d codes, %.1f tok/s)\n",
                (long long)result.t_generate_ms, result.n_tokens_generated, tok_s);
        fprintf(stderr, "  Audio decode:    %4lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:           %4lld ms  (%lld ms excl. load)\n",
                (long long)t_all, (long long)result.t_total_ms);
        fprintf(stderr, "  Audio:           %5.1f s   (%.1fx realtime)\n",
                audio_duration_s, rtf);
    }

    return result;
}

} // namespace vocal_tts
