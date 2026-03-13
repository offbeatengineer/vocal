#include "qwen3_asr.h"
#include "timing.h"

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <cctype>
#include <fstream>
#include <string>

extern "C" {

static void ggml_log_quiet(enum ggml_log_level level, const char * text, void * user_data) {
    (void)user_data;
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

// Extract detected language from ASR output (e.g. "language English..." -> "English")
static std::string detect_language(const std::string & text) {
    const std::string prefix = "language ";
    if (text.size() < prefix.size() || text.compare(0, prefix.size(), prefix) != 0) {
        return "";
    }
    size_t pos = prefix.size();
    if (pos >= text.size() || !std::isupper(static_cast<unsigned char>(text[pos]))) {
        return "";
    }
    ++pos;
    while (pos < text.size() && std::islower(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    return text.substr(prefix.size(), pos - prefix.size());
}

// Strip "language Xxx" prefix from transcript
static std::string extract_transcript(const std::string & text) {
    const std::string prefix = "language ";
    if (text.size() < prefix.size() || text.compare(0, prefix.size(), prefix) != 0) {
        return text;
    }
    size_t pos = prefix.size();
    if (pos >= text.size() || !std::isupper(static_cast<unsigned char>(text[pos]))) {
        return text;
    }
    ++pos;
    while (pos < text.size() && std::islower(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    // Skip whitespace after language tag
    while (pos < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[pos]);
        if (c >= 0x80 || !std::isspace(c)) break;
        ++pos;
    }
    return text.substr(pos);
}

static std::string escape_json(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 10);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static void write_output(const std::string & text, const char * output_path) {
    if (output_path) {
        std::ofstream out(output_path);
        if (!out) {
            fprintf(stderr, "error: cannot write to %s\n", output_path);
            return;
        }
        out << text << "\n";
        fprintf(stderr, "Output written to: %s\n", output_path);
    } else {
        printf("%s\n", text.c_str());
    }
}

int vocal_asr_run(const char * model_path, const char * audio_path,
                  const char * output_path, const char * language,
                  int n_threads, int max_tokens, bool json_output,
                  bool print_timing) {
    ggml_log_set(ggml_log_quiet, nullptr);

    qwen3_asr::Qwen3ASR asr;

    if (!asr.load_model(model_path)) {
        fprintf(stderr, "error: %s\n", asr.get_error().c_str());
        return 2;
    }

    qwen3_asr::transcribe_params params;
    params.max_tokens = max_tokens;
    params.language = language ? language : "";
    params.n_threads = n_threads;
    params.print_progress = false;
    params.print_timing = print_timing;

    auto result = asr.transcribe(audio_path, params);

    if (!result.success) {
        fprintf(stderr, "error: %s\n", result.error_msg.c_str());
        return 5;
    }

    // Strip "language Xxx" prefix
    std::string lang = detect_language(result.text);
    std::string transcript = extract_transcript(result.text);

    if (json_output) {
        char buf[256];
        std::string json = "{\n";
        json += "  \"text\": \"" + escape_json(transcript) + "\",\n";
        if (!lang.empty()) {
            json += "  \"language\": \"" + escape_json(lang) + "\",\n";
        }
        json += "  \"timing\": {\n";
        snprintf(buf, sizeof(buf),
                 "    \"load_ms\": %lld,\n"
                 "    \"mel_ms\": %lld,\n"
                 "    \"encode_ms\": %lld,\n"
                 "    \"decode_ms\": %lld,\n"
                 "    \"total_ms\": %lld,\n"
                 "    \"audio_duration_s\": %.3f\n",
                 (long long)result.t_load_ms,
                 (long long)result.t_mel_ms,
                 (long long)result.t_encode_ms,
                 (long long)result.t_decode_ms,
                 (long long)result.t_total_ms,
                 result.audio_duration_s);
        json += buf;
        json += "  },\n";
        snprintf(buf, sizeof(buf), "  \"tokens\": %zu\n", result.tokens.size());
        json += buf;
        json += "}";
        write_output(json, output_path);
    } else {
        write_output(transcript, output_path);
    }

    return 0;
}

int vocal_asr_align(const char * model_path, const char * aligner_path,
                    const char * audio_path, const char * transcript,
                    const char * output_path, const char * language,
                    int n_threads, bool json_output, bool print_timing) {
    ggml_log_set(ggml_log_quiet, nullptr);

    // Load audio
    float * raw_samples = nullptr;
    int n_samples = 0;
    int sample_rate = 0;

    // vocal_audio_read is declared in audio_io.h
    extern bool vocal_audio_read(const char *, float **, int *, int *, int);

    if (!vocal_audio_read(audio_path, &raw_samples, &n_samples, &sample_rate, 16000)) {
        fprintf(stderr, "error: failed to load audio: %s\n", audio_path);
        return 3;
    }

    // If no transcript provided, transcribe first
    std::string final_transcript;
    if (transcript && transcript[0]) {
        final_transcript = transcript;
    } else if (model_path) {
        // Use ASR to get transcript
        qwen3_asr::Qwen3ASR asr;
        if (!asr.load_model(model_path)) {
            fprintf(stderr, "error: %s\n", asr.get_error().c_str());
            free(raw_samples);
            return 2;
        }
        qwen3_asr::transcribe_params params;
        params.language = language ? language : "";
        params.n_threads = n_threads;
        params.print_timing = print_timing;

        auto result = asr.transcribe(raw_samples, n_samples, params);
        if (!result.success) {
            fprintf(stderr, "error: transcription failed: %s\n", result.error_msg.c_str());
            free(raw_samples);
            return 5;
        }
        final_transcript = extract_transcript(result.text);
        fprintf(stderr, "  Transcript: %s\n\n", final_transcript.c_str());
    } else {
        fprintf(stderr, "error: either --text or ASR model required for alignment\n");
        free(raw_samples);
        return 1;
    }

    // Load aligner and run alignment
    qwen3_asr::Qwen3ASR aligner_asr;

    // Load a dummy ASR model? No — we only need the aligner components.
    // Load aligner model (it contains both audio encoder and text decoder)
    if (!aligner_asr.load_model(aligner_path)) {
        // load_model loads encoder + decoder from the same file — this gives us mel filters too
        fprintf(stderr, "error: %s\n", aligner_asr.get_error().c_str());
        free(raw_samples);
        return 2;
    }

    if (!aligner_asr.load_aligner(aligner_path)) {
        fprintf(stderr, "error: %s\n", aligner_asr.get_error().c_str());
        free(raw_samples);
        return 2;
    }

    qwen3_asr::transcribe_params align_params;
    align_params.n_threads = n_threads;
    align_params.print_timing = print_timing;

    auto align_result = aligner_asr.align(raw_samples, n_samples, final_transcript, align_params);
    free(raw_samples);

    if (!align_result.success) {
        fprintf(stderr, "error: alignment failed: %s\n", align_result.error_msg.c_str());
        return 5;
    }

    // Output results
    if (json_output) {
        std::string json = "{\n";
        json += "  \"text\": \"" + escape_json(final_transcript) + "\",\n";
        json += "  \"words\": [\n";
        for (size_t i = 0; i < align_result.words.size(); ++i) {
            const auto & w = align_result.words[i];
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "    {\"word\": \"%s\", \"start\": %.3f, \"end\": %.3f}",
                     escape_json(w.word).c_str(), w.start_s, w.end_s);
            json += buf;
            if (i + 1 < align_result.words.size()) json += ",";
            json += "\n";
        }
        json += "  ],\n";
        {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "  \"timing\": {\n"
                     "    \"load_ms\": %lld,\n"
                     "    \"mel_ms\": %lld,\n"
                     "    \"encode_ms\": %lld,\n"
                     "    \"align_ms\": %lld,\n"
                     "    \"total_ms\": %lld,\n"
                     "    \"audio_duration_s\": %.3f\n"
                     "  }\n",
                     (long long)align_result.t_load_ms,
                     (long long)align_result.t_mel_ms,
                     (long long)align_result.t_encode_ms,
                     (long long)align_result.t_align_ms,
                     (long long)align_result.t_total_ms,
                     align_result.audio_duration_s);
            json += buf;
        }
        json += "}";
        write_output(json, output_path);
    } else {
        std::string out;
        for (const auto & w : align_result.words) {
            char buf[256];
            snprintf(buf, sizeof(buf), "[%.3f - %.3f] %s\n",
                     w.start_s, w.end_s, w.word.c_str());
            out += buf;
        }
        // Remove trailing newline for write_output (it adds one)
        if (!out.empty() && out.back() == '\n') out.pop_back();
        write_output(out, output_path);
    }

    return 0;
}

} // extern "C"
