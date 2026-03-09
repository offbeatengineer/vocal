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

    fprintf(stderr, "vocal asr\n");
    fprintf(stderr, "  Model: %s\n", model_path);
    fprintf(stderr, "  Audio: %s\n", audio_path);
    fprintf(stderr, "  Threads: %d\n", n_threads);
    fprintf(stderr, "\n");

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
                 "    \"mel_ms\": %lld,\n"
                 "    \"encode_ms\": %lld,\n"
                 "    \"decode_ms\": %lld,\n"
                 "    \"total_ms\": %lld\n",
                 (long long)result.t_mel_ms,
                 (long long)result.t_encode_ms,
                 (long long)result.t_decode_ms,
                 (long long)result.t_total_ms);
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

} // extern "C"
