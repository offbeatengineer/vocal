#include "vocal_server.h"
#include "vocal_common.h"
#include "vocal_model.h"
#include "audio_io.h"

// ASR
#include "qwen3_asr.h"

// TTS
#include "tts.h"

#include "ggml.h"
#include "httplib.h"
#include "dr_wav.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>
#include <signal.h>
#include <dirent.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string escape_json(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
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
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

// Extract "language Xxx" prefix from ASR output
static std::string detect_language(const std::string & text) {
    const char * prefix = "language ";
    size_t plen = strlen(prefix);
    if (text.size() < plen || text.compare(0, plen, prefix) != 0) return "";
    size_t pos = plen;
    if (pos >= text.size() || !(text[pos] >= 'A' && text[pos] <= 'Z')) return "";
    ++pos;
    while (pos < text.size() && text[pos] >= 'a' && text[pos] <= 'z') ++pos;
    return text.substr(plen, pos - plen);
}

// Strip "language Xxx" prefix
static std::string extract_transcript(const std::string & text) {
    const char * prefix = "language ";
    size_t plen = strlen(prefix);
    if (text.size() < plen || text.compare(0, plen, prefix) != 0) return text;
    size_t pos = plen;
    if (pos >= text.size() || !(text[pos] >= 'A' && text[pos] <= 'Z')) return text;
    ++pos;
    while (pos < text.size() && text[pos] >= 'a' && text[pos] <= 'z') ++pos;
    while (pos < text.size() && ((unsigned char)text[pos] < 0x80) &&
           (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r'))
        ++pos;
    return text.substr(pos);
}

// Detect audio format from Content-Type header
static const char * format_from_content_type(const std::string & ct) {
    if (ct.find("wav") != std::string::npos) return "wav";
    if (ct.find("mpeg") != std::string::npos || ct.find("mp3") != std::string::npos) return "mp3";
    if (ct.find("flac") != std::string::npos) return "flac";
    return "wav"; // default
}

// Minimal JSON value extraction (flat objects only).
// Returns the raw value string for a given key, empty string if not found.
static std::string json_get(const std::string & json, const std::string & key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos += needle.size();
    // skip whitespace and colon
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t')) ++pos;
    if (pos >= json.size()) return "";
    if (json[pos] == '"') {
        // string value
        ++pos;
        std::string val;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                val += json[pos + 1];
                pos += 2;
            } else {
                val += json[pos++];
            }
        }
        return val;
    }
    // number / bool / null
    size_t start = pos;
    while (pos < json.size() && json[pos] != ',' && json[pos] != '}' &&
           json[pos] != ' ' && json[pos] != '\n' && json[pos] != '\r')
        ++pos;
    return json.substr(start, pos - start);
}

// Encode float samples to in-memory WAV via drwav
static bool encode_wav_memory(const float * samples, int n_samples, int sample_rate,
                              std::vector<uint8_t> & out) {
    drwav_data_format fmt = {};
    fmt.container = drwav_container_riff;
    fmt.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    fmt.channels = 1;
    fmt.sampleRate = (drwav_uint32)sample_rate;
    fmt.bitsPerSample = 32;

    void * wav_data = nullptr;
    size_t wav_size = 0;
    drwav wav;
    if (!drwav_init_memory_write(&wav, &wav_data, &wav_size, &fmt, NULL)) {
        return false;
    }
    drwav_write_pcm_frames(&wav, (drwav_uint64)n_samples, samples);
    drwav_uninit(&wav);

    out.assign((uint8_t *)wav_data, (uint8_t *)wav_data + wav_size);
    drwav_free(wav_data, NULL);
    return true;
}

// Suppress verbose GGML logs
static void ggml_log_quiet(enum ggml_log_level level, const char * text, void * user_data) {
    (void)user_data;
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

// ---------------------------------------------------------------------------
// ServerContext — holds loaded models + mutexes
// ---------------------------------------------------------------------------

struct ServerContext {
    // ASR
    bool has_asr = false;
    qwen3_asr::Qwen3ASR asr;
    std::mutex asr_mutex;

    // TTS
    bool has_tts = false;
    vocal_tts::TTS tts;
    std::mutex tts_mutex;
    bool tts_has_encoders = false;

    // Config
    int n_threads = 4;
};

// Global server pointer for signal handler
static httplib::Server * g_server = nullptr;

static void signal_handler(int sig) {
    (void)sig;
    if (g_server) {
        g_server->stop();
    }
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

static void handle_health(const ServerContext & ctx, const httplib::Request &, httplib::Response & res) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "{\"status\":\"ok\",\"version\":\"%s\",\"models\":{\"asr\":%s,\"tts\":%s}}",
             VOCAL_VERSION,
             ctx.has_asr ? "true" : "false",
             ctx.has_tts ? "true" : "false");
    res.set_content(buf, "application/json");
}

static void handle_asr_transcribe(ServerContext & ctx, const httplib::Request & req,
                                   httplib::Response & res) {
    if (!ctx.has_asr) {
        res.status = 503;
        res.set_content("{\"error\":\"ASR model not loaded\"}", "application/json");
        return;
    }

    if (req.body.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"Empty request body\"}", "application/json");
        return;
    }

    // Parse query params
    std::string language;
    int max_tokens = 1024;
    if (req.has_param("language")) language = req.get_param_value("language");
    if (req.has_param("max_tokens")) max_tokens = std::atoi(req.get_param_value("max_tokens").c_str());

    // Decode uploaded audio from memory
    const char * fmt = format_from_content_type(req.get_header_value("Content-Type"));

    float * samples = nullptr;
    int n_samples = 0;
    int sample_rate = 0;
    if (!vocal_audio_read_memory(req.body.data(), req.body.size(), fmt,
                                  &samples, &n_samples, &sample_rate, 16000)) {
        res.status = 400;
        res.set_content("{\"error\":\"Failed to decode audio\"}", "application/json");
        return;
    }

    // Transcribe (serialized per model)
    qwen3_asr::transcribe_params params;
    params.max_tokens = max_tokens;
    params.language = language;
    params.n_threads = ctx.n_threads;
    params.print_progress = false;
    params.print_timing = false;

    qwen3_asr::transcribe_result result;
    {
        std::lock_guard<std::mutex> lock(ctx.asr_mutex);
        result = ctx.asr.transcribe(samples, n_samples, params);
    }
    free(samples);

    if (!result.success) {
        res.status = 500;
        std::string body = "{\"error\":\"" + escape_json(result.error_msg) + "\"}";
        res.set_content(body, "application/json");
        return;
    }

    std::string lang = detect_language(result.text);
    std::string transcript = extract_transcript(result.text);

    char json[8192];
    snprintf(json, sizeof(json),
             "{\"text\":\"%s\",\"language\":\"%s\",\"timing\":{\"total_ms\":%lld}}",
             escape_json(transcript).c_str(),
             escape_json(lang).c_str(),
             (long long)result.t_total_ms);
    res.set_content(json, "application/json");
}

static void handle_tts_synthesize(ServerContext & ctx, const httplib::Request & req,
                                   httplib::Response & res) {
    if (!ctx.has_tts) {
        res.status = 503;
        res.set_content("{\"error\":\"TTS model not loaded\"}", "application/json");
        return;
    }

    if (req.body.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"Empty request body\"}", "application/json");
        return;
    }

    // Parse JSON body
    std::string text = json_get(req.body, "text");
    if (text.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"Missing 'text' field\"}", "application/json");
        return;
    }

    vocal_tts::tts_params params;
    params.n_threads = ctx.n_threads;
    params.print_timing = false;

    std::string val;
    val = json_get(req.body, "voice_profile");
    if (!val.empty()) {
        // Resolve voice path
        char voice_path[4096];
        if (vocal_voice_path(val.c_str(), voice_path, sizeof(voice_path))) {
            params.voice_profile = voice_path;
        }
    }
    val = json_get(req.body, "speed");
    if (!val.empty()) params.speed = (float)std::atof(val.c_str());
    val = json_get(req.body, "temperature");
    if (!val.empty()) params.temperature = (float)std::atof(val.c_str());
    val = json_get(req.body, "top_k");
    if (!val.empty()) params.top_k = std::atoi(val.c_str());
    val = json_get(req.body, "rep_penalty");
    if (!val.empty()) params.rep_penalty = (float)std::atof(val.c_str());
    val = json_get(req.body, "seed");
    if (!val.empty()) params.seed = (uint32_t)std::atol(val.c_str());

    int64_t t0 = now_ms();
    vocal_tts::tts_result result;
    {
        std::lock_guard<std::mutex> lock(ctx.tts_mutex);
        result = ctx.tts.synthesize(text, params);
    }
    int64_t total_ms = now_ms() - t0;

    if (!result.success) {
        res.status = 500;
        std::string body = "{\"error\":\"" + escape_json(result.error_msg) + "\"}";
        res.set_content(body, "application/json");
        return;
    }

    // Encode WAV in memory
    std::vector<uint8_t> wav_data;
    if (!encode_wav_memory(result.audio.data(), (int)result.audio.size(),
                           result.sample_rate, wav_data)) {
        res.status = 500;
        res.set_content("{\"error\":\"Failed to encode WAV\"}", "application/json");
        return;
    }

    res.set_header("X-Vocal-Total-Ms", std::to_string(total_ms));
    res.set_header("X-Vocal-Generate-Ms", std::to_string(result.t_generate_ms));
    res.set_header("X-Vocal-Decode-Ms", std::to_string(result.t_decode_ms));
    res.set_header("X-Vocal-Tokens", std::to_string(result.n_tokens_generated));
    res.set_content(std::string((char *)wav_data.data(), wav_data.size()), "audio/wav");
}

static void handle_tts_clone(ServerContext & ctx, const httplib::Request & req,
                              httplib::Response & res) {
    if (!ctx.has_tts) {
        res.status = 503;
        res.set_content("{\"error\":\"TTS model not loaded\"}", "application/json");
        return;
    }
    if (!ctx.tts_has_encoders) {
        res.status = 503;
        res.set_content("{\"error\":\"Encoder models not loaded (clone not available)\"}", "application/json");
        return;
    }

    // Multipart form data
    if (!req.has_file("audio")) {
        res.status = 400;
        res.set_content("{\"error\":\"Missing 'audio' file in multipart form\"}", "application/json");
        return;
    }

    auto audio_file = req.get_file_value("audio");
    std::string text;
    std::string ref_text;

    if (req.has_file("text")) text = req.get_file_value("text").content;
    if (req.has_file("ref_text")) ref_text = req.get_file_value("ref_text").content;

    if (text.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"Missing 'text' field\"}", "application/json");
        return;
    }

    // Write uploaded audio to a temp file (TTS expects a file path for ref_audio)
    char tmp_path[] = "/tmp/vocal_clone_XXXXXX.wav";
    int fd = mkstemps(tmp_path, 4);
    if (fd < 0) {
        res.status = 500;
        res.set_content("{\"error\":\"Failed to create temp file\"}", "application/json");
        return;
    }
    write(fd, audio_file.content.data(), audio_file.content.size());
    close(fd);

    vocal_tts::tts_params params;
    params.n_threads = ctx.n_threads;
    params.print_timing = false;
    params.ref_audio_path = tmp_path;
    if (!ref_text.empty()) params.ref_text = ref_text;

    // Parse optional params from form fields
    std::string val;
    if (req.has_file("speed")) {
        val = req.get_file_value("speed").content;
        params.speed = (float)std::atof(val.c_str());
    }
    if (req.has_file("temperature")) {
        val = req.get_file_value("temperature").content;
        params.temperature = (float)std::atof(val.c_str());
    }
    if (req.has_file("top_k")) {
        val = req.get_file_value("top_k").content;
        params.top_k = std::atoi(val.c_str());
    }
    if (req.has_file("rep_penalty")) {
        val = req.get_file_value("rep_penalty").content;
        params.rep_penalty = (float)std::atof(val.c_str());
    }
    if (req.has_file("seed")) {
        val = req.get_file_value("seed").content;
        params.seed = (uint32_t)std::atol(val.c_str());
    }

    int64_t t0 = now_ms();
    vocal_tts::tts_result result;
    {
        std::lock_guard<std::mutex> lock(ctx.tts_mutex);
        result = ctx.tts.synthesize(text, params);
    }
    int64_t total_ms = now_ms() - t0;

    unlink(tmp_path); // clean up temp file

    if (!result.success) {
        res.status = 500;
        std::string body = "{\"error\":\"" + escape_json(result.error_msg) + "\"}";
        res.set_content(body, "application/json");
        return;
    }

    std::vector<uint8_t> wav_data;
    if (!encode_wav_memory(result.audio.data(), (int)result.audio.size(),
                           result.sample_rate, wav_data)) {
        res.status = 500;
        res.set_content("{\"error\":\"Failed to encode WAV\"}", "application/json");
        return;
    }

    res.set_header("X-Vocal-Total-Ms", std::to_string(total_ms));
    res.set_header("X-Vocal-Generate-Ms", std::to_string(result.t_generate_ms));
    res.set_header("X-Vocal-Decode-Ms", std::to_string(result.t_decode_ms));
    res.set_header("X-Vocal-Tokens", std::to_string(result.n_tokens_generated));
    res.set_content(std::string((char *)wav_data.data(), wav_data.size()), "audio/wav");
}

static void handle_voices(const httplib::Request &, httplib::Response & res) {
    const char * dir = vocal_voices_dir();
    std::string json = "{\"voices\":[";

    if (dir) {
        DIR * d = opendir(dir);
        if (d) {
            struct dirent * entry;
            bool first = true;
            while ((entry = readdir(d)) != NULL) {
                if (entry->d_name[0] == '.') continue;
                const char * ext = strrchr(entry->d_name, '.');
                if (!ext || strcmp(ext, ".voice") != 0) continue;

                if (!first) json += ",";
                first = false;

                // Name without extension
                std::string name(entry->d_name, ext - entry->d_name);
                json += "\"" + escape_json(name) + "\"";
            }
            closedir(d);
        }
    }

    json += "]}";
    res.set_content(json, "application/json");
}

// ---------------------------------------------------------------------------
// Main entry
// ---------------------------------------------------------------------------

extern "C" int vocal_serve_run(const struct vocal_serve_params * params) {
    ggml_log_set(ggml_log_quiet, nullptr);

    if (!params->load_asr && !params->load_tts) {
        fprintf(stderr, "error: at least one of --asr or --tts required\n");
        return 1;
    }

    ServerContext ctx;
    ctx.n_threads = params->n_threads;

    const char * model_dir = params->model_dir;

    // --- Load ASR ---
    if (params->load_asr) {
        const char * asr_model_name = params->use_large
            ? VOCAL_ASR_MODEL_LARGE_NAME : VOCAL_ASR_MODEL_NAME;

        char model_path[4096];
        if (!vocal_model_path(asr_model_name, model_dir, model_path, sizeof(model_path))) {
            fprintf(stderr, "error: could not resolve ASR model path\n");
            return 2;
        }
        if (!vocal_model_exists(asr_model_name, model_dir)) {
            fprintf(stderr, "error: ASR model not found at %s\n", model_path);
            fprintf(stderr, "Run: vocal download %s\n", params->use_large ? "asr-large" : "asr");
            return 2;
        }

        fprintf(stderr, "Loading ASR model: %s\n", model_path);
        if (!ctx.asr.load_model(model_path)) {
            fprintf(stderr, "error: failed to load ASR model: %s\n",
                    ctx.asr.get_error().c_str());
            return 2;
        }
        ctx.has_asr = true;
        fprintf(stderr, "ASR model loaded.\n");
    }

    // --- Load TTS ---
    if (params->load_tts) {
        const char * tts_model_name = params->use_large
            ? VOCAL_TTS_MODEL_LARGE_NAME : VOCAL_TTS_MODEL_NAME;
        const char * spk_encoder_name = params->use_large
            ? VOCAL_TTS_SPK_ENCODER_LARGE_NAME : VOCAL_TTS_SPK_ENCODER_NAME;

        char model_path[4096], tokenizer_path[4096], decoder_path[4096];
        char encoder_path[4096], spk_encoder_path[4096];

        if (!vocal_model_path(tts_model_name, model_dir, model_path, sizeof(model_path)) ||
            !vocal_model_exists(tts_model_name, model_dir)) {
            fprintf(stderr, "error: TTS model not found. Run: vocal download tts\n");
            return 2;
        }
        vocal_model_path(VOCAL_TTS_TOKENIZER_NAME, model_dir, tokenizer_path, sizeof(tokenizer_path));
        vocal_model_path(VOCAL_TTS_DECODER_NAME, model_dir, decoder_path, sizeof(decoder_path));

        if (!vocal_model_exists(VOCAL_TTS_DECODER_NAME, model_dir)) {
            fprintf(stderr, "error: TTS decoder not found. Run: vocal download tts\n");
            return 2;
        }

        fprintf(stderr, "Loading TTS model: %s\n", model_path);
        if (!ctx.tts.load(model_path, tokenizer_path, decoder_path)) {
            fprintf(stderr, "error: failed to load TTS model: %s\n",
                    ctx.tts.get_error().c_str());
            return 2;
        }
        ctx.has_tts = true;
        fprintf(stderr, "TTS model loaded.\n");

        // Try to load encoders for clone support (optional)
        vocal_model_path(VOCAL_TTS_ENCODER_NAME, model_dir, encoder_path, sizeof(encoder_path));
        vocal_model_path(spk_encoder_name, model_dir, spk_encoder_path, sizeof(spk_encoder_path));

        if (vocal_model_exists(VOCAL_TTS_ENCODER_NAME, model_dir) &&
            vocal_model_exists(spk_encoder_name, model_dir)) {
            if (ctx.tts.load_encoders(encoder_path, spk_encoder_path)) {
                ctx.tts_has_encoders = true;
                fprintf(stderr, "TTS encoders loaded (clone available).\n");
            } else {
                fprintf(stderr, "warning: failed to load encoders, clone endpoint disabled\n");
            }
        } else {
            fprintf(stderr, "Encoder models not found, clone endpoint disabled.\n");
        }
    }

    // --- Set up HTTP server ---
    httplib::Server svr;
    g_server = &svr;

    // Signal handler for graceful shutdown
    struct sigaction sa = {};
    sa.sa_handler = signal_handler;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    // Routes
    svr.Get("/health", [&ctx](const httplib::Request & req, httplib::Response & res) {
        handle_health(ctx, req, res);
    });

    svr.Post("/v1/asr/transcribe", [&ctx](const httplib::Request & req, httplib::Response & res) {
        handle_asr_transcribe(ctx, req, res);
    });

    svr.Post("/v1/tts/synthesize", [&ctx](const httplib::Request & req, httplib::Response & res) {
        handle_tts_synthesize(ctx, req, res);
    });

    svr.Post("/v1/tts/clone", [&ctx](const httplib::Request & req, httplib::Response & res) {
        handle_tts_clone(ctx, req, res);
    });

    svr.Get("/v1/voices", [](const httplib::Request & req, httplib::Response & res) {
        handle_voices(req, res);
    });

    const char * host = params->host ? params->host : "127.0.0.1";
    int port = params->port > 0 ? params->port : 8080;

    fprintf(stderr, "\nvocal server listening on http://%s:%d\n", host, port);
    fprintf(stderr, "  ASR:   %s\n", ctx.has_asr ? "enabled" : "disabled");
    fprintf(stderr, "  TTS:   %s\n", ctx.has_tts ? "enabled" : "disabled");
    fprintf(stderr, "  Clone: %s\n", ctx.tts_has_encoders ? "enabled" : "disabled");
    fprintf(stderr, "\nPress Ctrl+C to stop.\n\n");

    if (!svr.listen(host, port)) {
        // listen() returns false if it was stopped by signal or bind failure
        // If g_server was cleared by signal, it's a clean shutdown
        if (g_server == nullptr) {
            fprintf(stderr, "\nServer stopped.\n");
            return 0;
        }
        fprintf(stderr, "error: failed to start server on %s:%d\n", host, port);
        return 1;
    }

    fprintf(stderr, "\nServer stopped.\n");
    g_server = nullptr;
    return 0;
}
