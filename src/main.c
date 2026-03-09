#include "vocal.h"
#include "audio/audio_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Forward declarations for C++ ASR functions
#ifdef __cplusplus
extern "C" {
#endif

int vocal_asr_run(const char * model_path, const char * audio_path,
                  const char * output_path, const char * language,
                  int n_threads, int max_tokens, bool json_output,
                  bool print_timing);

int vocal_tts_run(const char * model_path, const char * tokenizer_path,
                  const char * decoder_path, const char * text,
                  const char * output_path,
                  int n_threads, float speed, bool print_timing);

#ifdef __cplusplus
}
#endif

// --- Usage ---

static void print_version(void) {
    printf("vocal %s\n", VOCAL_VERSION);
}

static void print_usage(void) {
    fprintf(stderr,
        "vocal %s - Local voice toolkit\n"
        "\n"
        "Usage: vocal <command> [options]\n"
        "\n"
        "Commands:\n"
        "  asr          Transcribe audio to text\n"
        "  tts          Synthesize speech from text\n"
        "  clone        Clone a voice from audio (coming soon)\n"
        "  download     Download models\n"
        "  models       List downloaded models\n"
        "  version      Print version\n"
        "\n"
        "Run 'vocal <command> --help' for command-specific help.\n",
        VOCAL_VERSION);
}

// --- ASR subcommand ---

static void print_asr_usage(void) {
    fprintf(stderr,
        "Usage: vocal asr [options]\n"
        "\n"
        "Transcribe audio to text using Qwen3-ASR.\n"
        "\n"
        "Options:\n"
        "  -f, --file <path>        Audio file (WAV, 16kHz mono) [required]\n"
        "  -o, --output <path>      Output file (default: stdout)\n"
        "  -m, --model <path>       Model file path (overrides default)\n"
        "  -l, --language <code>    Language hint\n"
        "  -t, --threads <n>        Thread count (default: 4)\n"
        "  --max-tokens <n>         Max tokens to generate (default: 1024)\n"
        "  --json                   Output JSON with timestamps\n"
        "  --no-timing              Don't print timing info\n"
        "  --model-dir <path>       Models directory override\n"
        "  -h, --help               Show this help\n");
}

static int cmd_asr(int argc, char ** argv) {
    const char * audio_path = NULL;
    const char * output_path = NULL;
    const char * model_override = NULL;
    const char * model_dir = NULL;
    const char * language = "";
    int n_threads = 4;
    int max_tokens = 1024;
    bool json_output = false;
    bool print_timing = true;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--file") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: -f requires argument\n"); return VOCAL_ERR_ARGS; }
            audio_path = argv[i];
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: -o requires argument\n"); return VOCAL_ERR_ARGS; }
            output_path = argv[i];
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: -m requires argument\n"); return VOCAL_ERR_ARGS; }
            model_override = argv[i];
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--language") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: -l requires argument\n"); return VOCAL_ERR_ARGS; }
            language = argv[i];
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: -t requires argument\n"); return VOCAL_ERR_ARGS; }
            n_threads = atoi(argv[i]);
        } else if (strcmp(argv[i], "--max-tokens") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --max-tokens requires argument\n"); return VOCAL_ERR_ARGS; }
            max_tokens = atoi(argv[i]);
        } else if (strcmp(argv[i], "--json") == 0) {
            json_output = true;
        } else if (strcmp(argv[i], "--no-timing") == 0) {
            print_timing = false;
        } else if (strcmp(argv[i], "--model-dir") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --model-dir requires argument\n"); return VOCAL_ERR_ARGS; }
            model_dir = argv[i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_asr_usage();
            return VOCAL_OK;
        } else {
            fprintf(stderr, "error: unknown option: %s\n", argv[i]);
            print_asr_usage();
            return VOCAL_ERR_ARGS;
        }
    }

    if (!audio_path) {
        fprintf(stderr, "error: audio file required (-f)\n\n");
        print_asr_usage();
        return VOCAL_ERR_ARGS;
    }

    // Resolve model path
    char model_path[4096];
    if (model_override) {
        snprintf(model_path, sizeof(model_path), "%s", model_override);
    } else {
        if (!vocal_model_path(VOCAL_ASR_MODEL_NAME, model_dir, model_path, sizeof(model_path))) {
            fprintf(stderr, "error: could not resolve model path\n");
            return VOCAL_ERR_MODEL;
        }
    }

    if (!vocal_model_exists(model_override ? NULL : VOCAL_ASR_MODEL_NAME,
                            model_override ? NULL : model_dir)) {
        // Check model_override path directly
        if (model_override) {
            // Direct path was given, just check it
        } else {
            fprintf(stderr, "error: ASR model not found at %s\n", model_path);
            fprintf(stderr, "Run: vocal download asr\n");
            return VOCAL_ERR_MODEL;
        }
    }

    return vocal_asr_run(model_path, audio_path, output_path, language,
                         n_threads, max_tokens, json_output, print_timing);
}

// --- TTS subcommand ---

static void print_tts_usage(void) {
    fprintf(stderr,
        "Usage: vocal tts [options]\n"
        "\n"
        "Synthesize speech from text using Qwen3-TTS.\n"
        "\n"
        "Options:\n"
        "  -t, --text <text>        Text to synthesize [required]\n"
        "  --stdin                  Read text from stdin\n"
        "  -o, --output <path>      Output WAV file [required]\n"
        "  -m, --model <path>       Model file path\n"
        "  --tokenizer <path>       Tokenizer file path\n"
        "  --speed <float>          Speed factor (default: 1.0)\n"
        "  --threads <n>            Thread count (default: 4)\n"
        "  --no-timing              Don't print timing info\n"
        "  --model-dir <path>       Models directory override\n"
        "  -h, --help               Show this help\n");
}

static int cmd_tts(int argc, char ** argv) {
    const char * text = NULL;
    const char * output_path = NULL;
    const char * model_override = NULL;
    const char * tokenizer_override = NULL;
    const char * model_dir = NULL;
    int n_threads = 4;
    float speed = 1.0f;
    bool print_timing = true;
    bool use_stdin = false;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--text") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: -t requires argument\n"); return VOCAL_ERR_ARGS; }
            text = argv[i];
        } else if (strcmp(argv[i], "--stdin") == 0) {
            use_stdin = true;
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: -o requires argument\n"); return VOCAL_ERR_ARGS; }
            output_path = argv[i];
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: -m requires argument\n"); return VOCAL_ERR_ARGS; }
            model_override = argv[i];
        } else if (strcmp(argv[i], "--tokenizer") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --tokenizer requires argument\n"); return VOCAL_ERR_ARGS; }
            tokenizer_override = argv[i];
        } else if (strcmp(argv[i], "--speed") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --speed requires argument\n"); return VOCAL_ERR_ARGS; }
            speed = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--threads") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --threads requires argument\n"); return VOCAL_ERR_ARGS; }
            n_threads = atoi(argv[i]);
        } else if (strcmp(argv[i], "--no-timing") == 0) {
            print_timing = false;
        } else if (strcmp(argv[i], "--model-dir") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --model-dir requires argument\n"); return VOCAL_ERR_ARGS; }
            model_dir = argv[i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_tts_usage();
            return VOCAL_OK;
        } else {
            fprintf(stderr, "error: unknown option: %s\n", argv[i]);
            print_tts_usage();
            return VOCAL_ERR_ARGS;
        }
    }

    // Read from stdin if requested
    static char stdin_buf[65536];
    if (use_stdin) {
        size_t total = 0;
        size_t n;
        while ((n = fread(stdin_buf + total, 1, sizeof(stdin_buf) - total - 1, stdin)) > 0) {
            total += n;
        }
        stdin_buf[total] = '\0';
        text = stdin_buf;
    }

    if (!text || !text[0]) {
        fprintf(stderr, "error: text required (-t or --stdin)\n\n");
        print_tts_usage();
        return VOCAL_ERR_ARGS;
    }

    if (!output_path) {
        fprintf(stderr, "error: output file required (-o)\n\n");
        print_tts_usage();
        return VOCAL_ERR_ARGS;
    }

    // Resolve model path
    char model_path[4096];
    if (model_override) {
        snprintf(model_path, sizeof(model_path), "%s", model_override);
    } else {
        if (!vocal_model_path(VOCAL_TTS_MODEL_NAME, model_dir, model_path, sizeof(model_path))) {
            fprintf(stderr, "error: could not resolve model path\n");
            return VOCAL_ERR_MODEL;
        }
        if (!vocal_model_exists(VOCAL_TTS_MODEL_NAME, model_dir)) {
            fprintf(stderr, "error: TTS model not found at %s\n", model_path);
            fprintf(stderr, "Run: vocal download tts\n");
            return VOCAL_ERR_MODEL;
        }
    }

    // Resolve tokenizer path
    char tokenizer_path[4096];
    if (tokenizer_override) {
        snprintf(tokenizer_path, sizeof(tokenizer_path), "%s", tokenizer_override);
    } else {
        if (!vocal_model_path(VOCAL_TTS_TOKENIZER_NAME, model_dir, tokenizer_path, sizeof(tokenizer_path))) {
            fprintf(stderr, "error: could not resolve tokenizer path\n");
            return VOCAL_ERR_MODEL;
        }
    }

    // Resolve decoder path
    char decoder_path[4096];
    if (!vocal_model_path(VOCAL_TTS_DECODER_NAME, model_dir, decoder_path, sizeof(decoder_path))) {
        fprintf(stderr, "error: could not resolve decoder path\n");
        return VOCAL_ERR_MODEL;
    }
    if (!vocal_model_exists(VOCAL_TTS_DECODER_NAME, model_dir)) {
        fprintf(stderr, "error: TTS decoder not found at %s\n", decoder_path);
        fprintf(stderr, "Run: vocal download tts\n");
        return VOCAL_ERR_MODEL;
    }

    return vocal_tts_run(model_path, tokenizer_path, decoder_path, text, output_path,
                         n_threads, speed, print_timing);
}

// --- Download subcommand ---

static void print_download_usage(void) {
    fprintf(stderr,
        "Usage: vocal download <model> [options]\n"
        "\n"
        "Download model files.\n"
        "\n"
        "Models:\n"
        "  asr          Qwen3-ASR 0.6B\n"
        "  tts          Qwen3-TTS 0.6B + tokenizer\n"
        "\n"
        "Options:\n"
        "  --model-dir <path>   Override model storage directory\n"
        "  -h, --help           Show this help\n");
}

static int cmd_download(int argc, char ** argv) {
    const char * model_dir = NULL;
    const char * model_type = NULL;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--model-dir") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --model-dir requires argument\n"); return VOCAL_ERR_ARGS; }
            model_dir = argv[i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_download_usage();
            return VOCAL_OK;
        } else if (argv[i][0] != '-') {
            model_type = argv[i];
        } else {
            fprintf(stderr, "error: unknown option: %s\n", argv[i]);
            return VOCAL_ERR_ARGS;
        }
    }

    if (!model_type) {
        fprintf(stderr, "error: specify model to download (e.g., 'vocal download asr')\n\n");
        print_download_usage();
        return VOCAL_ERR_ARGS;
    }

    if (strcmp(model_type, "asr") == 0) {
        return vocal_model_download(VOCAL_ASR_MODEL_URL, VOCAL_ASR_MODEL_NAME, model_dir);
    } else if (strcmp(model_type, "tts") == 0) {
        int ret = vocal_model_download(VOCAL_TTS_MODEL_URL, VOCAL_TTS_MODEL_NAME, model_dir);
        if (ret != 0) return ret;
        ret = vocal_model_download(VOCAL_TTS_TOKENIZER_URL, VOCAL_TTS_TOKENIZER_NAME, model_dir);
        if (ret != 0) return ret;
        return vocal_model_download(VOCAL_TTS_DECODER_URL, VOCAL_TTS_DECODER_NAME, model_dir);
    } else {
        fprintf(stderr, "error: unknown model type: %s\n", model_type);
        print_download_usage();
        return VOCAL_ERR_ARGS;
    }
}

// --- Main ---

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage();
        return VOCAL_ERR_ARGS;
    }

    const char * cmd = argv[1];

    if (strcmp(cmd, "asr") == 0) {
        return cmd_asr(argc - 2, argv + 2);
    } else if (strcmp(cmd, "download") == 0) {
        return cmd_download(argc - 2, argv + 2);
    } else if (strcmp(cmd, "models") == 0) {
        const char * model_dir = NULL;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
                model_dir = argv[++i];
            }
        }
        vocal_models_list(model_dir);
        return VOCAL_OK;
    } else if (strcmp(cmd, "version") == 0 || strcmp(cmd, "--version") == 0 || strcmp(cmd, "-v") == 0) {
        print_version();
        return VOCAL_OK;
    } else if (strcmp(cmd, "-h") == 0 || strcmp(cmd, "--help") == 0 || strcmp(cmd, "help") == 0) {
        print_usage();
        return VOCAL_OK;
    } else if (strcmp(cmd, "tts") == 0) {
        return cmd_tts(argc - 2, argv + 2);
    } else if (strcmp(cmd, "clone") == 0) {
        fprintf(stderr, "'vocal clone' is not yet implemented.\n");
        fprintf(stderr, "Coming in a future release.\n");
        return VOCAL_ERR_ARGS;
    } else {
        fprintf(stderr, "error: unknown command '%s'\n\n", cmd);
        print_usage();
        return VOCAL_ERR_ARGS;
    }
}
