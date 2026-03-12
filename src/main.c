#include "vocal.h"
#include "audio/audio_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Forward declarations for C++ ASR/TTS functions
#ifdef __cplusplus
extern "C" {
#endif

int vocal_asr_run(const char * model_path, const char * audio_path,
                  const char * output_path, const char * language,
                  int n_threads, int max_tokens, bool json_output,
                  bool print_timing);

struct vocal_sampling_params {
    float temperature;
    int top_k;
    float rep_penalty;
    unsigned int seed;  // 0 = random
};

int vocal_tts_run(const char * model_path, const char * tokenizer_path,
                  const char * decoder_path, const char * text,
                  const char * output_path,
                  int n_threads, float speed, bool print_timing,
                  struct vocal_sampling_params sampling);

int vocal_tts_clone(const char * model_path, const char * tokenizer_path,
                    const char * decoder_path, const char * encoder_path,
                    const char * spk_encoder_path,
                    const char * ref_audio_path, const char * ref_text,
                    const char * text, const char * output_path,
                    int n_threads, float speed, bool print_timing,
                    struct vocal_sampling_params sampling);

int vocal_tts_save_voice(const char * model_path, const char * tokenizer_path,
                          const char * decoder_path, const char * encoder_path,
                          const char * spk_encoder_path,
                          const char * ref_audio_path, const char * ref_text,
                          const char * save_path);

int vocal_tts_with_voice(const char * model_path, const char * tokenizer_path,
                          const char * decoder_path, const char * voice_path,
                          const char * text, const char * output_path,
                          int n_threads, float speed, bool print_timing,
                          struct vocal_sampling_params sampling);

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
        "  clone        Clone a voice from reference audio\n"
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
        "  -f, --file <path>        Audio file (WAV/MP3/FLAC, any sample rate) [required]\n"
        "  -o, --output <path>      Output file (default: stdout)\n"
        "  -m, --model <path>       Model file path (overrides default)\n"
        "  --large                  Use 1.7B model (default: 0.6B)\n"
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
    bool use_large = false;

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
        } else if (strcmp(argv[i], "--large") == 0) {
            use_large = true;
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

    const char * asr_model_name = use_large ? VOCAL_ASR_MODEL_LARGE_NAME : VOCAL_ASR_MODEL_NAME;

    // Resolve model path
    char model_path[4096];
    if (model_override) {
        snprintf(model_path, sizeof(model_path), "%s", model_override);
    } else {
        if (!vocal_model_path(asr_model_name, model_dir, model_path, sizeof(model_path))) {
            fprintf(stderr, "error: could not resolve model path\n");
            return VOCAL_ERR_MODEL;
        }
    }

    if (!vocal_model_exists(model_override ? NULL : asr_model_name,
                            model_override ? NULL : model_dir)) {
        // Check model_override path directly
        if (model_override) {
            // Direct path was given, just check it
        } else {
            fprintf(stderr, "error: ASR model not found at %s\n", model_path);
            fprintf(stderr, "Run: vocal download %s\n", use_large ? "asr-large" : "asr");
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
        "  --voice <name>           Use a saved voice profile\n"
        "  -m, --model <path>       Model file path\n"
        "  --large                  Use 1.7B model (default: 0.6B)\n"
        "  --tokenizer <path>       Tokenizer file path\n"
        "  --speed <float>          Speed factor (default: 1.0)\n"
        "  --threads <n>            Thread count (default: 4)\n"
        "  --temperature <float>    Sampling temperature (default: 0.9)\n"
        "  --top-k <n>              Top-k sampling (default: 50)\n"
        "  --rep-penalty <float>    Repetition penalty (default: 1.05)\n"
        "  --seed <n>               RNG seed (default: random)\n"
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
    const char * voice_name = NULL;
    int n_threads = 4;
    float speed = 1.0f;
    bool print_timing = true;
    bool use_stdin = false;
    bool use_large = false;
    struct vocal_sampling_params sampling = { 0.9f, 50, 1.05f, 0 };

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
        } else if (strcmp(argv[i], "--large") == 0) {
            use_large = true;
        } else if (strcmp(argv[i], "--tokenizer") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --tokenizer requires argument\n"); return VOCAL_ERR_ARGS; }
            tokenizer_override = argv[i];
        } else if (strcmp(argv[i], "--speed") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --speed requires argument\n"); return VOCAL_ERR_ARGS; }
            speed = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--threads") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --threads requires argument\n"); return VOCAL_ERR_ARGS; }
            n_threads = atoi(argv[i]);
        } else if (strcmp(argv[i], "--voice") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --voice requires argument\n"); return VOCAL_ERR_ARGS; }
            voice_name = argv[i];
        } else if (strcmp(argv[i], "--temperature") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --temperature requires argument\n"); return VOCAL_ERR_ARGS; }
            sampling.temperature = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--top-k") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --top-k requires argument\n"); return VOCAL_ERR_ARGS; }
            sampling.top_k = atoi(argv[i]);
        } else if (strcmp(argv[i], "--rep-penalty") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --rep-penalty requires argument\n"); return VOCAL_ERR_ARGS; }
            sampling.rep_penalty = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --seed requires argument\n"); return VOCAL_ERR_ARGS; }
            sampling.seed = (unsigned int)atol(argv[i]);
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

    const char * tts_model_name = use_large ? VOCAL_TTS_MODEL_LARGE_NAME : VOCAL_TTS_MODEL_NAME;

    // Resolve model path
    char model_path[4096];
    if (model_override) {
        snprintf(model_path, sizeof(model_path), "%s", model_override);
    } else {
        if (!vocal_model_path(tts_model_name, model_dir, model_path, sizeof(model_path))) {
            fprintf(stderr, "error: could not resolve model path\n");
            return VOCAL_ERR_MODEL;
        }
        if (!vocal_model_exists(tts_model_name, model_dir)) {
            fprintf(stderr, "error: TTS model not found at %s\n", model_path);
            if (use_large) {
                fprintf(stderr, "No hosted GGUF for 1.7B. Convert it yourself:\n");
                fprintf(stderr, "  huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir /tmp/Qwen3-TTS-1.7B\n");
                fprintf(stderr, "  python tools/convert_tts_to_gguf.py -i /tmp/Qwen3-TTS-1.7B -o %s\n", model_path);
            } else {
                fprintf(stderr, "Run: vocal download tts\n");
            }
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

    // If --voice specified, use voice profile path
    if (voice_name) {
        char voice_path[4096];
        if (!vocal_voice_path(voice_name, voice_path, sizeof(voice_path))) {
            fprintf(stderr, "error: could not resolve voice path\n");
            return VOCAL_ERR_IO;
        }
        // Check if the voice profile exists
        FILE * vf = fopen(voice_path, "rb");
        if (!vf) {
            fprintf(stderr, "error: voice profile not found: %s\n", voice_path);
            fprintf(stderr, "Create one with: vocal clone -f ref.wav --ref-text \"...\" --save %s\n", voice_name);
            return VOCAL_ERR_IO;
        }
        fclose(vf);

        return vocal_tts_with_voice(model_path, tokenizer_path, decoder_path,
                                     voice_path, text, output_path,
                                     n_threads, speed, print_timing, sampling);
    }

    return vocal_tts_run(model_path, tokenizer_path, decoder_path, text, output_path,
                         n_threads, speed, print_timing, sampling);
}

// --- Clone subcommand ---

static void print_clone_usage(void) {
    fprintf(stderr,
        "Usage: vocal clone [options]\n"
        "\n"
        "Clone a voice from reference audio and synthesize speech.\n"
        "Use --save to create a reusable voice profile.\n"
        "\n"
        "Options:\n"
        "  -f, --ref <path>         Reference audio file (WAV) [required]\n"
        "  --ref-text <text>        Transcript of reference audio (improves quality)\n"
        "  -t, --text <text>        Text to synthesize\n"
        "  --stdin                  Read synthesis text from stdin\n"
        "  -o, --output <path>      Output WAV file\n"
        "  --save <name>            Save voice profile for reuse (use with: vocal tts --voice <name>)\n"
        "  -m, --model <path>       Model file path\n"
        "  --large                  Use 1.7B model (default: 0.6B)\n"
        "  --tokenizer <path>       Tokenizer file path\n"
        "  --speed <float>          Speed factor (default: 1.0)\n"
        "  --threads <n>            Thread count (default: 4)\n"
        "  --temperature <float>    Sampling temperature (default: 0.9)\n"
        "  --top-k <n>              Top-k sampling (default: 50)\n"
        "  --rep-penalty <float>    Repetition penalty (default: 1.05)\n"
        "  --seed <n>               RNG seed (default: random)\n"
        "  --no-timing              Don't print timing info\n"
        "  --model-dir <path>       Models directory override\n"
        "  -h, --help               Show this help\n");
}

static int cmd_clone(int argc, char ** argv) {
    const char * ref_audio_path = NULL;
    const char * ref_text = NULL;
    const char * text = NULL;
    const char * output_path = NULL;
    const char * model_override = NULL;
    const char * tokenizer_override = NULL;
    const char * model_dir = NULL;
    const char * save_name = NULL;
    int n_threads = 4;
    float speed = 1.0f;
    bool print_timing = true;
    bool use_stdin = false;
    bool use_large = false;
    struct vocal_sampling_params sampling = { 0.9f, 50, 1.05f, 0 };

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--ref") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: -f requires argument\n"); return VOCAL_ERR_ARGS; }
            ref_audio_path = argv[i];
        } else if (strcmp(argv[i], "--ref-text") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --ref-text requires argument\n"); return VOCAL_ERR_ARGS; }
            ref_text = argv[i];
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--text") == 0) {
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
        } else if (strcmp(argv[i], "--large") == 0) {
            use_large = true;
        } else if (strcmp(argv[i], "--tokenizer") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --tokenizer requires argument\n"); return VOCAL_ERR_ARGS; }
            tokenizer_override = argv[i];
        } else if (strcmp(argv[i], "--speed") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --speed requires argument\n"); return VOCAL_ERR_ARGS; }
            speed = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--threads") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --threads requires argument\n"); return VOCAL_ERR_ARGS; }
            n_threads = atoi(argv[i]);
        } else if (strcmp(argv[i], "--save") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --save requires argument\n"); return VOCAL_ERR_ARGS; }
            save_name = argv[i];
        } else if (strcmp(argv[i], "--temperature") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --temperature requires argument\n"); return VOCAL_ERR_ARGS; }
            sampling.temperature = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--top-k") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --top-k requires argument\n"); return VOCAL_ERR_ARGS; }
            sampling.top_k = atoi(argv[i]);
        } else if (strcmp(argv[i], "--rep-penalty") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --rep-penalty requires argument\n"); return VOCAL_ERR_ARGS; }
            sampling.rep_penalty = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --seed requires argument\n"); return VOCAL_ERR_ARGS; }
            sampling.seed = (unsigned int)atol(argv[i]);
        } else if (strcmp(argv[i], "--no-timing") == 0) {
            print_timing = false;
        } else if (strcmp(argv[i], "--model-dir") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --model-dir requires argument\n"); return VOCAL_ERR_ARGS; }
            model_dir = argv[i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_clone_usage();
            return VOCAL_OK;
        } else {
            fprintf(stderr, "error: unknown option: %s\n", argv[i]);
            print_clone_usage();
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

    if (!ref_audio_path) {
        fprintf(stderr, "error: reference audio required (-f)\n\n");
        print_clone_usage();
        return VOCAL_ERR_ARGS;
    }

    // --save mode: encode and save voice profile, no synthesis needed
    if (save_name) {
        if (!ref_text || !ref_text[0]) {
            fprintf(stderr, "error: --ref-text required when saving voice profile\n\n");
            print_clone_usage();
            return VOCAL_ERR_ARGS;
        }

        char voice_path[4096];
        if (!vocal_voice_path(save_name, voice_path, sizeof(voice_path))) {
            fprintf(stderr, "error: could not resolve voice path\n");
            return VOCAL_ERR_IO;
        }

        // Ensure voices directory exists
        const char * vdir = vocal_voices_dir();
        if (vdir) {
            char mkdir_cmd[4200];
            snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p '%s'", vdir);
            (void)system(mkdir_cmd);
        }

        const char * tts_model_name = use_large ? VOCAL_TTS_MODEL_LARGE_NAME : VOCAL_TTS_MODEL_NAME;
        const char * spk_encoder_name = use_large ? VOCAL_TTS_SPK_ENCODER_LARGE_NAME : VOCAL_TTS_SPK_ENCODER_NAME;

        // Resolve model paths for encoding
        char model_path[4096], tokenizer_path[4096], decoder_path[4096];
        char encoder_path[4096], spk_encoder_path[4096];

        if (model_override) {
            snprintf(model_path, sizeof(model_path), "%s", model_override);
        } else {
            if (!vocal_model_path(tts_model_name, model_dir, model_path, sizeof(model_path)) ||
                !vocal_model_exists(tts_model_name, model_dir)) {
                fprintf(stderr, "error: TTS model not found at %s\n", model_path);
                if (use_large) {
                    fprintf(stderr, "No hosted GGUF for 1.7B. Convert it yourself:\n");
                    fprintf(stderr, "  huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir /tmp/Qwen3-TTS-1.7B\n");
                    fprintf(stderr, "  python tools/convert_tts_to_gguf.py -i /tmp/Qwen3-TTS-1.7B -o %s\n", model_path);
                } else {
                    fprintf(stderr, "Run: vocal download tts\n");
                }
                return VOCAL_ERR_MODEL;
            }
        }
        if (tokenizer_override) {
            snprintf(tokenizer_path, sizeof(tokenizer_path), "%s", tokenizer_override);
        } else {
            vocal_model_path(VOCAL_TTS_TOKENIZER_NAME, model_dir, tokenizer_path, sizeof(tokenizer_path));
        }
        vocal_model_path(VOCAL_TTS_DECODER_NAME, model_dir, decoder_path, sizeof(decoder_path));
        vocal_model_path(VOCAL_TTS_ENCODER_NAME, model_dir, encoder_path, sizeof(encoder_path));
        vocal_model_path(spk_encoder_name, model_dir, spk_encoder_path, sizeof(spk_encoder_path));

        if (!vocal_model_exists(VOCAL_TTS_ENCODER_NAME, model_dir)) {
            fprintf(stderr, "error: Codec encoder not found. Run: vocal download clone\n");
            return VOCAL_ERR_MODEL;
        }
        if (!vocal_model_exists(spk_encoder_name, model_dir)) {
            fprintf(stderr, "error: Speaker encoder not found.\n");
            if (use_large) {
                fprintf(stderr, "Convert the 1.7B model first (speaker encoder is embedded in the model GGUF).\n");
            } else {
                fprintf(stderr, "Run: vocal download clone\n");
            }
            return VOCAL_ERR_MODEL;
        }

        return vocal_tts_save_voice(model_path, tokenizer_path, decoder_path,
                                     encoder_path, spk_encoder_path,
                                     ref_audio_path, ref_text, voice_path);
    }

    // Synthesis mode: require text and output
    if (!text || !text[0]) {
        fprintf(stderr, "error: text required (-t or --stdin)\n\n");
        print_clone_usage();
        return VOCAL_ERR_ARGS;
    }

    if (!output_path) {
        fprintf(stderr, "error: output file required (-o)\n\n");
        print_clone_usage();
        return VOCAL_ERR_ARGS;
    }

    const char * tts_model_name = use_large ? VOCAL_TTS_MODEL_LARGE_NAME : VOCAL_TTS_MODEL_NAME;
    const char * spk_encoder_name = use_large ? VOCAL_TTS_SPK_ENCODER_LARGE_NAME : VOCAL_TTS_SPK_ENCODER_NAME;

    // Resolve model paths
    char model_path[4096];
    if (model_override) {
        snprintf(model_path, sizeof(model_path), "%s", model_override);
    } else {
        if (!vocal_model_path(tts_model_name, model_dir, model_path, sizeof(model_path))) {
            fprintf(stderr, "error: could not resolve model path\n");
            return VOCAL_ERR_MODEL;
        }
        if (!vocal_model_exists(tts_model_name, model_dir)) {
            fprintf(stderr, "error: TTS model not found at %s\n", model_path);
            if (use_large) {
                fprintf(stderr, "No hosted GGUF for 1.7B. Convert it yourself:\n");
                fprintf(stderr, "  huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir /tmp/Qwen3-TTS-1.7B\n");
                fprintf(stderr, "  python tools/convert_tts_to_gguf.py -i /tmp/Qwen3-TTS-1.7B -o %s\n", model_path);
            } else {
                fprintf(stderr, "Run: vocal download tts\n");
            }
            return VOCAL_ERR_MODEL;
        }
    }

    char tokenizer_path[4096];
    if (tokenizer_override) {
        snprintf(tokenizer_path, sizeof(tokenizer_path), "%s", tokenizer_override);
    } else {
        if (!vocal_model_path(VOCAL_TTS_TOKENIZER_NAME, model_dir, tokenizer_path, sizeof(tokenizer_path))) {
            fprintf(stderr, "error: could not resolve tokenizer path\n");
            return VOCAL_ERR_MODEL;
        }
    }

    char decoder_path[4096];
    if (!vocal_model_path(VOCAL_TTS_DECODER_NAME, model_dir, decoder_path, sizeof(decoder_path))) {
        fprintf(stderr, "error: could not resolve decoder path\n");
        return VOCAL_ERR_MODEL;
    }
    if (!vocal_model_exists(VOCAL_TTS_DECODER_NAME, model_dir)) {
        fprintf(stderr, "error: TTS decoder not found. Run: vocal download tts\n");
        return VOCAL_ERR_MODEL;
    }

    // Codec encoder loads from the tokenizer GGUF (same file as decoder)
    char encoder_path[4096];
    if (!vocal_model_path(VOCAL_TTS_ENCODER_NAME, model_dir, encoder_path, sizeof(encoder_path))) {
        fprintf(stderr, "error: could not resolve encoder path\n");
        return VOCAL_ERR_MODEL;
    }

    // Speaker encoder is in the main model GGUF
    char spk_encoder_path[4096];
    if (!vocal_model_path(spk_encoder_name, model_dir, spk_encoder_path, sizeof(spk_encoder_path))) {
        fprintf(stderr, "error: could not resolve speaker encoder path\n");
        return VOCAL_ERR_MODEL;
    }
    if (!vocal_model_exists(spk_encoder_name, model_dir)) {
        fprintf(stderr, "error: Speaker encoder not found.\n");
        if (use_large) {
            fprintf(stderr, "Convert the 1.7B model first (speaker encoder is embedded in the model GGUF).\n");
        } else {
            fprintf(stderr, "Run: vocal download tts\n");
        }
        return VOCAL_ERR_MODEL;
    }

    return vocal_tts_clone(model_path, tokenizer_path, decoder_path,
                            encoder_path, spk_encoder_path,
                            ref_audio_path, ref_text ? ref_text : "",
                            text, output_path,
                            n_threads, speed, print_timing, sampling);
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
        "  asr-large    Qwen3-ASR 1.7B\n"
        "  tts          Qwen3-TTS 0.6B + tokenizer + decoder\n"
        "  clone        TTS models (same as tts; encoders are embedded)\n"
        "\n"
        "Note: No hosted GGUF for TTS 1.7B. Use tools/convert_tts_to_gguf.py to convert.\n"
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
    } else if (strcmp(model_type, "asr-large") == 0) {
        return vocal_model_download(VOCAL_ASR_MODEL_LARGE_URL, VOCAL_ASR_MODEL_LARGE_NAME, model_dir);
    } else if (strcmp(model_type, "tts") == 0) {
        int ret = vocal_model_download(VOCAL_TTS_MODEL_URL, VOCAL_TTS_MODEL_NAME, model_dir);
        if (ret != 0) return ret;
        ret = vocal_model_download(VOCAL_TTS_TOKENIZER_URL, VOCAL_TTS_TOKENIZER_NAME, model_dir);
        if (ret != 0) return ret;
        return vocal_model_download(VOCAL_TTS_DECODER_URL, VOCAL_TTS_DECODER_NAME, model_dir);
    } else if (strcmp(model_type, "clone") == 0) {
        // Download all TTS models + encoder models
        int ret = vocal_model_download(VOCAL_TTS_MODEL_URL, VOCAL_TTS_MODEL_NAME, model_dir);
        if (ret != 0) return ret;
        ret = vocal_model_download(VOCAL_TTS_TOKENIZER_URL, VOCAL_TTS_TOKENIZER_NAME, model_dir);
        if (ret != 0) return ret;
        ret = vocal_model_download(VOCAL_TTS_DECODER_URL, VOCAL_TTS_DECODER_NAME, model_dir);
        if (ret != 0) return ret;
        ret = vocal_model_download(VOCAL_TTS_ENCODER_URL, VOCAL_TTS_ENCODER_NAME, model_dir);
        if (ret != 0) return ret;
        return vocal_model_download(VOCAL_TTS_SPK_ENCODER_URL, VOCAL_TTS_SPK_ENCODER_NAME, model_dir);
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
        return cmd_clone(argc - 2, argv + 2);
    } else if (strcmp(cmd, "voices") == 0) {
        vocal_voices_list();
        return VOCAL_OK;
    } else {
        fprintf(stderr, "error: unknown command '%s'\n\n", cmd);
        print_usage();
        return VOCAL_ERR_ARGS;
    }
}
