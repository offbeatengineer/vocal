#include "tts.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>

// dr_wav for writing output
#include "dr_wav.h"

extern "C" {

static void ggml_log_quiet_tts(enum ggml_log_level level, const char * text, void * user_data) {
    (void)user_data;
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

int vocal_tts_run(const char * model_path, const char * tokenizer_path,
                  const char * decoder_path, const char * text,
                  const char * output_path,
                  int n_threads, float speed, bool print_timing) {
    ggml_log_set(ggml_log_quiet_tts, nullptr);

    fprintf(stderr, "vocal tts\n");
    fprintf(stderr, "  Model: %s\n", model_path);
    fprintf(stderr, "  Tokenizer: %s\n", tokenizer_path);
    fprintf(stderr, "  Decoder: %s\n", decoder_path);
    fprintf(stderr, "  Text: \"%s\"\n", text);
    fprintf(stderr, "\n");

    vocal_tts::TTS tts;

    if (!tts.load(model_path, tokenizer_path, decoder_path)) {
        fprintf(stderr, "error: %s\n", tts.get_error().c_str());
        return 2;
    }

    vocal_tts::tts_params params;
    params.n_threads = n_threads;
    params.speed = speed;
    params.print_timing = print_timing;

    auto result = tts.synthesize(text, params);

    if (!result.success) {
        fprintf(stderr, "error: %s\n", result.error_msg.c_str());
        return 5;
    }

    // Write output WAV
    if (!output_path || !output_path[0]) {
        fprintf(stderr, "error: output path required (-o)\n");
        return 4;
    }

    // Write WAV using dr_wav
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 1;
    format.sampleRate = result.sample_rate;
    format.bitsPerSample = 32;

    drwav wav;
    if (!drwav_init_file_write(&wav, output_path, &format, NULL)) {
        fprintf(stderr, "error: failed to create WAV file: %s\n", output_path);
        return 4;
    }

    drwav_write_pcm_frames(&wav, result.audio.size(), result.audio.data());
    drwav_uninit(&wav);

    fprintf(stderr, "Output written to: %s\n", output_path);
    return 0;
}

} // extern "C"
