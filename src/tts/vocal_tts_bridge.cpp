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

static int write_wav(const char * path, const std::vector<float> & audio, int sample_rate) {
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 1;
    format.sampleRate = sample_rate;
    format.bitsPerSample = 32;

    drwav wav;
    if (!drwav_init_file_write(&wav, path, &format, NULL)) {
        fprintf(stderr, "error: failed to create WAV file: %s\n", path);
        return 4;
    }

    drwav_write_pcm_frames(&wav, audio.size(), audio.data());
    drwav_uninit(&wav);
    return 0;
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

    if (!output_path || !output_path[0]) {
        fprintf(stderr, "error: output path required (-o)\n");
        return 4;
    }

    int ret = write_wav(output_path, result.audio, result.sample_rate);
    if (ret != 0) return ret;

    fprintf(stderr, "Output written to: %s\n", output_path);
    return 0;
}

int vocal_tts_clone(const char * model_path, const char * tokenizer_path,
                    const char * decoder_path, const char * encoder_path,
                    const char * spk_encoder_path,
                    const char * ref_audio_path, const char * ref_text,
                    const char * text, const char * output_path,
                    int n_threads, float speed, bool print_timing) {
    ggml_log_set(ggml_log_quiet_tts, nullptr);

    fprintf(stderr, "vocal clone\n");
    fprintf(stderr, "  Model: %s\n", model_path);
    fprintf(stderr, "  Tokenizer: %s\n", tokenizer_path);
    fprintf(stderr, "  Decoder: %s\n", decoder_path);
    fprintf(stderr, "  Codec encoder: %s\n", encoder_path);
    fprintf(stderr, "  Speaker encoder: %s\n", spk_encoder_path);
    fprintf(stderr, "  Reference: %s\n", ref_audio_path);
    if (ref_text && ref_text[0]) {
        fprintf(stderr, "  Ref text: \"%s\"\n", ref_text);
    }
    fprintf(stderr, "  Text: \"%s\"\n", text);
    fprintf(stderr, "\n");

    vocal_tts::TTS tts;

    if (!tts.load(model_path, tokenizer_path, decoder_path)) {
        fprintf(stderr, "error: %s\n", tts.get_error().c_str());
        return 2;
    }

    if (!tts.load_encoders(encoder_path, spk_encoder_path)) {
        fprintf(stderr, "error: %s\n", tts.get_error().c_str());
        return 2;
    }

    vocal_tts::tts_params params;
    params.n_threads = n_threads;
    params.speed = speed;
    params.print_timing = print_timing;
    params.ref_audio_path = ref_audio_path;
    if (ref_text && ref_text[0]) {
        params.ref_text = ref_text;
    }

    auto result = tts.synthesize(text, params);

    if (!result.success) {
        fprintf(stderr, "error: %s\n", result.error_msg.c_str());
        return 5;
    }

    if (!output_path || !output_path[0]) {
        fprintf(stderr, "error: output path required (-o)\n");
        return 4;
    }

    int ret = write_wav(output_path, result.audio, result.sample_rate);
    if (ret != 0) return ret;

    fprintf(stderr, "Output written to: %s\n", output_path);
    return 0;
}

} // extern "C"
