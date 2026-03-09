#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "audio_io.h"

#include <stdio.h>
#include <stdlib.h>

bool vocal_wav_read(const char * path, float ** samples, int * n_samples,
                    int * sample_rate) {
    drwav wav;
    if (!drwav_init_file(&wav, path, NULL)) {
        fprintf(stderr, "error: failed to open WAV file: %s\n", path);
        return false;
    }

    // Read all frames as float, mono
    uint64_t total_frames = wav.totalPCMFrameCount;
    float * raw = (float *)malloc(total_frames * wav.channels * sizeof(float));
    if (!raw) {
        drwav_uninit(&wav);
        return false;
    }

    uint64_t frames_read = drwav_read_pcm_frames_f32(&wav, total_frames, raw);

    if (wav.channels > 1) {
        // Mix down to mono
        float * mono = (float *)malloc(frames_read * sizeof(float));
        if (!mono) {
            free(raw);
            drwav_uninit(&wav);
            return false;
        }
        for (uint64_t i = 0; i < frames_read; i++) {
            float sum = 0.0f;
            for (unsigned int ch = 0; ch < wav.channels; ch++) {
                sum += raw[i * wav.channels + ch];
            }
            mono[i] = sum / wav.channels;
        }
        free(raw);
        *samples = mono;
    } else {
        *samples = raw;
    }

    *n_samples = (int)frames_read;
    *sample_rate = (int)wav.sampleRate;

    drwav_uninit(&wav);
    return true;
}

bool vocal_wav_write(const char * path, const float * samples, int n_samples,
                     int sample_rate, int channels) {
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = channels;
    format.sampleRate = sample_rate;
    format.bitsPerSample = 32;

    drwav wav;
    if (!drwav_init_file_write(&wav, path, &format, NULL)) {
        fprintf(stderr, "error: failed to create WAV file: %s\n", path);
        return false;
    }

    drwav_write_pcm_frames(&wav, n_samples, samples);
    drwav_uninit(&wav);
    return true;
}
