#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#define DR_MP3_IMPLEMENTATION
#include "dr_mp3.h"

#define DR_FLAC_IMPLEMENTATION
#include "dr_flac.h"

#include "audio_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Resampling ---

bool vocal_resample(const float * samples, int n_samples, int src_rate,
                    int dst_rate, float ** out_samples, int * out_n_samples) {
    if (src_rate == dst_rate) {
        float * copy = (float *)malloc(n_samples * sizeof(float));
        if (!copy) return false;
        memcpy(copy, samples, n_samples * sizeof(float));
        *out_samples = copy;
        *out_n_samples = n_samples;
        return true;
    }

    double ratio = (double)dst_rate / src_rate;
    int out_len = (int)(n_samples * ratio);
    if (out_len <= 0) return false;

    float * out = (float *)malloc(out_len * sizeof(float));
    if (!out) return false;

    for (int i = 0; i < out_len; i++) {
        double src_idx = i / ratio;
        int idx0 = (int)src_idx;
        int idx1 = idx0 + 1;
        if (idx1 >= n_samples) idx1 = n_samples - 1;
        double frac = src_idx - idx0;
        out[i] = (float)((1.0 - frac) * samples[idx0] + frac * samples[idx1]);
    }

    *out_samples = out;
    *out_n_samples = out_len;
    return true;
}

// --- Mono mixdown helper ---

static float * mix_to_mono(const float * raw, uint64_t n_frames, unsigned int channels) {
    float * mono = (float *)malloc(n_frames * sizeof(float));
    if (!mono) return NULL;

    for (uint64_t i = 0; i < n_frames; i++) {
        float sum = 0.0f;
        for (unsigned int ch = 0; ch < channels; ch++) {
            sum += raw[i * channels + ch];
        }
        mono[i] = sum / channels;
    }
    return mono;
}

// --- WAV ---

bool vocal_wav_read(const char * path, float ** samples, int * n_samples,
                    int * sample_rate) {
    drwav wav;
    if (!drwav_init_file(&wav, path, NULL)) {
        fprintf(stderr, "error: failed to open WAV file: %s\n", path);
        return false;
    }

    uint64_t total_frames = wav.totalPCMFrameCount;
    float * raw = (float *)malloc(total_frames * wav.channels * sizeof(float));
    if (!raw) {
        drwav_uninit(&wav);
        return false;
    }

    uint64_t frames_read = drwav_read_pcm_frames_f32(&wav, total_frames, raw);

    if (wav.channels > 1) {
        float * mono = mix_to_mono(raw, frames_read, wav.channels);
        free(raw);
        if (!mono) { drwav_uninit(&wav); return false; }
        *samples = mono;
    } else {
        *samples = raw;
    }

    *n_samples = (int)frames_read;
    *sample_rate = (int)wav.sampleRate;

    drwav_uninit(&wav);
    return true;
}

// --- MP3 ---

static bool vocal_mp3_read(const char * path, float ** samples, int * n_samples,
                           int * sample_rate) {
    drmp3 mp3;
    if (!drmp3_init_file(&mp3, path, NULL)) {
        fprintf(stderr, "error: failed to open MP3 file: %s\n", path);
        return false;
    }

    uint64_t total_frames = mp3.totalPCMFrameCount;
    if (total_frames == 0 || total_frames == DRMP3_UINT64_MAX) {
        // Unknown length — read in chunks
        size_t capacity = 1024 * 1024;  // 1M frames initial
        float * buf = (float *)malloc(capacity * mp3.channels * sizeof(float));
        if (!buf) { drmp3_uninit(&mp3); return false; }

        uint64_t total_read = 0;
        while (1) {
            if (total_read >= capacity) {
                capacity *= 2;
                float * tmp = (float *)realloc(buf, capacity * mp3.channels * sizeof(float));
                if (!tmp) { free(buf); drmp3_uninit(&mp3); return false; }
                buf = tmp;
            }
            uint64_t to_read = capacity - total_read;
            uint64_t read = drmp3_read_pcm_frames_f32(&mp3, to_read,
                                buf + total_read * mp3.channels);
            if (read == 0) break;
            total_read += read;
        }
        total_frames = total_read;

        if (mp3.channels > 1) {
            float * mono = mix_to_mono(buf, total_frames, mp3.channels);
            free(buf);
            if (!mono) { drmp3_uninit(&mp3); return false; }
            *samples = mono;
        } else {
            *samples = buf;
        }
    } else {
        float * raw = (float *)malloc(total_frames * mp3.channels * sizeof(float));
        if (!raw) { drmp3_uninit(&mp3); return false; }

        uint64_t frames_read = drmp3_read_pcm_frames_f32(&mp3, total_frames, raw);

        if (mp3.channels > 1) {
            float * mono = mix_to_mono(raw, frames_read, mp3.channels);
            free(raw);
            if (!mono) { drmp3_uninit(&mp3); return false; }
            *samples = mono;
        } else {
            *samples = raw;
        }
        total_frames = frames_read;
    }

    *n_samples = (int)total_frames;
    *sample_rate = (int)mp3.sampleRate;

    drmp3_uninit(&mp3);
    return true;
}

// --- FLAC ---

static bool vocal_flac_read(const char * path, float ** samples, int * n_samples,
                            int * sample_rate) {
    drflac * flac = drflac_open_file(path, NULL);
    if (!flac) {
        fprintf(stderr, "error: failed to open FLAC file: %s\n", path);
        return false;
    }

    uint64_t total_frames = flac->totalPCMFrameCount;
    float * raw = (float *)malloc(total_frames * flac->channels * sizeof(float));
    if (!raw) { drflac_close(flac); return false; }

    uint64_t frames_read = drflac_read_pcm_frames_f32(flac, total_frames, raw);

    if (flac->channels > 1) {
        float * mono = mix_to_mono(raw, frames_read, flac->channels);
        free(raw);
        if (!mono) { drflac_close(flac); return false; }
        *samples = mono;
    } else {
        *samples = raw;
    }

    *n_samples = (int)frames_read;
    *sample_rate = (int)flac->sampleRate;

    drflac_close(flac);
    return true;
}

// --- In-memory readers ---

static bool vocal_wav_read_memory(const void * data, size_t data_size,
                                   float ** samples, int * n_samples, int * sample_rate) {
    drwav wav;
    if (!drwav_init_memory(&wav, data, data_size, NULL)) {
        fprintf(stderr, "error: failed to decode WAV from memory\n");
        return false;
    }

    uint64_t total_frames = wav.totalPCMFrameCount;
    float * raw = (float *)malloc(total_frames * wav.channels * sizeof(float));
    if (!raw) { drwav_uninit(&wav); return false; }

    uint64_t frames_read = drwav_read_pcm_frames_f32(&wav, total_frames, raw);

    if (wav.channels > 1) {
        float * mono = mix_to_mono(raw, frames_read, wav.channels);
        free(raw);
        if (!mono) { drwav_uninit(&wav); return false; }
        *samples = mono;
    } else {
        *samples = raw;
    }

    *n_samples = (int)frames_read;
    *sample_rate = (int)wav.sampleRate;
    drwav_uninit(&wav);
    return true;
}

static bool vocal_mp3_read_memory(const void * data, size_t data_size,
                                   float ** samples, int * n_samples, int * sample_rate) {
    drmp3 mp3;
    if (!drmp3_init_memory(&mp3, data, data_size, NULL)) {
        fprintf(stderr, "error: failed to decode MP3 from memory\n");
        return false;
    }

    // Read in chunks (totalPCMFrameCount may be unknown)
    size_t capacity = 1024 * 1024;
    float * buf = (float *)malloc(capacity * mp3.channels * sizeof(float));
    if (!buf) { drmp3_uninit(&mp3); return false; }

    uint64_t total_read = 0;
    while (1) {
        if (total_read >= capacity) {
            capacity *= 2;
            float * tmp = (float *)realloc(buf, capacity * mp3.channels * sizeof(float));
            if (!tmp) { free(buf); drmp3_uninit(&mp3); return false; }
            buf = tmp;
        }
        uint64_t read = drmp3_read_pcm_frames_f32(&mp3, capacity - total_read,
                            buf + total_read * mp3.channels);
        if (read == 0) break;
        total_read += read;
    }

    if (mp3.channels > 1) {
        float * mono = mix_to_mono(buf, total_read, mp3.channels);
        free(buf);
        if (!mono) { drmp3_uninit(&mp3); return false; }
        *samples = mono;
    } else {
        *samples = buf;
    }

    *n_samples = (int)total_read;
    *sample_rate = (int)mp3.sampleRate;
    drmp3_uninit(&mp3);
    return true;
}

static bool vocal_flac_read_memory(const void * data, size_t data_size,
                                    float ** samples, int * n_samples, int * sample_rate) {
    drflac * flac = drflac_open_memory(data, data_size, NULL);
    if (!flac) {
        fprintf(stderr, "error: failed to decode FLAC from memory\n");
        return false;
    }

    uint64_t total_frames = flac->totalPCMFrameCount;
    float * raw = (float *)malloc(total_frames * flac->channels * sizeof(float));
    if (!raw) { drflac_close(flac); return false; }

    uint64_t frames_read = drflac_read_pcm_frames_f32(flac, total_frames, raw);

    if (flac->channels > 1) {
        float * mono = mix_to_mono(raw, frames_read, flac->channels);
        free(raw);
        if (!mono) { drflac_close(flac); return false; }
        *samples = mono;
    } else {
        *samples = raw;
    }

    *n_samples = (int)frames_read;
    *sample_rate = (int)flac->sampleRate;
    drflac_close(flac);
    return true;
}

bool vocal_audio_read_memory(const void * data, size_t data_size,
                             const char * format_hint,
                             float ** samples, int * n_samples,
                             int * sample_rate, int target_sample_rate) {
    bool ok;

    if (format_hint && strcmp(format_hint, "mp3") == 0) {
        ok = vocal_mp3_read_memory(data, data_size, samples, n_samples, sample_rate);
    } else if (format_hint && strcmp(format_hint, "flac") == 0) {
        ok = vocal_flac_read_memory(data, data_size, samples, n_samples, sample_rate);
    } else {
        // WAV or unknown — try WAV
        ok = vocal_wav_read_memory(data, data_size, samples, n_samples, sample_rate);
    }

    if (!ok) return false;

    // Resample if requested
    if (target_sample_rate > 0 && *sample_rate != target_sample_rate) {
        float * resampled = NULL;
        int resampled_len = 0;
        if (!vocal_resample(*samples, *n_samples, *sample_rate,
                            target_sample_rate, &resampled, &resampled_len)) {
            free(*samples);
            *samples = NULL;
            return false;
        }
        free(*samples);
        *samples = resampled;
        *n_samples = resampled_len;
        *sample_rate = target_sample_rate;
    }

    return true;
}

// --- Unified reader ---

static const char * get_extension(const char * path) {
    const char * dot = strrchr(path, '.');
    return dot ? dot : "";
}

static int strcasecmp_ext(const char * a, const char * b) {
    while (*a && *b) {
        char ca = *a >= 'A' && *a <= 'Z' ? *a + 32 : *a;
        char cb = *b >= 'A' && *b <= 'Z' ? *b + 32 : *b;
        if (ca != cb) return ca - cb;
        a++; b++;
    }
    return (unsigned char)*a - (unsigned char)*b;
}

bool vocal_audio_read(const char * path, float ** samples, int * n_samples,
                      int * sample_rate, int target_sample_rate) {
    const char * ext = get_extension(path);
    bool ok;

    if (strcasecmp_ext(ext, ".wav") == 0) {
        ok = vocal_wav_read(path, samples, n_samples, sample_rate);
    } else if (strcasecmp_ext(ext, ".mp3") == 0) {
        ok = vocal_mp3_read(path, samples, n_samples, sample_rate);
    } else if (strcasecmp_ext(ext, ".flac") == 0) {
        ok = vocal_flac_read(path, samples, n_samples, sample_rate);
    } else {
        // Try WAV as fallback (dr_wav handles format validation)
        ok = vocal_wav_read(path, samples, n_samples, sample_rate);
    }

    if (!ok) return false;

    // Resample if requested and needed
    if (target_sample_rate > 0 && *sample_rate != target_sample_rate) {
        float * resampled = NULL;
        int resampled_len = 0;
        if (!vocal_resample(*samples, *n_samples, *sample_rate,
                            target_sample_rate, &resampled, &resampled_len)) {
            free(*samples);
            *samples = NULL;
            return false;
        }
        fprintf(stderr, "Resampled: %d Hz -> %d Hz (%d -> %d samples)\n",
                *sample_rate, target_sample_rate, *n_samples, resampled_len);
        free(*samples);
        *samples = resampled;
        *n_samples = resampled_len;
        *sample_rate = target_sample_rate;
    }

    return true;
}

// --- WAV write ---

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
