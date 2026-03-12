#ifndef VOCAL_AUDIO_IO_H
#define VOCAL_AUDIO_IO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Read an audio file (WAV, MP3, or FLAC) and return mono float samples
// normalized to [-1, 1]. Format is detected by file extension.
// If target_sample_rate > 0, resamples to that rate (e.g. 16000 for ASR).
// If target_sample_rate == 0, returns at original sample rate.
// Caller must free(*samples) when done.
// Returns true on success.
bool vocal_audio_read(const char * path, float ** samples, int * n_samples,
                      int * sample_rate, int target_sample_rate);

// Read a WAV file and return float samples normalized to [-1, 1].
// Supports any sample rate and channel count — will convert to mono.
// Caller must free(*samples) when done.
// Returns true on success.
bool vocal_wav_read(const char * path, float ** samples, int * n_samples,
                    int * sample_rate);

// Write float samples to a WAV file.
// samples must be normalized to [-1, 1].
// Returns true on success.
bool vocal_wav_write(const char * path, const float * samples, int n_samples,
                     int sample_rate, int channels);

// Read audio from an in-memory buffer (WAV, MP3, or FLAC).
// format_hint: "wav", "mp3", or "flac".
// If target_sample_rate > 0, resamples to that rate.
// Caller must free(*samples) when done.
// Returns true on success.
bool vocal_audio_read_memory(const void * data, size_t data_size,
                             const char * format_hint,
                             float ** samples, int * n_samples,
                             int * sample_rate, int target_sample_rate);

// Resample audio using linear interpolation.
// Caller must free(*out_samples) when done.
// Returns true on success.
bool vocal_resample(const float * samples, int n_samples, int src_rate,
                    int dst_rate, float ** out_samples, int * out_n_samples);

#ifdef __cplusplus
}
#endif

#endif // VOCAL_AUDIO_IO_H
