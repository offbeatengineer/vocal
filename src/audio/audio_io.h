#ifndef VOCAL_AUDIO_IO_H
#define VOCAL_AUDIO_IO_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif // VOCAL_AUDIO_IO_H
