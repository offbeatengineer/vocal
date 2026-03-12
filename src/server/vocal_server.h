#ifndef VOCAL_SERVER_H
#define VOCAL_SERVER_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct vocal_serve_params {
    const char * host;
    int port;
    bool load_asr;
    bool load_tts;
    bool use_large;
    const char * model_dir;
    int n_threads;
};

// Start the HTTP server (blocks until SIGINT).
// Returns 0 on clean shutdown, non-zero on error.
int vocal_serve_run(const struct vocal_serve_params * params);

#ifdef __cplusplus
}
#endif

#endif // VOCAL_SERVER_H
