#ifndef VOCAL_MODEL_H
#define VOCAL_MODEL_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Get the models directory path.
// Priority: --model-path flag > VOCAL_MODELS_DIR env > ~/.vocal/models/
// Returns a pointer to a static buffer. Not thread-safe.
const char * vocal_models_dir(const char * override_path);

// Get the full path to a specific model file.
// Writes into buf (must be at least bufsize bytes).
// Returns buf on success, NULL on failure.
char * vocal_model_path(const char * model_name, const char * override_dir,
                        char * buf, int bufsize);

// Check if a model file exists at the expected path.
bool vocal_model_exists(const char * model_name, const char * override_dir);

// Download a model from URL using system curl.
// Shows progress on stderr.
// Returns 0 on success, non-zero on failure.
int vocal_model_download(const char * url, const char * model_name,
                         const char * override_dir);

// List all downloaded models. Prints to stdout.
void vocal_models_list(const char * override_dir);

// Get the voices directory path (~/.vocal/voices/ by default).
const char * vocal_voices_dir(void);

// Get the full path to a voice profile file (~/.vocal/voices/<name>.voice).
// Writes into buf. Returns buf on success, NULL on failure.
char * vocal_voice_path(const char * name, char * buf, int bufsize);

// List all saved voice profiles. Prints to stdout.
void vocal_voices_list(void);

#ifdef __cplusplus
}
#endif

#endif // VOCAL_MODEL_H
