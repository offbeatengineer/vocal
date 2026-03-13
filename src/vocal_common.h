#ifndef VOCAL_COMMON_H
#define VOCAL_COMMON_H

#define VOCAL_VERSION "0.1.0"

// Default paths
#define VOCAL_DEFAULT_MODELS_DIR ".vocal/models"
#define VOCAL_DEFAULT_VOICES_DIR ".vocal/voices"

// Environment variables
#define VOCAL_ENV_MODELS_DIR "VOCAL_MODELS_DIR"

// Model identifiers — 0.6B (default)
#define VOCAL_ASR_MODEL_NAME "qwen3-asr-0.6b-f16.gguf"
#define VOCAL_TTS_MODEL_NAME "qwen3-tts-0.6b-f16.gguf"
#define VOCAL_TTS_TOKENIZER_NAME "tokenizer.json"
#define VOCAL_TTS_DECODER_NAME "qwen3-tts-tokenizer-f16.gguf"
// Codec encoder is embedded in the tokenizer GGUF (tok_enc.* tensors)
#define VOCAL_TTS_ENCODER_NAME VOCAL_TTS_DECODER_NAME
// Speaker encoder is embedded in the main model GGUF (spk_enc.* tensors)
#define VOCAL_TTS_SPK_ENCODER_NAME VOCAL_TTS_MODEL_NAME

// Forced aligner model (word-level timestamps)
#define VOCAL_ALIGNER_MODEL_NAME "qwen3-aligner-0.6b-f16.gguf"
#define VOCAL_ALIGNER_MODEL_URL "https://huggingface.co/offbeatengineer/qwen3-asr-gguf/resolve/main/qwen3-aligner-0.6b-f16.gguf"

// Model identifiers — 1.7B (--large)
#define VOCAL_ASR_MODEL_LARGE_NAME "qwen3-asr-1.7b-f16.gguf"
#define VOCAL_TTS_MODEL_LARGE_NAME "qwen3-tts-1.7b-f16.gguf"
#define VOCAL_TTS_SPK_ENCODER_LARGE_NAME VOCAL_TTS_MODEL_LARGE_NAME

// CustomVoice model identifiers
#define VOCAL_TTS_CUSTOM_MODEL_NAME "qwen3-tts-custom-0.6b-f16.gguf"
#define VOCAL_TTS_CUSTOM_MODEL_LARGE_NAME "qwen3-tts-custom-1.7b-f16.gguf"

// VoiceDesign model identifiers (1.7B only)
#define VOCAL_TTS_DESIGN_MODEL_NAME "qwen3-tts-design-1.7b-f16.gguf"

// HuggingFace URLs for GGUF models (offbeatengineer org)
#define VOCAL_ASR_MODEL_URL "https://huggingface.co/offbeatengineer/qwen3-asr-gguf/resolve/main/qwen3-asr-0.6b-f16.gguf"
#define VOCAL_ASR_MODEL_LARGE_URL "https://huggingface.co/offbeatengineer/qwen3-asr-gguf/resolve/main/qwen3-asr-1.7b-f16.gguf"
#define VOCAL_TTS_MODEL_URL "https://huggingface.co/offbeatengineer/qwen3-tts-gguf/resolve/main/qwen3-tts-0.6b-f16.gguf"
#define VOCAL_TTS_MODEL_LARGE_URL "https://huggingface.co/offbeatengineer/qwen3-tts-gguf/resolve/main/qwen3-tts-1.7b-f16.gguf"
#define VOCAL_TTS_TOKENIZER_URL "https://huggingface.co/offbeatengineer/qwen3-tts-gguf/resolve/main/tokenizer.json"
#define VOCAL_TTS_DECODER_URL "https://huggingface.co/offbeatengineer/qwen3-tts-gguf/resolve/main/qwen3-tts-tokenizer-f16.gguf"
#define VOCAL_TTS_ENCODER_URL VOCAL_TTS_DECODER_URL
#define VOCAL_TTS_SPK_ENCODER_URL VOCAL_TTS_MODEL_URL
#define VOCAL_TTS_SPK_ENCODER_LARGE_URL VOCAL_TTS_MODEL_LARGE_URL

// CustomVoice model URLs
#define VOCAL_TTS_CUSTOM_MODEL_URL "https://huggingface.co/offbeatengineer/qwen3-tts-gguf/resolve/main/qwen3-tts-custom-0.6b-f16.gguf"
#define VOCAL_TTS_CUSTOM_MODEL_LARGE_URL "https://huggingface.co/offbeatengineer/qwen3-tts-gguf/resolve/main/qwen3-tts-custom-1.7b-f16.gguf"

// VoiceDesign model URLs (1.7B only)
#define VOCAL_TTS_DESIGN_MODEL_URL "https://huggingface.co/offbeatengineer/qwen3-tts-gguf/resolve/main/qwen3-tts-design-1.7b-f16.gguf"

// Exit codes
#define VOCAL_OK          0
#define VOCAL_ERR_ARGS    1
#define VOCAL_ERR_MODEL   2
#define VOCAL_ERR_AUDIO   3
#define VOCAL_ERR_IO      4
#define VOCAL_ERR_RUNTIME 5

#endif // VOCAL_COMMON_H
