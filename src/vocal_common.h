#ifndef VOCAL_COMMON_H
#define VOCAL_COMMON_H

#define VOCAL_VERSION "0.1.0"

// Default paths
#define VOCAL_DEFAULT_MODELS_DIR ".vocal/models"
#define VOCAL_DEFAULT_VOICES_DIR ".vocal/voices"

// Environment variables
#define VOCAL_ENV_MODELS_DIR "VOCAL_MODELS_DIR"

// Model identifiers
#define VOCAL_ASR_MODEL_NAME "qwen3-asr-0.6b-f16.gguf"
#define VOCAL_ASR_MODEL_LARGE_NAME "qwen3-asr-1.7b-f16.gguf"
#define VOCAL_TTS_MODEL_NAME "qwen3-tts-0.6b-f16.gguf"
#define VOCAL_TTS_TOKENIZER_NAME "tokenizer.json"
#define VOCAL_TTS_DECODER_NAME "qwen3-tts-tokenizer-f16.gguf"
#define VOCAL_TTS_ENCODER_NAME "qwen3_tts_codec_encoder.onnx"
// Speaker encoder is embedded in the main model GGUF (spk_enc.* tensors)
#define VOCAL_TTS_SPK_ENCODER_NAME VOCAL_TTS_MODEL_NAME

// HuggingFace URLs for GGUF models
#define VOCAL_ASR_MODEL_URL "https://huggingface.co/FlippyDora/qwen3-asr-0.6b-GGUF/resolve/main/qwen3-asr-0.6b-f16.gguf"
#define VOCAL_TTS_MODEL_URL "https://huggingface.co/Volko76/Qwen3-TTS-12Hz-0.6B-Base-Qwen3tts.cpp_quants-GGUF/resolve/main/qwen3-tts-0.6b-f16.gguf"
#define VOCAL_TTS_TOKENIZER_URL "https://huggingface.co/cgisky/qwen3-tts-custom-gguf/resolve/main/tokenizer/tokenizer.json"
#define VOCAL_TTS_DECODER_URL "https://huggingface.co/Volko76/Qwen3-TTS-12Hz-0.6B-Base-Qwen3tts.cpp_quants-GGUF/resolve/main/qwen3-tts-tokenizer-f16.gguf"
#define VOCAL_TTS_ENCODER_URL "https://huggingface.co/cgisky/qwen3-tts-custom-gguf/resolve/main/onnx/qwen3_tts_codec_encoder.onnx"
#define VOCAL_TTS_SPK_ENCODER_URL VOCAL_TTS_MODEL_URL

// Exit codes
#define VOCAL_OK          0
#define VOCAL_ERR_ARGS    1
#define VOCAL_ERR_MODEL   2
#define VOCAL_ERR_AUDIO   3
#define VOCAL_ERR_IO      4
#define VOCAL_ERR_RUNTIME 5

#endif // VOCAL_COMMON_H
