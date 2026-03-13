# Vocal

Local voice toolkit — speech recognition, synthesis, and voice cloning. No cloud API, no Python, no Node.js. Single binary, runs on GPU.

Built on [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-0.6B) with [GGML](https://github.com/ggerganov/ggml) inference. Supports both 0.6B and 1.7B model sizes.

## Quick Start

```bash
# Build
make

# Download the ASR model (~1.8 GB)
./vocal download asr

# Transcribe audio
./vocal asr -f audio.wav
```

## Features

- **ASR**: Transcribe audio to text in 30+ languages (Qwen3-ASR, GPU-accelerated)
- **TTS**: Text-to-speech synthesis (Qwen3-TTS, GPU-accelerated)
- **Voice Cloning**: Clone voices from reference audio samples
- **HTTP Server**: Load models once, serve many requests over HTTP

## Requirements

- CMake 3.14+
- C17/C++17 compiler (Clang, GCC)
- Apple Silicon recommended (Metal GPU acceleration)

## Building

```bash
git clone --recursive https://github.com/user/vocal.git
cd vocal
make
```

If you already cloned without `--recursive`, init the submodules first:

```bash
git submodule update --init --recursive
```

For debug builds: `make debug`
For timing instrumentation: `make timing`

## Usage

### Speech Recognition (ASR)

```bash
# Basic transcription
./vocal asr -f audio.wav

# Use the 1.7B model for higher quality
./vocal asr --large -f audio.wav

# JSON output with timing
./vocal asr -f audio.wav --json

# Custom thread count
./vocal asr -f audio.wav -t 8

# Save to file
./vocal asr -f audio.wav -o transcript.txt
```

### Text-to-Speech (TTS)

```bash
# Download TTS models (~1 GB)
./vocal download tts

# Synthesize speech
./vocal tts -t "Hello world" -o output.wav

# Use the 1.7B model for higher quality
./vocal tts --large -t "Hello world" -o output.wav

# Read from stdin
echo "Hello world" | ./vocal tts --stdin -o output.wav
```

### Voice Cloning

```bash
# Download TTS models (voice cloning uses the same models)
./vocal download clone

# One-shot clone (encode + synthesize)
./vocal clone -f reference.wav --ref-text "What the speaker says in the reference" \
  -t "Text to synthesize in the cloned voice" -o output.wav

# Use the 1.7B model for higher quality
./vocal clone --large -f reference.wav --ref-text "What the speaker says" \
  -t "Text to synthesize" -o output.wav

# Save a voice profile for reuse (skips re-encoding each time)
./vocal clone -f reference.wav --ref-text "What the speaker says" --save myvoice

# Save with 1.7B model (voice profiles are model-size-specific)
./vocal clone --large -f reference.wav --ref-text "What the speaker says" --save myvoice-large

# Use a saved voice profile with TTS (match the model size used during save)
./vocal tts --voice myvoice -t "Text to synthesize" -o output.wav
./vocal tts --large --voice myvoice-large -t "Text to synthesize" -o output.wav

# List saved voice profiles
./vocal voices
```

### Model Management

```bash
# Download models (0.6B, default)
./vocal download asr      # ASR model (~1.8 GB)
./vocal download tts      # TTS model + tokenizer + decoder (~1 GB)
./vocal download clone    # TTS models (same as tts; encoders are embedded)

# Download 1.7B models
./vocal download asr-large      # ASR 1.7B (~4.4 GB)
./vocal download tts-large      # TTS 1.7B + tokenizer + decoder (~3.6 GB)
./vocal download clone-large    # TTS 1.7B models for voice cloning

# List downloaded models
./vocal models
```

### HTTP Server

Model loading takes 400ms–1.6s per invocation. The server mode loads models once at startup and serves many requests over HTTP, eliminating reload overhead.

```bash
# Start server with both models
./vocal serve --asr --tts --port 8080

# Or just ASR, with 1.7B model
./vocal serve --asr --large --port 8080
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check + loaded model status |
| `POST` | `/v1/asr/transcribe` | Transcribe audio (body: raw audio bytes) |
| `POST` | `/v1/tts/synthesize` | Synthesize speech (body: JSON) |
| `POST` | `/v1/tts/clone` | Voice cloning (body: multipart form) |
| `GET` | `/v1/voices` | List saved voice profiles |

```bash
# Health check
curl http://localhost:8080/health

# Transcribe audio
curl -X POST http://localhost:8080/v1/asr/transcribe \
  -H "Content-Type: audio/wav" --data-binary @audio.wav

# Synthesize speech (returns WAV)
curl -X POST http://localhost:8080/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","speed":1.0}' -o output.wav

# Clone a voice (multipart form)
curl -X POST http://localhost:8080/v1/tts/clone \
  -F audio=@ref.wav -F ref_text="Reference transcript" \
  -F text="Text to synthesize" -o cloned.wav
```

ASR and TTS can run concurrently (separate GGML backends), but concurrent requests of the same type are serialized via mutex. Timing info is returned in `X-Vocal-*` response headers for TTS endpoints.

### All Commands

```
vocal asr          Transcribe audio to text
vocal tts          Synthesize speech from text
vocal clone        Clone a voice from reference audio
vocal serve        Start HTTP server
vocal voices       List saved voice profiles
vocal download     Download models
vocal models       List downloaded models
vocal version      Print version
```

## Performance

On Apple Silicon with Metal GPU:

| Stage | Time (3s audio) |
|-------|-----------------|
| Mel spectrogram | ~2 ms |
| Audio encoding | ~50 ms |
| Text decoding | ~150 ms |
| **Total** | **~200 ms** |

## Quantization

The default models are F16 (half-precision). You can convert models to smaller quantized formats for reduced memory usage and faster loading.

### Quantized types

| Type | Size vs F16 | Notes |
|------|-------------|-------|
| `f16` | 1x (default) | Best quality |
| `q8_0` | ~0.5x | Minimal quality loss |
| `q4_k` | ~0.3x | TTS only, noticeable quality loss |

### Converting models

Requires Python with `gguf`, `torch`, `safetensors`, and `numpy`. Install them with:

```bash
pip install gguf torch safetensors numpy tqdm
```

**ASR** (supports `f16`, `f32`, `q8_0`):

```bash
# Download source model
hf download Qwen/Qwen3-ASR-0.6B --local-dir /tmp/Qwen3-ASR-0.6B

# Convert to Q8_0
python tools/convert_asr_to_gguf.py \
    -i /tmp/Qwen3-ASR-0.6B \
    -o ~/.vocal/models/qwen3-asr-0.6b-q8_0.gguf \
    -t q8_0

# Use it
./vocal asr -m ~/.vocal/models/qwen3-asr-0.6b-q8_0.gguf -f audio.wav
```

**TTS** (supports `f16`, `f32`, `q8_0`, `q4_k`):

```bash
# Download source model
hf download Qwen/Qwen3-TTS-12Hz-0.6B --local-dir /tmp/Qwen3-TTS-0.6B

# Convert to Q8_0
python tools/convert_tts_to_gguf.py \
    -i /tmp/Qwen3-TTS-0.6B \
    -o ~/.vocal/models/qwen3-tts-0.6b-q8_0.gguf \
    -t q8_0

# Use it
./vocal tts -m ~/.vocal/models/qwen3-tts-0.6b-q8_0.gguf -t "Hello world" -o output.wav
```

**1.7B models** work the same way — just substitute the model name:

```bash
hf download Qwen/Qwen3-ASR-1.7B --local-dir /tmp/Qwen3-ASR-1.7B
python tools/convert_asr_to_gguf.py -i /tmp/Qwen3-ASR-1.7B -o ~/.vocal/models/qwen3-asr-1.7b-q8_0.gguf -t q8_0

hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir /tmp/Qwen3-TTS-1.7B
python tools/convert_tts_to_gguf.py -i /tmp/Qwen3-TTS-1.7B -o ~/.vocal/models/qwen3-tts-1.7b-q8_0.gguf -t q8_0
```

The codec decoder (`qwen3-tts-tokenizer-f16.gguf`) and `tokenizer.json` are shared across all model sizes and don't need re-conversion — download them with `./vocal download tts`.

### Batch conversion

Convert all models at once:

```bash
python tools/convert_all.py -o output/
```

## Model Storage

Models are stored in `~/.vocal/models/` by default. Override with:
- `VOCAL_MODELS_DIR` environment variable
- `--model-dir` flag

## GPU Support

- **macOS**: Metal (automatic on Apple Silicon)
- **Linux/Windows**: CUDA (build with `VOCAL_CUDA=1 make`)
- **Fallback**: CPU with Accelerate (macOS) or OpenBLAS (Linux)

## Credits

- ASR engine based on [qwen3-asr.cpp](https://github.com/predict-woo/qwen3-asr.cpp) by predict-woo
- Audio I/O via [dr_wav](https://github.com/mackron/dr_libs) by David Reid
- HTTP server via [cpp-httplib](https://github.com/yhirose/cpp-httplib) by Yuji Hirose
- Tensor inference via [GGML](https://github.com/ggerganov/ggml) by Georgi Gerganov

## License

MIT
