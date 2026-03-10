# Vocal

Local voice toolkit — speech recognition, synthesis, and voice cloning. No cloud API, no Python, no Node.js. Single binary, runs on GPU.

Built on [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-0.6B) with [GGML](https://github.com/ggerganov/ggml) inference.

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

## Requirements

- CMake 3.14+
- C17/C++17 compiler (Clang, GCC)
- Apple Silicon recommended (Metal GPU acceleration)

## Building

```bash
git clone --recursive https://github.com/user/vocal.git
cd vocal

# Install ONNX Runtime (macOS arm64)
mkdir -p third_party/onnxruntime
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-osx-arm64-1.24.3.tgz | \
  tar xz --strip-components=1 -C third_party/onnxruntime

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

# JSON output with timing
./vocal asr -f audio.wav --json

# Custom thread count
./vocal asr -f audio.wav -t 8

# Save to file
./vocal asr -f audio.wav -o transcript.txt

# Use a specific model file
./vocal asr -f audio.wav -m /path/to/model.gguf
```

### Text-to-Speech (TTS)

```bash
# Download TTS models (~1 GB)
./vocal download tts

# Synthesize speech
./vocal tts -t "Hello world" -o output.wav

# Read from stdin
echo "Hello world" | ./vocal tts --stdin -o output.wav
```

### Voice Cloning

```bash
# Download voice cloning models (~1.3 GB total)
./vocal download clone

# Clone a voice (with reference transcript for best quality)
./vocal clone -f reference.wav --ref-text "What the speaker says in the reference" \
  -t "Text to synthesize in the cloned voice" -o output.wav

# Clone without transcript (lower quality, uses default speaker style)
./vocal clone -f reference.wav -t "Text to synthesize" -o output.wav
```

### Model Management

```bash
# Download models
./vocal download asr      # ASR model (~1.8 GB)
./vocal download tts      # TTS model + tokenizer + decoder (~1 GB)
./vocal download clone    # All TTS models + codec encoder + speaker encoder (~1.3 GB)

# List downloaded models
./vocal models
```

### All Commands

```
vocal asr          Transcribe audio to text
vocal tts          Synthesize speech from text
vocal clone        Clone a voice from reference audio
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
- Tensor inference via [GGML](https://github.com/ggerganov/ggml) by Georgi Gerganov

## License

MIT
