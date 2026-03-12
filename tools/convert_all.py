#!/usr/bin/env python3
"""
Convert official Qwen3 HuggingFace models to GGUF format for Vocal.

Produces all GGUF files + tokenizer.json from the official source models.
Downloads models automatically if not already cached locally.

Usage:
    # Convert all models (downloads ~15 GB on first run)
    python tools/convert_all.py -o output/

    # Convert only specific models
    python tools/convert_all.py -o output/ --asr-0.6b --tts-0.6b --codec --tokenizer

    # Use locally cached models instead of downloading
    python tools/convert_all.py -o output/ --cache-dir ~/.vocal

    # Dry run — show what would be done
    python tools/convert_all.py -o output/ --dry-run

Output files:
    qwen3-asr-0.6b-f16.gguf        (~1.8 GB)
    qwen3-asr-1.7b-f16.gguf        (~4.4 GB)
    qwen3-tts-0.6b-f16.gguf        (~1.8 GB)
    qwen3-tts-1.7b-f16.gguf        (~3.6 GB)
    qwen3-tts-tokenizer-f16.gguf   (~325 MB)
    tokenizer.json                  (~11 MB)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Official HuggingFace model repos
HF_REPOS = {
    "asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "tts-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "tts-1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

# Output filenames
OUTPUT_FILES = {
    "asr-0.6b": "qwen3-asr-0.6b-f16.gguf",
    "asr-1.7b": "qwen3-asr-1.7b-f16.gguf",
    "tts-0.6b": "qwen3-tts-0.6b-f16.gguf",
    "tts-1.7b": "qwen3-tts-1.7b-f16.gguf",
    "codec": "qwen3-tts-tokenizer-f16.gguf",
    "tokenizer": "tokenizer.json",
}

# Converter scripts (relative to repo root)
TOOLS_DIR = Path(__file__).parent

# Local directory names under cache-dir
CACHE_DIRS = {
    "asr-0.6b": "Qwen3-ASR-0.6B",
    "asr-1.7b": "Qwen3-ASR-1.7B",
    "tts-0.6b": "Qwen3-TTS-0.6B",
    "tts-1.7b": "Qwen3-TTS-1.7B",
}

ALL_TARGETS = ["asr-0.6b", "asr-1.7b", "tts-0.6b", "tts-1.7b", "codec", "tokenizer"]


def download_model(repo_id: str, local_dir: Path) -> None:
    """Download a HuggingFace model using the hf CLI."""
    if local_dir.exists() and any(local_dir.glob("*.safetensors")):
        logger.info(f"  Already cached: {local_dir}")
        return

    logger.info(f"  Downloading {repo_id} -> {local_dir}")
    cmd = ["hf", "download", repo_id, "--local-dir", str(local_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  Download failed: {result.stderr.strip()}")
        sys.exit(1)


def run_converter(script: str, input_dir: Path, output_path: Path, dtype: str = "f16") -> None:
    """Run a converter script."""
    cmd = [sys.executable, str(TOOLS_DIR / script), "-i", str(input_dir), "-o", str(output_path), "-t", dtype]
    logger.info(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  Converter failed:\n{result.stderr}")
        sys.exit(1)
    # Show summary lines
    for line in result.stderr.splitlines():
        if "Converted" in line or "complete" in line:
            logger.info(f"  {line.split(': ', 1)[-1]}")


def generate_tokenizer(input_dir: Path, output_path: Path) -> None:
    """Generate tokenizer.json from a Qwen3-TTS model directory."""
    logger.info(f"  Generating tokenizer.json from {input_dir}")
    cmd = [
        sys.executable, "-c",
        f"from transformers import AutoTokenizer; "
        f"t = AutoTokenizer.from_pretrained('{input_dir}'); "
        f"t.save_pretrained('{output_path.parent}'); "
        f"import shutil; shutil.move('{output_path.parent}/tokenizer.json', '{output_path}')"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  Tokenizer generation failed:\n{result.stderr}")
        sys.exit(1)


def resolve_cache_dir(cache_dir: Path, model_key: str) -> Path:
    """Resolve the local cache directory for a model."""
    return cache_dir / CACHE_DIRS[model_key]


def main():
    parser = argparse.ArgumentParser(
        description="Convert official Qwen3 models to GGUF format for Vocal"
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output directory for converted files",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path.home() / ".vocal",
        help="Directory for caching downloaded HF models (default: ~/.vocal)",
    )
    parser.add_argument(
        "--type", "-t", choices=["f16", "f32"], default="f16",
        help="Output data type (default: f16)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")

    # Target selection flags
    parser.add_argument("--asr-0.6b", action="store_true", help="Convert ASR 0.6B model")
    parser.add_argument("--asr-1.7b", action="store_true", help="Convert ASR 1.7B model")
    parser.add_argument("--tts-0.6b", action="store_true", help="Convert TTS 0.6B model")
    parser.add_argument("--tts-1.7b", action="store_true", help="Convert TTS 1.7B model")
    parser.add_argument("--codec", action="store_true", help="Convert codec (speech tokenizer)")
    parser.add_argument("--tokenizer", action="store_true", help="Generate tokenizer.json")

    args = parser.parse_args()

    # Determine targets — if none specified, convert all
    selected = {t for t in ALL_TARGETS if getattr(args, t.replace("-", "_").replace(".", ""), False)}
    if not selected:
        selected = set(ALL_TARGETS)

    output_dir = args.output
    cache_dir = args.cache_dir

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Cache directory:  {cache_dir}")
    logger.info(f"Targets: {', '.join(sorted(selected))}")

    if args.dry_run:
        for target in sorted(selected):
            out = output_dir / OUTPUT_FILES[target]
            if target in HF_REPOS:
                logger.info(f"  [{target}] Download {HF_REPOS[target]} -> {resolve_cache_dir(cache_dir, target)}")
            if target == "codec":
                logger.info(f"  [{target}] Convert codec from any TTS model -> {out}")
            elif target == "tokenizer":
                logger.info(f"  [{target}] Generate tokenizer.json -> {out}")
            else:
                logger.info(f"  [{target}] Convert -> {out}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download required models
    needed_downloads = set()
    for target in selected:
        if target in ("asr-0.6b", "asr-1.7b", "tts-0.6b", "tts-1.7b"):
            needed_downloads.add(target)
        elif target == "codec":
            # Codec comes from any TTS model (they share the same speech_tokenizer)
            needed_downloads.add("tts-0.6b")
        elif target == "tokenizer":
            needed_downloads.add("tts-0.6b")

    for model_key in sorted(needed_downloads):
        local = resolve_cache_dir(cache_dir, model_key)
        logger.info(f"[{model_key}] Ensuring model is available...")
        download_model(HF_REPOS[model_key], local)

    # Step 2: Convert
    for target in sorted(selected):
        out = output_dir / OUTPUT_FILES[target]
        if out.exists():
            logger.info(f"[{target}] Already exists: {out} (skipping)")
            continue

        logger.info(f"[{target}] Converting...")

        if target.startswith("asr-"):
            run_converter("convert_asr_to_gguf.py", resolve_cache_dir(cache_dir, target), out, args.type)
        elif target.startswith("tts-"):
            run_converter("convert_tts_to_gguf.py", resolve_cache_dir(cache_dir, target), out, args.type)
        elif target == "codec":
            run_converter("convert_codec_to_gguf.py", resolve_cache_dir(cache_dir, "tts-0.6b"), out, args.type)
        elif target == "tokenizer":
            generate_tokenizer(resolve_cache_dir(cache_dir, "tts-0.6b"), out)

        if out.exists():
            size_mb = out.stat().st_size / (1024 * 1024)
            logger.info(f"[{target}] Done: {out} ({size_mb:.0f} MB)")
        else:
            logger.error(f"[{target}] Output file not created!")
            sys.exit(1)

    logger.info("All conversions complete!")
    logger.info(f"Output files in {output_dir}:")
    for target in sorted(selected):
        out = output_dir / OUTPUT_FILES[target]
        if out.exists():
            size_mb = out.stat().st_size / (1024 * 1024)
            logger.info(f"  {out.name:40s} {size_mb:>8.0f} MB")


if __name__ == "__main__":
    main()
