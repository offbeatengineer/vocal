#!/usr/bin/env python3
"""
Convert Qwen3-TTS speech_tokenizer (Mimi codec) from HuggingFace SafeTensors to GGUF format.

The speech tokenizer lives in the `speech_tokenizer/` subdirectory of the full
Qwen3-TTS model repo (e.g. Qwen/Qwen3-TTS-12Hz-0.6B-Base).

Usage:
    # Download model
    huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir /tmp/Qwen3-TTS-0.6B

    # Convert
    python tools/convert_codec_to_gguf.py \\
        -i /tmp/Qwen3-TTS-0.6B \\
        -o qwen3-tts-tokenizer-f16.gguf \\
        -t f16
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

import gguf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Qwen3TTSCodecConverter:
    """Converter for Qwen3-TTS Mimi codec (speech_tokenizer) to GGUF format."""

    # ── Direct (non-indexed) tensor name mappings ──────────────────────

    TENSOR_MAP = {
        # Decoder VQ - first codebook projections
        "decoder.quantizer.rvq_first.input_proj.weight": "tok_dec.vq_first.input_proj.weight",
        "decoder.quantizer.rvq_first.output_proj.weight": "tok_dec.vq_first.output_proj.weight",
        # Decoder VQ - rest codebook projections
        "decoder.quantizer.rvq_rest.input_proj.weight": "tok_dec.vq_rest.input_proj.weight",
        "decoder.quantizer.rvq_rest.output_proj.weight": "tok_dec.vq_rest.output_proj.weight",
        # Decoder pre-conv
        "decoder.pre_conv.conv.weight": "tok_dec.pre_conv.weight",
        "decoder.pre_conv.conv.bias": "tok_dec.pre_conv.bias",
        # Decoder pre-transformer fixed tensors
        "decoder.pre_transformer.input_proj.weight": "tok_dec.pre_tfm.input_proj.weight",
        "decoder.pre_transformer.input_proj.bias": "tok_dec.pre_tfm.input_proj.bias",
        "decoder.pre_transformer.norm.weight": "tok_dec.pre_tfm.norm.weight",
        "decoder.pre_transformer.output_proj.weight": "tok_dec.pre_tfm.output_proj.weight",
        "decoder.pre_transformer.output_proj.bias": "tok_dec.pre_tfm.output_proj.bias",
        # Decoder blocks - first conv (index 0) and final snake + conv (indices 5, 6)
        "decoder.decoder.0.conv.weight": "tok_dec.dec.0.conv.weight",
        "decoder.decoder.0.conv.bias": "tok_dec.dec.0.conv.bias",
        "decoder.decoder.5.alpha": "tok_dec.dec.5.snake.alpha",
        "decoder.decoder.5.beta": "tok_dec.dec.5.snake.beta",
        "decoder.decoder.6.conv.weight": "tok_dec.dec.6.conv.weight",
        "decoder.decoder.6.conv.bias": "tok_dec.dec.6.conv.bias",
        # Encoder downsample
        "encoder.downsample.conv.weight": "tok_enc.downsample.weight",
    }

    # ── Decoder regex patterns ─────────────────────────────────────────

    DECODER_VQ_FIRST_PATTERNS = [
        (r"decoder\.quantizer\.rvq_first\.vq\.layers\.(\d+)\._codebook\.embedding_sum",
         "tok_dec.vq_first.{}.codebook"),
    ]

    DECODER_VQ_REST_PATTERNS = [
        (r"decoder\.quantizer\.rvq_rest\.vq\.layers\.(\d+)\._codebook\.embedding_sum",
         "tok_dec.vq_rest.{}.codebook"),
    ]

    DECODER_PRE_TFM_PATTERNS = [
        (r"decoder\.pre_transformer\.layers\.(\d+)\.input_layernorm\.weight",
         "tok_dec.pre_tfm.blk.{}.attn_norm.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn\.q_proj\.weight",
         "tok_dec.pre_tfm.blk.{}.attn_q.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn\.k_proj\.weight",
         "tok_dec.pre_tfm.blk.{}.attn_k.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn\.v_proj\.weight",
         "tok_dec.pre_tfm.blk.{}.attn_v.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn\.o_proj\.weight",
         "tok_dec.pre_tfm.blk.{}.attn_output.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn_layer_scale\.scale",
         "tok_dec.pre_tfm.blk.{}.attn_scale"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.post_attention_layernorm\.weight",
         "tok_dec.pre_tfm.blk.{}.ffn_norm.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.mlp\.gate_proj\.weight",
         "tok_dec.pre_tfm.blk.{}.ffn_gate.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.mlp\.up_proj\.weight",
         "tok_dec.pre_tfm.blk.{}.ffn_up.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.mlp\.down_proj\.weight",
         "tok_dec.pre_tfm.blk.{}.ffn_down.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.mlp_layer_scale\.scale",
         "tok_dec.pre_tfm.blk.{}.ffn_scale"),
    ]

    DECODER_UPSAMPLE_PATTERNS = [
        # Transpose conv
        (r"decoder\.upsample\.(\d+)\.0\.conv\.weight", "tok_dec.upsample.{}.conv.weight"),
        (r"decoder\.upsample\.(\d+)\.0\.conv\.bias", "tok_dec.upsample.{}.conv.bias"),
        # ConvNeXt dwconv
        (r"decoder\.upsample\.(\d+)\.1\.dwconv\.conv\.weight", "tok_dec.upsample.{}.dwconv.weight"),
        (r"decoder\.upsample\.(\d+)\.1\.dwconv\.conv\.bias", "tok_dec.upsample.{}.dwconv.bias"),
        # ConvNeXt norm
        (r"decoder\.upsample\.(\d+)\.1\.norm\.weight", "tok_dec.upsample.{}.norm.weight"),
        (r"decoder\.upsample\.(\d+)\.1\.norm\.bias", "tok_dec.upsample.{}.norm.bias"),
        # ConvNeXt pointwise convs
        (r"decoder\.upsample\.(\d+)\.1\.pwconv1\.weight", "tok_dec.upsample.{}.pwconv1.weight"),
        (r"decoder\.upsample\.(\d+)\.1\.pwconv1\.bias", "tok_dec.upsample.{}.pwconv1.bias"),
        (r"decoder\.upsample\.(\d+)\.1\.pwconv2\.weight", "tok_dec.upsample.{}.pwconv2.weight"),
        (r"decoder\.upsample\.(\d+)\.1\.pwconv2\.bias", "tok_dec.upsample.{}.pwconv2.bias"),
        # ConvNeXt gamma
        (r"decoder\.upsample\.(\d+)\.1\.gamma", "tok_dec.upsample.{}.gamma"),
    ]

    # Decoder blocks b=1..4: snake, transpose conv, residual sub-blocks
    DECODER_BLOCK_SNAKE_PATTERNS = [
        (r"decoder\.decoder\.([1-4])\.block\.0\.alpha", "tok_dec.dec.{}.snake.alpha"),
        (r"decoder\.decoder\.([1-4])\.block\.0\.beta", "tok_dec.dec.{}.snake.beta"),
    ]

    DECODER_BLOCK_CONV_T_PATTERNS = [
        (r"decoder\.decoder\.([1-4])\.block\.1\.conv\.weight", "tok_dec.dec.{}.conv_t.weight"),
        (r"decoder\.decoder\.([1-4])\.block\.1\.conv\.bias", "tok_dec.dec.{}.conv_t.bias"),
    ]

    DECODER_BLOCK_RES_PATTERNS = [
        (r"decoder\.decoder\.([1-4])\.block\.([2-4])\.act1\.alpha",
         "tok_dec.dec.{}.res.{}.act1.alpha"),
        (r"decoder\.decoder\.([1-4])\.block\.([2-4])\.act1\.beta",
         "tok_dec.dec.{}.res.{}.act1.beta"),
        (r"decoder\.decoder\.([1-4])\.block\.([2-4])\.conv1\.conv\.weight",
         "tok_dec.dec.{}.res.{}.conv1.weight"),
        (r"decoder\.decoder\.([1-4])\.block\.([2-4])\.conv1\.conv\.bias",
         "tok_dec.dec.{}.res.{}.conv1.bias"),
        (r"decoder\.decoder\.([1-4])\.block\.([2-4])\.act2\.alpha",
         "tok_dec.dec.{}.res.{}.act2.alpha"),
        (r"decoder\.decoder\.([1-4])\.block\.([2-4])\.act2\.beta",
         "tok_dec.dec.{}.res.{}.act2.beta"),
        (r"decoder\.decoder\.([1-4])\.block\.([2-4])\.conv2\.conv\.weight",
         "tok_dec.dec.{}.res.{}.conv2.weight"),
        (r"decoder\.decoder\.([1-4])\.block\.([2-4])\.conv2\.conv\.bias",
         "tok_dec.dec.{}.res.{}.conv2.bias"),
    ]

    # ── Encoder regex patterns ─────────────────────────────────────────

    ENCODER_CONV_PATTERNS = [
        (r"encoder\.encoder\.layers\.(\d+)\.conv\.weight", "tok_enc.conv.{}.weight"),
        (r"encoder\.encoder\.layers\.(\d+)\.conv\.bias", "tok_enc.conv.{}.bias"),
    ]

    ENCODER_RES_PATTERNS = [
        (r"encoder\.encoder\.layers\.(\d+)\.block\.(\d+)\.conv\.weight",
         "tok_enc.res.{}.blk.{}.weight"),
        (r"encoder\.encoder\.layers\.(\d+)\.block\.(\d+)\.conv\.bias",
         "tok_enc.res.{}.blk.{}.bias"),
    ]

    ENCODER_TFM_PATTERNS = [
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.input_layernorm\.weight",
         "tok_enc.blk.{}.attn_norm.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.input_layernorm\.bias",
         "tok_enc.blk.{}.attn_norm.bias"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn\.q_proj\.weight",
         "tok_enc.blk.{}.attn_q.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn\.k_proj\.weight",
         "tok_enc.blk.{}.attn_k.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn\.v_proj\.weight",
         "tok_enc.blk.{}.attn_v.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn\.o_proj\.weight",
         "tok_enc.blk.{}.attn_output.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn_layer_scale\.scale",
         "tok_enc.blk.{}.attn_scale"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.post_attention_layernorm\.weight",
         "tok_enc.blk.{}.ffn_norm.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.post_attention_layernorm\.bias",
         "tok_enc.blk.{}.ffn_norm.bias"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.mlp\.fc1\.weight",
         "tok_enc.blk.{}.ffn_up.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.mlp\.fc2\.weight",
         "tok_enc.blk.{}.ffn_down.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.mlp_layer_scale\.scale",
         "tok_enc.blk.{}.ffn_scale"),
    ]

    ENCODER_VQ_PROJ_MAP = {
        "encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight":
            "tok_enc.vq_semantic.input_proj.weight",
        "encoder.quantizer.semantic_residual_vector_quantizer.output_proj.weight":
            "tok_enc.vq_semantic.output_proj.weight",
        "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight":
            "tok_enc.vq_acoustic.input_proj.weight",
        "encoder.quantizer.acoustic_residual_vector_quantizer.output_proj.weight":
            "tok_enc.vq_acoustic.output_proj.weight",
    }

    ENCODER_VQ_CODEBOOK_PATTERNS = [
        (r"encoder\.quantizer\.semantic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.embed_sum",
         "tok_enc.vq_semantic.{}.codebook"),
        (r"encoder\.quantizer\.acoustic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.embed_sum",
         "tok_enc.vq_acoustic.{}.codebook"),
    ]

    # Maximum number of rvq_rest codebook layers to include
    MAX_RVQ_REST_LAYERS = 15

    def __init__(
        self,
        input_dir: Path,
        output_path: Path,
        output_type: str = "f16",
    ):
        self.input_dir = input_dir
        self.output_path = output_path
        self.output_type = output_type

        # Load config from speech_tokenizer subdirectory
        self.config = self._load_config()

        # Extract parameters
        self._extract_params()

    def _load_config(self) -> dict[str, Any]:
        """Load model configuration from speech_tokenizer/config.json."""
        config_path = self.input_dir / "speech_tokenizer" / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_params(self) -> None:
        """Extract model parameters from config."""
        dec_cfg = self.config.get("decoder_config", {})
        enc_cfg = self.config.get("encoder_config", {})

        self.sample_rate = self.config.get("input_sample_rate", 24000)
        self.num_codebooks = self.config.get("encoder_valid_num_quantizers", 16)
        self.codebook_size = dec_cfg.get("codebook_size", 2048)

        # Encoder architecture
        self.hidden_size = enc_cfg.get("hidden_size", 512)
        self.num_layers = enc_cfg.get("num_hidden_layers", 8)
        self.num_heads = enc_cfg.get("num_attention_heads", 8)
        self.codebook_dim = enc_cfg.get("codebook_dim", 256)
        self.num_quantizers = enc_cfg.get("num_quantizers", 32)

    def _should_skip(self, hf_name: str) -> bool:
        """Return True if this tensor should be skipped entirely."""
        # Skip boolean initialized flags
        if hf_name.endswith(".initialized"):
            return True
        # Skip cluster_usage — consumed during normalization, not emitted
        if hf_name.endswith(".cluster_usage"):
            return True
        return False

    def _map_tensor_name(self, hf_name: str) -> str | None:
        """Map HuggingFace tensor name to GGML convention."""
        # Check direct mapping first
        if hf_name in self.TENSOR_MAP:
            return self.TENSOR_MAP[hf_name]

        # Check encoder VQ projection direct map
        if hf_name in self.ENCODER_VQ_PROJ_MAP:
            return self.ENCODER_VQ_PROJ_MAP[hf_name]

        # ── Decoder patterns ──

        # VQ first codebook
        for pattern, template in self.DECODER_VQ_FIRST_PATTERNS:
            m = re.match(pattern, hf_name)
            if m:
                return template.format(m.group(1))

        # VQ rest codebook (only first MAX_RVQ_REST_LAYERS layers)
        for pattern, template in self.DECODER_VQ_REST_PATTERNS:
            m = re.match(pattern, hf_name)
            if m:
                idx = int(m.group(1))
                if idx >= self.MAX_RVQ_REST_LAYERS:
                    return None  # skip
                return template.format(m.group(1))

        # Pre-transformer layers
        for pattern, template in self.DECODER_PRE_TFM_PATTERNS:
            m = re.match(pattern, hf_name)
            if m:
                return template.format(m.group(1))

        # Upsample blocks
        for pattern, template in self.DECODER_UPSAMPLE_PATTERNS:
            m = re.match(pattern, hf_name)
            if m:
                return template.format(m.group(1))

        # Decoder block snake activation
        for pattern, template in self.DECODER_BLOCK_SNAKE_PATTERNS:
            m = re.match(pattern, hf_name)
            if m:
                return template.format(m.group(1))

        # Decoder block transpose conv
        for pattern, template in self.DECODER_BLOCK_CONV_T_PATTERNS:
            m = re.match(pattern, hf_name)
            if m:
                return template.format(m.group(1))

        # Decoder block residual sub-blocks (2-index patterns)
        for pattern, template in self.DECODER_BLOCK_RES_PATTERNS:
            m = re.match(pattern, hf_name)
            if m:
                return template.format(m.group(1), m.group(2))

        # ── Encoder patterns ──

        for pattern, template in self.ENCODER_CONV_PATTERNS:
            m = re.match(pattern, hf_name)
            if m:
                return template.format(m.group(1))

        for pattern, template in self.ENCODER_RES_PATTERNS:
            m = re.match(pattern, hf_name)
            if m:
                return template.format(m.group(1), m.group(2))

        for pattern, template in self.ENCODER_TFM_PATTERNS:
            m = re.match(pattern, hf_name)
            if m:
                return template.format(m.group(1))

        for pattern, template in self.ENCODER_VQ_CODEBOOK_PATTERNS:
            m = re.match(pattern, hf_name)
            if m:
                return template.format(m.group(1))

        return None

    def _get_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterate over all tensors from speech_tokenizer/model.safetensors."""
        st_dir = self.input_dir / "speech_tokenizer"
        safetensor_files = list(st_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(
                f"No safetensors files found in {st_dir}"
            )

        for sf_path in sorted(safetensor_files):
            logger.info(f"Loading tensors from {sf_path.name}")
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)

    def _convert_dtype(
        self, tensor: torch.Tensor, ggml_name: str
    ) -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
        """Convert tensor to appropriate dtype for GGUF.

        1D tensors (biases, norms, scales, snake alpha/beta) are always F32.
        2D/3D weight tensors use F16 or F32 based on --type.
        Conv1d 1x1 projections (shape [C,C,1]) are squeezed to 2D.
        """
        if tensor.dtype == torch.bfloat16:
            data = tensor.float().numpy()
        else:
            data = tensor.numpy()

        # Note: do NOT squeeze 1x1 conv projections — the C++ decoder expects
        # 3D tensors and accesses ne[2] for the output channel count.

        n_dims = len(data.shape)

        if n_dims == 3 and "weight" in ggml_name:
            logger.info(
                f"Conv1d weight {ggml_name}: shape {data.shape} "
                "[OC,IC,K] - GGUF will reverse to [K,IC,OC]"
            )

        # 1D tensors always F32
        if n_dims <= 1:
            return data.astype(np.float32), gguf.GGMLQuantizationType.F32

        if self.output_type == "f32":
            return data.astype(np.float32), gguf.GGMLQuantizationType.F32
        else:
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16

    def _add_metadata(self, writer: gguf.GGUFWriter) -> None:
        """Add model metadata to GGUF writer."""
        arch = "qwen3-tts-tokenizer"

        writer.add_name("Qwen3-TTS-Tokenizer")
        writer.add_type(gguf.GGUFType.MODEL)

        if self.output_type == "f32":
            ftype = gguf.LlamaFileType.ALL_F32
        else:
            ftype = gguf.LlamaFileType.MOSTLY_F16
        writer.add_file_type(ftype)

        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

        # Codec-level metadata
        writer.add_uint32("qwen3-tts.tokenizer.sample_rate", self.sample_rate)
        writer.add_uint32("qwen3-tts.tokenizer.num_codebooks", self.num_codebooks)
        writer.add_uint32("qwen3-tts.tokenizer.codebook_size", self.codebook_size)

        # Encoder architecture metadata
        writer.add_uint32(f"{arch}.encoder.hidden_size", self.hidden_size)
        writer.add_uint32(f"{arch}.encoder.num_layers", self.num_layers)
        writer.add_uint32(f"{arch}.encoder.num_heads", self.num_heads)
        writer.add_uint32(f"{arch}.encoder.codebook_dim", self.codebook_dim)
        writer.add_uint32(f"{arch}.codebook_size", self.codebook_size)
        writer.add_uint32(f"{arch}.encoder.num_quantizers", self.num_quantizers)
        writer.add_uint32(f"{arch}.encoder.valid_quantizers", self.num_codebooks)

        logger.info("Added model metadata")

    def convert(self) -> None:
        """Convert the speech tokenizer to GGUF format."""
        logger.info("Converting Qwen3-TTS speech tokenizer (Mimi codec) to GGUF")
        logger.info(f"Input: {self.input_dir}/speech_tokenizer/")
        logger.info(f"Output: {self.output_path}")
        logger.info(f"Output type: {self.output_type}")
        logger.info(
            f"Architecture: hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, num_heads={self.num_heads}, "
            f"codebook_dim={self.codebook_dim}, codebook_size={self.codebook_size}, "
            f"num_codebooks={self.num_codebooks}"
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        arch = "qwen3-tts-tokenizer"
        writer = gguf.GGUFWriter(path=None, arch=arch)

        self._add_metadata(writer)

        tensor_count = 0
        skipped_count = 0

        logger.info("Loading tensors...")
        all_tensors = list(self._get_tensors())

        # Build lookup for cluster_usage tensors (needed for codebook normalization)
        usage_map: dict[str, torch.Tensor] = {}
        for hf_name, tensor in all_tensors:
            if hf_name.endswith(".cluster_usage"):
                usage_map[hf_name] = tensor

        logger.info("Processing tensors...")
        for hf_name, tensor in tqdm(all_tensors, desc="Converting"):
            if self._should_skip(hf_name):
                logger.debug(f"  Skipping: {hf_name}")
                skipped_count += 1
                continue

            ggml_name = self._map_tensor_name(hf_name)

            if ggml_name is None:
                logger.warning(f"Skipping unmapped tensor: {hf_name}")
                skipped_count += 1
                continue

            # Normalize decoder codebook tensors: embedding_sum / cluster_usage
            # Encoder codebooks are stored raw (not normalized).
            if hf_name.endswith("._codebook.embedding_sum"):
                usage_key = hf_name.replace(".embedding_sum", ".cluster_usage")
                if usage_key in usage_map:
                    usage = usage_map[usage_key].float()
                    usage = torch.clamp(usage, min=1e-5)
                    tensor = tensor.float() / usage.unsqueeze(-1)
                    logger.info(f"  Normalized codebook: {hf_name}")

            data, dtype = self._convert_dtype(tensor, ggml_name)
            writer.add_tensor(ggml_name, data, raw_dtype=dtype)
            tensor_count += 1

            logger.debug(
                f"  {hf_name} -> {ggml_name} [{dtype.name}] {data.shape}"
            )

        logger.info(f"Converted {tensor_count} tensors, skipped {skipped_count}")

        logger.info(f"Writing GGUF file to {self.output_path}")
        writer.write_header_to_file(path=self.output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()

        logger.info("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS speech tokenizer (Mimi codec) to GGUF format"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to HuggingFace model directory (containing speech_tokenizer/ subdirectory)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output GGUF file path",
    )
    parser.add_argument(
        "--type", "-t",
        choices=["f16", "f32"],
        default="f16",
        help="Output data type (default: f16)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    converter = Qwen3TTSCodecConverter(
        input_dir=args.input,
        output_path=args.output,
        output_type=args.type,
    )
    converter.convert()


if __name__ == "__main__":
    main()
