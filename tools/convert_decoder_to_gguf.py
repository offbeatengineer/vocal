#!/usr/bin/env python3
"""Convert Qwen3-TTS ONNX decoder model to GGUF format.

Usage:
    python tools/convert_decoder_to_gguf.py \
        --onnx ~/.vocal/models/qwen3_tts_decoder.onnx \
        --output ~/.vocal/models/qwen3_tts_decoder.gguf
"""

import argparse
import numpy as np
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-TTS decoder ONNX to GGUF")
    parser.add_argument("--onnx", required=True, help="Path to ONNX decoder model")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    args = parser.parse_args()

    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        print("Error: pip install onnx", file=sys.stderr)
        sys.exit(1)

    try:
        import gguf
    except ImportError:
        print("Error: pip install gguf", file=sys.stderr)
        sys.exit(1)

    print(f"Loading ONNX model from {args.onnx}...")
    model = onnx.load(args.onnx)

    # --- Step 1: Build name mapping from ONNX graph ---
    # Trace which initializer feeds which named operation
    name_map = {}

    # Map opaque MatMul initializers via graph node outputs
    for node in model.graph.node:
        for inp in node.input:
            if not inp.startswith("onnx::"):
                continue

            # Extract semantic name from node output
            for out in node.output:
                # The output name encodes the path, e.g.:
                # /layers.0/self_attn/q_proj/MatMul_output_0
                # /decoder.1/block.0/Exp_output_0
                path = out

                if inp.startswith("onnx::MatMul_"):
                    if "/input_proj/MatMul" in path:
                        name_map[inp] = "decoder.pre_transformer.input_proj.weight"
                    elif "/output_proj" in path and "/norm/" in path:
                        # This is from /norm/Mul_1 -> output_proj
                        pass
                    elif "output_proj" in path:
                        name_map[inp] = "decoder.pre_transformer.output_proj.weight"
                    elif "/layers." in path and "/self_attn/" in path:
                        # Extract layer index and projection type
                        parts = path.split("/")
                        for p in parts:
                            if p.startswith("layers."):
                                layer_idx = p.split(".")[1]
                                break
                        if "q_proj" in path:
                            name_map[inp] = f"decoder.pre_transformer.layers.{layer_idx}.self_attn.q_proj.weight"
                        elif "k_proj" in path:
                            name_map[inp] = f"decoder.pre_transformer.layers.{layer_idx}.self_attn.k_proj.weight"
                        elif "v_proj" in path:
                            name_map[inp] = f"decoder.pre_transformer.layers.{layer_idx}.self_attn.v_proj.weight"
                        elif "o_proj" in path:
                            name_map[inp] = f"decoder.pre_transformer.layers.{layer_idx}.self_attn.o_proj.weight"
                    elif "/layers." in path and "/mlp/" in path:
                        parts = path.split("/")
                        for p in parts:
                            if p.startswith("layers."):
                                layer_idx = p.split(".")[1]
                                break
                        if "gate_proj" in path:
                            name_map[inp] = f"decoder.pre_transformer.layers.{layer_idx}.mlp.gate_proj.weight"
                        elif "up_proj" in path:
                            name_map[inp] = f"decoder.pre_transformer.layers.{layer_idx}.mlp.up_proj.weight"
                        elif "down_proj" in path:
                            name_map[inp] = f"decoder.pre_transformer.layers.{layer_idx}.mlp.down_proj.weight"
                    elif "/upsample." in path and "/pwconv" in path:
                        parts = path.split("/")
                        for p in parts:
                            if p.startswith("upsample."):
                                up_idx = p.split(".")[1]
                                break
                        if "pwconv1" in path:
                            name_map[inp] = f"decoder.upsample.{up_idx}.1.pwconv1.weight"
                        elif "pwconv2" in path:
                            name_map[inp] = f"decoder.upsample.{up_idx}.1.pwconv2.weight"

                elif inp.startswith("onnx::Exp_"):
                    if node.op_type == "Exp":
                        # Trace: /decoder.{blk}/block.{sub}/{act}/Exp{_1}_output_0
                        # or /decoder.5/Exp{_1}_output_0 (final)
                        parts = path.split("/")

                        # Find decoder block index
                        blk_idx = None
                        sub_idx = None
                        act_name = None
                        is_beta = "Exp_1" in path

                        for p in parts:
                            if p.startswith("decoder."):
                                blk_idx = p.split(".")[1]
                            elif p.startswith("block."):
                                sub_idx = p.split(".")[1]
                            elif p.startswith("act"):
                                act_name = p

                        if blk_idx is not None:
                            param = "beta" if is_beta else "alpha"
                            if sub_idx == "0" or sub_idx is None:
                                # Pre-TransConv SnakeBeta or final
                                name_map[inp] = f"decoder.decoder.{blk_idx}.snake.{param}"
                            elif sub_idx in ("2", "3", "4"):
                                # Residual unit SnakeBeta
                                act_idx = "1" if (act_name and "act1" in act_name) else "2"
                                name_map[inp] = f"decoder.decoder.{blk_idx}.block.{sub_idx}.snake{act_idx}.{param}"

    print(f"Mapped {len(name_map)} opaque tensors")

    # --- Step 2: Extract all initializer data ---
    tensors = {}
    for init in model.graph.initializer:
        data = numpy_helper.to_array(init).astype(np.float32)
        name = init.name

        # Use mapped name if available
        if name in name_map:
            name = name_map[name]
        # Otherwise keep original name (already properly named)

        # For Exp (SnakeBeta) tensors: squeeze batch dimension [1, C, 1] -> [C]
        if "snake" in name and data.ndim == 3:
            data = data.squeeze()

        tensors[name] = data

    print(f"Total tensors to write: {len(tensors)}")

    # --- Step 3: Write GGUF ---
    writer = gguf.GGUFWriter(args.output, "qwen3-tts-decoder")

    # Metadata
    writer.add_uint32("decoder.num_transformer_layers", 8)
    writer.add_uint32("decoder.transformer_dim", 512)
    writer.add_uint32("decoder.transformer_heads", 16)
    writer.add_uint32("decoder.transformer_head_dim", 64)
    writer.add_uint32("decoder.transformer_intermediate", 1024)
    writer.add_uint32("decoder.codebook_dim", 256)
    writer.add_uint32("decoder.codebook_size", 2048)
    writer.add_uint32("decoder.num_codebooks", 16)
    writer.add_uint32("decoder.decoder_dim", 1536)
    # Decoder block upsample rates: [8, 5, 4, 3]
    writer.add_array("decoder.upsample_rates", [8, 5, 4, 3])
    # Pre-decoder upsample ratios: [2, 2]
    writer.add_array("decoder.upsampling_ratios", [2, 2])
    writer.add_float32("decoder.rms_norm_eps", 1e-5)
    writer.add_float32("decoder.layer_scale_initial_scale", 0.01)

    # Write tensors
    for name, data in sorted(tensors.items()):
        writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Wrote {args.output} ({file_size:.1f} MB, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
