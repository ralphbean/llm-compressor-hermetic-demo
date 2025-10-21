#!/usr/bin/env python3
"""
LLM Compressor Quantization Script

This script applies one-shot quantization to the TinyLlama model using
SmoothQuant and GPTQ algorithms, reducing it to 8-bit weights and activations.

The model and dataset are hardcoded and must match the dependencies declared
in huggingface.lock.yaml for hermetic builds.
"""
import os
import argparse
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

# Hardcoded model and dataset - must match huggingface.lock.yaml
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET = "open_platypus"


def main():
    parser = argparse.ArgumentParser(description="Quantize TinyLlama model")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/TinyLlama-1.1B-Chat-v1.0-INT8",
        help="Output directory for quantized model"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=512,
        help="Number of calibration samples"
    )

    args = parser.parse_args()

    print(f"Starting quantization of {MODEL}")
    print(f"Using dataset: {DATASET}")
    print(f"Output directory: {args.output_dir}")

    # Define quantization recipe
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
    ]

    # Apply one-shot quantization
    oneshot(
        model=MODEL,
        dataset=DATASET,
        recipe=recipe,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
    )

    print(f"\nQuantization complete! Model saved to {os.path.abspath(args.output_dir)}")

    # Print output directory contents
    print("\nOutput files:")
    for root, dirs, files in os.walk(args.output_dir):
        level = root.replace(args.output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:  # Limit to first 10 files
            print(f'{subindent}{os.path.abspath(os.path.join(root, file))}')
        if len(files) > 10:
            print(f'{subindent}... and {len(files) - 10} more files')


if __name__ == "__main__":
    main()
