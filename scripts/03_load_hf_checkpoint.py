#!/usr/bin/env python3
"""Load a Hugging Face LLaMA 3.2 checkpoint directory and create a 4‑bit quantized version.

# Updated usage example reflecting correct arguments
Usage:
    python scripts/load_hf_checkpoint.py \
        --model_dir ./languagemodels/Llama-3.2-3B \
        --output_dir ./base_llama3_4bit

The script loads the model with `AutoModelForCausalLM` using 4‑bit quantization via `BitsAndBytesConfig` and saves both the model and tokenizer to the specified output directory.
The script loads the model with `transformers.AutoModelForCausalLM` using 4‑bit
quantization (bitsandbytes) and saves both the model and tokenizer to the
specified output directory.
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    parser = argparse.ArgumentParser(description="Load HF checkpoint directory and save 4‑bit quantized model")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the directory containing the Hugging Face checkpoint (e.g., cloned repo with config.json and consolidated.00.pth)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./base_llama3_4bit",
        help="Directory to save the 4‑bit quantized model and tokenizer"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve the model directory to an absolute path
    model_dir = os.path.abspath(args.model_dir)
    # Load the model in 4‑bit mode
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        ),
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)

    # Save the quantized model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Base 4‑bit checkpoint saved to {args.output_dir}")

if __name__ == "__main__":
    main()
