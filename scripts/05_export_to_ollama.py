#!/usr/bin/env python3
"""
05_export_to_ollama.py — Serve the fine-tuned model through Ollama.

The fine-tuned model is served so that it can be measured with the same harness
(scripts/bench_llm.py) as the base models:

  edgeiot-llama3.2-ft-merged  : LoRA merged into the dequantized NF4 weights it was
                                trained against (or, with --merge_base fp16, into the
                                original FP16 weights), converted to GGUF and quantized
                                with llama.cpp to the base model's format, then imported
                                by Ollama

With --with-adapter, a second variant FROM llama3.2:3b + ADAPTER (LoRA applied at
run time) is also created. Ollama 0.35.1 rejects LoRA adapters ("LoRA adapters are
no longer supported"), so this variant only works with older Ollama versions.

The Modelfile sets the instruction template the adapter was trained with.

Usage:
    python scripts/05_export_to_ollama.py \
        --base_dir languagemodels/Llama-3.2-3B-Instruct \
        --adapter_dir languagemodels/fine_tuned_llama3_lora
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEMPLATE = '''TEMPLATE """### Instruction:
{{ .Prompt }}

### Response:
"""
PARAMETER stop "<|eot_id|>"
PARAMETER stop "### Instruction:"
'''


def load_dequantized_nf4(base_dir: str):
    """Load the base model in the NF4 format used for training and replace every
    4-bit linear layer by an FP16 layer holding the *dequantized* NF4 weights.

    The adapter was trained against these weights, not against the original FP16
    weights; merging into the original weights changes the model's behaviour (see
    scripts/diagnose_ft_chain.py). Layers are moved to the CPU one by one, so that
    models whose FP16 weights do not fit in GPU memory can be merged as well.
    """
    import bitsandbytes as bnb
    import torch
    from bitsandbytes.functional import dequantize_4bit
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model = AutoModelForCausalLM.from_pretrained(
        base_dir, dtype=torch.float16, device_map={"": 0}, local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
    )
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, bnb.nn.Linear4bit):
                weight = dequantize_4bit(child.weight.data, child.weight.quant_state).to(torch.float16)
                layer = torch.nn.Linear(child.in_features, child.out_features,
                                        bias=child.bias is not None, dtype=torch.float16, device="cpu")
                layer.weight.data = weight.cpu()
                if child.bias is not None:
                    layer.bias.data = child.bias.data.to(torch.float16).cpu()
                setattr(parent, name, layer)
                del child, weight
        torch.cuda.empty_cache()
    model = model.to("cpu")
    for attr in ("quantization_config",):
        if hasattr(model.config, attr):
            delattr(model.config, attr)
    model.is_quantized = False
    model.hf_quantizer = None
    torch.cuda.empty_cache()
    return model


def create(name: str, modelfile: Path, quantize: str = None):
    cmd = ["ollama", "create", name, "-f", str(modelfile)]
    if quantize:
        cmd += ["--quantize", quantize]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default="languagemodels/Llama-3.2-3B-Instruct")
    p.add_argument("--adapter_dir", default="languagemodels/fine_tuned_llama3_lora")
    p.add_argument("--merged_dir", default="languagemodels/fine_tuned_llama3_merged")
    p.add_argument("--llama_cpp", default="tools/llama.cpp",
                   help="llama.cpp checkout with build/bin/llama-quantize")
    p.add_argument("--name", default="edgeiot-llama3.2-ft-merged", help="Ollama model name")
    p.add_argument("--qtype", default="Q4_K_M", help="llama.cpp quantization type, or F16 for no quantization")
    p.add_argument("--base_tag", default="llama3.2:3b", help="Ollama base model (adapter variant only)")
    p.add_argument("--merge_base", choices=["nf4", "fp16"], default="nf4",
                   help="merge into the dequantized NF4 weights the adapter was trained on "
                        "(default) or into the original FP16 weights")
    p.add_argument("--with-adapter", action="store_true",
                   help="also create the run-time adapter variant (older Ollama only)")
    args = p.parse_args()

    adapter_dir = (PROJECT_ROOT / args.adapter_dir).resolve()
    out = PROJECT_ROOT / "languagemodels" / "modelfiles"
    out.mkdir(parents=True, exist_ok=True)

    if args.with_adapter:
        mf = out / "Modelfile.ft-adapter"
        mf.write_text(f"FROM {args.base_tag}\nADAPTER {adapter_dir}\n" + TEMPLATE)
        create(args.name.replace("-merged", "-adapter"), mf)

    # 1. Merge the adapter into the FP16 checkpoint (skipped if already done)
    merged_dir = (PROJECT_ROOT / args.merged_dir).resolve()
    if not (merged_dir / "model.safetensors").exists():
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if args.merge_base == "nf4":
            base = load_dequantized_nf4(args.base_dir)
        else:
            base = AutoModelForCausalLM.from_pretrained(
                args.base_dir, dtype=torch.float16, device_map={"": "cpu"},
                low_cpu_mem_usage=True, local_files_only=True,
            )
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        AutoTokenizer.from_pretrained(args.base_dir, local_files_only=True).save_pretrained(merged_dir)
        del base, merged

    # Tokenizer files that save_pretrained does not write (e.g. Phi-3.5's SentencePiece
    # tokenizer.model) are needed by the GGUF converter: copy them from the base model.
    import shutil
    for extra in ("tokenizer.model", "added_tokens.json", "special_tokens_map.json"):
        src, dst = Path(args.base_dir) / extra, merged_dir / extra
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)

    # 2. Convert to GGUF (FP16) and quantize with llama.cpp to the base model's format.
    #    Ollama 0.35.1 only offers int4/int8/fp4/fp8 when it quantizes a safetensors
    #    import, so the GGUF is produced here.
    llama_cpp = (PROJECT_ROOT / args.llama_cpp).resolve()
    stem = args.name.replace("-merged", "")
    f16 = PROJECT_ROOT / "languagemodels" / f"{stem}-f16.gguf"
    unquantized = args.qtype.upper() in ("F16", "FP16")
    q4 = f16 if unquantized else PROJECT_ROOT / "languagemodels" / f"{stem}-{args.qtype}.gguf"
    if not f16.exists():
        subprocess.run([sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(merged_dir),
                        "--outtype", "f16", "--outfile", str(f16)], check=True)
    if not unquantized and not q4.exists():
        subprocess.run([str(llama_cpp / "build" / "bin" / "llama-quantize"), str(f16), str(q4), args.qtype],
                       check=True)

    # 3. Import the quantized GGUF into Ollama with the training template
    mf = out / f"Modelfile.{stem}"
    mf.write_text(f"FROM {q4}\n" + TEMPLATE)
    create(args.name, mf)

if __name__ == "__main__":
    main()
