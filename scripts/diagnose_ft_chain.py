#!/usr/bin/env python3
"""
diagnose_ft_chain.py — Where along the QLoRA chain does a fine-tuned model change?

Generates answers (greedy) for the first N evaluation rows with the trained
adapter applied to the base model in two precisions:
  nf4  : the 4-bit NF4 base the adapter was trained against (as in training)
  fp16 : the original FP16 base (what a merge into the FP16 weights serves)
and reports schema validity and canonical actions for each.

Usage:
    python scripts/diagnose_ft_chain.py --base_dir languagemodels/Qwen2.5-1.5B-Instruct \
        --adapter_dir languagemodels/fine_tuned_qwen25_lora --n 20
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edgeiot.semantic.actions import check_reply                                   # noqa: E402
from edgeiot.semantic.llm_reasoner import build_finetuned_prompt, semantic_fragment, wrap_finetuned  # noqa: E402


def load(base_dir, adapter_dir, precision):
    kwargs = dict(dtype=torch.float16, device_map={"": 0}, local_files_only=True)
    if precision == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(base_dir, **kwargs)
    return PeftModel.from_pretrained(model, adapter_dir).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", default=None)
    p.add_argument("--precisions", default="nf4,fp16")
    args = p.parse_args()

    run = sorted((PROJECT_ROOT / "outputs" / "rerun").glob("*"))[-1]
    rows = pd.read_csv(run / "llm_inputs.csv").sample(n=args.n, random_state=42)  # same rows as bench_llm --n
    tok = AutoTokenizer.from_pretrained(args.base_dir, local_files_only=True)
    results = {}
    for precision in args.precisions.split(","):
        model = load(args.base_dir, args.adapter_dir, precision)
        answers = []
        for _, r in rows.iterrows():
            tokens = r["tokens"].split() if isinstance(r["tokens"], str) and r["tokens"] else []
            prompt = wrap_finetuned(build_finetuned_prompt(r["state"], semantic_fragment(tokens)))
            ids = tok(prompt, return_tensors="pt").to(0)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=96, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            reply = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            answers.append({**check_reply(reply), "reply": reply})
        df = pd.DataFrame(answers)
        results[precision] = {"schema_valid": float(df["schema_valid"].mean()),
                              "canonical": float(df["action_canonical"].mean()),
                              "actions": df["action"].value_counts().to_dict(),
                              "examples": df["reply"].head(3).tolist()}
        print(precision, json.dumps({k: v for k, v in results[precision].items() if k != "examples"}), flush=True)
        del model
        torch.cuda.empty_cache()
    out = Path(args.out) if args.out else Path(args.adapter_dir) / "chain_diagnosis.json"
    previous = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**previous, **results}, indent=2))


if __name__ == "__main__":
    main()
