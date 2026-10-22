#!/usr/bin/env python3
"""
04_qlora_finetune.py — QLoRA fine-tuning of LLaMA 3.2 (3B) on the Jetson.

The FP16 checkpoint is quantized to 4-bit NF4 (double quantization, FP16
compute) when it is loaded; LoRA adapters (r=64, alpha=16, dropout 0.1) are
trained on q_proj and v_proj with a paged 32-bit AdamW optimizer. The loss is
computed on the response tokens only.

The same script fine-tunes Qwen 2.5 (q_proj, v_proj) and Phi-3.5-mini
(--target_modules qkv_proj, its fused query/key/value projection).

Usage:
    python scripts/04_qlora_finetune.py \
        --model_dir languagemodels/Llama-3.2-3B-Instruct \
        --train_file data/finetune/train.jsonl \
        --output_dir languagemodels/fine_tuned_llama3_lora
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="languagemodels/Llama-3.2-3B-Instruct")
parser.add_argument("--train_file", default="data/finetune/train.jsonl")
parser.add_argument("--output_dir", default="languagemodels/fine_tuned_llama3_lora")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--grad_accum", type=int, default=2)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--target_modules", default="q_proj,v_proj",
                    help="comma-separated attention projections (Phi-3.5 has a fused qkv_proj)")
args = parser.parse_args()

# ------------------------------------------------------------------
# Tokenizer and 4-bit NF4 base model
# ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_dir,
    dtype=torch.float16,
    device_map={"": 0},
    quantization_config=bnb_config,
    local_files_only=True,
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_cfg = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=args.target_modules.split(","),
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ------------------------------------------------------------------
# Dataset: loss on the response only
# ------------------------------------------------------------------
with open(args.train_file) as f:
    raw = [json.loads(line) for line in f if line.strip()]


def preprocess(example):
    prompt_ids = tokenizer(PROMPT_TEMPLATE.format(instruction=example["instruction"]))["input_ids"]
    answer_ids = tokenizer(example["output"], add_special_tokens=False)["input_ids"]
    answer_ids = answer_ids + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + answer_ids)[: args.max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[: args.max_length]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


dataset = [preprocess(example) for example in raw]
print("Training rows:", len(dataset))

training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    num_train_epochs=args.epochs,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    optim="paged_adamw_32bit",
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
)

torch.cuda.reset_peak_memory_stats()
start = time.time()
result = trainer.train()
elapsed = time.time() - start

trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

adapter_bytes = sum(p.stat().st_size for p in Path(args.output_dir).glob("adapter_model*"))
metrics = {
    "train_rows": len(dataset),
    "epochs": args.epochs,
    "optimizer_steps": result.global_step,
    "train_loss": result.training_loss,
    "loss_log": [e for e in trainer.state.log_history if "loss" in e],
    "train_seconds": elapsed,
    "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
    "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
    "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "target_modules": args.target_modules,
    "batch_size": args.batch_size,
    "grad_accum": args.grad_accum,
    "adapter_mb": adapter_bytes / 2**20,
}
Path(args.output_dir, "train_metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps({k: v for k, v in metrics.items() if k != "loss_log"}, indent=2))
print(f"LoRA adapter saved to: {args.output_dir}")
