#!/usr/bin/env python3

import argparse
import os
import torch
import torch.nn as nn

from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)

# ------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_dir",
    default="./languagemodels/llama-3.2-3B-think-fp16-fused-v6"
)

parser.add_argument(
    "--train_file_dir",
    help="Directory containing training JSONL file"
)

parser.add_argument(
    "--output_dir",
    default="./fine_tuned_llama3_lora"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=3
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=4
)

args = parser.parse_args()

# ------------------------------------------------------------------
# Resolve training dataset
# ------------------------------------------------------------------

if not args.train_file_dir:
    args.train_file_dir = input(
        "Please enter the directory containing the training JSONL file: "
    ).strip()

train_dir = os.path.abspath(args.train_file_dir)

if not os.path.isdir(train_dir):
    raise FileNotFoundError(
        f"Training directory not found: {train_dir}"
    )

jsonl_files = [
    f for f in os.listdir(train_dir)
    if f.lower().endswith(".jsonl")
]

if not jsonl_files:
    raise FileNotFoundError(
        f"No .jsonl file found in: {train_dir}"
    )

train_path = os.path.join(train_dir, jsonl_files[0])

print(f"Training file: {train_path}")

# ------------------------------------------------------------------
# Resolve model directory
# ------------------------------------------------------------------

model_path = os.path.abspath(args.model_dir)

if not os.path.isdir(model_path):
    repo_model_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            args.model_dir
        )
    )

    if os.path.isdir(repo_model_path):
        model_path = repo_model_path
    else:
        raise FileNotFoundError(
            f"Model directory not found:\n"
            f"  {model_path}\n"
            f"  {repo_model_path}"
        )

print(f"Model path: {model_path}")

# ------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)

# Llama usually has no pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

# NOTE:
# load_in_4bit=False = LoRA training
# load_in_4bit=True  = QLoRA training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    llm_int8_enable_fp32_cpu_offload=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True,
    local_files_only=True,
    offload_folder="offload",
    quantization_config=bnb_config,
)

# Safety for custom models
if not hasattr(model, "lm_head"):
    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size

    model.lm_head = nn.Linear(
        hidden_size,
        vocab_size,
        bias=False,
        dtype=torch.float16,
    )

if model.lm_head.weight.dtype != torch.float16:
    model.lm_head = model.lm_head.to(torch.float16)

# Required before LoRA
model = prepare_model_for_kbit_training(model)

# ------------------------------------------------------------------
# LoRA
# ------------------------------------------------------------------

lora_cfg = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_cfg)

model.print_trainable_parameters()

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

raw_dataset = load_dataset(
    "json",
    data_files=train_path,
    split="train",
)

print("Loaded rows:", len(raw_dataset))


def valid_record(example):
    instruction = example.get("instruction")
    output = example.get("output")

    return (
        isinstance(instruction, str)
        and instruction.strip()
        and isinstance(output, str)
        and output.strip()
    )


raw_dataset = raw_dataset.filter(valid_record)

print("Valid rows:", len(raw_dataset))

# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------

def preprocess(example):
    text = (
        f"### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Response:\n"
        f"{example['output']}"
    )

    enc = tokenizer(
        text,
        truncation=True,
        max_length=512,
    )

    enc["labels"] = enc["input_ids"].copy()

    return enc


dataset = raw_dataset.map(
    preprocess,
    remove_columns=raw_dataset.column_names,
)

print("Processed rows:", len(dataset))

# ------------------------------------------------------------------
# Dynamic padding collator
# ------------------------------------------------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ------------------------------------------------------------------
# Training arguments
# ------------------------------------------------------------------

use_fp16 = torch.cuda.is_available()

training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    learning_rate=2e-4,

    logging_steps=10,
    save_steps=100,

    eval_strategy="no",

    fp16=use_fp16,

    report_to="none",

    save_total_limit=2,

    remove_unused_columns=False,
)

# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------

print("Starting training...")
trainer.train()

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print()
print("=" * 60)
print("Training completed.")
print(f"LoRA adapter saved to: {args.output_dir}")
print("=" * 60)