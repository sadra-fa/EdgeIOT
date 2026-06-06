# Implementation Plan – Fine‑tune LLaMA 3.2 (3 B) via Ollama & Profile‑driven Prompt Engineering

## Confirmed choices

| Item | Choice |
|------|--------|
| Base model | **LLaMA 3.2 (3 B)** |
| Connectivity | Jetson has internet **and** Ollama is already installed (we’ll pull the model through Ollama). |
| Quality metric | Manual spot‑check of **10 examples** now **and** in later iterations, plus an automated **BLEU ≥ 0.85** for every fine‑tuned version. |

---

## Overview of the workflow

1. Export the existing LLM reasoning outputs into a LoRA‑compatible JSONL file.
2. Pull the 3 B LLaMA model with the Ollama CLI.
3. Load the Hugging Face LLaMA 3.2 (3 B) checkpoint (e.g., `consolidated.00.pth`) directly with Transformers for QLoRA fine‑tuning.
4. Fine‑tune the model with **4‑bit QLoRA** on the exported corpus.
5. Convert the fine‑tuned checkpoint back to GGUF and register it with Ollama.
6. Swap the pipeline to use the new model and re‑run the full benchmark.
7. Profile‑driven prompt engineering – test three prompt styles, pick the best, and store the choice in `config.yaml`.
8. Validate latency, quality (BLEU ≥ 0.85, manual spot‑check), and GPU‑load handling.
9. Document results for the dissertation.

All steps that require **your manual action** are marked with a **🛠️**.

---

## Detailed step‑by‑step plan

### 1️⃣ Export the training corpus

| Action | Command / Code | Reason |
|--------|----------------|--------|
| **🛠️ 1.1** Export reasoning outputs to `train_data.jsonl` | ```bash
python - <<'PY'
import json, pathlib
src = pathlib.Path('outputs/semantic_outputs.json')
records = json.load(src)
out = pathlib.Path('train_data.jsonl')
with out.open('w') as f:

    for r in records:
        prompt = " ".join(r['semantic_tokens'])
        response = r['llm_reasoning']
        f.write(json.dumps({'instruction': prompt, 'output': response}) + "\n")
PY
``` | Creates a line‑wise LoRA training file (`instruction` → `output`). |
| **🛠️ 1.2** Verify a few lines | `head -n 5 train_data.jsonl` | Ensure each line is a valid JSON object. |

---

### 2️⃣ Download LLaMA 3.2 (3 B) from Hugging Face

| Action | Command |
|--------|---------|
| **🛠️ 2.1** Download the checkpoint (e.g., `consolidated.00.pth`) using the Hugging Face CLI or `git lfs` | `git lfs install && git clone https://huggingface.co/<org>/llama-3.2-3b && cd llama-3.2-3b && ls -lh` |
| **🛠️ 2.2** Verify the file exists (`ls -lh consolidated.00.pth`) | `ls -lh ./consolidated.00.pth` |

---

### 3️⃣ Load Hugging Face checkpoint (for LoRA)

| Action | Command | Explanation |
|--------|---------|-------------|
| **🛠️ 3.1** Install `transformers` ≥ 4.40 | `pip install --upgrade transformers` |
| **🛠️ 3.2** Load the checkpoint with `AutoModelForCausalLM` (4‑bit) | ```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path = "./consolidated.00.pth"
model = AutoModelForCausalLM.from_pretrained(

    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.save_pretrained("./base_llama3_4bit")
tokenizer.save_pretrained("./base_llama3_4bit")
print("Base 4‑bit checkpoint saved to ./base_llama3_4bit")

``` |
| **🛠️ 3.3** Verify the checkpoint exists (`./base_llama3_4bit/pytorch_model.bin`) | `ls -R ./base_llama3_4bit` |

---

### 4️⃣ Run 4‑bit QLoRA fine‑tuning

| Action | Command |
|--------|---------|
| **🛠️ 4.1** Install fine‑tuning tools (if not already) | `pip install peft accelerate bitsandbytes` |
| **🛠️ 4.2** Create a `qlora_finetune.py` script (copy the example from the **QLoRA** repo, adjusting paths). Below is a minimal version you can paste into `scripts/qlora_finetune.py`: | ```python
#!/usr/bin/env python3
import argparse, torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="./base_llama3_4bit")
parser.add_argument("--train_file", default="train_data.jsonl")
parser.add_argument("--output_dir", default="./fine_tuned_llama3_lora")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True,
)
model = prepare_model_for_int8_training(model)

lora_cfg = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

raw_dataset = load_dataset("json", data_files=args.train_file, split="train")

def preprocess(example):
    tokens = tokenizer(
        example["instruction"],
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    labels = tokenizer(
        example["output"],
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    tokens["labels"] = labels["input_ids"]
    return {k: v.squeeze() for k, v in tokens.items()}

dataset = raw_dataset.map(preprocess, remove_columns=["instruction", "output"])

targs = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    eval_strategy="no",
)

trainer = Trainer(model=model, args=targs, train_dataset=dataset)
trainer.train()
trainer.save_model(args.output_dir)
print("Fine‑tuned LoRA checkpoint saved to", args.output_dir)
``` |
| **🛠️ 4.3** Run the trainer | ```bash
accelerate launch scripts/qlora_finetune.py \
   --model_dir ./base_llama3_4bit \
   --train_file train_data.jsonl \
   --output_dir ./fine_tuned_llama3_lora \
   --epochs 3 \
   --batch_size 4
``` |
| **🛠️ 4.4** After training, check size (~200 MB) | `du -h -d0 ./fine_tuned_llama3_lora` |
| **🛠️ 4.5** Spot‑check 10 generated outputs (manual) | ```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('./fine_tuned_llama3_lora', device_map='auto', torch_dtype=torch.float16)
tok = AutoTokenizer.from_pretrained('./fine_tuned_llama3_lora')
samples = [
    "temperature≈23.5 humidity≈45.2 motion≈0",
    "noise≈67.1 light≈300.0",
    # add 8 more representative prompts …
]
for p in samples:
    inp = tok(p, return_tensors='pt').to('cuda')
    out = model.generate(**inp, max_new_tokens=100)
    print(tok.decode(out[0], skip_special_tokens=True))
``` |
| **🛠️ 4.6** Compute BLEU against the original LLM (script example) | ```python
import json, nltk
orig = json.load(open('outputs/semantic_outputs.json'))
orig_map = {" ".join(r['semantic_tokens']): r['llm_reasoning'] for r in orig}
# Assume `new_out` holds the 10 strings from 4.5 in the same order
scores = []
for p, new in zip(samples, new_out):
    ref = orig_map[p]
    scores.append(nltk.translate.bleu_score.sentence_bleu([ref.split()], new.split()))
print('Avg BLEU =', sum(scores)/len(scores))
``` |
| **🛠️ 4.7** Verify **BLEU ≥ 0.85**. If not, increase epochs or LoRA rank and repeat step 4. | – |

---

### 5️⃣ Convert the fine‑tuned LoRA model back to GGUF for Ollama

| Action | Command |
|--------|---------|
| **🛠️ 5.1** Install the Ollama conversion helper (if not present) | `pip install ollama` |
| **🛠️ 5.2** Convert to GGUF | ```bash
ollama convert \
   --model_dir ./fine_tuned_llama3_lora \
   --output_dir ./ollama_finetuned_llama3 \
   --format gguf
``` |
| **🛠️ 5.3** Register the new model with Ollama | `ollama create llama3_finetuned ./ollama_finetuned_llama3` |
| **🛠️ 5.4** Quick sanity test | `ollama run llama3_finetuned "Explain why temperature is high"` |

---

### 6️⃣ Swap the pipeline to the new model

| Action | Command |
|--------|---------|
| **🛠️ 6.1** Run the full pipeline with the fine‑tuned model | `python main.py --model llama3_finetuned` |
| **🛠️ 6.2** Compare the new `jetson_benchmark.csv` with the baseline (look for **≥ 2× latency reduction**). |
| **🛠️ 6.3** Confirm the GPU‑load rows now show real percentages (no `nan`). |

---

### 7️⃣ Profile‑driven Prompt Engineering

1. **Add the helper** (once) to `edgeiot/semantic.py` (already provided in the earlier plan). 
2. **Run the micro‑benchmark** on a 50‑sample slice for three styles (`raw`, `compact`, `hybrid`).

| Action | Command |
|--------|---------|
| **🛠️ 7.1** Create `profile_prompt.py` (sample script) | ```python
import time, pandas as pd
from edgeiot.semantic import build_prompt
from edgeiot.semantic import OllamaReasoner

# load a small test set (adjust path if needed)
X_test = pd.read_json('outputs/semantic_outputs.json')
reasoner = OllamaReasoner(cfg.ollama)  # cfg from your config file
styles = ['raw','compact','hybrid']
for style in styles:
    lat, toks = [], []
    for i,row in X_test.iterrows():
        prompt = build_prompt(row, style)
        start = time.time()
        out = reasoner.reason(prompt, label=0)  # dummy label
        lat.append((time.time()-start)*1000)
        toks.append(out['tokens_out'])
    print(style, 'avg latency', sum(lat)/len(lat), 'avg tokens', sum(toks)/len(toks))
``` |
| **🛠️ 7.2** Plot latency vs. tokens (optional) | Use `matplotlib` inside the script or export to CSV and plot later. |
| **🛠️ 7.3** Choose the best style (lowest latency with BLEU ≥ 0.85). |
| **🛠️ 7.4** Record the choice in `config.yaml` | ```yaml
semantic:
  prompt_style: hybrid   # (or compact / raw)
``` |

---

### 8️⃣ Final verification & documentation

| Action | What to do |
|--------|-----------|
| **8.1** Run the *full* pipeline one more time (with the chosen prompt style) and generate the final `jetson_benchmark.csv`. |
| **8.2** Verify the following numbers are present: <br>• **Avg latency** (should be ≤ baseline/2). <br>• **Avg GPU load** (real %). <br>• **BLEU** (≥ 0.85). |
| **8.3** Spot‑check 10 examples manually (print to console) and ensure the reasoning still makes sense for your smart‑home domain. |
| **8.4** Add the benchmark tables, prompt‑style decision, and fine‑tuning hyper‑parameters to your dissertation chapter *“Edge‑IoT LLM optimisation”*. |
| **8.5** Commit all new scripts (`train_corpus.py`, `qlora_finetune.py`, `profile_prompt.py`) and the updated `config.yaml` to the repository. |
| **8.6** (Optional) Push the fine‑tuned GGUF model to a private HuggingFace repo for reproducibility. |

---

## 📁 Artifact checklist (auto‑generated)

| Artifact | Path | Status |
|----------|------|--------|
| `implementation_plan.md` | `.../implementation_plan.md` | ✅ Created |
| `train_corpus.py` (export script) | `scripts/train_corpus.py` | 👉 To be added (copy the code from step 1.1). |
| `qlora_finetune.py` | `scripts/qlora_finetune.py` | 👉 To be added (code in step 4.2). |
| `profile_prompt.py` | `scripts/profile_prompt.py` | 👉 To be added (code in step 7.1). |
| Updated `edgeiot/semantic.py` (helper `build_prompt`) | `edgeiot/semantic.py` | 👉 Add the function from step 7.1. |
| Updated `config.yaml` (prompt style) | `edgeiot/config.yaml` | 👉 Edit after step 7.4. |

---

## ✅ What you need to do now
1. Run steps 1–5 to obtain a fine‑tuned LLaMA 3.2 model.
2. Spot‑check the 10 outputs and ensure BLEU ≥ 0.85.
3. Execute the prompt‑engineering script (step 7), pick the best style, and update `config.yaml`.
4. Run the full pipeline with the new model (`python main.py --model llama3_finetuned`).
5. Verify the benchmark CSV and record the numbers for your dissertation.

When you have completed a step, let me know if anything failed or if you need tweaks to the hyper‑parameters, and we can iterate.
