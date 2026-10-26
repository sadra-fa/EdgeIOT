# Smart Home Edge AI Pipeline (EdgeIOT)

An edge AI pipeline for smart-home sensor data, developed for the MESW dissertation *Edge LLM Optimization* (FEUP). It preprocesses environmental readings, assigns a behavioural state, flags anomalies, translates the readings into semantic tokens, and asks a local Large Language Model (via Ollama) for one of seven smart-home actions in JSON – all running on-device on an **NVIDIA Jetson Orin Nano**. A rule-based baseline, a response cache and QLoRA fine-tuned models that run on the Jetson are included.

**Dataset.** `RealLife.csv` was downloaded from Kaggle (source page: https://data.mendeley.com/datasets/ckwvzf7pzt/1). Its content is synthetic (no timestamps, physically impossible values, mostly non-home locations) and its rows are independent samples, not a time series; it is used as a proxy stream of sensor readings.

---

## Features

1. **Data Processing & Normalisation**
   - Ingests raw smart home sensor metrics (temperature, humidity, noise_db, light_intensity, motion_level).
   - Handles missing values, deduplication, and one-hot encoding for categorical contexts (`location`, `behaviour`).
   - Standardises numerical features using `StandardScaler`.

2. **Clustering & Classification**
   - Applies PCA and Hierarchical Clustering (Ward linkage) to generate three pseudo-labels for distinct home behaviours (e.g., active, sleeping, away).
   - Trains a Decision Tree, a Random Forest and an XGBoost classifier on the pseudo-labels; XGBoost (with 5-fold cross-validation and SHAP values) feeds the downstream stages.
   - Generates confusion matrices, ROC curves, feature importance and model comparison plots.
   - `scripts/leakage_check.py` measures how much of the accuracy comes from the one-hot `behaviour` column.

3. **Window-Based Anomaly Detection**
   - Builds sliding windows of 5 consecutive rows (in stored order, since the dataset is not a true time series).
   - Trains an Isolation Forest (5 % contamination) on the windows; a row whose window is abnormal gets the `[ANOMALY_ALERT]` token.

4. **Semantic Tokenisation**
   - Translates raw continuous sensor values into discrete semantic tokens (e.g., `[TEMP_HIGH]`, `[NOISE_SPIKE]`, `[LOW_LIGHT]`).
   - Merges anomaly statuses into the token stream for downstream reasoning.
   - Writes the tokenized test rows to `llm_inputs.csv`, the input of all LLM experiments.

5. **Local LLM Reasoning (Ollama)**
   - Builds a *hybrid* prompt (preamble + `State: [STATE_…]` + observed events + instruction) and asks for `{"reasoning": …, "action_json": {"action": …}}`.
   - `edgeiot/semantic/actions.py` checks each reply against that schema (Action Syntactic Validity Rate) and maps it onto the canonical actions.
   - `main.py` discovers the local Ollama models, lets you pick one interactively, and runs a smoke test, a latency/throughput benchmark and a raw-vs-semantic token-efficiency comparison.

6. **Rule Baseline & Response Cache**
   - `rule_action` (`edgeiot/semantic/actions.py`) is a deterministic if-else controller over the closed token vocabulary (safety > comfort > lighting) that the LLM is compared with.
   - `ResponseCache` (`edgeiot/semantic/cache.py`) stores the decision for each (state, tokens) input the first time it is seen and reuses it afterwards; new entries are validated against the rule baseline (policy `rules`) or only flagged for review (policy `flag`).

7. **Instrumented LLM Benchmark (`scripts/bench_llm.py`)**
   - Logs Ollama's timing fields, wall-clock latency, the schema check of every reply, and `tegrastats` samples (GPU load, RAM, temperature, board power).
   - Experiments: `multi` (several models, same samples), `tokenization` (raw vs semantic input), `probe` (hand-written cases from `scripts/probe_cases.py`, LLM vs rule baseline) and `finetune` (base vs fine-tuned model).

8. **QLoRA Fine-Tuning on the Jetson**
   - Self-distilled training corpora: each base model answers the distinct training inputs, and only schema-valid replies are kept.
   - QLoRA (4-bit NF4, LoRA r=64) of LLaMA 3.2 3B, Qwen 2.5 1.5B and Phi-3.5-mini, trained on the Jetson itself.
   - Export to Ollama: adapter merged into the dequantized NF4 weights, converted to GGUF and quantized with llama.cpp.

9. **Smart-Home Brain Module & Demos**
   - `edgeiot/brain.py` accepts a sensor-data dict, builds a natural-language prompt (three styles: `hybrid`, `compact`, `raw`), queries the local Ollama REST API, and returns the raw reply, a parsed JSON action, and round-trip latency.
   - `scripts/run_demo.py` runs a fixed number of simulated sensor → LLM → CSV cycles; `scripts/stream_demo.py` runs an infinite stream until `Ctrl-C`.

10. **Rich Visualisations**
    - Automatically generates a complete suite of plots per `main.py` run, organised by pipeline stage: data exploration, clustering, classification, anomaly detection, tokenisation, and LLM benchmarking.

---

## Project Structure

```text
EdgeIOT/
├── edgeiot/
│   ├── __init__.py
│   ├── brain.py                  # Smart-home "brain" for the demos: prompt → LLM → parsed action
│   ├── config.py                 # Pipeline configuration and OllamaConfig dataclass
│   ├── data/                     # Data loading, cleaning, preprocessing, hierarchical clustering
│   ├── models/                   # Classifiers (DT, RF, XGBoost), evaluator, anomaly detection
│   ├── semantic/
│   │   ├── tokenization.py       # Semantic tokeniser and output records
│   │   ├── llm_reasoner.py       # Ollama model discovery, prompts, OllamaReasoner
│   │   ├── actions.py            # Canonical actions, JSON schema check, rule baseline
│   │   └── cache.py              # Response cache over the semantic vocabulary
│   └── viz/                      # Centralised plotting using Matplotlib
├── scripts/
│   ├── simulate_sensors.py       # Simulated sensor reading generator
│   ├── run_demo.py               # Batch baseline demo (N cycles → CSV)
│   ├── stream_demo.py            # Continuous streaming demo (infinite loop → CSV)
│   ├── bench_llm.py              # Instrumented LLM experiments (multi, tokenization, probe, finetune)
│   ├── probe_cases.py            # Hand-written probe cases (policy, unseen, freetext)
│   ├── leakage_check.py          # Accuracy with/without the behaviour column
│   ├── cache_replay.py           # Response cache replay over the test rows
│   ├── 01_download_model_from_hf.py  # Interactive Hugging Face model search & download
│   ├── 02_make_train_data.py     # Self-distilled fine-tuning corpus
│   ├── 03_load_hf_checkpoint.py  # Standalone 4-bit quantisation of a checkpoint (not used by the workflow)
│   ├── 04_qlora_finetune.py      # QLoRA training
│   ├── 05_export_to_ollama.py    # Merge, GGUF conversion, quantisation, Ollama import
│   ├── diagnose_ft_chain.py      # Fine-tuned answers in NF4 vs FP16 precision
│   ├── run_experiments.sh        # E1–E3 and corpus building
│   ├── run_finetune.sh           # E4 training of the three models
│   ├── run_ft_chain_all.sh       # E4 export, evaluation and quantisation-chain table
│   ├── analyze_results.py        # Tables, figures and numbers for the dissertation
│   ├── thesis_figures.py         # Print versions of the pipeline figures
│   └── generate_slides.py        # Presentation slides (.pptx)
├── data/finetune*/               # Training corpora (train.jsonl, teacher_log.csv, corpus_summary.json)
├── languagemodels/               # Hugging Face checkpoints, LoRA adapters, merged models (not tracked)
├── main.py                       # Primary entrypoint for the full ML pipeline
├── RealLife.csv                  # Dataset
├── diagrams.md                   # Architecture diagrams
├── requirements.txt              # Pipeline and benchmark dependencies
├── requirements-finetune.txt     # QLoRA and export dependencies
└── README.md
```

---

## Requirements

- Python 3.12 (results were produced with 3.12.3 on JetPack 7.2.1 / L4T R39.2.1, `MAXN_SUPER`)
- Pinned dependencies (exact versions used for all dissertation results):

  ```bash
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt              # pipeline and benchmark
  .venv/bin/pip install -r requirements-finetune.txt     # QLoRA (see comments in the file)
  ```

- [Ollama](https://ollama.com/) must be installed and running locally with at least one model pulled:

  ```bash
  ollama pull llama3.2:3b
  ollama serve          # keep running in a separate terminal
  ```

- For the export of fine-tuned models: a llama.cpp checkout (commit `c0fd786`) built in `tools/llama.cpp`, and the Hugging Face CLI (`hf`) for downloading checkpoints.

---

## Usage

### Full ML Pipeline

```bash
python main.py
```

#### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data <path>` | Path to the CSV dataset | `RealLife.csv` |
| `--model <name>` | Skip the interactive selector and use this Ollama model | *(interactive)* |
| `--skip-llm` | Run the ML pipeline but skip LLM reasoning | `False` |
| `--outdir <path>` | Custom base output directory | `outputs/` |

When `--model` and `--skip-llm` are both omitted, the script will interactively list available Ollama models and ask how many samples to process.

---

### LLM Experiments

```bash
python scripts/bench_llm.py multi --inputs outputs/rerun/<run>/llm_inputs.csv \
    --models llama3.2:3b phi3.5:latest qwen2.5:1.5b
python scripts/bench_llm.py tokenization --inputs outputs/rerun/<run>/llm_inputs.csv --models llama3.2:3b
python scripts/bench_llm.py probe --models llama3.2:3b qwen2.5:1.5b
```

| Argument | Description | Default |
|----------|-------------|---------|
| `experiment` | `multi`, `tokenization`, `probe` or `finetune` | required |
| `--inputs` | `llm_inputs.csv` written by `main.py` | – |
| `--models` | One or more Ollama model names | required |
| `--n` | Number of test samples | `100` |
| `--outdir` | Base directory for the results | `outputs/experiments` |
| `--idle-seconds` | Idle `tegrastats` reading before the run | `15` |
| `--tag` | Suffix of the experiment directory name | – |

Each run writes `outputs/experiments/<experiment>_<timestamp>/` with `requests.csv` (one row per request), `models.csv` (memory, cold start) and `meta.json` (options, idle readings, versions).

---

### Sensor Simulator

```bash
# Print 1 sensor reading (default)
python scripts/simulate_sensors.py

# Print 5 readings
python scripts/simulate_sensors.py -n 5

# Reproducible single reading (fixed random seed)
python scripts/simulate_sensors.py -n 1 --seed 42
```

| Argument | Description | Default |
|----------|-------------|---------|
| `-n`, `--num` | Number of readings to emit | `1` |
| `--seed` | Random seed for reproducibility | `None` |

Each reading contains: `timestamp` (UTC ISO-8601), `temperature` (°C, −50 … 65), `motion` (bool), `noise` (dB, 25 … 185).

---

### Batch Baseline Demo

Runs a **fixed number** of sensor → LLM → CSV cycles. Results are appended to a single persistent file (`outputs/baseline_demo.csv`) across all runs.

```bash
# 1 reading (default)
python scripts/run_demo.py

# 5 readings, batch mode
python scripts/run_demo.py -n 5

# 5 readings with 2-second intervals
python scripts/run_demo.py -n 5 -i 2

# Use the compact prompt style
python scripts/run_demo.py -n 3 --style compact

# Use a different model and custom output directory
python scripts/run_demo.py -n 10 --model phi3.5 --outdir results/
```

| Argument | Description | Default |
|----------|-------------|---------|
| `-n`, `--num` | Number of sensor cycles to run | `1` |
| `-i`, `--interval` | Seconds between cycles (`0` = back-to-back) | `0` |
| `--style` | Prompt style: `hybrid`, `compact`, or `raw` | `hybrid` |
| `--model` | Ollama model name | `llama3.2:3b` |
| `--outdir` | Base directory for the CSV log | `outputs/` |

**Output CSV** (`outputs/baseline_demo.csv`):

| Column | Description |
|--------|-------------|
| `timestamp` | UTC timestamp of the sensor reading |
| `temperature` | Temperature in °C |
| `motion` | Motion detected (True/False) |
| `noise` | Noise level in dB |
| `prompt_style` | Prompt style used |
| `prompt` | Full prompt sent to the LLM |
| `raw_reply` | Raw LLM response text |
| `action_json` | Parsed JSON action (e.g., `{"action": "adjust_ac"}`) |
| `latency_ms` | Round-trip LLM inference time in milliseconds |
| `model` | Ollama model used |

---

### Continuous Streaming Demo

Runs **indefinitely**: sensors drip-feed data at a configurable interval, the on-device LLM reasons about each reading in real time, and every result is streamed to `outputs/stream_demo.csv`. Press `Ctrl-C` to stop gracefully (the current cycle finishes, the CSV is flushed and a summary is printed).

```bash
# Stream as fast as possible (no delay)
python scripts/stream_demo.py

# Realistic pacing – one reading every 20 seconds
python scripts/stream_demo.py --interval 20

# Use a specific model and custom output directory
python scripts/stream_demo.py --interval 10 --model llama3.2:3b --outdir results/
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--interval` | Seconds between sensor reads (`0` = back-to-back) | `0` |
| `--model` | Ollama model name | `llama3.2:3b` |
| `--outdir` | Directory to store the CSV log | `outputs/` |

**What happens each cycle:**

1. `simulate_sensors.py` generates a sensor snapshot (temp, motion, noise).
2. `edgeiot/brain.py` builds a `hybrid` prompt and queries the local Ollama instance.
3. The LLM reply is parsed into a structured JSON action.
4. The result is printed to the terminal and appended to `outputs/stream_demo.csv`.

**Terminal output example:**

```
╔══════════════════════════════════════════════╗
║   EdgeIOT Smart-Home Stream (on-device LLM)  ║
╠══════════════════════════════════════════════╣
║  Model    : llama3.2:3b                      ║
║  Interval : 20.0 s                           ║
║  Log      : outputs/stream_demo.csv          ║
╚══════════════════════════════════════════════╝

  Streaming… press Ctrl-C to stop.

── Cycle 1 ─────────────────────────────────────
  Sensor  → temp=23.4°C  motion=no  noise=52.1 dB
  LLM     → The smart home should maintain its current state…
  Action  → {"action": "no_action"}
  Latency : 1847 ms
```

---

### QLoRA Fine-Tuning Pipeline

`scripts/run_finetune.sh` and `scripts/run_ft_chain_all.sh` run these steps for all three models; the commands below show them for LLaMA 3.2.

#### Step 1 – Download the base model from Hugging Face

```bash
python scripts/01_download_model_from_hf.py
```

Interactive: searches Hugging Face with the `hf` CLI, lets you pick a model, and downloads it to `languagemodels/<model-name>` (e.g. `languagemodels/Llama-3.2-3B-Instruct`).

#### Step 2 – Build the training corpus (self-distillation)

```bash
python scripts/02_make_train_data.py \
    --inputs outputs/rerun/<run>/llm_inputs.csv \
    --teacher llama3.2:3b \
    --outdir data/finetune
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--inputs` | `llm_inputs.csv` written by `main.py` | required |
| `--teacher` | Ollama model that answers the training inputs | `llama3.2:3b` |
| `--n` | Number of training rows to write | `1000` |
| `--eval-n` | Size of the held-out evaluation sample (excluded from training) | `100` |
| `--seed` | Random seed | `42` |
| `--outdir` | Output directory (`train.jsonl`, `teacher_log.csv`, `corpus_summary.json`) | `data/finetune` |

#### Step 3 – QLoRA fine-tuning

```bash
python scripts/04_qlora_finetune.py \
    --model_dir languagemodels/Llama-3.2-3B-Instruct \
    --train_file data/finetune/train.jsonl \
    --output_dir languagemodels/fine_tuned_llama3_lora
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_dir` | Hugging Face checkpoint (quantised to 4-bit NF4 when loaded) | `languagemodels/Llama-3.2-3B-Instruct` |
| `--train_file` | Training corpus (JSONL) | `data/finetune/train.jsonl` |
| `--output_dir` | Where the LoRA adapter and tokeniser will be saved | `languagemodels/fine_tuned_llama3_lora` |
| `--epochs` | Number of training epochs | `3` |
| `--batch_size` | Training batch size | `4` |
| `--grad_accum` | Gradient accumulation steps | `2` |
| `--max_length` | Maximum sequence length | `512` |
| `--target_modules` | LoRA target modules (`qkv_proj` for Phi-3.5) | `q_proj,v_proj` |

#### Step 4 – Export to Ollama

```bash
python scripts/05_export_to_ollama.py \
    --base_dir languagemodels/Llama-3.2-3B-Instruct \
    --adapter_dir languagemodels/fine_tuned_llama3_lora \
    --name edgeiot-llama3.2-ft-merged
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--base_dir` | Base Hugging Face checkpoint | `languagemodels/Llama-3.2-3B-Instruct` |
| `--adapter_dir` | Trained LoRA adapter | `languagemodels/fine_tuned_llama3_lora` |
| `--merged_dir` | Where the merged model is written | `languagemodels/fine_tuned_llama3_merged` |
| `--llama_cpp` | llama.cpp checkout | `tools/llama.cpp` |
| `--name` | Ollama model name | `edgeiot-llama3.2-ft-merged` |
| `--qtype` | llama.cpp quantisation type, or `F16` | `Q4_K_M` |
| `--merge_base` | Merge into the dequantized `nf4` weights or the original `fp16` weights | `nf4` |
| `--with-adapter` | Also create a base + runtime adapter variant (needs an older Ollama) | off |

The fine-tuned model is then evaluated with `scripts/bench_llm.py finetune`. `scripts/03_load_hf_checkpoint.py` (standalone 4-bit quantisation of a checkpoint) is no longer part of this workflow, because `04_qlora_finetune.py` quantises the model when it loads it.

---

## Outputs

| Path | Description |
|------|-------------|
| `outputs/<timestamp>_<model>/` | Timestamped folder per `main.py` run (`--outdir outputs/rerun` for the dissertation run) |
| `…/llm_inputs.csv` | Tokenized test rows: state, tokens, anomaly flag, standardised and raw readings |
| `…/pipeline_metrics.json` | Cluster sizes, PCA variance, classifier accuracy and CV, anomaly counts |
| `…/semantic_outputs.json` | Semantic output record of every test row |
| `…/final_outputs_with_reasoning.json` | First 10 records with the LLM's reasoning |
| `…/jetson_benchmark.csv` | Per-sample LLM benchmark stats (LLM runs only) |
| `…/token_efficiency.csv` | Raw vs semantic input latency (LLM runs only) |
| `…/plots/` | Auto-generated pipeline charts (PNG) |
| `outputs/rerun/<run>/leakage_check.json`, `cache_replay.json`, `cache_curve.csv` | Written by `leakage_check.py` and `cache_replay.py` |
| `outputs/experiments/<experiment>_<timestamp>/` | `bench_llm.py` results |
| `outputs/baseline_demo.csv` | Accumulated batch-demo results (all runs, all models) |
| `outputs/stream_demo.csv` | Accumulated streaming results |

---

## Prompt Styles (brain module)

| Style | Description |
|-------|-------------|
| `hybrid` *(default)* | Friendly description with bullet-pointed sensor values |
| `compact` | Terse one-liner with all values inline |
| `raw` | Dumps the raw JSON sensor dict directly into the prompt |

---

## Canonical Smart-Home Actions

All components use the same seven actions (`edgeiot/semantic/actions.py`); the demo module `brain.py` maps free-text keywords onto them and returns `other` when nothing matches:

`adjust_ac`, `lights_off`, `lights_on`, `security_alert`, `ventilation_on`, `dehumidifier_on`, `no_action`.

---

## Reproducing the Dissertation Results

Ollama 0.35.1 with the models `llama3.2:3b` (digest a80c4f17acd5), `phi3.5:latest` (61819fb370a3) and `qwen2.5:1.5b` (65ec06548149). Run nothing else on the GPU during the experiments.

```bash
# E0  pipeline, sanity checks, tokenized test rows (outputs/rerun/<run>/llm_inputs.csv)
.venv/bin/python main.py --skip-llm --data RealLife.csv --outdir outputs/rerun
.venv/bin/python scripts/leakage_check.py

# E1-E3 multi-model benchmark, tokenization (raw vs semantic), probe set; builds the
#       three QLoRA corpora (self-distilled from each base model)
bash scripts/run_experiments.sh outputs/rerun/<run>/llm_inputs.csv

# E4  QLoRA training of LLaMA 3.2, Qwen 2.5 and Phi-3.5, then export (adapter merged into the
#     dequantized NF4 weights, GGUF via llama.cpp commit c0fd786, Q4_K_M), base vs fine-tuned
#     evaluation, probe set and the quantization-chain table
bash scripts/run_finetune.sh
bash scripts/run_ft_chain_all.sh outputs/rerun/<run>/llm_inputs.csv

# E5  response cache replay
.venv/bin/python scripts/cache_replay.py

# Tables, figures and numbers of Chapter 5 -> dissertation/latex/
.venv/bin/python scripts/analyze_results.py
.venv/bin/python scripts/thesis_figures.py
```

`analyze_results.py` reads the latest pipeline run in `outputs/rerun/` and the latest directory of each experiment in `outputs/experiments/`, and writes `results_macros.tex`, `tables/*.tex` and `figures/results/*.pdf` under `dissertation/latex/`. The LaTeX source of the dissertation is not part of this repository.
