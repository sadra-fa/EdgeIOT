# Smart Home Edge AI Pipeline (EdgeIOT)

A comprehensive edge AI pipeline for smart home sensor data. This project processes raw time-series sensor data to detect anomalies, classify homeowner behaviors, and generate semantic reasoning using local Large Language Models (LLMs) via Ollama.

## Features

1. **Data Processing & Normalization**
   - Ingests raw smart home sensor metrics (temperature, humidity, noise_db, light_intensity, motion_level).
   - Handles missing values, deduplication, and one-hot encoding for categorical contexts.
   - Standardizes numerical features using StandardScaler.

2. **Clustering & Classification**
   - Applies PCA and Hierarchical Clustering (Ward linkage) to generate pseudo-labels for distinct home behaviors (e.g., active, sleeping, away).
   - Trains an XGBoost classifier for fast, robust behavior prediction on edge devices.
   - Generates and plots comprehensive evaluation metrics including confusion matrices, ROC curves, and feature importance.

3. **Time-Series Anomaly Detection**
   - Uses sliding sequence windows over time-series data.
   - Trains an Isolation Forest model to flag abnormal sensor patterns `[ANOMALY_ALERT]`.

4. **Semantic Tokenization**
   - Translates raw continuous sensor values into discrete semantic tokens (e.g., `[TEMP_HIGH]`, `[NOISE_SPIKE]`, `[LOW_LIGHT]`).
   - Merges anomaly statuses into the token stream for downstream reasoning.

5. **Local LLM Reasoning (Ollama)**
   - Automatically discovers available Ollama models on the host machine.
   - Provides an interactive CLI for model selection and batch sizing.
   - Performs offline semantic reasoning using lightweight, edge-compatible models (e.g., Llama 3.2, Phi-3).
   - Evaluates LLM performance metrics (latency, tokens/sec, RAM footprint).
   - Includes a token efficiency benchmark comparing reasoning on raw numeric data vs. semantic tokens.
   - Added CPU usage, GPU load, and GPU memory usage metrics to benchmark summary.

6. **Rich Visualizations**
   - Automatically generates a complete suite of plots in the `outputs/` directory.
   - Organizes plots by pipeline stage: data exploration, clustering, classification, anomaly detection, tokenization, and LLM benchmarking.

## Project Structure

```text
EdgeIOT/
├── edgeiot/
│   ├── config.py           # Pipeline configuration and dataclasses
│   ├── data/               # Data loading, cleaning, and preprocessing
│   ├── models/             # XGBoost wrapper, Anomaly Detection, Model Evaluator
│   ├── semantic/           # Semantic Tokenization, Output formatting, Ollama Reasoner
│   └── viz/                # Centralized plotting using Matplotlib
├── main.py                 # Primary entrypoint for the pipeline
├── RealLife.csv            # Example dataset (requires placement)
└── README.md               # This documentation
```

## Requirements

- Python 3.10+
- `pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap requests`
- [Ollama](https://ollama.com/) must be installed and running locally with at least one model pulled (e.g., `ollama run llama3.2`).

## Usage

Run the main pipeline script:

```bash
python main.py
```

### Additional Scripts

- **`scripts/03_load_hf_checkpoint.py`** – Loads a locally‑downloaded Hugging Face LLaMA 3.2 checkpoint, applies 4‑bit quantization with bitsandbytes, and saves a ready‑to‑fine‑tune model (`base_llama3_4bit`). This script is the first step of the fine‑tuning pipeline on the Jetson Orin Nano.

```bash
python scripts/03_load_hf_checkpoint.py \
    --model_dir ./languagemodels/Llama-3.2-3B \
    --output_dir ./base_llama3_4bit
```

**Inputs**
- `--model_dir`: Path to the original HuggingFace LLaMA 3.2 checkpoint directory.
- `--output_dir`: Destination directory for the quantized model (`base_llama3_4bit`).

**Outputs**
- A directory containing the quantized model files ready for LoRA fine‑tuning.


```bash
python scripts/03_load_hf_checkpoint.py \
    --model_dir ./languagemodels/Llama-3.2-3B \
    --output_dir ./base_llama3_4bit
```
### QLoRA Fine‑tuning (Script 04)

```bash
python3 scripts/04_qlora_finetune.py \
    --model_dir ./languagemodels/llama-3.2-3B-think-fp16-fused-v6 \
    --train_file_dir ./outputs/20260604_225550_llama3.2_3b \
    --output_dir ./fine_tuned_llama3_lora \
    --epochs 3 \
    --batch_size 4
```
**Inputs**
- `--model_dir`: Path to the base model directory (quantized 4-bit model).
- `--train_file_dir`: Directory containing the `train_data.jsonl` file.
- `--output_dir`: Directory where the fine‑tuned LoRA adapter and tokenizer will be saved.
- `--epochs`: Number of training epochs.
- `--batch_size`: Training batch size.

**Outputs**
- Fine‑tuned LoRA adapter saved in `--output_dir`.
- Updated tokenizer saved in the same folder.
- Training logs printed to console.
```


### Command-line Arguments

- `--data <path>`: Specify the path to the CSV dataset (default is `RealLife.csv` as configured).
- `--model <name>`: Bypass the interactive model selector and specify an Ollama model directly (e.g., `--model phi3.5`).
- `--skip-llm`: Run the traditional ML pipeline (classification, anomaly detection) but skip the LLM reasoning phase entirely.
- `--outdir <path>`: Specify a custom base output directory (defaults to `outputs/`).

### Interactive Prompts

When running without `--model` or `--skip-llm`, the script will:
1. List all available local Ollama models and ask you to select one.
2. Ask how many test samples to process through the LLM for benchmarking (leave blank to process all, or enter a number like `5` for a quick test).

## Outputs

All outputs and plots are saved to a timestamped folder inside the `outputs/` directory (e.g., `outputs/20260604_140337_llama3.2_3b/`).
- `plots/`: Contains subdirectories for each stage of the pipeline with generated PNG charts.
- `jetson_benchmark.csv`: Per-sample LLM reasoning statistics.

## 🚀 Quick Demo – Pre-Fine-Tuning Smart-Home Edge IoT

Run a single-shot smart-home demo to see the base LLaMA 3.2 (3 B) model
reasoning about simulated sensor data **before** any fine-tuning:

```bash
# Single reading (default)
python scripts/run_demo.py

# 5 readings, batch mode
python scripts/run_demo.py -n 5

# 5 readings with 2-second intervals (simulated stream)
python scripts/run_demo.py -n 5 -i 2

# Use a different prompt style
python scripts/run_demo.py -n 3 --style compact
```

### What happens

1. `simulate_sensors.py` generates a JSON reading with **temperature** (°C), **motion** (bool), and **noise** (dB).
2. `edgeiot/brain.py` builds a natural-language prompt and sends it to the local Ollama instance (`llama3.2:3b`).
3. The LLM returns a **plain-text** reply; the brain module also extracts a **JSON-structured action** (e.g., `{"action": "adjust_ac"}`).
4. Everything is logged to `outputs/baseline_demo_YYYYMMDD_HHMMSS.csv` with columns: `timestamp`, `temperature`, `motion`, `noise`, `prompt_style`, `prompt`, `raw_reply`, `action_json`, `latency_ms`.

### Prompt styles

| Style | Description |
|-------|-------------|
| `hybrid` (default) | Friendly description with bullet-pointed sensor values |
| `compact` | Terse one-liner |
| `raw` | Dumps the raw JSON into the prompt |

After fine-tuning, re-run the same demo to compare the LLM replies and latency against this baseline.

