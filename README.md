# Smart Home Edge AI Pipeline (EdgeIOT)

A comprehensive edge AI pipeline for smart home sensor data. This project processes raw time-series sensor data to detect anomalies, classify homeowner behaviours, and generate semantic reasoning using local Large Language Models (LLMs) via Ollama – all running on-device on an **NVIDIA Jetson Orin Nano**.

---

## Features

1. **Data Processing & Normalisation**
   - Ingests raw smart home sensor metrics (temperature, humidity, noise_db, light_intensity, motion_level).
   - Handles missing values, deduplication, and one-hot encoding for categorical contexts.
   - Standardises numerical features using `StandardScaler`.

2. **Clustering & Classification**
   - Applies PCA and Hierarchical Clustering (Ward linkage) to generate pseudo-labels for distinct home behaviours (e.g., active, sleeping, away).
   - Trains an XGBoost classifier for fast, robust behaviour prediction on edge devices.
   - Generates confusion matrices, ROC curves, and feature importance plots.

3. **Time-Series Anomaly Detection**
   - Uses sliding sequence windows over time-series data.
   - Trains an Isolation Forest model to flag abnormal sensor patterns `[ANOMALY_ALERT]`.

4. **Semantic Tokenisation**
   - Translates raw continuous sensor values into discrete semantic tokens (e.g., `[TEMP_HIGH]`, `[NOISE_SPIKE]`, `[LOW_LIGHT]`).
   - Merges anomaly statuses into the token stream for downstream reasoning.

5. **Local LLM Reasoning (Ollama)**
   - Automatically discovers available Ollama models on the host machine.
   - Provides an interactive CLI for model selection and batch sizing.
   - Performs offline semantic reasoning using lightweight, edge-compatible models (e.g., LLaMA 3.2, Phi-3).
   - Evaluates LLM performance metrics (latency, tokens/sec, RAM, GPU load, GPU memory).
   - Includes a token-efficiency benchmark comparing reasoning on raw numeric data vs. semantic tokens.

6. **Smart-Home Brain Module (`edgeiot/brain.py`)**
   - Accepts a sensor-data dict, builds a natural-language prompt (three styles: `hybrid`, `compact`, `raw`), queries the local Ollama REST API, and returns the raw reply, a parsed JSON action, and round-trip latency.
   - Action keywords are mapped to structured intents (e.g., `adjust_ac`, `security_alert`, `no_action`).

7. **Baseline Demo (`scripts/run_demo.py`)**
   - Batch-mode script for running a fixed number of sensor–LLM–log cycles.
   - All results are appended to a **single persistent CSV** (`outputs/baseline_demo.csv`) so that every run (across different models, styles, or sessions) accumulates in one place, including a `model` column.

8. **Streaming Demo (`scripts/stream_demo.py`)** *(new)*
   - Runs an **infinite sensor stream**: sensors emit data every `--interval` seconds, the on-device LLM reasons about each reading, and results are streamed to `outputs/stream_demo.csv`.
   - Graceful `Ctrl-C` shutdown – finishes the current cycle, flushes the CSV, and prints a summary.
   - Real-life simulation: mimics a continuously-running edge device where the LLM is always resident in memory.

9. **Rich Visualisations**
   - Automatically generates a complete suite of plots in the `outputs/` directory.
   - Organised by pipeline stage: data exploration, clustering, classification, anomaly detection, tokenisation, and LLM benchmarking.

---

## Project Structure

```text
EdgeIOT/
├── edgeiot/
│   ├── __init__.py
│   ├── brain.py            # Smart-home "brain": prompt → LLM → parsed action
│   ├── config.py           # Pipeline configuration and OllamaConfig dataclass
│   ├── data/               # Data loading, cleaning, and preprocessing
│   ├── models/             # XGBoost wrapper, Anomaly Detection, Model Evaluator
│   ├── semantic/           # Semantic Tokenisation, Output formatting, Ollama Reasoner
│   └── viz/                # Centralised plotting using Matplotlib
├── scripts/
│   ├── simulate_sensors.py # Simulated sensor reading generator
│   ├── run_demo.py         # Batch baseline demo (N cycles → CSV)
│   ├── stream_demo.py      # Continuous streaming demo (infinite loop → CSV)
│   ├── 01_download_model_from_hf.py
│   ├── 02_make_train_data.py
│   ├── 03_load_hf_checkpoint.py
│   └── 04_qlora_finetune.py
├── main.py                 # Primary entrypoint for the full ML pipeline
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  # or manually:
  pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap requests
  ```
- [Ollama](https://ollama.com/) must be installed and running locally with at least one model pulled:
  ```bash
  ollama pull llama3.2:3b
  ollama serve          # keep running in a separate terminal
  ```

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

### Continuous Streaming Demo *(new)*

Runs **indefinitely**: sensors drip-feed data at a configurable interval, the on-device LLM reasons about each reading in real time, and every result is streamed to `outputs/stream_demo.csv`. Press `Ctrl-C` to stop gracefully.

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

#### Step 1 – Download model from Hugging Face

```bash
python scripts/01_download_model_from_hf.py \
    --repo_id meta-llama/Llama-3.2-3B \
    --output_dir ./languagemodels/Llama-3.2-3B
```

#### Step 2 – Generate training data

```bash
python scripts/02_make_train_data.py \
    --input_csv RealLife.csv \
    --output_dir ./outputs/train_data
```

#### Step 3 – Load HF checkpoint & apply 4-bit quantisation

```bash
python scripts/03_load_hf_checkpoint.py \
    --model_dir ./languagemodels/Llama-3.2-3B \
    --output_dir ./base_llama3_4bit
```

| Argument | Description |
|----------|-------------|
| `--model_dir` | Path to the original HuggingFace LLaMA 3.2 checkpoint |
| `--output_dir` | Destination for the 4-bit quantised model |

#### Step 4 – QLoRA fine-tuning

```bash
python scripts/04_qlora_finetune.py \
    --model_dir ./base_llama3_4bit \
    --train_file_dir ./outputs/train_data \
    --output_dir ./fine_tuned_llama3_lora \
    --epochs 3 \
    --batch_size 4
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_dir` | Path to the base (quantised) model | required |
| `--train_file_dir` | Directory containing `train_data.jsonl` | required |
| `--output_dir` | Where the LoRA adapter and tokeniser will be saved | required |
| `--epochs` | Number of training epochs | `3` |
| `--batch_size` | Training batch size | `4` |

---

## Outputs

| Path | Description |
|------|-------------|
| `outputs/baseline_demo.csv` | Accumulated batch-demo results (all runs, all models) |
| `outputs/stream_demo.csv` | Accumulated streaming results |
| `outputs/<timestamp>_<model>/` | Timestamped folder per `main.py` run |
| `outputs/<timestamp>_<model>/plots/` | Auto-generated pipeline charts (PNG) |
| `outputs/<timestamp>_<model>/jetson_benchmark.csv` | Per-sample LLM benchmark stats |

---

## Prompt Styles (brain module)

| Style | Description |
|-------|-------------|
| `hybrid` *(default)* | Friendly description with bullet-pointed sensor values |
| `compact` | Terse one-liner with all values inline |
| `raw` | Dumps the raw JSON sensor dict directly into the prompt |

---

## Recognised Smart-Home Actions

The brain module maps LLM responses to the following structured intents:

| Action | Trigger keywords |
|--------|-----------------|
| `adjust_ac` | "turn on air conditioner", "cool the room", … |
| `turn_off_ac` | "turn off air conditioner", "disable cooling", … |
| `close_windows` | "close windows", "shut the windows" |
| `open_windows` | "open windows" |
| `turn_off_lights` | "turn off lights", "switch off lights" |
| `turn_on_lights` | "turn on lights", "switch on lights" |
| `reduce_noise` | "reduce noise", "lower the volume" |
| `security_alert` | "security alert", "intruder", "alarm" |
| `no_action` | "no action", "everything is normal" |
| `other` | No keyword matched |
