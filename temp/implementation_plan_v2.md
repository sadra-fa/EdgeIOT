# Implementation Plan v2 — Structured Plots, Python Packages, and Interactive Model Selection

This plan extends the original refactoring with three enhancements to the EdgeIOT pipeline.

---

## Enhancement 1: Structured Plot Output Folders

### Current Behaviour
All 11 plot images are saved flat into `outputs/plots/` with numbered prefixes (`01_timeseries.png`, `02_clustering_pca.png`, etc.).

### New Behaviour
Group plots into **named subfolders** by pipeline stage. Each folder gets a descriptive name:

```
outputs/
├── plots/
│   ├── 01_data_exploration/
│   │   └── timeseries.png
│   ├── 02_clustering/
│   │   └── clustering_pca.png
│   ├── 03_classification/
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   ├── roc.png
│   │   ├── model_comparison.png
│   │   └── shap_beeswarm.png
│   ├── 04_anomaly_detection/
│   │   └── anomaly.png
│   ├── 05_tokenization/
│   │   └── token_distribution.png
│   └── 06_llm_benchmark/
│       ├── llm_benchmark.png
│       └── token_efficiency.png
├── semantic_outputs.json
├── final_outputs_with_reasoning.json
├── jetson_benchmark.csv
└── token_efficiency.csv
```

### Files Modified
| File | Change |
|---|---|
| [config.py](file:///home/sadra/dev/EdgeIOT/config.py) | Replace single `plots_dir` with a `PlotDirsConfig` dataclass holding 6 subfolder paths. `__post_init__` creates all dirs. |
| [visualization.py](file:///home/sadra/dev/EdgeIOT/visualization.py) | `PipelineVisualizer.__init__` accepts `PlotDirsConfig`. Each plot method saves to its category subfolder. Remove numbered prefixes from filenames. |
| [main.py](file:///home/sadra/dev/EdgeIOT/main.py) | Pass `cfg.plots` (the new `PlotDirsConfig`) to `PipelineVisualizer`. |

---

## Enhancement 2: Reorganise Modules into Python Packages

### Current Behaviour
All `.py` files are flat in `/home/sadra/dev/EdgeIOT/`.

### New Behaviour
Group modules into logical sub-packages with `__init__.py` files for clean imports:

```
EdgeIOT/
├── main.py                         # entry point (stays at root)
├── requirements.txt
├── edgeiot/                        # ← top-level package
│   ├── __init__.py
│   ├── config.py                   # PipelineConfig, PlotDirsConfig, etc.
│   ├── data/                       # ← data handling sub-package
│   │   ├── __init__.py             # re-exports DataLoader, DataPreprocessor
│   │   ├── data_loader.py
│   │   └── clustering.py
│   ├── models/                     # ← ML models sub-package
│   │   ├── __init__.py             # re-exports classifiers, evaluator, anomaly
│   │   ├── classification.py
│   │   └── anomaly_detection.py
│   ├── semantic/                   # ← tokenization + LLM sub-package
│   │   ├── __init__.py             # re-exports tokenizer, encoder, reasoner
│   │   ├── tokenization.py
│   │   └── llm_reasoner.py
│   └── viz/                        # ← visualization sub-package
│       ├── __init__.py             # re-exports PipelineVisualizer
│       └── visualization.py
```

### Files Modified
| File | Change |
|---|---|
| All existing `.py` modules | Move into `edgeiot/` sub-packages. Update internal imports from bare `from config import ...` to relative `from edgeiot.config import ...` or relative `from ..config import ...`. |
| Each `__init__.py` | Re-export public classes for ergonomic imports (e.g. `from edgeiot.data import DataLoader`). |
| [main.py](file:///home/sadra/dev/EdgeIOT/main.py) | Update all imports to use the `edgeiot` package namespace. |

---

## Enhancement 3: Interactive Ollama Model Selection at Startup

### Current Behaviour
The Ollama model is either hard-coded (`llama3.2:3b`) or set via `--model` CLI flag.

### New Behaviour
When `--skip-llm` is **not** set and `--model` is **not** provided, the pipeline will:

1. Run `ollama list` via `subprocess` to discover locally installed models.
2. Print a numbered list to the console, e.g.:
   ```
   ╔════════════════════════════════════════════╗
   ║  Available Ollama Models                   ║
   ╠════════════════════════════════════════════╣
   ║  1.  llama3.2:3b          (2.0 GB)        ║
   ║  2.  phi3.5:latest        (3.8 GB)        ║
   ║  3.  mistral:7b           (4.1 GB)        ║
   ╚════════════════════════════════════════════╝
   Enter model number [1]: 
   ```
3. Wait for the user to type a number and press Enter.
4. Set `cfg.ollama.model` to the selected model name.
5. If Ollama is unreachable or no models are installed, log a warning and fall back to `--skip-llm` behaviour.

### Files Modified
| File | Change |
|---|---|
| [llm_reasoner.py](file:///home/sadra/dev/EdgeIOT/edgeiot/semantic/llm_reasoner.py) | Add a **standalone function** `discover_ollama_models() -> list[dict]` that runs `ollama list`, parses the output, and returns `[{"name": ..., "size": ...}, ...]`. Add `select_model_interactive(models) -> str` that displays the menu and reads input. |
| [main.py](file:///home/sadra/dev/EdgeIOT/main.py) | Before creating `OllamaReasoner`, if `--model` not given and `--skip-llm` not set: call `discover_ollama_models()`, then `select_model_interactive()`, and assign the result to `cfg.ollama.model`. |

---

## Execution Order

1. **Enhancement 2** (packages) first — creates the directory structure.
2. **Enhancement 1** (plot folders) second — modifies config + viz inside the new package layout.
3. **Enhancement 3** (model selection) last — adds new logic to `llm_reasoner.py` + `main.py`.

---

## Task Checklist

- [ ] Create `edgeiot/` package structure with `__init__.py` files
- [ ] Move modules into sub-packages (`data/`, `models/`, `semantic/`, `viz/`)
- [ ] Update all imports (internal cross-module + `main.py`)
- [ ] Add `PlotDirsConfig` dataclass to `config.py`
- [ ] Update `PipelineVisualizer` to save into category subfolders
- [ ] Update `main.py` to pass new plot dirs config
- [ ] Add `discover_ollama_models()` and `select_model_interactive()` to `llm_reasoner.py`
- [ ] Wire interactive model selection into `main.py`
- [ ] Delete old flat `.py` files from `EdgeIOT/` root (keep only `main.py` + `requirements.txt`)
- [ ] Verify with `python main.py --skip-llm`
