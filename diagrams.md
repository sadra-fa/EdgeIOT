# EdgeIOT Project Diagrams

Colours are the same in every diagram: blue = data/files, yellow = processing step, green = model or LLM, purple = decision, red = review/fallback.

## 1. ML Pipeline (`main.py`)
```mermaid
flowchart TD
    A["RealLife.csv"]:::data --> B["Load & clean<br/>missing values, duplicates"]:::step
    B --> C["One-hot location, behaviour<br/>StandardScaler on 5 sensors"]:::step

    C --> D["PCA + Ward clustering<br/>3 pseudo-labels"]:::step
    C --> G["Sliding windows of 5 rows"]:::step

    D --> F["Decision Tree, Random Forest, XGBoost<br/>80/20 split, 5-fold CV, SHAP"]:::model
    G --> I["Isolation Forest<br/>5% contamination"]:::model

    F -->|"state + confidence"| H["Semantic tokenizer<br/>[TEMP_HIGH], [LOW_LIGHT], ..."]:::step
    I -->|"[ANOMALY_ALERT]"| H

    H --> J["llm_inputs.csv<br/>pipeline_metrics.json"]:::data
    J -.->|"unless --skip-llm"| K["Ollama: smoke test, benchmark,<br/>raw vs semantic token efficiency"]:::model

    classDef data fill:#D6EAF8,stroke:#2E86C1,color:#1B2631
    classDef step fill:#FCF3CF,stroke:#B7950B,color:#1B2631
    classDef model fill:#D5F5E3,stroke:#1E8449,color:#1B2631
```

Every stage also writes its plots to `outputs/<run>/plots/`.

## 2. LLM Decision, Rule Baseline and Response Cache
```mermaid
flowchart TD
    A["State + tokens"]:::data --> B{"Cached?"}:::decide
    B -->|"hit"| C["Return cached action"]:::data
    B -->|"miss"| D["Hybrid prompt → Ollama LLM"]:::model
    D --> E["check_reply<br/>JSON schema, canonical action"]:::step
    E --> G{"In<br/>vocabulary?"}:::decide

    G -->|"yes"| H{"Matches<br/>rule_action?"}:::decide
    G -->|"no: unseen token,<br/>free text"| R["Store, flag for review"]:::review

    H -->|"yes"| S["Store LLM action"]:::data
    H -->|"no, policy rules"| T["Store rule action"]:::data
    H -->|"no, policy flag"| R

    classDef data fill:#D6EAF8,stroke:#2E86C1,color:#1B2631
    classDef step fill:#FCF3CF,stroke:#B7950B,color:#1B2631
    classDef model fill:#D5F5E3,stroke:#1E8449,color:#1B2631
    classDef decide fill:#E8DAEF,stroke:#7D3C98,color:#1B2631
    classDef review fill:#FADBD8,stroke:#C0392B,color:#1B2631
```

`scripts/cache_replay.py` replays the test rows through this cache, using LLaMA 3.2's logged replies instead of new inference.

## 3. Demo Streaming Loop (`scripts/stream_demo.py`)
```mermaid
sequenceDiagram
    participant S as simulate_sensors.py
    participant P as stream_demo.py
    participant B as brain.py
    participant M as Ollama LLM
    participant L as stream_demo.csv
    loop every --interval seconds
        S->>P: temperature, motion, noise
        P->>B: run(sensor)
        B->>M: hybrid prompt
        M-->>B: free-text reply
        B-->>P: keyword-matched action, latency
        P->>L: append row
    end
    Note over P,L: Ctrl-C finishes the cycle, flushes the CSV, prints a summary
```

`scripts/run_demo.py` runs the same cycle a fixed number of times and appends to `outputs/baseline_demo.csv`.

## 4. Training Corpus (`scripts/02_make_train_data.py`)
```mermaid
flowchart TD
    A["llm_inputs.csv"]:::data --> B["Drop the 100 evaluation rows"]:::step
    B --> C["Distinct state + tokens inputs"]:::step
    C --> D["Teacher = base model<br/>full hybrid prompt"]:::model
    D --> E{"Valid?"}:::decide
    E -->|"no: next seed,<br/>3 tries at most"| D
    E -->|"yes: schema-valid JSON"| F["Re-serialise as compact JSON"]:::step
    F --> G["Sample 1000 rows"]:::step
    G --> H["train.jsonl<br/>teacher_log.csv, corpus_summary.json"]:::data

    classDef data fill:#D6EAF8,stroke:#2E86C1,color:#1B2631
    classDef step fill:#FCF3CF,stroke:#B7950B,color:#1B2631
    classDef model fill:#D5F5E3,stroke:#1E8449,color:#1B2631
    classDef decide fill:#E8DAEF,stroke:#7D3C98,color:#1B2631
```

Each model gets its own self-distilled corpus: `data/finetune` (LLaMA 3.2), `data/finetune_qwen`, `data/finetune_phi`.

## 5. QLoRA Fine-Tuning and Export
```mermaid
flowchart LR
    subgraph T["04_qlora_finetune.py (on the Jetson)"]
        direction TB
        A["HF Instruct checkpoint"]:::data --> B["Load in 4-bit NF4<br/>double quant, FP16 compute"]:::step
        B --> D["LoRA r=64, alpha=16<br/>q_proj, v_proj<br/>qkv_proj for Phi"]:::step
        C["train.jsonl"]:::data --> D
        D --> E["Train on response tokens<br/>paged 32-bit AdamW"]:::model
        E --> F["LoRA adapter"]:::data
    end
    subgraph X["05_export_to_ollama.py"]
        direction TB
        G["Merge into dequantized NF4<br/>or original FP16 weights"]:::step --> H["GGUF FP16<br/>llama.cpp"]:::step
        H --> I["Quantize<br/>Q4_K_M, Q8_0, Q4_0, F16"]:::step
        I --> J["ollama create<br/>+ instruction template"]:::model
    end
    T -->|"adapter"| X

    classDef data fill:#D6EAF8,stroke:#2E86C1,color:#1B2631
    classDef step fill:#FCF3CF,stroke:#B7950B,color:#1B2631
    classDef model fill:#D5F5E3,stroke:#1E8449,color:#1B2631
    style T fill:#FBFCFC,stroke:#909497
    style X fill:#FBFCFC,stroke:#909497
```

`scripts/run_finetune.sh` trains LLaMA 3.2, Qwen 2.5 and Phi-3.5; `scripts/run_ft_chain_all.sh` exports every variant (`edgeiot-*-ft-merged` and the chain variants) and evaluates it. `scripts/diagnose_ft_chain.py` compares the adapter's answers on the NF4 and FP16 base to locate where along this chain a fine-tuned model changes.

## 6. Experiments and Analysis
```mermaid
flowchart TD
    A["E0: main.py --skip-llm<br/>leakage_check.py"]:::step --> B["outputs/rerun/&lt;run&gt;/<br/>llm_inputs.csv"]:::data

    B --> C["bench_llm.py<br/>E1 multi · E2 tokenization<br/>E3 probe · E4 finetune"]:::model
    B --> D["E5: cache_replay.py"]:::step

    C --> E["outputs/experiments/<br/>requests.csv, models.csv, meta.json"]:::data
    D --> F["cache_replay.json<br/>cache_curve.csv"]:::data

    E --> G["analyze_results.py<br/>thesis_figures.py"]:::step
    F --> G
    G --> H["dissertation/latex/<br/>macros, tables, figures"]:::data

    classDef data fill:#D6EAF8,stroke:#2E86C1,color:#1B2631
    classDef step fill:#FCF3CF,stroke:#B7950B,color:#1B2631
    classDef model fill:#D5F5E3,stroke:#1E8449,color:#1B2631
```

`bench_llm.py` also samples `tegrastats` (GPU load, RAM, temperature, board power) during every request; E3 uses the cases in `scripts/probe_cases.py`, and E4 the fine-tuned models from diagram 5.
