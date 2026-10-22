#!/usr/bin/env python3
"""
bench_llm.py — Instrumented LLM experiments for the dissertation.

Every request logs Ollama's four timing fields (prompt_eval_count,
prompt_eval_duration, eval_count, eval_duration), the wall-clock latency,
the schema check of the reply, and tegrastats samples (GPU load, RAM,
temperature, board power) taken while the request was running.

Experiments
-----------
  multi         same N test samples, semantic hybrid prompt, one model after another
  tokenization  same N test samples, raw vs semantic fragment (paired, order randomised)
  probe         hand-written cases (scripts/probe_cases.py), LLM vs rule baseline
  finetune      base model with the hybrid prompt vs fine-tuned model with its
                instruction prompt, on the same N test samples

Usage
-----
  python scripts/bench_llm.py multi --inputs outputs/rerun/<run>/llm_inputs.csv \
      --models llama3.2:3b phi3.5:latest qwen2.5:1.5b
  python scripts/bench_llm.py tokenization --inputs ... --models llama3.2:3b phi3.5:latest qwen2.5:1.5b
  python scripts/bench_llm.py probe --models llama3.2:3b qwen2.5:1.5b
  python scripts/bench_llm.py finetune --inputs ... --models llama3.2:3b edgeiot-llama3.2-ft
"""

import argparse
import json
import platform
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edgeiot.config import OllamaConfig                                   # noqa: E402
from edgeiot.semantic.actions import check_reply, rule_action             # noqa: E402
from edgeiot.semantic.llm_reasoner import (                               # noqa: E402
    OllamaReasoner, build_finetuned_prompt, build_hybrid_prompt,
    raw_fragment, semantic_fragment, wrap_finetuned,
)
from scripts.probe_cases import PROBE_CASES                               # noqa: E402

OLLAMA = "http://localhost:11434"
SENSORS = ["temperature", "humidity", "noise_db", "light_intensity", "motion_level"]
FINETUNED_MARKER = "-ft"


# ── tegrastats sampler ───────────────────────────────────────────────────────
class TegraMonitor:
    """Background tegrastats reader; `window(t0, t1)` summarises a time span."""

    _re = {
        "gpu":   re.compile(r"GR3D_FREQ (\d+)%"),
        "ram":   re.compile(r"RAM (\d+)/(\d+)MB"),
        "temp":  re.compile(r"tj@([\d.]+)C"),
        "power": re.compile(r"VDD_IN (\d+)mW"),
    }

    def __init__(self, interval_ms: int = 100):
        self.samples = []
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            now = time.time()
            vals = {}
            for key, rx in self._re.items():
                m = rx.search(line)
                if m:
                    vals[key] = float(m.group(1))
            if vals:
                self.samples.append((now, vals))

    def window(self, t0: float, t1: float) -> dict:
        rows = [v for t, v in self.samples if t0 <= t <= t1]
        out = {"n_tegra": len(rows)}
        for key in ("gpu", "ram", "temp", "power"):
            xs = [r[key] for r in rows if key in r]
            out[f"{key}_mean"] = sum(xs) / len(xs) if xs else None
            out[f"{key}_max"] = max(xs) if xs else None
        return out

    def stop(self):
        self.proc.terminate()


# ── helpers ──────────────────────────────────────────────────────────────────
def ollama_ps(model: str) -> dict:
    for m in requests.get(f"{OLLAMA}/api/ps", timeout=10).json().get("models", []):
        if model in (m["name"], m["model"]) or m["name"] == f"{model}:latest":
            return {"ps_size_mb": m["size"] / 2**20, "ps_vram_mb": m.get("size_vram", 0) / 2**20}
    return {"ps_size_mb": None, "ps_vram_mb": None}


def unload(model: str):
    requests.post(f"{OLLAMA}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    time.sleep(3)


def environment() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as e:
            return f"unavailable: {e}"
    import numpy, sklearn, xgboost  # noqa: E401
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "ollama": requests.get(f"{OLLAMA}/api/version", timeout=10).json().get("version"),
        "l4t": run(["head", "-1", "/etc/nv_tegra_release"]),
        "kernel": platform.release(),
        "power_mode": run(["nvpmodel", "-q"]),
        "python": platform.python_version(),
        "pandas": pd.__version__, "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
    }


def load_samples(inputs: str, n: int, out_dir: Path, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(inputs)
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    sample.to_csv(out_dir / "samples.csv", index=False)
    return sample


def save_partial(records, out_dir: Path):
    """Keep every finished request on disk, so an interrupted run loses nothing."""
    pd.DataFrame(records).to_csv(out_dir / "requests.partial.csv", index=False)


def tokens_of(row) -> list:
    return row["tokens"].split() if isinstance(row["tokens"], str) and row["tokens"] else []


def run_request(reasoner, monitor, prompt, meta: dict) -> dict:
    t0 = time.time()
    # Fine-tuned models get their training template verbatim (raw mode)
    r = reasoner.generate(prompt, raw=FINETUNED_MARKER in meta.get("model", ""))
    t1 = time.time()
    rec = dict(meta)
    rec.update(r)
    rec.update(check_reply(r["response"]))
    rec.update(monitor.window(t0, t1))
    rec["prompt"] = prompt
    rec["prompt_chars"] = len(prompt)
    return rec


def warmup(reasoner, monitor, model) -> dict:
    """First request loads the model; its load time is the cold start."""
    r = run_request(reasoner, monitor, build_hybrid_prompt("active", "[NO_EVENT]"),
                    {"model": model, "warmup": True})
    info = {"model": model, "cold_load_ms": r["load_ms"], "cold_wall_ms": r["wall_ms"]}
    info.update(ollama_ps(model))
    return info


def prompt_for(model: str, state: str, fragment: str) -> str:
    if FINETUNED_MARKER in model:
        return wrap_finetuned(build_finetuned_prompt(state, fragment))
    return build_hybrid_prompt(state, fragment)


# ── experiments ──────────────────────────────────────────────────────────────
def exp_samples(args, monitor, out_dir, condition_fn):
    """Shared loop for multi and finetune: one prompt per sample per model."""
    samples = load_samples(args.inputs, args.n, out_dir)
    records, model_info = [], []
    for model in args.models:
        reasoner = OllamaReasoner(OllamaConfig(model=model))
        model_info.append(warmup(reasoner, monitor, model))
        for i, row in samples.iterrows():
            tokens = tokens_of(row)
            prompt = condition_fn(model, row["state"], semantic_fragment(tokens))
            rec = run_request(reasoner, monitor, prompt, {
                "model": model, "sample": i, "row_index": row["row_index"],
                "state": row["state"], "tokens": " ".join(tokens),
                "anomaly": row["anomaly"], "rule_action": rule_action(row["state"], tokens),
            })
            records.append(rec)
            save_partial(records, out_dir)
            print(f"[{model}] {i+1}/{len(samples)} {rec['wall_ms']:.0f} ms "
                  f"out={rec['eval_count']} valid={rec['schema_valid']}", flush=True)
        model_info[-1].update(ollama_ps(model))
        unload(model)
    return records, model_info


def exp_tokenization(args, monitor, out_dir):
    samples = load_samples(args.inputs, args.n, out_dir)
    rng = random.Random(42)
    records, model_info = [], []
    for model in args.models:
        reasoner = OllamaReasoner(OllamaConfig(model=model))
        model_info.append(warmup(reasoner, monitor, model))
        for i, row in samples.iterrows():
            # Same definition as main.py: anomaly flag excluded in both conditions
            tokens = [t for t in tokens_of(row) if t != "[ANOMALY_ALERT]"]
            fragments = {
                "semantic": semantic_fragment(tokens),
                "raw": raw_fragment({s: row[f"z_{s}"] for s in SENSORS}),
            }
            order = ["semantic", "raw"]
            rng.shuffle(order)
            for cond in order:
                rec = run_request(reasoner, monitor,
                                  build_hybrid_prompt(row["state"], fragments[cond]), {
                    "model": model, "sample": i, "condition": cond,
                    "fragment": fragments[cond], "fragment_chars": len(fragments[cond]),
                    "n_predicates": len(tokens) if cond == "semantic" else None,
                    "state": row["state"],
                })
                records.append(rec)
                save_partial(records, out_dir)
            print(f"[{model}] {i+1}/{len(samples)}", flush=True)
        unload(model)
    return records, model_info


def exp_probe(args, monitor, out_dir):
    records, model_info = [], []
    for model in args.models:
        reasoner = OllamaReasoner(OllamaConfig(model=model))
        model_info.append(warmup(reasoner, monitor, model))
        for cid, group, state, tokens, note, acceptable in PROBE_CASES:
            fragment = semantic_fragment(tokens)
            if note:
                fragment += f'\nResident note: "{note}"'
            rec = run_request(reasoner, monitor, prompt_for(model, state, fragment), {
                "model": model, "case": cid, "group": group, "state": state,
                "tokens": " ".join(tokens), "note": note,
                "acceptable": "|".join(sorted(acceptable)),
            })
            rec["rule_action"] = rule_action(state, tokens)
            rec["llm_correct"] = rec["action"] in acceptable
            rec["rule_correct"] = rec["rule_action"] in acceptable
            records.append(rec)
            save_partial(records, out_dir)
            print(f"[{model}] {cid} llm={rec['action']} rule={rec['rule_action']}", flush=True)
        unload(model)
    return records, model_info


def main():
    p = argparse.ArgumentParser(description="Instrumented LLM experiments")
    p.add_argument("experiment", choices=["multi", "tokenization", "probe", "finetune"])
    p.add_argument("--inputs", help="llm_inputs.csv written by main.py")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--outdir", default="outputs/experiments")
    p.add_argument("--idle-seconds", type=int, default=15)
    p.add_argument("--tag", default="", help="suffix of the experiment directory name, e.g. qwen")
    args = p.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{args.experiment}-{args.tag}" if args.tag else args.experiment
    out_dir = PROJECT_ROOT / args.outdir / f"{name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    monitor = TegraMonitor()
    t_idle = time.time()
    time.sleep(args.idle_seconds)
    idle = monitor.window(t_idle, time.time())

    if args.experiment in ("multi", "finetune"):
        records, info = exp_samples(args, monitor, out_dir, prompt_for)
    elif args.experiment == "tokenization":
        records, info = exp_tokenization(args, monitor, out_dir)
    else:
        records, info = exp_probe(args, monitor, out_dir)
    monitor.stop()

    pd.DataFrame(records).to_csv(out_dir / "requests.csv", index=False)
    pd.DataFrame(info).to_csv(out_dir / "models.csv", index=False)
    meta = {"args": vars(args), "idle": idle, "environment": environment(),
            "options": OllamaReasoner(OllamaConfig()).options}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"Saved → {out_dir}")


if __name__ == "__main__":
    main()
