import logging
import time
import subprocess
from typing import Dict, List, Optional

import psutil
import requests

from edgeiot.config import OllamaConfig
from edgeiot.semantic.actions import CANONICAL_ACTIONS, check_reply

logger = logging.getLogger(__name__)

# Names of the Ward clusters, read from their composition in RealLife.csv
# (see pipeline_metrics.json: cluster_vs_behaviour and the location crosstab):
#   0 = studying/eating in cafes, libraries and study rooms
#   1 = walking/stressed in streets, traffic, offices and meeting rooms
#   2 = sleeping in bedrooms and dorm rooms
CLUSTER_NAMES: Dict[int, str] = {0: "sedentary", 1: "active", 2: "sleeping"}


def discover_ollama_models() -> List[Dict]:
    """Runs ollama list, parses output, and returns list of dicts with name and size."""
    try:
        res = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = res.stdout.strip().split('\n')[1:] # skip header
        models = []
        for line in lines:
            if not line: continue
            parts = line.split()
            # typical format: NAME ID SIZE MODIFIED
            name = parts[0]
            size_idx = -1
            for i, p in enumerate(parts):
                if p in ["GB", "MB", "KB", "B"]:
                    size_idx = i - 1
                    break
            
            if size_idx != -1 and size_idx < len(parts):
                size = f"{parts[size_idx]} {parts[size_idx+1]}"
            else:
                size = "Unknown"
            models.append({"name": name, "size": size})
        return models
    except Exception as e:
        logger.warning(f"Failed to discover Ollama models: {e}")
        return []

def select_model_interactive(models: List[Dict]) -> str:
    """Displays menu and reads input."""
    if not models:
        return ""
    
    print("╔" + "═" * 47 + "╗")
    print("   Available Ollama Models                   ")
    print("╠" + "═" * 47 + "╣")
    for i, m in enumerate(models):
        name_part = f"{i+1}.  {m['name']}"
        size_part = f"({m['size']})"
        line_content = f"{name_part:<30} {size_part:>13}"
        print(f"║  {line_content} ║")
    print("╚" + "═" * 47 + "╝")

    try:
        choice = input("Enter model number [1]: ").strip()
        if choice == "":
            return models[0]["name"]
        elif choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice)-1]["name"]
    except Exception:
        pass
    return models[0]["name"]

# ── Prompt construction ──────────────────────────────────────────────────────
# The preamble is identical in every condition; only the "Observed Events"
# fragment changes between the raw and the semantic representation.
PROMPT_PREAMBLE = (
    "System: You are an autonomous Edge AI smart home controller running "
    "locally on an NVIDIA Jetson.\n"
    "Analyze the inhabitant behavior and environmental tokens.\n"
)
PROMPT_INSTRUCTION = (
    'Output strict JSON with fields: "reasoning" (string, one sentence) and '
    '"action_json" (object with an "action" field). The action must be one of: '
    + ", ".join(CANONICAL_ACTIONS) + "."
)
DEFAULT_OPTIONS = {
    "temperature":    0.2,
    "top_p":          0.9,
    "top_k":          40,
    "repeat_penalty": 1.1,
    "num_ctx":        512,
    "num_predict":    128,
    "num_gpu":        99,
    "seed":           42,
}


def semantic_fragment(tokens: List[str]) -> str:
    """Variable part of the prompt in the semantic condition."""
    return " ".join(tokens) if tokens else "[NO_EVENT]"


def raw_fragment(z_values: Dict[str, float]) -> str:
    """Variable part of the prompt in the raw condition (standardised readings)."""
    return " ".join(f"{name}={value:.3f}" for name, value in z_values.items())


def build_hybrid_prompt(state: str, fragment: str) -> str:
    """Hybrid prompt (Listing 4.2): preamble + state + observed events + instruction."""
    return (
        PROMPT_PREAMBLE
        + f"State: [STATE_{state.upper()}]\n"
        + f"Observed Events: {fragment}\n"
        + PROMPT_INSTRUCTION
    )


def build_finetuned_prompt(state: str, fragment: str) -> str:
    """Instruction used by the fine-tuned model: state and events only, no preamble."""
    return f"State: [STATE_{state.upper()}]\nObserved Events: {fragment}"


# Template the adapter was trained with (scripts/04_qlora_finetune.py). The
# fine-tuned model is queried in Ollama's raw mode with this exact text, because
# Ollama applies the chat template embedded in the GGUF file otherwise.
FINETUNED_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def wrap_finetuned(instruction: str) -> str:
    return FINETUNED_TEMPLATE.format(instruction=instruction)


class OllamaReasoner:
    """Sends prompts to a locally hosted LLM via the Ollama REST API and records
    the latency decomposition reported by Ollama for every request."""

    def __init__(self, config: OllamaConfig, options: Optional[Dict] = None):
        self.url = config.url
        self.model = config.model
        self.options = dict(DEFAULT_OPTIONS, **(options or {}))

    def generate(self, prompt: str, keep_alive: str = "10m", raw: bool = False) -> Dict:
        """Send one prompt and return the reply together with Ollama's timing fields.

        Durations are converted from nanoseconds to milliseconds:
          prefill_ms = prompt_eval_duration, decode_ms = eval_duration.
        """
        start = time.perf_counter()
        try:
            resp = requests.post(
                self.url,
                json={
                    "model":      self.model,
                    "prompt":     prompt,
                    "stream":     False,
                    "options":    self.options,
                    "keep_alive": keep_alive,
                    "raw":        raw,
                },
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            error = ""
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            data, error = {}, str(e)
        wall_ms = (time.perf_counter() - start) * 1000

        eval_count = data.get("eval_count", 0)
        eval_ns = data.get("eval_duration", 0)
        return {
            "response":          data.get("response", "").strip(),
            "error":             error,
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "prefill_ms":        data.get("prompt_eval_duration", 0) / 1e6,
            "eval_count":        eval_count,
            "decode_ms":         eval_ns / 1e6,
            "load_ms":           data.get("load_duration", 0) / 1e6,
            "total_ms":          data.get("total_duration", 0) / 1e6,
            "wall_ms":           wall_ms,
            "tokens_per_sec":    eval_count / (eval_ns / 1e9) if eval_ns > 0 else 0.0,
            "capped":            eval_count >= self.options["num_predict"],
            "done_reason":       data.get("done_reason", ""),
        }

    def reason(self, semantic_tokens: List[str], cluster_label: int) -> Dict:
        """Build the hybrid prompt for one sample and query the model."""
        state = CLUSTER_NAMES.get(cluster_label, "unknown")
        prompt = build_hybrid_prompt(state, semantic_fragment(semantic_tokens))
        r = self.generate(prompt)
        r.update(check_reply(r["response"]))
        r["prompt"] = prompt
        r["latency_ms"] = r["wall_ms"]
        return r

    def smoke_test(self, outputs: List[Dict], n: int = 3) -> List[Dict]:
        """Run quick inference on the first *n* output records and log results."""
        results = []
        for i in range(min(n, len(outputs))):
            o = outputs[i]
            tokens = o["semantic_tokens"]
            label  = o["prediction"]["label"]
            result = self.reason(tokens, label)
            results.append(result)

            logger.info(
                f"Sample {i}  tokens={tokens or ['[NO_EVENT]']}  "
                f"behaviour={CLUSTER_NAMES.get(label, '?')}  "
                f"latency={result['wall_ms']:.0f}ms  "
                f"tps={result['tokens_per_sec']:.2f}"
            )
            logger.info(f"  → {result['response']}")
        return results

    def benchmark(
        self,
        outputs: List[Dict],
        progress_interval: int = 20,
        limit: int = None,
    ) -> List[Dict]:
        """
        Run LLM reasoning on every output record.
        Returns a list of per-sample benchmark dicts.
        """
        import numpy as np

        records: List[Dict] = []
        if limit is not None:
            outputs = outputs[:limit]
        total = len(outputs)
        logger.info(f"Starting benchmark on {total} samples...")

        for i, o in enumerate(outputs):
            tokens = o["semantic_tokens"]
            label  = o["prediction"]["label"]
            r      = self.reason(tokens, label)

            records.append({
                "sample":            i,
                "cluster":           label,
                "behaviour":         CLUSTER_NAMES.get(label, "?"),
                "n_tokens_in":       len(tokens),
                "is_anomaly":        "[ANOMALY_ALERT]" in tokens,
                "latency_ms":        round(r["wall_ms"], 1),
                "prompt_eval_count": r["prompt_eval_count"],
                "prefill_ms":        round(r["prefill_ms"], 1),
                "tokens_out":        r["eval_count"],
                "decode_ms":         round(r["decode_ms"], 1),
                "tokens_per_sec":    round(r["tokens_per_sec"], 2),
                "schema_valid":      r["schema_valid"],
                "action":            r["action"],
                "ram_mb":            round(psutil.Process().memory_info().rss / 1e6, 1),
                "reasoning":         r["response"],
            })

            if (i + 1) % progress_interval == 0:
                avg_lat = np.mean([x["latency_ms"] for x in records])
                logger.info(f"  [{i+1}/{total}]  avg latency so far: {avg_lat:.0f} ms")

        return records
