import logging
import time
import subprocess
from typing import Dict, List, Optional

import psutil
import requests

from edgeiot.config import OllamaConfig

logger = logging.getLogger(__name__)

# Human-readable cluster name mapping (matches original script)
CLUSTER_NAMES: Dict[int, str] = {0: "active", 1: "sleeping", 2: "away"}


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

class OllamaReasoner:
    """Sends semantic token prompts to a locally-hosted LLM via the Ollama REST API."""

    def __init__(self, config: OllamaConfig):
        self.url = config.url
        self.model = config.model
        self.options = {
            "temperature": 0.2,
            "num_predict": 80,      # keep short for edge speed
            "num_ctx":     512,     # smaller context = less VRAM
        }

    def reason(self, semantic_tokens: List[str], cluster_label: int) -> Dict:
        """
        Build a prompt from tokens + cluster label, send to Ollama, and return
        a dict with the LLM response and performance metrics.
        """
        token_str = " ".join(semantic_tokens) if semantic_tokens else "[NO_EVENT]"
        behaviour = CLUSTER_NAMES.get(cluster_label, "unknown")

        prompt = (
            "You are an AI assistant embedded in a smart home edge device.\n"
            f"Current sensor events : {token_str}\n"
            f"Predicted behaviour   : {behaviour}\n\n"
            "In one concise sentence, describe what is likely happening "
            "and if any action is needed."
        )

        start = time.perf_counter()
        try:
            resp = requests.post(
                self.url,
                json={
                    "model":   self.model,
                    "prompt":  prompt,
                    "stream":  False,
                    "options": self.options,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            reasoning     = data.get("response", "").strip()
            eval_count    = data.get("eval_count", 0)
            eval_duration = data.get("eval_duration", 0)       # nanoseconds
            load_duration = data.get("load_duration", 0)
            tps = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            reasoning      = f"[LLM ERROR: {e}]"
            eval_count     = 0
            eval_duration  = 0
            load_duration  = 0
            tps            = 0

        latency_ms = (time.perf_counter() - start) * 1000
        ram_mb     = psutil.Process().memory_info().rss / 1e6

        return {
            "reasoning":      reasoning,
            "latency_ms":     round(latency_ms, 1),
            "ram_mb":         round(ram_mb, 1),
            "tokens_out":     eval_count,
            "tokens_per_sec": round(tps, 2),
            "load_ms":        round(load_duration / 1e6, 1),
        }

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
                f"latency={result['latency_ms']}ms  "
                f"tps={result['tokens_per_sec']}"
            )
            logger.info(f"  → {result['reasoning']}")
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
                "sample":         i,
                "cluster":        label,
                "behaviour":      CLUSTER_NAMES.get(label, "?"),
                "n_tokens_in":    len(tokens),
                "is_anomaly":     "[ANOMALY_ALERT]" in tokens,
                "latency_ms":     r["latency_ms"],
                "tokens_out":     r["tokens_out"],
                "tokens_per_sec": r["tokens_per_sec"],
                "ram_mb":         r["ram_mb"],
                "reasoning":      r["reasoning"],
            })

            if (i + 1) % progress_interval == 0:
                avg_lat = np.mean([x["latency_ms"] for x in records])
                logger.info(f"  [{i+1}/{total}]  avg latency so far: {avg_lat:.0f} ms")

        return records
