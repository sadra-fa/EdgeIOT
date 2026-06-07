#!/usr/bin/env python3
"""
Stream‑demo: simulate sensor data → on‑device LLM → CSV log.

Usage:
    python scripts/stream_demo.py [--interval SECS] [--model MODEL] [--outdir DIR]

Options:
    --interval   Seconds between sensor reads (default: 0 → as fast as possible)
    --model      Ollama model name (default: llama3.2:3b)
    --outdir     Directory where the CSV is stored (default: outputs)
"""

import argparse
import csv
import signal
import time
import sys
import json
from pathlib import Path

# Ensure the scripts directory is on sys.path when running via `python scripts/stream_demo.py`
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Ensure the project root (the package directory) is on sys.path so `edgeiot` can be imported
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Local helpers (now importable after path adjustment)
from simulate_sensors import generate_reading
from edgeiot.brain import run as brain_run
from edgeiot.config import OllamaConfig

_stop_requested = False

def _handle_sigint(signum, frame):
    global _stop_requested
    _stop_requested = True
    print("\n🛑 Interrupt received – finishing current cycle…")

signal.signal(signal.SIGINT, _handle_sigint)


def main():
    parser = argparse.ArgumentParser(description="Stream sensor → LLM → CSV")
    parser.add_argument("--interval", type=float, default=0,
                        help="Seconds between sensor reads (default: 0)")
    parser.add_argument("--model", type=str, default="llama3.2:3b",
                        help="Ollama model name (default: llama3.2:3b)")
    parser.add_argument("--outdir", type=str, default="outputs",
                        help="Directory to store the CSV log")
    args = parser.parse_args()

    # Prepare CSV
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stream_demo.csv"
    FIELDNAMES = [
        "timestamp",
        "temperature",
        "motion",
        "noise",
        "prompt_style",
        "prompt",
        "raw_reply",
        "action_json",
        "latency_ms",
        "model",
    ]
    file_exists = csv_path.exists()
    csv_file = csv_path.open("a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    # LLM configuration
    config = OllamaConfig(model=args.model)

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║   EdgeIOT Smart-Home Stream (on-device LLM)  ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  Model    : {args.model:<34}║")
    print(f"║  Interval : {str(args.interval) + ' s':<34}║")
    print(f"║  Log      : {str(csv_path):<34}║")
    print(f"╚══════════════════════════════════════════════╝")
    print()
    print("  Streaming… press Ctrl-C to stop.")
    print()

    cycle = 0
    while not _stop_requested:
        cycle += 1
        sensor = generate_reading()

        print(f"── Cycle {cycle} ─────────────────────────────────────")
        print(f"  Sensor  → temp={sensor['temperature']}°C  "
              f"motion={'yes' if sensor['motion'] else 'no'}  "
              f"noise={sensor['noise']} dB")

        result = brain_run(sensor, config=config, style="hybrid")

        print(f"  LLM     → {result['raw_reply'][:120]}{'…' if len(result['raw_reply']) > 120 else ''}")
        print(f"  Action  → {json.dumps(result['action'])}")
        print(f"  Latency : {result['latency_ms']:.0f} ms")
        print()

        writer.writerow({
            "timestamp": sensor["timestamp"],
            "temperature": sensor["temperature"],
            "motion": sensor["motion"],
            "noise": sensor["noise"],
            "prompt_style": "hybrid",
            "prompt": result["prompt"],
            "raw_reply": result["raw_reply"],
            "action_json": json.dumps(result["action"]),
            "latency_ms": f"{result['latency_ms']:.2f}",
            "model": args.model,
        })
        csv_file.flush()
        if args.interval and not _stop_requested:
            time.sleep(args.interval)

    csv_file.close()
    print(f"\n✅ Stream stopped after {cycle} cycles – data saved to {csv_path}")

if __name__ == "__main__":
    main()
