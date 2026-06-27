#!/usr/bin/env python3
"""
run_demo.py – Single-shot smart-home Edge IoT demo.

Simulates sensor readings, sends each to the LLM via edgeiot.brain,
and logs every cycle (sensor JSON, prompt, raw reply, parsed action,
latency) into a timestamped CSV under outputs/.

Usage
-----
    python scripts/run_demo.py              # 1 reading  (default)
    python scripts/run_demo.py -n 5         # 5 readings, batch
    python scripts/run_demo.py -n 5 -i 2    # 5 readings, 2 s apart
    python scripts/run_demo.py --style raw   # use 'raw' prompt style
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure the project root is on sys.path so `edgeiot` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.simulate_sensors import generate_reading   # noqa: E402
from edgeiot.brain import run as brain_run               # noqa: E402
from edgeiot.config import OllamaConfig                  # noqa: E402

# CSV field definitions (including model column)
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
    "previous_action",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run the smart-home Edge IoT demo (pre-fine-tuning baseline)",
    )
    parser.add_argument(
        "-n", "--num", type=int, default=1,
        help="Number of simulated sensor cycles (default: 1)",
    )
    parser.add_argument(
        "-i", "--interval", type=float, default=0,
        help="Seconds to wait between cycles (0 = batch mode)",
    )
    parser.add_argument(
        "--style", type=str, default="hybrid",
        choices=["raw", "compact", "hybrid"],
        help="Prompt style to use (default: hybrid)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:3b",
        help="Ollama model name (default: llama3.2:3b)",
    )
    parser.add_argument(
        "--desired-temp",
        type=float,
        default=None,
        help="Desired temperature in °C (optional)",
    )
    parser.add_argument(
        "--outdir", type=str, default="outputs",
        help="Base directory for CSV logs (default: outputs/)",
    )
    args = parser.parse_args()

    # Prepare output directory and CSV file
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use a single persistent CSV file for all runs
    csv_path = out_dir / "baseline_demo.csv"

    config = OllamaConfig(model=args.model)

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║   EdgeIOT Smart-Home Demo (pre-fine-tune)    ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  Model  : {args.model:<34}║")
    print(f"║  Style  : {args.style:<34}║")
    print(f"║  Cycles : {args.num:<34}║")
    print(f"║  Log    : {str(csv_path):<34}║")
    print(f"╚══════════════════════════════════════════════╝")
    print()

    # Append to existing CSV; write header only if file is new
    file_exists = csv_path.exists()
    # History of events for this run (sensor + action)
    history: list[dict] = []
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for i in range(args.num):
            # 1. Simulate a sensor reading
            sensor = generate_reading()

            print(f"── Cycle {i + 1}/{args.num} ──────────────────────────────")
            print(f"  Sensor → temp={sensor['temperature']}°C  "
                  f"motion={'yes' if sensor['motion'] else 'no'}  "
                  f"noise={sensor['noise']} dB")

            # 2. Send to brain (prompt + Ollama + parse)
            # Determine previous action for CSV (empty for first cycle)
            previous_action = history[-1]["action"]["action"] if history else ""
            result = brain_run(
                sensor,
                config=config,
                style=args.style,
                desired_temp=args.desired_temp,
                history=history,
            )

            print(f"  LLM    → {result['raw_reply'][:500]}{'…' if len(result['raw_reply']) > 500 else ''}")
            print(f"  Action → {json.dumps(result['action'])}")
            print(f"  Latency: {result['latency_ms']:.0f} ms")
            print()

            # 3. Write to CSV
            writer.writerow({
                "timestamp":    sensor["timestamp"],
                "temperature":  sensor["temperature"],
                "motion":       sensor["motion"],
                "noise":        sensor["noise"],
                "prompt_style": args.style,
                "prompt":       result["prompt"],
                "raw_reply":    result["raw_reply"],
                "action_json":  json.dumps(result["action"]),
                "latency_ms":   f"{result['latency_ms']:.2f}",
                "model":        args.model,
                "previous_action": previous_action,
            })
            # Append this event to history for the next cycle
            history.append({"sensor": sensor, "action": result["action"]})
            f.flush()   # make sure partial results survive crashes

            # 4. Wait between cycles (if streaming)
            if args.interval and i < args.num - 1:
                time.sleep(args.interval)

    print(f"✅ Demo complete — results saved to {csv_path}")


if __name__ == "__main__":
    main()
