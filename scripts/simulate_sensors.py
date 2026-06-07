#!/usr/bin/env python3
"""
Simulate smart-home sensor readings for the EdgeIOT demo.

Generates JSON objects with temperature, motion, and noise values.
Non-deterministic by default; pass --seed for reproducible runs.

Usage:
    python scripts/simulate_sensors.py -n 5          # 5 readings, batch
    python scripts/simulate_sensors.py -n 1 --seed 42  # reproducible single reading
"""
import json
import random
import argparse
from datetime import datetime, timezone


def generate_reading() -> dict:
    """Return a single simulated sensor snapshot."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temperature": round(random.uniform(18.0, 35.0), 1),   # °C
        "motion": random.choice([True, False]),
        "noise": round(random.uniform(25.0, 85.0), 1),         # dB
    }


def main():
    parser = argparse.ArgumentParser(description="Simulate smart-home sensors")
    parser.add_argument(
        "-n", "--num", type=int, default=1,
        help="Number of readings to emit (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Optional random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    for _ in range(args.num):
        print(json.dumps(generate_reading()))


if __name__ == "__main__":
    main()
