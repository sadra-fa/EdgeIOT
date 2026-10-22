#!/usr/bin/env python3
"""
cache_replay.py — How much LLM work does a response cache save on the test set?

Replays the 6,000 tokenized test rows (llm_inputs.csv, in stored order) through
edgeiot.semantic.cache.ResponseCache. A cache miss needs one LLM decision; its
answer is taken from LLaMA 3.2's logged replies to the same input (the first
attempt, seed 42, of data/finetune/teacher_log.csv, which used the E1 prompt and
options), so no new inference is needed. The same replay with the raw fragments
as keys shows what happens without semantic tokenization.

Writes outputs/rerun/<run>/cache_replay.json and cache_curve.csv.
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edgeiot.semantic.actions import extract_answer                     # noqa: E402
from edgeiot.semantic.cache import ResponseCache, time_lookup           # noqa: E402

SENSORS = ["temperature", "humidity", "noise_db", "light_intensity", "motion_level"]
CHECKPOINTS = [10, 50, 100, 250, 500, 1000, 2000, 4000, 6000]


def llm_answers():
    log = pd.read_csv(PROJECT_ROOT / "data" / "finetune" / "teacher_log.csv")
    log["tokens"] = log["tokens"].fillna("")
    first = log[log["attempt"] == 0]
    answers = {}
    for _, r in first.iterrows():
        a = extract_answer(r["response"]) if isinstance(r["response"], str) else None
        answers[(r["state"], r["tokens"])] = (a["action_json"]["action"], a["reasoning"]) if a else (None, "")
    return answers


def replay(df, policy, answers):
    cache = ResponseCache(policy=policy)
    curve, llm_calls, unknown = [], 0, 0
    for i, row in enumerate(df.itertuples(index=False), 1):
        tokens = row.tokens.split() if row.tokens else []

        def ask():
            nonlocal llm_calls, unknown
            llm_calls += 1
            if (row.state, row.tokens) not in answers:
                unknown += 1
            return answers.get((row.state, row.tokens), (None, ""))

        cache.decide(row.state, tokens, ask)
        if i in CHECKPOINTS:
            curve.append({"rows": i, "llm_calls": llm_calls, "hit_rate": cache.hit_rate})
    return cache, curve, llm_calls, unknown


def main():
    run = sorted((PROJECT_ROOT / "outputs" / "rerun").glob("*"))[-1]
    df = pd.read_csv(run / "llm_inputs.csv")
    df["tokens"] = df["tokens"].fillna("")
    answers = llm_answers()

    results = {}
    for policy in ("rules", "flag"):
        cache, curve, calls, unknown = replay(df, policy, answers)
        results[policy] = {**cache.stats(), "llm_calls": calls, "answers_not_logged": unknown}
        if policy == "rules":
            pd.DataFrame(curve).to_csv(run / "cache_curve.csv", index=False)
            keys = [(s, t.split() if t else []) for s, t in zip(df["state"], df["tokens"])]
            results["lookup_us"] = time_lookup(cache, keys)

    raw_keys = df[[f"z_{s}" for s in SENSORS]].round(3).astype(str).agg(" ".join, axis=1)
    results["raw_distinct"] = int(raw_keys.nunique())
    results["raw_hit_rate"] = float((len(df) - raw_keys.nunique()) / len(df))
    results["rows"] = len(df)
    (run / "cache_replay.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
