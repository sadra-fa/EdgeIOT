#!/usr/bin/env python3
"""
02_make_train_data.py — Build the instruction-tuning corpus for the fine-tuned model.

Self-distillation with a validity filter:
  1. Take the tokenized test rows written by main.py (llm_inputs.csv) and drop
     the rows used for evaluation (the same seeded sample bench_llm.py draws).
  2. Ask the base model (teacher) once per distinct (state, tokens) combination,
     with the full hybrid prompt. Keep the reply if a JSON object with a string
     "reasoning" and a canonical action can be recovered from it (strictly, or
     from inside a code fence or after a preamble); otherwise retry with another
     seed, up to three times. The kept answer is re-serialised as plain JSON, so
     every training target is schema-valid.
  3. Draw --n rows from the remaining pool and write one JSONL record per row:
       instruction = state + observed events (no preamble, no action list)
       output      = the teacher's validated JSON, re-serialised compactly

Usage:
    python scripts/02_make_train_data.py --inputs outputs/rerun/<run>/llm_inputs.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edgeiot.config import OllamaConfig                                   # noqa: E402
from edgeiot.semantic.actions import check_reply, extract_answer                          # noqa: E402
from edgeiot.semantic.llm_reasoner import (                               # noqa: E402
    OllamaReasoner, build_finetuned_prompt, build_hybrid_prompt, semantic_fragment,
)


def main():
    p = argparse.ArgumentParser(description="Build the fine-tuning corpus")
    p.add_argument("--inputs", required=True, help="llm_inputs.csv written by main.py")
    p.add_argument("--teacher", default="llama3.2:3b")
    p.add_argument("--n", type=int, default=1000, help="training rows to write")
    p.add_argument("--eval-n", type=int, default=100, help="size of the held-out evaluation sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="data/finetune")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.inputs)
    df["tokens"] = df["tokens"].fillna("")
    held_out = df.sample(n=args.eval_n, random_state=args.seed)
    pool = df.drop(held_out.index)
    combos = pool[["state", "tokens"]].drop_duplicates().reset_index(drop=True)
    print(f"Pool: {len(pool)} rows, {len(combos)} distinct (state, tokens) combinations")

    teacher = OllamaReasoner(OllamaConfig(model=args.teacher))
    log, answers = [], {}
    for _, c in combos.iterrows():
        fragment = semantic_fragment(c["tokens"].split())
        prompt = build_hybrid_prompt(c["state"], fragment)
        for attempt, seed in enumerate((args.seed, 1, 2, 3)):
            teacher.options["seed"] = seed
            r = teacher.generate(prompt)
            check = check_reply(r["response"])
            answer = extract_answer(r["response"])
            log.append({"state": c["state"], "tokens": c["tokens"], "attempt": attempt,
                        "seed": seed, **check, "recovered": answer is not None,
                        "eval_count": r["eval_count"], "response": r["response"]})
            if answer is not None:
                answers[(c["state"], c["tokens"])] = json.dumps(answer, ensure_ascii=False)
                break
        status = "ok" if (c["state"], c["tokens"]) in answers else "dropped"
        print(f"  {c['state']:<9} {fragment:<50} {status}", flush=True)

    pd.DataFrame(log).to_csv(out_dir / "teacher_log.csv", index=False)
    usable = pool[[(s, t) in answers for s, t in zip(pool["state"], pool["tokens"])]]
    train = usable.sample(n=min(args.n, len(usable)), random_state=args.seed)

    with (out_dir / "train.jsonl").open("w") as f:
        for _, row in train.iterrows():
            record = {
                "instruction": build_finetuned_prompt(row["state"], semantic_fragment(row["tokens"].split())),
                "output": answers[(row["state"], row["tokens"])],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "pool_rows": len(pool), "combinations": len(combos),
        "valid_combinations": len(answers), "train_rows": len(train),
        "teacher": args.teacher, "held_out_rows": int(len(held_out)),
    }
    (out_dir / "corpus_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
