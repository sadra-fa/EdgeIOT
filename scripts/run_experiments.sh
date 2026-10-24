#!/usr/bin/env bash
# Runs the LLM experiments of the dissertation one after another (nothing else
# should use the GPU meanwhile). Usage: bash scripts/run_experiments.sh <llm_inputs.csv>
set -e
INPUTS=${1:-outputs/rerun/20261017_143419_llama3.2_3b/llm_inputs.csv}
MODELS="llama3.2:3b phi3.5:latest qwen2.5:1.5b"
PY=./.venv/bin/python
$PY scripts/bench_llm.py multi --inputs "$INPUTS" --n 100 --models $MODELS
$PY scripts/bench_llm.py tokenization --inputs "$INPUTS" --n 100 --models $MODELS
$PY scripts/bench_llm.py probe --models $MODELS
$PY scripts/02_make_train_data.py --inputs "$INPUTS" --teacher llama3.2:3b --n 1000 --outdir data/finetune
$PY scripts/02_make_train_data.py --inputs "$INPUTS" --teacher qwen2.5:1.5b --n 1000 --outdir data/finetune_qwen
$PY scripts/02_make_train_data.py --inputs "$INPUTS" --teacher phi3.5:latest --n 1000 --outdir data/finetune_phi
echo ALL_DONE
