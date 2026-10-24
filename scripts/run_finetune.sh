#!/usr/bin/env bash
# QLoRA training of the three models on their self-distilled corpora
# (data/finetune*, built by run_experiments.sh). Export and evaluation:
# run_ft_chain_all.sh. Usage: bash scripts/run_finetune.sh
set -e
PY=./.venv/bin/python
SP=$($PY -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$SP/nvpl/lib:$SP/nvidia/cu13/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Free the shared memory for training: unload every model Ollama has resident
for m in $(ollama ps | awk 'NR>1 {print $1}'); do ollama stop "$m"; done

$PY scripts/04_qlora_finetune.py --model_dir languagemodels/Llama-3.2-3B-Instruct \
    --train_file data/finetune/train.jsonl --output_dir languagemodels/fine_tuned_llama3_lora
$PY scripts/04_qlora_finetune.py --model_dir languagemodels/Qwen2.5-1.5B-Instruct \
    --train_file data/finetune_qwen/train.jsonl --output_dir languagemodels/fine_tuned_qwen25_lora
# Phi-3.5 has a fused query/key/value projection; a smaller batch is used if memory runs out
$PY scripts/04_qlora_finetune.py --model_dir languagemodels/Phi-3.5-mini-instruct \
    --train_file data/finetune_phi/train.jsonl --output_dir languagemodels/fine_tuned_phi35_lora \
    --target_modules qkv_proj \
  || $PY scripts/04_qlora_finetune.py --model_dir languagemodels/Phi-3.5-mini-instruct \
       --train_file data/finetune_phi/train.jsonl --output_dir languagemodels/fine_tuned_phi35_lora \
       --target_modules qkv_proj --batch_size 1 --grad_accum 8
echo TRAINING_DONE
