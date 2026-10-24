#!/usr/bin/env bash
# Export, evaluation and quantization-chain diagnosis of the three fine-tuned models.
# Run after run_finetune.sh. Usage: bash scripts/run_ft_chain_all.sh <llm_inputs.csv>
#
# Export chain (05_export_to_ollama.py, default --merge_base nf4):
#   adapter -> merged into the dequantized NF4 weights it was trained on
#           -> GGUF FP16 -> llama.cpp quantization -> Ollama
# Variants kept for the chain table (20 rows each):
#   *-ft-fp16merge : merged into the original FP16 weights, base format
#   *-ft-q8        : NF4 merge, Q8_0
#   *-ft-merged    : NF4 merge, Q4_K_M (main fine-tuned model)
#   *-ft-q40       : NF4 merge, Q4_0 (Phi only; Phi's base format)
set -e
INPUTS=$1
PY=./.venv/bin/python
SP=$($PY -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$SP/nvpl/lib:$SP/nvidia/cu13/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
free_gpu() { for m in $(ollama ps | awk 'NR>1 {print $1}'); do ollama stop "$m"; done; }
export_model() {  # base_dir adapter_dir merged_dir name qtype base_tag [merge_base]
  free_gpu
  $PY scripts/05_export_to_ollama.py --base_dir "$1" --adapter_dir "$2" --merged_dir "$3" \
      --name "$4" --qtype "$5" --base_tag "$6" --merge_base "${7:-nf4}"
}

LLAMA=languagemodels/Llama-3.2-3B-Instruct
QWEN=languagemodels/Qwen2.5-1.5B-Instruct
PHI=languagemodels/Phi-3.5-mini-instruct

# 1. LLaMA: NF4 merge in Q4_K_M and Q8_0, and the FP16-merge variant
export_model $LLAMA languagemodels/fine_tuned_llama3_lora languagemodels/fine_tuned_llama3_merged \
    edgeiot-llama3.2-ft-merged Q4_K_M llama3.2:3b
export_model $LLAMA languagemodels/fine_tuned_llama3_lora languagemodels/fine_tuned_llama3_merged \
    edgeiot-llama3.2-ft-q8 Q8_0 llama3.2:3b
export_model $LLAMA languagemodels/fine_tuned_llama3_lora languagemodels/fine_tuned_llama3_merged_fp16 \
    edgeiot-llama3.2-ft-fp16merge Q4_K_M llama3.2:3b fp16
echo LLAMA_EXPORT_DONE

# 2. Qwen: NF4 merge in Q4_K_M, Q8_0 and unquantized, and the FP16-merge variant
export_model $QWEN languagemodels/fine_tuned_qwen25_lora languagemodels/fine_tuned_qwen25_merged \
    edgeiot-qwen2.5-ft-merged Q4_K_M qwen2.5:1.5b
export_model $QWEN languagemodels/fine_tuned_qwen25_lora languagemodels/fine_tuned_qwen25_merged \
    edgeiot-qwen2.5-ft-q8 Q8_0 qwen2.5:1.5b
export_model $QWEN languagemodels/fine_tuned_qwen25_lora languagemodels/fine_tuned_qwen25_merged \
    edgeiot-qwen2.5-ft-f16 F16 qwen2.5:1.5b
export_model $QWEN languagemodels/fine_tuned_qwen25_lora languagemodels/fine_tuned_qwen25_merged_fp16 \
    edgeiot-qwen2.5-ft-fp16merge Q4_K_M qwen2.5:1.5b fp16
echo QWEN_EXPORT_DONE

# 3. Phi (adapter trained by run_finetune.sh)
# Phi's base model is Q4_0, but llama.cpp's plain Q4_0 breaks the fine-tuned Phi (it emits the
# end token at once; see the chain table), so the main fine-tuned Phi is served in Q4_K_M.
export_model $PHI languagemodels/fine_tuned_phi35_lora languagemodels/fine_tuned_phi35_merged \
    edgeiot-phi3.5-ft-merged Q4_K_M phi3.5:latest
export_model $PHI languagemodels/fine_tuned_phi35_lora languagemodels/fine_tuned_phi35_merged \
    edgeiot-phi3.5-ft-q40 Q4_0 phi3.5:latest
export_model $PHI languagemodels/fine_tuned_phi35_lora languagemodels/fine_tuned_phi35_merged \
    edgeiot-phi3.5-ft-q8 Q8_0 phi3.5:latest
echo PHI_EXPORT_DONE

# 4. Main evaluation: base vs fine-tuned on the same 100 rows, and the probe set
free_gpu
$PY scripts/bench_llm.py finetune --inputs "$INPUTS" --n 100 --models llama3.2:3b edgeiot-llama3.2-ft-merged
$PY scripts/bench_llm.py finetune --tag qwen --inputs "$INPUTS" --n 100 --models qwen2.5:1.5b edgeiot-qwen2.5-ft-merged
$PY scripts/bench_llm.py finetune --tag phi --inputs "$INPUTS" --n 100 --models phi3.5:latest edgeiot-phi3.5-ft-merged
$PY scripts/bench_llm.py probe --tag ft --models edgeiot-llama3.2-ft-merged edgeiot-qwen2.5-ft-merged edgeiot-phi3.5-ft-merged
echo MAIN_EVAL_DONE

# 5. Quantization-chain table: 20 rows per variant
$PY scripts/bench_llm.py finetune --tag chain --inputs "$INPUTS" --n 20 --models \
    edgeiot-llama3.2-ft-fp16merge edgeiot-llama3.2-ft-q8 edgeiot-llama3.2-ft-merged \
    edgeiot-qwen2.5-ft-fp16merge edgeiot-qwen2.5-ft-f16 edgeiot-qwen2.5-ft-q8 edgeiot-qwen2.5-ft-merged \
    edgeiot-phi3.5-ft-q40 edgeiot-phi3.5-ft-q8 edgeiot-phi3.5-ft-merged
free_gpu
for spec in "$LLAMA languagemodels/fine_tuned_llama3_lora" "$QWEN languagemodels/fine_tuned_qwen25_lora" \
            "$PHI languagemodels/fine_tuned_phi35_lora"; do
  set -- $spec
  $PY scripts/diagnose_ft_chain.py --base_dir "$1" --adapter_dir "$2" --n 20 --precisions nf4
done
echo ALL_FT_DONE
