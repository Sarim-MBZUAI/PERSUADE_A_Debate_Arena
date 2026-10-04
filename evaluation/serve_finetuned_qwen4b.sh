#!/bin/bash
# Serve a fine-tuned Qwen3-4B student (base checkpoint + LoRA adapter) for evaluation/test_finetuned_qwen4b.py.
# Usage: BASE_MODEL=<path to SFT or base model> LORA_ADAPTER=<path to LoRA adapter> bash evaluation/serve_finetuned_qwen4b.sh
: "${BASE_MODEL:?set BASE_MODEL to the base or SFT checkpoint}"
: "${LORA_ADAPTER:?set LORA_ADAPTER to the LoRA adapter directory}"

vllm serve "$BASE_MODEL" \
    --enable-lora \
    --lora-modules finetuned-qwen3-4b="$LORA_ADAPTER" \
    --served-model-name finetuned-qwen3-4b \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 24576 \
    --max-lora-rank 16 \
    --max-num-seqs 64 \
    --port 8000
