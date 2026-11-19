#!/usr/bin/env bash
# Serve Post-Trained `Qwen3.8-27B` Dense at "Flash" Latency (Single H100 80GB or L40S 48GB)
# Enables FP8 KV-cache + FlashInfer + Qwen3 reasoning & tool-call parsers (+ optional MTP in vLLM)
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/nvme/checkpoints/qwen38-27b-sre-stage3-final-merged}"
if [[ ! -d "${MODEL_PATH}" ]]; then
  MODEL_PATH="${FALLBACK_MODEL:-Qwen/Qwen3.8-27B-FP8}"
fi

ENGINE="${ENGINE:-sglang}"   # sglang | vllm
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"

if [[ "${ENGINE}" == "sglang" ]]; then
  exec python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --tp-size 1 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --attention-backend flashinfer \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --host "${HOST}" \
    --port "${PORT}"
else
  exec vllm serve "${MODEL_PATH}" \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --max-model-len 131072 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.88 \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
    --host "${HOST}" \
    --port "${PORT}"
fi
