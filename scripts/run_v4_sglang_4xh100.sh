#!/usr/bin/env bash
# Run-v4 Production Pipeline on GCP `a3-highgpu-4g` (4x H100 80GB):
#   1. Phase 1b SFT Calibration (~15 min): Resumes from Stage 1a `sft-lora64`, locks in
#      100%-exact Tier-1 tool schemas on 1,559 grounded real scenarios, and merges BF16 weights.
#   2. Launches SGLang TP2 Rollout Server on GPUs 0,1 (`fp8_e4m3` KV cache + FlashInfer).
#   3. Launches DDP Rank-64 Hybrid LoRA DAPO+GTPO RLVR on GPUs 2,3 (~10x faster than HF .generate(),
#      200 steps x 64 trajectories = 12,800 interactive SRE episodes in ~4 hrs).
set -euo pipefail

NVME_ROOT="${NVME_ROOT:-/mnt/nvme}"
GCS_PREFIX="${GCS_PREFIX:-gs://onflow-30k-qwen38-sre-rl/run-v4-sglang}"
export HF_HOME="${NVME_ROOT}/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export NCCL_NET_PLUGIN="none"
unset LD_LIBRARY_PATH || true

MERGED_P1B="${NVME_ROOT}/checkpoints_v4/phase1b-sft-merged"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_PID=""

stop_sglang() {
  if [[ -n "${SGLANG_PID}" ]]; then
    kill "${SGLANG_PID}" 2>/dev/null || true
  fi
}
trap stop_sglang EXIT

echo "=== [1/3] Phase 1b Exact-Schema SFT Calibration + BF16 Merge (4x H100) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 \
  scripts/train_sft_qwen38_lora.py \
  --config configs/sft_qwen38_27b_lora_4xh100.yaml \
  --resume-adapter-dir "${NVME_ROOT}/checkpoints_v2/sft-lora64" \
  --train-jsonl "${NVME_ROOT}/data_v4_phase1b/sft_phase1b_exact_schema.jsonl" \
  --output-dir "${NVME_ROOT}/checkpoints_v4/phase1b-sft-lora64" \
  --merge-output-dir "${MERGED_P1B}" \
  --max-steps 98

echo "=== Backing up Phase 1b SFT adapter to GCS ==="
gcloud storage rsync -r "${NVME_ROOT}/checkpoints_v4/phase1b-sft-lora64" \
  "${GCS_PREFIX}/phase1b-sft-lora64" --exclude="checkpoint-.*" --quiet || true

echo "=== [2/3] Launching SGLang TP2 FP8-KV Rollout Server on GPUs 0,1 ==="
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model-path "${MERGED_P1B}" \
  --served-model-name qwen38-sre \
  --trust-remote-code \
  --tp-size 2 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.82 \
  --attention-backend flashinfer \
  --chunked-prefill-size 16384 \
  --reasoning-parser qwen3 \
  --host 127.0.0.1 \
  --port "${SGLANG_PORT}" \
  > "${NVME_ROOT}/sglang_rollout_server.log" 2>&1 &
SGLANG_PID=$!

for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${SGLANG_PORT}/health" >/dev/null 2>&1; then
    echo "[sglang] Rollout server ready on port ${SGLANG_PORT}!"
    break
  fi
  sleep 5
done

echo "=== [3/3] Starting Split-GPU SGLang + DDP Multi-Turn DAPO+GTPO RLVR on GPUs 2,3 ==="
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29502 \
  scripts/train_rl_sglang_4xh100.py \
  --base-model-path "${MERGED_P1B}" \
  --sglang-url "http://127.0.0.1:${SGLANG_PORT}" \
  --served-model qwen38-sre \
  --sync-dir "${NVME_ROOT}/sglang_live_sync" \
  --output-dir "${NVME_ROOT}/checkpoints_v4/rl-sglang-dapo-gtpo" \
  --train-scenarios-jsonl "${NVME_ROOT}/data_v2/rl_real_train.jsonl" \
  --heldout-scenarios-jsonl "${NVME_ROOT}/data_v2/rl_real_heldout.jsonl" \
  --steps 200 \
  --prompts-per-step 8 \
  --group-size 8 \
  --rollout-workers 32 \
  --eval-heldout-count 40 \
  --max-turns 14 \
  --min-group-max-reward 0.50 \
  --lr 2e-6 \
  --kl-coef 0.04 \
  --eval-every 25 \
  --save-every 25

echo "=== Uploading final Run-v4 SGLang RLVR checkpoints & eval summary to GCS ==="
gcloud storage rsync -r "${NVME_ROOT}/checkpoints_v4/rl-sglang-dapo-gtpo" \
  "${GCS_PREFIX}/rl-sglang-dapo-gtpo" --quiet || true
