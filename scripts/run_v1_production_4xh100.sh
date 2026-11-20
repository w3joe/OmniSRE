#!/usr/bin/env bash
# Full Production OmniSRE v1.0 Run on GCP `a3-highgpu-4g` (4x H100 80GB, 3TB NVMe)
# - Uses in-memory cached real production telemetry (`DejaVu` Bank + `RCAEval` GKE + optional `OpenRCA`)
# - Runs 250 steps of Masked CISPO (zero KL) + TITO + 2x PODS + 25% Offline Synthesis
# - Syncs SGLang weights every 2 steps (max_staleness <= 2, cutting sync overhead 50%)
# - Automatically backs up `step-025..250` & `best_checkpoint` to GCS for Spot preemption safety
# - Runs final `1x` vs. `5x` Tree Ensemble evaluation on 100% unseen real systems (`DejaVu-C` + `RE3-TT`)
set -euo pipefail

NVME_ROOT="${NVME_ROOT:-/mnt/nvme}"
GCS_V1="${GCS_V1:-gs://onflow-30k-qwen38-sre-rl/omnisre-v1.0}"
BASE_MODEL="${BASE_MODEL:-${NVME_ROOT}/checkpoints_v6/omnisre-v5-merged}"
STEPS="${STEPS:-250}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_PID=""

export HF_HOME="${NVME_ROOT}/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export NCCL_NET_PLUGIN="none"
unset LD_LIBRARY_PATH || true

stop_sglang() {
  if [[ -n "${SGLANG_PID}" ]]; then
    kill "${SGLANG_PID}" 2>/dev/null || true
  fi
}
trap stop_sglang EXIT

echo "=== [v1.0 1/2] Launching SGLang 0.5.23 TP2 on GPUs 0,1 ==="
PATH="/mnt/nvme/venv-sglang/bin:${PATH}" CUDA_VISIBLE_DEVICES=0,1 \
  /mnt/nvme/venv-sglang/bin/python3 -m sglang.launch_server \
  --model-path "${BASE_MODEL}" \
  --served-model-name omnisre-27b-v1 \
  --trust-remote-code \
  --tp-size 2 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.82 \
  --attention-backend flashinfer \
  --chunked-prefill-size 16384 \
  --host 127.0.0.1 \
  --port "${SGLANG_PORT}" \
  > "${NVME_ROOT}/sglang_v1_server.log" 2>&1 &
SGLANG_PID=$!

READY=0
for _ in $(seq 1 90); do
  if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
    echo "[v1.0] ERROR: SGLang exited!" >&2
    tail -n 50 "${NVME_ROOT}/sglang_v1_server.log" >&2 || true
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${SGLANG_PORT}/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 5
done
[[ "${READY}" -eq 1 ]] || exit 1

echo "=== [v1.0 2/2] Running Full OmniSRE v1.0 (${STEPS} steps, in-memory real telemetry, sync-every=2, GCS Spot backups) ==="
CUDA_VISIBLE_DEVICES=2,3 /mnt/nvme/venv-sre-rl/bin/torchrun --nproc_per_node=2 --master_port=29502 \
  scripts/train_rl_v6_real_telemetry_4xh100.py \
  --base-model-path "${BASE_MODEL}" \
  --sglang-url "http://127.0.0.1:${SGLANG_PORT}" \
  --sync-dir "${NVME_ROOT}/sglang_live_sync_v1" \
  --output-dir "${NVME_ROOT}/checkpoints_v1/omnisre-27b-v1.0" \
  --train-index-jsonl "${NVME_ROOT}/real_telemetry/index/real_telemetry_train.jsonl" \
  --heldout-index-jsonl "${NVME_ROOT}/real_telemetry/index/real_telemetry_heldout.jsonl" \
  --steps "${STEPS}" \
  --prompts-per-step 8 \
  --group-size 8 \
  --rollout-workers 24 \
  --eval-heldout-count 40 \
  --max-turns 12 \
  --lr 3e-6 \
  --sync-every 2 \
  --eval-every 25 \
  --save-every 25 \
  --gcs-backup-prefix "${GCS_V1}"

gcloud storage rsync -r "${NVME_ROOT}/checkpoints_v1/omnisre-27b-v1.0" "${GCS_V1}/final" --quiet
echo "=== OmniSRE v1.0 COMPLETE & BACKED UP TO ${GCS_V1} ==="
