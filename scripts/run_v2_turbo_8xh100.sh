#!/usr/bin/env bash
# OmniSRE v2.0 High-Throughput Raw-Exploration RLVR Launcher (`8x H100` or `4x H100`)
# - Auto-detects GPU count:
#     * `8x H100` (`a3-highgpu-8g`): Launches `6 x TP=1` FP8 SGLang servers (GPUs 0..5) + `2 x DDP` Trainer (GPUs 6,7)
#     * `4x H100` (`a3-highgpu-4g`): Launches `2 x TP=1` FP8 SGLang servers (GPUs 0,1)  + `2 x DDP` Trainer (GPUs 2,3)
# - Uses `/dev/shm` RAM-disk weight sync (`~3-5s` broadcast reload across all SGLang workers)
# - Runs 60-100 turn raw DuckDB SQL/PromQL/Log/Trace haystack investigations with `compact_workspace`
set -euo pipefail

NVME_ROOT="${NVME_ROOT:-/mnt/nvme}"
GCS_BUCKET="${GCS_BUCKET:-gs://onflow-30k-qwen38-sre-rl}"
GCS_V2="${GCS_V2:-${GCS_BUCKET}/omnisre-v2.0}"
BASE_MODEL="${BASE_MODEL:-${NVME_ROOT}/omnisre-v1.0}"
STEPS="${STEPS:-400}"
MAX_TURNS="${MAX_TURNS:-60}"
BASE_PORT="${BASE_PORT:-30000}"

export HF_HOME="${NVME_ROOT}/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export NCCL_NET_PLUGIN="none"
unset LD_LIBRARY_PATH || true

NUM_GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
if [[ "${NUM_GPUS}" -ge 8 ]]; then
  ROLLOUT_GPUS=(0 1 2 3 4 5)
  TRAINER_GPUS="6,7"
  WORKERS=192
  OVERSAMPLE=3
else
  ROLLOUT_GPUS=(0 1)
  TRAINER_GPUS="2,3"
  WORKERS=96
  OVERSAMPLE=2
fi

SGLANG_PIDS=()
cleanup() {
  for pid in "${SGLANG_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT

echo "=== [v2.0 1/2] Launching ${#ROLLOUT_GPUS[@]} Independent TP=1 FP8 SGLang Servers ==="
URLS=()
for idx in "${!ROLLOUT_GPUS[@]}"; do
  GPU_ID="${ROLLOUT_GPUS[$idx]}"
  PORT=$((BASE_PORT + idx))
  URLS+=("http://127.0.0.1:${PORT}")
  PATH="${NVME_ROOT}/venv-sglang/bin:${PATH}" CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${NVME_ROOT}/venv-sglang/bin/python3" -m sglang.launch_server \
    --model-path "${BASE_MODEL}" \
    --served-model-name omnisre-27b-v2 \
    --trust-remote-code \
    --tp-size 1 \
    --quantization fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --attention-backend flashinfer \
    --chunked-prefill-size 16384 \
    --host 127.0.0.1 \
    --port "${PORT}" \
    > "${NVME_ROOT}/sglang_v2_gpu${GPU_ID}.log" 2>&1 &
  SGLANG_PIDS+=("$!")
done

for idx in "${!ROLLOUT_GPUS[@]}"; do
  PORT=$((BASE_PORT + idx))
  READY=0
  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      READY=1
      break
    fi
    sleep 4
  done
  [[ "${READY}" -eq 1 ]] || { echo "ERROR: SGLang port ${PORT} failed to start"; exit 1; }
done
URL_CSV=$(IFS=,; echo "${URLS[*]}")
echo "[v2.0] All ${#ROLLOUT_GPUS[@]} TP=1 FP8 SGLang servers ready: ${URL_CSV}"

echo "=== [v2.0 2/2] Running OmniSRE v2.0 Turbo RLVR on GPUs ${TRAINER_GPUS} (sync via /dev/shm) ==="
CUDA_VISIBLE_DEVICES="${TRAINER_GPUS}" "${NVME_ROOT}/venv-sre-rl/bin/torchrun" --nproc_per_node=2 --master_port=29504 \
  scripts/train_rl_v2_turbo_8xh100.py \
  --base-model-path "${BASE_MODEL}" \
  --sglang-urls "${URL_CSV}" \
  --sync-dir "/dev/shm/omnisre_v2_sync" \
  --output-dir "${NVME_ROOT}/checkpoints_v2/omnisre-27b-v2.0" \
  --steps "${STEPS}" \
  --prompts-per-step 12 \
  --oversample-factor "${OVERSAMPLE}" \
  --group-size 8 \
  --rollout-workers "${WORKERS}" \
  --max-turns "${MAX_TURNS}" \
  --distractors 40 \
  --max-train-seq-len 5632 \
  --lr 3e-6 \
  --min-lr 3e-7 \
  --sync-every 2 \
  --eval-every 25 \
  --save-every 25 \
  --gcs-backup-prefix "${GCS_V2}"

gcloud storage rsync -r "${NVME_ROOT}/checkpoints_v2/omnisre-27b-v2.0" "${GCS_V2}/final" --quiet
echo "=== OmniSRE v2.0 COMPLETE & BACKED UP TO ${GCS_V2} ==="
