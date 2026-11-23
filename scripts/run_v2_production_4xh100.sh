#!/usr/bin/env bash
# End-to-End OmniSRE v2.0 Production Pipeline Built on the Proven `v1.0` Architecture (`4x H100`)
#   Stage 1: Build Expanded 22-Fault K8s + OpenRCA/DejaVu/RCAEval Dataset (`~3,000` SFT + `~2,600` RL)
#   Stage 2: Assistant-Only Masked Rank-64 Hybrid LoRA Calibration SFT (`~15 min`) + Clean BF16 Merge
#   Stage 3: 80-Step Multi-World Masked CISPO RLVR (`2 x TP=1` BF16 SGLang + `/dev/shm` + Cosine LR + Live Kind K8s)
#   Stage 4: Standalone BF16 Merge (`best_checkpoint`) + GCS Backup
set -euo pipefail

NVME_ROOT="${NVME_ROOT:-/mnt/nvme}"
GCS_BUCKET="${GCS_BUCKET:-gs://onflow-30k-qwen38-sre-rl}"
GCS_V2="${GCS_V2:-${GCS_BUCKET}/omnisre-v2.0}"
BASE_V1="${BASE_V1:-${NVME_ROOT}/omnisre-v1.0}"
RL_STEPS="${RL_STEPS:-80}"
ENABLE_LIVE_KIND="${ENABLE_LIVE_KIND:-1}"
NUM_KIND_CLUSTERS="${NUM_KIND_CLUSTERS:-4}"

export CLOUDSDK_STORAGE_CHECK_HASHES="never"
export HF_HOME="${NVME_ROOT}/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export NCCL_NET_PLUGIN="none"
unset LD_LIBRARY_PATH || true

SGLANG_PIDS=()
cleanup() {
  for pid in "${SGLANG_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT

echo "=== [v2.0 1/5] Indexing Real Telemetry (DejaVu + RCAEval + OpenRCA) & Building Expanded 22-Fault Dataset ==="
mkdir -p "${NVME_ROOT}/real_telemetry/raw" "${NVME_ROOT}/real_telemetry/extracted" "${NVME_ROOT}/real_telemetry/index"
if [[ ! -d "${NVME_ROOT}/real_telemetry/extracted/DejaVu-C" ]]; then
  gcloud storage rsync -r "${GCS_BUCKET}/real-telemetry-raw" "${NVME_ROOT}/real_telemetry/raw" --quiet
fi
"${NVME_ROOT}/venv-sre-rl/bin/python3" scripts/index_real_telemetry.py \
  --raw-dir "${NVME_ROOT}/real_telemetry/raw" \
  --extracted-dir "${NVME_ROOT}/real_telemetry/extracted" \
  --out-dir "${NVME_ROOT}/real_telemetry/index"

if [[ -d "${NVME_ROOT}/real_telemetry/openrca/OpenRCA/dataset" ]]; then
  "${NVME_ROOT}/venv-sre-rl/bin/python3" scripts/download_openrca.py \
    --work-dir "${NVME_ROOT}/real_telemetry/openrca" \
    --out-dir "${NVME_ROOT}/real_telemetry/index" \
    --skip-download
fi

if [[ ! -f "${NVME_ROOT}/data_v2_expanded/sft_v2_expanded_calibration.jsonl" ]]; then
  "${NVME_ROOT}/venv-sre-rl/bin/python3" scripts/build_v2_expanded_dataset.py \
    --out-dir "${NVME_ROOT}/data_v2_expanded" \
    --existing-rl-train-jsonl "${NVME_ROOT}/data_v2/rl_real_train.jsonl" \
    --real-telemetry-train-jsonl "${NVME_ROOT}/real_telemetry/index/real_telemetry_train.jsonl" \
    --openrca-train-jsonl "${NVME_ROOT}/real_telemetry/index/openrca_train.jsonl"
  gcloud storage rsync -r "${NVME_ROOT}/data_v2_expanded" "${GCS_V2}/dataset-expanded" --quiet
fi

echo "=== [v2.0 2/5] Running Assistant-Masked Calibration SFT Starting from OmniSRE v1.0 ==="
SFT_MERGED="${NVME_ROOT}/omnisre-v2-sft-merged"
CUDA_VISIBLE_DEVICES=0,1,2,3 "${NVME_ROOT}/venv-sre-rl/bin/torchrun" --nproc_per_node=4 --master_port=29501 \
  scripts/train_sft_qwen38_lora.py \
  --config configs/sft_v2_expanded_4xh100.yaml \
  --model-path "${BASE_V1}" \
  --train-jsonl "${NVME_ROOT}/data_v2_expanded/sft_v2_expanded_calibration.jsonl" \
  --output-dir "${NVME_ROOT}/checkpoints_v2/sft-lora64" \
  --merge-output-dir "${SFT_MERGED}"
gcloud storage rsync -r "${SFT_MERGED}" "${GCS_V2}/sft-merged-27b-bf16" --quiet

if [[ "${ENABLE_LIVE_KIND}" == "1" ]] && command -v kind >/dev/null 2>&1 && command -v kubectl >/dev/null 2>&1; then
  echo "=== [v2.0 3/5] Bootstrapping ${NUM_KIND_CLUSTERS} Isolated Live Kind K8s Clusters ==="
  NUM_CLUSTERS="${NUM_KIND_CLUSTERS}" LEASE_ROOT="${NVME_ROOT}/kind-leases" bash scripts/bootstrap_kind_sregym_pool.sh
fi

echo "=== [v2.0 4/5] Launching 2x TP=1 Lossless BF16 SGLang Fleet (GPUs 0,1) + DDP RLVR (GPUs 2,3) ==="
for g in 0 1; do
  PORT=$((30000 + g))
  PATH="${NVME_ROOT}/venv-sglang/bin:${PATH}" CUDA_VISIBLE_DEVICES="${g}" \
    "${NVME_ROOT}/venv-sglang/bin/python3" -m sglang.launch_server \
    --model-path "${SFT_MERGED}" \
    --served-model-name omnisre-27b-v2 \
    --trust-remote-code --tp-size 1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 \
    --max-mamba-cache-size 115 --mem-fraction-static 0.85 --context-length 8192 \
    --disable-prefill-cuda-graph --attention-backend flashinfer --chunked-prefill-size 8192 \
    --host 127.0.0.1 --port "${PORT}" \
    > "${NVME_ROOT}/sglang_v2_gpu${g}.log" 2>&1 &
  SGLANG_PIDS+=("$!")
done
for p in 30000 30001; do
  for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:${p}/health" >/dev/null 2>&1 && break
    sleep 3
  done
done

CUDA_VISIBLE_DEVICES=2,3 "${NVME_ROOT}/venv-sre-rl/bin/torchrun" --nproc_per_node=2 --master_port=29502 \
  scripts/train_rl_v2_omnisre_4xh100.py \
  --base-model-path "${SFT_MERGED}" \
  --sglang-urls "http://127.0.0.1:30000,http://127.0.0.1:30001" \
  --sync-dir "/dev/shm/omnisre_v2_sync" \
  --output-dir "${NVME_ROOT}/checkpoints_v2/omnisre-27b-v2.0" \
  --tier1-train-json "${NVME_ROOT}/data_v2_expanded/rl_v2_train.jsonl" \
  --tier1-heldout-json "${NVME_ROOT}/data_v2_expanded/rl_v2_heldout.jsonl" \
  --steps "${RL_STEPS}" \
  --gcs-backup-prefix "${GCS_V2}/rlvr"

cleanup
SGLANG_PIDS=()

echo "=== [v2.0 5/5] Merging Best Checkpoint into Standalone BF16 ==="
BEST_ADAPTER="${NVME_ROOT}/checkpoints_v2/omnisre-27b-v2.0/best_checkpoint"
[[ -d "${BEST_ADAPTER}" ]] || BEST_ADAPTER="${NVME_ROOT}/checkpoints_v2/omnisre-27b-v2.0"
MERGED_V2="${NVME_ROOT}/omnisre-27b-v2.0-merged"
"${NVME_ROOT}/venv-sre-rl/bin/python3" scripts/merge_standalone_bf16.py \
  --base-model-dir "${SFT_MERGED}" \
  --adapter-dir "${BEST_ADAPTER}" \
  --out-dir "${MERGED_V2}"
gcloud storage rsync -r "${MERGED_V2}" "${GCS_V2}/merged-27b-bf16" --quiet
gcloud storage rsync -r "${NVME_ROOT}/checkpoints_v2/omnisre-27b-v2.0" "${GCS_V2}/final" --quiet
echo "=== OmniSRE v2.0 COMPLETE & BACKED UP TO ${GCS_V2} ==="
