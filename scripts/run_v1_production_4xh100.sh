#!/usr/bin/env bash
# Full Flagship OmniSRE v1.0 Multi-World RLVR Run on GCP `a3-highgpu-4g` (4x H100 80GB, 3TB NVMe)
# Upgrades included:
#  1. Fixed DejaVu-B/C service & Oracle wait-event fault taxonomy + in-memory `np.searchsorted` CSV cache
#  2. Multi-World RLVR mix: 50% Real Telemetry (DejaVu + RCAEval + OpenRCA) + 25% Write Remediation (Live Kind K8s + Tier-1) + 25% Offline Synthesis
#  3. 24-turn horizon + 60% MCP/observability tool-schema randomization + dual budget footers
#  4. 250 steps, `--sync-every 2`, GCS Spot backups (`step-025..250`, `best_checkpoint`), & true `5x` Hierarchical Tree Ensemble (`2+3 -> 1`)
set -euo pipefail

NVME_ROOT="${NVME_ROOT:-/mnt/nvme}"
GCS_BUCKET="${GCS_BUCKET:-gs://onflow-30k-qwen38-sre-rl}"
GCS_V1="${GCS_V1:-${GCS_BUCKET}/omnisre-v1.0}"
BASE_MODEL="${BASE_MODEL:-${NVME_ROOT}/checkpoints_v6/omnisre-v5-merged}"
STEPS="${STEPS:-250}"
MAX_TURNS="${MAX_TURNS:-24}"
ENABLE_OPENRCA="${ENABLE_OPENRCA:-0}"
ENABLE_LIVE_KIND="${ENABLE_LIVE_KIND:-1}"
NUM_KIND_CLUSTERS="${NUM_KIND_CLUSTERS:-4}"
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

echo "=== [v1.0 1/4] Preparing Real Telemetry + Tier-1 Write-Remediation Corpora ==="
mkdir -p "${NVME_ROOT}/real_telemetry/raw" "${NVME_ROOT}/real_telemetry/extracted" "${NVME_ROOT}/real_telemetry/index" "${NVME_ROOT}/data_phase1b"
if [[ ! -d "${NVME_ROOT}/real_telemetry/extracted/DejaVu-C" ]]; then
  gcloud storage rsync -r "${GCS_BUCKET}/real-telemetry-raw" "${NVME_ROOT}/real_telemetry/raw" --quiet
fi
/mnt/nvme/venv-sre-rl/bin/python3 scripts/index_real_telemetry.py \
  --raw-dir "${NVME_ROOT}/real_telemetry/raw" \
  --extracted-dir "${NVME_ROOT}/real_telemetry/extracted" \
  --out-dir "${NVME_ROOT}/real_telemetry/index"

if [[ ! -f "${NVME_ROOT}/data_phase1b/rlvr_scenarios_train.json" ]]; then
  gcloud storage rsync -r "${GCS_BUCKET}/dataset-v4-phase1b" "${NVME_ROOT}/data_phase1b" --quiet
fi

if [[ "${ENABLE_OPENRCA}" == "1" ]]; then
  echo "=== [v1.0 1b/4] Downloading & Indexing microsoft/OpenRCA (Bank/Market/Telecom) ==="
  /mnt/nvme/venv-sre-rl/bin/pip install -q gdown
  /mnt/nvme/venv-sre-rl/bin/python3 scripts/download_openrca.py \
    --work-dir "${NVME_ROOT}/real_telemetry/openrca" \
    --out-dir "${NVME_ROOT}/real_telemetry/index"
fi

if [[ "${ENABLE_LIVE_KIND}" == "1" ]] && command -v kind >/dev/null 2>&1 && command -v kubectl >/dev/null 2>&1; then
  echo "=== [v1.0 2/4] Bootstrapping ${NUM_KIND_CLUSTERS} Isolated Live Kind K8s Clusters ==="
  NUM_CLUSTERS="${NUM_KIND_CLUSTERS}" LEASE_ROOT="${NVME_ROOT}/kind-leases" bash scripts/bootstrap_kind_sregym_pool.sh
else
  echo "=== [v1.0 2/4] Skipping Live Kind pool (ENABLE_LIVE_KIND=${ENABLE_LIVE_KIND} or kind/kubectl absent) ==="
fi

echo "=== [v1.0 3/4] Launching SGLang 0.5.23 TP2 on GPUs 0,1 ==="
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

echo "=== [v1.0 4/4] Running Full OmniSRE v1.0 Multi-World RLVR (${STEPS} steps, max_turns=${MAX_TURNS}, 60% MCP rand, sync-every=2) ==="
CUDA_VISIBLE_DEVICES=2,3 /mnt/nvme/venv-sre-rl/bin/torchrun --nproc_per_node=2 --master_port=29502 \
  scripts/train_rl_v1_omnisre_4xh100.py \
  --base-model-path "${BASE_MODEL}" \
  --sglang-url "http://127.0.0.1:${SGLANG_PORT}" \
  --sync-dir "${NVME_ROOT}/sglang_live_sync_v1" \
  --output-dir "${NVME_ROOT}/checkpoints_v1/omnisre-27b-v1.0" \
  --train-index-jsonl "${NVME_ROOT}/real_telemetry/index/real_telemetry_train.jsonl" \
  --heldout-index-jsonl "${NVME_ROOT}/real_telemetry/index/real_telemetry_heldout.jsonl" \
  --openrca-train-jsonl "${NVME_ROOT}/real_telemetry/index/openrca_train.jsonl" \
  --openrca-heldout-jsonl "${NVME_ROOT}/real_telemetry/index/openrca_heldout.jsonl" \
  --tier1-train-json "${NVME_ROOT}/data_phase1b/rlvr_scenarios_train.json" \
  --tier1-heldout-json "${NVME_ROOT}/data_phase1b/rlvr_scenarios_heldout.json" \
  --kind-lease-dir "${NVME_ROOT}/kind-leases" \
  --steps "${STEPS}" \
  --prompts-per-step 8 \
  --group-size 8 \
  --rollout-workers 24 \
  --eval-heldout-count 40 \
  --max-turns "${MAX_TURNS}" \
  --harness-rand-prob 0.60 \
  --lr 3e-6 \
  --sync-every 2 \
  --eval-every 25 \
  --save-every 25 \
  --gcs-backup-prefix "${GCS_V1}"

gcloud storage rsync -r "${NVME_ROOT}/checkpoints_v1/omnisre-27b-v1.0" "${GCS_V1}/final" --quiet
echo "=== OmniSRE v1.0 COMPLETE & BACKED UP TO ${GCS_V1} ==="
