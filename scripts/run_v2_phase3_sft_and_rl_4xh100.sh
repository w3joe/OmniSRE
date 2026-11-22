#!/usr/bin/env bash
# End-to-End OmniSRE v2.0 Phase-3 SFT Warm-Up + Short 40-Step RLVR Polish + Full Benchmark (`4x H100`)
# Total expected runtime on `a3-highgpu-4g`: ~2.5 - 3.0 hours (~$70 - $85 total Spot)
set -euo pipefail

NVME_ROOT="${NVME_ROOT:-/mnt/nvme}"
GCS_BUCKET="${GCS_BUCKET:-gs://onflow-30k-qwen38-sre-rl}"
GCS_OUT="${GCS_OUT:-${GCS_BUCKET}/omnisre-v2.0-phase3}"
RL_STEPS="${RL_STEPS:-40}"
VM_NAME="${VM_NAME:-}"
VM_ZONE="${VM_ZONE:-asia-northeast3-c}"
GCP_PROJECT="${GCP_PROJECT:-onflow-30k}"

export CLOUDSDK_STORAGE_CHECK_HASHES="never"
export HF_HOME="${NVME_ROOT}/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export NCCL_NET_PLUGIN="none"
unset LD_LIBRARY_PATH || true

echo "=== [1/4] Building Verified Zero-Contamination Phase-3 SFT Dataset ==="
"${NVME_ROOT}/venv-sre-rl/bin/python3" scripts/build_phase3_v2_sft_dataset.py \
  --train-index-jsonl "${NVME_ROOT}/real_telemetry/index/real_telemetry_train.jsonl" \
  --openrca-train-jsonl "${NVME_ROOT}/real_telemetry/index/openrca_train.jsonl" \
  --phase1b-jsonl "data/phase1b_sft_v4/sft_phase1b_exact_schema.jsonl" \
  --out-jsonl "${NVME_ROOT}/data_phase3/sft_phase3_v2_mixed.jsonl"
gcloud storage rsync -r "${NVME_ROOT}/data_phase3" "${GCS_OUT}/dataset-phase3" --quiet

echo "=== [2/4] Running 1-Epoch Phase-3 Rank-64 LoRA SFT Warm-Up (~20 min on 4x H100) ==="
SFT_MERGED="${NVME_ROOT}/omnisre-v2-phase3-sft-merged"
CUDA_VISIBLE_DEVICES=0,1,2,3 "${NVME_ROOT}/venv-sre-rl/bin/torchrun" --nproc_per_node=4 --master_port=29501 \
  scripts/train_sft_qwen38_lora.py \
  --config configs/sft_v2_phase3_4xh100.yaml \
  --model-path "${NVME_ROOT}/omnisre-v1.0" \
  --train-jsonl "${NVME_ROOT}/data_phase3/sft_phase3_v2_mixed.jsonl" \
  --output-dir "${NVME_ROOT}/checkpoints_phase3/sft-lora64" \
  --merge-output-dir "${SFT_MERGED}"
gcloud storage rsync -r "${SFT_MERGED}" "${GCS_OUT}/sft-merged-27b-bf16" --quiet

if [[ "${RL_STEPS}" -gt 0 ]]; then
  echo "=== [3/4] Running ${RL_STEPS}-Step On-Policy RLVR Polish Starting from Phase-3 SFT ==="
  STEPS="${RL_STEPS}" START_STEP=1 MAX_TURNS=35 OVERSAMPLE=2 MAX_SEGMENTS_PER_STEP=96 SYNC_EVERY=3 \
    EVAL_EVERY=20 SAVE_EVERY=20 MERGE_STANDALONE=1 \
    BASE_MODEL="${SFT_MERGED}" \
    OUTPUT_DIR="${NVME_ROOT}/checkpoints_phase3/rlvr" \
    GCS_V2="${GCS_OUT}/rlvr" \
    bash scripts/run_v2_turbo_8xh100.sh
  FINAL_MODEL="${NVME_ROOT}/omnisre-27b-v2.0-merged"
else
  FINAL_MODEL="${SFT_MERGED}"
fi

echo "=== [4/4] Full Symmetric Benchmark: OmniSRE v2.0 (Phase-3) vs OmniSRE v1.0 ==="
BENCH_PIDS=()
for g in 0 1; do
  PORT=$((30100 + g))
  PATH="${NVME_ROOT}/venv-sglang/bin:${PATH}" CUDA_VISIBLE_DEVICES="${g}" \
    "${NVME_ROOT}/venv-sglang/bin/python3" -m sglang.launch_server \
    --model-path "${FINAL_MODEL}" --served-model-name omnisre-27b-v2 \
    --trust-remote-code --tp-size 1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 \
    --max-mamba-cache-size 115 --mem-fraction-static 0.85 --context-length 8192 \
    --disable-prefill-cuda-graph --attention-backend flashinfer --chunked-prefill-size 8192 \
    --host 127.0.0.1 --port "${PORT}" > "${NVME_ROOT}/bench_p3_gpu${g}.log" 2>&1 &
  BENCH_PIDS+=("$!")
done
for g in 2 3; do
  PORT=$((30100 + g))
  PATH="${NVME_ROOT}/venv-sglang/bin:${PATH}" CUDA_VISIBLE_DEVICES="${g}" \
    "${NVME_ROOT}/venv-sglang/bin/python3" -m sglang.launch_server \
    --model-path "${NVME_ROOT}/omnisre-v1.0" --served-model-name omnisre-27b-v1 \
    --trust-remote-code --tp-size 1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 \
    --max-mamba-cache-size 115 --mem-fraction-static 0.85 --context-length 8192 \
    --disable-prefill-cuda-graph --attention-backend flashinfer --chunked-prefill-size 8192 \
    --host 127.0.0.1 --port "${PORT}" > "${NVME_ROOT}/bench_v1_gpu${g}.log" 2>&1 &
  BENCH_PIDS+=("$!")
done
for p in 30100 30101 30102 30103; do
  for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:${p}/health" >/dev/null 2>&1 && break
    sleep 3
  done
done

"${NVME_ROOT}/venv-sre-rl/bin/python3" scripts/eval_v2_full_suite.py \
  --v2-urls "http://127.0.0.1:30100,http://127.0.0.1:30101" \
  --baseline-urls "http://127.0.0.1:30102,http://127.0.0.1:30103" \
  --tokenizer-path "${FINAL_MODEL}" \
  --max-turns 35 \
  --workers 48 \
  --out-json "${NVME_ROOT}/v2_phase3_vs_v1_results.json"

gcloud storage cp "${NVME_ROOT}/v2_phase3_vs_v1_results.json" "${GCS_OUT}/benchmarks/v2_phase3_vs_v1_results.json" --quiet
for pid in "${BENCH_PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done

if [[ -n "${VM_NAME}" ]]; then
  gcloud compute instances delete "${VM_NAME}" --zone="${VM_ZONE}" --project="${GCP_PROJECT}" --quiet || true
fi
