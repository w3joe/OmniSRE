#!/usr/bin/env bash
# Launch Stage 1 Cold-Start SFT on GCP `a3-highgpu-4g` (4x NVIDIA H100 80GB)
set -euo pipefail

NVME_ROOT="${NVME_ROOT:-/mnt/nvme}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export UNSLOTH_COMPILE_DISABLE="1"
export HF_HOME="${NVME_ROOT}/hf_cache"
export WANDB_DIR="${NVME_ROOT}/wandb"

# Ensure datasets exist on NVMe before starting
if [[ ! -f "${NVME_ROOT}/data/sft_coldstart_train.jsonl" ]]; then
  echo "[stage1] Generating Dataset A & Dataset B onto ${NVME_ROOT}/data..."
  python3 scripts/build_datasets.py \
    --output-dir "${NVME_ROOT}/data" \
    --num-scenarios 3600 \
    --sft-trajectories 2400 \
    --seed 42
fi

torchrun --nproc_per_node=4 --master_port=29501 \
  scripts/train_sft_qwen38_lora.py \
  --config configs/sft_qwen38_27b_lora_4xh100.yaml \
  --merge-output-dir "${NVME_ROOT}/checkpoints/qwen38-27b-sre-sft-merged"
