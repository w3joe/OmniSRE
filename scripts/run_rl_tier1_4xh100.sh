#!/usr/bin/env bash
# Launch Stage 2 Tier-1 Snapshot Multi-Turn RLVR (DAPO + GTPO) on GCP `a3-highgpu-4g` (4x H100)
set -euo pipefail

NVME_ROOT="${NVME_ROOT:-/mnt/nvme}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="${NVME_ROOT}/hf_cache"
export WANDB_DIR="${NVME_ROOT}/wandb"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-${NVME_ROOT}/checkpoints/qwen38-27b-sre-sft-merged}"
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[stage2] SFT merged checkpoint not found at ${MODEL_PATH}; falling back to Qwen/Qwen3.8-27B"
  MODEL_PATH="Qwen/Qwen3.8-27B"
fi

python3 -m verl.trainer.main_ppo \
  --config-path="${PWD}/configs" \
  --config-name="rl_tier1_dapo_gtpo_4xh100" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  data.train_files="${NVME_ROOT}/data/rl_tier1_train.parquet" \
  data.val_files="${NVME_ROOT}/data/rl_tier1_val.parquet" \
  custom_reward_function.path="${PWD}/src/sre_rl/verl_ext/reward_manager.py" \
  custom_reward_function.name="compute_score" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${PWD}/configs/tools/sre_tool_config.yaml"
