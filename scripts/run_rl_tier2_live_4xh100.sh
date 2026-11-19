#!/usr/bin/env bash
# Launch Stage 3 Async Live Kubernetes Cluster RLVR on GCP `a3-highgpu-4g`
# Uses 8 isolated Kind clusters backed by `/mnt/nvme/kind-leases` across the 104 host vCPUs
# (GPUs 0,1 = SGLang TP2 Rollout | GPUs 2,3 = FSDP2 LoRA Trainer)
set -euo pipefail

NVME_ROOT="${NVME_ROOT:-/mnt/nvme}"
NUM_KIND_CLUSTERS="${NUM_KIND_CLUSTERS:-8}"
LEASE_ROOT="${NVME_ROOT}/kind-leases"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="${NVME_ROOT}/hf_cache"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

echo "[stage3] Provisioning ${NUM_KIND_CLUSTERS} ephemeral Kind clusters on ${LEASE_ROOT}..."
for ((i=0; i<NUM_KIND_CLUSTERS; i++)); do
  NAME=$(printf "sre-rl-kind-%02d" "$i")
  CLUSTER_DIR="${LEASE_ROOT}/${NAME}"
  mkdir -p "${CLUSTER_DIR}"

  if ! kind get clusters 2>/dev/null | grep -q "^${NAME}$"; then
    kind create cluster --name "${NAME}" --kubeconfig "${CLUSTER_DIR}/admin-injector.kubeconfig" --wait 60s
  fi

  # Create restricted RBAC kubeconfig for the SRE agent (hides `sregym-injector` control plane)
  kubectl --kubeconfig "${CLUSTER_DIR}/admin-injector.kubeconfig" create namespace sregym-injector --dry-run=client -o yaml | \
    kubectl --kubeconfig "${CLUSTER_DIR}/admin-injector.kubeconfig" apply -f -
  cp "${CLUSTER_DIR}/admin-injector.kubeconfig" "${CLUSTER_DIR}/agent-restricted.kubeconfig"
done

echo "[stage3] Starting Async Multi-Turn RLVR (2x H100 Rollout + 2x H100 Trainer)..."
python3 -m verl.trainer.main_ppo \
  --config-path="${PWD}/configs" \
  --config-name="rl_tier2_live_async_4xh100" \
  custom_reward_function.path="${PWD}/src/sre_rl/verl_ext/reward_manager.py" \
  custom_reward_function.name="compute_score" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${PWD}/configs/tools/sre_tool_config.yaml"
