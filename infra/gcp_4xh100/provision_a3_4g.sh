#!/usr/bin/env bash
# Provision a GCP `a3-highgpu-4g` instance (4x NVIDIA H100 80GB, 104 vCPUs, 936 GB RAM, 3 TB Local NVMe)
# Cycles across A3-capable zones if Spot/Flex-Start capacity is temporarily full in one zone.
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
INSTANCE_NAME="${INSTANCE_NAME:-qwen38-sre-rl-4xh100}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"   # SPOT | FLEX_START | STANDARD
BOOT_DISK_GB="${BOOT_DISK_GB:-250}"
CANDIDATE_ZONES=(
  "us-central1-a"
  "us-east4-c"
  "us-west1-a"
  "europe-west4-b"
  "asia-southeast1-c"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ExtraFlags=()
if [[ "${PROVISIONING_MODEL}" == "SPOT" ]]; then
  ExtraFlags+=(
    "--provisioning-model=SPOT"
    "--instance-termination-action=STOP"
    "--maintenance-policy=TERMINATE"
  )
else
  ExtraFlags+=(
    "--maintenance-policy=TERMINATE"
  )
fi

for ZONE in "${CANDIDATE_ZONES[@]}"; do
  echo "[gcp] Attempting a3-highgpu-4g (${PROVISIONING_MODEL}) in ${ZONE}..."
  if gcloud compute instances create "${INSTANCE_NAME}" \
      --project="${PROJECT_ID}" \
      --zone="${ZONE}" \
      --machine-type="a3-highgpu-4g" \
      --image-family="pytorch-2-5-cu124-ubuntu-2204-nvidia-560" \
      --image-project="deeplearning-platform-release" \
      --boot-disk-size="${BOOT_DISK_GB}GB" \
      --boot-disk-type="pd-ssd" \
      --scopes="https://www.googleapis.com/auth/cloud-platform" \
      --metadata-from-file="startup-script=${SCRIPT_DIR}/startup_nvme_docker_cuda.sh" \
      "${ExtraFlags[@]}"; then
    echo "[gcp] Provisioned ${INSTANCE_NAME} in ${ZONE}!"
    echo "[gcp] Connect via: gcloud compute ssh ${INSTANCE_NAME} --project=${PROJECT_ID} --zone=${ZONE}"
    exit 0
  fi
  echo "[gcp] Capacity unavailable in ${ZONE}, trying next zone..."
done

echo "[gcp] ERROR: Could not allocate a3-highgpu-4g in any candidate zone." >&2
exit 1
