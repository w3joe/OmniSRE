#!/usr/bin/env bash
# Provision a GCP `a3-highgpu-4g` instance (4x NVIDIA H100 80GB, 104 vCPUs, 936 GB RAM, 3 TB Local NVMe)
# Cycles across A3-capable zones if Spot/Flex-Start capacity is temporarily full in one zone.
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-}"
INSTANCE_NAME="${INSTANCE_NAME:-qwen38-sre-rl-4xh100}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"   # SPOT | FLEX_START | STANDARD
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2404-nvidia-580}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-balanced}"
BOOT_DISK_GB="${BOOT_DISK_GB:-250}"
CANDIDATE_ZONES=(
  "europe-west4-b"
  "europe-west4-c"
  "us-central1-a"
  "us-central1-b"
  "us-central1-c"
  "us-east4-c"
  "us-east4-a"
  "us-east5-a"
  "us-west1-a"
  "us-west4-a"
  "europe-west1-b"
  "europe-west2-b"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ExtraFlags=()
if [[ -n "${GCLOUD_ACCOUNT}" ]]; then
  ExtraFlags+=("--account=${GCLOUD_ACCOUNT}")
fi
if [[ "${PROVISIONING_MODEL}" == "SPOT" ]]; then
  ExtraFlags+=(
    "--provisioning-model=SPOT"
    "--instance-termination-action=STOP"
    "--maintenance-policy=TERMINATE"
  )
elif [[ "${PROVISIONING_MODEL}" == "FLEX_START" ]]; then
  ExtraFlags+=(
    "--provisioning-model=FLEX_START"
    "--instance-termination-action=STOP"
    "--maintenance-policy=TERMINATE"
    "--reservation-affinity=none"
    "--max-run-duration=${MAX_RUN_DURATION:-12h}"
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
      --image-family="${IMAGE_FAMILY}" \
      --image-project="deeplearning-platform-release" \
      --boot-disk-size="${BOOT_DISK_GB}GB" \
      --boot-disk-type="${BOOT_DISK_TYPE}" \
      --scopes="https://www.googleapis.com/auth/cloud-platform" \
      --metadata-from-file="startup-script=${SCRIPT_DIR}/startup_nvme_docker_cuda.sh" \
      "${ExtraFlags[@]}"; then
    echo "[gcp] Provisioned ${INSTANCE_NAME} in ${ZONE}!"
    echo "[gcp] Connect via: gcloud compute ssh ${INSTANCE_NAME} --project=${PROJECT_ID} --zone=${ZONE}"
    exit 0
  fi
  echo "[gcp] Capacity/quota unavailable in ${ZONE}, trying next zone..."
done

echo "[gcp] ERROR: Could not allocate a3-highgpu-4g in any candidate zone." >&2
exit 1
