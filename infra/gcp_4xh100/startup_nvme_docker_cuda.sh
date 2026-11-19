#!/usr/bin/env bash
# Startup script for GCP `a3-highgpu-4g`:
# 1. Stripes the 8x 375GB Local NVMe SSDs into a 3 TB RAID-0 filesystem at `/mnt/nvme`
# 2. Tunes Linux kernel inotify/conntrack limits for running 8-12 concurrent Kind K8s clusters
# 3. Moves Docker data-root onto `/mnt/nvme/docker` to eliminate etcd/image I/O bottlenecks
set -euo pipefail

NVME_MOUNT="/mnt/nvme"

if ! mountpoint -q "${NVME_MOUNT}"; then
  apt-get update -y && apt-get install -y mdadm xfsprogs jq rsync python3-dev build-essential
  NVME_DEVS=($(ls /dev/disk/by-id/google-local-nvme-ssd-* 2>/dev/null || true))
  if [[ ${#NVME_DEVS[@]} -gt 0 ]]; then
    if [[ ! -e /dev/md0 ]]; then
      mdadm --create /dev/md0 --level=0 --raid-devices="${#NVME_DEVS[@]}" "${NVME_DEVS[@]}" --force --run
      mkfs.xfs -f /dev/md0
    fi
    mkdir -p "${NVME_MOUNT}"
    mount -o discard,noatime /dev/md0 "${NVME_MOUNT}"
    chmod 1777 "${NVME_MOUNT}"
  else
    mkdir -p "${NVME_MOUNT}"
  fi
fi

mkdir -p \
  "${NVME_MOUNT}/hf_cache" \
  "${NVME_MOUNT}/data" \
  "${NVME_MOUNT}/checkpoints" \
  "${NVME_MOUNT}/wandb" \
  "${NVME_MOUNT}/docker" \
  "${NVME_MOUNT}/kind-leases"
chmod 1777 \
  "${NVME_MOUNT}/hf_cache" \
  "${NVME_MOUNT}/data" \
  "${NVME_MOUNT}/checkpoints" \
  "${NVME_MOUNT}/wandb" \
  "${NVME_MOUNT}/kind-leases"

# Mandatory sysctl tuning for multi-cluster Kind + high-concurrency SGLang HTTP workers
cat <<'EOF' >/etc/sysctl.d/99-sre-rl-kind.conf
fs.inotify.max_user_instances = 8192
fs.inotify.max_user_watches = 1048576
fs.file-max = 2097152
vm.swappiness = 1
net.core.somaxconn = 65535
EOF
sysctl --system

# Point Docker storage to the 3 TB RAID-0 NVMe array
if command -v docker >/dev/null 2>&1; then
  mkdir -p /etc/docker
  cat <<EOF >/etc/docker/daemon.json
{
  "data-root": "${NVME_MOUNT}/docker",
  "storage-driver": "overlay2",
  "default-ulimits": {
    "nofile": { "Name": "nofile", "Hard": 1048576, "Soft": 1048576 }
  }
}
EOF
  systemctl restart docker || true
fi

echo "[startup] /mnt/nvme ready ($(df -h "${NVME_MOUNT}" | tail -n 1))"
