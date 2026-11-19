#!/usr/bin/env bash
# Bootstrap Python + CUDA 12 + Gated DeltaNet kernels + SGLang + veRL + Kind on GCP a3-highgpu-4g
set -euo pipefail

NVME_ROOT="${NVME_ROOT:-/mnt/nvme}"
export HF_HOME="${NVME_ROOT}/hf_cache"

echo "[1/5] Installing uv, kind, and kubectl..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v kind >/dev/null 2>&1; then
  curl -Lo /usr/local/bin/kind https://kind.sigs.k8s.io/dl/v0.26.0/kind-linux-amd64
  chmod +x /usr/local/bin/kind
fi

if ! command -v kubectl >/dev/null 2>&1; then
  K8S_VER="$(curl -L -s https://dl.k8s.io/release/stable.txt)"
  curl -Lo /usr/local/bin/kubectl "https://dl.k8s.io/release/${K8S_VER}/bin/linux/amd64/kubectl"
  chmod +x /usr/local/bin/kubectl
fi

echo "[2/5] Creating Python 3.12 training environment on NVMe..."
uv venv "${NVME_ROOT}/venv-sre-rl" --python 3.12
# shellcheck disable=SC1091
source "${NVME_ROOT}/venv-sre-rl/bin/activate"

echo "[3/5] Installing PyTorch, Transformers >=5.8 (Qwen3.8 qwen3_5 arch), FlashAttention-2 & Gated DeltaNet Triton kernels..."
uv pip install --upgrade pip wheel setuptools packaging ninja
uv pip install "torch>=2.5.1" "transformers>=5.8.1" "accelerate>=1.3.0" "peft>=0.14.0" "trl>=0.15.0" "datasets>=3.2.0" "pyarrow>=18.0.0" "wandb>=0.19.0"
# Fused linear-attention kernels used by Qwen3.8-27B's 48 Gated DeltaNet blocks:
uv pip install flash-attn --no-build-isolation
uv pip install flash-linear-attention causal-conv1d --no-build-isolation

echo "[4/5] Installing SGLang + veRL (HybridFlow) + qwen38-sre-rl..."
uv pip install "sglang[all]"
if [[ ! -d "${NVME_ROOT}/verl" ]]; then
  git clone https://github.com/verl-project/verl.git "${NVME_ROOT}/verl"
fi
uv pip install -e "${NVME_ROOT}/verl"
uv pip install -e ".[train,dev]"

echo "[5/5] Pre-warming HuggingFace snapshot for Qwen/Qwen3.8-27B onto /mnt/nvme/hf_cache..."
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get("QWEN38_MODEL_ID", "Qwen/Qwen3.8-27B")
print(f"Downloading {model_id} to {os.environ['HF_HOME']}...")
snapshot_download(repo_id=model_id, resume_download=True)
print("Download complete.")
PY

echo "[setup] Environment ready. Activate via: source ${NVME_ROOT}/venv-sre-rl/bin/activate"
