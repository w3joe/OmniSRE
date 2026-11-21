#!/usr/bin/env bash
# Official Third-Party External Benchmark Runner on GCP `a3-highgpu-4g` (4x H100 80GB, 3TB NVMe)
# Evaluates `OmniSRE v1.0` (GPUs 0,1) vs. Base `Qwen/Qwen3.8-27B` (GPUs 2,3) side-by-side on:
#   1. Official `microsoft/OpenRCA` (Dockerized Python RCA-Agent sandbox across Bank, Telecom & Market)
#   2. Official Live `SREGym-Lite` (4-node Kind Kubernetes cluster + Calico + Stratus agent, 17 problems)
# Streams intermediate & final CSVs/JSONs to GCS and auto-deletes the VM when complete.
set -euo pipefail

NVME_ROOT="${NVME_ROOT:-/mnt/nvme}"
GCS_BUCKET="${GCS_BUCKET:-gs://onflow-30k-qwen38-sre-rl}"
GCS_OUT="${GCS_OUT:-${GCS_BUCKET}/omnisre-v1.0/external-benchmarks}"
OPENRCA_TASKS_PER_DS="${OPENRCA_TASKS_PER_DS:-15}"
SREGYM_TIMEOUT_S="${SREGYM_TIMEOUT_S:-480}"
VM_NAME="${VM_NAME:-}"
VM_ZONE="${VM_ZONE:-asia-northeast3-c}"
GCP_PROJECT="${GCP_PROJECT:-onflow-30k}"

export HF_HOME="${NVME_ROOT}/hf_cache"
export TOKENIZERS_PARALLELISM="false"
export NCCL_NET_PLUGIN="none"
unset LD_LIBRARY_PATH || true
export PATH="$HOME/.local/bin:/usr/local/bin:${PATH}"

PID_OMNI=""
PID_BASE=""
cleanup() {
  [[ -n "${PID_OMNI}" ]] && kill "${PID_OMNI}" 2>/dev/null || true
  [[ -n "${PID_BASE}" ]] && kill "${PID_BASE}" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "${NVME_ROOT}/ext_bench" "${NVME_ROOT}/ext_results" "${NVME_ROOT}/omnisre-v1.0" "${NVME_ROOT}/docker"

echo "=== [1/6] Installing Docker >=25 (/mnt/nvme/docker), Kind v0.33.0, Kubectl, & Helm v4.3.2 ==="
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo mkdir -p /etc/docker "${NVME_ROOT}/docker" /run/udev
echo "{\"data-root\": \"${NVME_ROOT}/docker\"}" | sudo tee /etc/docker/daemon.json >/dev/null
sudo systemctl enable --now docker
sudo systemctl restart docker
sudo chmod 666 /var/run/docker.sock
sudo sysctl -w fs.inotify.max_user_instances=1024 fs.inotify.max_user_watches=1048576

if ! command -v kind >/dev/null 2>&1; then
  curl -fsSLo /tmp/kind https://kind.sigs.k8s.io/dl/v0.33.0/kind-linux-amd64
  chmod +x /tmp/kind && sudo mv /tmp/kind /usr/local/bin/kind
fi
if ! command -v kubectl >/dev/null 2>&1; then
  KVER=$(curl -L -s https://dl.k8s.io/release/stable.txt)
  curl -fsSLo /tmp/kubectl "https://dl.k8s.io/release/${KVER}/bin/linux/amd64/kubectl"
  chmod +x /tmp/kubectl && sudo mv /tmp/kubectl /usr/local/bin/kubectl
fi
if ! command -v helm >/dev/null 2>&1; then
  curl -fsSL https://get.helm.sh/helm-v4.3.2-linux-amd64.tar.gz -o /tmp/helm.tgz
  tar -xzf /tmp/helm.tgz -C /tmp
  sudo mv /tmp/linux-amd64/helm /usr/local/bin/helm
fi
docker --version && kind --version && kubectl version --client && helm version

echo "=== [2/6] Pulling OmniSRE v1.0 (51 GiB BF16) + Caching Base Qwen/Qwen3.8-27B ==="
if [[ ! -f "${NVME_ROOT}/omnisre-v1.0/model.safetensors.index.json" ]]; then
  gcloud storage rsync -r "${GCS_BUCKET}/omnisre-v1.0/merged-27b-bf16" "${NVME_ROOT}/omnisre-v1.0" --quiet
fi
"${NVME_ROOT}/venv-sglang/bin/python3" -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3.8-27B")'

echo "=== [3/6] Launching Parallel 2+2 SGLang TP2 Servers (0.0.0.0:30000 OmniSRE | 0.0.0.0:30001 Base) ==="
PATH="${NVME_ROOT}/venv-sglang/bin:${PATH}" CUDA_VISIBLE_DEVICES=0,1 \
  "${NVME_ROOT}/venv-sglang/bin/python3" -m sglang.launch_server \
  --model-path "${NVME_ROOT}/omnisre-v1.0" \
  --served-model-name omnisre-27b-v1 \
  --trust-remote-code --tp-size 2 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.80 \
  --attention-backend flashinfer --chunked-prefill-size 16384 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --host 0.0.0.0 --port 30000 \
  > "${NVME_ROOT}/ext_results/sglang_omnisre.log" 2>&1 &
PID_OMNI=$!

PATH="${NVME_ROOT}/venv-sglang/bin:${PATH}" CUDA_VISIBLE_DEVICES=2,3 \
  "${NVME_ROOT}/venv-sglang/bin/python3" -m sglang.launch_server \
  --model-path "Qwen/Qwen3.8-27B" \
  --served-model-name qwen38-27b-base \
  --trust-remote-code --tp-size 2 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.80 \
  --attention-backend flashinfer --chunked-prefill-size 16384 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --host 0.0.0.0 --port 30001 \
  > "${NVME_ROOT}/ext_results/sglang_base.log" 2>&1 &
PID_BASE=$!

for PORT in 30000 30001; do
  READY=0
  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "[ext-bench] SGLang healthy on port ${PORT}"
      READY=1
      break
    fi
    sleep 5
  done
  [[ "${READY}" -eq 1 ]] || { echo "ERROR: SGLang port ${PORT} failed"; exit 1; }
done

DOCKER_GW_IP=$(docker network inspect bridge -f '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null || echo "172.17.0.1")
echo "[ext-bench] Docker bridge gateway IP: ${DOCKER_GW_IP}"

echo "=== [4/6] Official Benchmark 1/2: microsoft/OpenRCA (Dockerized RCA-Agent) ==="
OPENRCA_DIR="${NVME_ROOT}/ext_bench/OpenRCA"
if [[ ! -d "${OPENRCA_DIR}" ]]; then
  git clone --depth 1 https://github.com/microsoft/OpenRCA.git "${OPENRCA_DIR}"
fi
uv venv --clear "${NVME_ROOT}/venv-openrca" --python 3.11
uv pip install --python "${NVME_ROOT}/venv-openrca/bin/python" -r "${OPENRCA_DIR}/requirements.txt" "numpy<2" openai huggingface_hub gdown pandas loguru nbformat pyyaml

mkdir -p "${OPENRCA_DIR}/dataset"
if gcloud storage ls "${GCS_BUCKET}/openrca-raw/READY" >/dev/null 2>&1; then
  echo "[openrca] Restoring cached OpenRCA dataset from GCS..."
  gcloud storage rsync -r "${GCS_BUCKET}/openrca-raw" "${OPENRCA_DIR}/dataset" --quiet
else
  echo "[openrca] Downloading OpenRCA (Bank, Market, Telecom) via Hugging Face Hub (cdreetz/OpenRCA)..."
  "${NVME_ROOT}/venv-sglang/bin/python3" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("cdreetz/OpenRCA", repo_type="dataset", local_dir="${OPENRCA_DIR}/dataset")
PY
  if [[ -f "${OPENRCA_DIR}/dataset/Bank/query.csv" || -f "${OPENRCA_DIR}/dataset/Telecom/query.csv" || -f "${OPENRCA_DIR}/dataset/Market/cloudbed-1/query.csv" ]]; then
    echo "ok" | gcloud storage cp - "${GCS_BUCKET}/openrca-raw/READY" --quiet
    gcloud storage rsync -r "${OPENRCA_DIR}/dataset" "${GCS_BUCKET}/openrca-raw" --quiet || true
  fi
fi
sudo chmod -R a+rX "${OPENRCA_DIR}/dataset"

docker build -t openrca-executor:local "${OPENRCA_DIR}/rca/baseline/rca_agent/sandbox"

cat <<EOF >"${OPENRCA_DIR}/config_omnisre.json"
{"gemini3pro": {"model": "omnisre-27b-v1", "api_key": "sglang", "base_url": "http://127.0.0.1:30000/v1", "temperature": 0.0}}
EOF
cat <<EOF >"${OPENRCA_DIR}/config_base.json"
{"gemini3pro": {"model": "qwen38-27b-base", "api_key": "sglang", "base_url": "http://127.0.0.1:30001/v1", "temperature": 0.0}}
EOF

END_IDX=$((OPENRCA_TASKS_PER_DS - 1))
cd "${OPENRCA_DIR}"
AVAILABLE_OPENRCA_DS=()
for DS in "Telecom" "Bank" "Market/cloudbed-1"; do
  if [[ -f "${OPENRCA_DIR}/dataset/${DS}/query.csv" && -d "${OPENRCA_DIR}/dataset/${DS}/telemetry" ]]; then
    AVAILABLE_OPENRCA_DS+=("${DS}")
  fi
done
echo "[openrca] Available datasets with query.csv + telemetry: ${AVAILABLE_OPENRCA_DS[*]:-NONE}"

for DS in "${AVAILABLE_OPENRCA_DS[@]}"; do
  echo "[openrca] Running OmniSRE v1.0 & Base Qwen3.8-27B concurrently on ${DS} (0..${END_IDX})..."
  API_CONFIG_PATH="${OPENRCA_DIR}/config_omnisre.json" \
    "${NVME_ROOT}/venv-openrca/bin/python" -m rca.run_agent_standard \
    --dataset "${DS}" --start_idx 0 --end_idx "${END_IDX}" --tag omnisre_v1 \
    > "${NVME_ROOT}/ext_results/openrca_omnisre_${DS//\//_}.log" 2>&1 &
  P1=$!

  API_CONFIG_PATH="${OPENRCA_DIR}/config_base.json" \
    "${NVME_ROOT}/venv-openrca/bin/python" -m rca.run_agent_standard \
    --dataset "${DS}" --start_idx 0 --end_idx "${END_IDX}" --tag base_qwen38 \
    > "${NVME_ROOT}/ext_results/openrca_base_${DS//\//_}.log" 2>&1 &
  P2=$!

  wait "${P1}" || echo "[openrca] WARN: OmniSRE exited non-zero on ${DS}"
  wait "${P2}" || echo "[openrca] WARN: Base exited non-zero on ${DS}"
done

# Summarize OpenRCA results into clean JSON
"${NVME_ROOT}/venv-openrca/bin/python" - <<'PY'
import glob, json, os, pandas as pd
root = "/mnt/nvme/ext_bench/OpenRCA/test/result"
summary = {"omnisre_v1": {}, "base_qwen38": {}}
for csv_path in sorted(glob.glob(f"{root}/**/*.csv", recursive=True)):
    rel = os.path.relpath(csv_path, root)
    ds = os.path.dirname(rel)
    fname = os.path.basename(rel)
    key = "omnisre_v1" if "omnisre_v1" in fname else ("base_qwen38" if "base_qwen38" in fname else None)
    if not key:
        continue
    df = pd.read_csv(csv_path)
    scores = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    summary[key][ds] = {
        "tasks": int(len(df)),
        "strict_accuracy": round(float((scores == 1.0).mean()), 4) if len(df) else 0.0,
        "partial_score_mean": round(float(scores.mean()), 4) if len(df) else 0.0,
    }
out_p = "/mnt/nvme/ext_results/openrca_summary.json"
with open(out_p, "w") as fh:
    json.dump(summary, fh, indent=2)
print("OPENRCA_SUMMARY:", json.dumps(summary))
PY
cp -r "${OPENRCA_DIR}/test/result" "${NVME_ROOT}/ext_results/openrca_csvs" 2>/dev/null || true
gcloud storage rsync -r "${NVME_ROOT}/ext_results" "${GCS_OUT}" --quiet

echo "=== [5/6] Official Benchmark 2/2: Live SREGym-Lite on 4-Node Kind Kubernetes Cluster ==="
SREGYM_DIR="${NVME_ROOT}/ext_bench/SREGym"
if [[ ! -d "${SREGYM_DIR}" ]]; then
  git clone --depth 1 --recurse-submodules https://github.com/SREGym/SREGym.git "${SREGYM_DIR}"
fi
cd "${SREGYM_DIR}"
uv sync
bash kind/setup_kind_cluster.sh x86
kubectl get nodes -o wide

# Run SREGym-Lite for OmniSRE v1.0, then Base Qwen3.8-27B (both judged identically by port 30001)
export JUDGE_API_BASE="http://127.0.0.1:30001/v1"
export JUDGE_API_KEY="sglang"
export OPENAI_API_KEY="sglang"
export AGENT_API_KEY="sglang"

echo "[sregym] Running Live SREGym-Lite (17 K8s problems) for OmniSRE v1.0..."
AGENT_API_BASE="http://${DOCKER_GW_IP}:30000/v1" \
  uv run python main.py \
  --suite sregym-lite \
  --agent stratus \
  --model openai/omnisre-27b-v1 \
  --judge-model openai/qwen38-27b-base \
  --internet-access open \
  --agent-timeout "${SREGYM_TIMEOUT_S}" \
  > "${NVME_ROOT}/ext_results/sregym_omnisre.log" 2>&1 || echo "[sregym] WARN: OmniSRE SREGym exited non-zero"
cp -r "${SREGYM_DIR}/results" "${NVME_ROOT}/ext_results/sregym_runs_after_omnisre" 2>/dev/null || true
gcloud storage rsync -r "${NVME_ROOT}/ext_results" "${GCS_OUT}" --quiet

echo "[sregym] Running Live SREGym-Lite (17 K8s problems) for Base Qwen3.8-27B..."
AGENT_API_BASE="http://${DOCKER_GW_IP}:30001/v1" \
  uv run python main.py \
  --suite sregym-lite \
  --agent stratus \
  --model openai/qwen38-27b-base \
  --judge-model openai/qwen38-27b-base \
  --internet-access open \
  --agent-timeout "${SREGYM_TIMEOUT_S}" \
  > "${NVME_ROOT}/ext_results/sregym_base.log" 2>&1 || echo "[sregym] WARN: Base SREGym exited non-zero"
cp -r "${SREGYM_DIR}/results" "${NVME_ROOT}/ext_results/sregym_runs_all" 2>/dev/null || true

# Summarize SREGym CSVs + combine with OpenRCA into final_external_benchmarks.json
"${NVME_ROOT}/venv-openrca/bin/python" - <<'PY'
import glob, json, os, pandas as pd

openrca_p = "/mnt/nvme/ext_results/openrca_summary.json"
openrca = json.load(open(openrca_p)) if os.path.exists(openrca_p) else {}

sregym_csvs = sorted(glob.glob("/mnt/nvme/ext_bench/SREGym/results/*/*_stratus_results.csv"))
sregym_runs = []
for p in sregym_csvs:
    df = pd.read_csv(p)
    cols = list(df.columns)
    diag_col = next((c for c in cols if "diagnosis" in c and ("success" in c or c == "diagnosis")), None)
    mit_col = next((c for c in cols if "mitigation" in c and ("success" in c or c == "mitigation")), None)
    sregym_runs.append({
        "csv": os.path.relpath(p, "/mnt/nvme/ext_bench/SREGym/results"),
        "problems": int(len(df)),
        "columns": cols,
        "rows": df.to_dict(orient="records"),
        "diagnosis_pass_rate": round(float(df[diag_col].isin([True, "True", 1, "1", "pass", "passed"]).mean()), 4) if diag_col and len(df) else None,
        "mitigation_pass_rate": round(float(df[mit_col].isin([True, "True", 1, "1", "pass", "passed"]).mean()), 4) if mit_col and len(df) else None,
    })

final_report = {
    "openrca_official": openrca,
    "sregym_lite_live_k8s": sregym_runs,
}
with open("/mnt/nvme/ext_results/final_external_benchmarks.json", "w") as fh:
    json.dump(final_report, fh, indent=2, default=str)
print("FINAL_EXTERNAL_BENCHMARKS:", json.dumps(final_report, default=str)[:2000])
PY

echo "=== [6/6] Syncing Final External Benchmark Results to GCS & Auto-Deleting VM ==="
gcloud storage rsync -r "${NVME_ROOT}/ext_results" "${GCS_OUT}" --quiet
cleanup
if [[ -n "${VM_NAME}" ]]; then
  gcloud compute instances delete "${VM_NAME}" --zone="${VM_ZONE}" --project="${GCP_PROJECT}" --quiet
fi
