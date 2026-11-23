#!/usr/bin/env bash
# Stage 3 Live Kubernetes Sandbox Pool Bootstrap for GCP `a3-highgpu-4g` (104 vCPUs, 3 TB NVMe)
# Provisions N isolated `Kind` clusters backed by `/mnt/nvme/kind-leases`, applies strict
# anti-cheat RBAC (`infra/k8s/sre_agent_rbac.yaml`), and mints a restricted agent kubeconfig
# that cannot see or tamper with `sregym-injector` / `kube-system`.
set -euo pipefail

NUM_CLUSTERS="${NUM_CLUSTERS:-8}"
LEASE_ROOT="${LEASE_ROOT:-/mnt/nvme/kind-leases}"
BASE_INGRESS_PORT="${BASE_INGRESS_PORT:-18080}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${LEASE_ROOT}"
docker pull busybox:1.36 >/dev/null 2>&1 || true

for ((i=0; i<NUM_CLUSTERS; i++)); do
  NAME=$(printf "sre-rl-kind-%02d" "$i")
  PORT=$((BASE_INGRESS_PORT + i))
  DIR="${LEASE_ROOT}/${NAME}"
  mkdir -p "${DIR}"

  cat <<EOF >"${DIR}/kind-config.yaml"
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30080
        hostPort: ${PORT}
        protocol: TCP
EOF

  if ! kind get clusters 2>/dev/null | grep -q "^${NAME}$"; then
    echo "[kind] Creating cluster ${NAME} (ingress port ${PORT})..."
    kind create cluster \
      --name "${NAME}" \
      --config "${DIR}/kind-config.yaml" \
      --kubeconfig "${DIR}/admin-injector.kubeconfig" \
      --wait 90s
  fi
  kind load docker-image busybox:1.36 --name "${NAME}" >/dev/null 2>&1 || true

  # Apply least-privilege agent RBAC & sample target app deployment
  kubectl --kubeconfig "${DIR}/admin-injector.kubeconfig" apply -f "${REPO_ROOT}/infra/k8s/sre_agent_rbac.yaml"

  # Mint time-bound ServiceAccount token kubeconfig strictly limited to `sre-rl-agent` in `app-prod`
  TOKEN=$(kubectl --kubeconfig "${DIR}/admin-injector.kubeconfig" -n app-prod create token sre-rl-agent --duration=24h)
  SERVER=$(kubectl --kubeconfig "${DIR}/admin-injector.kubeconfig" config view --minify -o jsonpath='{.clusters[0].cluster.server}')
  CA_DATA=$(kubectl --kubeconfig "${DIR}/admin-injector.kubeconfig" config view --raw --minify -o jsonpath='{.clusters[0].cluster.certificate-authority-data}')

  cat <<EOF >"${DIR}/agent-restricted.kubeconfig"
apiVersion: v1
kind: Config
clusters:
  - name: ${NAME}
    cluster:
      certificate-authority-data: ${CA_DATA}
      server: ${SERVER}
contexts:
  - name: sre-agent@${NAME}
    context:
      cluster: ${NAME}
      namespace: app-prod
      user: sre-rl-agent
current-context: sre-agent@${NAME}
users:
  - name: sre-rl-agent
    user:
      token: ${TOKEN}
EOF

  echo "[kind] Cluster ${NAME} ready (agent RBAC isolated to app-prod)."
done

echo "[kind] All ${NUM_CLUSTERS} isolated Stage 3 Kind clusters provisioned under ${LEASE_ROOT}."
