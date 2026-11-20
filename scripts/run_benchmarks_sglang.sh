#!/usr/bin/env bash
# Side-by-Side Public Benchmark Runner: `Qwen/Qwen3.8-27B` (Base) vs. `OmniSRE-27B`
# Launches SGLang FP8-KV on 1-2x H100, runs `--mode subset` or `--mode full`, and outputs
# a unified Markdown + JSON comparison table.
#
# Usage:
#   bash scripts/run_benchmarks_sglang.sh --mode subset --benchmarks "internal_heldout rcaeval openrca"
#   bash scripts/run_benchmarks_sglang.sh --mode full   --benchmarks "internal_heldout rcaeval openrca sregym"
set -euo pipefail

MODE="subset"
BENCHMARKS="internal_heldout rcaeval"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.8-27B}"
OMNISRE_MODEL="${OMNISRE_MODEL:-/mnt/nvme/checkpoints_v4/omnisre-27b-merged}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/mnt/nvme/venv-sglang/bin/python3}"
EVAL_PYTHON="${EVAL_PYTHON:-/mnt/nvme/venv-sre-rl/bin/python3}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-2}"
OUT_DIR="${OUT_DIR:-/mnt/nvme/benchmark_reports}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --benchmarks) BENCHMARKS="$2"; shift 2 ;;
    --omnisre-model) OMNISRE_MODEL="$2"; shift 2 ;;
    --tp-size) TP_SIZE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${OUT_DIR}"
SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

run_for_model() {
  local MODEL_PATH="$1"
  local SERVED_NAME="$2"
  local REPORT_JSON="${OUT_DIR}/${SERVED_NAME}_${MODE}.json"

  cleanup
  echo "=== Serving ${SERVED_NAME} (${MODEL_PATH}) via SGLang TP${TP_SIZE} ==="
  PATH="$(dirname "${SGLANG_PYTHON}"):${PATH}" "${SGLANG_PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --trust-remote-code \
    --tp-size "${TP_SIZE}" \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --attention-backend flashinfer \
    --reasoning-parser qwen3 \
    --host 127.0.0.1 \
    --port "${PORT}" \
    > "${OUT_DIR}/sglang_${SERVED_NAME}.log" 2>&1 &
  SERVER_PID=$!

  local READY=0
  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      READY=1
      break
    fi
    sleep 5
  done
  if [[ "${READY}" -ne 1 ]]; then
    echo "[eval] ERROR: SGLang failed to start for ${SERVED_NAME}" >&2
    tail -n 40 "${OUT_DIR}/sglang_${SERVED_NAME}.log" >&2 || true
    exit 1
  fi

  # shellcheck disable=SC2086
  "${EVAL_PYTHON}" scripts/eval_benchmarks.py \
    --mode "${MODE}" \
    --benchmarks ${BENCHMARKS} \
    --model "${SERVED_NAME}" \
    --endpoint-url "http://127.0.0.1:${PORT}/v1" \
    --output-json "${REPORT_JSON}"

  cleanup
  SERVER_PID=""
}

run_for_model "${BASE_MODEL}" "qwen3.8-27b-base"
run_for_model "${OMNISRE_MODEL}" "omnisre-27b"

echo "=== Side-by-side reports saved under ${OUT_DIR}/ ==="
ls -lh "${OUT_DIR}"/*.json
