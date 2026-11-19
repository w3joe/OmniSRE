# `qwen38-sre-rl` — RLVR Post-Training for `Qwen3.8-27B` Dense Autonomous SRE Agents

End-to-end **Reinforcement Learning with Verifiable Rewards (RLVR)** post-training pipeline using **Multi-Turn DAPO + GTPO** to specialize **`Qwen/Qwen3.8-27B` (Dense)** for production Site Reliability Engineering (Triage $\rightarrow$ Localization $\rightarrow$ Root Cause Analysis $\rightarrow$ Safe Remediation), engineered out-of-the-box for **GCP `a3-highgpu-4g` (`4× NVIDIA H100 80GB`)**.

---

## Why `Qwen/Qwen3.8-27B` (Dense) + Multi-Turn DAPO/GTPO?

1. **Hybrid `3:1` Architecture (`48 × Gated DeltaNet` + `16 × Gated Attention`):**
   * Only 16 of 64 layers require standard KV-cache (4 KV heads, `head_dim=256` $\rightarrow$ **~2.0 GB per 64K sequence in FP8** vs. ~20 GB on traditional 30B dense transformers), letting `4× H100` run high-concurrency 32K–64K multi-turn telemetry rollouts without OOMing.
   * Standard LoRA configs miss 75% of Qwen3.8's sequence-mixing layers. `sre_rl.model.qwen38_lora` explicitly targets both Gated Attention (`q/k/v/o_proj`), SwiGLU (`gate/up/down_proj`), **and** Gated DeltaNet projections (`in_proj_qkvz`, `in_proj_ba`, `in_proj`, `out_proj`, `ba_proj`) while freezing the Vision Tower and MTP heads.
2. **Calibrated Thinking (`preserve_thinking` + Overthinking Regularization):**
   * Retains active diagnostic hypotheses across recent turns via bounded `preserve_thinking` while applying a soft per-turn `<think>` token budget (`low=180`, `medium=380`, `high=800`) to cure Qwen3.8-27B's verbose overthinking during time-critical SEV-1 pages.
3. **Anti-Reward-Hacking SRE Environment Design:**
   * **Read-Before-Write Separation:** Hard guardrail rejects mutations prior to $\ge 2$ diagnostic reads.
   * **Persistent Faults & Metastable Retry Storms:** Blind `restart_workload` on persistent ConfigMap/OOM/Canary defects re-crashes during burn-in ($0$ mitigation credit); metastable retry cascades require *both* root-config patching and replica scale-out.
   * **Benign False Alarms & Adversarial Log Injection Traps:** Prevents "action bias" (restarting healthy workloads) and terminates episodes (`-1.0`) if the agent executes prompt injections embedded inside untrusted HTTP/log lines or snoops on `sregym-injector` / `chaos-mesh` namespaces.

---

## Repository Structure

```text
qwen38-sre-rl/
├── configs/
│   ├── sft_qwen38_27b_lora_4xh100.yaml       # Stage 1 Cold-Start SFT (Rank-64 Hybrid LoRA)
│   ├── rl_tier1_dapo_gtpo_4xh100.yaml        # Stage 2 Tier-1 Snapshot RLVR (veRL + SGLang TP2)
│   ├── rl_tier2_live_async_4xh100.yaml       # Stage 3 Async Live K8s RLVR (2 Rollout + 2 Trainer GPUs)
│   └── tools/sre_tool_config.yaml            # Multi-turn read/write/terminal tool schemas
├── infra/gcp_4xh100/
│   ├── provision_a3_4g.sh                    # Multi-zone GCP Spot/Flex-Start a3-highgpu-4g launcher
│   ├── main.tf                               # Terraform for a3-highgpu-4g + GCS checkpoint bucket
│   ├── startup_nvme_docker_cuda.sh           # 3 TB RAID-0 Local NVMe (/mnt/nvme) + Docker/Kind tuning
│   └── setup_verl_qwen38_env.sh              # PyTorch + flash-linear-attention + SGLang + veRL setup
├── scripts/
│   ├── build_datasets.py                     # Generates Dataset A (SFT) + Dataset B (RLVR Parquet/JSONL)
│   ├── train_sft_qwen38_lora.py              # Stage 1 distributed LoRA SFT + BF16 merge exporter
│   ├── run_sft_4xh100.sh                     # Stage 1 launcher
│   ├── run_rl_tier1_4xh100.sh                # Stage 2 Tier-1 Snapshot RLVR launcher
│   ├── run_rl_tier2_live_4xh100.sh           # Stage 3 Live Kind Cluster Pool + Async RLVR launcher
│   ├── run_eval_pass_hat_k.py                # Evaluates Pass@1, Pass^5, TNR Safety & False-Positive Rate
│   └── serve_flash_sglang.sh                 # Single-GPU FP8 + FlashInfer / vLLM MTP serving
├── src/sre_rl/
│   ├── model/                                # Qwen3.8-27B DeltaNet LoRA discovery & thinking parser
│   ├── data/                                 # PII/Secret scrubber, combinatorial fault factory, splits
│   ├── env/                                  # Tier-1 sub-ms simulator, Tier-2 Kind driver, 4K output limiter
│   ├── safety/                               # Read-before-write state machine & destructive/injection firewall
│   ├── rewards/                              # Verifiable composite reward + Multi-Turn DAPO/GTPO math
│   ├── verl_ext/                             # veRL multi-turn tool router & `compute_score` hook
│   └── eval/                                 # Pass@k, Pass^k (all-k reliability), production gate checks
└── tests/                                    # End-to-end unit & integration test suite
```

---

## Quickstart (Local Verification & Dataset Generation)

```bash
# 1. Install package & dev dependencies
uv sync --extra dev

# 2. Run full test suite (verifies DeltaNet LoRA targets, DAPO/GTPO math, Tier-1 simulator, rewards, scrubber)
uv run pytest -v

# 3. Generate de-contaminated Dataset A (SFT) & Dataset B (Tier-1 RLVR + Unseen Topology/Fault Eval)
make gen-dataset

# 4. Run Pass^5 reliability evaluation comparing policies
uv run python scripts/run_eval_pass_hat_k.py \
  --scenarios data/generated/eval_heldout_scenarios.json \
  --policy expert --k 5
```

---

## Running on GCP `a3-highgpu-4g` (`4× H100 80GB`, `104 vCPUs`, `936 GB RAM`, `3 TB NVMe`)

### 1. Provision the Node
```bash
export GCP_PROJECT_ID="your-gcp-project"
bash infra/gcp_4xh100/provision_a3_4g.sh
```
On boot, `startup_nvme_docker_cuda.sh` stripes the 8× 375 GB local NVMe disks into `/mnt/nvme` (3 TB RAID-0), moves Docker/`kind` storage onto NVMe, and raises `fs.inotify` limits for concurrent Kubernetes sandboxes.

### 2. Bootstrap `Qwen3.8-27B` + Gated DeltaNet Kernels + `veRL`
```bash
bash infra/gcp_4xh100/setup_verl_qwen38_env.sh
source /mnt/nvme/venv-sre-rl/bin/activate
```

### 3. Execute the 3-Stage Post-Training Pipeline
```bash
# Stage 1: Cold-Start SFT (~2-3 hrs on 4x H100) — aligns tool protocol & read-before-write discipline
bash scripts/run_sft_4xh100.sh

# Stage 2: Tier-1 Deterministic Snapshot RLVR (~800 steps, DAPO + GTPO, SGLang FP8 KV cache)
bash scripts/run_rl_tier1_4xh100.sh

# Stage 3: Tier-2 Live Kind Cluster Async RLVR (GPUs 0,1 Rollout | GPUs 2,3 Trainer | 8 NVMe Kind clusters)
bash scripts/run_rl_tier2_live_4xh100.sh
```

### 4. Gate on `Pass^5` & Serve at "Flash" Speed
```bash
# Start single-GPU FP8 SGLang server (or ENGINE=vllm for 3-token qwen3_5_mtp speculative decoding)
bash scripts/serve_flash_sglang.sh

# Verify Pass^5 >= 60%, Safety Violation Rate == 0.0%, False-Positive Action Rate < 2%
python3 scripts/run_eval_pass_hat_k.py \
  --scenarios /mnt/nvme/data/eval_heldout_scenarios.json \
  --policy sglang_endpoint \
  --endpoint-url http://127.0.0.1:30000/v1/chat/completions \
  --k 5
```
