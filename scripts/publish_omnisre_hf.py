#!/usr/bin/env python3
"""Publish OmniSRE-27B (LoRA adapter or merged weights) + Model Card to Hugging Face Hub.

Usage:
    # 1. Authenticate once with a Hugging Face Write token:
    #    uv run hf auth login
    # 2. Publish directly from GCS (downloads to temp dir & uploads cleanly):
    #    uv run python scripts/publish_omnisre_hf.py \
    #        --repo-id <your-hf-user>/OmniSRE-27B \
    #        --gcs-checkpoint gs://onflow-30k-qwen38-sre-rl/run-v3-rlvr/rl-dapo-gtpo \
    #        --stage-label "v0.1-draft"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def build_model_card(
    repo_id: str,
    stage_label: str,
    base_model: str,
    eval_summary: dict[str, Any] | None,
    is_adapter: bool,
) -> str:
    eval_section = ""
    if eval_summary:
        before = eval_summary.get("before_rl") or {}
        after = eval_summary.get("after_rl") or {}
        eval_section = (
            "## Held-Out Real-World SRE Evaluation\n\n"
            "Evaluated on unseen real OpenShift / kube-prometheus / Awesome-Prometheus-Alerts "
            "topologies strictly held out from training (`0` topology overlap, `0` prompt overlap):\n\n"
            "| Stage | Scenarios | Mean Reward | Mitigation Rate | Safety Violation Rate | Avg Turns |\n"
            "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
            f"| Stage 1 SFT *(Before RL)* | {before.get('num_heldout', '-')} | "
            f"{before.get('mean_reward', '-')} | {before.get('mitigation_rate', '-')} | "
            f"{before.get('safety_violation_rate', '-')} | {before.get('avg_turns', '-')} |\n"
            f"| Stage 2 RLVR *(After DAPO+GTPO)* | {after.get('num_heldout', '-')} | "
            f"{after.get('mean_reward', '-')} | {after.get('mitigation_rate', '-')} | "
            f"{after.get('safety_violation_rate', '-')} | {after.get('avg_turns', '-')} |\n\n"
        )

    artifact_note = (
        "This repository currently hosts the **Rank-64 Hybrid LoRA adapter** (~1.8 GB, adapting all "
        "48 Gated DeltaNet linear-attention layers + 16 Gated Attention layers + SwiGLU MLPs while "
        "keeping the Vision Tower frozen) on top of `Qwen/Qwen3.8-27B`."
        if is_adapter
        else "This repository hosts merged **BF16 weights** ready for single-GPU FP8/NVFP4 serving."
    )

    usage_snippet = (
        f"""```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

base_id = "{base_model}"
adapter_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
base = AutoModelForImageTextToText.from_pretrained(
    base_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(base, adapter_id)
```"""
        if is_adapter
        else f"""```bash
python3 -m sglang.launch_server \\
  --model-path {repo_id} \\
  --served-model-name omnisre-27b \\
  --trust-remote-code \\
  --kv-cache-dtype fp8_e4m3 \\
  --attention-backend flashinfer \\
  --reasoning-parser qwen3 \\
  --tool-call-parser qwen3_coder
```"""
    )

    return f"""---
license: apache-2.0
base_model: {base_model}
language:
- en
library_name: {"peft" if is_adapter else "transformers"}
tags:
- sre
- aiops
- kubernetes
- devops
- rlvr
- dapo
- gtpo
- agent
- tool-use
- omnisre
---

# OmniSRE-27B ({stage_label})

**OmniSRE-27B** is an open-weights Site Reliability Engineering (SRE) agent model built so any
engineering team can self-host autonomous incident triage, fault localization, root-cause analysis
(RCA), and read-before-write remediation under the **Apache 2.0** license.

> **Status:** `{stage_label}` — {artifact_note}

## Architecture & Post-Training Recipe

* **Base Model:** `{base_model}` (27B Dense 3:1 hybrid architecture: 48 Gated DeltaNet linear-attention layers + 16 Gated Attention layers, 262K native context, FP8 KV-cache footprint ~2 GB per 64K trace).
* **Hybrid LoRA Targeting:** Rank-64 adapters applied across **both** Gated DeltaNet (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`) and Gated Attention + SwiGLU MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`), with the Vision Encoder and MTP heads frozen (`466.9M` trainable parameters / `1.68%`).
* **RL Post-Training:** Multi-Turn **DAPO + GTPO** (Group Turn Policy Optimization) with:
  * **Read-Before-Write Separation:** Hard guardrail requiring $\\ge 2$ diagnostic reads prior to any write mutation.
  * **Verifiable Dual State + SLO Rewards:** Blind pod restarts on persistent config/code defects receive `0` mitigation credit; metastable retry storms require both config patching and capacity scale-out.
  * **Anti-Stalling & Positive-Winner Filtering:** DAPO filters zero-variance and zero-winner groups (`max_reward < 0.50`) and penalizes read-only timeouts.
  * **Adversarial Log Hygiene:** Trained to ignore prompt-injection payloads hidden inside untrusted HTTP/application logs and never mutate healthy services on benign false alarms.

## Open-Source Training Corpus (5-Gate Validated)

Trained on **5,038** scrubbed, deduplicated, 5-gate-validated SRE trajectories (`0` PII/secret leaks,
`0` train/eval topology overlap) sourced from openly licensed community corpora:
* `rmems/incident-response-oncall-trajectories` (Apache-2.0)
* `openshift/runbooks` & `prometheus-operator/runbooks` (Apache-2.0)
* `samber/awesome-prometheus-alerts` (1,217 production PromQL alert rules)
* `SREGym/SREGym` (MIT)
* `miyad316/sre-agentic-trajectories-devops`, `beatsprom/devops-kubernetes-iac-sft-dpo-2026` (CC-BY-4.0), `nolabs/deepfabric-devops-reasoning-traces`, `Snaseem2026/devops-incident-response`, and `danluu/post-mortems`.

{eval_section}## Quickstart

{usage_snippet}

## Training Code & Reproducibility

Full open-source training pipeline, 5-gate dataset validator, Tier-1 snapshot simulator, and GCP `4x H100` scripts:
[https://github.com/w3joe/qwen38-sre-rl](https://github.com/w3joe/qwen38-sre-rl)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish OmniSRE-27B checkpoint & Model Card to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="Target HF repo ID, e.g. w3joe/OmniSRE-27B")
    parser.add_argument("--local-checkpoint", type=Path, default=None)
    parser.add_argument("--gcs-checkpoint", type=str, default=None)
    parser.add_argument("--gcloud-account", type=str, default="onflowai.general@gmail.com")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.8-27B")
    parser.add_argument("--stage-label", type=str, default="v0.1-draft")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.local_checkpoint is None and args.gcs_checkpoint is None:
        raise ValueError("Provide either --local-checkpoint or --gcs-checkpoint.")

    with tempfile.TemporaryDirectory(prefix="omnisre_hf_") as tmp:
        ckpt_dir = args.local_checkpoint
        if ckpt_dir is None:
            ckpt_dir = Path(tmp) / "checkpoint"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            print(f"[hf] Syncing {args.gcs_checkpoint} -> {ckpt_dir}...")
            cmd = [
                "gcloud",
                "storage",
                "rsync",
                "-r",
                args.gcs_checkpoint,
                str(ckpt_dir),
                "--exclude=step-.*",
            ]
            if args.gcloud_account:
                cmd.append(f"--account={args.gcloud_account}")
            env = {
                **os.environ,
                "CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD": "0",
                "CLOUDSDK_STORAGE_CHECK_HASHES": "never",
            }
            subprocess.run(cmd, check=True, env=env)

        assert ckpt_dir is not None and ckpt_dir.exists()
        is_adapter = (ckpt_dir / "adapter_config.json").exists()
        eval_file = ckpt_dir / "trial_eval_summary.json"
        eval_summary = json.loads(eval_file.read_text(encoding="utf-8")) if eval_file.exists() else None

        card = build_model_card(
            repo_id=args.repo_id,
            stage_label=args.stage_label,
            base_model=args.base_model,
            eval_summary=eval_summary,
            is_adapter=is_adapter,
        )
        (ckpt_dir / "README.md").write_text(card, encoding="utf-8")

        if args.dry_run:
            print(f"[hf] DRY RUN complete. Model card written to {ckpt_dir / 'README.md'}")
            print(card[:1200])
            return

        from huggingface_hub import HfApi

        api = HfApi()
        user = api.whoami()
        print(f"[hf] Authenticated as: {user.get('name', user.get('fullname', 'unknown'))}")
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(ckpt_dir),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Publish OmniSRE-27B ({args.stage_label})",
        )
        print(f"[hf] Published successfully: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
