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

<p align="center">
  <img src="assets/omnisre_hero.png" alt="OmniSRE-27B Hero Banner" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/w3joe/qwen38-sre-rl"><img src="https://img.shields.io/badge/GitHub-Code%20%26%20RLVR%20Pipeline-181717?style=for-the-badge&logo=github" alt="GitHub"/></a>
  <img src="https://img.shields.io/badge/License-Apache%202.0-10B981?style=for-the-badge" alt="Apache 2.0"/>
  <img src="https://img.shields.io/badge/Release-{stage_label.replace('-', '--')}-38BDF8?style=for-the-badge" alt="Release"/>
  <img src="https://img.shields.io/badge/Hardware-1x%20H100%20%2F%20L40S%20FP8-A78BFA?style=for-the-badge" alt="Hardware"/>
</p>

> **OmniSRE** is an open-weights Site Reliability Engineering agent built so any team can self-host autonomous incident triage, fault localization, RCA, and read-before-write remediation. {artifact_note}

---

### Agent Architecture & Safety Guardrails

<p align="center">
  <img src="assets/omnisre_architecture.png" alt="OmniSRE Read-Write Separated Architecture" width="100%"/>
</p>

---

### Training Dynamics *(4× NVIDIA H100)*

<p align="center">
  <img src="assets/omnisre_training_curves.png" alt="OmniSRE Training Loss and Accuracy Curves" width="100%"/>
</p>

{eval_section}---

### Open-Source SRE Corpus & 5-Gate Validation

<p align="center">
  <img src="assets/omnisre_dataset_infographic.png" alt="OmniSRE Dataset Breakdown and 5-Gate Validation" width="100%"/>
</p>

---

### Quickstart

{usage_snippet}

<details>
<summary><b>Open-Source Corpus Credits & Training Specs</b></summary>

* **Base:** `{base_model}` (27B Dense 3:1 Hybrid Gated DeltaNet + Gated Attention, 262K native context)
* **Adapter:** Rank-64 Hybrid LoRA (`466.9M` trainable params / `1.68%`; Vision Encoder & MTP frozen)
* **Data Sources:** `rmems/incident-response-oncall-trajectories` (Apache-2.0), `openshift/runbooks` (Apache-2.0), `prometheus-operator/runbooks` (Apache-2.0), `samber/awesome-prometheus-alerts`, `SREGym/SREGym` (MIT), `miyad316/sre-agentic-trajectories-devops`, `beatsprom/devops-kubernetes-iac-sft-dpo-2026` (CC-BY-4.0), `nolabs/deepfabric-devops-reasoning-traces`, `Snaseem2026/devops-incident-response`, `danluu/post-mortems`.
* **Pipeline Repo:** [github.com/w3joe/qwen38-sre-rl](https://github.com/w3joe/qwen38-sre-rl)

</details>
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
