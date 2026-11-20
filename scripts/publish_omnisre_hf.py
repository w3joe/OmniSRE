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

        def row(name: str, d: dict[str, Any]) -> str:
            return (
                f"| {name} | {d.get('num_heldout', '-')} | {d.get('mean_reward', '-')} | "
                f"{d.get('mitigation_rate', '-')} | {d.get('safety_violation_rate', '-')} | "
                f"{d.get('avg_turns', '-')} |\n"
            )

        eval_section = (
            "## Held-out evaluation\n\n"
            "Unseen OpenShift, kube-prometheus and Prometheus-alert topologies "
            "(zero topology or prompt overlap with training).\n\n"
            "| Stage | Scenarios | Reward | Mitigated | Safety violations | Turns |\n"
            "| :-- | --: | --: | --: | --: | --: |\n"
            + row("SFT", before)
            + row("SFT + RLVR", after)
            + "\n"
        )

    artifact = f"Rank-64 LoRA adapter for `{base_model}`" if is_adapter else "Merged BF16 weights"

    usage_snippet = (
        f"""```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{base_model}", trust_remote_code=True)
base = AutoModelForImageTextToText.from_pretrained(
    "{base_model}", dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(base, "{repo_id}")
```"""
        if is_adapter
        else f"""```bash
python3 -m sglang.launch_server --model-path {repo_id} \\
  --served-model-name omnisre-27b --trust-remote-code \\
  --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer \\
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
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

<img src="assets/omnisre_hero.png" alt="OmniSRE 27B" width="100%"/>

**OmniSRE** is an open-weights site reliability engineering agent: it triages, localizes, \
finds root cause and remediates, and it never writes before it reads.

{artifact} · `{stage_label}` · Apache 2.0 · \
[Code & training pipeline](https://github.com/w3joe/qwen38-sre-rl)

<br/>

<img src="assets/omnisre_architecture.png" alt="Read-before-write agent loop and hybrid backbone" \
width="100%"/>

<br/>

<img src="assets/omnisre_training_curves.png" alt="Stage 1a and Phase 1b training loss and \
token accuracy" width="100%"/>

{eval_section}<br/>

<img src="assets/omnisre_dataset_infographic.png" alt="Open SRE corpus and five validation \
gates" width="100%"/>

## Quickstart

{usage_snippet}

<details>
<summary>Training details</summary>

| | |
| :-- | :-- |
| Base | `{base_model}` · 64 layers, 3 : 1 Gated DeltaNet / Gated Attention, 262K context |
| Adapter | LoRA r = 64 on all linear projections · 466.9M trainable (1.68%) |
| Frozen | Vision tower, MTP head |
| Stage 1a | 5,038 real SRE traces · 157 steps · loss 3.34 → 0.175 |
| Phase 1b | 1,559 exact-schema traces · 98 steps · loss 0.057 · 98.5% token accuracy |
| Next | Read-before-write RLVR with DAPO + GTPO |
| Hardware | 4 × H100 BF16 for training · 1 × H100 / L40S FP8 for serving |

</details>

<details>
<summary>Data sources</summary>

`rmems/incident-response-oncall-trajectories` (Apache-2.0) ·
`nolabs/deepfabric-devops-reasoning-traces` ·
`miyad316/sre-agentic-trajectories-devops` ·
`beatsprom/devops-kubernetes-iac-sft-dpo-2026` (CC-BY-4.0) ·
`Snaseem2026/devops-incident-response` ·
`samber/awesome-prometheus-alerts` ·
`openshift/runbooks` (Apache-2.0) ·
`prometheus-operator/runbooks` (Apache-2.0) ·
`SREGym/SREGym` (MIT) ·
`danluu/post-mortems`

</details>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish OmniSRE-27B checkpoint & Model Card to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id", type=str, required=True, help="Target HF repo ID, e.g. w3joe/OmniSRE-27B"
    )
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
        eval_summary = (
            json.loads(eval_file.read_text(encoding="utf-8")) if eval_file.exists() else None
        )

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
        api.create_repo(
            repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True
        )
        api.upload_folder(
            folder_path=str(ckpt_dir),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Publish OmniSRE-27B ({args.stage_label})",
        )
        print(f"[hf] Published successfully: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
