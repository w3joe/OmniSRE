#!/usr/bin/env python3
"""Merge a PEFT Rank-64 Hybrid LoRA checkpoint into a clean standalone BF16 OmniSRE model.

Verifies all invariants required for zero-wrapper SGLang / vLLM / `transformers` loading:
- Zero residual `.base_layer.` or `.lora_*` keys in `model.safetensors.index.json`
- Exact `1,184` canonical backbone keys for `Qwen3.8-27B`
- Preserves `processor_config.json`, `chat_template.jinja`, `generation_config.json`, and tokenizer files
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def merge_peft_to_standalone_bf16(
    base_model_dir: Path,
    adapter_dir: Path,
    out_dir: Path,
    max_shard_size: str = "5GB",
    expected_keys: int = 1184,
) -> dict[str, int]:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        str(base_model_dir),
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged = peft_model.merge_and_unload()

    clean_sd = {
        k.replace(".base_layer.", "."): v.to(torch.bfloat16)
        for k, v in merged.state_dict().items()
        if ".lora_A." not in k and ".lora_B." not in k and ".lora_embedding_" not in k
    }
    n_base_layer = sum(1 for k in clean_sd if "base_layer" in k)
    n_lora = sum(1 for k in clean_sd if "lora" in k.lower())
    if n_base_layer != 0 or n_lora != 0:
        raise RuntimeError(f"Residual PEFT keys detected: base_layer={n_base_layer}, lora={n_lora}")
    if expected_keys > 0 and len(clean_sd) != expected_keys:
        raise RuntimeError(f"Expected {expected_keys} canonical keys, got {len(clean_sd)}")

    merged.save_pretrained(str(out_dir), state_dict=clean_sd, safe_serialization=True, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(str(out_dir))

    for extra_name in (
        "processor_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        src_file = base_model_dir / extra_name
        if src_file.exists() and not (out_dir / extra_name).exists():
            shutil.copy2(src_file, out_dir / extra_name)

    idx_path = out_dir / "model.safetensors.index.json"
    if idx_path.exists():
        wmap = json.loads(idx_path.read_text(encoding="utf-8")).get("weight_map", {})
        bad = [k for k in wmap if "base_layer" in k or "lora" in k.lower()]
        if bad:
            raise RuntimeError(f"Index contains residual PEFT keys: {bad[:5]}")

    stats = {"total_keys": len(clean_sd), "base_layer_keys": n_base_layer, "lora_keys": n_lora}
    print(f"[merge] Clean standalone BF16 saved -> {out_dir} | {json.dumps(stats)}", flush=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge OmniSRE LoRA adapter into standalone BF16 weights")
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-shard-size", type=str, default="5GB")
    parser.add_argument("--expected-keys", type=int, default=1184)
    args = parser.parse_args()

    merge_peft_to_standalone_bf16(
        base_model_dir=args.base_model_dir,
        adapter_dir=args.adapter_dir,
        out_dir=args.out_dir,
        max_shard_size=args.max_shard_size,
        expected_keys=args.expected_keys,
    )


if __name__ == "__main__":
    main()
