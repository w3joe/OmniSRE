#!/usr/bin/env python3
"""Stage 1 Cold-Start SFT Trainer for `Qwen/Qwen3.8-27B` (Dense) on 4x H100.

Applies Rank-64 BF16 LoRA across BOTH Gated Attention (16 layers) AND Gated DeltaNet
(48 layers) + SwiGLU MLP projections while explicitly freezing the Vision Tower and
MTP heads, then exports a merged BF16 checkpoint ready for Stage 2 `veRL` RLVR.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sre_rl.model.qwen38_lora import (
    Qwen38LoRASpec,
    build_qwen38_peft_lora_kwargs,
    discover_qwen38_lora_targets,
    freeze_non_lora_subsystems,
)
from sre_rl.model.thinking import format_multiturn_preserve_thinking


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 Qwen3.8-27B Dense LoRA SFT")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sft_qwen38_27b_lora_4xh100.yaml"),
    )
    parser.add_argument(
        "--merge-output-dir",
        type=Path,
        default=Path("/mnt/nvme/checkpoints/qwen38-27b-sre-sft-merged"),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Optional max optimizer steps override for smoke-testing.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merging LoRA into a 54GB BF16 checkpoint at end of run.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_id = cfg["model"]["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
    )

    # 1. Freeze Vision Tower & Multi-Token Prediction (MTP) heads
    spec = Qwen38LoRASpec(
        rank=int(cfg["lora"]["rank"]),
        alpha=int(cfg["lora"]["alpha"]),
        dropout=float(cfg["lora"]["dropout"]),
        freeze_vision_tower=bool(cfg["model"].get("freeze_vision_tower", True)),
        freeze_mtp_heads=bool(cfg["model"].get("freeze_mtp_heads", True)),
    )
    freeze_stats = freeze_non_lora_subsystems(model, spec=spec)
    print(f"[sft] Frozen non-LLM params: {freeze_stats}")

    # 2. Verify Gated DeltaNet + Attention LoRA target modules exist in model tree
    module_pairs = [(name, type(mod).__name__) for name, mod in model.named_modules()]
    discovered = discover_qwen38_lora_targets(module_pairs, require_deltanet=True)
    print(f"[sft] Verified Qwen3.8-27B hybrid LoRA leaf targets: {discovered}")

    peft_kwargs = build_qwen38_peft_lora_kwargs(spec)
    peft_kwargs["target_modules"] = discovered
    lora_cfg = LoraConfig(**peft_kwargs)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 3. Load cold-start SRE trajectories & apply bounded `preserve_thinking`
    raw_ds = load_dataset("json", data_files=cfg["data"]["train_jsonl"], split="train")
    preserve = bool(cfg["data"].get("preserve_thinking", True))

    def _format_row(example):
        msgs = format_multiturn_preserve_thinking(
            example["messages"], preserve_thinking=preserve
        )
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    train_ds = raw_ds.map(_format_row, remove_columns=raw_ds.column_names)

    t_cfg = cfg["training"]
    sft_args = SFTConfig(
        output_dir=t_cfg["output_dir"],
        dataset_text_field="text",
        max_length=int(cfg["data"]["max_seq_length"]),
        num_train_epochs=float(t_cfg["num_train_epochs"]),
        max_steps=int(args.max_steps),
        per_device_train_batch_size=int(t_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(t_cfg["gradient_accumulation_steps"]),
        learning_rate=float(t_cfg["learning_rate"]),
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_steps=int(t_cfg.get("warmup_steps", 15)),
        weight_decay=float(t_cfg["weight_decay"]),
        max_grad_norm=float(t_cfg["max_grad_norm"]),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=int(t_cfg["logging_steps"]),
        save_steps=int(t_cfg["save_steps"]),
        seed=int(t_cfg["seed"]),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        args=sft_args,
    )
    trainer.train()
    trainer.save_model(t_cfg["output_dir"])

    # 4. Merge LoRA into BF16 base checkpoint on rank 0 for Stage 2 veRL initialization
    if not args.skip_merge and trainer.accelerator.is_main_process:
        print(f"[sft] Merging LoRA adapter into {args.merge_output_dir} for Stage 2 RLVR...")
        merged = model.merge_and_unload()
        args.merge_output_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(args.merge_output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.merge_output_dir)
        print("[sft] Merged checkpoint saved.")


if __name__ == "__main__":
    main()
