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
        "--resume-adapter-dir",
        type=Path,
        default=None,
        help="Optional existing LoRA adapter dir to continue Phase 1b SFT from.",
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=None,
        help="Optional override for SFT training JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for SFT LoRA adapter output dir.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Optional max optimizer steps override for smoke-testing.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Optional override for base model path (e.g., /mnt/nvme/omnisre-v1.0 for Phase-3 SFT).",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merging LoRA into a 54GB BF16 checkpoint at end of run.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    import json
    import shutil

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_id = args.model_path.strip() if args.model_path.strip() else cfg["model"]["model_name_or_path"]
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

    # 2. Attach fresh hybrid LoRA OR resume from Stage 1a adapter for Phase 1b schema calibration
    if (
        args.resume_adapter_dir is not None
        and args.resume_adapter_dir.exists()
        and (args.resume_adapter_dir / "adapter_config.json").exists()
    ):
        print(f"[sft] Resuming Phase 1b SFT from adapter: {args.resume_adapter_dir}")
        model = PeftModel.from_pretrained(model, str(args.resume_adapter_dir), is_trainable=True)
    else:
        module_pairs = [(name, type(mod).__name__) for name, mod in model.named_modules()]
        discovered = discover_qwen38_lora_targets(module_pairs, require_deltanet=True)
        print(f"[sft] Verified Qwen3.8-27B hybrid LoRA leaf targets: {discovered}")

        peft_kwargs = build_qwen38_peft_lora_kwargs(spec)
        peft_kwargs["target_modules"] = discovered
        lora_cfg = LoraConfig(**peft_kwargs)
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 3. Load cold-start SRE trajectories & apply bounded `preserve_thinking`
    # Load via Python dict projection (`{"messages": ...}`) so heterogeneous metadata columns
    # across multi-source scraped corpora never trigger PyArrow chunk CastError.
    train_jsonl_path = Path(args.train_jsonl) if args.train_jsonl is not None else Path(cfg["data"]["train_jsonl"])
    preserve = bool(cfg["data"].get("preserve_thinking", True))
    formatted_texts: list[dict[str, str]] = []
    with train_jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = format_multiturn_preserve_thinking(
                row["messages"], preserve_thinking=preserve
            )
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            formatted_texts.append({"text": text})
    train_ds = Dataset.from_list(formatted_texts)

    t_cfg = cfg["training"]
    out_dir = str(args.output_dir) if args.output_dir is not None else t_cfg["output_dir"]
    assistant_only = bool(cfg["data"].get("assistant_only_loss", True))

    sft_kwargs: dict[str, object] = {
        "output_dir": out_dir,
        "dataset_text_field": "text",
        "max_length": int(cfg["data"]["max_seq_length"]),
        "num_train_epochs": float(t_cfg["num_train_epochs"]),
        "max_steps": int(args.max_steps),
        "per_device_train_batch_size": int(t_cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(t_cfg["gradient_accumulation_steps"]),
        "learning_rate": float(t_cfg["learning_rate"]),
        "lr_scheduler_type": t_cfg["lr_scheduler_type"],
        "warmup_steps": int(t_cfg.get("warmup_steps", 15)),
        "weight_decay": float(t_cfg["weight_decay"]),
        "max_grad_norm": float(t_cfg["max_grad_norm"]),
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": int(t_cfg["logging_steps"]),
        "save_steps": int(t_cfg["save_steps"]),
        "seed": int(t_cfg["seed"]),
        "report_to": "none",
    }
    sft_args = SFTConfig(**sft_kwargs)  # type: ignore[arg-type]

    # Version-proof assistant-only label masking collator (`labels = -100` outside `<|im_start|>assistant ... <|im_end|>`)
    im_start_assistant = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    im_end_toks = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_id = im_end_toks[0] if im_end_toks else int(tokenizer.eos_token_id)

    def _assistant_only_collator(features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        batch_ids: list[list[int]] = []
        for f in features:
            if "input_ids" in f:
                ids = [int(x) for x in f["input_ids"]]  # type: ignore[union-attr]
            else:
                ids = tokenizer.encode(str(f["text"]), add_special_tokens=False)[: int(cfg["data"]["max_seq_length"])]
            batch_ids.append(ids)
        max_l = max(len(x) for x in batch_ids)
        pad_id = int(tokenizer.pad_token_id)
        in_ten = torch.full((len(batch_ids), max_l), pad_id, dtype=torch.long)
        att_ten = torch.zeros((len(batch_ids), max_l), dtype=torch.long)
        lbl_ten = torch.full((len(batch_ids), max_l), -100, dtype=torch.long)

        pat_len = len(im_start_assistant)
        for row_i, ids in enumerate(batch_ids):
            in_ten[row_i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            att_ten[row_i, : len(ids)] = 1
            if not assistant_only or pat_len == 0:
                lbl_ten[row_i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                continue
            pos = 0
            while pos <= len(ids) - pat_len:
                if ids[pos : pos + pat_len] == im_start_assistant:
                    start_gen = pos + pat_len
                    end_gen = start_gen
                    while end_gen < len(ids) and ids[end_gen] != im_end_id:
                        end_gen += 1
                    if end_gen < len(ids) and ids[end_gen] == im_end_id:
                        end_gen += 1
                    lbl_ten[row_i, start_gen:end_gen] = torch.tensor(ids[start_gen:end_gen], dtype=torch.long)
                    pos = end_gen
                else:
                    pos += 1
        return {"input_ids": in_ten, "attention_mask": att_ten, "labels": lbl_ten}

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        args=sft_args,
        data_collator=_assistant_only_collator if assistant_only else None,
    )
    trainer.train()
    trainer.save_model(out_dir)

    # 4. Merge LoRA into BF16 base checkpoint on rank 0 for Stage 2 SGLang/veRL initialization
    trainer.accelerator.wait_for_everyone()
    if not args.skip_merge and trainer.accelerator.is_main_process:
        print(f"[sft] Merging LoRA adapter into {args.merge_output_dir} for Stage 2 RLVR...")
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        merged = unwrapped.merge_and_unload()
        clean_sd = {
            k.replace(".base_layer.", "."): v.to(torch.bfloat16)
            for k, v in merged.state_dict().items()
            if ".lora_A." not in k and ".lora_B." not in k and ".lora_embedding_" not in k
        }
        args.merge_output_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(args.merge_output_dir, state_dict=clean_sd, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(args.merge_output_dir)
        src_dir = Path(model_id)
        if src_dir.is_dir():
            for extra_name in (
                "processor_config.json",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
                "chat_template.jinja",
                "generation_config.json",
            ):
                if (src_dir / extra_name).exists() and not (args.merge_output_dir / extra_name).exists():
                    shutil.copy2(src_dir / extra_name, args.merge_output_dir / extra_name)
        print(f"[sft] Clean standalone BF16 checkpoint saved ({len(clean_sd)} keys).")
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
