#!/usr/bin/env python3
"""10x Faster Split-GPU SGLang + DDP Multi-Turn DAPO+GTPO RLVR Trainer for 4x H100 (`a3-highgpu-4g`).

Hardware Partition on 4x NVIDIA H100 80GB:
- GPUs 0, 1: Persistent SGLang TP2 Rollout Server (`--kv-cache-dtype fp8_e4m3`,
  FlashInfer, Radix Prefix Caching). Drives 32-64 concurrent multi-turn SRE episodes
  at ~1,800-2,500 aggregate tok/s (~25-40s per 64-trajectory batch vs ~25m in HF `.generate()`).
- GPUs 2, 3: DDP Rank-64 Hybrid LoRA (`Qwen3.8-27B` Gated DeltaNet + Gated Attention)
  DAPO + GTPO Policy Trainer using `logits_to_keep` (only computes 248K-vocab `lm_head`
  on active assistant `<think>` + `<tool_call>` positions) and zero-extra-VRAM reference KL.

After each optimizer step, rank 0 exports merged BF16 weights to `/mnt/nvme/sglang_live_sync`
and invokes SGLang's `/update_weights_from_disk` endpoint for zero-restart live policy sync.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import random
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from sre_rl.env.tier1_snapshot_env import Tier1SnapshotEnvironment
from sre_rl.model.qwen38_lora import (
    Qwen38LoRASpec,
    build_qwen38_peft_lora_kwargs,
    discover_qwen38_lora_targets,
    freeze_non_lora_subsystems,
)
from sre_rl.model.thinking import build_sre_system_prompt, format_multiturn_preserve_thinking
from sre_rl.rewards.dapo_gtpo import DAPOConfig, RolloutGroupItem, compute_gtpo_hybrid_advantages
from sre_rl.rewards.verifiable_reward import score_episode
from sre_rl.schemas import IncidentScenario


def _post_json(url: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _chat_template_ids(
    tokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    if isinstance(encoded, list):
        return [int(t) for t in encoded]
    ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(t) for t in ids]


def rollout_episode_via_sglang(
    sglang_base_url: str,
    served_model: str,
    tokenizer,
    scenario: IncidentScenario,
    max_turns: int = 14,
    max_new_tokens: int = 320,
    max_seq_len: int = 8192,
    temperature: float = 0.85,
) -> tuple[list[int], RolloutGroupItem, dict[str, float]]:
    """Run one full multi-turn SRE episode against the live SGLang TP2 rollout server."""
    env = Tier1SnapshotEnvironment(max_turns=max_turns, max_output_chars=2800)
    alert = env.reset(scenario)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_sre_system_prompt(scenario.reasoning_effort)},
        {"role": "user", "content": alert},
    ]

    full_ids: list[int] = _chat_template_ids(tokenizer, messages, add_generation_prompt=False)
    turn_spans: list[tuple[int, int]] = []
    turn_rewards: list[float] = []
    chat_url = f"{sglang_base_url.rstrip('/')}/v1/chat/completions"

    for _ in range(max_turns):
        # Preserve ALL prior thoughts during RL rollout construction so token offsets
        # in `turn_spans` remain 100% bit-exact against final `full_ids` across all turns.
        fmt_msgs = format_multiturn_preserve_thinking(
            messages, preserve_thinking=True, keep_last_k_thoughts=max_turns
        )
        prompt_ids = _chat_template_ids(tokenizer, fmt_msgs, add_generation_prompt=True)
        if len(prompt_ids) >= max_seq_len - max_new_tokens:
            break

        resp = _post_json(
            chat_url,
            {
                "model": served_model,
                "messages": fmt_msgs,
                "temperature": temperature,
                "top_p": 0.95 if temperature > 0.0 else 1.0,
                "max_tokens": max_new_tokens,
            },
            timeout=90,
        )
        msg_obj = resp["choices"][0]["message"]
        reasoning = (msg_obj.get("reasoning_content") or "").strip()
        content = (msg_obj.get("content") or "").strip()
        if reasoning and "<think>" not in content:
            assistant_text = f"<think>{reasoning}</think>\n{content}"
        else:
            assistant_text = content
        if not assistant_text.strip():
            break

        outcome = env.step(assistant_text)
        messages.append({"role": "assistant", "content": assistant_text})

        # Compute exact token span directly from ChatML template prefix diff (zero drift!)
        after_assistant_ids = _chat_template_ids(
            tokenizer,
            format_multiturn_preserve_thinking(
                messages, preserve_thinking=True, keep_last_k_thoughts=max_turns
            ),
            add_generation_prompt=False,
        )
        start_pos = len(prompt_ids)
        end_pos = len(after_assistant_ids)
        if end_pos <= start_pos:
            break
        full_ids = after_assistant_ids
        turn_spans.append((start_pos, end_pos))
        turn_rewards.append(float(outcome.step_shaping_reward))

        messages.append(
            {"role": "user", "content": f"<tool_response>\n{outcome.observation}\n</tool_response>"}
        )
        if outcome.done:
            break

    ep = env.finalize_episode()
    breakdown = score_episode(scenario, ep)
    item = RolloutGroupItem(
        outcome_reward=float(breakdown.total),
        turn_rewards=turn_rewards,
        turn_token_spans=turn_spans,
        total_seq_len=len(full_ids),
    )
    stats = {
        "reward": float(breakdown.total),
        "mitigated": float(breakdown.mitigated),
        "safety_violated": float(breakdown.safety_violated),
        "turns": float(len(turn_spans)),
    }
    return full_ids, item, stats


def _compute_masked_logprobs(
    model,
    input_ids: torch.Tensor,
    action_mask: torch.Tensor,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_target_pos = torch.nonzero(action_mask[1:] > 0.0, as_tuple=False).squeeze(-1) + 1
    if active_target_pos.numel() == 0:
        return torch.zeros(0, device=input_ids.device, dtype=torch.float32), active_target_pos

    pred_positions = active_target_pos - 1
    logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=pred_positions).logits
    target_ids = input_ids[0, active_target_pos]
    chunks: list[torch.Tensor] = []
    for start in range(0, active_target_pos.numel(), chunk_size):
        end = min(active_target_pos.numel(), start + chunk_size)
        sub_lp = torch.log_softmax(logits[0, start:end, :].float(), dim=-1)
        chunks.append(torch.gather(sub_lp, dim=-1, index=target_ids[start:end].unsqueeze(-1)).squeeze(-1))
    del logits
    return torch.cat(chunks, dim=0), active_target_pos


def sync_lora_weights_to_sglang(
    peft_model,
    tokenizer,
    sglang_base_url: str,
    sync_dir: Path,
    base_model_path: str,
) -> float:
    """Atomically export merged BF16 weights to NVMe RAID-0 and hot-reload SGLang."""
    t0 = time.time()
    tmp_dir = sync_dir.parent / f"{sync_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    was_training = peft_model.training
    peft_model.eval()
    peft_model.merge_adapter()
    try:
        peft_model.base_model.model.save_pretrained(tmp_dir, safe_serialization=True)
        tokenizer.save_pretrained(tmp_dir)
        # Preserve multimodal preprocessor JSONs if present in base_model_path
        src_base = Path(base_model_path)
        if src_base.is_dir():
            for fname in ("preprocessor_config.json", "video_preprocessor_config.json"):
                if (src_base / fname).exists():
                    shutil.copy2(src_base / fname, tmp_dir / fname)
    finally:
        peft_model.unmerge_adapter()
        if was_training:
            peft_model.train()

    if sync_dir.exists():
        shutil.rmtree(sync_dir)
    tmp_dir.rename(sync_dir)

    resp = _post_json(
        f"{sglang_base_url.rstrip('/')}/update_weights_from_disk",
        {"model_path": str(sync_dir)},
        timeout=180,
    )
    if isinstance(resp, dict) and resp.get("success") is False:
        raise RuntimeError(f"SGLang weight sync failed: {resp}")
    return time.time() - t0


def _load_verl_jsonl(path: Path) -> list[IncidentScenario]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        gt = obj["reward_model"]["ground_truth"]
        rows.append(
            IncidentScenario.model_validate_json(gt)
            if isinstance(gt, str)
            else IncidentScenario.model_validate(gt)
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Split-GPU SGLang + DDP Multi-Turn DAPO+GTPO RLVR")
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--sglang-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--served-model", type=str, default="qwen38-sre")
    parser.add_argument("--sync-dir", type=Path, default=Path("/mnt/nvme/sglang_live_sync"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/nvme/checkpoints_v4/rl-sglang"))
    parser.add_argument("--train-scenarios-jsonl", type=Path, required=True)
    parser.add_argument("--heldout-scenarios-jsonl", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--rollout-workers", type=int, default=32)
    parser.add_argument("--eval-heldout-count", type=int, default=40)
    parser.add_argument("--max-turns", type=int, default=14)
    parser.add_argument("--min-group-max-reward", type=float, default=0.50)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--kl-coef", type=float, default=0.04)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    spec = Qwen38LoRASpec(rank=64, alpha=64)
    freeze_non_lora_subsystems(base_model, spec=spec)
    targets = discover_qwen38_lora_targets(
        [(n, type(m).__name__) for n, m in base_model.named_modules()],
        require_deltanet=True,
    )
    peft_kwargs = build_qwen38_peft_lora_kwargs(spec)
    peft_kwargs["target_modules"] = targets
    peft_model = get_peft_model(base_model, LoraConfig(**peft_kwargs))
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ddp_model = DDP(peft_model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        [p for p in ddp_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    train_scenarios = _load_verl_jsonl(args.train_scenarios_jsonl)
    heldout_scenarios = _load_verl_jsonl(args.heldout_scenarios_jsonl)
    random.Random(args.seed).shuffle(train_scenarios)
    random.Random(args.seed + 7).shuffle(heldout_scenarios)
    eval_slice = heldout_scenarios[: args.eval_heldout_count]

    dapo_cfg = DAPOConfig(
        group_size=args.group_size,
        clip_eps_low=0.20,
        clip_eps_high=0.28,
        gamma=0.92,
        outcome_dampening_lambda=0.30,
        min_group_max_reward=args.min_group_max_reward,
        gate_positive_shaping_on_success=True,
    )

    def _eval_heldout(label: str) -> dict[str, Any]:
        if local_rank != 0:
            return {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.rollout_workers) as pool:
            futs = [
                pool.submit(
                    rollout_episode_via_sglang,
                    args.sglang_url,
                    args.served_model,
                    tokenizer,
                    sc,
                    args.max_turns,
                    320,
                    8192,
                    0.0,
                )
                for sc in eval_slice
            ]
            stats = [f.result()[2] for f in futs]
        summary = {
            "eval_stage": label,
            "num_heldout": len(stats),
            "mean_reward": round(float(np.mean([s["reward"] for s in stats])), 4),
            "mitigation_rate": round(float(np.mean([s["mitigated"] for s in stats])), 4),
            "safety_violation_rate": round(float(np.mean([s["safety_violated"] for s in stats])), 4),
            "avg_turns": round(float(np.mean([s["turns"] for s in stats])), 2),
        }
        print(f"[heldout-eval] {json.dumps(summary)}", flush=True)
        return summary

    before_eval = _eval_heldout("before_rl_phase1b_sft")
    dist.barrier()

    cursor = 0
    filtered_zero_var = 0
    filtered_zero_winner = 0

    for step in range(1, args.steps + 1):
        t0 = time.time()

        # Rank 0 dispatches concurrent SGLang rollouts & performs DAPO group filtering
        payload_for_ranks: list[tuple[list[int], np.ndarray, np.ndarray]] = []
        step_metrics: dict[str, Any] = {}

        if local_rank == 0:
            accepted_groups = 0
            all_rewards: list[float] = []
            all_mit: list[float] = []
            all_safe: list[float] = []
            all_turns: list[float] = []

            max_attempts_this_step = args.prompts_per_step * 5
            attempts_this_step = 0

            while accepted_groups < args.prompts_per_step and attempts_this_step < max_attempts_this_step:
                needed = args.prompts_per_step - accepted_groups
                batch_sc = [
                    train_scenarios[(cursor + j) % len(train_scenarios)]
                    for j in range(needed)
                ]
                cursor += len(batch_sc)
                attempts_this_step += len(batch_sc)

                jobs = [(sc_i, sc) for sc_i, sc in enumerate(batch_sc) for _ in range(args.group_size)]
                grouped: dict[int, list[tuple[list[int], RolloutGroupItem, dict[str, float]]]] = {
                    i: [] for i in range(len(batch_sc))
                }
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.rollout_workers) as pool:
                    fut_map = {
                        pool.submit(
                            rollout_episode_via_sglang,
                            args.sglang_url,
                            args.served_model,
                            tokenizer,
                            sc,
                            args.max_turns,
                            320,
                            8192,
                            0.85,
                        ): sc_i
                        for sc_i, sc in jobs
                    }
                    for fut, sc_i in fut_map.items():
                        grouped[sc_i].append(fut.result())

                for sc_i in range(len(batch_sc)):
                    if accepted_groups >= args.prompts_per_step:
                        break
                    traces = grouped[sc_i]
                    items = [t[1] for t in traces]
                    stats = [t[2] for t in traces]
                    rew = np.array([it.outcome_reward for it in items], dtype=np.float64)

                    if float(np.std(rew)) < dapo_cfg.min_group_std:
                        filtered_zero_var += 1
                        continue
                    if float(np.max(rew)) < dapo_cfg.min_group_max_reward:
                        filtered_zero_winner += 1
                        continue

                    advs, masks = compute_gtpo_hybrid_advantages(items, config=dapo_cfg)
                    for idx, tr in enumerate(traces):
                        # Pre-filter empty-mask rollouts BEFORE broadcast so every DDP rank
                        # executes the EXACT same number of backward passes (zero NCCL deadlock!)
                        if float(np.sum(masks[idx][1:])) > 0.0 and len(tr[0]) > 2:
                            payload_for_ranks.append((tr[0], advs[idx], masks[idx]))
                    all_rewards.extend(rew.tolist())
                    all_mit.extend(s["mitigated"] for s in stats)
                    all_safe.extend(s["safety_violated"] for s in stats)
                    all_turns.extend(s["turns"] for s in stats)
                    accepted_groups += 1

            # Truncate to exact multiple of `world_size` so all DDP ranks match micro-step count
            usable = (len(payload_for_ranks) // world_size) * world_size
            payload_for_ranks = payload_for_ranks[:usable]

            if payload_for_ranks:
                step_metrics = {
                    "rl_step": step,
                    "trajectories": len(payload_for_ranks),
                    "batch_mean_reward": round(float(np.mean(all_rewards)), 4),
                    "batch_max_reward": round(float(np.max(all_rewards)), 4),
                    "mitigation_rate": round(float(np.mean(all_mit)), 3),
                    "safety_violation_rate": round(float(np.mean(all_safe)), 3),
                    "avg_turns": round(float(np.mean(all_turns)), 2),
                    "filtered_zero_var": filtered_zero_var,
                    "filtered_zero_winner": filtered_zero_winner,
                }

        # Broadcast rollouts from rank 0 to all trainer ranks (GPUs 2 & 3)
        obj_box = [payload_for_ranks]
        dist.broadcast_object_list(obj_box, src=0)
        all_samples = obj_box[0]
        if not all_samples:
            if local_rank == 0:
                print(f"[rlvr] Step {step}: no informative trajectories after {max_attempts_this_step} attempts; skipping update.", flush=True)
            continue

        my_samples = [s for i, s in enumerate(all_samples) if i % world_size == local_rank]

        ddp_model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        denom = float(len(my_samples))

        for s_idx, (seq_ids, adv_np, mask_np) in enumerate(my_samples):
            input_ids = torch.tensor([seq_ids], dtype=torch.long, device=device)
            adv_t = torch.tensor(adv_np, dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                old_lp, active_pos = _compute_masked_logprobs(peft_model, input_ids, mask_t)
                with peft_model.disable_adapter():
                    ref_lp, _ = _compute_masked_logprobs(peft_model, input_ids, mask_t)
            torch.cuda.empty_cache()

            # Use `no_sync()` on all accumulation micro-steps except the last -> 1 clean AllReduce/step
            sync_ctx = ddp_model.no_sync() if s_idx < len(my_samples) - 1 else contextlib.nullcontext()
            with sync_ctx:
                new_lp, _ = _compute_masked_logprobs(ddp_model, input_ids, mask_t)
                active_adv = adv_t[active_pos]
                ratio = torch.exp(torch.clamp(new_lp - old_lp, -15.0, 15.0))
                clipped = torch.clamp(ratio, 1.0 - dapo_cfg.clip_eps_low, 1.0 + dapo_cfg.clip_eps_high)
                policy_loss = -torch.minimum(ratio * active_adv, clipped * active_adv).mean()

                log_ref_diff = torch.clamp(ref_lp - new_lp, -10.0, 10.0)
                kl_loss = args.kl_coef * (torch.exp(log_ref_diff) - log_ref_diff - 1.0).mean()

                loss = (policy_loss + kl_loss) / denom
                loss.backward()
            loss_accum += float(loss.detach().cpu())
            del loss, policy_loss, kl_loss, new_lp, old_lp, ref_lp
            torch.cuda.empty_cache()

        grad_norm = float(torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0))
        optimizer.step()

        # Hot-sync updated weights to SGLang TP2 server on GPUs 0 & 1
        sync_s = 0.0
        if local_rank == 0:
            sync_s = sync_lora_weights_to_sglang(
                peft_model, tokenizer, args.sglang_url, args.sync_dir, args.base_model_path
            )
            step_metrics["loss"] = round(loss_accum, 4)
            step_metrics["grad_norm"] = round(grad_norm, 3)
            step_metrics["sync_s"] = round(sync_s, 1)
            step_metrics["step_time_s"] = round(time.time() - t0, 1)
            print(json.dumps(step_metrics), flush=True)

            if args.save_every > 0 and step % args.save_every == 0:
                ckpt = args.output_dir / f"step-{step:03d}"
                ckpt.mkdir(parents=True, exist_ok=True)
                peft_model.save_pretrained(ckpt)

            if args.eval_every > 0 and step % args.eval_every == 0:
                _eval_heldout(f"step_{step:03d}")
        dist.barrier()

    after_eval = _eval_heldout("after_rl_sglang_final")
    if local_rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        (args.output_dir / "trial_eval_summary.json").write_text(
            json.dumps(
                {
                    "completed_rl_updates": args.steps,
                    "before_rl": before_eval,
                    "after_rl": after_eval,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
