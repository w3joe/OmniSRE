#!/usr/bin/env python3
"""Standalone 4x H100 Multi-Turn DAPO + GTPO RLVR Trainer for `Qwen/Qwen3.8-27B` (Dense).

Runs interactive multi-turn SRE rollouts across 4x H100 GPUs in parallel:
- Each rank generates `rollouts_per_gpu` (default 2 -> `G = 4 * 2 = 8` group rollouts)
  against `Tier1SnapshotEnvironment` with `preserve_thinking`.
- Scores episodes via deterministic `score_episode` (Detection + Localization + RCA +
  Dual State/SLO Mitigation - Safety/Overthinking Penalties).
- Applies **DAPO Dynamic Zero-Variance Group Filtering** (`std < 1e-4` skipped).
- Computes **GTPO Hybrid Turn + Outcome Advantages** strictly over assistant `<think>`
  and `<tool_call>` tokens (`action_mask = 0` on system & `<tool_response>` tokens).
- Computes reference policy log-probs with **zero extra VRAM** via `model.disable_adapter()`.
- Updates Rank-64 Hybrid (Gated DeltaNet + Gated Attention + SwiGLU) LoRA weights using
  DAPO asymmetric clipping (`[1 - 0.20, 1 + 0.28]`) + low-variance KL regularization.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from sre_rl.data.scenario_factory import generate_scenario_suite
from sre_rl.data.splits import split_scenarios_decontaminated
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


def _compute_Selected_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log-probabilities aligned with `input_ids` (`shape: [seq_len]`).

    Position `t` stores `log P(input_ids[t] | input_ids[:t])` (`0.0` at `t=0`).
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:].unsqueeze(-1)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_lp = torch.gather(log_probs, dim=-1, index=shift_labels).squeeze(-1).squeeze(0)
    pad_zero = torch.zeros(1, device=token_lp.device, dtype=token_lp.dtype)
    return torch.cat([pad_zero, token_lp], dim=0)


def _chat_template_ids(
    tokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> list[int]:
    """Handle both transformers v4 (`list[int]`) and v5 (`BatchEncoding`) return types."""
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(encoded, list):
        return [int(t) for t in encoded]
    if hasattr(encoded, "input_ids"):
        ids = encoded.input_ids
    elif isinstance(encoded, dict) and "input_ids" in encoded:
        ids = encoded["input_ids"]
    else:
        raise TypeError(f"Unexpected apply_chat_template output type: {type(encoded)}")
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(t) for t in ids]


@torch.no_grad()
def rollout_multi_turn_episode(
    unwrapped_model,
    tokenizer,
    scenario: IncidentScenario,
    device: torch.device,
    max_turns: int = 8,
    max_new_tokens_per_turn: int = 256,
    max_seq_len: int = 8192,
    temperature: float = 0.9,
) -> tuple[list[int], RolloutGroupItem, dict[str, float]]:
    """Run one interactive multi-turn SRE episode and track exact assistant token spans."""
    unwrapped_model.eval()
    env = Tier1SnapshotEnvironment(max_turns=max_turns, max_output_chars=3000)
    initial_alert = env.reset(scenario)

    messages = [
        {"role": "system", "content": build_sre_system_prompt(scenario.reasoning_effort)},
        {"role": "user", "content": initial_alert},
    ]

    full_ids: list[int] = _chat_template_ids(
        tokenizer, messages, add_generation_prompt=False
    )
    turn_spans: list[tuple[int, int]] = []
    turn_rewards: list[float] = []

    for _ in range(max_turns):
        formatted_msgs = format_multiturn_preserve_thinking(messages, preserve_thinking=True)
        prompt_ids: list[int] = _chat_template_ids(
            tokenizer, formatted_msgs, add_generation_prompt=True
        )
        if len(prompt_ids) >= max_seq_len - max_new_tokens_per_turn:
            break

        inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(inp)
        gen_out = unwrapped_model.generate(
            input_ids=inp,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens_per_turn,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            top_p=0.95 if temperature > 0.0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        gen_tokens = gen_out[0, len(prompt_ids) :].tolist()
        if not gen_tokens:
            break

        assistant_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        outcome = env.step(assistant_text)

        # Track exact assistant token span inside concatenated rollout sequence
        start_idx = len(prompt_ids)
        full_ids = prompt_ids + gen_tokens
        end_idx = len(full_ids)
        turn_spans.append((start_idx, end_idx))
        turn_rewards.append(float(outcome.step_shaping_reward))

        messages.append({"role": "assistant", "content": assistant_text})
        messages.append(
            {"role": "user", "content": f"<tool_response>\n{outcome.observation}\n</tool_response>"}
        )
        if outcome.done:
            break

    ep = env.finalize_episode()
    breakdown = score_episode(scenario, ep)
    group_item = RolloutGroupItem(
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
    return full_ids, group_item, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="4x H100 Multi-Turn DAPO + GTPO RLVR Trainer")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3.8-27B")
    parser.add_argument(
        "--sft-adapter-dir",
        type=Path,
        default=Path("/mnt/nvme/checkpoints/qwen38-27b-sre-sft-lora64"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/nvme/checkpoints/qwen38-27b-sre-rl-dapo-gtpo"),
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-scenario-attempts", type=int, default=60)
    parser.add_argument("--eval-heldout-count", type=int, default=12)
    parser.add_argument("--rollouts-per-gpu", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + local_rank * 1000)

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)

    spec = Qwen38LoRASpec(rank=64, alpha=64)
    freeze_non_lora_subsystems(base_model, spec=spec)

    if args.sft_adapter_dir.exists() and (args.sft_adapter_dir / "adapter_config.json").exists():
        if local_rank == 0:
            print(f"[rlvr] Initializing policy from SFT LoRA adapter: {args.sft_adapter_dir}")
        peft_model = PeftModel.from_pretrained(
            base_model, str(args.sft_adapter_dir), is_trainable=True
        )
    else:
        module_pairs = [(n, type(m).__name__) for n, m in base_model.named_modules()]
        targets = discover_qwen38_lora_targets(module_pairs, require_deltanet=True)
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

    splits = split_scenarios_decontaminated(
        generate_scenario_suite(num_scenarios=400, seed=args.seed)
    )
    scenarios = splits.train
    heldout_eval_scenarios = splits.eval_heldout[: args.eval_heldout_count]

    def _run_distributed_heldout_eval(label: str) -> dict[str, float]:
        local_eval_stats = []
        for idx, sc in enumerate(heldout_eval_scenarios):
            if idx % world_size != local_rank:
                continue
            _, _, st = rollout_multi_turn_episode(
                unwrapped_model=peft_model,
                tokenizer=tokenizer,
                scenario=sc,
                device=device,
                max_turns=args.max_turns,
                temperature=0.0,
            )
            local_eval_stats.append(st)
        all_eval_lists: list[list[dict[str, float]]] = [None] * world_size  # type: ignore[list-item]
        dist.all_gather_object(all_eval_lists, local_eval_stats)
        flat = [x for sub in all_eval_lists for x in sub]
        summary = {
            "eval_stage": label,
            "num_heldout": len(flat),
            "mean_reward": round(float(np.mean([x["reward"] for x in flat])), 4),
            "mitigation_rate": round(float(np.mean([x["mitigated"] for x in flat])), 4),
            "safety_violation_rate": round(float(np.mean([x["safety_violated"] for x in flat])), 4),
            "avg_turns": round(float(np.mean([x["turns"] for x in flat])), 2),
        }
        if local_rank == 0:
            print(f"[heldout-eval] {json.dumps(summary)}", flush=True)
        return summary

    dapo_cfg = DAPOConfig(
        group_size=world_size * args.rollouts_per_gpu,
        clip_eps_low=0.20,
        clip_eps_high=0.28,
        gamma=0.92,
        outcome_dampening_lambda=0.30,
    )

    before_eval = _run_distributed_heldout_eval("before_rl_sft_checkpoint")

    completed_updates = 0
    filtered_zero_var = 0
    scenario_cursor = 0

    while completed_updates < args.steps and scenario_cursor < args.max_scenario_attempts:
        t0 = time.time()
        scenario = scenarios[scenario_cursor % len(scenarios)]
        scenario_cursor += 1

        # 1. Parallel Group Rollout across all 4x H100 GPUs (G = world_size * rollouts_per_gpu)
        local_traces: list[tuple[list[int], RolloutGroupItem, dict[str, float]]] = []
        for _ in range(args.rollouts_per_gpu):
            trace = rollout_multi_turn_episode(
                unwrapped_model=peft_model,
                tokenizer=tokenizer,
                scenario=scenario,
                device=device,
                max_turns=args.max_turns,
            )
            local_traces.append(trace)

        gathered_items: list[list[RolloutGroupItem]] = [None] * world_size  # type: ignore[list-item]
        gathered_stats: list[list[dict[str, float]]] = [None] * world_size  # type: ignore[list-item]
        dist.all_gather_object(gathered_items, [t[1] for t in local_traces])
        dist.all_gather_object(gathered_stats, [t[2] for t in local_traces])

        flat_group_items = [item for rank_items in gathered_items for item in rank_items]
        flat_stats = [st for rank_stats in gathered_stats for st in rank_stats]
        group_rewards = np.array([it.outcome_reward for it in flat_group_items], dtype=np.float64)

        # 2. DAPO Dynamic Zero-Variance Group Filtering
        if float(np.std(group_rewards)) < dapo_cfg.min_group_std:
            filtered_zero_var += 1
            if local_rank == 0:
                print(
                    f"[dapo-filter] Skipped zero-variance group on {scenario.scenario_id} "
                    f"(uniform reward={group_rewards[0]:.3f}, total_filtered={filtered_zero_var})"
                )
            continue

        # 3. GTPO Hybrid Turn + Outcome Advantage Calculation
        all_advs, all_masks = compute_gtpo_hybrid_advantages(flat_group_items, config=dapo_cfg)
        my_offset = local_rank * args.rollouts_per_gpu

        # 4. Compute Reference & Old Log-Probs + DAPO Asymmetric Clipped Policy Gradient Update
        ddp_model.train()
        optimizer.zero_grad()
        total_loss_val = 0.0

        for r_idx in range(args.rollouts_per_gpu):
            seq_ids_list = local_traces[r_idx][0]
            adv_np = all_advs[my_offset + r_idx]
            mask_np = all_masks[my_offset + r_idx]

            input_ids = torch.tensor([seq_ids_list], dtype=torch.long, device=device)
            adv_t = torch.tensor(adv_np, dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)
            active_count = torch.clamp(mask_t.sum(), min=1.0)

            with torch.no_grad():
                old_logits = peft_model(input_ids=input_ids).logits
                old_lp = _compute_Selected_logprobs(old_logits, input_ids)
                with peft_model.disable_adapter():
                    ref_logits = peft_model(input_ids=input_ids).logits
                    ref_lp = _compute_Selected_logprobs(ref_logits, input_ids)

            new_logits = ddp_model(input_ids=input_ids).logits
            new_lp = _compute_Selected_logprobs(new_logits, input_ids)

            log_ratio = (new_lp - old_lp) * mask_t
            ratio = torch.exp(torch.clamp(log_ratio, -15.0, 15.0))
            clipped_ratio = torch.clamp(
                ratio, 1.0 - dapo_cfg.clip_eps_low, 1.0 + dapo_cfg.clip_eps_high
            )
            surr = torch.minimum(ratio * adv_t, clipped_ratio * adv_t) * mask_t
            policy_loss = -surr.sum() / active_count

            # Low-variance Schulman KL estimator: exp(ref - new) - (ref - new) - 1
            log_ref_diff = torch.clamp((ref_lp - new_lp) * mask_t, -10.0, 10.0)
            kl_per_tok = (torch.exp(log_ref_diff) - log_ref_diff - 1.0) * mask_t
            kl_loss = args.kl_coef * (kl_per_tok.sum() / active_count)

            loss = (policy_loss + kl_loss) / args.rollouts_per_gpu
            loss.backward()
            total_loss_val += float(loss.detach().cpu())

        grad_norm = float(torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0))
        optimizer.step()
        completed_updates += 1

        if local_rank == 0:
            mean_r = float(np.mean(group_rewards))
            max_r = float(np.max(group_rewards))
            mit_rate = float(np.mean([s["mitigated"] for s in flat_stats]))
            avg_turns = float(np.mean([s["turns"] for s in flat_stats]))
            dt = time.time() - t0
            print(
                json.dumps(
                    {
                        "rl_step": completed_updates,
                        "scenario": scenario.scenario_id,
                        "loss": round(total_loss_val, 4),
                        "grad_norm": round(grad_norm, 3),
                        "group_mean_reward": round(mean_r, 4),
                        "group_max_reward": round(max_r, 4),
                        "group_std_reward": round(float(np.std(group_rewards)), 4),
                        "mitigation_rate": round(mit_rate, 3),
                        "avg_turns": round(avg_turns, 2),
                        "filtered_groups": filtered_zero_var,
                        "step_time_s": round(dt, 1),
                    }
                ),
                flush=True,
            )

    after_eval = _run_distributed_heldout_eval("after_rl_dapo_gtpo")

    if local_rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        (args.output_dir / "trial_eval_summary.json").write_text(
            json.dumps(
                {
                    "completed_rl_updates": completed_updates,
                    "filtered_zero_var_groups": filtered_zero_var,
                    "before_rl": before_eval,
                    "after_rl": after_eval,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[rlvr] Saved post-trained Qwen3.8-27B SRE LoRA adapter to {args.output_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
