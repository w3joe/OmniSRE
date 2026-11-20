#!/usr/bin/env python3
"""Standalone 4x H100 Multi-Turn DAPO + GTPO RLVR Trainer for `Qwen/Qwen3.8-27B` (Dense).

Run-v3 Anti-Stalling & Batched Multi-Prompt RLVR Architecture:
- Shuffles real-world grounded RLVR scenarios across all topologies/alert families.
- Generates `rollouts_per_gpu` rollouts **in parallel per GPU** via left-padded batched
  `.generate()` (`G = 4 * rollouts_per_gpu = 8` rollouts per incident scenario).
- Applies **DAPO Positive-Winner + Zero-Variance Filtering** (`std >= 1e-4` AND
  `max_reward >= min_group_max_reward`) so zero-winner groups can NEVER reinforce
  read-only timeout stalling over remediation attempts.
- Computes **Completion-Gated GTPO Hybrid Turn + Outcome Advantages** strictly over
  assistant `<think>` + `<tool_call>` tokens (`action_mask = 0` on env observations).
- Accumulates gradients across `prompts_per_step` (default `4` -> `32` trajectories per
  optimizer update) with 512-token chunked 248K-vocab log-probs (~0.5 GB peak logit VRAM)
  and zero-extra-VRAM reference KL via `model.disable_adapter()`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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


def _compute_masked_logprobs_with_logits_to_keep(
    model,
    input_ids: torch.Tensor,
    action_mask: torch.Tensor,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log-probs ONLY at active assistant token positions using `logits_to_keep`.

    Avoids materializing `[1, seq_len, 248320]` logits across 8K-token `<tool_response>`
    context, saving >10 GB of H100 VRAM per GPU on 14-turn Qwen3.8-27B trajectories.

    Returns `(active_logprobs, active_token_positions)` where `active_token_positions`
    are the target indices `t >= 1` with `action_mask[t] > 0`.
    """
    active_target_pos = torch.nonzero(action_mask[1:] > 0.0, as_tuple=False).squeeze(-1) + 1
    if active_target_pos.numel() == 0:
        empty = torch.zeros(0, device=input_ids.device, dtype=torch.float32)
        return empty, active_target_pos

    # Predict token at position `t` using hidden state at `t - 1`
    pred_positions = active_target_pos - 1
    logits = model(
        input_ids=input_ids,
        use_cache=False,
        logits_to_keep=pred_positions,
    ).logits  # shape: [1, num_active, vocab_size]

    num_active = active_target_pos.numel()
    target_ids = input_ids[0, active_target_pos]
    chunks: list[torch.Tensor] = []
    for start in range(0, num_active, chunk_size):
        end = min(num_active, start + chunk_size)
        sub_logits = logits[0, start:end, :].float()
        sub_labels = target_ids[start:end].unsqueeze(-1)
        sub_lp = torch.log_softmax(sub_logits, dim=-1)
        chunks.append(torch.gather(sub_lp, dim=-1, index=sub_labels).squeeze(-1))
    del logits
    return torch.cat(chunks, dim=0), active_target_pos


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
def rollout_batch_multi_turn(
    unwrapped_model,
    tokenizer,
    scenarios: list[IncidentScenario],
    device: torch.device,
    max_turns: int = 12,
    max_new_tokens_per_turn: int = 256,
    max_seq_len: int = 7168,
    temperature: float = 0.85,
) -> list[tuple[list[int], RolloutGroupItem, dict[str, float]]]:
    """Run `len(scenarios)` multi-turn SRE episodes concurrently on one GPU via left-padded batching."""
    unwrapped_model.eval()
    bsz = len(scenarios)
    envs = [Tier1SnapshotEnvironment(max_turns=max_turns, max_output_chars=2800) for _ in range(bsz)]
    messages_list: list[list[dict[str, str]]] = []
    full_ids_list: list[list[int]] = []
    turn_spans_list: list[list[tuple[int, int]]] = [[] for _ in range(bsz)]
    turn_rewards_list: list[list[float]] = [[] for _ in range(bsz)]
    done_flags = [False] * bsz

    for i, sc in enumerate(scenarios):
        alert = envs[i].reset(sc)
        msgs = [
            {"role": "system", "content": build_sre_system_prompt(sc.reasoning_effort)},
            {"role": "user", "content": alert},
        ]
        messages_list.append(msgs)
        full_ids_list.append(_chat_template_ids(tokenizer, msgs, add_generation_prompt=False))

    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    eos_id = int(tokenizer.eos_token_id)

    for _ in range(max_turns):
        active_indices: list[int] = []
        active_prompts: list[list[int]] = []
        for i in range(bsz):
            if done_flags[i]:
                continue
            fmt_msgs = format_multiturn_preserve_thinking(
                messages_list[i], preserve_thinking=True, keep_last_k_thoughts=max_turns
            )
            p_ids = _chat_template_ids(tokenizer, fmt_msgs, add_generation_prompt=True)
            if len(p_ids) >= max_seq_len - max_new_tokens_per_turn:
                done_flags[i] = True
                continue
            active_indices.append(i)
            active_prompts.append(p_ids)

        if not active_indices:
            break

        # Left-pad active prompts for batched autoregressive decode
        max_p = max(len(p) for p in active_prompts)
        padded_inp = torch.full((len(active_indices), max_p), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((len(active_indices), max_p), dtype=torch.long, device=device)
        for row_i, p_ids in enumerate(active_prompts):
            offset = max_p - len(p_ids)
            padded_inp[row_i, offset:] = torch.tensor(p_ids, dtype=torch.long, device=device)
            attn_mask[row_i, offset:] = 1

        gen_out = unwrapped_model.generate(
            input_ids=padded_inp,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens_per_turn,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            top_p=0.95 if temperature > 0.0 else None,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            use_cache=True,
        )

        for row_i, idx in enumerate(active_indices):
            raw_gen = gen_out[row_i, max_p:].tolist()
            # Trim any trailing pad/eos beyond the first EOS token
            gen_tokens: list[int] = []
            for tok in raw_gen:
                if tok == pad_id and gen_tokens:
                    break
                gen_tokens.append(int(tok))
                if tok == eos_id:
                    break

            if not gen_tokens:
                done_flags[idx] = True
                continue

            assistant_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            outcome = envs[idx].step(assistant_text)

            p_ids = active_prompts[row_i]
            start_pos = len(p_ids)
            full_ids_list[idx] = p_ids + gen_tokens
            end_pos = len(full_ids_list[idx])
            turn_spans_list[idx].append((start_pos, end_pos))
            turn_rewards_list[idx].append(float(outcome.step_shaping_reward))

            messages_list[idx].append({"role": "assistant", "content": assistant_text})
            messages_list[idx].append(
                {
                    "role": "user",
                    "content": f"<tool_response>\n{outcome.observation}\n</tool_response>",
                }
            )
            if outcome.done:
                done_flags[idx] = True

    results: list[tuple[list[int], RolloutGroupItem, dict[str, float]]] = []
    for i, sc in enumerate(scenarios):
        ep = envs[i].finalize_episode()
        breakdown = score_episode(sc, ep)
        item = RolloutGroupItem(
            outcome_reward=float(breakdown.total),
            turn_rewards=turn_rewards_list[i],
            turn_token_spans=turn_spans_list[i],
            total_seq_len=len(full_ids_list[i]),
        )
        stats = {
            "reward": float(breakdown.total),
            "mitigated": float(breakdown.mitigated),
            "safety_violated": float(breakdown.safety_violated),
            "turns": float(len(turn_spans_list[i])),
        }
        results.append((full_ids_list[i], item, stats))
    return results


def _load_verl_jsonl_scenarios(path: Path) -> list[IncidentScenario]:
    loaded: list[IncidentScenario] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        gt = row["reward_model"]["ground_truth"]
        sc = (
            IncidentScenario.model_validate_json(gt)
            if isinstance(gt, str)
            else IncidentScenario.model_validate(gt)
        )
        loaded.append(sc)
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="4x H100 Batched Multi-Turn DAPO + GTPO RLVR Trainer")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3.8-27B")
    parser.add_argument(
        "--sft-adapter-dir",
        type=Path,
        default=Path("/mnt/nvme/checkpoints_v2/sft-lora64"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/nvme/checkpoints_v3/rl-dapo-gtpo"),
    )
    parser.add_argument("--train-scenarios-jsonl", type=Path, default=None)
    parser.add_argument("--heldout-scenarios-jsonl", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--max-scenario-attempts", type=int, default=160)
    parser.add_argument("--eval-heldout-count", type=int, default=24)
    parser.add_argument("--rollouts-per-gpu", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=14)
    parser.add_argument("--min-group-max-reward", type=float, default=0.50)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--kl-coef", type=float, default=0.04)
    parser.add_argument("--save-every", type=int, default=5)
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
            print(
                f"[rlvr] Merging SFT adapter {args.sft_adapter_dir} into base weights so "
                "disable_adapter() anchors KL strictly to pi_SFT (not raw base model)!",
                flush=True,
            )
        sft_peft = PeftModel.from_pretrained(base_model, str(args.sft_adapter_dir))
        base_model = sft_peft.merge_and_unload()
        del sft_peft
        torch.cuda.empty_cache()

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

    if args.train_scenarios_jsonl is not None and args.train_scenarios_jsonl.exists():
        scenarios = _load_verl_jsonl_scenarios(args.train_scenarios_jsonl)
        if args.heldout_scenarios_jsonl is not None and args.heldout_scenarios_jsonl.exists():
            heldout_pool = _load_verl_jsonl_scenarios(args.heldout_scenarios_jsonl)
        else:
            heldout_pool = list(scenarios[-args.eval_heldout_count :])
    else:
        splits = split_scenarios_decontaminated(
            generate_scenario_suite(num_scenarios=400, seed=args.seed)
        )
        scenarios = splits.train
        heldout_pool = splits.eval_heldout

    # CRITICAL FIX #1: Deterministically shuffle BOTH train and held-out pools across all ranks
    # so training never sees consecutive single-component alerts (e.g. 8x OVN in a row).
    random.Random(args.seed).shuffle(scenarios)
    random.Random(args.seed + 7).shuffle(heldout_pool)
    heldout_eval_scenarios = heldout_pool[: args.eval_heldout_count]

    if local_rank == 0:
        print(
            f"[rlvr] Shuffled {len(scenarios)} train & {len(heldout_eval_scenarios)} held-out scenarios "
            f"(max_turns={args.max_turns}, prompts_per_step={args.prompts_per_step}, "
            f"min_group_max_reward={args.min_group_max_reward})",
            flush=True,
        )

    def _run_distributed_heldout_eval(label: str) -> dict[str, float]:
        my_scenarios = [
            sc for idx, sc in enumerate(heldout_eval_scenarios) if idx % world_size == local_rank
        ]
        local_eval_stats: list[dict[str, float]] = []
        # Evaluate in mini-batches of 2 per GPU concurrently for ~2x eval speedup
        for start in range(0, len(my_scenarios), 2):
            batch_sc = my_scenarios[start : start + 2]
            batch_out = rollout_batch_multi_turn(
                unwrapped_model=peft_model,
                tokenizer=tokenizer,
                scenarios=batch_sc,
                device=device,
                max_turns=args.max_turns,
                temperature=0.0,
            )
            local_eval_stats.extend(st for _, _, st in batch_out)

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
        min_group_max_reward=args.min_group_max_reward,
        gate_positive_shaping_on_success=True,
    )

    before_eval = _run_distributed_heldout_eval("before_rl_sft_checkpoint")

    completed_updates = 0
    filtered_zero_var = 0
    filtered_zero_winner = 0
    scenario_cursor = 0

    while completed_updates < args.steps and scenario_cursor < args.max_scenario_attempts:
        step_t0 = time.time()
        ddp_model.train()
        optimizer.zero_grad()

        accepted_prompts = 0
        step_loss_accum = 0.0
        step_rewards_all: list[float] = []
        step_mitigations_all: list[float] = []
        step_safety_all: list[float] = []
        step_turns_all: list[float] = []
        step_scenarios_used: list[str] = []

        while (
            accepted_prompts < args.prompts_per_step
            and scenario_cursor < args.max_scenario_attempts
        ):
            scenario = scenarios[scenario_cursor % len(scenarios)]
            scenario_cursor += 1

            # Batched parallel rollout generation on each GPU (G = 4 * rollouts_per_gpu = 8)
            local_traces = rollout_batch_multi_turn(
                unwrapped_model=peft_model,
                tokenizer=tokenizer,
                scenarios=[scenario] * args.rollouts_per_gpu,
                device=device,
                max_turns=args.max_turns,
                temperature=0.85,
            )

            gathered_items: list[list[RolloutGroupItem]] = [None] * world_size  # type: ignore[list-item]
            gathered_stats: list[list[dict[str, float]]] = [None] * world_size  # type: ignore[list-item]
            dist.all_gather_object(gathered_items, [t[1] for t in local_traces])
            dist.all_gather_object(gathered_stats, [t[2] for t in local_traces])

            flat_group_items = [item for rank_items in gathered_items for item in rank_items]
            flat_stats = [st for rank_stats in gathered_stats for st in rank_stats]
            group_rewards = np.array([it.outcome_reward for it in flat_group_items], dtype=np.float64)
            group_std = float(np.std(group_rewards))
            group_max = float(np.max(group_rewards))

            # CRITICAL FIX #2: DAPO Zero-Variance AND Positive-Winner Filtering
            if group_std < dapo_cfg.min_group_std:
                filtered_zero_var += 1
                if local_rank == 0:
                    print(
                        f"[dapo-filter] Skipped zero-variance group on {scenario.scenario_id} "
                        f"(reward={group_rewards[0]:.3f})",
                        flush=True,
                    )
                continue

            if group_max < dapo_cfg.min_group_max_reward:
                filtered_zero_winner += 1
                if local_rank == 0:
                    print(
                        f"[dapo-filter] Skipped zero-winner group on {scenario.scenario_id} "
                        f"(max_reward={group_max:.3f} < {dapo_cfg.min_group_max_reward:.2f}) — preventing stall reinforcement",
                        flush=True,
                    )
                continue

            # CRITICAL FIX #3: Completion-Gated GTPO Hybrid Advantages
            all_advs, all_masks = compute_gtpo_hybrid_advantages(flat_group_items, config=dapo_cfg)
            my_offset = local_rank * args.rollouts_per_gpu
            denom = float(args.prompts_per_step * args.rollouts_per_gpu)

            ddp_model.train()
            for r_idx in range(args.rollouts_per_gpu):
                seq_ids_list = local_traces[r_idx][0]
                adv_np = all_advs[my_offset + r_idx]
                mask_np = all_masks[my_offset + r_idx]

                input_ids = torch.tensor([seq_ids_list], dtype=torch.long, device=device)
                adv_t = torch.tensor(adv_np, dtype=torch.float32, device=device)
                mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)

                with torch.no_grad():
                    old_lp, active_pos = _compute_masked_logprobs_with_logits_to_keep(
                        peft_model, input_ids, mask_t
                    )
                    if active_pos.numel() == 0:
                        continue
                    with peft_model.disable_adapter():
                        ref_lp, _ = _compute_masked_logprobs_with_logits_to_keep(
                            peft_model, input_ids, mask_t
                        )
                torch.cuda.empty_cache()

                new_lp, _ = _compute_masked_logprobs_with_logits_to_keep(
                    ddp_model, input_ids, mask_t
                )
                active_adv = adv_t[active_pos]

                log_ratio = new_lp - old_lp
                ratio = torch.exp(torch.clamp(log_ratio, -15.0, 15.0))
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - dapo_cfg.clip_eps_low, 1.0 + dapo_cfg.clip_eps_high
                )
                surr = torch.minimum(ratio * active_adv, clipped_ratio * active_adv)
                policy_loss = -surr.mean()

                # Low-variance Schulman KL estimator anchored to Stage 1 SFT policy
                log_ref_diff = torch.clamp(ref_lp - new_lp, -10.0, 10.0)
                kl_per_tok = torch.exp(log_ref_diff) - log_ref_diff - 1.0
                kl_loss = args.kl_coef * kl_per_tok.mean()

                loss = (policy_loss + kl_loss) / denom
                loss.backward()
                step_loss_accum += float(loss.detach().cpu())
                del loss, policy_loss, kl_loss, new_lp, old_lp, ref_lp
                torch.cuda.empty_cache()

            accepted_prompts += 1
            step_rewards_all.extend(group_rewards.tolist())
            step_mitigations_all.extend(s["mitigated"] for s in flat_stats)
            step_safety_all.extend(s["safety_violated"] for s in flat_stats)
            step_turns_all.extend(s["turns"] for s in flat_stats)
            step_scenarios_used.append(scenario.scenario_id)

        if accepted_prompts == 0:
            break

        grad_norm = float(torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0))
        optimizer.step()
        completed_updates += 1

        if local_rank == 0:
            dt = time.time() - step_t0
            print(
                json.dumps(
                    {
                        "rl_step": completed_updates,
                        "prompts_in_batch": accepted_prompts,
                        "loss": round(step_loss_accum, 4),
                        "grad_norm": round(grad_norm, 3),
                        "batch_mean_reward": round(float(np.mean(step_rewards_all)), 4),
                        "batch_max_reward": round(float(np.max(step_rewards_all)), 4),
                        "mitigation_rate": round(float(np.mean(step_mitigations_all)), 3),
                        "safety_violation_rate": round(float(np.mean(step_safety_all)), 3),
                        "avg_turns": round(float(np.mean(step_turns_all)), 2),
                        "filtered_zero_var": filtered_zero_var,
                        "filtered_zero_winner": filtered_zero_winner,
                        "step_time_s": round(dt, 1),
                    }
                ),
                flush=True,
            )
            if args.save_every > 0 and completed_updates % args.save_every == 0:
                ckpt_dir = args.output_dir / f"step-{completed_updates:03d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                peft_model.save_pretrained(ckpt_dir)

    after_eval = _run_distributed_heldout_eval("after_rl_dapo_gtpo_v3")

    if local_rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        (args.output_dir / "trial_eval_summary.json").write_text(
            json.dumps(
                {
                    "completed_rl_updates": completed_updates,
                    "prompts_per_step": args.prompts_per_step,
                    "max_turns": args.max_turns,
                    "filtered_zero_var_groups": filtered_zero_var,
                    "filtered_zero_winner_groups": filtered_zero_winner,
                    "before_rl": before_eval,
                    "after_rl": after_eval,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[rlvr] Saved Run-v3 post-trained Qwen3.8-27B SRE LoRA adapter to {args.output_dir}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
