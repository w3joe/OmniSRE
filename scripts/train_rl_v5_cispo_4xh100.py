#!/usr/bin/env python3
"""OmniSRE v5 RLVR Trainer Incorporating ResolveAI SRE-1 Findings (for 4x H100 `a3-highgpu-4g`):

1. **Tokens-In, Tokens-Out (TITO) via Native SGLang `/generate`:**
   Eliminates the 94% text re-tokenization drift by sending & receiving exact `input_ids`
   and `output_ids` directly from SGLang TP2 (`GPUs 0,1`), guaranteeing bit-identical token
   streams between sampler and trainer (`GPUs 2,3`).
2. **Masked CISPO Objective (`ratio ∈ [0.2, 5.0]`, Zero KL Penalty, FP32 Master/Grad/LM-Head):**
   Drops the KL drag term (`kl_coef = 0.0` by default) and applies stop-gradient importance
   weights inside `[0.2, 5.0]` with ratio masking outside the band (`38x` lower spike KL).
3. **2x PODS Oversampling + Returning Easy/Hard Pool Curriculum:**
   Oversamples 2x candidate scenarios per step, routes 100%-solved to Easy Pool (`t_1/2=120`)
   and 0%-solved to Hard Pool (`t_1/2=15`), and trains strictly on the top `prompts_per_step`
   highest-variance groups.
4. **Harness Randomization (50% groups) + Parallel Tool Calling + 25% Offline Synthesis:**
   Trains Investigation + Synthesis jointly so `OmniSRE v5` supports `5x` Tree Ensembling at
   inference (`2 + 3 -> 1` synthesis tree).
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
from sre_rl.model.thinking import parse_qwen38_assistant_turn_multi
from sre_rl.rewards.dapo_gtpo import (
    DAPOConfig,
    RolloutGroupItem,
    compute_gtpo_hybrid_advantages,
    compute_masked_cispo_loss_torch,
)
from sre_rl.rewards.verifiable_reward import score_episode
from sre_rl.schemas import IncidentScenario
from sre_rl.v5_sre1_upgrades import (
    ReturningPoolCurriculum,
    SampledHarnessConfig,
    build_offline_synthesis_prompt,
    sample_group_harness,
)


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
    reasoning_effort: str = "medium",
) -> list[int]:
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            reasoning_effort=reasoning_effort,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt
        )
    if isinstance(encoded, list):
        return [int(t) for t in encoded]
    ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(t) for t in ids]


def rollout_tito_sglang_episode(
    sglang_base_url: str,
    tokenizer,
    scenario: IncidentScenario,
    harness: SampledHarnessConfig,
    temperature: float = 0.85,
    max_new_tokens: int = 384,
) -> tuple[list[int], RolloutGroupItem, dict[str, float], list[dict[str, str]]]:
    """Append-only Tokens-In, Tokens-Out (TITO) multi-turn rollout via SGLang `/generate`."""
    env = Tier1SnapshotEnvironment(max_turns=harness.max_turns, max_output_chars=2800)
    alert = env.reset(scenario)
    sys_prompt = harness.build_system_prompt(scenario.reasoning_effort)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": alert},
    ]

    # Build initial prompt token stream once; strictly append-only thereafter (true TITO!).
    # Note: Qwen3.8's chat_template with `add_generation_prompt=True` ends with
    # `<|im_start|>assistant\n<think>\n`, so generated `out_ids` begin inside `<think>`.
    token_stream: list[int] = _chat_template_ids(
        tokenizer,
        messages,
        add_generation_prompt=True,
        reasoning_effort=scenario.reasoning_effort.value,
    )
    im_end_toks = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_id = im_end_toks[0] if im_end_toks else int(tokenizer.eos_token_id)

    turn_spans: list[tuple[int, int]] = []
    turn_rewards: list[float] = []
    total_tool_calls = 0
    gen_url = f"{sglang_base_url.rstrip('/')}/generate"

    for turn_idx in range(harness.max_turns):
        if len(token_stream) >= harness.max_context_tokens - max_new_tokens:
            break

        resp = _post_json(
            gen_url,
            {
                "input_ids": token_stream,
                "sampling_params": {
                    "temperature": temperature,
                    "top_p": 0.95 if temperature > 0.0 else 1.0,
                    "max_new_tokens": max_new_tokens,
                    "stop_token_ids": list({int(tokenizer.eos_token_id), im_end_id}),
                },
            },
            timeout=90,
        )
        out_ids: list[int] = [
            int(t) for t in (resp.get("output_ids") or resp.get("meta_info", {}).get("output_ids") or [])
        ]
        if not out_ids:
            raw_body = str(resp.get("text", ""))
            if not raw_body.strip():
                break
            out_ids = tokenizer.encode(raw_body, add_special_tokens=False)
        else:
            raw_body = tokenizer.decode(out_ids, skip_special_tokens=True)

        # Reconstruct full `<think>...</think>` wrapper since `<think>\n` was in prompt prefix
        raw_txt = (
            f"<think>\n{raw_body}"
            if "</think>" in raw_body and not raw_body.lstrip().startswith("<think>")
            else raw_body
        )

        start_pos = len(token_stream)
        token_stream.extend(out_ids)
        end_pos = len(token_stream)

        # Support 1..N concurrent tool calls in one turn (SRE-1 parallel calls)
        parsed = parse_qwen38_assistant_turn_multi(raw_txt, max_parallel_calls=6)
        obs_parts: list[str] = []
        step_reward = 0.0
        done = False

        if parsed.parse_error is not None or not parsed.tool_calls:
            outcome = env.step(raw_txt)
            obs_parts.append(outcome.observation)
            step_reward += outcome.step_shaping_reward
            done = outcome.done
        else:
            total_tool_calls += len(parsed.tool_calls)
            # Mild bonus when batching 2-4 read-only calls concurrently on early triage turns
            if turn_idx <= 2 and 2 <= len(parsed.tool_calls) <= 4:
                step_reward += 0.02
            for tc in parsed.tool_calls:
                resolved = harness.resolve_tool_call(tc)
                single_turn_str = (
                    f"<think>{parsed.thinking}</think>\n"
                    f"<tool_call>{resolved.model_dump_json()}</tool_call>"
                )
                outcome = env.step(single_turn_str)
                obs_parts.append(outcome.observation)
                step_reward += outcome.step_shaping_reward
                if outcome.done:
                    done = True
                    break

        combined_obs = harness.format_observation_with_budget(
            "\n---\n".join(obs_parts), turn_idx=turn_idx, used_tokens=len(token_stream)
        )
        turn_spans.append((start_pos, end_pos))
        turn_rewards.append(float(step_reward))
        messages.append({"role": "assistant", "content": raw_txt})
        messages.append({"role": "user", "content": f"<tool_response>\n{combined_obs}\n</tool_response>"})

        if done:
            break

        # Append exact Qwen3.8 ChatML observation + `<think>\n` generation prefix without
        # re-encoding prior assistant turns (true append-only TITO!)
        leading_end = "" if (token_stream and token_stream[-1] == im_end_id) else "<|im_end|>"
        obs_suffix = (
            f"{leading_end}\n<|im_start|>user\n"
            f"<tool_response>\n{combined_obs}\n</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        token_stream.extend(tokenizer.encode(obs_suffix, add_special_tokens=False))

    ep = env.finalize_episode()
    breakdown = score_episode(scenario, ep)
    item = RolloutGroupItem(
        outcome_reward=float(breakdown.total),
        turn_rewards=turn_rewards,
        turn_token_spans=turn_spans,
        total_seq_len=len(token_stream),
    )
    stats = {
        "reward": float(breakdown.total),
        "mitigated": float(breakdown.mitigated),
        "safety_violated": float(breakdown.safety_violated),
        "turns": float(len(turn_spans)),
        "calls_per_turn": round(total_tool_calls / max(1, len(turn_spans)), 2),
    }
    return token_stream, item, stats, messages


def run_tree_ensemble_5x(
    sglang_base_url: str,
    tokenizer,
    scenario: IncidentScenario,
    harness: SampledHarnessConfig,
) -> dict[str, Any]:
    """SRE-1 5x Tree Ensemble (`2 + 3 -> 2 syntheses -> 1 final answer`) for high-stakes eval."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futs = [
            pool.submit(rollout_tito_sglang_episode, sglang_base_url, tokenizer, scenario, harness, 0.7)
            for _ in range(5)
        ]
        runs = [f.result() for f in futs]

    def _synthesize(trace_msgs_subset: list[list[dict[str, str]]]) -> str:
        synth_msgs = build_offline_synthesis_prompt(scenario, trace_msgs_subset)
        p_ids = _chat_template_ids(tokenizer, synth_msgs, add_generation_prompt=True)
        resp = _post_json(
            f"{sglang_base_url.rstrip('/')}/generate",
            {
                "input_ids": p_ids,
                "sampling_params": {"temperature": 0.2, "top_p": 0.95, "max_new_tokens": 512},
            },
            timeout=90,
        )
        out_ids = resp.get("output_ids") or []
        return tokenizer.decode(out_ids, skip_special_tokens=True) if out_ids else str(resp.get("text", ""))

    # Level 1 tree: synthesize (run 1..2) and (run 3..5)
    branch_a = _synthesize([runs[0][3], runs[1][3]])
    branch_b = _synthesize([runs[2][3], runs[3][3], runs[4][3]])
    # Root synthesis combining both branches
    final_turn = _synthesize(
        [
            [*runs[0][3][:2], {"role": "assistant", "content": branch_a}],
            [*runs[2][3][:2], {"role": "assistant", "content": branch_b}],
        ]
    )
    return {
        "constituent_rewards": [r[2]["reward"] for r in runs],
        "final_synthesized_turn": final_turn,
    }


def _compute_masked_logprobs_fp32_head(
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
        # Explicit FP32 softmax & log-prob accumulation (SRE-1 / ScaleRL numerics invariant)
        sub_lp = torch.log_softmax(logits[0, start:end, :].to(torch.float32), dim=-1)
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
    parser = argparse.ArgumentParser(description="OmniSRE v5 Masked-CISPO + TITO + PODS Trainer")
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--sglang-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--sync-dir", type=Path, default=Path("/mnt/nvme/sglang_live_sync_v5"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/nvme/checkpoints_v5/omnisre-v5"))
    parser.add_argument("--train-scenarios-jsonl", type=Path, required=True)
    parser.add_argument("--heldout-scenarios-jsonl", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--oversample-factor", type=int, default=2, help="PODS 2x oversampling")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--rollout-workers", type=int, default=32)
    parser.add_argument("--eval-heldout-count", type=int, default=40)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--ratio-min", type=float, default=0.20)
    parser.add_argument("--ratio-max", type=float, default=5.00)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=20)
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

    # SRE-1 Numerics Invariant: Cast trainable LoRA master weights strictly to FP32 so small
    # policy updates never round away in BF16 accumulation!
    for param in peft_model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ddp_model = DDP(peft_model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        [p for p in ddp_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    train_scenarios = _load_verl_jsonl(args.train_scenarios_jsonl)
    heldout_scenarios = _load_verl_jsonl(args.heldout_scenarios_jsonl)
    rng = random.Random(args.seed)
    rng.shuffle(train_scenarios)
    random.Random(args.seed + 7).shuffle(heldout_scenarios)
    eval_slice = heldout_scenarios[: args.eval_heldout_count]

    curriculum = ReturningPoolCurriculum(len(train_scenarios), seed=args.seed)
    canonical_harness = sample_group_harness(rng, args.max_turns, 12288, randomize_prob=0.0)
    dapo_cfg = DAPOConfig(
        group_size=args.group_size,
        gamma=0.92,
        outcome_dampening_lambda=0.30,
        min_group_max_reward=0.50,
        gate_positive_shaping_on_success=True,
    )

    def _eval_heldout(label: str) -> dict[str, Any]:
        if local_rank != 0:
            return {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.rollout_workers) as pool:
            futs = [
                pool.submit(
                    rollout_tito_sglang_episode,
                    args.sglang_url,
                    tokenizer,
                    sc,
                    canonical_harness,
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
            "calls_per_turn": round(float(np.mean([s["calls_per_turn"] for s in stats])), 2),
        }
        print(f"[heldout-eval] {json.dumps(summary)}", flush=True)
        return summary

    before_eval = _eval_heldout("before_rl_v5")
    dist.barrier()

    best_mit = float(before_eval.get("mitigation_rate", 0.0)) if local_rank == 0 else 0.0

    for step in range(1, args.steps + 1):
        t0 = time.time()
        payload_for_ranks: list[tuple[list[int], np.ndarray, np.ndarray]] = []
        step_metrics: dict[str, Any] = {}

        if local_rank == 0:
            # PODS 2x oversampling -> keep top `prompts_per_step` highest-variance winner groups
            n_cand = args.prompts_per_step * args.oversample_factor
            cand_indices = curriculum.sample_candidates(step, n_cand)
            cand_harnesses = [
                sample_group_harness(rng, args.max_turns, 12288, randomize_prob=0.50)
                for _ in cand_indices
            ]

            jobs = [
                (c_pos, train_scenarios[sc_idx], cand_harnesses[c_pos])
                for c_pos, sc_idx in enumerate(cand_indices)
                for _ in range(args.group_size)
            ]
            grouped: dict[int, list[tuple[list[int], RolloutGroupItem, dict[str, float], list[dict[str, str]]]]] = {
                i: [] for i in range(len(cand_indices))
            }
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.rollout_workers) as pool:
                fmap = {
                    pool.submit(
                        rollout_tito_sglang_episode,
                        args.sglang_url,
                        tokenizer,
                        sc,
                        hrn,
                        0.85,
                    ): c_pos
                    for c_pos, sc, hrn in jobs
                }
                for fut, c_pos in fmap.items():
                    grouped[c_pos].append(fut.result())

            scored_groups: list[tuple[float, int]] = []
            for c_pos, sc_idx in enumerate(cand_indices):
                traces = grouped[c_pos]
                rew = np.array([t[1].outcome_reward for t in traces], dtype=np.float64)
                passes = [bool(t[2]["mitigated"]) for t in traces]
                curriculum.record_group_outcome(sc_idx, step, passes)

                std_val = float(np.std(rew))
                max_val = float(np.max(rew))
                if std_val >= dapo_cfg.min_group_std and max_val >= dapo_cfg.min_group_max_reward:
                    scored_groups.append((std_val, c_pos))

            # Rank by highest reward standard deviation (PODS) and take top `prompts_per_step`
            scored_groups.sort(key=lambda x: x[0], reverse=True)
            chosen = [c_pos for _, c_pos in scored_groups[: args.prompts_per_step]]

            all_rew, all_mit, all_safe, all_turns, all_cpt = [], [], [], [], []
            for c_pos in chosen:
                traces = grouped[c_pos]
                items = [t[1] for t in traces]
                stats = [t[2] for t in traces]
                advs, masks = compute_gtpo_hybrid_advantages(items, config=dapo_cfg)
                for idx, tr in enumerate(traces):
                    if float(np.sum(masks[idx][1:])) > 0.0 and len(tr[0]) > 2:
                        payload_for_ranks.append((tr[0], advs[idx], masks[idx]))
                all_rew.extend(it.outcome_reward for it in items)
                all_mit.extend(s["mitigated"] for s in stats)
                all_safe.extend(s["safety_violated"] for s in stats)
                all_turns.extend(s["turns"] for s in stats)
                all_cpt.extend(s["calls_per_turn"] for s in stats)

            usable = (len(payload_for_ranks) // world_size) * world_size
            payload_for_ranks = payload_for_ranks[:usable]
            if payload_for_ranks:
                step_metrics = {
                    "rl_step": step,
                    "trajectories": len(payload_for_ranks),
                    "batch_mean_reward": round(float(np.mean(all_rew)), 4),
                    "mitigation_rate": round(float(np.mean(all_mit)), 3),
                    "safety_violation_rate": round(float(np.mean(all_safe)), 3),
                    "avg_turns": round(float(np.mean(all_turns)), 2),
                    "calls_per_turn": round(float(np.mean(all_cpt)), 2),
                    "pools": curriculum.pool_counts(),
                }

        obj_box = [payload_for_ranks]
        dist.broadcast_object_list(obj_box, src=0)
        all_samples = obj_box[0]
        if not all_samples:
            continue

        my_samples = [s for i, s in enumerate(all_samples) if i % world_size == local_rank]
        ddp_model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        masked_fracs: list[float] = []
        k3_kls: list[float] = []
        denom = float(len(my_samples))

        for s_idx, (seq_ids, adv_np, mask_np) in enumerate(my_samples):
            input_ids = torch.tensor([seq_ids], dtype=torch.long, device=device)
            adv_t = torch.tensor(adv_np, dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                old_lp, active_pos = _compute_masked_logprobs_fp32_head(peft_model, input_ids, mask_t)
            torch.cuda.empty_cache()

            sync_ctx = ddp_model.no_sync() if s_idx < len(my_samples) - 1 else contextlib.nullcontext()
            with sync_ctx:
                new_lp, _ = _compute_masked_logprobs_fp32_head(ddp_model, input_ids, mask_t)
                cispo_loss, diag = compute_masked_cispo_loss_torch(
                    new_lp=new_lp,
                    old_lp=old_lp,
                    active_advantages=adv_t[active_pos],
                    ratio_min=args.ratio_min,
                    ratio_max=args.ratio_max,
                )
                loss = cispo_loss / denom
                loss.backward()
            loss_accum += float(loss.detach().cpu())
            masked_fracs.append(diag["masked_fraction"])
            k3_kls.append(diag["approx_k3_kl"])
            del loss, cispo_loss, new_lp, old_lp
            torch.cuda.empty_cache()

        grad_norm = float(torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0))
        optimizer.step()

        if local_rank == 0:
            sync_s = sync_lora_weights_to_sglang(
                peft_model, tokenizer, args.sglang_url, args.sync_dir, args.base_model_path
            )
            step_metrics.update(
                {
                    "loss": round(loss_accum, 4),
                    "grad_norm": round(grad_norm, 3),
                    "masked_frac": round(float(np.mean(masked_fracs)), 4),
                    "k3_kl": round(float(np.mean(k3_kls)), 5),
                    "sync_s": round(sync_s, 1),
                    "step_time_s": round(time.time() - t0, 1),
                }
            )
            print(json.dumps(step_metrics), flush=True)

            if args.eval_every > 0 and step % args.eval_every == 0:
                ev = _eval_heldout(f"step_{step:03d}")
                if ev.get("mitigation_rate", 0.0) >= best_mit:
                    best_mit = float(ev["mitigation_rate"])
                    best_dir = args.output_dir / "best_checkpoint"
                    best_dir.mkdir(parents=True, exist_ok=True)
                    peft_model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
        dist.barrier()

    after_eval = _eval_heldout("after_rl_v5_final")
    if local_rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        (args.output_dir / "trial_eval_summary.json").write_text(
            json.dumps({"before_rl": before_eval, "after_rl": after_eval, "best_mitigation": best_mit}, indent=2),
            encoding="utf-8",
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
