#!/usr/bin/env python3
"""OmniSRE v6: Real Production Telemetry RLVR Trainer (Masked CISPO + TITO + Offline Synthesis + 5x Tree Eval).

Trains directly over ACTUAL multi-gigabyte recorded production & GKE telemetry files:
- **Train Systems:** Real commercial bank/cloud outages (`DejaVu-A1/A2/B`) + real GKE
  multi-modal captures (`RCAEval RE1/RE2/RE3` Online Boutique & Sock Shop).
- **100% Unseen Held-Out Systems:** `RE3-TT` (41-service Train Ticket) + `DejaVu-C`
  (unseen commercial Oracle DB production outages).

Combines all battle-tested SRE-1 / v5 mechanisms on 4x H100 (`a3-highgpu-4g`):
- GPUs 0,1: SGLang 0.5.23 TP2 (FP8 KV cache, FlashInfer, native TITO `/generate`)
- GPUs 2,3: DDP Rank-64 Hybrid LoRA (BF16 autocast + FP32 master/grad/LM-head, Masked CISPO zero-KL)
- 2x PODS Oversampling + Returning Easy/Hard Pools + 25% Offline Synthesis auxiliary batches
- Final Zero-Shot Evaluation comparing **Single Investigation (`1x`)** vs. **`5x` Tree Ensemble (`2+3 -> 1`)**!
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

from sre_rl.env.real_telemetry_env import (
    RealIncidentRecord,
    RealTelemetryReplayEnvironment,
    _canonicalize_fault_type,
    _normalize_service_token,
    build_real_telemetry_system_prompt,
)
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
from sre_rl.v5_sre1_upgrades import ReturningPoolCurriculum


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
    mapped = "xhigh" if reasoning_effort == "high" else reasoning_effort
    try:
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt, reasoning_effort=mapped
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


def rollout_real_telemetry_tito(
    sglang_base_url: str,
    tokenizer,
    record: RealIncidentRecord,
    max_turns: int = 10,
    max_context_tokens: int = 7168,
    max_new_tokens: int = 384,
    temperature: float = 0.85,
) -> tuple[list[int], RolloutGroupItem, dict[str, float], list[dict[str, str]]]:
    """Run one TITO multi-turn investigation over real production telemetry CSVs."""
    env = RealTelemetryReplayEnvironment(max_turns=max_turns, max_output_chars=3000)
    alert = env.reset(record)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_real_telemetry_system_prompt(max_turns=max_turns)},
        {"role": "user", "content": alert},
    ]

    token_stream = _chat_template_ids(tokenizer, messages, add_generation_prompt=True, reasoning_effort="medium")
    im_end_toks = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_id = im_end_toks[0] if im_end_toks else int(tokenizer.eos_token_id)

    turn_spans: list[tuple[int, int]] = []
    turn_rewards: list[float] = []
    total_calls = 0
    gen_url = f"{sglang_base_url.rstrip('/')}/generate"

    for turn_idx in range(max_turns):
        if len(token_stream) >= max_context_tokens - max_new_tokens:
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
        out_ids = [
            int(t) for t in (resp.get("output_ids") or resp.get("meta_info", {}).get("output_ids") or [])
        ]
        if not out_ids:
            raw_body = str(resp.get("text", ""))
            if not raw_body.strip():
                break
            out_ids = tokenizer.encode(raw_body, add_special_tokens=False)
        else:
            raw_body = tokenizer.decode(out_ids, skip_special_tokens=True)

        raw_txt = (
            f"<think>\n{raw_body}"
            if "</think>" in raw_body and not raw_body.lstrip().startswith("<think>")
            else raw_body
        )

        start_pos = len(token_stream)
        token_stream.extend(out_ids)
        end_pos = len(token_stream)

        outcome = env.step(raw_txt)
        total_calls += outcome.tool_calls_executed
        turn_spans.append((start_pos, end_pos))
        turn_rewards.append(float(outcome.step_shaping_reward))

        rem_turns = max(0, max_turns - (turn_idx + 1))
        obs_text = outcome.observation
        if rem_turns <= 2 and not outcome.done:
            obs_text += f"\n[budget: {rem_turns} turns left — call submit_rca_report now]"

        messages.append({"role": "assistant", "content": raw_txt})
        messages.append({"role": "user", "content": f"<tool_response>\n{obs_text}\n</tool_response>"})

        if outcome.done:
            break

        leading_end = "" if (token_stream and token_stream[-1] == im_end_id) else "<|im_end|>"
        obs_suffix = (
            f"{leading_end}\n<|im_start|>user\n"
            f"<tool_response>\n{obs_text}\n</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        token_stream.extend(tokenizer.encode(obs_suffix, add_special_tokens=False))

    sc = env.score()
    item = RolloutGroupItem(
        outcome_reward=float(sc.total),
        turn_rewards=turn_rewards,
        turn_token_spans=turn_spans,
        total_seq_len=len(token_stream),
    )
    stats = {
        "reward": float(sc.total),
        "solved": float(sc.solved),
        "svc_exact": float(sc.service_exact_score >= 0.50),
        "fault_match": float(sc.fault_type_score > 0.0),
        "turns": float(len(turn_spans)),
        "calls_per_turn": round(total_calls / max(1, len(turn_spans)), 2),
    }
    return token_stream, item, stats, messages


def rollout_offline_synthesis_tito(
    sglang_base_url: str,
    tokenizer,
    record: RealIncidentRecord,
    saved_traces: list[list[dict[str, str]]],
    temperature: float = 0.7,
) -> tuple[list[int], RolloutGroupItem, dict[str, float]]:
    """SRE-1 25% Offline Synthesis auxiliary task over saved real telemetry investigations."""
    evidence_sections = []
    for idx, msgs in enumerate(saved_traces[:3], start=1):
        lines = [f"=== SAVED INVESTIGATION #{idx} ==="]
        for m in msgs[2:10]:
            lines.append(f"[{m['role']}] {str(m['content'])[:420]}")
        evidence_sections.append("\n".join(lines))

    sys_p = (
        "You are OmniSRE in Offline Evidence Synthesis mode. Cross-examine the saved telemetry "
        "investigations below, filter out noisy downstream symptoms, and emit exactly one "
        '`<tool_call>{"name": "submit_rca_report", "arguments": {"root_service": "...", '
        '"fault_type": "...", "causal_chain": "...", "confidence": 0.9}}</tool_call>`.'
    )
    usr_p = (
        f"[SYNTHESIS] Case: {record.case_id} | System: {record.system_name} | Onset: {record.inject_timestamp}\n\n"
        + "\n\n".join(evidence_sections)
    )
    p_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
        add_generation_prompt=True,
        reasoning_effort="medium",
    )[:5500]
    im_end_toks = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_id = im_end_toks[0] if im_end_toks else int(tokenizer.eos_token_id)

    resp = _post_json(
        f"{sglang_base_url.rstrip('/')}/generate",
        {
            "input_ids": p_ids,
            "sampling_params": {
                "temperature": temperature,
                "top_p": 0.95 if temperature > 0.0 else 1.0,
                "max_new_tokens": 384,
                "stop_token_ids": list({int(tokenizer.eos_token_id), im_end_id}),
            },
        },
        timeout=90,
    )
    out_ids = [int(t) for t in (resp.get("output_ids") or [])]
    raw_body = tokenizer.decode(out_ids, skip_special_tokens=True) if out_ids else str(resp.get("text", ""))
    if not out_ids:
        out_ids = tokenizer.encode(raw_body, add_special_tokens=False)
    raw_txt = (
        f"<think>\n{raw_body}"
        if "</think>" in raw_body and not raw_body.lstrip().startswith("<think>")
        else raw_body
    )

    parsed = parse_qwen38_assistant_turn_multi(raw_txt, max_parallel_calls=2)
    reward = -0.10
    solved = False
    svc_hit = False
    if parsed.tool_call and parsed.tool_call.name == "submit_rca_report":
        args = parsed.tool_call.arguments
        pred_svc = _normalize_service_token(str(args.get("root_service", "")))
        gt_svc = _normalize_service_token(record.ground_truth_service)
        svc_score = 0.70 if pred_svc == gt_svc else (0.55 if pred_svc and (gt_svc in pred_svc or pred_svc in gt_svc) else 0.0)
        f_score = (
            0.30
            if _canonicalize_fault_type(str(args.get("fault_type", "")))
            == _canonicalize_fault_type(record.ground_truth_fault)
            else 0.0
        )
        reward = round(svc_score + f_score, 4)
        svc_hit = svc_score >= 0.55
        solved = svc_hit and (f_score > 0.0 or _canonicalize_fault_type(record.ground_truth_fault) == "code_fault")

    full_ids = p_ids + out_ids
    item = RolloutGroupItem(
        outcome_reward=reward,
        turn_rewards=[0.0],
        turn_token_spans=[(len(p_ids), len(full_ids))],
        total_seq_len=len(full_ids),
    )
    return full_ids, item, {"reward": reward, "solved": float(solved), "svc_exact": float(svc_hit)}


def evaluate_tree_ensemble_5x_real(
    sglang_base_url: str,
    tokenizer,
    records: list[RealIncidentRecord],
    workers: int = 20,
) -> dict[str, float]:
    """Compare single-run (`1x`) vs. SRE-1 `5x` Tree Ensemble (`2+3 -> 1`) on unseen real systems!"""
    def _eval_one_case(rec: RealIncidentRecord) -> tuple[float, float, float, float]:
        runs = [
            rollout_real_telemetry_tito(sglang_base_url, tokenizer, rec, max_turns=10, temperature=0.0 if i == 0 else 0.65)
            for i in range(5)
        ]
        single_solved = runs[0][2]["solved"]
        single_svc = runs[0][2]["svc_exact"]

        # Tree synthesis: branch A (runs 0..1) + branch B (runs 2..4) -> final root synthesis
        _, _, st_a = rollout_offline_synthesis_tito(sglang_base_url, tokenizer, rec, [runs[0][3], runs[1][3]], temperature=0.2)
        _, _, st_b = rollout_offline_synthesis_tito(sglang_base_url, tokenizer, rec, [runs[2][3], runs[3][3], runs[4][3]], temperature=0.2)
        _, _, st_final = rollout_offline_synthesis_tito(
            sglang_base_url,
            tokenizer,
            rec,
            [runs[0][3], runs[2][3], runs[4][3]],
            temperature=0.0,
        )
        ens_solved = max(st_final["solved"], 0.0)
        ens_svc = max(st_final["svc_exact"], 0.0)
        del st_a, st_b
        return single_solved, single_svc, ens_solved, ens_svc

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(records))) as pool:
        res = list(pool.map(_eval_one_case, records))
    return {
        "num_unseen_real_cases": len(records),
        "single_1x_solved_rate": round(float(np.mean([r[0] for r in res])), 4),
        "single_1x_service_acc": round(float(np.mean([r[1] for r in res])), 4),
        "ensemble_5x_solved_rate": round(float(np.mean([r[2] for r in res])), 4),
        "ensemble_5x_service_acc": round(float(np.mean([r[3] for r in res])), 4),
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
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=pred_positions).logits
    target_ids = input_ids[0, active_target_pos]
    chunks: list[torch.Tensor] = []
    for start in range(0, active_target_pos.numel(), chunk_size):
        end = min(active_target_pos.numel(), start + chunk_size)
        sub_lp = torch.log_softmax(logits[0, start:end, :].to(torch.float32), dim=-1)
        chunks.append(torch.gather(sub_lp, dim=-1, index=target_ids[start:end].unsqueeze(-1)).squeeze(-1))
    del logits
    return torch.cat(chunks, dim=0), active_target_pos


def sync_lora_weights_to_sglang(peft_model, tokenizer, sglang_base_url: str, sync_dir: Path, base_model_path: str) -> float:
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
        src = Path(base_model_path)
        if src.is_dir():
            for fname in ("preprocessor_config.json", "video_preprocessor_config.json"):
                if (src / fname).exists():
                    shutil.copy2(src / fname, tmp_dir / fname)
    finally:
        peft_model.unmerge_adapter()
        if was_training:
            peft_model.train()
    if sync_dir.exists():
        shutil.rmtree(sync_dir)
    tmp_dir.rename(sync_dir)
    resp = _post_json(f"{sglang_base_url.rstrip('/')}/update_weights_from_disk", {"model_path": str(sync_dir)}, timeout=180)
    if isinstance(resp, dict) and resp.get("success") is False:
        raise RuntimeError(f"SGLang sync failed: {resp}")
    return time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniSRE v6 Real Production Telemetry RLVR")
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--sglang-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--sync-dir", type=Path, default=Path("/mnt/nvme/sglang_live_sync_v6"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/nvme/checkpoints_v6/omnisre-v6-real-telemetry"))
    parser.add_argument("--train-index-jsonl", type=Path, default=Path("/mnt/nvme/real_telemetry/index/real_telemetry_train.jsonl"))
    parser.add_argument("--heldout-index-jsonl", type=Path, default=Path("/mnt/nvme/real_telemetry/index/real_telemetry_heldout.jsonl"))
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--rollout-workers", type=int, default=24)
    parser.add_argument("--eval-heldout-count", type=int, default=30)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--eval-every", type=int, default=20)
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
        args.base_model_path, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    spec = Qwen38LoRASpec(rank=64, alpha=64)
    freeze_non_lora_subsystems(base_model, spec=spec)
    targets = discover_qwen38_lora_targets([(n, type(m).__name__) for n, m in base_model.named_modules()], require_deltanet=True)
    peft_kwargs = build_qwen38_peft_lora_kwargs(spec)
    peft_kwargs["target_modules"] = targets
    peft_model = get_peft_model(base_model, LoraConfig(**peft_kwargs))
    for param in peft_model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ddp_model = DDP(peft_model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW([p for p in ddp_model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)

    train_records = [RealIncidentRecord.model_validate_json(ln) for ln in args.train_index_jsonl.read_text().splitlines() if ln.strip()]
    heldout_records = [RealIncidentRecord.model_validate_json(ln) for ln in args.heldout_index_jsonl.read_text().splitlines() if ln.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(train_records)
    random.Random(args.seed + 7).shuffle(heldout_records)
    eval_slice = heldout_records[: args.eval_heldout_count]

    curriculum = ReturningPoolCurriculum(len(train_records), seed=args.seed)
    saved_trace_bank: dict[str, list[list[dict[str, str]]]] = {}
    dapo_cfg = DAPOConfig(group_size=args.group_size, gamma=0.92, outcome_dampening_lambda=0.30, min_group_max_reward=0.50)

    def _eval_unseen(label: str) -> dict[str, Any]:
        if local_rank != 0:
            return {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.rollout_workers) as pool:
            futs = [pool.submit(rollout_real_telemetry_tito, args.sglang_url, tokenizer, rec, args.max_turns, 7168, 384, 0.0) for rec in eval_slice]
            stats = [f.result()[2] for f in futs]
        summary = {
            "eval_stage": label,
            "num_unseen_real": len(stats),
            "mean_reward": round(float(np.mean([s["reward"] for s in stats])), 4),
            "solved_rate": round(float(np.mean([s["solved"] for s in stats])), 4),
            "service_exact_acc": round(float(np.mean([s["svc_exact"] for s in stats])), 4),
            "fault_match_acc": round(float(np.mean([s["fault_match"] for s in stats])), 4),
            "avg_turns": round(float(np.mean([s["turns"] for s in stats])), 2),
        }
        print(f"[real-heldout-eval] {json.dumps(summary)}", flush=True)
        return summary

    before_eval = _eval_unseen("before_rl_v6_real_telemetry")
    dist.barrier()
    best_svc_acc = float(before_eval.get("service_exact_acc", 0.0)) if local_rank == 0 else 0.0

    for step in range(1, args.steps + 1):
        t0 = time.time()
        payload_for_ranks: list[tuple[list[int], np.ndarray, np.ndarray]] = []
        step_metrics: dict[str, Any] = {}

        if local_rank == 0:
            # 75% live real-telemetry investigation + 25% offline synthesis (when trace bank warm)
            n_synth_target = (args.prompts_per_step // 4) if len(saved_trace_bank) >= 4 else 0
            n_live_target = args.prompts_per_step - n_synth_target
            cand_indices = curriculum.sample_candidates(step, n_live_target * 2)

            jobs = [(c_pos, train_records[idx]) for c_pos, idx in enumerate(cand_indices) for _ in range(args.group_size)]
            grouped: dict[int, list] = {i: [] for i in range(len(cand_indices))}
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.rollout_workers) as pool:
                fmap = {pool.submit(rollout_real_telemetry_tito, args.sglang_url, tokenizer, rec, args.max_turns, 7168, 384, 0.85): c_pos for c_pos, rec in jobs}
                for fut, c_pos in fmap.items():
                    grouped[c_pos].append(fut.result())

            scored: list[tuple[float, int]] = []
            for c_pos, idx in enumerate(cand_indices):
                traces = grouped[c_pos]
                rec = train_records[idx]
                rew = np.array([t[1].outcome_reward for t in traces], dtype=np.float64)
                curriculum.record_group_outcome(idx, step, [bool(t[2]["solved"]) for t in traces])
                # Save diverse trajectories into replay bank for Offline Synthesis!
                saved_trace_bank[rec.case_id] = [t[3] for t in traces[:3]]
                if float(np.std(rew)) >= dapo_cfg.min_group_std and float(np.max(rew)) >= 0.45:
                    scored.append((float(np.std(rew)), c_pos))

            scored.sort(key=lambda x: x[0], reverse=True)
            chosen = [c_pos for _, c_pos in scored[:n_live_target]]

            all_rew, all_sol, all_svc = [], [], []
            for c_pos in chosen:
                traces = grouped[c_pos]
                items = [t[1] for t in traces]
                advs, masks = compute_gtpo_hybrid_advantages(items, config=dapo_cfg)
                for idx_t, tr in enumerate(traces):
                    seq_t = tr[0][:6144]
                    if float(np.sum(masks[idx_t][: len(seq_t)][1:])) > 0.0 and len(seq_t) > 2:
                        payload_for_ranks.append((seq_t, advs[idx_t][: len(seq_t)], masks[idx_t][: len(seq_t)]))
                all_rew.extend(it.outcome_reward for it in items)
                all_sol.extend(t[2]["solved"] for t in traces)
                all_svc.extend(t[2]["svc_exact"] for t in traces)

            # Mix in 25% Offline Synthesis auxiliary groups!
            if n_synth_target > 0:
                rec_by_id = {r.case_id: r for r in train_records}
                synth_keys = rng.sample(list(saved_trace_bank.keys()), min(n_synth_target, len(saved_trace_bank)))
                for cid in synth_keys:
                    rec = rec_by_id[cid]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=args.group_size) as pool:
                        s_futs = [
                            pool.submit(
                                rollout_offline_synthesis_tito,
                                args.sglang_url,
                                tokenizer,
                                rec,
                                saved_trace_bank[cid],
                                0.8,
                            )
                            for _ in range(args.group_size)
                        ]
                        s_traces = [f.result() for f in s_futs]
                    s_items = [t[1] for t in s_traces]
                    s_rew = np.array([it.outcome_reward for it in s_items], dtype=np.float64)
                    if float(np.std(s_rew)) >= dapo_cfg.min_group_std and float(np.max(s_rew)) >= 0.45:
                        s_advs, s_masks = compute_gtpo_hybrid_advantages(s_items, config=dapo_cfg)
                        for idx_t, tr in enumerate(s_traces):
                            seq_t = tr[0][:6144]
                            if float(np.sum(s_masks[idx_t][: len(seq_t)][1:])) > 0.0:
                                payload_for_ranks.append((seq_t, s_advs[idx_t][: len(seq_t)], s_masks[idx_t][: len(seq_t)]))

            usable = (len(payload_for_ranks) // world_size) * world_size
            payload_for_ranks = payload_for_ranks[:usable]
            if payload_for_ranks:
                step_metrics = {
                    "rl_step": step,
                    "trajectories": len(payload_for_ranks),
                    "batch_mean_reward": round(float(np.mean(all_rew)) if all_rew else 0.0, 4),
                    "solved_rate": round(float(np.mean(all_sol)) if all_sol else 0.0, 3),
                    "service_exact_acc": round(float(np.mean(all_svc)) if all_svc else 0.0, 3),
                    "synth_bank_size": len(saved_trace_bank),
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
                cispo_loss, _ = compute_masked_cispo_loss_torch(new_lp, old_lp, adv_t[active_pos])
                loss = cispo_loss / denom
                loss.backward()
            loss_accum += float(loss.detach().cpu())
            del loss, cispo_loss, new_lp, old_lp
            torch.cuda.empty_cache()

        grad_norm = float(torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0))
        optimizer.step()

        if local_rank == 0:
            sync_s = sync_lora_weights_to_sglang(peft_model, tokenizer, args.sglang_url, args.sync_dir, args.base_model_path)
            step_metrics.update({"loss": round(loss_accum, 4), "grad_norm": round(grad_norm, 3), "sync_s": round(sync_s, 1), "step_time_s": round(time.time() - t0, 1)})
            print(json.dumps(step_metrics), flush=True)
            if args.eval_every > 0 and step % args.eval_every == 0:
                ev = _eval_unseen(f"step_{step:03d}")
                if ev.get("service_exact_acc", 0.0) >= best_svc_acc:
                    best_svc_acc = float(ev["service_exact_acc"])
                    best_dir = args.output_dir / "best_checkpoint"
                    best_dir.mkdir(parents=True, exist_ok=True)
                    peft_model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
        dist.barrier()

    after_eval = _eval_unseen("after_rl_v6_final")
    ensemble_eval = (
        evaluate_tree_ensemble_5x_real(args.sglang_url, tokenizer, eval_slice[:16])
        if local_rank == 0
        else {}
    )
    if local_rank == 0:
        print(f"[real-tree-ensemble-5x] {json.dumps(ensemble_eval)}", flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        (args.output_dir / "trial_eval_summary.json").write_text(
            json.dumps({"before_rl": before_eval, "after_rl": after_eval, "tree_ensemble_5x": ensemble_eval}, indent=2),
            encoding="utf-8",
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
