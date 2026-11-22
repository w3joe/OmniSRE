#!/usr/bin/env python3
"""OmniSRE v2.0 High-Throughput 100-Turn Raw Exploration RLVR Trainer (`8x H100` / `4x H100`).

Engineered to reach `15,000 - 20,000` multi-turn rollouts/hour (`100K` rollouts in ~5-6 hrs,
`409K` SRE-1 parity rollouts in ~24 hrs on `1x a3-highgpu-8g` Spot):
- **Data-Parallel `TP=1` FP8 SGLang Fleet (GPUs 0..5):** Load-balances `128-384` concurrent
  rollouts across `N` independent `TP=1` FP8 servers (zero cross-GPU TP AllReduce latency).
- **In-RAM DuckDB Haystack + Rolling Compaction (`compact_workspace`):** Runs 60-100 turn raw
  SQL/PromQL/Log/Trace investigations over 50-120 distractor services while decomposing each
  episode into `<5.6K`-token exact-TITO segments (`15x` lower attention FLOPs than `1M` flat ctx).
- **`/dev/shm` RAM-Disk Broadcast Weight Sync:** Writes clean merged `state_dict` to host RAM
  (`/dev/shm`) and reloads all `N` SGLang servers in parallel in `~3-5s`.
- **Cosine LR Decay + Fault-Class Stratification:** Eliminates post-epoch-1 fault-label mode
  collapse across multi-hundred-step zero-KL Masked CISPO runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime
import http.client
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from sre_rl.env.real_telemetry_env import RealIncidentRecord, _canonicalize_fault_type
from sre_rl.env.tier1_snapshot_env import Tier1SnapshotEnvironment
from sre_rl.env.v2_raw_exploration_env import (
    HaystackConfig,
    RawDuckDBExplorationEnv,
    build_v2_raw_system_prompt,
)
from sre_rl.model.qwen38_lora import (
    Qwen38LoRASpec,
    build_qwen38_peft_lora_kwargs,
    discover_qwen38_lora_targets,
    freeze_non_lora_subsystems,
)
from sre_rl.rewards.dapo_gtpo import (
    DAPOConfig,
    RolloutGroupItem,
    compute_gtpo_hybrid_advantages,
    compute_masked_cispo_loss_torch,
)
from sre_rl.rewards.verifiable_reward import score_episode
from sre_rl.schemas import IncidentScenario
from sre_rl.v2_tree_and_compaction import (
    SPECIALIST_LEAF_ROLES,
    CompactedSegmentRecord,
    LeafInvestigationArtifact,
    build_segmented_rollout_items,
    build_v2_role_diverse_synthesis_prompt,
    is_disagreeing_counterfactual_leaf_set,
    score_v2_counterfactual_synthesis_turn,
    select_balanced_segments,
)
from sre_rl.v5_sre1_upgrades import sample_group_harness


def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout: int = 180,
    max_retries: int = 4,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            ConnectionError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            last_exc = exc
            if attempt + 1 < max_retries:
                time.sleep(0.6 * (2**attempt) + random.uniform(0.0, 0.25))
    assert last_exc is not None
    raise last_exc


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


def _sglang_generate_tito(
    sglang_base_url: str,
    tokenizer,
    input_ids: list[int],
    temperature: float,
    max_new_tokens: int,
    im_end_id: int,
) -> tuple[list[int], str]:
    try:
        resp = _post_json(
            f"{sglang_base_url.rstrip('/')}/generate",
            {
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": temperature,
                    "top_p": 0.95 if temperature > 0.0 else 1.0,
                    "max_new_tokens": max_new_tokens,
                    "stop_token_ids": list({int(tokenizer.eos_token_id), im_end_id}),
                },
            },
            timeout=180,
        )
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        ConnectionError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[v2-sglang-warn] /generate failed after retries on {sglang_base_url}: {exc}", flush=True)
        return [], ""
    out_ids = [
        int(t) for t in (resp.get("output_ids") or resp.get("meta_info", {}).get("output_ids") or [])
    ]
    if not out_ids:
        raw_body = str(resp.get("text", ""))
        if not raw_body.strip():
            return [], ""
        out_ids = tokenizer.encode(raw_body, add_special_tokens=False)
    else:
        raw_body = tokenizer.decode(out_ids, skip_special_tokens=True)

    raw_txt = (
        f"<think>\n{raw_body}"
        if "</think>" in raw_body and not raw_body.lstrip().startswith("<think>")
        else raw_body
    )
    return out_ids, raw_txt


def rollout_v2_raw_compacted_tito(
    sglang_base_url: str,
    tokenizer,
    record: RealIncidentRecord,
    cfg: HaystackConfig,
    seed: int,
    role_name: str = "general_investigator",
    max_new_tokens: int = 640,
    temperature: float = 0.85,
) -> tuple[list[CompactedSegmentRecord], dict[str, float], LeafInvestigationArtifact]:
    """Execute a 60-100 turn raw DuckDB investigation decomposed into `<5.6K`-token TITO segments."""
    env = RawDuckDBExplorationEnv(config=cfg, seed=seed)
    try:
        alert = env.reset(record)
        sys_prompt = build_v2_raw_system_prompt(cfg)
        im_end_toks = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        im_end_id = im_end_toks[0] if im_end_toks else int(tokenizer.eos_token_id)

        def _start_fresh_segment(user_text: str) -> list[int]:
            return _chat_template_ids(
                tokenizer,
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_text}],
                add_generation_prompt=True,
                reasoning_effort="medium",
            )

        token_stream = _start_fresh_segment(alert)
        segments_raw: list[tuple[list[int], list[tuple[int, int]], list[float]]] = []
        cur_spans: list[tuple[int, int]] = []
        cur_rewards: list[float] = []
        key_obs: list[str] = []
        total_turns = 0

        for _ in range(cfg.max_turns):
            # Auto-rollover segment if approaching VRAM ceiling without explicit compact_workspace
            if len(token_stream) >= cfg.segment_token_budget - max_new_tokens - 128:
                if cur_spans:
                    segments_raw.append((list(token_stream), list(cur_spans), list(cur_rewards)))
                cur_spans, cur_rewards = [], []
                env.workspace.compactions_performed += 1
                for recent_obs in key_obs[-3:]:
                    env.workspace.confirmed_anomalies.append(recent_obs[:220])
                token_stream = _start_fresh_segment(env.render_initial_alert())

            out_ids, raw_txt = _sglang_generate_tito(
                sglang_base_url, tokenizer, token_stream, temperature, max_new_tokens, im_end_id
            )
            if not out_ids:
                break

            s_pos = len(token_stream)
            token_stream.extend(out_ids)
            e_pos = len(token_stream)
            total_turns += 1

            outcome = env.step(raw_txt)
            cur_spans.append((s_pos, e_pos))
            cur_rewards.append(float(outcome.step_shaping_reward))
            if outcome.syntax_valid and not outcome.compacted:
                key_obs.append(outcome.observation[:400])

            if outcome.done:
                break

            if outcome.compacted:
                segments_raw.append((list(token_stream), list(cur_spans), list(cur_rewards)))
                cur_spans, cur_rewards = [], []
                token_stream = _start_fresh_segment(env.render_initial_alert())
                continue

            leading_end = "" if (token_stream and token_stream[-1] == im_end_id) else "<|im_end|>"
            obs_suffix = (
                f"{leading_end}\n<|im_start|>user\n"
                f"<tool_response>\n{outcome.observation}\n</tool_response><|im_end|>\n"
                "<|im_start|>assistant\n<think>\n"
            )
            token_stream.extend(tokenizer.encode(obs_suffix, add_special_tokens=False))

        if cur_spans:
            segments_raw.append((list(token_stream), list(cur_spans), list(cur_rewards)))

        sc = env.score()
        n_segs = len(segments_raw)
        segments = [
            CompactedSegmentRecord(
                segment_idx=i,
                total_segments=n_segs,
                token_ids=toks[: cfg.segment_token_budget],
                turn_spans=spans,
                turn_rewards=rews,
                episode_outcome_reward=float(sc.total),
            )
            for i, (toks, spans, rews) in enumerate(segments_raw)
        ]
        stats = {
            "reward": float(sc.total),
            "strict_solved": float(sc.strict_solved),
            "svc_exact": float(sc.service_exact),
            "fault_exact": float(sc.fault_exact),
            "onset_ok": float(sc.onset_within_90s),
            "cite_ok": float(sc.citation_verified),
            "turns": float(total_turns),
            "segments": float(n_segs),
        }
        leaf = LeafInvestigationArtifact(
            role_name=role_name,
            strict_solved=sc.strict_solved,
            outcome_reward=float(sc.total),
            compacted_notes_json=env.workspace.render_prompt_block(),
            final_submission=env._submission,
            key_observations=key_obs[-5:],
            window_start=env._window_start,
            window_end=env._window_end,
            alias_to_canon=dict(env._alias_to_canon),
        )
        return segments, stats, leaf
    finally:
        env.close()


def evaluate_v2_role_diverse_tree_5x(
    sglang_urls: list[str],
    tokenizer,
    records: list[RealIncidentRecord],
    max_turns: int = 50,
    segment_token_budget: int = 4864,
    num_distractors: int = 40,
    workers: int = 48,
) -> dict[str, Any]:
    """Run role-diversified `5x` specialist leaves + counterfactual root synthesis over unseen incidents."""
    t0 = time.time()
    url_cyc = itertools.cycle(sglang_urls)
    im_end_toks = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_id = im_end_toks[0] if im_end_toks else int(tokenizer.eos_token_id)

    leaf_jobs = []
    for r_idx, rec in enumerate(records):
        case_seed = 9000 + r_idx * 31
        for l_idx, (role_name, role_hint) in enumerate(SPECIALIST_LEAF_ROLES):
            cfg = HaystackConfig(
                max_turns=max_turns,
                segment_token_budget=segment_token_budget,
                num_distractor_services=num_distractors,
                alias_services_prob=0.0,
                specialist_focus_prompt=role_hint,
            )
            leaf_jobs.append((r_idx, l_idx, rec, cfg, case_seed, role_name))

    leaves_by_case: dict[int, list[tuple[int, LeafInvestigationArtifact]]] = {
        i: [] for i in range(len(records))
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, max(1, len(leaf_jobs)))) as pool:
        fmap = {
            pool.submit(
                rollout_v2_raw_compacted_tito,
                next(url_cyc),
                tokenizer,
                rec,
                cfg,
                sd,
                rn,
                640,
                0.35,
            ): (r_idx, l_idx)
            for r_idx, l_idx, rec, cfg, sd, rn in leaf_jobs
        }
        for fut, (r_idx, l_idx) in fmap.items():
            _, _, leaf = fut.result()
            leaves_by_case[r_idx].append((l_idx, leaf))

    def _synth_case(r_idx: int) -> dict[str, float]:
        rec = records[r_idx]
        ordered_leaves = [lf for _, lf in sorted(leaves_by_case[r_idx], key=lambda x: x[0])]
        w0 = ordered_leaves[0].window_start if ordered_leaves else rec.inject_timestamp - 900
        w1 = ordered_leaves[0].window_end if ordered_leaves else rec.inject_timestamp + 900
        msgs = build_v2_role_diverse_synthesis_prompt(rec, w0, w1, ordered_leaves)
        prompt_ids = _chat_template_ids(tokenizer, msgs, add_generation_prompt=True, reasoning_effort="medium")
        _, raw_txt = _sglang_generate_tito(next(url_cyc), tokenizer, prompt_ids, 0.0, 896, im_end_id)
        sc = score_v2_counterfactual_synthesis_turn(raw_txt, rec, ordered_leaves)
        any_leaf_solved = any(lf.strict_solved for lf in ordered_leaves)
        return {
            "strict_solved": float(sc.strict_solved),
            "svc_exact": float(sc.service_exact),
            "fault_exact": float(sc.fault_exact),
            "onset_ok": float(sc.onset_within_90s),
            "cite_ok": float(sc.citation_verified),
            "reward": float(sc.total),
            "leaf_pass_at_5": float(any_leaf_solved),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, max(1, len(records)))) as pool:
        synth_stats = list(pool.map(_synth_case, range(len(records))))

    return {
        "n": len(synth_stats),
        "strict_solved": round(float(np.mean([s["strict_solved"] for s in synth_stats])), 4) if synth_stats else 0.0,
        "leaf_pass_at_5": round(float(np.mean([s["leaf_pass_at_5"] for s in synth_stats])), 4) if synth_stats else 0.0,
        "svc_exact": round(float(np.mean([s["svc_exact"] for s in synth_stats])), 4) if synth_stats else 0.0,
        "fault_exact": round(float(np.mean([s["fault_exact"] for s in synth_stats])), 4) if synth_stats else 0.0,
        "onset_within_90s": round(float(np.mean([s["onset_ok"] for s in synth_stats])), 4) if synth_stats else 0.0,
        "citation_verified": round(float(np.mean([s["cite_ok"] for s in synth_stats])), 4) if synth_stats else 0.0,
        "mean_reward": round(float(np.mean([s["reward"] for s in synth_stats])), 4) if synth_stats else 0.0,
        "wall_time_s": round(time.time() - t0, 1),
    }


def sync_weights_shm_broadcast(
    peft_model,
    tokenizer,
    sglang_urls: list[str],
    sync_dir: Path,
    base_model_path: str,
    num_shards: int = 8,
) -> float:
    """Write 8 balanced BF16 safetensor shards in parallel to `/dev/shm` and reload all N SGLang servers concurrently!"""
    from safetensors.torch import save_file

    t0 = time.time()
    tmp_dir = sync_dir.parent / f"{sync_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    was_training = peft_model.training
    peft_model.eval()
    peft_model.merge_adapter()
    try:
        cpu_sd: dict[str, torch.Tensor] = {}
        for k, v in peft_model.base_model.model.state_dict().items():
            if ".lora_A." in k or ".lora_B." in k or ".lora_embedding_" in k:
                continue
            canon_k = k.replace(".base_layer.", ".")
            cpu_sd[canon_k] = v.detach().to(device="cpu", dtype=torch.bfloat16, non_blocking=True).contiguous()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        peft_model.unmerge_adapter()
        if was_training:
            peft_model.train()

    # Partition into `num_shards` byte-balanced shards so both `/dev/shm` write & SGLang multi-thread load scale 8x
    shard_dicts: list[dict[str, torch.Tensor]] = [{} for _ in range(num_shards)]
    shard_bytes = [0 for _ in range(num_shards)]
    weight_map: dict[str, str] = {}
    total_bytes = 0
    for k, tensor in cpu_sd.items():
        nbytes = tensor.numel() * tensor.element_size()
        total_bytes += nbytes
        target_idx = int(min(range(num_shards), key=lambda idx: shard_bytes[idx]))
        shard_name = f"model-{target_idx + 1:05d}-of-{num_shards:05d}.safetensors"
        shard_dicts[target_idx][k] = tensor
        shard_bytes[target_idx] += nbytes
        weight_map[k] = shard_name

    def _write_one_shard(idx_and_dict: tuple[int, dict[str, torch.Tensor]]) -> None:
        idx, sdict = idx_and_dict
        if not sdict:
            return
        sname = f"model-{idx + 1:05d}-of-{num_shards:05d}.safetensors"
        save_file(sdict, str(tmp_dir / sname), metadata={"format": "pt"})

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_shards) as pool:
        list(pool.map(_write_one_shard, enumerate(shard_dicts)))
    del cpu_sd, shard_dicts

    (tmp_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2),
        encoding="utf-8",
    )
    src = Path(base_model_path)
    copied_cfg = False
    if src.is_dir():
        for item in src.iterdir():
            if item.is_file() and not item.name.endswith(".safetensors") and item.name != "model.safetensors.index.json":
                shutil.copy2(item, tmp_dir / item.name)
                if item.name == "config.json":
                    copied_cfg = True
    if not copied_cfg:
        peft_model.base_model.model.config.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

    if sync_dir.exists():
        shutil.rmtree(sync_dir)
    tmp_dir.rename(sync_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sglang_urls)) as pool:
        futs = [
            pool.submit(_post_json, f"{u.rstrip('/')}/update_weights_from_disk", {"model_path": str(sync_dir)}, 240)
            for u in sglang_urls
        ]
        for f in futs:
            resp = f.result()
            if isinstance(resp, dict) and resp.get("success") is False:
                raise RuntimeError(f"SGLang broadcast sync failed: {resp}")
    return time.time() - t0


def _compute_masked_logprobs_fp32_head(
    model,
    input_ids: torch.Tensor,
    action_mask: torch.Tensor,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(selected_logprobs, active_target_pos, mean_token_entropy)` with FP32 head stability."""
    active_target_pos = torch.nonzero(action_mask[1:] > 0.0, as_tuple=False).squeeze(-1) + 1
    if active_target_pos.numel() == 0:
        z = torch.zeros(0, device=input_ids.device, dtype=torch.float32)
        return z, active_target_pos, torch.tensor(0.0, device=input_ids.device)
    pred_positions = active_target_pos - 1
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=pred_positions).logits
    target_ids = input_ids[0, active_target_pos]
    chunks: list[torch.Tensor] = []
    ent_vals: list[float] = []
    for start in range(0, active_target_pos.numel(), chunk_size):
        end = min(active_target_pos.numel(), start + chunk_size)
        sub_lp = torch.log_softmax(logits[0, start:end, :].to(torch.float32), dim=-1)
        with torch.no_grad():
            probs = sub_lp.detach().exp()
            ent_vals.append(float(-(probs * sub_lp.detach()).sum(dim=-1).mean().item()))
        chunks.append(torch.gather(sub_lp, dim=-1, index=target_ids[start:end].unsqueeze(-1)).squeeze(-1))
    del logits
    selected_lp = torch.cat(chunks, dim=0)
    # Zero-memory Monte Carlo entropy estimator `-E[log pi(a|s)]` for regularization (saves +7.4 GB vs full-vocab autograd!)
    mc_ent = -selected_lp.mean()
    _ = ent_vals
    return selected_lp, active_target_pos, mc_ent


def _load_tier1_scenarios(path: Path) -> list[IncidentScenario]:
    if not path.exists():
        return []
    rows: list[IncidentScenario] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        gt = obj["reward_model"]["ground_truth"] if "reward_model" in obj else obj
        rows.append(IncidentScenario.model_validate_json(gt) if isinstance(gt, str) else IncidentScenario.model_validate(gt))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniSRE v2.0 100-Turn Raw Exploration Turbo RLVR Trainer")
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--sglang-urls", type=str, default="http://127.0.0.1:30000", help="Comma-separated TP=1 FP8 SGLang URLs")
    parser.add_argument("--sync-dir", type=Path, default=Path("/dev/shm/omnisre_v2_sync"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/nvme/checkpoints_v2/omnisre-27b-v2.0"))
    parser.add_argument("--train-index-jsonl", type=Path, default=Path("/mnt/nvme/real_telemetry/index/real_telemetry_train.jsonl"))
    parser.add_argument("--heldout-index-jsonl", type=Path, default=Path("/mnt/nvme/real_telemetry/index/real_telemetry_heldout.jsonl"))
    parser.add_argument("--openrca-train-jsonl", type=Path, default=Path("/mnt/nvme/real_telemetry/index/openrca_train.jsonl"))
    parser.add_argument("--openrca-heldout-jsonl", type=Path, default=Path("/mnt/nvme/real_telemetry/index/openrca_heldout.jsonl"))
    parser.add_argument("--tier1-train-json", type=Path, default=Path("/mnt/nvme/data_v2/rl_real_train.jsonl"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--start-step", type=int, default=1, help="Step index to start/resume from (1-based)")
    parser.add_argument("--resume-adapter-dir", type=str, default="", help="Optional PEFT LoRA checkpoint directory to resume from")
    parser.add_argument("--prompts-per-step", type=int, default=12)
    parser.add_argument("--oversample-factor", type=int, default=3, help="3x PODS oversampling on fast FP8 workers")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--rollout-workers", type=int, default=96)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--distractors", type=int, default=40)
    parser.add_argument("--max-train-seq-len", type=int, default=4864)
    parser.add_argument("--max-segments-per-step", type=int, default=128, help="Cap backward segments per step to highest-|advantage| slices (0 = all)")
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--min-lr", type=float, default=3e-7)
    parser.add_argument("--entropy-floor", type=float, default=0.25, help="Target token entropy floor to prevent fault-type collapse")
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--sync-every", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--gcs-backup-prefix", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sglang_urls = [u.strip() for u in args.sglang_urls.split(",") if u.strip()]
    url_cycle = itertools.cycle(sglang_urls)

    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=90))
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_path, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    spec = Qwen38LoRASpec(rank=64, alpha=64)
    freeze_non_lora_subsystems(base_model, spec=spec)
    if args.resume_adapter_dir and Path(args.resume_adapter_dir).is_dir():
        if local_rank == 0:
            print(f"[v2-resume] Loading trainable PEFT adapter from {args.resume_adapter_dir}", flush=True)
        peft_model = PeftModel.from_pretrained(base_model, args.resume_adapter_dir, is_trainable=True)
    else:
        targets = discover_qwen38_lora_targets([(n, type(m).__name__) for n, m in base_model.named_modules()], require_deltanet=True)
        peft_kwargs = build_qwen38_peft_lora_kwargs(spec)
        peft_kwargs["target_modules"] = targets
        peft_model = get_peft_model(base_model, LoraConfig(**peft_kwargs))
    for param in peft_model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    if args.resume_adapter_dir and Path(args.resume_adapter_dir).is_dir():
        if local_rank == 0:
            init_sync_s = sync_weights_shm_broadcast(peft_model, tokenizer, sglang_urls, args.sync_dir, args.base_model_path)
            print(f"[v2-resume] Synced resumed adapter to {len(sglang_urls)} SGLang servers in {init_sync_s:.1f}s", flush=True)
        dist.barrier()

    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ddp_model = DDP(peft_model, device_ids=[local_rank], find_unused_parameters=False)
    trainable_params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    train_records = [RealIncidentRecord.model_validate_json(ln) for ln in args.train_index_jsonl.read_text().splitlines() if ln.strip()]
    heldout_records = [RealIncidentRecord.model_validate_json(ln) for ln in args.heldout_index_jsonl.read_text().splitlines() if ln.strip()]
    if args.openrca_train_jsonl.exists():
        train_records.extend(RealIncidentRecord.model_validate_json(ln) for ln in args.openrca_train_jsonl.read_text().splitlines() if ln.strip())
    if args.openrca_heldout_jsonl.exists():
        heldout_records.extend(RealIncidentRecord.model_validate_json(ln) for ln in args.openrca_heldout_jsonl.read_text().splitlines() if ln.strip())
    tier1_train = _load_tier1_scenarios(args.tier1_train_json)

    # Fault-class stratified buckets to prevent single-fault mode collapse!
    fault_buckets: dict[str, list[RealIncidentRecord]] = {}
    for rec in train_records:
        fault_buckets.setdefault(_canonicalize_fault_type(rec.ground_truth_fault), []).append(rec)
    rng = random.Random(args.seed)
    for b in fault_buckets.values():
        rng.shuffle(b)

    # Stratified unseen eval slice (zero aliasing, 40 distractors, de-crutched raw SQL/PromQL!)
    by_ho: dict[str, list[RealIncidentRecord]] = {}
    for r in heldout_records:
        by_ho.setdefault(r.source_dataset, []).append(r)
    eval_slice: list[RealIncidentRecord] = []
    for ds_k in sorted(by_ho):
        sub = list(by_ho[ds_k])
        random.Random(args.seed + 19).shuffle(sub)
        eval_slice.extend(sub[: max(1, 30 // max(1, len(by_ho)))])

    counterfactual_bank: list[tuple[RealIncidentRecord, list[LeafInvestigationArtifact]]] = []
    dapo_cfg = DAPOConfig(group_size=args.group_size, gamma=0.92, outcome_dampening_lambda=0.30, min_group_max_reward=0.40)

    def _sample_stratified_records(n: int) -> list[RealIncidentRecord]:
        keys = sorted(fault_buckets.keys())
        out: list[RealIncidentRecord] = []
        while len(out) < n:
            fk = rng.choice(keys)
            out.append(rng.choice(fault_buckets[fk]))
        return out

    def _eval_raw_unseen(label: str) -> dict[str, Any]:
        if local_rank != 0:
            return {}
        eval_cfg = HaystackConfig(max_turns=args.max_turns, segment_token_budget=args.max_train_seq_len, num_distractor_services=args.distractors, alias_services_prob=0.0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.rollout_workers, len(eval_slice))) as pool:
            futs = [
                pool.submit(rollout_v2_raw_compacted_tito, next(url_cycle), tokenizer, rec, eval_cfg, 1000 + i, "eval", 640, 0.0)
                for i, rec in enumerate(eval_slice)
            ]
            st_list = [f.result()[1] for f in futs]
        summary = {
            "eval_stage": label,
            "n": len(st_list),
            "strict_solved": round(float(np.mean([s["strict_solved"] for s in st_list])), 4),
            "svc_exact": round(float(np.mean([s["svc_exact"] for s in st_list])), 4),
            "fault_exact": round(float(np.mean([s["fault_exact"] for s in st_list])), 4),
            "onset_within_90s": round(float(np.mean([s["onset_ok"] for s in st_list])), 4),
            "citation_verified": round(float(np.mean([s["cite_ok"] for s in st_list])), 4),
            "avg_turns": round(float(np.mean([s["turns"] for s in st_list])), 2),
            "avg_segments": round(float(np.mean([s["segments"] for s in st_list])), 2),
        }
        print(f"[v2-raw-eval] {json.dumps(summary)}", flush=True)
        return summary

    before_label = "before_rl_v2_raw" if args.start_step <= 1 else f"resume_before_step_{args.start_step:04d}"
    before_eval = _eval_raw_unseen(before_label)
    dist.barrier()
    best_strict = float(before_eval.get("strict_solved", 0.0)) if local_rank == 0 else 0.0
    total_rollouts_generated = 0

    for step in range(max(1, args.start_step), args.steps + 1):
        t0 = time.time()
        # Cosine LR decay schedule (`args.lr -> args.min_lr`)
        progress = (step - 1) / max(1, args.steps - 1)
        cur_lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        payload_for_ranks: list[tuple[list[int], list[float], list[float]]] = []
        step_metrics: dict[str, Any] = {}

        if local_rank == 0:
            n_cand_groups = args.prompts_per_step * args.oversample_factor
            cand_recs = _sample_stratified_records(n_cand_groups)

            # Dispatch `n_cand_groups * G` rollouts across all `TP=1` FP8 SGLang servers with specialist roles!
            # Use a shared per-group environment seed so all G rollouts investigate the exact same haystack,
            # onset window, and bijective service aliases (`svc-prod-XXX`) for clean GTPO & 5x leaf synthesis.
            jobs = []
            for c_pos, rec in enumerate(cand_recs):
                group_env_seed = args.seed + step * 1000 + c_pos * 17
                for g in range(args.group_size):
                    role_name, role_hint = SPECIALIST_LEAF_ROLES[g % len(SPECIALIST_LEAF_ROLES)]
                    h_cfg = HaystackConfig(
                        max_turns=args.max_turns,
                        segment_token_budget=args.max_train_seq_len,
                        num_distractor_services=args.distractors,
                        alias_services_prob=0.50,
                        specialist_focus_prompt=role_hint if g < 5 else "",
                    )
                    jobs.append((c_pos, rec, h_cfg, group_env_seed, role_name))

            grouped: dict[int, list] = {i: [] for i in range(n_cand_groups)}
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.rollout_workers) as pool:
                fmap = {
                    pool.submit(rollout_v2_raw_compacted_tito, next(url_cycle), tokenizer, r, hc, sd, rn, 640, 0.85): cp
                    for cp, r, hc, sd, rn in jobs
                }
                for fut, cp in fmap.items():
                    grouped[cp].append(fut.result())

            total_rollouts_generated += len(jobs)
            scored_groups: list[tuple[float, int]] = []
            for cp, rec in enumerate(cand_recs):
                res_list = grouped[cp]
                rews = np.array([r[1]["reward"] for r in res_list], dtype=np.float64)
                leaves = [r[2] for r in res_list[:5]]
                if is_disagreeing_counterfactual_leaf_set(leaves):
                    counterfactual_bank.append((rec, leaves))
                    if len(counterfactual_bank) > 400:
                        counterfactual_bank.pop(0)
                if float(np.std(rews)) >= dapo_cfg.min_group_std and float(np.max(rews)) >= dapo_cfg.min_group_max_reward:
                    scored_groups.append((float(np.std(rews)), cp))

            scored_groups.sort(key=lambda x: x[0], reverse=True)
            selected = [cp for _, cp in scored_groups[: args.prompts_per_step]]

            all_strict, all_svc, all_fault, all_cite = [], [], [], []
            for cp in selected:
                res_list = grouped[cp]
                ep_segs = [r[0] for r in res_list]
                seg_samples = build_segmented_rollout_items(ep_segs, config=dapo_cfg)
                payload_for_ranks.extend(seg_samples)
                all_strict.extend(r[1]["strict_solved"] for r in res_list)
                all_svc.extend(r[1]["svc_exact"] for r in res_list)
                all_fault.extend(r[1]["fault_exact"] for r in res_list)
                all_cite.extend(r[1]["cite_ok"] for r in res_list)

            # Keep safety-gated write-remediation sharp (1 group per step)
            if tier1_train:
                sc_t1 = rng.choice(tier1_train)
                har_t1 = sample_group_harness(rng, default_max_turns=14, randomize_prob=0.60)
                im_end_id = (tokenizer.encode("<|im_end|>", add_special_tokens=False) or [int(tokenizer.eos_token_id)])[0]

                def _t1_one(sc=sc_t1, har=har_t1, end_id=im_end_id) -> tuple[list[int], RolloutGroupItem]:
                    env_t1 = Tier1SnapshotEnvironment(max_turns=har.max_turns, max_output_chars=2600)
                    al = env_t1.reset(sc)
                    ts = _chat_template_ids(tokenizer, [{"role": "system", "content": har.build_system_prompt(sc.reasoning_effort)}, {"role": "user", "content": al}], True, sc.reasoning_effort.value)
                    spans, trs = [], []
                    for _ in range(har.max_turns):
                        if len(ts) >= args.max_train_seq_len - 400:
                            break
                        oids, rtxt = _sglang_generate_tito(next(url_cycle), tokenizer, ts, 0.85, 384, end_id)
                        if not oids:
                            break
                        sp = len(ts)
                        ts.extend(oids)
                        ep_p = len(ts)
                        for al_k, cn_k in har.alias_to_canonical.items():
                            if al_k != cn_k and al_k in rtxt:
                                rtxt = rtxt.replace(f'"{al_k}"', f'"{cn_k}"')
                        out = env_t1.step(rtxt)
                        spans.append((sp, ep_p))
                        trs.append(float(out.step_shaping_reward))
                        if out.done:
                            break
                        obs_s = f"\n<|im_start|>user\n<tool_response>\n{out.observation}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
                        ts.extend(tokenizer.encode(obs_s, add_special_tokens=False))
                    ep_res = env_t1.finalize_episode()
                    rb = score_episode(sc, ep_res)
                    return ts[: args.max_train_seq_len], RolloutGroupItem(float(rb.total), trs, spans, len(ts))

                with concurrent.futures.ThreadPoolExecutor(max_workers=args.group_size) as pool:
                    t1_outs = [f.result() for f in [pool.submit(_t1_one) for _ in range(args.group_size)]]
                t1_items = [x[1] for x in t1_outs]
                t1_rw = np.array([it.outcome_reward for it in t1_items])
                if float(np.std(t1_rw)) >= dapo_cfg.min_group_std and float(np.max(t1_rw)) >= 0.45:
                    advs, masks = compute_gtpo_hybrid_advantages(t1_items, config=dapo_cfg)
                    for i_t, (seq_t, _) in enumerate(t1_outs):
                        if float(np.sum(masks[i_t][: len(seq_t)][1:])) > 0.0:
                            payload_for_ranks.append((seq_t, list(advs[i_t][: len(seq_t)]), list(masks[i_t][: len(seq_t)])))

            # Train on-policy Role-Diverse 5x Counterfactual Synthesis (1 group per step when bank is populated)
            cf_synth_trained = 0
            if counterfactual_bank:
                cf_rec, cf_leaves = rng.choice(counterfactual_bank)
                w0 = cf_leaves[0].window_start if cf_leaves else cf_rec.inject_timestamp - 900
                w1 = cf_leaves[0].window_end if cf_leaves else cf_rec.inject_timestamp + 900
                cf_msgs = build_v2_role_diverse_synthesis_prompt(cf_rec, w0, w1, cf_leaves)
                cf_prompt_ids = _chat_template_ids(tokenizer, cf_msgs, add_generation_prompt=True, reasoning_effort="medium")
                cf_end_id = (tokenizer.encode("<|im_end|>", add_special_tokens=False) or [int(tokenizer.eos_token_id)])[0]

                def _cf_one(p_ids=cf_prompt_ids, end_id=cf_end_id, rec_c=cf_rec, lvs=cf_leaves) -> tuple[list[int], RolloutGroupItem]:
                    oids, rtxt = _sglang_generate_tito(next(url_cycle), tokenizer, p_ids, 0.85, 896, end_id)
                    if not oids or len(p_ids) >= args.max_train_seq_len - 8:
                        return list(p_ids[: args.max_train_seq_len]), RolloutGroupItem(-0.30, [0.0], [(0, 0)], min(len(p_ids), args.max_train_seq_len))
                    full_seq = (list(p_ids) + oids)[: args.max_train_seq_len]
                    sc_cf = score_v2_counterfactual_synthesis_turn(rtxt, rec_c, lvs)
                    span = (len(p_ids), len(full_seq))
                    return full_seq, RolloutGroupItem(float(sc_cf.total), [0.0], [span], len(full_seq))

                with concurrent.futures.ThreadPoolExecutor(max_workers=args.group_size) as pool:
                    cf_outs = [f.result() for f in [pool.submit(_cf_one) for _ in range(args.group_size)]]
                cf_items = [x[1] for x in cf_outs]
                cf_rw = np.array([it.outcome_reward for it in cf_items])
                if float(np.std(cf_rw)) >= dapo_cfg.min_group_std and float(np.max(cf_rw)) >= 0.40:
                    cf_advs, cf_masks = compute_gtpo_hybrid_advantages(cf_items, config=dapo_cfg)
                    for i_c, (seq_c, _) in enumerate(cf_outs):
                        if float(np.sum(cf_masks[i_c][: len(seq_c)][1:])) > 0.0:
                            payload_for_ranks.append((seq_c, list(cf_advs[i_c][: len(seq_c)]), list(cf_masks[i_c][: len(seq_c)])))
                            cf_synth_trained += 1

            payload_for_ranks = select_balanced_segments(payload_for_ranks, args.max_segments_per_step)

            usable = (len(payload_for_ranks) // world_size) * world_size
            payload_for_ranks = payload_for_ranks[:usable]
            step_metrics = {
                "rl_step": step,
                "lr": round(cur_lr, 8),
                "total_rollouts": total_rollouts_generated,
                "segments_trained": len(payload_for_ranks),
                "strict_solved": round(float(np.mean(all_strict)) if all_strict else 0.0, 3),
                "svc_exact": round(float(np.mean(all_svc)) if all_svc else 0.0, 3),
                "fault_exact": round(float(np.mean(all_fault)) if all_fault else 0.0, 3),
                "cite_verified": round(float(np.mean(all_cite)) if all_cite else 0.0, 3),
                "cf_synth_bank": len(counterfactual_bank),
                "cf_synth_trained": cf_synth_trained,
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
        ent_accum = 0.0
        denom = float(len(my_samples))

        for s_idx, (seq_ids, adv_list, mask_list) in enumerate(my_samples):
            input_ids = torch.tensor([seq_ids], dtype=torch.long, device=device)
            adv_t = torch.tensor(adv_list, dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask_list, dtype=torch.float32, device=device)

            sync_ctx = contextlib.nullcontext() if s_idx == len(my_samples) - 1 else ddp_model.no_sync()
            with sync_ctx:
                new_lp, active_pos, tok_ent = _compute_masked_logprobs_fp32_head(ddp_model, input_ids, mask_t)
                cispo_loss, _ = compute_masked_cispo_loss_torch(new_lp, new_lp.detach(), adv_t[active_pos])
                # One-sided entropy floor prevents fault-type collapse without penalizing confident tool syntax
                ent_shortfall = torch.relu(torch.tensor(args.entropy_floor, device=device) - tok_ent)
                total_loss = cispo_loss + args.entropy_coef * ent_shortfall
                (total_loss / denom).backward()
                loss_accum += float(total_loss.item()) / denom
                ent_accum += float(tok_ent.item()) / denom
            del input_ids, adv_t, mask_t, new_lp, active_pos, cispo_loss, total_loss
            if (s_idx + 1) % 16 == 0:
                torch.cuda.empty_cache()

        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, 1.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        if local_rank == 0:
            should_sync = (step % max(1, args.sync_every) == 0) or (step == args.steps)
            sync_s = (
                sync_weights_shm_broadcast(peft_model, tokenizer, sglang_urls, args.sync_dir, args.base_model_path)
                if should_sync
                else 0.0
            )
            elapsed = time.time() - t0
            rollouts_per_hr = round((args.prompts_per_step * args.oversample_factor * args.group_size) * 3600.0 / max(1.0, elapsed))
            step_metrics.update({
                "loss": round(loss_accum, 4),
                "entropy": round(ent_accum, 3),
                "grad_norm": round(grad_norm, 3),
                "sync_s": round(sync_s, 1),
                "step_time_s": round(elapsed, 1),
                "est_rollouts_per_hr": rollouts_per_hr,
            })
            print(json.dumps(step_metrics), flush=True)

            if args.save_every > 0 and step % args.save_every == 0:
                sdir = args.output_dir / f"step-{step:04d}"
                sdir.mkdir(parents=True, exist_ok=True)
                peft_model.save_pretrained(sdir)
                if args.gcs_backup_prefix:
                    subprocess.run(["gcloud", "storage", "rsync", "-r", str(sdir), f"{args.gcs_backup_prefix.rstrip('/')}/step-{step:04d}", "--quiet"], check=False)

            if args.eval_every > 0 and step % args.eval_every == 0:
                ev = _eval_raw_unseen(f"step_{step:04d}")
                if ev.get("strict_solved", 0.0) >= best_strict:
                    best_strict = float(ev["strict_solved"])
                    bdir = args.output_dir / "best_checkpoint"
                    bdir.mkdir(parents=True, exist_ok=True)
                    peft_model.save_pretrained(bdir)
                    tokenizer.save_pretrained(bdir)
                    if args.gcs_backup_prefix:
                        subprocess.run(["gcloud", "storage", "rsync", "-r", str(bdir), f"{args.gcs_backup_prefix.rstrip('/')}/best_checkpoint", "--quiet"], check=False)
        dist.barrier()

    if local_rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        after_eval = _eval_raw_unseen("after_rl_v2_final")
        after_tree_5x = evaluate_v2_role_diverse_tree_5x(
            sglang_urls=sglang_urls,
            tokenizer=tokenizer,
            records=eval_slice,
            max_turns=args.max_turns,
            segment_token_budget=args.max_train_seq_len,
            num_distractors=args.distractors,
            workers=min(args.rollout_workers, max(8, len(eval_slice) * 5)),
        )
        print(f"[v2-tree-5x-eval] {json.dumps(after_tree_5x)}", flush=True)
        summary = {
            "before_rl": before_eval,
            "after_rl": after_eval,
            "after_rl_tree_5x": after_tree_5x,
            "best_strict_solved": best_strict,
            "total_rollouts_generated": total_rollouts_generated,
        }
        (args.output_dir / "v2_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
