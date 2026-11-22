#!/usr/bin/env python3
"""Full Apples-to-Apples Benchmark Suite for `OmniSRE v2.0` (`1x` Greedy + Role-Diverse `5x` Tree).

Evaluates `OmniSRE v2.0` head-to-head against `OmniSRE v1.0` or Base `Qwen/Qwen3.8-27B` across:
1. **Unseen De-Crutched Raw Exploration (`1x` Greedy, `50` turns, `40` distractors):**
   - `OpenRCA` held-out (`Telecom` + `Market/cloudbed-2`)
   - `DejaVu-C` (`99` commercial Oracle DB wait-event outages)
   - `RCAEval RE3-TT` (`30` 41-service Train Ticket GKE microservice failures)
2. **Role-Diversified `5x` Specialist Tree + Counterfactual Synthesis (`evaluate_v2_role_diverse_tree_5x`):**
   - Evaluates `leaf_pass_at_5` and synthesized `strict_solved`, `svc_exact`, `fault_exact`, `onset_within_90s`, and `citation_verified`.
3. **Unseen Policy-Gated Kubernetes Write Remediation (`60 x 3 = 180` rollouts):**
   - Verifies zero regression on `Pass@1`, `Pass^3`, `Pass@3`, and `0.0%` greedy safety violations.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from sre_rl.env.real_telemetry_env import RealIncidentRecord
from sre_rl.env.v2_raw_exploration_env import HaystackConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_v1_full_suite import eval_tier1_write
from train_rl_v2_turbo_8xh100 import (
    evaluate_v2_role_diverse_tree_5x,
    rollout_v2_raw_compacted_tito,
)


def eval_v2_raw_1x(
    sglang_urls: list[str],
    tokenizer,
    records: list[RealIncidentRecord],
    max_turns: int = 50,
    segment_token_budget: int = 4864,
    num_distractors: int = 40,
    workers: int = 48,
) -> dict[str, Any]:
    t0 = time.time()
    url_cyc = itertools.cycle(sglang_urls)
    cfg = HaystackConfig(
        max_turns=max_turns,
        segment_token_budget=segment_token_budget,
        num_distractor_services=num_distractors,
        alias_services_prob=0.0,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, max(1, len(records)))) as pool:
        futs = [
            (
                rec.source_dataset,
                pool.submit(
                    rollout_v2_raw_compacted_tito,
                    next(url_cyc),
                    tokenizer,
                    rec,
                    cfg,
                    5000 + idx,
                    "eval_1x",
                    640,
                    0.0,
                ),
            )
            for idx, rec in enumerate(records)
        ]
        rows = [(ds, f.result()[1]) for ds, f in futs]

    by_ds: dict[str, dict[str, Any]] = {}
    for ds in sorted({d for d, _ in rows}):
        sub = [st for d, st in rows if d == ds]
        by_ds[ds] = {
            "count": len(sub),
            "strict_solved": round(float(np.mean([s["strict_solved"] for s in sub])), 4),
            "svc_exact": round(float(np.mean([s["svc_exact"] for s in sub])), 4),
            "fault_exact": round(float(np.mean([s["fault_exact"] for s in sub])), 4),
            "onset_within_90s": round(float(np.mean([s["onset_ok"] for s in sub])), 4),
            "citation_verified": round(float(np.mean([s["cite_ok"] for s in sub])), 4),
            "avg_turns": round(float(np.mean([s["turns"] for s in sub])), 2),
            "avg_segments": round(float(np.mean([s["segments"] for s in sub])), 2),
        }

    all_st = [st for _, st in rows]
    return {
        "num_cases": len(all_st),
        "strict_solved": round(float(np.mean([s["strict_solved"] for s in all_st])), 4) if all_st else 0.0,
        "svc_exact": round(float(np.mean([s["svc_exact"] for s in all_st])), 4) if all_st else 0.0,
        "fault_exact": round(float(np.mean([s["fault_exact"] for s in all_st])), 4) if all_st else 0.0,
        "onset_within_90s": round(float(np.mean([s["onset_ok"] for s in all_st])), 4) if all_st else 0.0,
        "citation_verified": round(float(np.mean([s["cite_ok"] for s in all_st])), 4) if all_st else 0.0,
        "mean_reward": round(float(np.mean([s["reward"] for s in all_st])), 4) if all_st else 0.0,
        "avg_turns": round(float(np.mean([s["turns"] for s in all_st])), 2) if all_st else 0.0,
        "avg_segments": round(float(np.mean([s["segments"] for s in all_st])), 2) if all_st else 0.0,
        "wall_time_s": round(time.time() - t0, 1),
        "by_dataset": by_ds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniSRE v2.0 Full Raw Exploration & 5x Tree Benchmark Suite")
    parser.add_argument("--v2-urls", type=str, required=True, help="Comma-separated SGLang URLs for OmniSRE v2.0")
    parser.add_argument("--baseline-urls", type=str, default="", help="Optional comma-separated SGLang URLs for baseline")
    parser.add_argument("--tokenizer-path", type=str, default="/mnt/nvme/omnisre-27b-v2.0-merged")
    parser.add_argument("--heldout-real-jsonl", type=Path, default=Path("/mnt/nvme/real_telemetry/index/real_telemetry_heldout.jsonl"))
    parser.add_argument("--openrca-heldout-jsonl", type=Path, default=Path("/mnt/nvme/real_telemetry/index/openrca_heldout.jsonl"))
    parser.add_argument("--heldout-tier1-jsonl", type=Path, default=Path("/mnt/nvme/data_v2/rl_real_heldout.jsonl"))
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--distractors", type=int, default=40)
    parser.add_argument("--tree-5x-count", type=int, default=45)
    parser.add_argument("--tier1-limit", type=int, default=60)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--out-json", type=Path, default=Path("/mnt/nvme/v2_full_benchmark_results.json"))
    args = parser.parse_args()

    v2_urls = [u.strip() for u in args.v2_urls.split(",") if u.strip()]
    base_urls = [u.strip() for u in args.baseline_urls.split(",") if u.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    heldout_records: list[RealIncidentRecord] = []
    if args.heldout_real_jsonl.exists():
        heldout_records.extend(
            RealIncidentRecord.model_validate_json(ln)
            for ln in args.heldout_real_jsonl.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        )
    if args.openrca_heldout_jsonl.exists():
        heldout_records.extend(
            RealIncidentRecord.model_validate_json(ln)
            for ln in args.openrca_heldout_jsonl.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        )

    by_ds: dict[str, list[RealIncidentRecord]] = {}
    for rec in heldout_records:
        by_ds.setdefault(rec.source_dataset, []).append(rec)
    tree_slice: list[RealIncidentRecord] = []
    per_ds = max(1, args.tree_5x_count // max(1, len(by_ds)))
    for ds_name in sorted(by_ds):
        bucket = list(by_ds[ds_name])
        random.Random(42).shuffle(bucket)
        tree_slice.extend(bucket[:per_ds])

    print(f"[v2-bench] Evaluating {len(heldout_records)} unseen raw incidents (1x Greedy, parallel v2 vs v1)...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as top_pool:
        fut_v2_1x = top_pool.submit(
            eval_v2_raw_1x, v2_urls, tokenizer, heldout_records, args.max_turns, 4864, args.distractors, args.workers
        )
        fut_base_1x = (
            top_pool.submit(
                eval_v2_raw_1x, base_urls, tokenizer, heldout_records, args.max_turns, 4864, args.distractors, args.workers
            )
            if base_urls
            else None
        )
        v2_raw_1x = fut_v2_1x.result()
        base_raw_1x = fut_base_1x.result() if fut_base_1x else None
    print("[v2-bench] OmniSRE v2.0 Raw 1x:", json.dumps(v2_raw_1x), flush=True)
    if base_raw_1x:
        print("[v2-bench] Baseline (v1.0) Raw 1x:", json.dumps(base_raw_1x), flush=True)

    print(f"[v2-bench] Evaluating Role-Diverse 5x Specialist Tree on {len(tree_slice)} unseen incidents (parallel v2 vs v1)...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as top_pool:
        fut_v2_5x = top_pool.submit(
            evaluate_v2_role_diverse_tree_5x, v2_urls, tokenizer, tree_slice, args.max_turns, 4864, args.distractors, args.workers
        )
        fut_base_5x = (
            top_pool.submit(
                evaluate_v2_role_diverse_tree_5x, base_urls, tokenizer, tree_slice, args.max_turns, 4864, args.distractors, args.workers
            )
            if base_urls
            else None
        )
        v2_tree_5x = fut_v2_5x.result()
        base_tree_5x = fut_base_5x.result() if fut_base_5x else None
    print("[v2-bench] OmniSRE v2.0 Role-Diverse 5x Tree:", json.dumps(v2_tree_5x), flush=True)
    if base_tree_5x:
        print("[v2-bench] Baseline (v1.0) Role-Diverse 5x Tree:", json.dumps(base_tree_5x), flush=True)

    v2_tier1 = None
    base_tier1 = None
    if args.heldout_tier1_jsonl.exists():
        print(f"[v2-bench] Evaluating {args.tier1_limit} unseen Tier-1 write-remediation topologies (k=3)...", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as top_pool:
            fut_t1_v2 = top_pool.submit(
                eval_tier1_write, v2_urls[0], tokenizer, args.heldout_tier1_jsonl, args.tier1_limit, 3, min(32, args.workers)
            )
            fut_t1_base = (
                top_pool.submit(
                    eval_tier1_write, base_urls[0], tokenizer, args.heldout_tier1_jsonl, args.tier1_limit, 3, min(32, args.workers)
                )
                if base_urls
                else None
            )
            v2_tier1 = fut_t1_v2.result()
            base_tier1 = fut_t1_base.result() if fut_t1_base else None
        print("[v2-bench] OmniSRE v2.0 Tier-1 Write:", json.dumps(v2_tier1), flush=True)
        if base_tier1:
            print("[v2-bench] Baseline (v1.0) Tier-1 Write:", json.dumps(base_tier1), flush=True)

    report = {
        "checkpoint": "OmniSRE v2.0 (Step 50)",
        "v2_raw_unseen_1x": v2_raw_1x,
        "baseline_raw_unseen_1x": base_raw_1x,
        "v2_role_diverse_tree_5x": v2_tree_5x,
        "baseline_role_diverse_tree_5x": base_tree_5x,
        "v2_tier1_write_remediation": v2_tier1,
        "baseline_tier1_write_remediation": base_tier1,
    }
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[v2-bench] Saved full v2.0 benchmark report -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
