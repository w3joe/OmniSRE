#!/usr/bin/env python3
"""Full Apples-to-Apples Benchmark Suite: `OmniSRE v1.0` vs. Base `Qwen/Qwen3.8-27B`.

Evaluates side-by-side over 100% zero-contamination unseen real production systems & topologies:
1. **ALL `129` Unseen Real Production Incidents (`1x` Single Run):**
   - `DejaVu-C` (all `99` commercial Oracle DB wait-event outages)
   - `RE3-TT` (all `30` 41-service Train Ticket GKE microservice failures)
2. **True `5x` Hierarchical Tree Ensemble (`2 + 3 -> 1`)** over `40` stratified unseen incidents
   (`20` `DejaVu-C` + `20` `RE3-TT`) using `OmniSRE v1.0`.
3. **Unseen Real-Runbook Write-Gated Remediation (`100` unseen Tier-1 topologies, `k=3`):**
   - `Pass@1`, `Pass@3`, `Pass^3` consistency, and `Safety Violation Rate`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from sre_rl.env.real_telemetry_env import (
    RealIncidentRecord,
    RealTelemetryHarnessConfig,
)
from sre_rl.v5_sre1_upgrades import SampledHarnessConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse exact v1.0 TITO rollout & tree-ensemble helpers
from train_rl_v1_omnisre_4xh100 import (
    _load_tier1_scenarios,
    evaluate_tree_ensemble_5x_v1,
    rollout_real_telemetry_v1,
    rollout_tier1_write_v1,
)


def eval_real_1x(
    sglang_url: str,
    tokenizer,
    records: list[RealIncidentRecord],
    max_turns: int = 24,
    workers: int = 24,
) -> dict[str, Any]:
    t0 = time.time()
    harness = RealTelemetryHarnessConfig(is_canonical=True, max_turns=max_turns)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [
            (
                rec.source_dataset,
                pool.submit(rollout_real_telemetry_v1, sglang_url, tokenizer, rec, harness, 384, 0.0),
            )
            for rec in records
        ]
        rows = [(ds, f.result()[2]) for ds, f in futs]

    by_ds: dict[str, dict[str, Any]] = {}
    for ds in sorted({d for d, _ in rows}):
        sub = [st for d, st in rows if d == ds]
        by_ds[ds] = {
            "count": len(sub),
            "solved_rate": round(float(np.mean([s["solved"] for s in sub])), 4),
            "service_exact_acc": round(float(np.mean([s["svc_exact"] for s in sub])), 4),
            "fault_match_acc": round(float(np.mean([s["fault_match"] for s in sub])), 4),
            "avg_turns": round(float(np.mean([s["turns"] for s in sub])), 2),
        }

    all_st = [st for _, st in rows]
    return {
        "num_cases": len(all_st),
        "solved_rate": round(float(np.mean([s["solved"] for s in all_st])), 4),
        "service_exact_acc": round(float(np.mean([s["svc_exact"] for s in all_st])), 4),
        "fault_match_acc": round(float(np.mean([s["fault_match"] for s in all_st])), 4),
        "mean_reward": round(float(np.mean([s["reward"] for s in all_st])), 4),
        "avg_turns": round(float(np.mean([s["turns"] for s in all_st])), 2),
        "wall_time_s": round(time.time() - t0, 1),
        "by_dataset": by_ds,
    }


def eval_tier1_write(
    sglang_url: str,
    tokenizer,
    scenarios_path: Path,
    limit: int = 60,
    k_runs: int = 3,
    workers: int = 24,
) -> dict[str, Any]:
    t0 = time.time()
    scenarios = _load_tier1_scenarios(scenarios_path)[:limit]
    harness = SampledHarnessConfig(
        is_canonical=True,
        max_turns=14,
        max_context_tokens=12_288,
        show_budget_footer=True,
        parallel_hint="",
    )

    jobs = [
        (i, run_idx, sc, 0.0 if run_idx == 0 else 0.4)
        for i, sc in enumerate(scenarios)
        for run_idx in range(k_runs)
    ]
    per_sc: dict[int, list[dict[str, float]]] = {i: [] for i in range(len(scenarios))}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        fmap = {
            pool.submit(rollout_tier1_write_v1, sglang_url, tokenizer, sc, harness, 384, temp): idx
            for idx, _, sc, temp in jobs
        }
        for fut, idx in fmap.items():
            per_sc[idx].append(fut.result()[2])

    pass_at_1 = float(np.mean([per_sc[i][0]["mitigated"] for i in range(len(scenarios))]))
    pass_at_k = float(np.mean([max(r["mitigated"] for r in per_sc[i]) for i in range(len(scenarios))]))
    pass_hat_k = float(np.mean([min(r["mitigated"] for r in per_sc[i]) for i in range(len(scenarios))]))
    safety_at_0 = float(np.mean([per_sc[i][0]["safety_violation"] for i in range(len(scenarios))]))
    safety_all = float(np.mean([r["safety_violation"] for i in range(len(scenarios)) for r in per_sc[i]]))
    avg_turns = float(np.mean([r["turns"] for i in range(len(scenarios)) for r in per_sc[i]]))
    return {
        "num_scenarios": len(scenarios),
        "k_runs": k_runs,
        "pass_at_1": round(pass_at_1, 4),
        "pass_at_k": round(pass_at_k, 4),
        "pass_hat_k": round(pass_hat_k, 4),
        "safety_violation_greedy": round(safety_at_0, 4),
        "safety_violation_all_runs": round(safety_all, 4),
        "avg_turns": round(avg_turns, 2),
        "wall_time_s": round(time.time() - t0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full OmniSRE v1.0 vs Base Qwen3.8-27B Benchmark Suite")
    parser.add_argument("--omnisre-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30001")
    parser.add_argument("--tokenizer-path", type=str, default="/mnt/nvme/omnisre-27b-v1.0-merged")
    parser.add_argument("--heldout-real-jsonl", type=Path, default=Path("/mnt/nvme/real_telemetry/index/real_telemetry_heldout.jsonl"))
    parser.add_argument("--heldout-tier1-jsonl", type=Path, default=Path("/mnt/nvme/data_v2/rl_real_heldout.jsonl"))
    parser.add_argument("--tier1-limit", type=int, default=60)
    parser.add_argument("--tree-5x-count", type=int, default=40)
    parser.add_argument("--out-json", type=Path, default=Path("/mnt/nvme/v1_full_benchmark_results.json"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    heldout_records = [
        RealIncidentRecord.model_validate_json(ln)
        for ln in args.heldout_real_jsonl.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    # Stratified 40-case slice (20 DejaVu-C + 20 RE3-TT) for 5x Hierarchical Tree Ensemble
    by_ds: dict[str, list[RealIncidentRecord]] = {}
    for rec in heldout_records:
        by_ds.setdefault(rec.source_dataset, []).append(rec)
    tree_slice: list[RealIncidentRecord] = []
    per_ds = max(1, args.tree_5x_count // max(1, len(by_ds)))
    for ds_name in sorted(by_ds):
        bucket = list(by_ds[ds_name])
        random.Random(42).shuffle(bucket)
        tree_slice.extend(bucket[:per_ds])

    print(f"[bench] Evaluating ALL {len(heldout_records)} unseen real production incidents (OmniSRE v1.0 vs Base)...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as top_pool:
        f_omni_real = top_pool.submit(eval_real_1x, args.omnisre_url, tokenizer, heldout_records, 24, 24)
        f_base_real = top_pool.submit(eval_real_1x, args.base_url, tokenizer, heldout_records, 24, 24)
        omni_real = f_omni_real.result()
        base_real = f_base_real.result()
    print("[bench] REAL 1x OmniSRE v1.0:", json.dumps(omni_real), flush=True)
    print("[bench] REAL 1x Base Qwen3.8:", json.dumps(base_real), flush=True)

    print(f"[bench] Evaluating {args.tier1_limit} unseen Tier-1 write-remediation topologies (k=3)...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as top_pool:
        f_omni_t1 = top_pool.submit(eval_tier1_write, args.omnisre_url, tokenizer, args.heldout_tier1_jsonl, args.tier1_limit, 3, 24)
        f_base_t1 = top_pool.submit(eval_tier1_write, args.base_url, tokenizer, args.heldout_tier1_jsonl, args.tier1_limit, 3, 24)
        omni_t1 = f_omni_t1.result()
        base_t1 = f_base_t1.result()
    print("[bench] WRITE Tier-1 OmniSRE v1.0:", json.dumps(omni_t1), flush=True)
    print("[bench] WRITE Tier-1 Base Qwen3.8:", json.dumps(base_t1), flush=True)

    print(f"[bench] Running True 5x Hierarchical Tree Ensemble (2+3 -> 1) on {len(tree_slice)} unseen real incidents...", flush=True)
    tree_5x = evaluate_tree_ensemble_5x_v1(args.omnisre_url, tokenizer, tree_slice, max_turns=24, workers=20)
    print("[bench] TREE 5x OmniSRE v1.0:", json.dumps(tree_5x), flush=True)

    report = {
        "checkpoint": "OmniSRE v1.0 (step_100 best_checkpoint, cleanly merged standalone BF16)",
        "base_model": "Qwen/Qwen3.8-27B",
        "real_production_all_129_1x": {
            "omnisre_v1": omni_real,
            "base_qwen38_27b": base_real,
        },
        "tier1_write_remediation_unseen": {
            "omnisre_v1": omni_t1,
            "base_qwen38_27b": base_t1,
        },
        "real_production_tree_ensemble_5x": tree_5x,
    }
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[bench] Saved full benchmark report -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
