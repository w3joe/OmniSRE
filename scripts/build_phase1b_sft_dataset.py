#!/usr/bin/env python3
"""Build Phase 1b Exact-Schema SFT Calibration Dataset (strictly train-only, zero held-out overlap).

Solves the Stage 1 SFT tool-schema dilution bottleneck:
- Stage 1a (`5,038` traces) gave Qwen3.8-27B broad real-world OpenSRE, K8s & DevOps reasoning.
- Phase 1b generates `1,559` purified, 100%-exact-schema multi-turn trajectories strictly from
  `rl_real_train.jsonl` (NEVER touching `rl_real_heldout.jsonl`!), including:
    * Real OpenShift / Kube-Prometheus / PromQL alert diagnosis + `search_runbooks` lookup
    * Exact `apply_remediation(namespace, target_object, patch)` + `rollback_release` + `scale_resource`
    * Post-write `query_metrics` SLO verification + `submit_incident_report` closeout
    * Benign false-alarm abstention (0 writes) + external outage `escalate_to_human`
    * Adversarial log prompt-injection resistance + metastable two-step recovery
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from sre_rl.data.trajectory_builder import (
    export_jsonl,
    purify_trajectory,
    synthesize_expert_trajectory,
)
from sre_rl.data.validator import scan_text_for_leaks, validate_sft_row
from sre_rl.env.tier1_snapshot_env import Tier1SnapshotEnvironment
from sre_rl.model.thinking import build_sre_system_prompt
from sre_rl.rewards.verifiable_reward import score_episode
from sre_rl.schemas import IncidentClassification, IncidentScenario


def _fmt_turn(thought: str, tool_name: str, arguments: dict[str, Any]) -> str:
    call_json = json.dumps({"name": tool_name, "arguments": arguments})
    return f"<think>{thought}</think>\n<tool_call>{call_json}</tool_call>"


def synthesize_runbook_grounded_trajectory(
    scenario: IncidentScenario,
    include_runbook_lookup: bool = True,
) -> dict[str, Any]:
    """Generate a high-diversity exact-schema trace that also exercises `search_runbooks` & `get_traces`."""
    if not include_runbook_lookup or scenario.fault.classification != IncidentClassification.REAL_INCIDENT:
        traj = synthesize_expert_trajectory(scenario)
        traj["messages"] = purify_trajectory(traj["messages"])
        return traj

    env = Tier1SnapshotEnvironment(max_turns=14)
    initial_prompt = env.reset(scenario)
    fault = scenario.fault

    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_sre_system_prompt(scenario.reasoning_effort)},
        {"role": "user", "content": initial_prompt},
    ]

    turns: list[str] = [
        _fmt_turn(
            "Verify active SLO error-budget burn across services before touching any resource.",
            "query_metrics",
            {"metric": "http_requests_error_ratio"},
        ),
        _fmt_turn(
            f"Consult runbooks for {fault.root_service} / {fault.fault_id} to check documented root causes and safe remediation keys.",
            "search_runbooks",
            {"query": f"{fault.root_service} {fault.category.value}"},
        ),
        _fmt_turn(
            f"Inspect Kubernetes object {fault.root_object} in {fault.root_namespace} to confirm the live misconfiguration.",
            "k8s_inspect",
            {"namespace": fault.root_namespace, "name": fault.root_object},
        ),
    ]

    if fault.expected_remediation_tool == "rollback_release":
        turns.append(
            _fmt_turn(
                f"Check recent git commits on {fault.root_service} to confirm release regression.",
                "git_recent_changes",
                {"service": fault.root_service},
            )
        )
        turns.append(
            _fmt_turn(
                f"Confirmed canary regression on {fault.root_object}. Roll back to previous stable revision.",
                "rollback_release",
                {"namespace": fault.root_namespace, "deployment": fault.root_object},
            )
        )
    else:
        patch_payload = {
            k: v for k, v in fault.expected_patch_fields.items() if k != "min_recovery_replicas"
        }
        turns.append(
            _fmt_turn(
                f"Root cause confirmed on {fault.root_object} ({fault.causal_mechanism[:120]}). Apply minimal targeted patch.",
                "apply_remediation",
                {
                    "namespace": fault.root_namespace,
                    "target_object": fault.root_object,
                    "patch": patch_payload,
                },
            )
        )
        if fault.requires_secondary_drain_or_scale:
            replicas = int(fault.expected_patch_fields.get("min_recovery_replicas", 6))
            turns.append(
                _fmt_turn(
                    f"Metastable retry backlog persists on {fault.root_service}; scale to {replicas} replicas to drain queue.",
                    "scale_resource",
                    {
                        "namespace": fault.root_namespace,
                        "workload": f"{fault.root_service}-deploy",
                        "replicas": replicas,
                    },
                )
            )

    turns.append(
        _fmt_turn(
            f"Verify {fault.root_service} error ratio returned below SLO threshold post-fix.",
            "query_metrics",
            {"service": fault.root_service, "metric": "http_requests_error_ratio"},
        )
    )
    turns.append(
        _fmt_turn(
            "SLO recovery verified with zero regressions. Submit structured closeout report.",
            "submit_incident_report",
            {
                "classification": "real_incident",
                "root_namespace": fault.root_namespace,
                "root_service": fault.root_service,
                "root_object": fault.root_object,
                "causal_mechanism": fault.causal_mechanism,
                "remediation_summary": f"Remediated {fault.root_object} and verified SLO recovery.",
            },
        )
    )

    for a_turn in turns:
        out = env.step(a_turn)
        messages.append({"role": "assistant", "content": a_turn})
        messages.append({"role": "user", "content": f"<tool_response>\n{out.observation}\n</tool_response>"})
        if out.done:
            break

    ep = env.finalize_episode()
    reward = score_episode(scenario, ep)
    return {
        "scenario_id": scenario.scenario_id,
        "topology_name": scenario.topology_name,
        "fault_id": fault.fault_id,
        "source": "phase1b_exact_schema_calibration",
        "reasoning_effort": scenario.reasoning_effort.value,
        "messages": purify_trajectory(messages),
        "reward": reward.total,
        "mitigated": reward.mitigated,
    }


def _load_scenarios(jsonl_path: Path) -> list[IncidentScenario]:
    out = []
    for ln in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        gt = row["reward_model"]["ground_truth"]
        out.append(
            IncidentScenario.model_validate_json(gt)
            if isinstance(gt, str)
            else IncidentScenario.model_validate(gt)
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 1b exact-schema SFT calibration dataset")
    parser.add_argument("--train-rl-jsonl", type=Path, default=Path("data/real_sre_v2/rl_real_train.jsonl"))
    parser.add_argument("--heldout-rl-jsonl", type=Path, default=Path("data/real_sre_v2/rl_real_heldout.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/phase1b_sft_v4"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_scenarios = _load_scenarios(args.train_rl_jsonl)
    heldout_scenarios = _load_scenarios(args.heldout_rl_jsonl)
    heldout_topos = {s.topology_name for s in heldout_scenarios}
    heldout_ids = {s.scenario_id for s in heldout_scenarios}

    rows: list[dict[str, Any]] = []
    invalid_count = 0
    leak_count = 0

    for idx, sc in enumerate(train_scenarios):
        # Double-assert zero leakage from held-out split
        assert sc.scenario_id not in heldout_ids
        assert sc.topology_name not in heldout_topos

        # Mix runbook-lookup traces (60%) and log-first traces (40%) for strategy diversity
        use_runbook = (idx % 5) < 3
        traj = synthesize_runbook_grounded_trajectory(sc, include_runbook_lookup=use_runbook)
        if not traj["mitigated"] or traj["reward"] < 0.90:
            invalid_count += 1
            continue

        ok, _errs, _toks, has_tc = validate_sft_row(traj)
        joined = "\n".join(m["content"] for m in traj["messages"])
        leaks = scan_text_for_leaks(joined)
        if leaks:
            leak_count += len(leaks)
            continue
        if not ok or not has_tc:
            invalid_count += 1
            continue
        rows.append(traj)

    rng.shuffle(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = export_jsonl(rows, args.output_dir / "sft_phase1b_exact_schema.jsonl")

    # Verify zero prompt-hash overlap with held-out scenarios
    sft_hashes = {
        hashlib.sha256(r["messages"][1]["content"].encode("utf-8")).hexdigest()[:16]
        for r in rows
    }
    heldout_overlap = sum(
        1
        for sc in heldout_scenarios
        if hashlib.sha256(sc.alert_pager_payload.encode("utf-8")).hexdigest()[:16] in sft_hashes
    )
    assert heldout_overlap == 0, f"Contamination detected: {heldout_overlap}"
    assert leak_count == 0, f"PII/secret leaks detected: {leak_count}"

    summary = {
        "phase1b_rows": len(rows),
        "invalid_dropped": invalid_count,
        "pii_secret_leaks": leak_count,
        "heldout_contamination_overlap": heldout_overlap,
        "categories": dict(Counter(sc.fault.category.value for sc in train_scenarios)),
        "classifications": dict(Counter(sc.fault.classification.value for sc in train_scenarios)),
        "output_jsonl": str(out_jsonl),
    }
    (args.output_dir / "phase1b_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
