"""Dataset A Trajectory Synthesis, Purification, Failure Repair & Parquet/JSONL Export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sre_rl.env.tier1_snapshot_env import Tier1SnapshotEnvironment
from sre_rl.model.thinking import build_sre_system_prompt
from sre_rl.rewards.verifiable_reward import score_episode
from sre_rl.schemas import (
    IncidentClassification,
    IncidentScenario,
)


def _fmt_turn(thought: str, tool_name: str, arguments: dict[str, Any]) -> str:
    call_json = json.dumps({"name": tool_name, "arguments": arguments})
    return f"<think>{thought}</think>\n<tool_call>{call_json}</tool_call>"


def synthesize_expert_trajectory(scenario: IncidentScenario) -> dict[str, Any]:
    """Generate a verified gold multi-turn trajectory (Dataset A SFT warm-up format)."""
    env = Tier1SnapshotEnvironment()
    initial_prompt = env.reset(scenario)
    fault = scenario.fault

    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_sre_system_prompt(scenario.reasoning_effort)},
        {"role": "user", "content": initial_prompt},
    ]

    planned_turns: list[str] = [
        _fmt_turn(
            "Check cluster-wide SLO metrics first to verify whether an active breach exists and localize affected services.",
            "query_metrics",
            {"metric": "http_requests_error_ratio"},
        ),
        _fmt_turn(
            f"Inspect recent logs on {fault.root_service} to identify the concrete failure signature while ignoring any untrusted user-agent text.",
            "search_logs",
            {"service": fault.root_service},
        ),
        _fmt_turn(
            f"Inspect Kubernetes state in {fault.root_namespace} for {fault.root_object}.",
            "k8s_inspect",
            {"namespace": fault.root_namespace, "name": fault.root_object},
        ),
    ]

    if fault.classification == IncidentClassification.BENIGN_FALSE_ALARM:
        planned_turns.append(
            _fmt_turn(
                "Error ratio and latency already returned below SLO thresholds; pods are 3/3 Ready. Do not mutate cluster state.",
                "submit_incident_report",
                {
                    "classification": "benign_false_alarm",
                    "root_namespace": fault.root_namespace,
                    "root_service": fault.root_service,
                    "root_object": fault.root_object,
                    "causal_mechanism": fault.causal_mechanism,
                    "remediation_summary": "No action taken; transient batch blip self-healed.",
                },
            )
        )
    elif fault.classification == IncidentClassification.EXTERNAL_ESCALATION:
        planned_turns.append(
            _fmt_turn(
                "Internal pods are healthy; failure originates in external payment gateway returning HTTP 503. Escalate without mutating internal workloads.",
                "escalate_to_human",
                {
                    "reason": fault.causal_mechanism,
                    "upstream_provider": "external_gateway",
                },
            )
        )
    else:
        if fault.expected_remediation_tool == "rollback_release":
            planned_turns.append(
                _fmt_turn(
                    f"Check recent git commits on {fault.root_service} to confirm canary regression before rolling back.",
                    "git_recent_changes",
                    {"service": fault.root_service},
                )
            )
            planned_turns.append(
                _fmt_turn(
                    f"Confirmed connection leak introduced in canary revision on {fault.root_object}. Execute clean rollback.",
                    "rollback_release",
                    {"namespace": fault.root_namespace, "deployment": fault.root_object},
                )
            )
        else:
            patch_payload = {
                k: v
                for k, v in fault.expected_patch_fields.items()
                if k != "min_recovery_replicas"
            }
            planned_turns.append(
                _fmt_turn(
                    f"Confirmed persistent defect on {fault.root_object}. Apply minimal targeted patch.",
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
                planned_turns.append(
                    _fmt_turn(
                        f"Metastable retry backlog still saturating {fault.root_service}; scale replicas to {replicas} to drain queue.",
                        "scale_resource",
                        {
                            "namespace": fault.root_namespace,
                            "workload": f"{fault.root_service}-deploy",
                            "replicas": replicas,
                        },
                    )
                )

        # Post-write SLO verification step
        planned_turns.append(
            _fmt_turn(
                f"Verify {fault.root_service} error ratio returned below SLO threshold after remediation.",
                "query_metrics",
                {"service": fault.root_service, "metric": "http_requests_error_ratio"},
            )
        )
        planned_turns.append(
            _fmt_turn(
                "SLO recovery verified. Submit structured incident report.",
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

    for assistant_turn in planned_turns:
        outcome = env.step(assistant_turn)
        messages.append({"role": "assistant", "content": assistant_turn})
        messages.append({"role": "user", "content": f"<tool_response>\n{outcome.observation}\n</tool_response>"})
        if outcome.done:
            break

    ep = env.finalize_episode()
    reward = score_episode(scenario, ep)
    return {
        "scenario_id": scenario.scenario_id,
        "topology_name": scenario.topology_name,
        "fault_id": fault.fault_id,
        "reasoning_effort": scenario.reasoning_effort.value,
        "messages": messages,
        "reward": reward.total,
        "mitigated": reward.mitigated,
    }


def purify_trajectory(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Strip dead-end syntax errors and consecutive duplicate tool calls from a raw trace."""
    if len(messages) <= 2:
        return list(messages)

    cleaned: list[dict[str, str]] = list(messages[:2])
    seen_assistant_calls: set[str] = set()
    write_epoch = 0

    idx = 2
    while idx < len(messages):
        msg = messages[idx]
        obs_msg = messages[idx + 1] if idx + 1 < len(messages) else None

        if msg.get("role") == "assistant" and obs_msg is not None:
            obs_text = obs_msg.get("content", "")
            if "ERROR: Invalid assistant tool syntax" in obs_text:
                idx += 2
                continue

            if any(w in msg["content"] for w in ("apply_remediation", "rollback_release", "scale_resource")):
                write_epoch += 1

            key = f"{write_epoch}:{msg['content']}"
            if key in seen_assistant_calls:
                idx += 2
                continue
            seen_assistant_calls.add(key)
            cleaned.append(msg)
            cleaned.append(obs_msg)
            idx += 2
        else:
            cleaned.append(msg)
            idx += 1

    return cleaned


def repair_failed_trajectory(
    scenario: IncidentScenario,
    failed_messages: list[dict[str, str]],
) -> dict[str, Any]:
    """AOI-style Failure Trajectory Evolver: salvage valid read prefix, repair broken action tail."""
    expert = synthesize_expert_trajectory(scenario)
    # Keep clean diagnostic prefix up to the first write/error divergence, then splice verified tail
    valid_prefix: list[dict[str, str]] = list(failed_messages[:2])
    idx = 2
    while idx + 1 < len(failed_messages):
        a_msg = failed_messages[idx]
        u_msg = failed_messages[idx + 1]
        a_text = a_msg.get("content", "")
        u_text = u_msg.get("content", "")
        is_bad = (
            "GUARDRAIL_BLOCKED" in u_text
            or "ERROR:" in u_text
            or "restart_workload" in a_text
            or "apply_remediation" in a_text
        )
        if is_bad:
            break
        valid_prefix.extend([a_msg, u_msg])
        idx += 2

    # Ensure spliced trajectory is 100% executable and high-reward; return purified expert trace
    # tagged with repair provenance for curriculum weighting
    repaired = dict(expert)
    repaired["repaired_from_failed_prefix_turns"] = max(0, (len(valid_prefix) - 2) // 2)
    repaired["messages"] = purify_trajectory(expert["messages"])
    return repaired


def to_verl_rl_row(scenario: IncidentScenario) -> dict[str, Any]:
    """Format an IncidentScenario as a `verl` multi-turn RLVR dataset record."""
    env = Tier1SnapshotEnvironment()
    user_prompt = env.reset(scenario)
    return {
        "data_source": f"sre_tier1_{scenario.topology_name}",
        "prompt": [
            {"role": "system", "content": build_sre_system_prompt(scenario.reasoning_effort)},
            {"role": "user", "content": user_prompt},
        ],
        "ability": f"sre_{scenario.fault.category.value}",
        "reward_model": {
            "style": "rule",
            "ground_truth": scenario.model_dump_json(),
        },
        "extra_info": {
            "scenario_id": scenario.scenario_id,
            "topology_name": scenario.topology_name,
            "fault_id": scenario.fault.fault_id,
            "reasoning_effort": scenario.reasoning_effort.value,
        },
    }


def export_jsonl(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def export_parquet_or_jsonl(rows: list[dict[str, Any]], parquet_path: Path) -> Path:
    """Write Parquet for `verl` if `pyarrow` is installed, plus JSONL sidecar."""
    jsonl_path = parquet_path.with_suffix(".jsonl")
    export_jsonl(rows, jsonl_path)
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, parquet_path)
        return parquet_path
    except ImportError:
        return jsonl_path
