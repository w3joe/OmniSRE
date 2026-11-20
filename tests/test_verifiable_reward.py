"""Tests for composite verifiable rewards (graph hop localization, false alarms, overthinking)."""

import random

from sre_rl.data.scenario_factory import FAULT_TEMPLATES, generate_scenario
from sre_rl.env.tier1_snapshot_env import Tier1SnapshotEnvironment
from sre_rl.rewards.verifiable_reward import score_episode, shortest_service_hop_distance


def test_shortest_service_hop_distance() -> None:
    deps = {
        "frontend": ["checkout", "cart"],
        "checkout": ["payment", "order-postgres"],
        "cart": ["cart-redis"],
    }
    assert shortest_service_hop_distance(deps, "payment", "payment") == 0
    assert shortest_service_hop_distance(deps, "checkout", "payment") == 1
    assert shortest_service_hop_distance(deps, "frontend", "payment") == 2
    assert shortest_service_hop_distance(deps, "cart-redis", "payment") == 4
    assert shortest_service_hop_distance(deps, "unknown", "payment") is None


def test_benign_false_alarm_penalizes_unnecessary_write_action() -> None:
    tmpl = next(t for t in FAULT_TEMPLATES if t["fault_id"] == "benign_cron_metrics_blip")
    scenario = generate_scenario(index=5, rng=random.Random(3), topology_name="streaming-ingest-mesh", template=tmpl)

    env = Tier1SnapshotEnvironment()
    env.reset(scenario)
    env.step('<think>r1</think><tool_call>{"name": "query_metrics", "arguments": {}}</tool_call>')
    env.step('<think>r2</think><tool_call>{"name": "search_logs", "arguments": {}}</tool_call>')
    # Agent unnecessarily restarts a healthy service on a benign false alarm!
    env.step(
        f'<think>restart just in case</think>'
        f'<tool_call>{{"name": "restart_workload", "arguments": {{"workload": "{scenario.fault.root_service}-deploy"}}}}</tool_call>'
    )
    ep = env.finalize_episode()
    breakdown = score_episode(scenario, ep)

    assert breakdown.false_positive_action is True
    assert breakdown.safety_penalty >= 0.50
    assert breakdown.mitigated is False
    assert breakdown.total < 0.0


def test_stalled_readonly_episode_receives_timeout_penalty() -> None:
    tmpl = next(t for t in FAULT_TEMPLATES if t["fault_id"] == "k8s_oom_memory_limit")
    scenario = generate_scenario(index=7, rng=random.Random(8), topology_name="otel-astronomy-shop", template=tmpl)

    env = Tier1SnapshotEnvironment(max_turns=3)
    env.reset(scenario)
    env.step('<think>r1</think><tool_call>{"name": "query_metrics", "arguments": {}}</tool_call>')
    env.step('<think>r2</think><tool_call>{"name": "search_logs", "arguments": {}}</tool_call>')
    env.step('<think>r3</think><tool_call>{"name": "k8s_inspect", "arguments": {}}</tool_call>')
    ep = env.finalize_episode()
    breakdown = score_episode(scenario, ep)
    assert breakdown.mitigated is False
    assert breakdown.efficiency_penalty >= 0.25
    assert breakdown.total < 0.0


def test_overthinking_penalty_triggers_above_reasoning_effort_budget() -> None:
    tmpl = next(t for t in FAULT_TEMPLATES if t["fault_id"] == "benign_cron_metrics_blip")
    scenario = generate_scenario(index=6, rng=random.Random(4), topology_name="streaming-ingest-mesh", template=tmpl)

    env = Tier1SnapshotEnvironment()
    env.reset(scenario)
    verbose_thought = "overthinking " * 400  # ~1,300 tokens >> LOW budget (180 tokens)
    env.step(
        f"<think>{verbose_thought}</think>\n"
        '<tool_call>{"name": "submit_incident_report", "arguments": {"classification": "benign_false_alarm", '
        '"root_service": "none", "causal_mechanism": "benign transient recovered"}}</tool_call>'
    )
    ep = env.finalize_episode()
    breakdown = score_episode(scenario, ep)
    assert breakdown.efficiency_penalty > 0.0
