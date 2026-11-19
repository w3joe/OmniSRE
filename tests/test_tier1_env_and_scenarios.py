"""Tests for Tier-1 Snapshot Simulator, persistent faults vs restarts, and metastable cascades."""

import random

from sre_rl.data.scenario_factory import FAULT_TEMPLATES, generate_scenario
from sre_rl.data.trajectory_builder import synthesize_expert_trajectory
from sre_rl.env.tier1_snapshot_env import Tier1SnapshotEnvironment
from sre_rl.rewards.verifiable_reward import score_episode


def _find_template(fault_id: str) -> dict:
    return next(t for t in FAULT_TEMPLATES if t["fault_id"] == fault_id)


def test_blind_pod_restart_fails_on_persistent_oom_defect() -> None:
    rng = random.Random(7)
    scenario = generate_scenario(
        index=1,
        rng=rng,
        topology_name="otel-astronomy-shop",
        template=_find_template("k8s_oom_memory_limit"),
    )
    env = Tier1SnapshotEnvironment()
    env.reset(scenario)

    env.step('<think>Read metrics</think><tool_call>{"name": "query_metrics", "arguments": {}}</tool_call>')
    env.step('<think>Read logs</think><tool_call>{"name": "search_logs", "arguments": {}}</tool_call>')
    restart_out = env.step(
        f'<think>Blindly restart pods</think>'
        f'<tool_call>{{"name": "restart_workload", "arguments": {{"workload": "{scenario.fault.root_service}-deploy"}}}}</tool_call>'
    )
    assert "persistent fault remains active" in restart_out.observation

    ep = env.finalize_episode()
    reward = score_episode(scenario, ep)
    assert ep.slo_burn_in_passed is False
    assert reward.mitigation_score == 0.0
    assert reward.mitigated is False


def test_metastable_retry_storm_requires_both_patch_and_scale() -> None:
    rng = random.Random(11)
    scenario = generate_scenario(
        index=2,
        rng=rng,
        topology_name="payment-ledger-core",
        template=_find_template("metastable_retry_storm_cascade"),
    )
    env = Tier1SnapshotEnvironment()
    env.reset(scenario)

    env.step('<think>Read 1</think><tool_call>{"name": "query_metrics", "arguments": {}}</tool_call>')
    env.step('<think>Read 2</think><tool_call>{"name": "k8s_inspect", "arguments": {}}</tool_call>')

    # Step 1: Patch MAX_RETRIES=2 alone -> root config fixed, but retry queue still overloaded!
    p_out = env.step(
        f'<think>Patch retry cap</think>'
        f'<tool_call>{{"name": "apply_remediation", "arguments": {{"namespace": "{scenario.fault.root_namespace}", '
        f'"target_object": "{scenario.fault.root_object}", "patch": {{"MAX_RETRIES": "2"}}}}}}</tool_call>'
    )
    assert "metastable overload" in p_out.observation
    assert env.is_fully_mitigated is False

    # Step 2: Scale replicas to 6 -> drains backlog and restores SLO!
    env.step(
        f'<think>Scale replicas to drain backlog</think>'
        f'<tool_call>{{"name": "scale_resource", "arguments": {{"namespace": "{scenario.fault.root_namespace}", '
        f'"workload": "{scenario.fault.root_service}-deploy", "replicas": 6}}}}</tool_call>'
    )
    assert env.is_fully_mitigated is True


def test_expert_trajectory_achieves_full_mitigation_across_all_fault_templates() -> None:
    rng = random.Random(99)
    for idx, tmpl in enumerate(FAULT_TEMPLATES):
        sc = generate_scenario(
            index=idx,
            rng=rng,
            topology_name="otel-astronomy-shop",
            template=tmpl,
            inject_adversarial_log=(idx % 2 == 0),
        )
        traj = synthesize_expert_trajectory(sc)
        assert traj["mitigated"] is True, f"Failed mitigation on {tmpl['fault_id']}"
        assert traj["reward"] >= 0.90, f"Low reward {traj['reward']} on {tmpl['fault_id']}"
