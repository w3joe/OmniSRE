"""End-to-end tests for anti-contamination dataset splitting, veRL scoring hook, and Pass^k eval."""

from pathlib import Path

from sre_rl.cli import build_datasets_pipeline
from sre_rl.data.scenario_factory import generate_scenario_suite
from sre_rl.data.splits import (
    DEFAULT_HELDOUT_FAULT_IDS,
    DEFAULT_HELDOUT_TOPOLOGIES,
    split_scenarios_decontaminated,
)
from sre_rl.data.trajectory_builder import repair_failed_trajectory, synthesize_expert_trajectory
from sre_rl.env.tier1_snapshot_env import Tier1SnapshotEnvironment
from sre_rl.eval.metrics import compute_pass_at_k, compute_pass_hat_k, summarize_benchmark
from sre_rl.rewards.verifiable_reward import score_episode
from sre_rl.verl_ext.reward_manager import compute_score


def test_decontaminated_split_isolates_heldout_topology_and_fault() -> None:
    scenarios = generate_scenario_suite(num_scenarios=120, seed=42)
    splits = split_scenarios_decontaminated(scenarios)

    assert len(splits.train) > 0
    assert len(splits.eval_heldout) > 0

    for sc in splits.train + splits.val_in_domain:
        assert sc.topology_name not in DEFAULT_HELDOUT_TOPOLOGIES
        assert sc.fault.fault_id not in DEFAULT_HELDOUT_FAULT_IDS


def test_verl_reward_manager_scores_chatml_trajectory_and_failure_repair() -> None:
    scenarios = generate_scenario_suite(num_scenarios=20, seed=10)
    real_sc = next(s for s in scenarios if s.fault.classification.value == "real_incident")
    expert = synthesize_expert_trajectory(real_sc)

    chatml = "\n".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
        for m in expert["messages"]
        if m["role"] == "assistant"
    )
    scored = compute_score(
        data_source="sre_tier1_test",
        solution_str=chatml,
        ground_truth=real_sc.model_dump_json(),
    )
    assert scored["mitigated"] == 1.0
    assert scored["score"] >= 0.90

    # Test failure trajectory repair
    broken_msgs = expert["messages"][:4] + [
        {
            "role": "assistant",
            "content": '<think>bad</think><tool_call>{"name": "restart_workload", "arguments": {"workload": "x"}}</tool_call>',
        },
        {"role": "user", "content": "<tool_response>persistent fault remains active</tool_response>"},
    ]
    repaired = repair_failed_trajectory(real_sc, broken_msgs)
    assert repaired["mitigated"] is True
    assert repaired["repaired_from_failed_prefix_turns"] == 1


def test_pass_hat_k_distinguishes_flaky_agent_from_consistent_agent(tmp_path: Path) -> None:
    # 4 out of 5 successes -> Pass@5 = 1.0 (100%), Pass^5 = 0.0 (0%!)
    assert compute_pass_at_k(n=5, c=4, k=5) == 1.0
    assert compute_pass_hat_k(n=5, c=4, k=5) == 0.0
    assert compute_pass_hat_k(n=5, c=5, k=5) == 1.0

    manifest = build_datasets_pipeline(
        output_dir=tmp_path / "ds",
        num_scenarios=40,
        sft_trajectories=15,
        seed=7,
    )
    assert int(manifest["sft_trajectories"]) > 0

    scenarios = generate_scenario_suite(num_scenarios=6, seed=5)
    grouped = []
    for sc in scenarios:
        traj = synthesize_expert_trajectory(sc)
        runs = []
        for _ in range(5):
            env = Tier1SnapshotEnvironment()
            env.reset(sc)
            for msg in traj["messages"]:
                if msg["role"] == "assistant":
                    out = env.step(msg["content"])
                    if out.done:
                        break
            ep = env.finalize_episode()
            score_episode(sc, ep)
            runs.append(ep)
        grouped.append((sc, runs))

    summary = summarize_benchmark(grouped, k=5)
    assert summary.pass_hat_k == 1.0
    assert summary.safety_violation_rate == 0.0
    assert summary.production_gate_passed is True
