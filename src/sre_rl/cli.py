"""CLI entrypoints for dataset construction and Pass^k evaluation."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from sre_rl.data.scenario_factory import generate_scenario_suite
from sre_rl.data.splits import split_scenarios_decontaminated
from sre_rl.data.trajectory_builder import (
    export_jsonl,
    export_parquet_or_jsonl,
    purify_trajectory,
    synthesize_expert_trajectory,
    to_verl_rl_row,
)
from sre_rl.env.tier1_snapshot_env import Tier1SnapshotEnvironment
from sre_rl.eval.metrics import summarize_benchmark
from sre_rl.rewards.verifiable_reward import score_episode
from sre_rl.schemas import IncidentScenario


def build_datasets_pipeline(
    output_dir: Path,
    num_scenarios: int = 600,
    sft_trajectories: int = 400,
    seed: int = 42,
) -> dict[str, str | int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = generate_scenario_suite(num_scenarios=num_scenarios, seed=seed)
    splits = split_scenarios_decontaminated(scenarios)

    # 1. Export Dataset A: Cold-Start SFT trajectories (from de-contaminated train split only!)
    sft_rows = []
    for sc in splits.train[:sft_trajectories]:
        traj = synthesize_expert_trajectory(sc)
        traj["messages"] = purify_trajectory(traj["messages"])
        sft_rows.append(traj)
    sft_path = export_jsonl(sft_rows, output_dir / "sft_coldstart_train.jsonl")

    # 2. Export Dataset B: veRL RLVR train / val / held-out datasets (Parquet + JSONL)
    rl_train_rows = [to_verl_rl_row(sc) for sc in splits.train]
    rl_val_rows = [to_verl_rl_row(sc) for sc in splits.val_in_domain]
    rl_heldout_rows = [to_verl_rl_row(sc) for sc in splits.eval_heldout]

    train_pq = export_parquet_or_jsonl(rl_train_rows, output_dir / "rl_tier1_train.parquet")
    val_pq = export_parquet_or_jsonl(rl_val_rows, output_dir / "rl_tier1_val.parquet")
    heldout_pq = export_parquet_or_jsonl(rl_heldout_rows, output_dir / "rl_tier1_heldout.parquet")

    # Save full scenario objects for standalone `Pass^k` evaluation harness
    eval_json = output_dir / "eval_heldout_scenarios.json"
    eval_json.write_text(
        json.dumps([sc.model_dump() for sc in splits.eval_heldout], indent=2),
        encoding="utf-8",
    )

    manifest = {
        "total_scenarios": len(scenarios),
        "train_scenarios": len(splits.train),
        "val_in_domain_scenarios": len(splits.val_in_domain),
        "eval_heldout_scenarios": len(splits.eval_heldout),
        "sft_trajectories": len(sft_rows),
        "sft_path": str(sft_path),
        "rl_train_path": str(train_pq),
        "rl_val_path": str(val_pq),
        "rl_heldout_path": str(heldout_pq),
        "eval_scenarios_json": str(eval_json),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _run_simulated_policy(
    scenario: IncidentScenario,
    policy: str,
    rng: random.Random,
):
    """Run reference policies (`expert`, `oracle_noisy`, `naive_restarter`) or OpenAI-compatible SGLang endpoint."""
    if policy == "expert":
        traj = synthesize_expert_trajectory(scenario)
        env = Tier1SnapshotEnvironment()
        env.reset(scenario)
        for msg in traj["messages"]:
            if msg["role"] == "assistant":
                out = env.step(msg["content"])
                if out.done:
                    break
        ep = env.finalize_episode()
        score_episode(scenario, ep)
        return ep

    if policy == "oracle_noisy":
        # Simulates a strong post-trained policy (~90% per-run accuracy, 0% safety violations)
        if rng.random() < 0.92:
            return _run_simulated_policy(scenario, "expert", rng)
        env = Tier1SnapshotEnvironment()
        env.reset(scenario)
        env.step(
            '<think>Inspect metrics first.</think>\n'
            '<tool_call>{"name": "query_metrics", "arguments": {"metric": "http_requests_error_ratio"}}</tool_call>'
        )
        env.step(
            '<think>Submit partial report without completing remediation.</think>\n'
            f'<tool_call>{{"name": "submit_incident_report", "arguments": {{"classification": "{scenario.fault.classification.value}", "root_service": "{scenario.fault.root_service}"}}}}</tool_call>'
        )
        ep = env.finalize_episode()
        score_episode(scenario, ep)
        return ep

    if policy == "naive_restarter":
        # Classic untrained agent anti-pattern: blindly restarts pods on every alert
        env = Tier1SnapshotEnvironment()
        env.reset(scenario)
        env.step(
            '<think>Check metrics.</think>\n'
            '<tool_call>{"name": "query_metrics", "arguments": {}}</tool_call>'
        )
        env.step(
            '<think>Check pods.</think>\n'
            f'<tool_call>{{"name": "k8s_inspect", "arguments": {{"namespace": "{scenario.fault.root_namespace}"}}}}</tool_call>'
        )
        env.step(
            '<think>Restart pod to fix it.</think>\n'
            f'<tool_call>{{"name": "restart_workload", "arguments": {{"workload": "{scenario.fault.root_service}-deploy"}}}}</tool_call>'
        )
        ep = env.finalize_episode()
        score_episode(scenario, ep)
        return ep

    raise ValueError(f"Unknown local benchmark policy: {policy}")


def main_build_datasets() -> None:
    parser = argparse.ArgumentParser(description="Build SRE Dataset A (SFT) & Dataset B (RLVR).")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--num-scenarios", type=int, default=600)
    parser.add_argument("--sft-trajectories", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_datasets_pipeline(
        output_dir=args.output_dir,
        num_scenarios=args.num_scenarios,
        sft_trajectories=args.sft_trajectories,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


def main_eval() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SRE policy on Pass^k & Safety Gates.")
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=["expert", "oracle_noisy", "naive_restarter", "sglang_endpoint"],
        default="expert",
    )
    parser.add_argument("--endpoint-url", type=str, default="http://127.0.0.1:30000/v1/chat/completions")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.8-27B")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    raw_list = json.loads(args.scenarios.read_text(encoding="utf-8"))
    scenarios = [IncidentScenario.model_validate(item) for item in raw_list]
    rng = random.Random(args.seed)

    grouped = []
    for sc in scenarios:
        runs = []
        for _ in range(args.k):
            if args.policy == "sglang_endpoint":
                runs.append(_run_sglang_endpoint_episode(sc, args.endpoint_url, args.model_name))
            else:
                runs.append(_run_simulated_policy(sc, args.policy, rng))
        grouped.append((sc, runs))

    summary = summarize_benchmark(grouped, k=args.k)
    print(json.dumps(asdict(summary), indent=2))


def _run_sglang_endpoint_episode(
    scenario: IncidentScenario,
    endpoint_url: str,
    model_name: str,
):
    """Evaluate a live SGLang / vLLM server serving Qwen3.8-27B with preserve_thinking."""
    import urllib.request

    from sre_rl.model.thinking import build_sre_system_prompt, format_multiturn_preserve_thinking

    env = Tier1SnapshotEnvironment()
    initial = env.reset(scenario)
    messages = [
        {"role": "system", "content": build_sre_system_prompt(scenario.reasoning_effort)},
        {"role": "user", "content": initial},
    ]

    for _ in range(env.max_turns):
        payload = json.dumps(
            {
                "model": model_name,
                "messages": format_multiturn_preserve_thinking(messages, preserve_thinking=True),
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 1024,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            endpoint_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assistant_text = body["choices"][0]["message"]["content"]
        outcome = env.step(assistant_text)
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append(
            {"role": "user", "content": f"<tool_response>\n{outcome.observation}\n</tool_response>"}
        )
        if outcome.done:
            break

    ep = env.finalize_episode()
    score_episode(scenario, ep)
    return ep
