"""Tests for OmniSRE v5 upgrades (Masked CISPO, Parallel Tool Calls, Harness Randomization, Pools)."""

import random

import pytest

from sre_rl.data.scenario_factory import generate_scenario_suite
from sre_rl.model.thinking import parse_qwen38_assistant_turn_multi
from sre_rl.rewards.dapo_gtpo import compute_masked_cispo_loss_torch
from sre_rl.schemas import ToolCall
from sre_rl.v5_sre1_upgrades import (
    ReturningPoolCurriculum,
    build_offline_synthesis_prompt,
    sample_group_harness,
)


def test_masked_cispo_drops_out_of_band_ratios_and_preserves_in_band_gradients() -> None:
    torch = pytest.importorskip("torch")
    old_lp = torch.tensor([-2.0, -2.0, -2.0], dtype=torch.float32)
    # Token 0: ratio = exp(0.1) = 1.105 (in band [0.2, 5.0])
    # Token 1: ratio = exp(3.0) = 20.08 (above 5.0 -> masked out!)
    # Token 2: ratio = exp(-2.5) = 0.082 (below 0.2 -> masked out!)
    new_lp = torch.tensor([-1.9, 1.0, -4.5], dtype=torch.float32, requires_grad=True)
    adv = torch.tensor([1.5, 1.5, -1.0], dtype=torch.float32)

    loss, diag = compute_masked_cispo_loss_torch(new_lp, old_lp, adv, ratio_min=0.2, ratio_max=5.0)
    loss.backward()

    assert diag["masked_fraction"] == round(2 / 3, 4)
    assert new_lp.grad is not None
    assert new_lp.grad[0].item() != 0.0
    assert new_lp.grad[1].item() == 0.0
    assert new_lp.grad[2].item() == 0.0


def test_multi_parallel_tool_calls_parsing_and_cap() -> None:
    raw = (
        "<think>Batch metrics and logs in parallel.</think>\n"
        '<tool_call>{"name": "query_metrics", "arguments": {"metric": "error"}}</tool_call>\n'
        '<tool_call>{"name": "search_logs", "arguments": {"service": "checkout"}}</tool_call>'
    )
    parsed = parse_qwen38_assistant_turn_multi(raw, max_parallel_calls=6)
    assert parsed.parse_error is None
    assert len(parsed.tool_calls) == 2
    assert parsed.tool_calls[0].name == "query_metrics"
    assert parsed.tool_calls[1].name == "search_logs"

    too_many = "\n".join(
        '<tool_call>{"name": "query_metrics", "arguments": {}}</tool_call>' for _ in range(9)
    )
    capped = parse_qwen38_assistant_turn_multi(too_many, max_parallel_calls=6)
    assert capped.parse_error is not None and "exceeded_parallel_call_cap" in capped.parse_error


def test_harness_randomization_alias_resolution_and_returning_pools() -> None:
    rng = random.Random(12)
    harness = sample_group_harness(rng, randomize_prob=1.0)
    assert harness.is_canonical is False

    alias = harness.canonical_to_alias["query_metrics"]
    resolved = harness.resolve_tool_call(ToolCall(name=alias, arguments={"service": "api"}))
    assert resolved.name == "query_metrics"

    curr = ReturningPoolCurriculum(num_scenarios=10, easy_half_life_steps=2.0, hard_half_life_steps=1.0, seed=3)
    assert curr.record_group_outcome(0, current_step=1, pass_flags=[True, True, True]) == "easy"
    assert curr.record_group_outcome(1, current_step=1, pass_flags=[False, False, False]) == "hard"
    counts = curr.pool_counts()
    assert counts["easy_pool"] == 1 and counts["hard_pool"] == 1

    sc = generate_scenario_suite(num_scenarios=2, seed=1)[0]
    synth = build_offline_synthesis_prompt(
        sc, [[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "obs"}]]
    )
    assert len(synth) == 2 and "SAVED INVESTIGATION #1" in synth[1]["content"]
