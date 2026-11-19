"""Custom Verifiable Reward Entrypoint for `verl` (`custom_reward_function.path`).

Compatible with `verl.trainer.main_ppo` via:
    custom_reward_function.path=src/sre_rl/verl_ext/reward_manager.py
    custom_reward_function.name=compute_score
"""

from __future__ import annotations

import re
from typing import Any

from sre_rl.env.tier1_snapshot_env import Tier1SnapshotEnvironment
from sre_rl.rewards.verifiable_reward import score_episode
from sre_rl.schemas import IncidentScenario

ASSISTANT_TURN_SPLIT_RE = re.compile(
    r"(?:<\|im_start\|>assistant\s*)(.*?)(?:<\|im_end\|>|$)",
    re.DOTALL,
)


def extract_assistant_turns(solution_str: str) -> list[str]:
    """Extract assistant turns from either ChatML-formatted rollouts or raw single/multi turns."""
    chatml_turns = ASSISTANT_TURN_SPLIT_RE.findall(solution_str)
    if chatml_turns:
        return [t.strip() for t in chatml_turns if t.strip()]

    # Fallback: split per `<tool_call>...</tool_call>` block preserving preceding `<think>`
    chunks: list[str] = []
    cursor = 0
    for match in re.finditer(r"</tool_call>", solution_str):
        end = match.end()
        segment = solution_str[cursor:end]
        # Strip any interleaved <tool_response>...</tool_response> from previous turn
        segment = re.sub(r"<tool_response>.*?</tool_response>", "", segment, flags=re.DOTALL).strip()
        if segment:
            chunks.append(segment)
        cursor = end

    if chunks:
        return chunks
    return [solution_str.strip()] if solution_str.strip() else []


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | dict[str, Any],
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic RLVR scoring hook invoked by `verl` at the end of a multi-turn rollout."""
    del data_source, extra_info
    scenario = (
        IncidentScenario.model_validate_json(ground_truth)
        if isinstance(ground_truth, str)
        else IncidentScenario.model_validate(ground_truth)
    )

    env = Tier1SnapshotEnvironment()
    env.reset(scenario)

    turns = extract_assistant_turns(solution_str)
    for turn_text in turns:
        outcome = env.step(turn_text)
        if outcome.done:
            break

    episode = env.finalize_episode()
    breakdown = score_episode(scenario, episode)
    return {
        "score": float(breakdown.total),
        "detection_score": breakdown.detection_score,
        "localization_score": breakdown.localization_score,
        "rca_score": breakdown.rca_score,
        "mitigation_score": breakdown.mitigation_score,
        "stage_bonus": breakdown.stage_bonus,
        "safety_penalty": breakdown.safety_penalty,
        "efficiency_penalty": breakdown.efficiency_penalty,
        "safety_violated": float(breakdown.safety_violated),
        "false_positive_action": float(breakdown.false_positive_action),
        "mitigated": float(breakdown.mitigated),
    }
