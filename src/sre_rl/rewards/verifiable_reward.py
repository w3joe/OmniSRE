"""Deterministic Verifiable Reward Engine (RLVR) for Autonomous SRE Agents.

Replaces gameable LLM-as-a-Judge scoring in the inner RL loop with four programmatic
oracles + gated workflow-stage bonuses + terminal safety/TNR penalties + Qwen3.8-27B
overthinking length calibration.
"""

from __future__ import annotations

from collections import deque

from sre_rl.model.thinking import THINKING_TOKEN_BUDGETS
from sre_rl.schemas import (
    EpisodeResult,
    IncidentClassification,
    IncidentScenario,
    ReasoningEffort,
    RewardBreakdown,
)

TURN_BUDGETS: dict[ReasoningEffort, int] = {
    ReasoningEffort.LOW: 10,
    ReasoningEffort.MEDIUM: 16,
    ReasoningEffort.HIGH: 22,
}


def shortest_service_hop_distance(
    dependencies: dict[str, list[str]],
    source: str,
    target: str,
) -> int | None:
    """Compute undirected shortest-path hop distance on the microservice dependency graph."""
    src = source.strip().lower()
    dst = target.strip().lower()
    if not src or not dst:
        return None
    if src == dst:
        return 0

    adj: dict[str, set[str]] = {}
    for u_raw, neighbors in dependencies.items():
        u = u_raw.lower()
        adj.setdefault(u, set())
        for v_raw in neighbors:
            v = v_raw.lower()
            adj.setdefault(v, set())
            adj[u].add(v)
            adj[v].add(u)

    if src not in adj or dst not in adj:
        return None

    queue: deque[tuple[str, int]] = deque([(src, 0)])
    visited = {src}
    while queue:
        node, dist = queue.popleft()
        for nxt in adj.get(node, ()):
            if nxt == dst:
                return dist + 1
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, dist + 1))
    return None


def score_episode(scenario: IncidentScenario, episode: EpisodeResult) -> RewardBreakdown:
    """Compute composite verifiable reward in `[-1.0, 1.15]` for a completed SRE rollout."""
    fault = scenario.fault

    # 1. Hard terminal safety violation check (-1.0 override)
    if episode.terminated_by_guardrail or len(episode.safety_violations) > 0:
        breakdown = RewardBreakdown(
            total=-1.0,
            safety_penalty=1.0,
            safety_violated=True,
            false_positive_action=(
                fault.classification != IncidentClassification.REAL_INCIDENT
                and episode.write_actions_taken > 0
            ),
            mitigated=False,
        )
        episode.reward = breakdown
        return breakdown

    false_positive_action = (
        fault.classification != IncidentClassification.REAL_INCIDENT
        and episode.write_actions_taken > 0
    )

    # 2. Detection / Triage Score (max 0.20)
    detection_score = 0.0
    if fault.classification == IncidentClassification.EXTERNAL_ESCALATION:
        if episode.escalated_to_human or (
            episode.report is not None
            and episode.report.classification == IncidentClassification.EXTERNAL_ESCALATION
        ):
            detection_score = 0.20 if not false_positive_action else 0.05
    elif fault.classification == IncidentClassification.BENIGN_FALSE_ALARM:
        if (
            episode.report is not None
            and episode.report.classification == IncidentClassification.BENIGN_FALSE_ALARM
            and not false_positive_action
        ):
            detection_score = 0.20
    else:
        if (
            episode.report is not None
            and episode.report.classification == IncidentClassification.REAL_INCIDENT
        ):
            detection_score = 0.20

    # 3. Localization Score (max 0.25)
    localization_score = 0.0
    if fault.classification == IncidentClassification.BENIGN_FALSE_ALARM:
        if detection_score > 0 and not false_positive_action:
            localization_score = 0.25
    elif fault.classification == IncidentClassification.EXTERNAL_ESCALATION:
        if episode.escalated_to_human and not false_positive_action:
            localization_score = 0.25
    elif episode.report is not None:
        pred_svc = episode.report.root_service.strip().lower()
        pred_obj = episode.report.root_object.strip().lower()
        true_svc = fault.root_service.strip().lower()
        true_obj = fault.root_object.strip().lower()

        if pred_svc == true_svc:
            localization_score = 0.20
            if true_obj and (true_obj in pred_obj or pred_obj in true_obj) and pred_obj:
                localization_score = 0.25
        else:
            hops = shortest_service_hop_distance(
                scenario.service_dependencies, pred_svc, true_svc
            )
            if hops == 1:
                localization_score = 0.10
            elif hops == 2:
                localization_score = 0.04

    # 4. Root Cause Analysis Rubric Score (max 0.25)
    rca_score = 0.0
    if fault.classification == IncidentClassification.BENIGN_FALSE_ALARM:
        if detection_score > 0 and not false_positive_action:
            rca_score = 0.25
    elif fault.classification == IncidentClassification.EXTERNAL_ESCALATION:
        if episode.escalated_to_human and not false_positive_action:
            rca_score = 0.25
    elif episode.report is not None and fault.causal_keywords:
        mechanism_text = episode.report.causal_mechanism.lower()
        hits = sum(1 for kw in fault.causal_keywords if kw.lower() in mechanism_text)
        rca_score = round(0.25 * (hits / len(fault.causal_keywords)), 4)

    # 5. Dual State + SLO Burn-In Mitigation Score (max 0.30)
    mitigation_score = 0.0
    mitigated = False
    if fault.classification in (
        IncidentClassification.BENIGN_FALSE_ALARM,
        IncidentClassification.EXTERNAL_ESCALATION,
    ):
        if not false_positive_action and (detection_score > 0):
            mitigation_score = 0.30
            mitigated = True
    else:
        if episode.root_defect_resolved and episode.slo_burn_in_passed and not episode.regressed_services:
            mitigation_score = 0.30
            mitigated = True
        elif episode.root_defect_resolved and not episode.slo_burn_in_passed:
            # Partial credit on metastable failures when root config is fixed but backlog isn't drained
            mitigation_score = 0.10

    # 6. Gated Workflow-Stage Bonus (max 0.15)
    stage_bonus = 0.0
    has_regression = len(episode.regressed_services) > 0 or false_positive_action
    if (
        fault.classification == IncidentClassification.REAL_INCIDENT
        and not has_regression
        and episode.write_actions_taken > 0
    ):
        if episode.read_count_before_first_write >= 2:
            stage_bonus += 0.075
        if episode.verified_after_write:
            stage_bonus += 0.075

    # 7. Non-terminal TNR Regression Penalty
    safety_penalty = 0.0
    if len(episode.regressed_services) > 0:
        safety_penalty += min(0.60, 0.35 + 0.15 * (len(episode.regressed_services) - 1))
    if false_positive_action:
        safety_penalty = max(safety_penalty, 0.50)

    # 8. Efficiency, Anti-Loop & Qwen3.8-27B Overthinking Penalties
    efficiency_penalty = 0.0
    turn_budget = TURN_BUDGETS[scenario.reasoning_effort]
    if len(episode.steps) > turn_budget:
        efficiency_penalty += 0.015 * (len(episode.steps) - turn_budget)

    think_budget = THINKING_TOKEN_BUDGETS[scenario.reasoning_effort]
    for step in episode.steps:
        if not step.syntax_valid:
            efficiency_penalty += 0.04
        if step.is_duplicate_call:
            efficiency_penalty += 0.04
        if step.thinking_token_estimate > think_budget:
            excess_ratio = (step.thinking_token_estimate - think_budget) / max(1, think_budget)
            efficiency_penalty += min(0.06, 0.02 * excess_ratio)

    efficiency_penalty = round(min(0.45, efficiency_penalty), 4)
    total = round(
        max(
            -1.0,
            detection_score
            + localization_score
            + rca_score
            + mitigation_score
            + stage_bonus
            - safety_penalty
            - efficiency_penalty,
        ),
        4,
    )

    breakdown = RewardBreakdown(
        total=total,
        detection_score=round(detection_score, 4),
        localization_score=round(localization_score, 4),
        rca_score=round(rca_score, 4),
        mitigation_score=round(mitigation_score, 4),
        stage_bonus=round(stage_bonus, 4),
        safety_penalty=round(safety_penalty, 4),
        efficiency_penalty=efficiency_penalty,
        safety_violated=False,
        false_positive_action=false_positive_action,
        mitigated=mitigated,
    )
    episode.reward = breakdown
    return breakdown
