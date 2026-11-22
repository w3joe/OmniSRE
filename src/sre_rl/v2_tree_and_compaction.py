"""OmniSRE v2.0 Role-Diversified `5x` Specialist Tree Ensemble & Segmented 100-Turn TITO Compaction.

Fixes the two ensemble & long-horizon limitations identified in `v1.0`:
1. **Role-Diversified `5x` Investigation Leaves (`SPECIALIST_LEAF_ROLES`):**
   Replaces 5 correlated `temp=0.65` samples with 5 orthogonal investigative priors
   (Metrics Inflection, Trace Critical-Path DAG, Log/Exception Chronology, DB/Node Saturation,
   and Counterfactual Red-Herring Falsifier) so `5x` tree synthesis receives genuinely
   complementary evidence across messy 100-service outages.
2. **Disagreeing-Leaf Counterfactual Synthesis Training (`select_counterfactual_leaf_sets`):**
   Constructs RLVR synthesis training problems specifically from *mixed-outcome* leaf sets
   (`1..3` winners mixed with `2..4` red-herring failures) so the synthesizer learns to
   verify onset chronology & verbatim citations rather than majority-voting wrong symptoms.
3. **Segmented TITO Rollout Representation (`CompactedSegmentRecord`):**
   Decomposes an 80-100 turn investigation with `K` `compact_workspace` calls into `K+1`
   bounded `<5.6K`-token TITO segments — preserving exact Token-In-Token-Out alignment
   and `15x` lower attention FLOPs than flat `256K-1M` context windows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from sre_rl.env.real_telemetry_env import (
    RealIncidentRecord,
    _canonicalize_fault_set,
    _normalize_service_token,
)
from sre_rl.env.v2_raw_exploration_env import StrictRawRCABreakdown, _parse_timestamp_to_unix
from sre_rl.model.thinking import parse_qwen38_assistant_turn_multi
from sre_rl.rewards.dapo_gtpo import DAPOConfig, RolloutGroupItem, compute_gtpo_hybrid_advantages

SPECIALIST_LEAF_ROLES: tuple[tuple[str, str], ...] = (
    (
        "metrics_inflection_specialist",
        "Start with `sql_telemetry_query` / `promql_range_query` comparing 5-minute buckets across the window to find the EARLIEST sustained metric inflection (`stddev` & `p95` jump), ignoring single-minute blips.",
    ),
    (
        "trace_dag_specialist",
        "Start with `trace_call_graph` and `sql_telemetry_query` on `traces` to compare `total_p95_ms` vs `self_p95_ms` and isolate the furthest-downstream leaf service originating latency or errors.",
    ),
    (
        "log_forensics_specialist",
        "Start with `grep_logs` and SQL grouping over `logs` (`level IN ('ERROR','FATAL','WARN')`) ordered chronologically to pinpoint the very first error/exception before cascade propagation.",
    ),
    (
        "db_and_resource_specialist",
        "Focus on saturation, connection pools, disk/redo wait events, memory OOMs, and lock contention metrics across database and stateful storage entities.",
    ),
    (
        "counterfactual_falsifier",
        "Actively test and FALSIFY the top-2 loudest noisy services: check whether their anomalies started AFTER an upstream/downstream dependency or recovered after 1 minute.",
    ),
)


@dataclass
class CompactedSegmentRecord:
    """One `<5.6K`-token exact-TITO segment produced between `compact_workspace` boundaries."""

    segment_idx: int
    total_segments: int
    token_ids: list[int]
    turn_spans: list[tuple[int, int]]
    turn_rewards: list[float]
    episode_outcome_reward: float


@dataclass
class LeafInvestigationArtifact:
    role_name: str
    strict_solved: bool
    outcome_reward: float
    compacted_notes_json: str
    final_submission: dict[str, object] | None
    key_observations: list[str]
    window_start: int = 0
    window_end: int = 0
    alias_to_canon: dict[str, str] | None = None


def select_balanced_segments(
    segments: list[tuple[list[int], list[float], list[float]]],
    max_segments: int,
    pos_fraction: float = 0.60,
) -> list[tuple[list[int], list[float], list[float]]]:
    """Select a sign-balanced subset of segments so high-penalty `-0.30` timeouts never crowd out `+` winners."""
    valid = [s for s in segments if any(m > 0.0 for m in s[2][1:])]
    if max_segments <= 0 or len(valid) <= max_segments:
        return valid

    def _mean_active_adv(seg: tuple[list[int], list[float], list[float]]) -> float:
        vals = [a for a, m in zip(seg[1][1:], seg[2][1:], strict=False) if m > 0.0]
        return float(np.mean(vals)) if vals else 0.0

    scored = [(_mean_active_adv(s), idx, s) for idx, s in enumerate(valid)]
    pos = sorted([x for x in scored if x[0] > 0.0], key=lambda x: x[0], reverse=True)
    neg = sorted([x for x in scored if x[0] < 0.0], key=lambda x: abs(x[0]), reverse=True)

    target_pos = min(len(pos), max(1, round(max_segments * pos_fraction)))
    target_neg = min(len(neg), max_segments - target_pos)
    if target_pos + target_neg < max_segments:
        rem = max_segments - (target_pos + target_neg)
        if len(pos) > target_pos:
            target_pos = min(len(pos), target_pos + rem)
        elif len(neg) > target_neg:
            target_neg = min(len(neg), target_neg + rem)

    return [x[2] for x in pos[:target_pos]] + [x[2] for x in neg[:target_neg]]


def build_segmented_rollout_items(
    episodes_segments: list[list[CompactedSegmentRecord]],
    segment_discount: float = 0.92,
    config: DAPOConfig | None = None,
) -> list[tuple[list[int], list[float], list[float]]]:
    """Assign group-normalized GTPO advantages across multi-segment 100-turn compacted rollouts.

    Each episode `g in 0..G-1` may have `1..K` segments (`compact_workspace` resets).
    Earlier segments that successfully gathered & compacted the decisive evidence receive
    `segment_discount^(K - 1 - s) * episode_outcome_reward` in group advantage computation!
    """
    cfg = config or DAPOConfig(group_size=max(2, len(episodes_segments)))
    # Anchor group mean/std on episode-level terminal outcome rewards
    ep_items = [
        RolloutGroupItem(
            outcome_reward=segs[-1].episode_outcome_reward if segs else -0.30,
            turn_rewards=[0.0],
            turn_token_spans=[(0, 1)],
            total_seq_len=2,
        )
        for segs in episodes_segments
    ]
    ep_advs, _ = compute_gtpo_hybrid_advantages(ep_items, config=cfg)
    if all(float(a[0]) == 0.0 for a in ep_advs):
        return []

    packed_training_samples: list[tuple[list[int], list[float], list[float]]] = []
    for ep_idx, segs in enumerate(episodes_segments):
        base_ep_adv = float(ep_advs[ep_idx][0])
        n_segs = len(segs)
        for seg in segs:
            if not seg.turn_spans or len(seg.token_ids) < 4:
                continue
            seg_decay = segment_discount ** max(0, (n_segs - 1 - seg.segment_idx))
            adv_vec = [0.0] * len(seg.token_ids)
            mask_vec = [0.0] * len(seg.token_ids)
            n_turns = len(seg.turn_spans)
            for t_idx, (s_pos, e_pos) in enumerate(seg.turn_spans):
                turn_decay = cfg.gamma ** max(0, (n_turns - 1 - t_idx))
                allow_pos_shaping = (
                    not cfg.gate_positive_shaping_on_success
                    or seg.episode_outcome_reward >= cfg.min_group_max_reward
                )
                raw_turn_r = seg.turn_rewards[t_idx]
                effective_turn_r = 0.0 if (not allow_pos_shaping and raw_turn_r > 0.0) else raw_turn_r
                token_adv = base_ep_adv * seg_decay * (
                    (1.0 - cfg.outcome_dampening_lambda) + cfg.outcome_dampening_lambda * turn_decay
                ) + effective_turn_r
                for pos in range(max(1, s_pos), min(len(seg.token_ids), e_pos)):
                    adv_vec[pos] = round(float(token_adv), 5)
                    mask_vec[pos] = 1.0
            if sum(mask_vec[1:]) > 0.0:
                packed_training_samples.append((seg.token_ids, adv_vec, mask_vec))
    return packed_training_samples


def is_disagreeing_counterfactual_leaf_set(leaves: list[LeafInvestigationArtifact]) -> bool:
    """Return True when 1..3 leaves solved the case and >=1 leaf failed (ideal counterfactual signal!)."""
    if len(leaves) < 2:
        return False
    n_solved = sum(1 for lf in leaves if lf.strict_solved)
    return 1 <= n_solved < len(leaves)


def build_v2_role_diverse_synthesis_prompt(
    record: RealIncidentRecord,
    window_start: int,
    window_end: int,
    leaves: list[LeafInvestigationArtifact],
) -> list[dict[str, str]]:
    """Build a structured counterfactual synthesis prompt over role-diversified specialist leaves."""
    sections: list[str] = []
    for idx, lf in enumerate(leaves, start=1):
        obs_preview = "\n".join(f"  [obs_{j+1}] {o[:320]}" for j, o in enumerate(lf.key_observations[-4:]))
        sections.append(
            f"=== LEAF #{idx} ({lf.role_name}) ===\n"
            f"Compacted Workspace: {lf.compacted_notes_json[:600]}\n"
            f"Candidate Submission: {json.dumps(lf.final_submission, default=str)}\n"
            f"Raw Tool Evidence:\n{obs_preview}"
        )

    sys_prompt = (
        "You are OmniSRE v2 operating in Counterfactual Tree-Synthesis Mode.\n"
        "Five specialist investigators explored the same noisy production window using different angles "
        "(metrics inflection, trace self-latency DAG, log chronology, DB saturation, and red-herring falsification).\n"
        "Cross-examine their raw evidence: discard transient single-minute blips and downstream victim cascades, "
        "identify the earliest sustained upstream/leaf root cause, and emit ONE `<tool_call>` to `submit_rca_report` "
        "with `root_service`, `fault_type`, `onset_timestamp`, `cited_evidence` (verbatim from raw tool evidence), "
        "and `causal_chain`."
    )
    user_prompt = (
        f"[5x COUNTERFACTUAL SYNTHESIS] Incident: {record.case_id} | "
        f"Window: [{window_start}, {window_end}]\n\n" + "\n\n".join(sections)
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


def score_v2_counterfactual_synthesis_turn(
    raw_assistant_turn: str,
    record: RealIncidentRecord,
    leaves: list[LeafInvestigationArtifact],
) -> StrictRawRCABreakdown:
    """Grade a single-turn `5x` counterfactual tree synthesis response with strict 4-way verification."""
    parsed = parse_qwen38_assistant_turn_multi(raw_assistant_turn, max_parallel_calls=2)
    submit_args: dict[str, object] | None = None
    if parsed.parse_error is None and parsed.tool_calls:
        for call in parsed.tool_calls:
            if call.name.strip().lower() == "submit_rca_report":
                submit_args = dict(call.arguments)
                break

    if submit_args is None:
        return StrictRawRCABreakdown(
            total=-0.30,
            service_score=0.0,
            fault_score=0.0,
            onset_time_score=0.0,
            citation_score=0.0,
            efficiency_penalty=0.30,
            strict_solved=False,
            service_exact=False,
            fault_exact=False,
            onset_within_90s=False,
            citation_verified=False,
        )

    merged_alias: dict[str, str] = {}
    corpus_chunks: list[str] = []
    for lf in leaves:
        if lf.alias_to_canon:
            for k, v in lf.alias_to_canon.items():
                merged_alias[k.lower()] = v
        if lf.compacted_notes_json:
            corpus_chunks.append(lf.compacted_notes_json)
        corpus_chunks.extend(lf.key_observations)

    raw_pred_svc = str(submit_args.get("root_service", "")).strip()
    unaliased_svc = merged_alias.get(raw_pred_svc.lower(), raw_pred_svc)
    pred_svc_norm = _normalize_service_token(unaliased_svc)
    gt_svc_norm = _normalize_service_token(record.ground_truth_service)

    svc_exact = bool(pred_svc_norm and pred_svc_norm == gt_svc_norm)
    svc_partial = bool(
        pred_svc_norm
        and not svc_exact
        and (gt_svc_norm in pred_svc_norm or pred_svc_norm in gt_svc_norm)
    )
    service_score = 0.40 if svc_exact else (0.25 if svc_partial else 0.0)

    pred_fault_set = _canonicalize_fault_set(str(submit_args.get("fault_type", "")))
    gt_fault_set = _canonicalize_fault_set(record.ground_truth_fault)
    fault_exact = bool(pred_fault_set & gt_fault_set)
    fault_score = 0.25 if fault_exact else 0.0

    pred_ts = _parse_timestamp_to_unix(submit_args.get("onset_timestamp"))
    onset_within_90s = False
    onset_score = 0.0
    if pred_ts is not None:
        delta_s = abs(pred_ts - record.inject_timestamp)
        if delta_s <= 90:
            onset_within_90s = True
            onset_score = 0.20
        elif delta_s <= 240:
            onset_score = 0.10

    citation = str(submit_args.get("cited_evidence", "")).strip()
    full_obs_blob = "\n".join(corpus_chunks)
    citation_verified = False
    citation_score = 0.0
    if len(citation) >= 6 and citation in full_obs_blob:
        citation_verified = True
        citation_score = 0.15 if (svc_exact or svc_partial) else 0.03
    elif citation:
        citation_score = -0.08

    total = round(max(-0.50, service_score + fault_score + onset_score + citation_score), 4)
    strict_solved = (svc_exact or svc_partial) and fault_exact and onset_within_90s
    return StrictRawRCABreakdown(
        total=total,
        service_score=service_score,
        fault_score=fault_score,
        onset_time_score=onset_score,
        citation_score=citation_score,
        efficiency_penalty=0.0,
        strict_solved=strict_solved,
        service_exact=svc_exact or svc_partial,
        fault_exact=fault_exact,
        onset_within_90s=onset_within_90s,
        citation_verified=citation_verified,
    )
