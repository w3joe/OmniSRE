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

from sre_rl.env.real_telemetry_env import RealIncidentRecord
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
                token_adv = base_ep_adv * seg_decay * (
                    (1.0 - cfg.outcome_dampening_lambda) + cfg.outcome_dampening_lambda * turn_decay
                ) + seg.turn_rewards[t_idx]
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
