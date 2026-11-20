"""OmniSRE v5 Enhancements Inspired by ResolveAI SRE-1 Ablations (Nov 2026):

1. **Harness & Schema Randomization (`SampledHarnessConfig`):**
   Randomizes tool aliases, prompt wording, parallel-call hints, and explicit turn/context
   budget headers across 50% of RLVR rollout groups (while keeping 50% in the canonical
   deployment harness) so the policy transfers cleanly across SREGym, OpenRCA, ITBench,
   and real production stacks without overfitting to one prompt template.
2. **Returning Easy/Hard Pool Curriculum (`ReturningPoolCurriculum`):**
   Instead of permanently discarding all-solved or zero-solved prompts, moves `100%`-pass
   prompts into an Easy Pool (returning with half-life ~120 steps to catch regressions)
   and `0%`-pass prompts into a Hard Pool (returning with half-life ~15 steps as the
   policy strengthens), while ranking active candidate groups by reward variance.
3. **Offline Evidence Synthesis & Tree-Ensemble Inference (`SynthesisWorkspace`):**
   Constructs offline synthesis problems (25% of v5 RLVR batches) from saved multi-turn
   evidence bundles and supports `3x` / `5x` hierarchical tree synthesis (`2 + 3 -> 1`)
   to boost single-run incident resolution on hard live benchmarks.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field

from sre_rl.model.thinking import THINKING_TOKEN_BUDGETS, build_sre_system_prompt
from sre_rl.schemas import IncidentScenario, ReasoningEffort, ToolCall

PARALLEL_CALL_HINTS: tuple[str, ...] = (
    "Multiple `<tool_call>` blocks emitted in the same turn execute concurrently.",
    "To minimize MTTR, batch independent read-only checks (e.g. metrics + logs + k8s_inspect) as parallel `<tool_call>` blocks in one turn (up to 6 per turn).",
    "Concurrent tool execution is enabled: emit 2-4 read `<tool_call>` blocks together during initial triage.",
    "",
)

TOOL_ALIAS_BANKS: dict[str, tuple[str, ...]] = {
    "query_metrics": ("query_metrics", "promql_query", "metrics_inspect"),
    "search_logs": ("search_logs", "log_search", "loki_logs"),
    "get_traces": ("get_traces", "trace_spans", "otel_traces"),
    "k8s_inspect": ("k8s_inspect", "kubectl_inspect", "cluster_get"),
    "git_recent_changes": ("git_recent_changes", "deploy_history", "git_commits"),
    "search_runbooks": ("search_runbooks", "lookup_runbook", "sop_search"),
    "apply_remediation": ("apply_remediation", "patch_resource", "k8s_remediate"),
    "rollback_release": ("rollback_release", "rollout_undo", "helm_rollback"),
    "scale_resource": ("scale_resource", "scale_deployment", "k8s_scale"),
    "submit_incident_report": ("submit_incident_report", "submit_rca_report", "close_incident"),
    "escalate_to_human": ("escalate_to_human", "page_human_oncall", "escalate_incident"),
}


@dataclass(frozen=True)
class SampledHarnessConfig:
    is_canonical: bool
    max_turns: int
    max_context_tokens: int
    show_budget_footer: bool
    parallel_hint: str
    canonical_to_alias: dict[str, str] = field(default_factory=dict)
    alias_to_canonical: dict[str, str] = field(default_factory=dict)

    def build_system_prompt(self, effort: ReasoningEffort) -> str:
        base = build_sre_system_prompt(effort)
        if self.is_canonical:
            return base + f"\n{self.parallel_hint}".rstrip()

        # Replace canonical tool names in system prompt with sampled group aliases
        prompt = base
        for canon, alias in self.canonical_to_alias.items():
            prompt = prompt.replace(f"`{canon}`", f"`{alias}`")
        budget_line = (
            f"\nBudget: max_turns={self.max_turns}, max_context_tokens={self.max_context_tokens}."
            if self.show_budget_footer
            else ""
        )
        hint_line = f"\n{self.parallel_hint}" if self.parallel_hint else ""
        return f"{prompt}{budget_line}{hint_line}"

    def format_observation_with_budget(
        self,
        observation: str,
        turn_idx: int,
        used_tokens: int,
    ) -> str:
        if not self.show_budget_footer:
            return observation
        rem_turns = max(0, self.max_turns - (turn_idx + 1))
        rem_toks = max(0, self.max_context_tokens - used_tokens)
        footer = f"\n[budget: turns_remaining={rem_turns}/{self.max_turns} | tokens_remaining={rem_toks}]"
        if rem_turns <= 2:
            footer += "\n[WARNING: Budget nearly exhausted — submit final incident report now.]"
        return observation + footer

    def resolve_tool_call(self, call: ToolCall) -> ToolCall:
        canon = self.alias_to_canonical.get(call.name, call.name)
        if canon == call.name:
            return call
        return ToolCall(name=canon, arguments=dict(call.arguments))


def sample_group_harness(
    rng: random.Random,
    default_max_turns: int = 14,
    default_max_tokens: int = 12_288,
    randomize_prob: float = 0.50,
) -> SampledHarnessConfig:
    """50% canonical deployment harness, 50% randomized tool aliases & dual budgets."""
    if rng.random() >= randomize_prob:
        return SampledHarnessConfig(
            is_canonical=True,
            max_turns=default_max_turns,
            max_context_tokens=default_max_tokens,
            show_budget_footer=True,
            parallel_hint=PARALLEL_CALL_HINTS[1],
        )

    c2a: dict[str, str] = {}
    a2c: dict[str, str] = {}
    for canon, choices in TOOL_ALIAS_BANKS.items():
        picked = rng.choice(choices)
        c2a[canon] = picked
        a2c[picked] = canon

    sampled_turns = rng.choice((10, 12, 14, 16))
    sampled_ctx = rng.choice((6144, 7168, 8192))
    return SampledHarnessConfig(
        is_canonical=False,
        max_turns=sampled_turns,
        max_context_tokens=sampled_ctx,
        show_budget_footer=rng.random() < 0.80,
        parallel_hint=rng.choice(PARALLEL_CALL_HINTS),
        canonical_to_alias=c2a,
        alias_to_canonical=a2c,
    )


@dataclass
class PoolEntry:
    scenario_idx: int
    entered_step: int
    pool_type: str  # "easy" | "hard"


class ReturningPoolCurriculum:
    """SRE-1 style Easy/Hard pool scheduler with exponential half-life return + PODS variance ranking."""

    def __init__(
        self,
        num_scenarios: int,
        easy_half_life_steps: float = 120.0,
        hard_half_life_steps: float = 15.0,
        seed: int = 42,
    ) -> None:
        self.rng = random.Random(seed)
        self.active_queue: list[int] = list(range(num_scenarios))
        self.rng.shuffle(self.active_queue)
        self.easy_pool: dict[int, PoolEntry] = {}
        self.hard_pool: dict[int, PoolEntry] = {}
        self.easy_decay = math.log(2.0) / max(1.0, easy_half_life_steps)
        self.hard_decay = math.log(2.0) / max(1.0, hard_half_life_steps)
        self._cursor = 0

    def _wake_returning_prompts(self, current_step: int) -> None:
        for pool, rate in ((self.easy_pool, self.easy_decay), (self.hard_pool, self.hard_decay)):
            to_return: list[int] = []
            for s_idx, entry in pool.items():
                age = max(1, current_step - entry.entered_step)
                prob_return = 1.0 - math.exp(-rate * (1.0 if age >= 1 else 0.0))
                if self.rng.random() < prob_return:
                    to_return.append(s_idx)
            for s_idx in to_return:
                pool.pop(s_idx, None)
                self.active_queue.append(s_idx)

    def sample_candidates(self, current_step: int, count: int) -> list[int]:
        """Draw `count` candidate scenario indices (recommended 2x oversample for PODS)."""
        self._wake_returning_prompts(current_step)
        if not self.active_queue:
            # Fallback if every single scenario moved out temporarily
            all_pooled = list(self.easy_pool.keys()) + list(self.hard_pool.keys())
            self.rng.shuffle(all_pooled)
            return all_pooled[:count]

        picked: list[int] = []
        while len(picked) < count:
            idx = self.active_queue[self._cursor % len(self.active_queue)]
            self._cursor += 1
            if idx not in self.easy_pool and idx not in self.hard_pool:
                picked.append(idx)
            if self._cursor > len(self.active_queue) * 3 and not picked:
                break
        return picked

    def record_group_outcome(
        self,
        scenario_idx: int,
        current_step: int,
        pass_flags: list[bool],
    ) -> str:
        """Route prompt to Easy Pool (all pass), Hard Pool (none pass), or keep Active."""
        if all(pass_flags):
            self.easy_pool[scenario_idx] = PoolEntry(scenario_idx, current_step, "easy")
            return "easy"
        if not any(pass_flags):
            self.hard_pool[scenario_idx] = PoolEntry(scenario_idx, current_step, "hard")
            return "hard"
        return "active"

    def pool_counts(self) -> dict[str, int]:
        active = sum(
            1
            for idx in set(self.active_queue)
            if idx not in self.easy_pool and idx not in self.hard_pool
        )
        return {
            "active": active,
            "easy_pool": len(self.easy_pool),
            "hard_pool": len(self.hard_pool),
        }


def build_offline_synthesis_prompt(
    scenario: IncidentScenario,
    saved_investigations: list[list[dict[str, str]]],
) -> list[dict[str, str]]:
    """Build an SRE-1 style Offline Evidence Synthesis prompt from 1..N prior traces.

    Used for BOTH:
    1. Training the 25% offline-synthesis RLVR auxiliary task
    2. Inference-time `3x` and `5x` Tree Ensembling (`2 + 3 -> 1`)
    """
    evidence_blocks: list[str] = []
    for run_i, msgs in enumerate(saved_investigations, start=1):
        lines = [f"=== SAVED INVESTIGATION #{run_i} ==="]
        line_no = 1
        for m in msgs[2:]:
            role = m.get("role", "")
            txt = str(m.get("content", "")).strip()
            for raw_ln in txt.splitlines()[:35]:
                lines.append(f"[file:run_{run_i}.log:L{line_no:03d}] ({role}) {raw_ln[:220]}")
                line_no += 1
        evidence_blocks.append("\n".join(lines))

    budget = THINKING_TOKEN_BUDGETS[ReasoningEffort.HIGH]
    sys_prompt = (
        "You are OmniSRE operating in Offline Evidence Synthesis mode.\n"
        f"Keep `<think>` focused (<= {budget} tokens). You have NO live cluster write tools.\n"
        "Cross-examine the saved investigation logs below, falsify any red-herring hypotheses "
        "where counterfactual evidence disagrees, cite `[file:run_X.log:Lnnn]` lines, and emit "
        "exactly one `<tool_call>{\"name\": \"submit_incident_report\", \"arguments\": {...}}</tool_call>` "
        "with `classification`, `root_namespace`, `root_service`, `root_object`, `causal_mechanism`, "
        "and `confidence` in [0.0, 1.0]."
    )
    user_prompt = (
        f"[{scenario.severity}] Alert: {scenario.alert_title}\n"
        f"Pager Payload: {scenario.alert_pager_payload}\n"
        f"Topology Dependencies: {json.dumps(scenario.service_dependencies)}\n\n"
        + "\n\n".join(evidence_blocks)
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
