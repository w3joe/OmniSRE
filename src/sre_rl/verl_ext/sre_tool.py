"""Stateful Multi-Turn SRE Tool Router compatible with `verl` SGLang AgentLoop / BaseTool.

Each rollout trajectory gets an isolated `instance_id` backed by either:
- `Tier1SnapshotEnvironment` (sub-ms deterministic telemetry replay, default for 80% of steps)
- `Tier2LiveK8sEnvironment` (leased Kind cluster from `/mnt/nvme` pool for Stage 3 live RL)
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

from sre_rl.env.base import BaseSREEnvironment
from sre_rl.env.tier1_snapshot_env import Tier1SnapshotEnvironment
from sre_rl.env.tier2_live_k8s_env import (
    KindClusterLease,
    KindClusterLeasePool,
    Tier2LiveK8sEnvironment,
)
from sre_rl.rewards.verifiable_reward import score_episode
from sre_rl.schemas import IncidentScenario, ToolCall


class SREMultiTurnToolRouter:
    """Manages per-rollout environment instances across concurrent SGLang async workers."""

    def __init__(
        self,
        mode: str = "tier1",
        max_turns: int = 24,
        live_pool: KindClusterLeasePool | None = None,
    ) -> None:
        self.mode = mode
        self.max_turns = max_turns
        self.live_pool = live_pool
        self._lock = threading.Lock()
        self._envs: dict[str, tuple[IncidentScenario, BaseSREEnvironment, KindClusterLease | None]] = {}

    async def create(
        self,
        instance_id: str | None = None,
        ground_truth: str | dict[str, Any] | None = None,
        **_: Any,
    ) -> str:
        iid = instance_id or str(uuid.uuid4())
        if isinstance(ground_truth, str):
            scenario = IncidentScenario.model_validate_json(ground_truth)
        elif isinstance(ground_truth, dict):
            scenario = IncidentScenario.model_validate(ground_truth)
        else:
            raise TypeError("`ground_truth` must contain serialized `IncidentScenario`.")

        lease: KindClusterLease | None = None
        if self.mode == "tier2":
            if self.live_pool is None:
                self.live_pool = KindClusterLeasePool.from_nvme_pool()
            lease = self.live_pool.acquire()
            env: BaseSREEnvironment = Tier2LiveK8sEnvironment(
                lease=lease, max_turns=self.max_turns
            )
        else:
            env = Tier1SnapshotEnvironment(max_turns=self.max_turns)

        env.reset(scenario)
        with self._lock:
            self._envs[iid] = (scenario, env, lease)
        return iid

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        tool_name: str | None = None,
        **_: Any,
    ) -> tuple[str, float, dict[str, Any]]:
        """Execute a structured tool invocation and return `(observation, step_reward, meta)`."""
        with self._lock:
            scenario, env, _ = self._envs[instance_id]

        resolved_name = tool_name or str(parameters.get("name", ""))
        resolved_args = parameters.get("arguments", parameters) if "name" in parameters else parameters
        raw_turn = (
            f"<think>tool_dispatch</think>\n"
            f"<tool_call>{ToolCall(name=resolved_name, arguments=resolved_args).model_dump_json()}</tool_call>"
        )
        outcome = env.step(raw_turn)
        meta = {
            "done": outcome.done,
            "syntax_valid": outcome.syntax_valid,
            "is_duplicate_call": outcome.is_duplicate_call,
            "safety_violation": outcome.safety_violation,
            "scenario_id": scenario.scenario_id,
        }
        return outcome.observation, outcome.step_shaping_reward, meta

    async def calc_reward(self, instance_id: str, **_: Any) -> float:
        with self._lock:
            scenario, env, _ = self._envs[instance_id]
        ep = env.finalize_episode()
        breakdown = score_episode(scenario, ep)
        return breakdown.total

    async def release(self, instance_id: str, **_: Any) -> None:
        with self._lock:
            popped = self._envs.pop(instance_id, None)
        if popped is not None:
            _, _, lease = popped
            if lease is not None and self.live_pool is not None:
                self.live_pool.release(lease)
