"""Abstract multi-turn SRE environment contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from sre_rl.schemas import EpisodeResult, IncidentScenario, ToolCall


@dataclass
class StepOutcome:
    observation: str
    done: bool
    syntax_valid: bool = True
    is_duplicate_call: bool = False
    safety_violation: str | None = None
    step_shaping_reward: float = 0.0


class BaseSREEnvironment(ABC):
    """Unified interface shared by Tier-1 Snapshot sims and Tier-2 Live K8s sandboxes."""

    @abstractmethod
    def reset(self, scenario: IncidentScenario) -> str:
        """Reset environment to initial fault state and return initial alert prompt."""

    @abstractmethod
    def step(self, raw_assistant_turn: str) -> StepOutcome:
        """Parse and execute one assistant `<think>` + `<tool_call>` turn."""

    @abstractmethod
    def execute_tool(self, tool_call: ToolCall) -> tuple[str, bool, bool]:
        """Execute a validated ToolCall. Returns `(observation, done, syntax_valid)`."""

    @abstractmethod
    def finalize_episode(self) -> EpisodeResult:
        """Return full episode trace and ground-truth state/SLO oracle checks."""
