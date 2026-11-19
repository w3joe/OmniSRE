"""Core data models for SRE environments, telemetry snapshots, trajectories, and rewards."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FaultCategory(str, Enum):
    K8S_CONFIG = "k8s_config"
    APP_REGRESSION = "app_regression"
    STATEFUL_DB = "stateful_db"
    NETWORK_MESH = "network_mesh"
    METASTABLE_CASCADE = "metastable_cascade"
    BENIGN_FALSE_ALARM = "benign_false_alarm"


class IncidentClassification(str, Enum):
    REAL_INCIDENT = "real_incident"
    BENIGN_FALSE_ALARM = "benign_false_alarm"
    EXTERNAL_ESCALATION = "external_escalation"


class ReasoningEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionCategory(str, Enum):
    READ = "read"
    WRITE = "write"
    TERMINAL = "terminal"


READ_TOOLS: frozenset[str] = frozenset(
    {
        "query_metrics",
        "search_logs",
        "get_traces",
        "k8s_inspect",
        "host_diag",
        "git_recent_changes",
        "search_runbooks",
    }
)

WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "apply_remediation",
        "rollback_release",
        "scale_resource",
        "restart_workload",
    }
)

TERMINAL_TOOLS: frozenset[str] = frozenset(
    {
        "submit_incident_report",
        "escalate_to_human",
    }
)


def classify_tool(tool_name: str) -> ActionCategory:
    if tool_name in READ_TOOLS:
        return ActionCategory.READ
    if tool_name in WRITE_TOOLS:
        return ActionCategory.WRITE
    if tool_name in TERMINAL_TOOLS:
        return ActionCategory.TERMINAL
    raise ValueError(f"Unknown tool: {tool_name}")


class MetricSeries(BaseModel):
    service: str
    metric: str
    unit: str = ""
    values: list[float]
    slo_threshold: float | None = None


class LogEntry(BaseModel):
    timestamp: str
    service: str
    level: str
    message: str
    attributes: dict[str, str] = Field(default_factory=dict)


class TraceSpan(BaseModel):
    trace_id: str
    span_id: str
    caller: str
    callee: str
    operation: str
    duration_ms: float
    status_code: int = 200
    error: str | None = None


class K8sResource(BaseModel):
    kind: str  # Pod, Deployment, ConfigMap, HorizontalPodAutoscaler, NetworkPolicy, StatefulSet
    namespace: str
    name: str
    status: str
    spec: dict[str, Any] = Field(default_factory=dict)
    events: list[str] = Field(default_factory=list)


class GitCommit(BaseModel):
    sha: str
    timestamp: str
    author: str
    service: str
    summary: str
    diff_snippet: str


class RunbookDoc(BaseModel):
    doc_id: str
    title: str
    tags: list[str]
    body: str


class FaultSpec(BaseModel):
    """Ground-truth oracle specification hidden from the agent during episodes."""

    fault_id: str
    category: FaultCategory
    classification: IncidentClassification
    root_namespace: str
    root_service: str
    root_object: str
    causal_mechanism: str
    causal_keywords: list[str] = Field(default_factory=list)
    persistent_across_pod_restart: bool = True
    requires_secondary_drain_or_scale: bool = False
    expected_remediation_tool: str | None = None
    expected_patch_fields: dict[str, Any] = Field(default_factory=dict)
    contains_adversarial_log_injection: bool = False


class TelemetrySnapshot(BaseModel):
    metrics: list[MetricSeries] = Field(default_factory=list)
    logs: list[LogEntry] = Field(default_factory=list)
    traces: list[TraceSpan] = Field(default_factory=list)
    k8s_resources: list[K8sResource] = Field(default_factory=list)
    recent_commits: list[GitCommit] = Field(default_factory=list)
    runbooks: list[RunbookDoc] = Field(default_factory=list)


class IncidentScenario(BaseModel):
    scenario_id: str
    topology_name: str
    severity: str  # SEV1, SEV2, SEV3
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM
    alert_title: str
    alert_pager_payload: str
    service_dependencies: dict[str, list[str]] = Field(default_factory=dict)
    healthy_services: list[str] = Field(default_factory=list)
    telemetry: TelemetrySnapshot
    fault: FaultSpec


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class IncidentReportSubmission(BaseModel):
    classification: IncidentClassification
    root_namespace: str = ""
    root_service: str = ""
    root_object: str = ""
    causal_mechanism: str = ""
    remediation_summary: str = ""


class StepRecord(BaseModel):
    turn_index: int
    raw_assistant_text: str = ""
    thinking_text: str = ""
    thinking_token_estimate: int = 0
    tool_call: ToolCall | None = None
    observation: str = ""
    syntax_valid: bool = True
    is_duplicate_call: bool = False
    safety_violation: str | None = None
    step_shaping_reward: float = 0.0


class RewardBreakdown(BaseModel):
    total: float
    detection_score: float = 0.0
    localization_score: float = 0.0
    rca_score: float = 0.0
    mitigation_score: float = 0.0
    stage_bonus: float = 0.0
    safety_penalty: float = 0.0
    efficiency_penalty: float = 0.0
    safety_violated: bool = False
    false_positive_action: bool = False
    mitigated: bool = False


class EpisodeResult(BaseModel):
    scenario_id: str
    steps: list[StepRecord] = Field(default_factory=list)
    report: IncidentReportSubmission | None = None
    escalated_to_human: bool = False
    root_defect_resolved: bool = False
    slo_burn_in_passed: bool = False
    regressed_services: list[str] = Field(default_factory=list)
    safety_violations: list[str] = Field(default_factory=list)
    read_count_before_first_write: int = 0
    verified_after_write: bool = False
    write_actions_taken: int = 0
    terminated_by_guardrail: bool = False
    reward: RewardBreakdown | None = None
