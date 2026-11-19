"""Tier-1 Deterministic Telemetry-Snapshot Simulator (~80% of RLVR training steps).

Executes multi-turn SRE diagnostic queries (`query_metrics`, `search_logs`, `get_traces`,
`k8s_inspect`, `host_diag`, `git_recent_changes`, `search_runbooks`) in sub-millisecond
wall-clock time against frozen incident snapshots, while simulating realistic stateful
transitions for write actions (`apply_remediation`, `rollback_release`, `scale_resource`,
`restart_workload`) including:
- Persistent misconfigurations that re-crash if pods are blindly restarted
- Metastable retry storms that require both root-fix AND capacity/drain stabilization
- Transactional No-Regression (TNR) tracking if healthy bystander services are mutated
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from sre_rl.env.base import BaseSREEnvironment, StepOutcome
from sre_rl.env.output_limiter import bound_tool_output
from sre_rl.model.thinking import parse_qwen38_assistant_turn
from sre_rl.safety.guardrails import SafetyGuardrailTracker
from sre_rl.schemas import (
    EpisodeResult,
    FaultCategory,
    IncidentClassification,
    IncidentReportSubmission,
    IncidentScenario,
    K8sResource,
    StepRecord,
    ToolCall,
)


class Tier1SnapshotEnvironment(BaseSREEnvironment):
    def __init__(
        self,
        max_turns: int = 24,
        min_reads_before_write: int = 2,
        max_output_chars: int = 12_000,
    ) -> None:
        self.max_turns = max_turns
        self.min_reads_before_write = min_reads_before_write
        self.max_output_chars = max_output_chars

        self._scenario: IncidentScenario | None = None
        self._resources: list[K8sResource] = []
        self._guardrail = SafetyGuardrailTracker(min_reads_before_write=min_reads_before_write)
        self._steps: list[StepRecord] = []
        self._seen_calls: set[str] = set()
        self._primary_fixed: bool = False
        self._secondary_stabilized: bool = False
        self._regressed_services: set[str] = set()
        self._report: IncidentReportSubmission | None = None
        self._escalated: bool = False
        self._terminated_by_guardrail: bool = False
        self._done: bool = False

    def reset(self, scenario: IncidentScenario) -> str:
        self._scenario = scenario
        self._resources = copy.deepcopy(scenario.telemetry.k8s_resources)
        self._guardrail = SafetyGuardrailTracker(
            min_reads_before_write=self.min_reads_before_write
        )
        self._steps = []
        self._seen_calls = set()
        self._primary_fixed = (
            scenario.fault.classification != IncidentClassification.REAL_INCIDENT
        )
        self._secondary_stabilized = not scenario.fault.requires_secondary_drain_or_scale
        self._regressed_services = set()
        self._report = None
        self._escalated = False
        self._terminated_by_guardrail = False
        self._done = False

        return (
            f"[{scenario.severity}] Alert: {scenario.alert_title}\n"
            f"Topology: {scenario.topology_name}\n"
            f"Pager Payload:\n{scenario.alert_pager_payload}\n"
            f"Services in topology: {', '.join(sorted(scenario.service_dependencies.keys()))}"
        )

    @property
    def scenario(self) -> IncidentScenario:
        if self._scenario is None:
            raise RuntimeError("Environment must be reset(scenario) before stepping.")
        return self._scenario

    @property
    def is_fully_mitigated(self) -> bool:
        return (
            self._primary_fixed
            and self._secondary_stabilized
            and len(self._regressed_services) == 0
        )

    def step(self, raw_assistant_turn: str) -> StepOutcome:
        if self._done:
            return StepOutcome(observation="Episode already finished.", done=True)

        turn_idx = len(self._steps)
        parsed = parse_qwen38_assistant_turn(raw_assistant_turn)

        if parsed.parse_error is not None or parsed.tool_call is None:
            err_msg = parsed.parse_error or "missing_tool_call"
            obs = (
                f"ERROR: Invalid assistant tool syntax ({err_msg}). "
                "Emit exactly one valid `<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>`."
            )
            done = (turn_idx + 1) >= self.max_turns
            self._done = done
            self._steps.append(
                StepRecord(
                    turn_index=turn_idx,
                    raw_assistant_text=raw_assistant_turn,
                    thinking_text=parsed.thinking,
                    thinking_token_estimate=parsed.thinking_tokens,
                    tool_call=None,
                    observation=obs,
                    syntax_valid=False,
                    step_shaping_reward=-0.05,
                )
            )
            return StepOutcome(
                observation=obs,
                done=done,
                syntax_valid=False,
                step_shaping_reward=-0.05,
            )

        call = parsed.tool_call
        call_sig = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
        # Re-querying read tools AFTER a write action is verification, not a duplicate!
        sig_with_epoch = f"w{self._guardrail.write_count}:{call_sig}"
        is_duplicate = sig_with_epoch in self._seen_calls
        self._seen_calls.add(sig_with_epoch)

        decision = self._guardrail.evaluate(call)
        if not decision.allowed:
            obs = f"GUARDRAIL_BLOCKED [{decision.violation_code}]: {decision.reason}"
            if decision.terminal_violation:
                self._terminated_by_guardrail = True
                self._done = True
                shaping = -1.0
            else:
                self._done = (turn_idx + 1) >= self.max_turns
                shaping = -0.08

            self._steps.append(
                StepRecord(
                    turn_index=turn_idx,
                    raw_assistant_text=raw_assistant_turn,
                    thinking_text=parsed.thinking,
                    thinking_token_estimate=parsed.thinking_tokens,
                    tool_call=call,
                    observation=obs,
                    syntax_valid=not (decision.violation_code or "").startswith("hallucinated_tool"),
                    is_duplicate_call=is_duplicate,
                    safety_violation=decision.violation_code if decision.terminal_violation else None,
                    step_shaping_reward=shaping,
                )
            )
            return StepOutcome(
                observation=obs,
                done=self._done,
                syntax_valid=not (decision.violation_code or "").startswith("hallucinated_tool"),
                is_duplicate_call=is_duplicate,
                safety_violation=decision.violation_code if decision.terminal_violation else None,
                step_shaping_reward=shaping,
            )

        raw_obs, tool_done, syntax_valid = self.execute_tool(call)
        bounded_obs = bound_tool_output(raw_obs, max_chars=self.max_output_chars)
        done = tool_done or ((turn_idx + 1) >= self.max_turns)
        self._done = done

        shaping = self._compute_turn_shaping(
            call=call,
            syntax_valid=syntax_valid,
            is_duplicate=is_duplicate,
        )

        self._steps.append(
            StepRecord(
                turn_index=turn_idx,
                raw_assistant_text=raw_assistant_turn,
                thinking_text=parsed.thinking,
                thinking_token_estimate=parsed.thinking_tokens,
                tool_call=call,
                observation=bounded_obs,
                syntax_valid=syntax_valid,
                is_duplicate_call=is_duplicate,
                step_shaping_reward=shaping,
            )
        )
        return StepOutcome(
            observation=bounded_obs,
            done=done,
            syntax_valid=syntax_valid,
            is_duplicate_call=is_duplicate,
            step_shaping_reward=shaping,
        )

    def _compute_turn_shaping(
        self,
        call: ToolCall,
        syntax_valid: bool,
        is_duplicate: bool,
    ) -> float:
        """Small dense turn shaping signal used by GTPO discounted returns."""
        if not syntax_valid:
            return -0.05
        if is_duplicate:
            return -0.05

        fault = self.scenario.fault
        args_blob = json.dumps(call.arguments).lower()
        if call.name in ("query_metrics", "search_logs", "get_traces", "k8s_inspect", "git_recent_changes"):
            # Reward narrowing onto the true root service/object without rewarding random spam
            if fault.root_service and fault.root_service.lower() in args_blob:
                return 0.03
            return 0.005
        if call.name in ("apply_remediation", "rollback_release", "scale_resource"):
            if self.is_fully_mitigated:
                return 0.12
            if len(self._regressed_services) > 0:
                return -0.25
        return 0.0

    def execute_tool(self, tool_call: ToolCall) -> tuple[str, bool, bool]:
        name = tool_call.name
        args = tool_call.arguments

        if name == "query_metrics":
            return self._tool_query_metrics(args), False, True
        if name == "search_logs":
            return self._tool_search_logs(args), False, True
        if name == "get_traces":
            return self._tool_get_traces(args), False, True
        if name == "k8s_inspect":
            return self._tool_k8s_inspect(args), False, True
        if name == "host_diag":
            return self._tool_host_diag(args), False, True
        if name == "git_recent_changes":
            return self._tool_git_recent_changes(args), False, True
        if name == "search_runbooks":
            return self._tool_search_runbooks(args), False, True
        if name == "apply_remediation":
            return self._tool_apply_remediation(args), False, True
        if name == "rollback_release":
            return self._tool_rollback_release(args), False, True
        if name == "scale_resource":
            return self._tool_scale_resource(args), False, True
        if name == "restart_workload":
            return self._tool_restart_workload(args), False, True
        if name == "submit_incident_report":
            return self._tool_submit_report(args)
        if name == "escalate_to_human":
            self._escalated = True
            reason = str(args.get("reason", "Escalated by SRE agent."))
            return f"INCIDENT_ESCALATED: {reason}", True, True

        return f"Unknown tool: {name}", False, False

    def _tool_query_metrics(self, args: dict[str, Any]) -> str:
        service_filter = str(args.get("service", "")).strip().lower()
        metric_filter = str(args.get("metric", "")).strip().lower()

        rows: list[dict[str, Any]] = []
        for series in self.scenario.telemetry.metrics:
            if service_filter and service_filter not in series.service.lower():
                continue
            if metric_filter and metric_filter not in series.metric.lower():
                continue

            vals = list(series.values)
            if self.is_fully_mitigated and vals:
                # Post-mitigation metric recovery simulation
                baseline = vals[0]
                vals = vals[-3:] + [round(baseline * 1.02, 4), round(baseline * 0.99, 4)]
            elif series.service in self._regressed_services and vals:
                vals = vals + [round(max(vals) * 3.5 + 0.5, 4)]

            latest = vals[-1] if vals else 0.0
            peak = max(vals) if vals else 0.0
            breached = (
                series.slo_threshold is not None and latest > series.slo_threshold
            )
            rows.append(
                {
                    "service": series.service,
                    "metric": series.metric,
                    "unit": series.unit,
                    "recent_window": vals[-6:],
                    "latest": latest,
                    "peak": peak,
                    "slo_threshold": series.slo_threshold,
                    "slo_breached": breached,
                }
            )

        if not rows:
            return json.dumps({"matches": 0, "hint": "Try broader `service` or `metric` filter."})
        return json.dumps({"matches": len(rows), "series": rows}, indent=2)

    def _tool_search_logs(self, args: dict[str, Any]) -> str:
        service_filter = str(args.get("service", "")).strip().lower()
        level_filter = str(args.get("level", "")).strip().upper()
        pattern_str = str(args.get("pattern", "")).strip()

        try:
            regex = re.compile(pattern_str, re.IGNORECASE) if pattern_str else None
        except re.error as exc:
            return f"LOGQL_SYNTAX_ERROR: Invalid regex '{pattern_str}': {exc}"

        matched = []
        for entry in self.scenario.telemetry.logs:
            if service_filter and service_filter not in entry.service.lower():
                continue
            if level_filter and entry.level.upper() != level_filter:
                continue
            if regex and not regex.search(entry.message):
                continue
            matched.append(
                f"{entry.timestamp} [{entry.level}] svc={entry.service} msg={entry.message}"
            )

        if self.is_fully_mitigated and (not level_filter or level_filter == "INFO"):
            matched.append(
                f"2026-11-19T12:15:00Z [INFO] svc={self.scenario.fault.root_service} "
                "msg=Healthchecks passing; error rate returned within SLO budget."
            )

        if not matched:
            return "No matching log entries found."
        return "\n".join(matched[:60])

    def _tool_get_traces(self, args: dict[str, Any]) -> str:
        service_filter = str(args.get("service", "")).strip().lower()
        errors_only = bool(args.get("errors_only", False))

        spans = []
        for sp in self.scenario.telemetry.traces:
            if service_filter and (
                service_filter not in sp.caller.lower()
                and service_filter not in sp.callee.lower()
            ):
                continue
            if errors_only and sp.status_code < 400 and not sp.error:
                continue
            status = 200 if self.is_fully_mitigated else sp.status_code
            dur = round(sp.duration_ms * 0.12, 1) if self.is_fully_mitigated else sp.duration_ms
            spans.append(
                {
                    "trace_id": sp.trace_id,
                    "edge": f"{sp.caller} -> {sp.callee}",
                    "operation": sp.operation,
                    "duration_ms": dur,
                    "status_code": status,
                    "error": None if self.is_fully_mitigated else sp.error,
                }
            )
        return json.dumps({"span_count": len(spans), "spans": spans[:40]}, indent=2)

    def _tool_k8s_inspect(self, args: dict[str, Any]) -> str:
        ns_filter = str(args.get("namespace", "")).strip().lower()
        kind_filter = str(args.get("kind", "")).strip().lower()
        name_filter = str(args.get("name", "")).strip().lower()

        found = []
        for res in self._resources:
            if ns_filter and ns_filter != res.namespace.lower():
                continue
            if kind_filter and kind_filter not in res.kind.lower():
                continue
            if name_filter and name_filter not in res.name.lower():
                continue
            found.append(res.model_dump())

        if not found:
            return json.dumps({"resources": [], "note": "No matching Kubernetes objects."})
        return json.dumps({"resources": found}, indent=2)

    def _tool_host_diag(self, args: dict[str, Any]) -> str:
        service = str(args.get("service", self.scenario.fault.root_service))
        check = str(args.get("check", "summary")).lower()
        fault = self.scenario.fault
        if service == fault.root_service and not self.is_fully_mitigated:
            if fault.category == FaultCategory.STATEFUL_DB:
                return (
                    f"[host_diag:{service}:{check}] iostat await=184.2ms util=99.4%; "
                    "active_locks=47 waiting_txns=128"
                )
            if fault.category == FaultCategory.NETWORK_MESH:
                return (
                    f"[host_diag:{service}:{check}] ss -s: TCP SYN-SENT=412 retrans=28.4%; "
                    "envoy upstream_cx_connect_timeout spiking"
                )
        return f"[host_diag:{service}:{check}] loadavg=0.82 disk_used=41% tcp_retrans=0.02% OK"

    def _tool_git_recent_changes(self, args: dict[str, Any]) -> str:
        service_filter = str(args.get("service", "")).strip().lower()
        commits = [
            c.model_dump()
            for c in self.scenario.telemetry.recent_commits
            if not service_filter or service_filter in c.service.lower()
        ]
        return json.dumps({"recent_commits": commits}, indent=2)

    def _tool_search_runbooks(self, args: dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip().lower()
        tokens = [t for t in re.split(r"\W+", query) if len(t) >= 2]
        scored = []
        for doc in self.scenario.telemetry.runbooks:
            haystack = f"{doc.title} {' '.join(doc.tags)} {doc.body}".lower()
            hits = sum(1 for tok in tokens if tok in haystack) if tokens else 1
            if hits > 0:
                scored.append((hits, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return json.dumps(
            {"runbooks": [d.model_dump() for _, d in scored[:3]]},
            indent=2,
        )

    def _mark_bystander_regression_if_unrelated(self, target_name: str) -> None:
        fault = self.scenario.fault
        if fault.classification != IncidentClassification.REAL_INCIDENT:
            # Any write mutation on a benign false alarm or external escalation causes regression
            self._regressed_services.add(target_name or "cluster")
            return

        target_lower = target_name.lower()
        if (
            fault.root_service.lower() not in target_lower
            and fault.root_object.lower() not in target_lower
        ):
            for healthy in self.scenario.healthy_services:
                if healthy.lower() in target_lower:
                    self._regressed_services.add(healthy)

    def _update_resource_state_on_fix(self) -> None:
        for res in self._resources:
            if (
                self.scenario.fault.root_service.lower() in res.name.lower()
                or self.scenario.fault.root_object.lower() in res.name.lower()
            ) and self.is_fully_mitigated:
                res.status = "Healthy/Running"
                res.events.append("Normal Reconciled: SLO checks passing after remediation.")

    def _tool_apply_remediation(self, args: dict[str, Any]) -> str:
        target = str(args.get("target_object", args.get("service", "")))
        patch = args.get("patch", {})
        if not isinstance(patch, dict) or not target:
            return "ERROR: `apply_remediation` requires `target_object` (str) and `patch` (dict)."

        self._mark_bystander_regression_if_unrelated(target)
        fault = self.scenario.fault

        target_matches = (
            fault.root_object.lower() in target.lower()
            or fault.root_service.lower() in target.lower()
        )
        expected = {
            k: v for k, v in fault.expected_patch_fields.items() if k != "min_recovery_replicas"
        }
        patch_matches = bool(expected) and all(
            str(patch.get(k, "")).lower() == str(v).lower() for k, v in expected.items()
        )

        # Also update resource spec in snapshot state
        for res in self._resources:
            if res.name.lower() == target.lower() or target.lower() in res.name.lower():
                res.spec.update(patch)

        if target_matches and patch_matches:
            self._primary_fixed = True
            self._update_resource_state_on_fix()
            if not self._secondary_stabilized:
                return (
                    f"PATCH_APPLIED on {target}: Root configuration updated, BUT upstream retry queue "
                    "saturation persists (metastable overload). Scale workload or drain retry backlog "
                    "to restore SLOs."
                )
            return f"PATCH_APPLIED on {target}: Reconciliation succeeded; rollout healthy."

        return (
            f"PATCH_APPLIED on {target} with {json.dumps(patch)}: "
            "Resource updated; verify metrics and events to confirm whether SLOs recovered."
        )

    def _tool_rollback_release(self, args: dict[str, Any]) -> str:
        deployment = str(args.get("deployment", args.get("service", "")))
        if not deployment:
            return "ERROR: `rollback_release` requires `deployment`."

        self._mark_bystander_regression_if_unrelated(deployment)
        fault = self.scenario.fault
        if (
            fault.expected_remediation_tool == "rollback_release"
            and (
                fault.root_service.lower() in deployment.lower()
                or fault.root_object.lower() in deployment.lower()
            )
        ):
            self._primary_fixed = True
            self._update_resource_state_on_fix()
            return f"ROLLBACK_COMPLETE: {deployment} reverted to previous stable revision."

        return f"ROLLBACK_EXECUTED on {deployment}: Revision rolled back; check SLO metrics."

    def _tool_scale_resource(self, args: dict[str, Any]) -> str:
        workload = str(args.get("workload", args.get("service", "")))
        replicas = int(args.get("replicas", 0))
        if not workload or replicas <= 0:
            return "ERROR: `scale_resource` requires `workload` and positive `replicas`."

        if replicas == 0:
            self._regressed_services.add(workload)
            return f"WARNING: Scaled {workload} to 0 replicas; service unavailable."

        fault = self.scenario.fault
        if fault.classification != IncidentClassification.REAL_INCIDENT:
            self._regressed_services.add(workload)
            return f"SCALED {workload} to {replicas} replicas (unnecessary capacity mutation)."

        if (
            fault.requires_secondary_drain_or_scale
            and fault.root_service.lower() in workload.lower()
            and replicas >= int(fault.expected_patch_fields.get("min_recovery_replicas", 6))
        ):
            self._secondary_stabilized = True
            self._update_resource_state_on_fix()
            return (
                f"SCALED {workload} to {replicas} replicas: Retry storm backlog draining; "
                "latency stabilizing if primary fault is also patched."
            )

        self._mark_bystander_regression_if_unrelated(workload)
        return f"SCALED {workload} to {replicas} replicas."

    def _tool_restart_workload(self, args: dict[str, Any]) -> str:
        workload = str(args.get("workload", args.get("service", "")))
        if not workload:
            return "ERROR: `restart_workload` requires `workload`."

        self._mark_bystander_regression_if_unrelated(workload)
        fault = self.scenario.fault
        if fault.persistent_across_pod_restart:
            return (
                f"WORKLOAD_RESTARTED: Pods for {workload} recreated, "
                "but persistent fault remains active — new pods are failing SLO checks again."
            )

        if fault.root_service.lower() in workload.lower():
            self._primary_fixed = True
            self._update_resource_state_on_fix()
            return f"WORKLOAD_RESTARTED: Transient state cleared on {workload}."
        return f"WORKLOAD_RESTARTED: {workload} pods cycled."

    def _tool_submit_report(self, args: dict[str, Any]) -> tuple[str, bool, bool]:
        raw_cls = str(args.get("classification", "real_incident")).strip().lower()
        try:
            cls = IncidentClassification(raw_cls)
        except ValueError:
            valid_vals = [c.value for c in IncidentClassification]
            return (
                f"ERROR: Invalid classification '{raw_cls}'. Must be one of {valid_vals}.",
                False,
                False,
            )

        self._report = IncidentReportSubmission(
            classification=cls,
            root_namespace=str(args.get("root_namespace", "")),
            root_service=str(args.get("root_service", "")),
            root_object=str(args.get("root_object", "")),
            causal_mechanism=str(args.get("causal_mechanism", "")),
            remediation_summary=str(args.get("remediation_summary", "")),
        )
        return "INCIDENT_REPORT_RECORDED. Episode complete.", True, True

    def finalize_episode(self) -> EpisodeResult:
        fault = self.scenario.fault
        if fault.classification != IncidentClassification.REAL_INCIDENT:
            # Benign false alarm or external upstream outage: SLO stays clean iff agent didn't break anything
            root_resolved = len(self._regressed_services) == 0
            slo_passed = len(self._regressed_services) == 0
        else:
            root_resolved = self._primary_fixed
            slo_passed = self.is_fully_mitigated

        return EpisodeResult(
            scenario_id=self.scenario.scenario_id,
            steps=list(self._steps),
            report=self._report,
            escalated_to_human=self._escalated,
            root_defect_resolved=root_resolved,
            slo_burn_in_passed=slo_passed,
            regressed_services=sorted(self._regressed_services),
            safety_violations=list(self._guardrail.violations),
            read_count_before_first_write=(
                self._guardrail.read_count_at_first_write
                if self._guardrail.read_count_at_first_write is not None
                else self._guardrail.read_count
            ),
            verified_after_write=self._guardrail.verified_after_write,
            write_actions_taken=self._guardrail.write_count,
            terminated_by_guardrail=self._terminated_by_guardrail,
        )
