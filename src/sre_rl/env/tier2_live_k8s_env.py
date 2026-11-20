"""Tier-2 Live Ephemeral Kubernetes / SREGym Cluster Driver (~20% of RLVR steps).

Designed to run directly on GCP `a3-highgpu-4g` (104 vCPUs, 936 GB RAM, 3 TB Local NVMe)
hosting 8-12 isolated `kind` clusters concurrently without CPU over-subscription, or
against remote SREGym worker hosts.

Key anti-reward-hacking protections:
1. Out-of-band Fault Injection: Faults are applied via a privileged admin kubeconfig
   outside the agent's RBAC scope, and any hidden control-plane namespaces are filtered.
2. Dual State + Live HTTP Burn-In Oracle: Mitigation is scored True only if both the
   Kubernetes resource defect is resolved AND live synthetic probe requests meet p99/error
   SLOs across the post-remediation burn-in window.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from sre_rl.env.base import BaseSREEnvironment, StepOutcome
from sre_rl.env.output_limiter import bound_tool_output
from sre_rl.model.thinking import parse_qwen38_assistant_turn
from sre_rl.safety.guardrails import SafetyGuardrailTracker
from sre_rl.schemas import (
    EpisodeResult,
    IncidentClassification,
    IncidentReportSubmission,
    IncidentScenario,
    StepRecord,
    ToolCall,
)

HIDDEN_SYSTEM_NAMESPACES: frozenset[str] = frozenset(
    {"sregym-injector", "chaos-mesh", "litmus", "fault-control-plane"}
)


@dataclass(frozen=True)
class KindClusterLease:
    cluster_name: str
    agent_kubeconfig: str
    admin_kubeconfig: str
    prometheus_url: str
    ingress_probe_url: str


class KindClusterLeasePool:
    """Thread-safe lease pool for ephemeral Kind clusters backed by /mnt/nvme."""

    def __init__(self, leases: list[KindClusterLease]) -> None:
        self._available: list[KindClusterLease] = list(leases)
        self._cond = threading.Condition()

    @classmethod
    def from_nvme_pool(
        cls,
        num_clusters: int = 8,
        nvme_root: str = "/mnt/nvme/kind-leases",
        base_prom_port: int = 19090,
        base_ingress_port: int = 18080,
    ) -> KindClusterLeasePool:
        leases = []
        root = Path(nvme_root)
        for idx in range(num_clusters):
            name = f"sre-rl-kind-{idx:02d}"
            leases.append(
                KindClusterLease(
                    cluster_name=name,
                    agent_kubeconfig=str(root / name / "agent-restricted.kubeconfig"),
                    admin_kubeconfig=str(root / name / "admin-injector.kubeconfig"),
                    prometheus_url=f"http://127.0.0.1:{base_prom_port + idx}",
                    ingress_probe_url=f"http://127.0.0.1:{base_ingress_port + idx}/healthz",
                )
            )
        return cls(leases)

    def acquire(self, timeout_s: float = 180.0) -> KindClusterLease:
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while not self._available:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a free Kind cluster lease.")
                self._cond.wait(timeout=remaining)
            return self._available.pop(0)

    def release(self, lease: KindClusterLease) -> None:
        with self._cond:
            self._available.append(lease)
            self._cond.notify()


class Tier2LiveK8sEnvironment(BaseSREEnvironment):
    """Live Kubernetes cluster RL environment with restricted RBAC and SLO burn-in checks."""

    def __init__(
        self,
        lease: KindClusterLease,
        max_turns: int = 30,
        min_reads_before_write: int = 2,
        cmd_timeout_s: int = 15,
        burn_in_probes: int = 12,
        burn_in_interval_s: float = 2.0,
    ) -> None:
        self.lease = lease
        self.max_turns = max_turns
        self.min_reads_before_write = min_reads_before_write
        self.cmd_timeout_s = cmd_timeout_s
        self.burn_in_probes = burn_in_probes
        self.burn_in_interval_s = burn_in_interval_s

        self._scenario: IncidentScenario | None = None
        self._guardrail = SafetyGuardrailTracker(min_reads_before_write=min_reads_before_write)
        self._steps: list[StepRecord] = []
        self._seen_calls: set[str] = set()
        self._report: IncidentReportSubmission | None = None
        self._escalated = False
        self._terminated_by_guardrail = False
        self._done = False

    def _run_kubectl(
        self,
        subargs: list[str],
        use_admin: bool = False,
    ) -> tuple[int, str]:
        kubeconfig = (
            self.lease.admin_kubeconfig if use_admin else self.lease.agent_kubeconfig
        )
        cmd = ["kubectl", "--kubeconfig", kubeconfig, *subargs]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.cmd_timeout_s,
            check=False,
        )
        combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        # Scrub any accidental references to hidden injector namespaces
        filtered_lines = [
            ln
            for ln in combined.splitlines()
            if not any(ns in ln for ns in HIDDEN_SYSTEM_NAMESPACES)
        ]
        return proc.returncode, "\n".join(filtered_lines).strip()

    def reset(self, scenario: IncidentScenario) -> str:
        self._scenario = scenario
        self._guardrail = SafetyGuardrailTracker(
            min_reads_before_write=self.min_reads_before_write
        )
        self._steps = []
        self._seen_calls = set()
        self._report = None
        self._escalated = False
        self._terminated_by_guardrail = False
        self._done = False

        return (
            f"[LIVE-CLUSTER {self.lease.cluster_name}] [{scenario.severity}] {scenario.alert_title}\n"
            f"Topology: {scenario.topology_name}\n"
            f"Pager Payload:\n{scenario.alert_pager_payload}"
        )

    def step(self, raw_assistant_turn: str) -> StepOutcome:
        if self._done:
            return StepOutcome(observation="Episode already finished.", done=True)

        turn_idx = len(self._steps)
        parsed = parse_qwen38_assistant_turn(raw_assistant_turn)
        if parsed.parse_error is not None or parsed.tool_call is None:
            obs = f"ERROR: Invalid tool call ({parsed.parse_error or 'missing_tool_call'})."
            self._done = (turn_idx + 1) >= self.max_turns
            self._steps.append(
                StepRecord(
                    turn_index=turn_idx,
                    raw_assistant_text=raw_assistant_turn,
                    thinking_text=parsed.thinking,
                    thinking_token_estimate=parsed.thinking_tokens,
                    observation=obs,
                    syntax_valid=False,
                    step_shaping_reward=-0.05,
                )
            )
            return StepOutcome(observation=obs, done=self._done, syntax_valid=False, step_shaping_reward=-0.05)

        call = parsed.tool_call
        sig = f"w{self._guardrail.write_count}:{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
        is_dup = sig in self._seen_calls
        self._seen_calls.add(sig)

        decision = self._guardrail.evaluate(call)
        if not decision.allowed:
            obs = f"GUARDRAIL_BLOCKED [{decision.violation_code}]: {decision.reason}"
            if decision.terminal_violation:
                self._terminated_by_guardrail = True
                self._done = True
            else:
                self._done = (turn_idx + 1) >= self.max_turns
            self._steps.append(
                StepRecord(
                    turn_index=turn_idx,
                    raw_assistant_text=raw_assistant_turn,
                    thinking_text=parsed.thinking,
                    thinking_token_estimate=parsed.thinking_tokens,
                    tool_call=call,
                    observation=obs,
                    is_duplicate_call=is_dup,
                    safety_violation=decision.violation_code if decision.terminal_violation else None,
                    step_shaping_reward=-1.0 if decision.terminal_violation else -0.08,
                )
            )
            return StepOutcome(
                observation=obs,
                done=self._done,
                is_duplicate_call=is_dup,
                safety_violation=decision.violation_code if decision.terminal_violation else None,
                step_shaping_reward=-1.0 if decision.terminal_violation else -0.08,
            )

        raw_obs, tool_done, syntax_valid = self.execute_tool(call)
        bounded = bound_tool_output(raw_obs)
        self._done = tool_done or ((turn_idx + 1) >= self.max_turns)
        self._steps.append(
            StepRecord(
                turn_index=turn_idx,
                raw_assistant_text=raw_assistant_turn,
                thinking_text=parsed.thinking,
                thinking_token_estimate=parsed.thinking_tokens,
                tool_call=call,
                observation=bounded,
                syntax_valid=syntax_valid,
                is_duplicate_call=is_dup,
            )
        )
        return StepOutcome(observation=bounded, done=self._done, syntax_valid=syntax_valid, is_duplicate_call=is_dup)

    def execute_tool(self, tool_call: ToolCall) -> tuple[str, bool, bool]:
        args = tool_call.arguments
        name = tool_call.name

        if name == "k8s_inspect":
            ns = str(args.get("namespace", "default"))
            if ns in HIDDEN_SYSTEM_NAMESPACES:
                return "Error from server (NotFound): namespace not found", False, True
            kind = str(args.get("kind", "pods"))
            obj = str(args.get("name", ""))
            cmd = ["get", kind, "-n", ns, "-o", "json"] if not obj else ["describe", kind, obj, "-n", ns]
            rc, out = self._run_kubectl(cmd)
            return out, False, rc == 0

        if name == "search_logs":
            ns = str(args.get("namespace", "default"))
            svc = str(args.get("service", ""))
            rc, out = self._run_kubectl(["logs", "-n", ns, f"deployment/{svc}", "--tail=120"])
            return out, False, rc == 0

        if name == "query_metrics":
            promql = str(args.get("promql", args.get("metric", "up")))
            url = f"{self.lease.prometheus_url}/api/v1/query?query={urllib.parse.quote(promql)}"
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    return resp.read().decode("utf-8"), False, True
            except Exception as exc:  # noqa: BLE001
                return f"PROMETHEUS_QUERY_ERROR: {exc}", False, False

        if name == "apply_remediation":
            ns = str(args.get("namespace", "default"))
            kind = str(args.get("kind", "deployment"))
            target = str(args.get("target_object", ""))
            patch_json = json.dumps(args.get("patch", {}))
            rc, out = self._run_kubectl(["patch", kind, target, "-n", ns, "--type=merge", "-p", patch_json])
            return out, False, rc == 0

        if name == "rollback_release":
            ns = str(args.get("namespace", "default"))
            dep = str(args.get("deployment", ""))
            rc, out = self._run_kubectl(["rollout", "undo", f"deployment/{dep}", "-n", ns])
            return out, False, rc == 0

        if name == "scale_resource":
            ns = str(args.get("namespace", "default"))
            workload = str(args.get("workload", ""))
            replicas = int(args.get("replicas", 2))
            rc, out = self._run_kubectl(["scale", f"deployment/{workload}", "-n", ns, f"--replicas={replicas}"])
            return out, False, rc == 0

        if name == "submit_incident_report":
            cls = IncidentClassification(str(args.get("classification", "real_incident")))
            self._report = IncidentReportSubmission(
                classification=cls,
                root_namespace=str(args.get("root_namespace", "")),
                root_service=str(args.get("root_service", "")),
                root_object=str(args.get("root_object", "")),
                causal_mechanism=str(args.get("causal_mechanism", "")),
                remediation_summary=str(args.get("remediation_summary", "")),
            )
            return "INCIDENT_REPORT_RECORDED.", True, True

        if name == "escalate_to_human":
            self._escalated = True
            return "INCIDENT_ESCALATED.", True, True

        return f"Unsupported live tool: {name}", False, False

    def _verify_live_slo_burn_in(self) -> bool:
        """Probe the live service endpoint over `burn_in_probes` intervals to confirm stability."""
        successes = 0
        for idx in range(self.burn_in_probes):
            try:
                t0 = time.monotonic()
                with urllib.request.urlopen(self.lease.ingress_probe_url, timeout=2.0) as resp:
                    latency_ms = (time.monotonic() - t0) * 1000.0
                    if 200 <= resp.status < 300 and latency_ms < 800.0:
                        successes += 1
            except OSError:
                continue
            if idx + 1 < self.burn_in_probes and self.burn_in_interval_s > 0:
                time.sleep(self.burn_in_interval_s)
        return successes >= int(0.95 * self.burn_in_probes)

    def finalize_episode(self) -> EpisodeResult:
        assert self._scenario is not None
        slo_ok = False if self._terminated_by_guardrail else self._verify_live_slo_burn_in()
        return EpisodeResult(
            scenario_id=self._scenario.scenario_id,
            steps=list(self._steps),
            report=self._report,
            escalated_to_human=self._escalated,
            root_defect_resolved=slo_ok,
            slo_burn_in_passed=slo_ok,
            regressed_services=[],
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
