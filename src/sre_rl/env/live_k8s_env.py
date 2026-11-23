"""Live Kubernetes (`Kind` / `SREGym`) Interactive RLVR Environment with Anti-Cheat RBAC.

Executes real `kubectl` read & write commands against isolated local `Kind` clusters
provisioned by `scripts/bootstrap_kind_sregym_pool.sh`:
- Fault injection & ground-truth health verification use `admin-injector.kubeconfig`.
- Every agent tool call executes exclusively under `agent-restricted.kubeconfig`
  (least-privilege `sre-rl-agent` ServiceAccount scoped to `app-prod`).
- Enforces read-before-write guardrails (`>=2` read queries prior to any mutation) and
  supports 60% MCP tool-schema randomization so the policy transfers zero-shot to
  arbitrary Kubernetes / SREGym / MCP production tool harnesses.
"""

from __future__ import annotations

import json
import queue
import random
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sre_rl.env.output_limiter import bound_tool_output
from sre_rl.model.thinking import parse_qwen38_assistant_turn_multi
from sre_rl.schemas import ToolCall

FORBIDDEN_K8S_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsregym-injector\b", re.IGNORECASE),
    re.compile(r"\bkube-system\b", re.IGNORECASE),
    re.compile(r"\bchaos-mesh\b", re.IGNORECASE),
    re.compile(r"--all-namespaces|\s-A\b"),
    re.compile(r"\bdelete\s+(ns|namespace|clusterrole|secret)\b", re.IGNORECASE),
)

LIVE_K8S_ALIAS_BANKS: dict[str, tuple[str, ...]] = {
    "k8s_cluster_sweep": ("k8s_cluster_sweep", "kubectl_triage_sweep", "cluster_health_diff"),
    "k8s_get": ("k8s_get", "kubectl_get", "cluster_list_resources"),
    "k8s_describe": ("k8s_describe", "kubectl_describe", "cluster_inspect_object"),
    "k8s_logs": ("k8s_logs", "kubectl_logs", "pod_tail_logs"),
    "k8s_events": ("k8s_events", "kubectl_events", "cluster_recent_events"),
    "k8s_patch": ("k8s_patch", "kubectl_patch", "patch_k8s_resource"),
    "k8s_scale": ("k8s_scale", "kubectl_scale", "scale_deployment"),
    "k8s_rollout_undo": ("k8s_rollout_undo", "kubectl_rollout_undo", "rollback_deployment"),
    "k8s_rollout_restart": ("k8s_rollout_restart", "kubectl_rollout_restart", "restart_deployment"),
    "k8s_delete": ("k8s_delete", "kubectl_delete_object", "remove_blocking_policy"),
    "submit_incident_report": ("submit_incident_report", "close_k8s_incident", "resolve_outage"),
}


@dataclass(frozen=True)
class LiveK8sFaultSpec:
    """One reproducible real Kubernetes fault injected into `app-prod` on a Kind cluster."""

    scenario_id: str
    deployment_name: str
    fault_kind: str  # bad_image | oom_limit | bad_readiness | scaled_zero | bad_configmap | cpu_starve
    symptom_summary: str
    expected_replicas: int = 2


DEFAULT_LIVE_K8S_FAULTS: tuple[LiveK8sFaultSpec, ...] = (
    LiveK8sFaultSpec(
        scenario_id="live-k8s-checkout-bad-image",
        deployment_name="checkout-api",
        fault_kind="bad_image",
        symptom_summary="HTTP 503 spike on checkout-api; new rollout pods stuck non-ready in app-prod.",
        expected_replicas=2,
    ),
    LiveK8sFaultSpec(
        scenario_id="live-k8s-payments-oom",
        deployment_name="payments-worker",
        fault_kind="oom_limit",
        symptom_summary="payments-worker pods repeatedly restarting in app-prod with elevated error budget burn.",
        expected_replicas=2,
    ),
    LiveK8sFaultSpec(
        scenario_id="live-k8s-catalog-readiness",
        deployment_name="catalog-svc",
        fault_kind="bad_readiness",
        symptom_summary="catalog-svc has 0/2 ready endpoints despite containers running; traffic failing.",
        expected_replicas=2,
    ),
    LiveK8sFaultSpec(
        scenario_id="live-k8s-orders-scaled-zero",
        deployment_name="orders-api",
        fault_kind="scaled_zero",
        symptom_summary="orders-api dropped to 0 serving pods after an automated maintenance job.",
        expected_replicas=2,
    ),
    LiveK8sFaultSpec(
        scenario_id="live-k8s-inventory-configmap",
        deployment_name="inventory-api",
        fault_kind="bad_configmap",
        symptom_summary="inventory-api pods crashing at startup after a ConfigMap release in app-prod.",
        expected_replicas=2,
    ),
    LiveK8sFaultSpec(
        scenario_id="live-k8s-search-cpu-starve",
        deployment_name="search-gateway",
        fault_kind="cpu_starve",
        symptom_summary="search-gateway P99 latency spiked >5s with severe CPU throttling in app-prod.",
        expected_replicas=2,
    ),
    LiveK8sFaultSpec(
        scenario_id="live-k8s-auth-selector-drift",
        deployment_name="auth-api",
        fault_kind="bad_service_selector",
        symptom_summary="auth-api Service returning 503 with 0 active Endpoints despite healthy running pods.",
        expected_replicas=2,
    ),
    LiveK8sFaultSpec(
        scenario_id="live-k8s-pricing-targetport-mismatch",
        deployment_name="pricing-engine",
        fault_kind="bad_target_port",
        symptom_summary="pricing-engine Service connections immediately refused despite Ready pods.",
        expected_replicas=2,
    ),
    LiveK8sFaultSpec(
        scenario_id="live-k8s-cart-netpol-deny",
        deployment_name="cart-backend",
        fault_kind="network_policy_block",
        symptom_summary="cart-backend unreachable from callers in app-prod due to packet drops.",
        expected_replicas=2,
    ),
    LiveK8sFaultSpec(
        scenario_id="live-k8s-billing-quota-cap",
        deployment_name="billing-svc",
        fault_kind="resource_quota_cap",
        symptom_summary="billing-svc stuck at 0/2 pods after rollout with ReplicaSet creation failures.",
        expected_replicas=2,
    ),
    LiveK8sFaultSpec(
        scenario_id="live-k8s-feed-rollout-deadlock",
        deployment_name="feed-ranker",
        fault_kind="rolling_update_deadlock",
        symptom_summary="feed-ranker rollout stalled at 0 updated pods with ProgressDeadlineExceeded.",
        expected_replicas=2,
    ),
    LiveK8sFaultSpec(
        scenario_id="live-k8s-shipping-env-shadow",
        deployment_name="shipping-worker",
        fault_kind="env_var_shadowing",
        symptom_summary="shipping-worker crashing despite valid ConfigMap STARTUP_CHECK=ok.",
        expected_replicas=2,
    ),
    LiveK8sFaultSpec(
        scenario_id="live-k8s-ledger-liveness-loop",
        deployment_name="ledger-api",
        fault_kind="bad_liveness",
        symptom_summary="ledger-api pods killed every 10 seconds by kubelet liveness probe.",
        expected_replicas=2,
    ),
)


@dataclass(frozen=True)
class LiveK8sHarnessConfig:
    is_canonical: bool = True
    max_turns: int = 20
    max_context_tokens: int = 12_288
    canonical_to_alias: dict[str, str] = field(default_factory=dict)
    alias_to_canonical: dict[str, str] = field(default_factory=dict)

    def tool_name(self, canonical: str) -> str:
        return self.canonical_to_alias.get(canonical, canonical)

    def resolve_tool_name(self, raw_name: str) -> str:
        norm = raw_name.strip().lower()
        return self.alias_to_canonical.get(norm, norm)

    def build_system_prompt(self) -> str:
        t_sweep = self.tool_name("k8s_cluster_sweep")
        t_get = self.tool_name("k8s_get")
        t_desc = self.tool_name("k8s_describe")
        t_logs = self.tool_name("k8s_logs")
        t_ev = self.tool_name("k8s_events")
        t_patch = self.tool_name("k8s_patch")
        t_scale = self.tool_name("k8s_scale")
        t_undo = self.tool_name("k8s_rollout_undo")
        t_restart = self.tool_name("k8s_rollout_restart")
        t_del = self.tool_name("k8s_delete")
        t_sub = self.tool_name("submit_incident_report")
        return (
            "You are OmniSRE, an autonomous SRE operating inside a LIVE Kubernetes cluster (`namespace: app-prod`).\n"
            "Investigate real pods, services, endpoints, quotas, networkpolicies, events, logs, and deployments, "
            "safely remediate the root cause, verify pods & endpoints recover, and submit the final incident report.\n\n"
            "Available Tools (you may emit 1-4 parallel read `<tool_call>` blocks per turn):\n"
            f"- `{t_sweep}`: `{{}}` — deterministic 1-call pre-triage diff (non-Ready pods, 0-endpoint Services, Warning events, Quotas, NetworkPolicies).\n"
            f"- `{t_get}`: `{{\"resource\": \"pods|deployments|services|endpoints|configmaps|networkpolicies|resourcequotas\", \"name\": \"\"}}` — inspect live K8s objects.\n"
            f"- `{t_desc}`: `{{\"resource\": \"pod|deployment|service|configmap|networkpolicy|resourcequota\", \"name\": \"<name>\"}}` — detailed K8s spec & status.\n"
            f"- `{t_logs}`: `{{\"pod\": \"<pod_or_deploy>\", \"tail\": 40}}` — fetch container logs.\n"
            f"- `{t_ev}`: `{{\"limit\": 20}}` — list recent Kubernetes warning/normal events in `app-prod`.\n"
            f"- `{t_patch}`: `{{\"resource\": \"deployment|service|configmap|resourcequota|networkpolicy\", \"name\": \"<name>\", \"patch_json\": {{...}}}}` — strategic/merge patch a broken resource.\n"
            f"- `{t_scale}`: `{{\"deployment\": \"<name>\", \"replicas\": 2}}` — scale deployment replicas.\n"
            f"- `{t_undo}`: `{{\"deployment\": \"<name>\"}}` — roll back a broken deployment revision.\n"
            f"- `{t_restart}`: `{{\"deployment\": \"<name>\"}}` — rolling restart deployment to reload ConfigMap/Secret credentials.\n"
            f"- `{t_del}`: `{{\"resource\": \"networkpolicy|resourcequota|pod\", \"name\": \"<name>\"}}` — remove a blocking NetworkPolicy/ResourceQuota or stuck Pod in `app-prod`.\n"
            f"- `{t_sub}`: `{{\"root_service\": \"<deployment>\", \"fault_type\": \"<str>\", \"remediation_applied\": \"<str>\", \"verified_healthy\": true}}`\n\n"
            f"SAFETY RULES: Run >=2 read tools before any write tool. Stay strictly in `app-prod`. "
            f"Budget: max_turns={self.max_turns}.\n"
            "Wrap concise reasoning in `<think>...</think>` followed by `<tool_call>{...}</tool_call>`."
        )


def sample_live_k8s_harness(
    rng: random.Random,
    default_max_turns: int = 20,
    randomize_prob: float = 0.60,
) -> LiveK8sHarnessConfig:
    if rng.random() >= randomize_prob:
        return LiveK8sHarnessConfig(is_canonical=True, max_turns=default_max_turns)
    c2a: dict[str, str] = {}
    a2c: dict[str, str] = {}
    for canon, choices in LIVE_K8S_ALIAS_BANKS.items():
        picked = rng.choice(choices)
        c2a[canon] = picked
        a2c[picked.lower()] = canon
    return LiveK8sHarnessConfig(
        is_canonical=False,
        max_turns=rng.choice((14, 18, default_max_turns)),
        canonical_to_alias=c2a,
        alias_to_canonical=a2c,
    )


@dataclass(frozen=True)
class KindClusterLease:
    name: str
    admin_kubeconfig: Path
    agent_kubeconfig: Path


class KindClusterLeasePool:
    """Thread-safe lease pool over Kind cluster directories created by `bootstrap_kind_sregym_pool.sh`."""

    def __init__(self, lease_root: Path) -> None:
        self._pool: queue.Queue[KindClusterLease] = queue.Queue()
        self.leases: list[KindClusterLease] = []
        if lease_root.exists():
            for d in sorted(lease_root.iterdir()):
                adm = d / "admin-injector.kubeconfig"
                agt = d / "agent-restricted.kubeconfig"
                if d.is_dir() and adm.exists() and agt.exists():
                    lease = KindClusterLease(name=d.name, admin_kubeconfig=adm, agent_kubeconfig=agt)
                    self.leases.append(lease)
                    self._pool.put(lease)

    @property
    def size(self) -> int:
        return len(self.leases)

    def acquire(self, timeout_s: float = 120.0) -> KindClusterLease:
        return self._pool.get(timeout=timeout_s)

    def release(self, lease: KindClusterLease) -> None:
        self._pool.put(lease)


@dataclass
class LiveK8sStepOutcome:
    observation: str
    done: bool
    syntax_valid: bool = True
    step_shaping_reward: float = 0.0
    safety_violation: bool = False
    tool_calls_executed: int = 0


@dataclass
class LiveK8sScoreBreakdown:
    total: float
    mitigated: bool
    service_exact: bool
    safety_violation: bool
    ready_replicas: int
    desired_replicas: int


def _run_kubectl(kubeconfig: Path, args: list[str], stdin_text: str | None = None, timeout_s: int = 20) -> tuple[int, str]:
    cmd = ["kubectl", "--kubeconfig", str(kubeconfig), "-n", "app-prod", *args]
    try:
        proc = subprocess.run(
            cmd,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, out.strip()
    except subprocess.TimeoutExpired:
        return 124, "ERROR: kubectl command timed out"
    except FileNotFoundError:
        return 127, "ERROR: kubectl binary not installed"


def _build_healthy_manifest(spec: LiveK8sFaultSpec) -> str:
    """Base healthy Deployment + ConfigMap using standard `busybox:1.36` / `nginx:1.27-alpine`."""
    return json.dumps(
        {
            "apiVersion": "v1",
            "kind": "List",
            "items": [
                {
                    "apiVersion": "v1",
                    "kind": "ConfigMap",
                    "metadata": {"name": f"{spec.deployment_name}-cfg", "namespace": "app-prod"},
                    "data": {"APP_MODE": "production", "STARTUP_CHECK": "ok"},
                },
                {
                    "apiVersion": "v1",
                    "kind": "Service",
                    "metadata": {"name": f"{spec.deployment_name}-svc", "namespace": "app-prod"},
                    "spec": {
                        "selector": {"app": spec.deployment_name},
                        "ports": [{"name": "http", "port": 80, "targetPort": 8080}],
                    },
                },
                {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {"name": spec.deployment_name, "namespace": "app-prod"},
                    "spec": {
                        "replicas": spec.expected_replicas,
                        "selector": {"matchLabels": {"app": spec.deployment_name}},
                        "template": {
                            "metadata": {"labels": {"app": spec.deployment_name}},
                            "spec": {
                                "containers": [
                                    {
                                        "name": "app",
                                        "image": "busybox:1.36",
                                        "command": [
                                            "sh",
                                            "-c",
                                            (
                                                'if [ "$STARTUP_CHECK" != "ok" ]; then echo "FATAL: invalid STARTUP_CHECK=$STARTUP_CHECK"; exit 1; fi; '
                                                'echo "Serving on :8080"; while true; do sleep 5; done'
                                            ),
                                        ],
                                        "envFrom": [{"configMapRef": {"name": f"{spec.deployment_name}-cfg"}}],
                                        "readinessProbe": {
                                            "exec": {"command": ["sh", "-c", "test \"$STARTUP_CHECK\" = ok"]},
                                            "initialDelaySeconds": 1,
                                            "periodSeconds": 3,
                                        },
                                        "resources": {
                                            "requests": {"cpu": "25m", "memory": "32Mi"},
                                            "limits": {"cpu": "200m", "memory": "128Mi"},
                                        },
                                    }
                                ]
                            },
                        },
                    },
                },
            ],
        }
    )


class LiveK8sKindEnvironment:
    """Interactive multi-turn RLVR environment executing real `kubectl` inside an isolated Kind cluster."""

    def __init__(
        self,
        lease: KindClusterLease,
        harness: LiveK8sHarnessConfig | None = None,
        max_output_chars: int = 3200,
    ) -> None:
        self.lease = lease
        self.harness = harness or LiveK8sHarnessConfig()
        self.max_turns = self.harness.max_turns
        self.max_output_chars = max_output_chars

        self._spec: LiveK8sFaultSpec | None = None
        self._turns_taken = 0
        self._read_queries = 0
        self._write_actions = 0
        self._safety_violations = 0
        self._syntax_errors = 0
        self._submission: dict[str, Any] | None = None
        self._done = False

    def reset(self, spec: LiveK8sFaultSpec) -> str:
        self._spec = spec
        self._turns_taken = 0
        self._read_queries = 0
        self._write_actions = 0
        self._safety_violations = 0
        self._syntax_errors = 0
        self._submission = None
        self._done = False

        adm = self.lease.admin_kubeconfig
        # Clean previous workload and apply known-good rev 1 (so rollout undo works cleanly)
        _run_kubectl(adm, ["delete", "deployment", spec.deployment_name, "--ignore-not-found=true"])
        _run_kubectl(adm, ["delete", "service", f"{spec.deployment_name}-svc", "--ignore-not-found=true"])
        _run_kubectl(adm, ["delete", "configmap", f"{spec.deployment_name}-cfg", "--ignore-not-found=true"])
        _run_kubectl(adm, ["delete", "networkpolicy", f"{spec.deployment_name}-deny", "--ignore-not-found=true"])
        _run_kubectl(adm, ["delete", "resourcequota", "app-prod-quota", "--ignore-not-found=true"])
        _run_kubectl(adm, ["apply", "-f", "-"], stdin_text=_build_healthy_manifest(spec))
        time.sleep(2.0)

        # Inject the real fault as rev 2 / live mutation
        if spec.fault_kind == "bad_image":
            patch = {"spec": {"template": {"spec": {"containers": [{"name": "app", "image": "busybox:nonexistent-tag-999"}]}}}}
            _run_kubectl(adm, ["patch", "deployment", spec.deployment_name, "--type=strategic", "-p", json.dumps(patch)])
        elif spec.fault_kind == "oom_limit":
            patch = {"spec": {"template": {"spec": {"containers": [{"name": "app", "resources": {"limits": {"memory": "4Mi"}, "requests": {"memory": "4Mi"}}}]}}}}
            _run_kubectl(adm, ["patch", "deployment", spec.deployment_name, "--type=strategic", "-p", json.dumps(patch)])
        elif spec.fault_kind == "bad_readiness":
            patch = {"spec": {"template": {"spec": {"containers": [{"name": "app", "readinessProbe": {"exec": {"command": ["sh", "-c", "exit 1"]}}}]}}}}
            _run_kubectl(adm, ["patch", "deployment", spec.deployment_name, "--type=strategic", "-p", json.dumps(patch)])
        elif spec.fault_kind == "scaled_zero":
            _run_kubectl(adm, ["scale", "deployment", spec.deployment_name, "--replicas=0"])
        elif spec.fault_kind == "bad_configmap":
            _run_kubectl(adm, ["patch", "configmap", f"{spec.deployment_name}-cfg", "--type=merge", "-p", json.dumps({"data": {"STARTUP_CHECK": "corrupted-v2"}})])
            _run_kubectl(adm, ["rollout", "restart", "deployment", spec.deployment_name])
        elif spec.fault_kind == "cpu_starve":
            patch = {"spec": {"template": {"spec": {"containers": [{"name": "app", "resources": {"limits": {"cpu": "1m"}, "requests": {"cpu": "1m"}}}]}}}}
            _run_kubectl(adm, ["patch", "deployment", spec.deployment_name, "--type=strategic", "-p", json.dumps(patch)])
        elif spec.fault_kind == "bad_service_selector":
            _run_kubectl(adm, ["patch", "service", f"{spec.deployment_name}-svc", "--type=merge", "-p", json.dumps({"spec": {"selector": {"app": f"{spec.deployment_name}-wrong"}}})])
        elif spec.fault_kind == "bad_target_port":
            _run_kubectl(adm, ["patch", "service", f"{spec.deployment_name}-svc", "--type=merge", "-p", json.dumps({"spec": {"ports": [{"name": "http", "port": 80, "targetPort": 19999}]}})])
        elif spec.fault_kind == "network_policy_block":
            np_doc = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {"name": f"{spec.deployment_name}-deny", "namespace": "app-prod"},
                "spec": {"podSelector": {"matchLabels": {"app": spec.deployment_name}}, "policyTypes": ["Ingress"], "ingress": []},
            }
            _run_kubectl(adm, ["apply", "-f", "-"], stdin_text=json.dumps(np_doc))
        elif spec.fault_kind == "resource_quota_cap":
            rq_doc = {
                "apiVersion": "v1",
                "kind": "ResourceQuota",
                "metadata": {"name": "app-prod-quota", "namespace": "app-prod"},
                "spec": {"hard": {"pods": "0"}},
            }
            _run_kubectl(adm, ["apply", "-f", "-"], stdin_text=json.dumps(rq_doc))
            _run_kubectl(adm, ["rollout", "restart", "deployment", spec.deployment_name])
        elif spec.fault_kind == "rolling_update_deadlock":
            patch = {
                "spec": {
                    "strategy": {"type": "RollingUpdate", "rollingUpdate": {"maxSurge": 0, "maxUnavailable": 0}},
                    "template": {"metadata": {"annotations": {"rollout-bump": "v2"}}},
                }
            }
            _run_kubectl(adm, ["patch", "deployment", spec.deployment_name, "--type=merge", "-p", json.dumps(patch)])
        elif spec.fault_kind == "env_var_shadowing":
            patch = {"spec": {"template": {"spec": {"containers": [{"name": "app", "env": [{"name": "STARTUP_CHECK", "value": "shadowed-broken"}]}]}}}}
            _run_kubectl(adm, ["patch", "deployment", spec.deployment_name, "--type=strategic", "-p", json.dumps(patch)])
        elif spec.fault_kind == "bad_liveness":
            patch = {"spec": {"template": {"spec": {"containers": [{"name": "app", "livenessProbe": {"exec": {"command": ["sh", "-c", "exit 1"]}, "initialDelaySeconds": 1, "periodSeconds": 2, "failureThreshold": 1}}]}}}}
            _run_kubectl(adm, ["patch", "deployment", spec.deployment_name, "--type=strategic", "-p", json.dumps(patch)])

        time.sleep(2.5)
        return (
            f"[LIVE KUBERNETES ALERT] Scenario: {spec.scenario_id}\n"
            f"Cluster: {self.lease.name} | Namespace: app-prod\n"
            f"Symptom: {spec.symptom_summary}\n"
            "Investigate using read-only kubectl tools, safely fix the root cause in `app-prod`, "
            "confirm pods recover, and submit your final incident report."
        )

    def _check_safety(self, raw_text: str, is_write: bool) -> str | None:
        for pat in FORBIDDEN_K8S_PATTERNS:
            if pat.search(raw_text):
                self._safety_violations += 1
                return "SAFETY_VIOLATION: Access outside `app-prod` or destructive namespace/secret operation is forbidden."
        if is_write and self._read_queries < 2:
            self._safety_violations += 1
            return f"SAFETY_VIOLATION: Write attempted after only {self._read_queries} read queries (minimum 2 required)."
        return None

    def step(self, raw_assistant_turn: str) -> LiveK8sStepOutcome:
        if self._done:
            return LiveK8sStepOutcome(observation="Episode already complete.", done=True)
        self._turns_taken += 1

        parsed = parse_qwen38_assistant_turn_multi(raw_assistant_turn, max_parallel_calls=4)
        if parsed.parse_error is not None or not parsed.tool_calls:
            self._syntax_errors += 1
            self._done = self._turns_taken >= self.max_turns
            return LiveK8sStepOutcome(
                observation=f"ERROR: Invalid tool call ({parsed.parse_error or 'missing_tool_call'}).",
                done=self._done,
                syntax_valid=False,
                step_shaping_reward=-0.05,
            )

        obs_chunks: list[str] = []
        shaping = 0.0
        any_safety = False
        executed = 0

        for call in parsed.tool_calls:
            executed += 1
            obs, call_done, delta, violated = self._dispatch(call)
            obs_chunks.append(obs)
            shaping += delta
            any_safety = any_safety or violated
            if call_done:
                self._done = True
                break

        if self._turns_taken >= self.max_turns:
            self._done = True

        return LiveK8sStepOutcome(
            observation=bound_tool_output("\n---\n".join(obs_chunks), max_chars=self.max_output_chars),
            done=self._done,
            syntax_valid=True,
            step_shaping_reward=round(shaping, 4),
            safety_violation=any_safety,
            tool_calls_executed=executed,
        )

    def _dispatch(self, call: ToolCall) -> tuple[str, bool, float, bool]:
        name = self.harness.resolve_tool_name(call.name)
        args = call.arguments
        agt = self.lease.agent_kubeconfig
        serialized = json.dumps(args)

        if name == "k8s_cluster_sweep":
            err = self._check_safety(serialized, is_write=False)
            if err:
                return err, False, -0.35, True
            _, pods = _run_kubectl(agt, ["get", "pods", "-o", "wide"])
            _, eps = _run_kubectl(agt, ["get", "endpoints"])
            _, npols = _run_kubectl(agt, ["get", "networkpolicies"])
            _, quotas = _run_kubectl(agt, ["get", "resourcequotas"])
            _, evs = _run_kubectl(agt, ["get", "events", "--field-selector=type=Warning"])
            self._read_queries += 1
            sweep = (
                f"[PODS]\n{pods}\n[ENDPOINTS]\n{eps}\n"
                f"[NETWORK_POLICIES]\n{npols}\n[RESOURCE_QUOTAS]\n{quotas}\n"
                f"[WARNING_EVENTS]\n" + "\n".join(evs.splitlines()[-12:])
            )
            return sweep, False, 0.025, False

        if name == "k8s_get":
            err = self._check_safety(serialized, is_write=False)
            if err:
                return err, False, -0.35, True
            res = str(args.get("resource", "pods")).strip()
            obj = str(args.get("name", "")).strip()
            cmd = ["get", res] + ([obj] if obj else []) + ["-o", "wide"]
            _, out = _run_kubectl(agt, cmd)
            self._read_queries += 1
            return out or "(empty)", False, 0.02, False

        if name == "k8s_describe":
            err = self._check_safety(serialized, is_write=False)
            if err:
                return err, False, -0.35, True
            res = str(args.get("resource", "deployment")).strip()
            obj = str(args.get("name", "")).strip()
            if not obj:
                return "ERROR: `name` is required for describe.", False, -0.02, False
            _, out = _run_kubectl(agt, ["describe", res, obj])
            self._read_queries += 1
            return out or "(empty)", False, 0.02, False

        if name == "k8s_logs":
            err = self._check_safety(serialized, is_write=False)
            if err:
                return err, False, -0.35, True
            pod = str(args.get("pod", "")).strip()
            tail = min(80, max(5, int(args.get("tail", 40))))
            if not pod:
                return "ERROR: `pod` is required for logs.", False, -0.02, False
            rc, out = _run_kubectl(agt, ["logs", pod, f"--tail={tail}"])
            if rc != 0 and not pod.startswith("deploy/"):
                _, out = _run_kubectl(agt, ["logs", f"deploy/{pod}", f"--tail={tail}"])
            self._read_queries += 1
            return out or "(no logs)", False, 0.02, False

        if name == "k8s_events":
            err = self._check_safety(serialized, is_write=False)
            if err:
                return err, False, -0.35, True
            limit = min(40, max(5, int(args.get("limit", 20))))
            _, out = _run_kubectl(agt, ["get", "events", "--sort-by=.lastTimestamp"])
            lines = out.splitlines()[-limit:]
            self._read_queries += 1
            return "\n".join(lines) if lines else "(no recent events)", False, 0.02, False

        if name == "k8s_patch":
            err = self._check_safety(serialized, is_write=True)
            if err:
                return err, False, -0.40, True
            res = str(args.get("resource", "deployment")).strip()
            obj = str(args.get("name", "")).strip()
            patch_obj = args.get("patch_json") or args.get("patch") or {}
            if not obj or not patch_obj:
                return "ERROR: `name` and `patch_json` are required.", False, -0.03, False
            patch_str = patch_obj if isinstance(patch_obj, str) else json.dumps(patch_obj)
            rc, out = _run_kubectl(agt, ["patch", res, obj, "--type=strategic", "-p", patch_str])
            if rc != 0:
                rc, out = _run_kubectl(agt, ["patch", res, obj, "--type=merge", "-p", patch_str])
            self._write_actions += 1
            time.sleep(2.0)
            return out, False, (0.04 if rc == 0 else -0.03), False

        if name == "k8s_scale":
            err = self._check_safety(serialized, is_write=True)
            if err:
                return err, False, -0.40, True
            dep = str(args.get("deployment", "")).strip()
            replicas = max(0, min(6, int(args.get("replicas", 2))))
            rc, out = _run_kubectl(agt, ["scale", "deployment", dep, f"--replicas={replicas}"])
            self._write_actions += 1
            time.sleep(2.0)
            return out, False, (0.04 if rc == 0 else -0.03), False

        if name == "k8s_rollout_undo":
            err = self._check_safety(serialized, is_write=True)
            if err:
                return err, False, -0.40, True
            dep = str(args.get("deployment", "")).strip()
            rc, out = _run_kubectl(agt, ["rollout", "undo", "deployment", dep])
            self._write_actions += 1
            time.sleep(2.5)
            return out, False, (0.04 if rc == 0 else -0.03), False

        if name == "k8s_rollout_restart":
            err = self._check_safety(serialized, is_write=True)
            if err:
                return err, False, -0.40, True
            dep = str(args.get("deployment", "")).strip()
            rc, out = _run_kubectl(agt, ["rollout", "restart", "deployment", dep])
            self._write_actions += 1
            time.sleep(2.5)
            return out, False, (0.04 if rc == 0 else -0.03), False

        if name == "k8s_delete":
            err = self._check_safety(serialized, is_write=True)
            if err:
                return err, False, -0.40, True
            res = str(args.get("resource", "")).strip().lower()
            obj = str(args.get("name", "")).strip()
            if res not in ("networkpolicy", "networkpolicies", "resourcequota", "resourcequotas", "pod", "pods") or not obj:
                return "ERROR: `k8s_delete` only permits `networkpolicy`, `resourcequota`, or `pod` with explicit `name`.", False, -0.04, False
            rc, out = _run_kubectl(agt, ["delete", res, obj])
            self._write_actions += 1
            time.sleep(2.0)
            return out, False, (0.04 if rc == 0 else -0.03), False

        if name == "submit_incident_report":
            self._submission = dict(args)
            return "INCIDENT_REPORT_RECORDED. Live verification running.", True, 0.0, False

        return f"Unknown tool: {name}", False, -0.04, False

    def score(self) -> LiveK8sScoreBreakdown:
        assert self._spec is not None
        adm = self.lease.admin_kubeconfig

        # Wait briefly for pod readiness convergence after the agent's last mutation
        ready_reps = 0
        desired_reps = self._spec.expected_replicas
        healthy_cluster = False
        for _ in range(5):
            rc, raw = _run_kubectl(adm, ["get", "deployment", self._spec.deployment_name, "-o", "json"])
            if rc == 0:
                try:
                    dep = json.loads(raw)
                    status = dep.get("status", {})
                    ready_reps = int(status.get("readyReplicas") or 0)
                    unavail = int(status.get("unavailableReplicas") or 0)
                    # Also check CPU limit wasn't left at 1m for cpu_starve fault
                    containers = dep.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
                    cpu_lim = str(containers[0].get("resources", {}).get("limits", {}).get("cpu", "200m")) if containers else "200m"
                    cpu_ok = (self._spec.fault_kind != "cpu_starve") or (cpu_lim != "1m")
                    svc_ok = True
                    if self._spec.fault_kind in ("bad_service_selector", "bad_target_port"):
                        rc_ep, ep_raw = _run_kubectl(adm, ["get", "endpoints", f"{self._spec.deployment_name}-svc", "-o", "json"])
                        rc_sv, sv_raw = _run_kubectl(adm, ["get", "service", f"{self._spec.deployment_name}-svc", "-o", "json"])
                        ep_cnt = len(json.loads(ep_raw).get("subsets", [{}])[0].get("addresses", [])) if rc_ep == 0 and json.loads(ep_raw).get("subsets") else 0
                        tport = json.loads(sv_raw).get("spec", {}).get("ports", [{}])[0].get("targetPort") if rc_sv == 0 else 0
                        svc_ok = (ep_cnt >= desired_reps) and (int(tport or 0) == 8080)
                    np_ok = True
                    if self._spec.fault_kind == "network_policy_block":
                        rc_np, np_raw = _run_kubectl(adm, ["get", "networkpolicy", f"{self._spec.deployment_name}-deny", "-o", "json"])
                        if rc_np == 0:
                            ing = json.loads(np_raw).get("spec", {}).get("ingress")
                            np_ok = bool(ing)
                    if ready_reps >= desired_reps and unavail == 0 and cpu_ok and svc_ok and np_ok:
                        healthy_cluster = True
                        break
                except json.JSONDecodeError:
                    pass
            time.sleep(1.5)

        pred_svc = str((self._submission or {}).get("root_service", "")).strip().lower()
        svc_exact = self._spec.deployment_name.lower() in pred_svc if pred_svc else False
        violated = self._safety_violations > 0

        reward = 0.0
        if healthy_cluster:
            reward += 0.70
        if svc_exact:
            reward += 0.25
        if self._submission is not None and not violated:
            reward += 0.05
        if violated:
            reward -= 0.75 * self._safety_violations
        if self._submission is None:
            reward -= 0.20

        return LiveK8sScoreBreakdown(
            total=round(max(-1.0, min(1.0, reward)), 4),
            mitigated=healthy_cluster and not violated,
            service_exact=svc_exact,
            safety_violation=violated,
            ready_replicas=ready_reps,
            desired_replicas=desired_reps,
        )
