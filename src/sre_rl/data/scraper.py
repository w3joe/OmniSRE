"""Real-World Open-Source SRE Dataset Scraper, Normalizer & RLVR Grounding Pipeline.

Collects, scrubs (`TelemetryScrubber`), deduplicates, and normalizes public, open-licensed
SRE corpora into Qwen3.8-27B `<think>` + `<tool_call>` SFT trajectories, a production
PromQL/Runbook Knowledge Base, and real-world grounded RLVR verification scenarios:

1. `rmems/incident-response-oncall-trajectories` (Apache-2.0):
   Multi-turn OpenSRE (`alert -> evidence -> RCA -> rollback -> verify`) trajectories
   featuring explicit red-herring false leads and falsification steps.
2. `miyad316/sre-agentic-trajectories-devops`:
   Multi-turn Kubernetes/Linux/DB/Cloud SRE tool-calling trajectories.
3. `beatsprom/devops-kubernetes-iac-sft-dpo-2026` (CC-BY-4.0):
   Kubernetes incident RCA, SRE triage, thought chains, and DPO pushback pairs.
4. `Snaseem2026/devops-incident-response`:
   Incident descriptions, multi-turn troubleshooting, RCA, and resolution steps.
5. `nolabs/deepfabric-devops-reasoning-traces`:
   Step-by-step infrastructure/DevOps diagnostic reasoning traces.
6. `openshift/runbooks` (Apache-2.0) & `prometheus-operator/runbooks` (Apache-2.0):
   420+ real production Kubernetes/OpenShift/etcd/kube-apiserver/CoreDNS alert runbooks.
7. `samber/awesome-prometheus-alerts`:
   1,217 real PromQL alert rules across 99 production services.
8. `SREGym/SREGym` (MIT):
   190+ real-world Kubernetes/OS/eBPF/metastable fault problem specifications.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import os
import random
import re
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sre_rl.data.scrubber import TelemetryScrubber
from sre_rl.model.thinking import build_sre_system_prompt
from sre_rl.schemas import (
    FaultCategory,
    FaultSpec,
    GitCommit,
    IncidentClassification,
    IncidentScenario,
    K8sResource,
    LogEntry,
    MetricSeries,
    ReasoningEffort,
    RunbookDoc,
    TelemetrySnapshot,
    TraceSpan,
)

USER_AGENT = "qwen38-sre-rl-dataset-builder/0.1"


def _http_get_bytes(url: str, timeout: int = 30) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {hf_token}"
    gh_token = os.environ.get("GITHUB_TOKEN")
    if gh_token and "github.com" in url:
        headers["Authorization"] = f"Bearer {gh_token}"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_get_json(url: str, timeout: int = 30) -> Any:
    return json.loads(_http_get_bytes(url, timeout=timeout).decode("utf-8"))


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def _normalize_tool_name(raw_name: str) -> str:
    """Map external dataset tool names onto our canonical read/write SRE tool namespace."""
    n = raw_name.strip().lower()
    mapping = {
        "read": "search_runbooks",
        "read_file": "search_runbooks",
        "promql": "query_metrics",
        "query_prometheus": "query_metrics",
        "get_metrics": "query_metrics",
        "metrics_query": "query_metrics",
        "logs": "search_logs",
        "query_logs": "search_logs",
        "loki_query": "search_logs",
        "k8s_get_pods": "k8s_inspect",
        "k8s_describe": "k8s_inspect",
        "k8s_get_events": "k8s_inspect",
        "kubectl_get": "k8s_inspect",
        "kubectl_describe": "k8s_inspect",
        "kubectl_logs": "search_logs",
        "get_trace": "get_traces",
        "jaeger_query": "get_traces",
        "ssh_exec": "host_diag",
        "shell": "host_diag",
        "bash": "host_diag",
        "git_log": "git_recent_changes",
        "helm_history": "git_recent_changes",
        "helm_rollback": "rollback_release",
        "kubectl_rollout_undo": "rollback_release",
        "kubectl_scale": "scale_resource",
        "kubectl_patch": "apply_remediation",
        "kubectl_apply": "apply_remediation",
        "kubectl_rollout_restart": "restart_workload",
    }
    return mapping.get(n, raw_name)


@dataclass
class ScrapeStats:
    source_counts: dict[str, int] = field(default_factory=dict)
    dedup_skipped: int = 0
    quality_skipped: int = 0
    errors: list[str] = field(default_factory=list)


class RealSREDatasetScraper:
    """Concurrent multi-source open SRE corpus scraper & normalizer."""

    def __init__(
        self,
        scrubber: TelemetryScrubber | None = None,
        max_workers: int = 12,
        seed: int = 42,
    ) -> None:
        self.scrubber = scrubber or TelemetryScrubber(salt="qwen38-real-sre-v2")
        self.max_workers = max_workers
        self.rng = random.Random(seed)
        self._seen_hashes: set[str] = set()
        self.stats = ScrapeStats()

    def _register_unique(self, key_text: str) -> bool:
        h = _content_hash(key_text)
        if h in self._seen_hashes:
            self.stats.dedup_skipped += 1
            return False
        self._seen_hashes.add(h)
        return True

    def _scrub_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return [
            {"role": m["role"], "content": self.scrubber.scrub_text(str(m["content"]))}
            for m in messages
        ]

    # ------------------------------------------------------------------
    # Source 1: rmems/incident-response-oncall-trajectories (Apache-2.0)
    # ------------------------------------------------------------------
    def scrape_rmems_oncall(self, max_batches: int = 350, max_records: int = 3500) -> list[dict[str, Any]]:
        """Download Apache-2.0 OpenSRE multi-turn incident trajectories with false leads."""
        results: list[dict[str, Any]] = []
        try:
            info = _http_get_json(
                "https://huggingface.co/api/datasets/rmems/incident-response-oncall-trajectories"
            )
            all_files = sorted(
                s["rfilename"]
                for s in info.get("siblings", [])
                if s["rfilename"].startswith("data/raw/") and s["rfilename"].endswith(".jsonl")
            )
            # Sample deterministically across all rounds for diversity
            if len(all_files) > max_batches:
                step = max(1, len(all_files) // max_batches)
                selected_files = all_files[::step][:max_batches]
            else:
                selected_files = all_files
        except Exception as exc:  # noqa: BLE001
            self.stats.errors.append(f"rmems_list_error: {exc}")
            return results

        def _fetch_one(rel_path: str) -> list[dict[str, Any]]:
            url = (
                "https://huggingface.co/datasets/rmems/"
                f"incident-response-oncall-trajectories/resolve/main/{rel_path}"
            )
            rows_out: list[dict[str, Any]] = []
            try:
                raw_text = _http_get_bytes(url, timeout=25).decode("utf-8", errors="ignore")
                for line in raw_text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    rows_out.append(obj)
            except Exception:  # noqa: BLE001
                return []
            return rows_out

        raw_episodes: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for batch in pool.map(_fetch_one, selected_files):
                raw_episodes.extend(batch)
                if len(raw_episodes) >= max_records * 2:
                    break

        for ep in raw_episodes:
            if len(results) >= max_records:
                break
            reward_meta = ep.get("reward") or {}
            if not reward_meta.get("success", True):
                self.stats.quality_skipped += 1
                continue
            steps = ep.get("steps") or []
            if len(steps) < 3 or len(steps) > 28:
                self.stats.quality_skipped += 1
                continue

            goal = str(ep.get("goal", "")).strip()
            if not goal or not self._register_unique(f"rmems:{goal}:{ep.get('rca', '')[:120]}"):
                continue

            effort = ReasoningEffort.HIGH if len(steps) >= 12 else ReasoningEffort.MEDIUM
            messages: list[dict[str, str]] = [
                {"role": "system", "content": build_sre_system_prompt(effort)},
                {"role": "user", "content": goal},
            ]

            for st in steps:
                thought = str(st.get("decision_basis", "")).strip()
                tc = st.get("tool_call") or {}
                tool_name = _normalize_tool_name(str(tc.get("name", "host_diag")))
                tool_args = tc.get("args", tc.get("arguments", {}))
                if not isinstance(tool_args, dict):
                    tool_args = {"input": str(tool_args)}
                obs = str(st.get("observation", "")).strip()[:3200]

                call_json = json.dumps({"name": tool_name, "arguments": tool_args})
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"<think>{thought}</think>\n<tool_call>{call_json}</tool_call>",
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"<tool_response>\n{obs}\n</tool_response>",
                    }
                )

            # Append structured final incident report closing turn
            rca_summary = str(ep.get("outcome") or ep.get("rca") or "Incident resolved.")[:1200]
            final_call = json.dumps(
                {
                    "name": "submit_incident_report",
                    "arguments": {
                        "classification": "real_incident",
                        "causal_mechanism": rca_summary,
                        "remediation_summary": str(ep.get("remediate", "remediated")),
                    },
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "<think>Root cause falsified false leads and remediation verified. "
                        "Submitting structured incident closeout report.</think>\n"
                        f"<tool_call>{final_call}</tool_call>"
                    ),
                }
            )

            results.append(
                {
                    "id": str(ep.get("id", f"rmems-{len(results)}")),
                    "source": "rmems/incident-response-oncall-trajectories",
                    "license": "Apache-2.0",
                    "reasoning_effort": effort.value,
                    "has_false_lead": bool(ep.get("false_lead")),
                    "num_turns": len(steps) + 1,
                    "messages": self._scrub_messages(messages),
                }
            )

        self.stats.source_counts["rmems/incident-response-oncall-trajectories"] = len(results)
        return results

    # ------------------------------------------------------------------
    # Source 2: miyad316/sre-agentic-trajectories-devops (2,000 traces)
    # ------------------------------------------------------------------
    def scrape_miyad316_sre_trajectories(self, max_records: int = 2000) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        url = (
            "https://huggingface.co/datasets/miyad316/"
            "sre-agentic-trajectories-devops/resolve/main/sample_2000.jsonl"
        )
        try:
            raw_text = _http_get_bytes(url, timeout=60).decode("utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001
            self.stats.errors.append(f"miyad316_error: {exc}")
            return results

        for idx, line in enumerate(raw_text.splitlines()):
            if len(results) >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                self.stats.quality_skipped += 1
                continue

            raw_msgs = obj.get("messages") or []
            if len(raw_msgs) < 4:
                self.stats.quality_skipped += 1
                continue

            meta = obj.get("metadata") or {}
            diff = str(meta.get("difficulty", "medium")).lower()
            effort = (
                ReasoningEffort.HIGH
                if diff == "hard"
                else (ReasoningEffort.LOW if diff == "easy" else ReasoningEffort.MEDIUM)
            )

            messages: list[dict[str, str]] = [
                {"role": "system", "content": build_sre_system_prompt(effort)}
            ]
            first_user = ""

            for m in raw_msgs:
                role = str(m.get("role", ""))
                if role == "system":
                    continue
                if role == "user":
                    content = str(m.get("content", "")).strip()
                    if not first_user:
                        first_user = content
                    messages.append({"role": "user", "content": content})
                elif role == "tool":
                    obs = str(m.get("content", "")).strip()[:3200]
                    messages.append(
                        {"role": "user", "content": f"<tool_response>\n{obs}\n</tool_response>"}
                    )
                elif role == "assistant":
                    thought = str(m.get("content") or m.get("reasoning_content") or "").strip()
                    tcs = m.get("tool_calls") or []
                    if tcs:
                        fn = tcs[0].get("function") or {}
                        tname = _normalize_tool_name(str(fn.get("name", "host_diag")))
                        raw_args = fn.get("arguments", {})
                        if isinstance(raw_args, str):
                            try:
                                raw_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                raw_args = {"raw": raw_args}
                        call_str = json.dumps({"name": tname, "arguments": raw_args})
                        body = (
                            f"<think>{thought}</think>\n<tool_call>{call_str}</tool_call>"
                            if thought
                            else f"<tool_call>{call_str}</tool_call>"
                        )
                        messages.append({"role": "assistant", "content": body})
                    elif thought:
                        messages.append({"role": "assistant", "content": thought})

            if not first_user or not self._register_unique(f"miyad316:{first_user[:240]}"):
                continue

            results.append(
                {
                    "id": str(obj.get("id", f"miyad316-{idx}")),
                    "source": "miyad316/sre-agentic-trajectories-devops",
                    "license": "open-hf",
                    "category": str(meta.get("category", "sre")),
                    "subcategory": str(meta.get("subcategory", "general")),
                    "reasoning_effort": effort.value,
                    "num_turns": sum(1 for m in messages if m["role"] == "assistant"),
                    "messages": self._scrub_messages(messages),
                }
            )

        self.stats.source_counts["miyad316/sre-agentic-trajectories-devops"] = len(results)
        return results

    # ------------------------------------------------------------------
    # Source 3: beatsprom/devops-kubernetes-iac-sft-dpo-2026 (CC-BY-4.0)
    # ------------------------------------------------------------------
    def scrape_beatsprom_k8s_sre(self, max_records: int = 1500) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        rows = self._download_hf_parquet_rows(
            "beatsprom/devops-kubernetes-iac-sft-dpo-2026", max_rows=max_records
        )
        for idx, row in enumerate(rows):
            prompt = str(row.get("user_prompt", "")).strip()
            thought = str(row.get("thought_chain", "")).strip()
            chosen = str(row.get("chosen_response", "")).strip()
            if not prompt or not chosen:
                continue
            if not self._register_unique(f"beatsprom:{prompt[:240]}"):
                continue

            risk = str(row.get("risk_tier", "")).upper()
            effort = ReasoningEffort.HIGH if "CRITICAL" in risk else ReasoningEffort.MEDIUM
            assistant_content = (
                f"<think>{thought}</think>\n{chosen}" if thought else chosen
            )
            messages = [
                {"role": "system", "content": build_sre_system_prompt(effort)},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_content},
            ]
            # Preserve rejected response as DPO/preference metadata for optional Stage 1.5 DPO
            results.append(
                {
                    "id": str(row.get("id", f"beatsprom-{idx}")),
                    "source": "beatsprom/devops-kubernetes-iac-sft-dpo-2026",
                    "license": "CC-BY-4.0",
                    "category": str(row.get("domain", "kubernetes_sre")),
                    "reasoning_effort": effort.value,
                    "num_turns": 1,
                    "messages": self._scrub_messages(messages),
                    "rejected_response": self.scrubber.scrub_text(
                        str(row.get("rejected_response", ""))[:2000]
                    ),
                }
            )

        self.stats.source_counts["beatsprom/devops-kubernetes-iac-sft-dpo-2026"] = len(results)
        return results

    # ------------------------------------------------------------------
    # Source 4: nolabs/deepfabric-devops-reasoning-traces
    # ------------------------------------------------------------------
    def scrape_deepfabric_devops(self, max_records: int = 1000) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        rows = self._download_hf_parquet_rows(
            "nolabs/deepfabric-devops-reasoning-traces", max_rows=max_records
        )
        for idx, row in enumerate(rows):
            question = str(row.get("question", "")).strip()
            cot = str(row.get("chain_of_thought", "")).strip()
            final_ans = str(row.get("final_answer", "")).strip()
            if not question or not final_ans:
                continue
            if not self._register_unique(f"deepfabric:{question[:240]}"):
                continue

            trace_steps = row.get("reasoning_trace") or []
            if isinstance(trace_steps, list) and trace_steps:
                step_lines = [
                    f"Step {s.get('step_number', i + 1)}: {s.get('thought', '')}"
                    for i, s in enumerate(trace_steps)
                    if isinstance(s, dict)
                ]
                thought_block = "\n".join(step_lines) if step_lines else cot
            else:
                thought_block = cot

            messages = [
                {"role": "system", "content": build_sre_system_prompt(ReasoningEffort.MEDIUM)},
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": f"<think>{thought_block}</think>\n{final_ans}",
                },
            ]
            results.append(
                {
                    "id": f"deepfabric-{idx:05d}",
                    "source": "nolabs/deepfabric-devops-reasoning-traces",
                    "license": "open-hf",
                    "category": "devops_reasoning",
                    "reasoning_effort": ReasoningEffort.MEDIUM.value,
                    "num_turns": 1,
                    "messages": self._scrub_messages(messages),
                }
            )

        self.stats.source_counts["nolabs/deepfabric-devops-reasoning-traces"] = len(results)
        return results

    # ------------------------------------------------------------------
    # Source 5: Snaseem2026/devops-incident-response
    # ------------------------------------------------------------------
    def scrape_snaseem_incidents(self, max_records: int = 1000) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        rows = self._download_hf_parquet_rows(
            "Snaseem2026/devops-incident-response", max_rows=max_records
        )
        if not rows:
            # Fallback to JSONL/JSON files directly from repo siblings
            rows = self._download_hf_json_rows(
                "Snaseem2026/devops-incident-response", max_rows=max_records
            )

        for idx, row in enumerate(rows):
            title = str(row.get("title", "")).strip()
            desc = str(row.get("description", "")).strip()
            rca = str(row.get("root_cause", "")).strip()
            if not title or not rca:
                continue
            if not self._register_unique(f"snaseem:{title}:{rca[:150]}"):
                continue

            sev = str(row.get("severity", "high")).upper()
            symptoms = row.get("symptoms") or []
            res_steps = row.get("resolution_steps") or []
            conv = row.get("troubleshooting_conversation") or []

            prompt = (
                f"[{sev}] Incident: {title}\n"
                f"Category: {row.get('category', 'infra')} | Env: {row.get('environment', 'prod')}\n"
                f"Description: {desc}\n"
                f"Symptoms: {json.dumps(symptoms) if isinstance(symptoms, list) else symptoms}"
            )
            thought_lines = []
            if isinstance(conv, list):
                for turn in conv[:6]:
                    if isinstance(turn, dict):
                        thought_lines.append(
                            f"{turn.get('role', turn.get('speaker', 'engineer'))}: "
                            f"{turn.get('message', turn.get('content', ''))}"
                        )
                    else:
                        thought_lines.append(str(turn))
            thought_text = "\n".join(thought_lines) or f"Analyze symptoms and isolate root cause: {rca}"

            report_call = json.dumps(
                {
                    "name": "submit_incident_report",
                    "arguments": {
                        "classification": "real_incident",
                        "root_service": str(row.get("category", "cluster")),
                        "causal_mechanism": rca,
                        "remediation_summary": (
                            "; ".join(str(x) for x in res_steps[:6])
                            if isinstance(res_steps, list)
                            else str(res_steps)
                        ),
                    },
                }
            )
            messages = [
                {"role": "system", "content": build_sre_system_prompt(ReasoningEffort.MEDIUM)},
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": f"<think>{thought_text}</think>\n<tool_call>{report_call}</tool_call>",
                },
            ]
            results.append(
                {
                    "id": str(row.get("incident_id", f"snaseem-{idx:05d}")),
                    "source": "Snaseem2026/devops-incident-response",
                    "license": "open-hf",
                    "category": str(row.get("category", "devops")),
                    "reasoning_effort": ReasoningEffort.MEDIUM.value,
                    "num_turns": 1,
                    "messages": self._scrub_messages(messages),
                }
            )

        self.stats.source_counts["Snaseem2026/devops-incident-response"] = len(results)
        return results

    # ------------------------------------------------------------------
    # Source 6: samber/awesome-prometheus-alerts (1,217 real PromQL rules)
    # ------------------------------------------------------------------
    def scrape_awesome_prometheus_alerts(self) -> list[dict[str, Any]]:
        url = "https://raw.githubusercontent.com/samber/awesome-prometheus-alerts/master/_data/rules.yml"
        rules_out: list[dict[str, Any]] = []
        try:
            doc = yaml.safe_load(_http_get_bytes(url, timeout=25).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.stats.errors.append(f"awesome_prom_error: {exc}")
            return rules_out

        for group in doc.get("groups") or []:
            group_name = str(group.get("name", "general"))
            for svc in group.get("services") or []:
                svc_name = str(svc.get("name", "unknown"))
                for exp in svc.get("exporters") or []:
                    exp_name = str(exp.get("name", ""))
                    for rule in exp.get("rules") or []:
                        if not isinstance(rule, dict):
                            continue
                        alert_name = str(rule.get("name", "")).strip()
                        query = str(rule.get("query", "")).strip()
                        if not alert_name or not query:
                            continue
                        rules_out.append(
                            {
                                "kb_type": "promql_alert_rule",
                                "source": "samber/awesome-prometheus-alerts",
                                "group": group_name,
                                "service": svc_name,
                                "exporter": exp_name,
                                "alert_name": alert_name,
                                "description": str(rule.get("description", "")).strip(),
                                "promql": query,
                                "severity": str(rule.get("severity", "warning")).lower(),
                                "for_duration": str(rule.get("for", "2m")),
                            }
                        )

        # Scrub any secrets inside PromQL alert rules too
        for rule in rules_out:
            rule["description"] = self.scrubber.scrub_text(rule["description"])
            rule["promql"] = self.scrubber.scrub_text(rule["promql"])

        self.stats.source_counts["samber/awesome-prometheus-alerts"] = len(rules_out)
        return rules_out

    # ------------------------------------------------------------------
    # Source 7: openshift/runbooks + prometheus-operator/runbooks (Tarball!)
    # ------------------------------------------------------------------
    def scrape_github_runbook_tarball(
        self,
        repo: str,
        branch: str,
        path_prefix: str,
        license_id: str = "Apache-2.0",
    ) -> list[dict[str, Any]]:
        """Download full GitHub repo tarball in 1 HTTP call (avoids GitHub API rate limits!)."""
        url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}"
        docs: list[dict[str, Any]] = []
        try:
            tar_bytes = _http_get_bytes(url, timeout=60)
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
                for member in tf.getmembers():
                    if not member.isfile() or not member.name.endswith(".md"):
                        continue
                    # Strip leading `<repo>-<branch>/`
                    rel = member.name.split("/", 1)[1] if "/" in member.name else member.name
                    if not rel.startswith(path_prefix):
                        continue
                    if rel.endswith(("_index.md", "README.md")):
                        continue
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    body = fh.read().decode("utf-8", errors="ignore").strip()
                    if len(body) < 120:
                        continue
                    alert_name = Path(rel).stem
                    component = Path(rel).parent.name
                    docs.append(
                        {
                            "kb_type": "production_runbook",
                            "source": repo,
                            "license": license_id,
                            "component": component,
                            "alert_name": alert_name,
                            "path": rel,
                            "body": self.scrubber.scrub_text(body[:6000]),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            self.stats.errors.append(f"tarball_{repo}_error: {exc}")

        self.stats.source_counts[repo] = len(docs)
        return docs

    # ------------------------------------------------------------------
    # Source 8: SREGym/SREGym fault specifications (MIT, Tarball)
    # ------------------------------------------------------------------
    def scrape_sregym_fault_specs(self) -> list[dict[str, Any]]:
        url = "https://codeload.github.com/SREGym/SREGym/tar.gz/refs/heads/main"
        specs: list[dict[str, Any]] = []
        try:
            tar_bytes = _http_get_bytes(url, timeout=60)
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
                for member in tf.getmembers():
                    if not member.isfile() or not member.name.endswith(".py"):
                        continue
                    rel = member.name.split("/", 1)[1] if "/" in member.name else member.name
                    if not rel.startswith("sregym/conductor/problems/"):
                        continue
                    if rel.endswith(("__init__.py", "base.py", "registry.py")):
                        continue
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    code = fh.read().decode("utf-8", errors="ignore")
                    prob_id = Path(rel).stem
                    doc_match = re.search(r'"""(.*?)"""', code, re.DOTALL)
                    docstring = doc_match.group(1).strip() if doc_match else ""
                    specs.append(
                        {
                            "kb_type": "sregym_fault_spec",
                            "source": "SREGym/SREGym",
                            "license": "MIT",
                            "problem_id": prob_id,
                            "path": rel,
                            "summary": self.scrubber.scrub_text(
                                docstring[:1000] or prob_id.replace("_", " ")
                            ),
                            "code_excerpt": self.scrubber.scrub_text(code[:2000]),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            self.stats.errors.append(f"sregym_error: {exc}")

        self.stats.source_counts["SREGym/SREGym"] = len(specs)
        return specs

    # ------------------------------------------------------------------
    # Source 9: ibm-research/ITBench-Trajectories (Real SRE ReAct runs)
    # ------------------------------------------------------------------
    def scrape_itbench_trajectories(
        self,
        max_sessions: int = 105,
        min_judge_score: float = 0.25,
    ) -> list[dict[str, Any]]:
        """Download real IBM Research ITBench SRE ReAct trajectories & normalize tool calls."""
        results: list[dict[str, Any]] = []
        try:
            info = _http_get_json(
                "https://huggingface.co/api/datasets/ibm-research/ITBench-Trajectories"
            )
            all_files = {s["rfilename"] for s in info.get("siblings", [])}
            sre_sessions = sorted(
                f for f in all_files if "/sre/" in f and f.endswith("session.jsonl")
            )[:max_sessions]
        except Exception as exc:  # noqa: BLE001
            self.stats.errors.append(f"itbench_list_error: {exc}")
            return results

        def _fetch_session(sess_path: str) -> dict[str, Any] | None:
            base_dir = sess_path.rsplit("/", 1)[0]
            judge_path = f"{base_dir}/judge_output.json"
            judge_score = 1.0
            if judge_path in all_files:
                try:
                    j_url = (
                        "https://huggingface.co/datasets/ibm-research/"
                        f"ITBench-Trajectories/resolve/main/{judge_path}"
                    )
                    j_obj = _http_get_json(j_url, timeout=20)
                    scores = (j_obj.get("eval_result") or {}).get("scores") or {}
                    rc_score = scores.get("root_cause_entity") or {}
                    f1 = rc_score.get("f1", rc_score.get("recall", rc_score.get("accuracy")))
                    if isinstance(f1, (int, float)):
                        judge_score = float(f1)
                except Exception:  # noqa: BLE001
                    judge_score = 0.5

            if judge_score < min_judge_score:
                return None

            s_url = (
                "https://huggingface.co/datasets/ibm-research/"
                f"ITBench-Trajectories/resolve/main/{sess_path}"
            )
            try:
                raw_lines = _http_get_bytes(s_url, timeout=30).decode("utf-8", errors="ignore").splitlines()
            except Exception:  # noqa: BLE001
                return None

            user_goal = ""
            pending_thought = ""
            messages: list[dict[str, str]] = [
                {"role": "system", "content": build_sre_system_prompt(ReasoningEffort.HIGH)}
            ]
            turn_count = 0

            for ln in raw_lines:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    ev = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                ev_type = ev.get("type")
                payload = ev.get("payload") or {}
                p_type = payload.get("type")

                if ev_type == "event_msg" and p_type == "user_message" and not user_goal:
                    user_goal = str(payload.get("message", "")).strip()
                    if user_goal:
                        messages.append({"role": "user", "content": user_goal[:3000]})
                elif ev_type == "response_item" and p_type == "reasoning":
                    summary = payload.get("summary") or []
                    if isinstance(summary, list):
                        parts = [
                            str(x.get("text", x)) if isinstance(x, dict) else str(x)
                            for x in summary
                        ]
                        txt = " ".join(p.strip() for p in parts if p.strip())
                    else:
                        txt = str(payload.get("content") or summary or "").strip()
                    if txt:
                        pending_thought = txt[:1000]
                elif ev_type == "response_item" and p_type == "function_call":
                    if not user_goal:
                        user_goal = f"Investigate and diagnose ITBench SRE incident ({base_dir})."
                        messages.append({"role": "user", "content": user_goal})
                    if turn_count >= 22:
                        continue
                    tname = _normalize_tool_name(str(payload.get("name", "host_diag")))
                    raw_args = payload.get("arguments", {})
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            raw_args = {"command": raw_args[:1200]}
                    if not isinstance(raw_args, dict):
                        raw_args = {"input": str(raw_args)[:1200]}
                    thought = pending_thought or f"Execute {tname} to inspect cluster telemetry."
                    pending_thought = ""
                    call_json = json.dumps({"name": tname, "arguments": raw_args})
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"<think>{thought}</think>\n<tool_call>{call_json}</tool_call>",
                        }
                    )
                    turn_count += 1
                elif (
                    ev_type == "response_item"
                    and p_type == "function_call_output"
                    and messages
                    and messages[-1]["role"] == "assistant"
                ):
                    obs = str(payload.get("output", "")).strip()[:2800]
                    messages.append(
                        {"role": "user", "content": f"<tool_response>\n{obs}\n</tool_response>"}
                    )

            if turn_count < 2:
                return None

            # Ensure conversation ends on an assistant turn
            if messages[-1]["role"] == "user":
                close_call = json.dumps(
                    {
                        "name": "submit_incident_report",
                        "arguments": {
                            "classification": "real_incident",
                            "root_service": base_dir.split("/")[-2],
                            "causal_mechanism": "ITBench SRE root-cause localization completed.",
                        },
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "<think>Evidence gathered across cluster telemetry; submit root cause report.</think>\n"
                            f"<tool_call>{close_call}</tool_call>"
                        ),
                    }
                )

            return {
                "id": f"itbench-{base_dir.replace('/', '_')}",
                "source": "ibm-research/ITBench-Trajectories",
                "license": "Apache-2.0",
                "judge_score": round(judge_score, 3),
                "reasoning_effort": ReasoningEffort.HIGH.value,
                "num_turns": sum(1 for m in messages if m["role"] == "assistant"),
                "messages": messages,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for item in pool.map(_fetch_session, sre_sessions):
                if item is None:
                    self.stats.quality_skipped += 1
                    continue
                if not self._register_unique(f"itbench:{item['id']}"):
                    continue
                item["messages"] = self._scrub_messages(item["messages"])
                results.append(item)

        self.stats.source_counts["ibm-research/ITBench-Trajectories"] = len(results)
        return results

    # ------------------------------------------------------------------
    # Source 10: danluu/post-mortems (Real Production Outage Index)
    # ------------------------------------------------------------------
    def scrape_danluu_postmortems(self) -> list[dict[str, Any]]:
        url = "https://raw.githubusercontent.com/danluu/post-mortems/master/README.md"
        entries: list[dict[str, Any]] = []
        try:
            text = _http_get_bytes(url, timeout=20).decode("utf-8", errors="ignore")
            link_re = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)\.\s*(.+)")
            for idx, line in enumerate(text.splitlines()):
                m = link_re.search(line.strip())
                if not m:
                    continue
                title, link, summary = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                if len(summary) < 30:
                    continue
                entries.append(
                    {
                        "kb_type": "real_outage_postmortem",
                        "source": "danluu/post-mortems",
                        "postmortem_id": f"pm-{idx:04d}",
                        "title": self.scrubber.scrub_text(title),
                        "url": link,
                        "summary": self.scrubber.scrub_text(summary[:1500]),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            self.stats.errors.append(f"danluu_postmortems_error: {exc}")

        self.stats.source_counts["danluu/post-mortems"] = len(entries)
        return entries

    # ------------------------------------------------------------------
    # Grounding: Convert Scraped Production Runbooks + PromQL Alerts into
    # Verifiable RLVR `IncidentScenario` Bundles & SFT Traces
    # ------------------------------------------------------------------
    def build_grounded_rlvr_scenarios_from_kb(
        self,
        runbooks: list[dict[str, Any]],
        promql_rules: list[dict[str, Any]],
        max_scenarios: int = 800,
    ) -> list[IncidentScenario]:
        """Ground executable Tier-1 RLVR scenarios in real OpenShift/Kube-Prometheus runbooks & PromQL."""
        scenarios: list[IncidentScenario] = []

        # Map PromQL rules by lowercase keyword for cross-linking
        prom_by_service: dict[str, list[dict[str, Any]]] = {}
        for r in promql_rules:
            prom_by_service.setdefault(r["service"].lower(), []).append(r)

        # 1. Build scenarios from real production runbooks (OpenShift + kube-prometheus)
        for idx, rb in enumerate(runbooks):
            if len(scenarios) >= max_scenarios // 2:
                break
            alert = rb["alert_name"]
            comp = re.sub(r"[^a-z0-9\-]", "-", rb["component"].lower()).strip("-") or "kube-system"
            body = rb["body"]

            # Extract concrete diagnosis/mitigation keywords from the real runbook text
            cat = self._infer_fault_category(f"{alert} {comp} {body}")
            sev = "SEV1" if any(k in alert.lower() for k in ("down", "crash", "oom", "errorbudget", "outage")) else "SEV2"
            effort = ReasoningEffort.HIGH if sev == "SEV1" else ReasoningEffort.MEDIUM

            # Pull key technical terms from the runbook for deterministic RCA rubric scoring
            keywords = self._extract_rca_keywords(alert, comp, body)
            root_svc = f"{comp[:24]}-svc"
            root_obj = f"{comp[:24]}-config"
            ns = f"{comp[:18]}-ns"
            entry_svc = "kube-ingress"

            patch_field, bad_val, good_val = self._infer_patch_pair(alert, body)

            scenarios.append(
                IncidentScenario(
                    scenario_id=f"real-rb-{idx:04d}-{alert.lower()[:32]}",
                    topology_name=f"k8s-{comp[:20]}",
                    severity=sev,
                    reasoning_effort=effort,
                    alert_title=f"Alert Firing: {alert} ({comp})",
                    alert_pager_payload=(
                        f"alertname={alert} component={comp} namespace={ns} "
                        f"source={rb['source']} runbook={rb['path']}"
                    ),
                    service_dependencies={
                        entry_svc: [root_svc, "monitoring-agent"],
                        root_svc: ["cluster-etcd"],
                        "monitoring-agent": [],
                        "cluster-etcd": [],
                    },
                    healthy_services=["monitoring-agent", "cluster-etcd"],
                    telemetry=TelemetrySnapshot(
                        metrics=[
                            MetricSeries(
                                service=root_svc,
                                metric="http_requests_error_ratio",
                                unit="ratio",
                                values=[0.002, 0.004, 0.072, 0.195, 0.284, 0.312],
                                slo_threshold=0.02,
                            ),
                            MetricSeries(
                                service=entry_svc,
                                metric="http_requests_error_ratio",
                                unit="ratio",
                                values=[0.001, 0.002, 0.031, 0.088, 0.121, 0.135],
                                slo_threshold=0.02,
                            ),
                            MetricSeries(
                                service="monitoring-agent",
                                metric="http_requests_error_ratio",
                                unit="ratio",
                                values=[0.001] * 6,
                                slo_threshold=0.02,
                            ),
                        ],
                        logs=[
                            LogEntry(
                                timestamp="2026-11-19T14:05:10Z",
                                service=root_svc,
                                level="ERROR",
                                message=f"Alert {alert} triggered on {comp}: {body[:240].replace(chr(10), ' ')}",
                            )
                        ],
                        traces=[
                            TraceSpan(
                                trace_id=f"tr-rb-{idx:04d}",
                                span_id="sp-01",
                                caller=entry_svc,
                                callee=root_svc,
                                operation=f"CHECK /{comp}/health",
                                duration_ms=1920.0,
                                status_code=503,
                                error=f"{alert}: {patch_field}={bad_val}",
                            )
                        ],
                        k8s_resources=[
                            K8sResource(
                                kind="ConfigMap",
                                namespace=ns,
                                name=root_obj,
                                status=f"Degraded ({alert})",
                                spec={patch_field: bad_val, "alert_source": alert},
                                events=[f"Warning {alert}: {patch_field}={bad_val} violates SLO policy"],
                            ),
                            K8sResource(
                                kind="Deployment",
                                namespace=ns,
                                name=f"{root_svc}-deploy",
                                status="Degraded (1/3 Ready)",
                                spec={"replicas": 3},
                                events=[f"Warning Unhealthy: Triggered {alert}"],
                            ),
                        ],
                        recent_commits=[
                            GitCommit(
                                sha=f"rb{idx:05x}",
                                timestamp="2026-11-19T13:45:00Z",
                                author="gitops-controller",
                                service=root_svc,
                                summary=f"Update {root_obj} ({patch_field}={bad_val})",
                                diff_snippet=f"- {patch_field}: {good_val}\n+ {patch_field}: {bad_val}",
                            )
                        ],
                        runbooks=[
                            RunbookDoc(
                                doc_id=f"rb-{alert}",
                                title=f"Runbook: {alert} ({rb['source']})",
                                tags=[comp, alert.lower(), cat.value],
                                body=body[:2400],
                            )
                        ],
                    ),
                    fault=FaultSpec(
                        fault_id=f"real_runbook_{alert.lower()[:36]}",
                        category=cat,
                        classification=IncidentClassification.REAL_INCIDENT,
                        root_namespace=ns,
                        root_service=root_svc,
                        root_object=root_obj,
                        causal_mechanism=f"{alert} on {comp}: misconfigured {patch_field}={bad_val} (expected {good_val})",
                        causal_keywords=keywords,
                        persistent_across_pod_restart=True,
                        expected_remediation_tool="apply_remediation",
                        expected_patch_fields={patch_field: good_val},
                    ),
                )
            )

        # 2. Build scenarios from real PromQL rules (samber/awesome-prometheus-alerts)
        for idx, rule in enumerate(promql_rules):
            if len(scenarios) >= max_scenarios:
                break
            svc_slug = re.sub(r"[^a-z0-9\-]", "-", rule["service"].lower()).strip("-") or "service"
            alert_slug = re.sub(r"[^a-z0-9\-]", "-", rule["alert_name"].lower()).strip("-")[:32]
            cat = self._infer_fault_category(f"{rule['group']} {rule['service']} {rule['alert_name']} {rule['description']}")
            sev = "SEV1" if rule["severity"] == "critical" else "SEV2"
            ns = f"{svc_slug[:18]}-prod"
            root_svc = svc_slug[:24]
            root_obj = f"{root_svc}-config"
            patch_field, bad_val, good_val = self._infer_patch_pair(rule["alert_name"], rule["description"])

            scenarios.append(
                IncidentScenario(
                    scenario_id=f"real-prom-{idx:04d}-{alert_slug}",
                    topology_name=f"prom-{svc_slug[:20]}",
                    severity=sev,
                    reasoning_effort=ReasoningEffort.MEDIUM,
                    alert_title=f"[{rule['service']}] {rule['alert_name']}",
                    alert_pager_payload=(
                        f"alertname={rule['alert_name']} service={rule['service']} "
                        f"group={rule['group']} promql=`{rule['promql']}` "
                        f"description='{rule['description']}'"
                    ),
                    service_dependencies={
                        "edge-proxy": [root_svc, "telemetry-collector"],
                        root_svc: [],
                        "telemetry-collector": [],
                    },
                    healthy_services=["telemetry-collector"],
                    telemetry=TelemetrySnapshot(
                        metrics=[
                            MetricSeries(
                                service=root_svc,
                                metric="http_requests_error_ratio",
                                unit="ratio",
                                values=[0.002, 0.003, 0.09, 0.21, 0.29, 0.33],
                                slo_threshold=0.02,
                            )
                        ],
                        logs=[
                            LogEntry(
                                timestamp="2026-11-19T14:20:00Z",
                                service=root_svc,
                                level="ERROR",
                                message=(
                                    f"Prometheus rule {rule['alert_name']} firing "
                                    f"(query: {rule['promql'][:180]}): {rule['description']}"
                                ),
                            )
                        ],
                        traces=[
                            TraceSpan(
                                trace_id=f"tr-prom-{idx:04d}",
                                span_id="sp-01",
                                caller="edge-proxy",
                                callee=root_svc,
                                operation=f"CALL /{root_svc}",
                                duration_ms=1750.0,
                                status_code=503,
                                error=rule["description"][:140],
                            )
                        ],
                        k8s_resources=[
                            K8sResource(
                                kind="ConfigMap",
                                namespace=ns,
                                name=root_obj,
                                status=f"Alerting ({rule['alert_name']})",
                                spec={patch_field: bad_val, "promql_trigger": rule["promql"][:160]},
                                events=[f"Warning {rule['alert_name']}: {rule['description'][:180]}"],
                            )
                        ],
                        recent_commits=[
                            GitCommit(
                                sha=f"pm{idx:05x}",
                                timestamp="2026-11-19T14:00:00Z",
                                author="infra-bot",
                                service=root_svc,
                                summary=f"Modify {root_obj} {patch_field}={bad_val}",
                                diff_snippet=f"- {patch_field}: {good_val}\n+ {patch_field}: {bad_val}",
                            )
                        ],
                        runbooks=[
                            RunbookDoc(
                                doc_id=f"prom-{alert_slug}",
                                title=f"PromQL Alert: {rule['service']} — {rule['alert_name']}",
                                tags=[rule["service"].lower(), rule["group"].lower(), cat.value],
                                body=(
                                    f"PromQL Expression:\n{rule['promql']}\n\n"
                                    f"Description: {rule['description']}\n"
                                    f"Remediation: Inspect `{root_obj}` and restore `{patch_field}={good_val}`."
                                ),
                            )
                        ],
                    ),
                    fault=FaultSpec(
                        fault_id=f"real_prom_{alert_slug}",
                        category=cat,
                        classification=IncidentClassification.REAL_INCIDENT,
                        root_namespace=ns,
                        root_service=root_svc,
                        root_object=root_obj,
                        causal_mechanism=(
                            f"{rule['alert_name']} ({rule['service']}): {rule['description']} — "
                            f"fix {patch_field} from {bad_val} to {good_val}"
                        ),
                        causal_keywords=self._extract_rca_keywords(
                            rule["alert_name"], rule["service"], rule["description"]
                        ),
                        persistent_across_pod_restart=True,
                        expected_remediation_tool="apply_remediation",
                        expected_patch_fields={patch_field: good_val},
                    ),
                )
            )

        return scenarios

    def _download_hf_parquet_rows(self, repo_id: str, max_rows: int) -> list[dict[str, Any]]:
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError:
            self.stats.errors.append(f"pyarrow_missing_for_{repo_id}")
            return []

        rows: list[dict[str, Any]] = []
        try:
            info = _http_get_json(f"https://huggingface.co/api/datasets/{repo_id}")
            parquets = [
                s["rfilename"]
                for s in info.get("siblings", [])
                if s["rfilename"].endswith(".parquet")
            ]
            for rel in parquets[:4]:
                url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel}"
                raw = _http_get_bytes(url, timeout=60)
                table = pq.read_table(io.BytesIO(raw))
                for r in table.to_pylist():
                    rows.append(r)
                    if len(rows) >= max_rows:
                        return rows
        except Exception as exc:  # noqa: BLE001
            self.stats.errors.append(f"hf_parquet_{repo_id}_error: {exc}")
        return rows

    def _download_hf_json_rows(self, repo_id: str, max_rows: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            info = _http_get_json(f"https://huggingface.co/api/datasets/{repo_id}")
            files = [
                s["rfilename"]
                for s in info.get("siblings", [])
                if s["rfilename"].endswith((".jsonl", ".json"))
                and not s["rfilename"].lower().startswith("dataset")
            ]
            for rel in files[:4]:
                url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel}"
                txt = _http_get_bytes(url, timeout=45).decode("utf-8", errors="ignore").strip()
                if rel.endswith(".jsonl"):
                    for ln in txt.splitlines():
                        if ln.strip():
                            rows.append(json.loads(ln))
                            if len(rows) >= max_rows:
                                return rows
                else:
                    parsed = json.loads(txt)
                    if isinstance(parsed, list):
                        rows.extend(parsed[: max_rows - len(rows)])
                    elif isinstance(parsed, dict):
                        for v in parsed.values():
                            if isinstance(v, list):
                                rows.extend(v[: max_rows - len(rows)])
        except Exception as exc:  # noqa: BLE001
            self.stats.errors.append(f"hf_json_{repo_id}_error: {exc}")
        return rows

    @staticmethod
    def _infer_fault_category(text: str) -> FaultCategory:
        t = text.lower()
        if any(k in t for k in ("postgres", "mysql", "redis", "mongo", "kafka", "etcd", "disk", "storage", "clickhouse", "cassandra")):
            return FaultCategory.STATEFUL_DB
        if any(k in t for k in ("dns", "envoy", "istio", "ingress", "tls", "cert", "network", "tcp", "haproxy", "nginx", "504", "502")):
            return FaultCategory.NETWORK_MESH
        if any(k in t for k in ("retry", "cascade", "throttl", "saturation", "queue")):
            return FaultCategory.METASTABLE_CASCADE
        if any(k in t for k in ("oom", "kube", "pod", "node", "probe", "hpa", "daemonset", "deployment", "pvc")):
            return FaultCategory.K8S_CONFIG
        return FaultCategory.APP_REGRESSION

    @staticmethod
    def _infer_patch_pair(alert: str, desc: str) -> tuple[str, str, str]:
        t = f"{alert} {desc}".lower()
        if any(k in t for k in ("oom", "memory", "ram")):
            return ("memory_limit", "256Mi", "2Gi")
        if any(k in t for k in ("cpu", "throttl")):
            return ("cpu_limit", "200m", "2000m")
        if any(k in t for k in ("timeout", "latency", "slow", "504")):
            return ("timeout_ms", "100", "5000")
        if any(k in t for k in ("connection", "pool", "max_conn")):
            return ("max_connections", "20", "300")
        if any(k in t for k in ("disk", "storage", "volume", "space")):
            return ("storage_retention_days", "365", "14")
        if any(k in t for k in ("cert", "tls", "ssl", "expir")):
            return ("tls_auto_renew", "false", "true")
        return ("reconcile_policy", "broken", "healthy")

    @staticmethod
    def _extract_rca_keywords(alert: str, svc: str, text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{3,}", f"{alert} {svc} {text[:200]}")
        stop = {"alert", "firing", "with", "that", "this", "from", "have", "been", "more", "than", "when", "into"}
        picked: list[str] = []
        for tok in tokens:
            low = tok.lower()
            if low not in stop and low not in picked:
                picked.append(low)
            if len(picked) >= 3:
                break
        return picked or [svc.lower()]
