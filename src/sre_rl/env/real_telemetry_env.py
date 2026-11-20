"""Interactive RLVR Environment Over REAL Multi-Gigabyte Production & K8s Telemetry.

Supports actual recorded multi-modal observability corpora (zero synthetic templates):
- **RCAEval (`RE1`/`RE2`/`RE3`):** Real GKE Kubernetes captures with 6-20 MB Prometheus
  `metrics.csv`, 10-20 MB `logs.csv`, 25-40 MB Jaeger `traces.csv`, and `inject_time.txt`.
- **DejaVu (`A1`/`A2`/`B`/`C`/`D`):** Real commercial bank & enterprise cloud production
  failure telemetry (`metrics.csv`, `faults.csv`, `graph.yml`).

Anti-leakage & ResolveAI SRE-1 invariants enforced:
1. Blind incident IDs: directory/file names (`carts_cpu`, `adservice_f3`) are NEVER exposed.
2. Strict Time-Pinning: every metric/log/trace query is pinned to `[t_inject - 300s, t_inject + 240s]`.
3. Evidence-Grounded Verifiable Grading: requires >=2 real telemetry queries prior to
   `submit_rca_report`, plus verifiable matching of `(root_service, fault_type)` and
   evidence grounding against real queried observations.
"""

from __future__ import annotations

import csv
import json
import random
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from sre_rl.env.output_limiter import bound_tool_output
from sre_rl.model.thinking import parse_qwen38_assistant_turn_multi
from sre_rl.schemas import ToolCall


class RealIncidentRecord(BaseModel):
    """Metadata pointer for one real recorded production/K8s incident on disk."""

    case_id: str
    source_dataset: str           # e.g. RE2-SS, RE3-OB, RE3-TT, DejaVu-A1, DejaVu-C
    system_name: str              # sock-shop, online-boutique, train-ticket, bank-a1, oracle-db-c
    inject_timestamp: int         # Unix seconds of fault onset
    ground_truth_service: str     # e.g. carts, adservice, docker_003, bjclpdb01
    ground_truth_fault: str       # e.g. cpu, mem, disk, delay, loss, f3, Row Lock Contention
    metrics_path: str
    simple_metrics_path: str | None = None
    logs_path: str | None = None
    traces_path: str | None = None
    pod_node_path: str | None = None
    graph_path: str | None = None
    is_dejavu_long_format: bool = False
    single_entity_alert_host: str | None = None  # Set for single-host DB corpora (e.g. DejaVu-C)
    candidate_services: list[str] = Field(default_factory=list)


@dataclass
class RealStepOutcome:
    observation: str
    done: bool
    syntax_valid: bool = True
    step_shaping_reward: float = 0.0
    tool_calls_executed: int = 0


@dataclass
class RealRCARewardBreakdown:
    total: float
    service_exact_score: float = 0.0
    fault_type_score: float = 0.0
    evidence_grounding_bonus: float = 0.0
    efficiency_penalty: float = 0.0
    solved: bool = False
    predicted_service: str = ""
    predicted_fault: str = ""


REAL_TELEMETRY_ALIAS_BANKS: dict[str, tuple[str, ...]] = {
    "list_services": ("list_services", "k8s_service_inventory", "topology_list_entities", "cmdb_list_services"),
    "query_metric_anomalies": ("query_metric_anomalies", "promql_top_anomalies", "datadog_anomaly_rank", "grafana_detect_spikes"),
    "query_service_metrics": ("query_service_metrics", "promql_service_kpis", "datadog_host_metrics", "metrics_get_series"),
    "search_logs": ("search_logs", "loki_search_logs", "splunk_grep_events", "kibana_query_logs"),
    "query_traces": ("query_traces", "jaeger_trace_spans", "otel_slow_traces", "tempo_query_spans"),
    "submit_rca_report": ("submit_rca_report", "pagerduty_submit_rca", "close_incident_rca", "resolve_root_cause"),
}

_REAL_PARALLEL_HINTS: tuple[str, ...] = (
    "You may emit 1-4 parallel `<tool_call>` blocks per turn for fast triage.",
    "Batch independent read-only telemetry queries (metrics + logs + traces) in one turn (up to 4 parallel `<tool_call>` blocks).",
    "Concurrent read execution is supported: inspect topology and top anomalies together on turn 1.",
    "Run parallel read-only queries to narrow root cause quickly before calling the final submission tool.",
)


@dataclass(frozen=True)
class RealTelemetryHarnessConfig:
    """SRE-1 style tool-schema randomization & dual-budget harness for real telemetry RLVR."""

    is_canonical: bool = True
    max_turns: int = 24
    max_context_tokens: int = 12_288
    show_budget_footer: bool = True
    parallel_hint: str = _REAL_PARALLEL_HINTS[0]
    canonical_to_alias: dict[str, str] = field(default_factory=dict)
    alias_to_canonical: dict[str, str] = field(default_factory=dict)

    def tool_name(self, canonical: str) -> str:
        return self.canonical_to_alias.get(canonical, canonical)

    def resolve_tool_name(self, raw_name: str) -> str:
        norm = raw_name.strip().lower()
        return self.alias_to_canonical.get(norm, norm)

    def build_system_prompt(self) -> str:
        t_list = self.tool_name("list_services")
        t_anom = self.tool_name("query_metric_anomalies")
        t_svc = self.tool_name("query_service_metrics")
        t_logs = self.tool_name("search_logs")
        t_trc = self.tool_name("query_traces")
        t_sub = self.tool_name("submit_rca_report")
        return (
            "You are OmniSRE, an autonomous Site Reliability Engineer investigating a REAL production incident "
            "using live-captured observability telemetry (Prometheus/DB metrics, application logs, and distributed traces).\n\n"
            f"Available Tools ({self.parallel_hint}):\n"
            f"- `{t_list}`: `{{}}` — list services/entities and pod-to-node placement.\n"
            f"- `{t_anom}`: `{{\"service\": \"\", \"top_k\": 15}}` — rank KPIs by incident z-score vs. pre-incident baseline.\n"
            f"- `{t_svc}`: `{{\"service\": \"<name>\", \"metric_filter\": \"\"}}` — inspect baseline vs. incident stats for one service/host.\n"
            f"- `{t_logs}`: `{{\"service\": \"\", \"pattern\": \"\", \"errors_only\": true}}` — grep real container logs within the alert window.\n"
            f"- `{t_trc}`: `{{\"service\": \"\", \"errors_only\": false, \"min_duration_ms\": 0}}` — inspect slow/failing distributed trace spans.\n"
            f"- `{t_sub}`: `{{\"root_service\": \"<exact_service>\", \"fault_type\": \"<cpu|mem|disk|delay|loss|socket|code_fault|lock_contention|other>\", "
            "\"causal_chain\": \"...\", \"confidence\": 0.9}` — finalize diagnosis.\n\n"
            f"Protocol: Run >=2 telemetry queries before calling `{t_sub}`. Never guess from filenames. "
            f"Budget: max_turns={self.max_turns}, max_context_tokens={self.max_context_tokens}.\n"
            "Wrap concise reasoning in `<think>...</think>` followed by `<tool_call>{...}</tool_call>`."
        )

    def format_budget_footer(self, turn_idx: int, used_tokens: int) -> str:
        if not self.show_budget_footer:
            return ""
        rem_turns = max(0, self.max_turns - (turn_idx + 1))
        rem_toks = max(0, self.max_context_tokens - used_tokens)
        sub_name = self.tool_name("submit_rca_report")
        footer = f"\n[budget: turns_remaining={rem_turns}/{self.max_turns} | tokens_remaining={rem_toks}]"
        if rem_turns <= 2:
            footer += f"\n[WARNING: Budget nearly exhausted — call `{sub_name}` now.]"
        return footer


def sample_real_telemetry_harness(
    rng: random.Random,
    default_max_turns: int = 24,
    default_max_tokens: int = 12_288,
    randomize_prob: float = 0.60,
) -> RealTelemetryHarnessConfig:
    """Sample canonical (40%) or randomized MCP/observability tool aliases & dual budgets (60%)."""
    if rng.random() >= randomize_prob:
        return RealTelemetryHarnessConfig(
            is_canonical=True,
            max_turns=default_max_turns,
            max_context_tokens=default_max_tokens,
            show_budget_footer=True,
            parallel_hint=_REAL_PARALLEL_HINTS[0],
        )

    c2a: dict[str, str] = {}
    a2c: dict[str, str] = {}
    for canon, choices in REAL_TELEMETRY_ALIAS_BANKS.items():
        picked = rng.choice(choices)
        c2a[canon] = picked
        a2c[picked.lower()] = canon

    turn_choices = tuple(sorted({max(8, default_max_turns // 2), max(12, default_max_turns - 6), default_max_turns}))
    ctx_choices = tuple(sorted({max(6144, default_max_tokens - 4096), max(8192, default_max_tokens - 2048), default_max_tokens}))
    return RealTelemetryHarnessConfig(
        is_canonical=False,
        max_turns=rng.choice(turn_choices),
        max_context_tokens=rng.choice(ctx_choices),
        show_budget_footer=rng.random() < 0.85,
        parallel_hint=rng.choice(_REAL_PARALLEL_HINTS),
        canonical_to_alias=c2a,
        alias_to_canonical=a2c,
    )


def build_real_telemetry_system_prompt(
    max_turns: int = 12,
    harness: RealTelemetryHarnessConfig | None = None,
) -> str:
    if harness is not None:
        return harness.build_system_prompt()
    return RealTelemetryHarnessConfig(is_canonical=True, max_turns=max_turns).build_system_prompt()


def _normalize_service_token(name: str) -> str:
    cleaned = name.strip().lower()
    for prefix in ("metric.platform.", "platform.", "pod/", "svc/", "deployment/"):
        cleaned = cleaned.removeprefix(prefix)
    return re.sub(r"[^a-z0-9]", "", cleaned)


def _canonicalize_single_fault(token: str) -> set[str]:
    t = token.strip().lower()
    if not t or re.fullmatch(r"\d+", t):
        return {"other"}

    out: set[str] = set()
    if any(k in t for k in ("cpu", "throttl")):
        out.add("cpu")
    if any(k in t for k in ("mem", "oom", "heap")):
        out.add("mem")
    if any(k in t for k in ("logfile", "dbfile", "asm control", "disk", "io", "storage", "fs", "redo")):
        out.add("disk")
    if "dbfile" in t:
        out.add("lock_contention")
    if any(k in t for k in ("delay", "latenc", "slow", "timeout")):
        out.add("delay")
    if any(k in t for k in ("loss", "drop", "packet")):
        out.add("loss")
    if "network" in t and not ({"delay", "loss"} & out):
        # Generic "OS Network" fault without delay/loss subtype accepts any network-class diagnosis
        out.update({"delay", "loss", "socket"})
    if any(k in t for k in ("socket", "conn", "port", "tcp")):
        out.add("socket")
    if any(
        k in t
        for k in (
            "lock",
            "contention",
            "deadlock",
            "global cache",
            "buffer busy",
            "cachelock",
            "execution",
            "parse",
            "sql",
            "db close",
            "db  close",
            "db connection",
        )
    ):
        out.add("lock_contention")
    if re.fullmatch(r"f\d+", t) or any(k in t for k in ("code", "bug", "app", "exception", "500")):
        out.add("code_fault")

    return out or {t}


def _canonicalize_fault_set(raw: str) -> set[str]:
    """Map single or compound (`;`-delimited) real fault labels into the canonical fault taxonomy."""
    parts = [p.strip() for p in re.split(r"[;,|/]+", raw) if p.strip()]
    if not parts:
        return {"other"}
    merged: set[str] = set()
    for part in parts:
        merged.update(_canonicalize_single_fault(part))
    return merged or {"other"}


def _canonicalize_fault_type(raw: str) -> str:
    """Backwards-compatible primary canonical fault label."""
    fset = _canonicalize_fault_set(raw)
    for pref in ("cpu", "mem", "disk", "delay", "loss", "socket", "lock_contention", "code_fault", "other"):
        if pref in fset:
            return pref
    return min(fset)


@dataclass
class _LoadedMetricsCache:
    """Compact numerical representation of one incident's real time-series metrics."""

    services: list[str]
    # List of (service, metric_name, pre_mean, pre_std, inc_mean, inc_p95, inc_max, z_score)
    summary_rows: list[tuple[str, str, float, float, float, float, float, float]] = field(
        default_factory=list
    )


# Process-wide thread-safe caches so 24 concurrent rollout threads NEVER re-parse
# 75 MB DejaVu CSVs or re-scan identical Prometheus files across G=8 group rollouts!
_CACHE_LOCK = threading.Lock()
_DEJAVU_RAW_TABLE_CACHE: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
_CASE_SUMMARY_CACHE: dict[tuple[str, int, int, int, str], _LoadedMetricsCache] = {}
_CACHE_STATS: dict[str, int] = {"dejavu_file_parses": 0, "case_cache_hits": 0}


def clear_real_telemetry_caches() -> None:
    with _CACHE_LOCK:
        _DEJAVU_RAW_TABLE_CACHE.clear()
        _CASE_SUMMARY_CACHE.clear()
        _CACHE_STATS["dejavu_file_parses"] = 0
        _CACHE_STATS["case_cache_hits"] = 0


def get_real_telemetry_cache_stats() -> dict[str, int]:
    with _CACHE_LOCK:
        return dict(_CACHE_STATS)


class RealTelemetryReplayEnvironment:
    """Interactive multi-turn RLVR environment backed by real CSV telemetry files on disk."""

    def __init__(
        self,
        max_turns: int = 12,
        pre_window_s: int = 300,
        post_window_s: int = 240,
        max_output_chars: int = 3500,
        harness: RealTelemetryHarnessConfig | None = None,
    ) -> None:
        self.harness = harness or RealTelemetryHarnessConfig(is_canonical=True, max_turns=max_turns)
        self.max_turns = self.harness.max_turns if harness is not None else max_turns
        self.pre_window_s = pre_window_s
        self.post_window_s = post_window_s
        self.max_output_chars = max_output_chars

        self._record: RealIncidentRecord | None = None
        self._metrics_cache: _LoadedMetricsCache | None = None
        self._turns_taken = 0
        self._read_queries = 0
        self._queried_modalities: set[str] = set()
        self._seen_calls: set[str] = set()
        self._duplicate_calls = 0
        self._syntax_errors = 0
        self._submission: dict[str, Any] | None = None
        self._done = False

    def reset(self, record: RealIncidentRecord) -> str:
        self._record = record
        self._metrics_cache = None
        self._turns_taken = 0
        self._read_queries = 0
        self._queried_modalities = set()
        self._seen_calls = set()
        self._duplicate_calls = 0
        self._syntax_errors = 0
        self._submission = None
        self._done = False

        modalities = ["prometheus_metrics"]
        if record.logs_path and Path(record.logs_path).exists():
            modalities.append("container_logs")
        if record.traces_path and Path(record.traces_path).exists():
            modalities.append("distributed_traces")

        host_line = (
            f"Alerted Host/Instance: {record.single_entity_alert_host}\n"
            if record.single_entity_alert_host
            else ""
        )
        return (
            f"[REAL PRODUCTION ALERT] Case: {record.case_id}\n"
            f"System: {record.system_name}\n"
            f"{host_line}"
            f"Alert Onset Timestamp (Unix): {record.inject_timestamp}\n"
            f"Available Telemetry: {', '.join(modalities)}\n"
            "Symptom: Anomalous SLI degradation (elevated latency / error budget burn / wait-event spike) detected.\n"
            "Investigate the real telemetry and pinpoint the root-cause service and fault type."
        )

    @property
    def record(self) -> RealIncidentRecord:
        if self._record is None:
            raise RuntimeError("Call reset(record) before stepping.")
        return self._record

    def _ensure_metrics_loaded(self) -> _LoadedMetricsCache:
        if self._metrics_cache is not None:
            return self._metrics_cache

        rec = self.record
        t0 = rec.inject_timestamp
        t_start = t0 - self.pre_window_s
        t_end = t0 + self.post_window_s

        if rec.is_dejavu_long_format:
            raw_p = Path(rec.metrics_path)
            norm_p = raw_p.with_name("metrics.norm.csv")
            target_p = norm_p if norm_p.exists() else raw_p
        else:
            mpath = Path(rec.simple_metrics_path or rec.metrics_path)
            target_p = mpath if mpath.exists() else Path(rec.metrics_path)

        default_host = rec.single_entity_alert_host or "cluster-global"
        cache_key = (str(target_p), t_start, t0, t_end, default_host)
        with _CACHE_LOCK:
            cached = _CASE_SUMMARY_CACHE.get(cache_key)
            if cached is not None:
                _CACHE_STATS["case_cache_hits"] += 1
                self._metrics_cache = cached
                return cached

        if rec.is_dejavu_long_format:
            cache = self._load_dejavu_long_metrics(target_p, t_start, t0, t_end, default_host=default_host)
        else:
            cache = self._load_wide_prometheus_metrics(target_p, t_start, t0, t_end)

        with _CACHE_LOCK:
            # Cap case summary cache at 1,500 entries (~30 MB RAM)
            if len(_CASE_SUMMARY_CACHE) > 1500:
                _CASE_SUMMARY_CACHE.clear()
            _CASE_SUMMARY_CACHE[cache_key] = cache

        self._metrics_cache = cache
        return cache

    @staticmethod
    def _load_wide_prometheus_metrics(
        path: Path,
        t_start: int,
        t_inject: int,
        t_end: int,
    ) -> _LoadedMetricsCache:
        if not path.exists():
            return _LoadedMetricsCache(services=[])

        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            if len(header) < 2:
                return _LoadedMetricsCache(services=[])

            col_names = header[1:]
            pre_vals: list[list[float]] = [[] for _ in col_names]
            inc_vals: list[list[float]] = [[] for _ in col_names]

            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    ts = int(float(row[0]))
                except ValueError:
                    continue
                if ts < t_start or ts > t_end:
                    continue
                bucket = pre_vals if ts < t_inject else inc_vals
                for i, raw_v in enumerate(row[1 : len(col_names) + 1]):
                    if not raw_v:
                        continue
                    try:
                        v = float(raw_v)
                        if np.isfinite(v):
                            bucket[i].append(v)
                    except ValueError:
                        continue

        services_set: set[str] = set()
        summary_rows: list[tuple[str, str, float, float, float, float, float, float]] = []

        for idx, col in enumerate(col_names):
            # RCAEval columns look like `carts_cpu`, `adservice_container-cpu-system-seconds-total`
            svc = col.split("_", 1)[0] if "_" in col else col
            services_set.add(svc)
            p_arr = np.array(pre_vals[idx], dtype=np.float64)
            i_arr = np.array(inc_vals[idx], dtype=np.float64)
            if p_arr.size < 3 or i_arr.size < 3:
                continue
            p_mean = float(np.mean(p_arr))
            p_std = float(np.std(p_arr))
            i_mean = float(np.mean(i_arr))
            i_p95 = float(np.percentile(i_arr, 95))
            i_max = float(np.max(i_arr))
            # Robust floor avoids sparse 0.0 -> 0.002 jitter tying at z=999
            denom = max(p_std, abs(p_mean) * 0.10, 0.15)
            z_score = abs(i_mean - p_mean) / denom
            z_p95 = abs(i_p95 - p_mean) / denom
            score = float(min(250.0, max(z_score, 0.75 * z_p95)))
            summary_rows.append((svc, col, p_mean, p_std, i_mean, i_p95, i_max, score))

        summary_rows.sort(key=lambda r: r[7], reverse=True)
        return _LoadedMetricsCache(services=sorted(services_set), summary_rows=summary_rows)

    @staticmethod
    def _get_or_parse_dejavu_table(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        key = str(path)
        with _CACHE_LOCK:
            existing = _DEJAVU_RAW_TABLE_CACHE.get(key)
            if existing is not None:
                return existing

        # Parse once per CSV file into sorted (timestamps_int64, values_float32) arrays
        raw_ts: dict[str, list[int]] = {}
        raw_vals: dict[str, list[float]] = {}
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    ts = int(float(row["timestamp"]))
                    val = float(row["value"])
                except (KeyError, ValueError):
                    continue
                if not np.isfinite(val):
                    continue
                name = str(row.get("name", "")).strip()
                if not name:
                    continue
                raw_ts.setdefault(name, []).append(ts)
                raw_vals.setdefault(name, []).append(val)

        indexed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, ts_list in raw_ts.items():
            ts_arr = np.array(ts_list, dtype=np.int64)
            v_arr = np.array(raw_vals[name], dtype=np.float32)
            order = np.argsort(ts_arr)
            indexed[name] = (ts_arr[order], v_arr[order])

        with _CACHE_LOCK:
            _DEJAVU_RAW_TABLE_CACHE[key] = indexed
            _CACHE_STATS["dejavu_file_parses"] += 1
        return indexed

    @classmethod
    def _load_dejavu_long_metrics(
        cls,
        path: Path,
        t_start: int,
        t_inject: int,
        t_end: int,
        default_host: str = "cluster-global",
    ) -> _LoadedMetricsCache:
        """Query pre-indexed in-memory DejaVu table in <3ms using binary search (`searchsorted`)."""
        if not path.exists():
            return _LoadedMetricsCache(services=[])

        table = cls._get_or_parse_dejavu_table(path)
        services_set: set[str] = set()
        summary_rows: list[tuple[str, str, float, float, float, float, float, float]] = []

        for metric_key, (ts_arr, v_arr) in table.items():
            i0 = int(np.searchsorted(ts_arr, t_start, side="left"))
            i_mid = int(np.searchsorted(ts_arr, t_inject, side="left"))
            i1 = int(np.searchsorted(ts_arr, t_end, side="right"))

            p_arr = v_arr[i0:i_mid]
            i_arr = v_arr[i_mid:i1]
            if p_arr.size < 2 or i_arr.size < 2:
                continue

            svc = metric_key.split("##", 1)[0] or default_host
            services_set.add(svc)
            p_mean = float(np.mean(p_arr))
            p_std = float(np.std(p_arr))
            i_mean = float(np.mean(i_arr))
            i_p95 = float(np.percentile(i_arr, 95))
            i_max = float(np.max(i_arr))
            denom = max(p_std, abs(p_mean) * 0.10, 0.25)
            local_z = max(abs(i_mean - p_mean) / denom, 0.75 * abs(i_p95 - p_mean) / denom)
            z = float(min(250.0, local_z))
            summary_rows.append((svc, metric_key, p_mean, p_std, i_mean, i_p95, i_max, z))

        summary_rows.sort(key=lambda r: r[7], reverse=True)
        return _LoadedMetricsCache(services=sorted(services_set), summary_rows=summary_rows)

    def step(self, raw_assistant_turn: str) -> RealStepOutcome:
        if self._done:
            return RealStepOutcome(observation="Episode already complete.", done=True)

        self._turns_taken += 1
        parsed = parse_qwen38_assistant_turn_multi(raw_assistant_turn, max_parallel_calls=6)
        if parsed.parse_error is not None or not parsed.tool_calls:
            self._syntax_errors += 1
            done = self._turns_taken >= self.max_turns
            self._done = done
            return RealStepOutcome(
                observation=f"ERROR: Invalid tool syntax ({parsed.parse_error or 'no_tool_call'}).",
                done=done,
                syntax_valid=False,
                step_shaping_reward=-0.05,
            )

        obs_chunks: list[str] = []
        shaping = 0.0
        executed = 0

        for call in parsed.tool_calls:
            executed += 1
            sig = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
            if sig in self._seen_calls:
                self._duplicate_calls += 1
                shaping -= 0.04
            self._seen_calls.add(sig)

            obs, call_done, delta = self._dispatch_tool(call)
            obs_chunks.append(obs)
            shaping += delta
            if call_done:
                self._done = True
                break

        if self._turns_taken >= self.max_turns:
            self._done = True

        combined = bound_tool_output("\n---\n".join(obs_chunks), max_chars=self.max_output_chars)
        return RealStepOutcome(
            observation=combined,
            done=self._done,
            syntax_valid=True,
            step_shaping_reward=round(shaping, 4),
            tool_calls_executed=executed,
        )

    def _dispatch_tool(self, call: ToolCall) -> tuple[str, bool, float]:
        name = self.harness.resolve_tool_name(call.name)
        args = call.arguments
        rec = self.record

        if name == "list_services":
            cache = self._ensure_metrics_loaded()
            svcs = cache.services or rec.candidate_services
            pod_info = ""
            if rec.pod_node_path and Path(rec.pod_node_path).exists():
                pod_lines = Path(rec.pod_node_path).read_text(encoding="utf-8", errors="ignore").splitlines()[:25]
                pod_info = "\nPod placement:\n" + "\n".join(pod_lines)
            self._read_queries += 1
            self._queried_modalities.add("topology")
            return json.dumps({"system": rec.system_name, "services": svcs}) + pod_info, False, 0.01

        if name == "query_metric_anomalies":
            cache = self._ensure_metrics_loaded()
            svc_filter = str(args.get("service", "")).strip().lower()
            top_k = min(25, max(1, int(args.get("top_k", 15))))
            rows = [
                {
                    "service": r[0],
                    "metric": r[1],
                    "baseline_mean": round(r[2], 4),
                    "incident_mean": round(r[4], 4),
                    "incident_p95": round(r[5], 4),
                    "z_score": round(r[7], 2),
                }
                for r in cache.summary_rows
                if not svc_filter or svc_filter in r[0].lower() or svc_filter in r[1].lower()
            ][:top_k]
            self._read_queries += 1
            self._queried_modalities.add("metrics")
            return json.dumps({"top_anomalies": rows}, indent=2), False, 0.02

        if name == "query_service_metrics":
            cache = self._ensure_metrics_loaded()
            svc = str(args.get("service", "")).strip().lower()
            mfilt = str(args.get("metric_filter", "")).strip().lower()
            rows = [
                {
                    "metric": r[1],
                    "baseline_mean": round(r[2], 4),
                    "incident_mean": round(r[4], 4),
                    "incident_p95": round(r[5], 4),
                    "incident_max": round(r[6], 4),
                    "z_score": round(r[7], 2),
                }
                for r in cache.summary_rows
                if (not svc or svc in r[0].lower() or svc in r[1].lower())
                and (not mfilt or mfilt in r[1].lower())
            ][:18]
            self._read_queries += 1
            self._queried_modalities.add("metrics")
            hit_root = _normalize_service_token(svc) == _normalize_service_token(rec.ground_truth_service)
            return json.dumps({"service": svc, "metrics": rows}, indent=2), False, (0.03 if hit_root else 0.01)

        if name == "search_logs":
            self._read_queries += 1
            self._queried_modalities.add("logs")
            if not rec.logs_path or not Path(rec.logs_path).exists():
                return "No container logs recorded for this system; inspect metrics/traces.", False, 0.0
            return self._query_real_logs(
                Path(rec.logs_path),
                service=str(args.get("service", "")).strip().lower(),
                pattern=str(args.get("pattern", "")).strip(),
                errors_only=bool(args.get("errors_only", False)),
                limit=min(40, max(5, int(args.get("limit", 25)))),
            ), False, 0.02

        if name == "query_traces":
            self._read_queries += 1
            self._queried_modalities.add("traces")
            if not rec.traces_path or not Path(rec.traces_path).exists():
                return "No distributed traces recorded for this system; inspect metrics/logs.", False, 0.0
            return self._query_real_traces(
                Path(rec.traces_path),
                service=str(args.get("service", "")).strip().lower(),
                errors_only=bool(args.get("errors_only", False)),
                min_duration_ms=float(args.get("min_duration_ms", 0.0)),
            ), False, 0.02

        if name == "submit_rca_report":
            if self._read_queries < 2:
                msg = (
                    f"REJECTED: Only {self._read_queries} telemetry queries executed; "
                    "gather >=2 observations before submitting root cause."
                )
                return msg, False, -0.10
            self._submission = dict(args)
            return "RCA_REPORT_RECORDED. Investigation complete.", True, 0.0

        return f"Unknown tool: {name}", False, -0.05

    def _query_real_logs(
        self,
        path: Path,
        service: str,
        pattern: str,
        errors_only: bool,
        limit: int,
    ) -> str:
        t0 = self.record.inject_timestamp
        t_min, t_max = t0 - 60, t0 + self.post_window_s
        try:
            regex = re.compile(pattern, re.IGNORECASE) if pattern else None
        except re.error as exc:
            return f"REGEX_ERROR: {exc}"

        err_re = re.compile(r"(error|err=|warn|fail|timeout|refused|5\d\d|exception|panic|oom)", re.IGNORECASE)
        hits: list[str] = []
        scanned = 0

        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                scanned += 1
                if scanned > 80_000:
                    break
                # Timestamp in RCAEval logs.csv is nanoseconds (`1705595459012034902`) or seconds
                raw_ts = row.get("timestamp", "")
                if raw_ts:
                    try:
                        ts_val = int(float(raw_ts))
                        if ts_val > 10**15:
                            ts_val //= 10**9
                        elif ts_val > 10**12:
                            ts_val //= 10**3
                        if ts_val < t_min or ts_val > t_max:
                            continue
                    except ValueError:
                        pass

                svc = str(row.get("container_name") or row.get("service") or "")
                msg = str(row.get("message") or "")
                lvl = str(row.get("level") or "")
                err_col = str(row.get("error") or "")

                if service and service not in svc.lower():
                    continue
                if errors_only and not (err_col or err_re.search(f"{lvl} {msg}")):
                    continue
                if regex and not regex.search(msg):
                    continue
                hits.append(f"[{svc}] {lvl} {msg[:240]}".strip())
                if len(hits) >= limit:
                    break

        if not hits:
            return "No matching log lines in incident window. Try broadening service/pattern or setting errors_only=false."
        return "\n".join(hits)

    def _query_real_traces(
        self,
        path: Path,
        service: str,
        errors_only: bool,
        min_duration_ms: float,
    ) -> str:
        t0 = self.record.inject_timestamp
        t_min, t_max = t0 - 60, t0 + self.post_window_s
        op_durations: dict[tuple[str, str], list[float]] = {}
        op_errors: dict[tuple[str, str], int] = {}
        scanned = 0

        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                scanned += 1
                if scanned > 120_000:
                    break
                raw_ts = row.get("startTimeMillis") or row.get("startTime") or ""
                if raw_ts:
                    try:
                        ts_val = int(float(raw_ts))
                        if ts_val > 10**15:
                            ts_val //= 10**6
                        elif ts_val > 10**12:
                            ts_val //= 10**3
                        if ts_val < t_min or ts_val > t_max:
                            continue
                    except ValueError:
                        pass

                svc = str(row.get("serviceName") or "")
                if service and service not in svc.lower():
                    continue
                op = str(row.get("operationName") or row.get("methodName") or "unknown")
                try:
                    # Jaeger duration in traces.csv is microseconds
                    dur_ms = float(row.get("duration", 0.0)) / 1000.0
                except ValueError:
                    dur_ms = 0.0
                status = str(row.get("statusCode", "")).strip()
                is_err = status not in ("", "0", "0.0", "200", "200.0", "OK")

                if errors_only and not is_err:
                    continue
                if dur_ms < min_duration_ms:
                    continue

                key = (svc, op)
                op_durations.setdefault(key, []).append(dur_ms)
                if is_err:
                    op_errors[key] = op_errors.get(key, 0) + 1

        summary = []
        for (svc, op), durs in op_durations.items():
            arr = np.array(durs, dtype=np.float64)
            summary.append(
                {
                    "service": svc,
                    "operation": op[:80],
                    "spans": int(arr.size),
                    "errors": op_errors.get((svc, op), 0),
                    "p50_ms": round(float(np.percentile(arr, 50)), 2),
                    "p95_ms": round(float(np.percentile(arr, 95)), 2),
                    "max_ms": round(float(np.max(arr)), 2),
                }
            )
        summary.sort(key=lambda r: (r["errors"] > 0, r["p95_ms"]), reverse=True)
        return json.dumps({"trace_summary": summary[:20]}, indent=2)

    def score(self) -> RealRCARewardBreakdown:
        rec = self.record
        if self._submission is None:
            # Stall timeout penalty
            return RealRCARewardBreakdown(total=-0.25, efficiency_penalty=0.25, solved=False)

        pred_svc = str(self._submission.get("root_service", "")).strip()
        pred_fault = str(self._submission.get("fault_type", "")).strip()
        gt_svc_norm = _normalize_service_token(rec.ground_truth_service)
        pred_svc_norm = _normalize_service_token(pred_svc)

        # 1. Root-Cause Service Match
        svc_score = 0.0
        if pred_svc_norm and (pred_svc_norm == gt_svc_norm):
            svc_score = 0.65
        elif pred_svc_norm and (gt_svc_norm in pred_svc_norm or pred_svc_norm in gt_svc_norm):
            svc_score = 0.50

        # 2. Fault Type Match (handles compound `;`-delimited & Oracle wait-event fault taxonomies)
        gt_fault_set = _canonicalize_fault_set(rec.ground_truth_fault)
        pred_fault_set = _canonicalize_fault_set(pred_fault)
        fault_matched = bool(pred_fault_set & gt_fault_set)
        fault_score = 0.25 if fault_matched else 0.0

        # For single-entity DB instances (e.g. DejaVu-C where the alerted host is given in the alert),
        # re-weight so the model MUST diagnose the true wait-event fault mechanism to earn high reward!
        if rec.single_entity_alert_host:
            svc_weight = 0.25 if svc_score >= 0.50 else 0.0
            fault_weight = 0.65 if fault_matched else 0.0
            grounding_bonus = 0.10 if (fault_matched and self._read_queries >= 2) else 0.0
            eff_pen = min(0.35, 0.04 * self._duplicate_calls + 0.05 * self._syntax_errors)
            total = round(max(-0.5, svc_weight + fault_weight + grounding_bonus - eff_pen), 4)
            solved = (svc_score >= 0.50) and (fault_matched or gt_fault_set == {"other"})
            return RealRCARewardBreakdown(
                total=total,
                service_exact_score=svc_score,
                fault_type_score=fault_score,
                evidence_grounding_bonus=grounding_bonus,
                efficiency_penalty=round(eff_pen, 4),
                solved=solved,
                predicted_service=pred_svc,
                predicted_fault=pred_fault,
            )

        # 3. Multi-Modal Evidence Grounding Bonus (max 0.10, gated on correct service!)
        grounding_bonus = 0.0
        if svc_score >= 0.50 and len(self._queried_modalities) >= 2:
            grounding_bonus = 0.10
        elif svc_score >= 0.50 and self._read_queries >= 2:
            grounding_bonus = 0.05

        eff_pen = min(0.35, 0.04 * self._duplicate_calls + 0.05 * self._syntax_errors)
        total = round(max(-0.5, svc_score + fault_score + grounding_bonus - eff_pen), 4)
        solved = svc_score >= 0.50 and (
            fault_matched or ("code_fault" in gt_fault_set) or (gt_fault_set == {"other"})
        )

        return RealRCARewardBreakdown(
            total=total,
            service_exact_score=svc_score,
            fault_type_score=fault_score,
            evidence_grounding_bonus=grounding_bonus,
            efficiency_penalty=round(eff_pen, 4),
            solved=solved,
            predicted_service=pred_svc,
            predicted_fault=pred_fault,
        )
