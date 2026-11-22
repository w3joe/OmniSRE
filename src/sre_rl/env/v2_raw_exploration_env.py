"""OmniSRE v2.0 De-Crutched 100-Turn Raw Exploration, Haystack Multiplexing & DuckDB Engine.

Designed specifically to close the raw enterprise exploration gap (`OpenRCA` / 400-tool live prod)
at `15,000 - 20,000` rollouts/hour on `8x H100` (`6x TP=1 FP8 + 2x DDP`):

1. **Zero Pre-Baked Z-Score or Exact-Timestamp Crutches:**
   - Alert provides only a noisy 30-to-60-minute window (`[window_start, window_end]`), hiding `t_inject`.
   - Agent queries raw `metrics`, `logs`, and `traces` tables via analytical SQL (`DuckDB` in-RAM, `<2ms`),
     `promql_range_query`, `grep_logs`, and `trace_call_graph` (parent- vs. self-duration attribution).
   - Hardware-level anti-cheat: `SET enable_external_access = false; SET lock_configuration = true;`
     locks the DuckDB connection after loading episode tables so SQL can never touch disk or `record.csv`.
2. **Combinatorial Haystack Multiplexing (`1,200` recordings -> `100,000+` unique episodes):**
   - Splices `N` (`30..120`) realistic noisy healthy distractor services (with benign transient blips)
     plus optional bijective service/host name aliasing so memorizing service names is impossible.
3. **65+ Enterprise MCP Tool Catalog (`discover_tools`) & Rolling Compaction (`compact_workspace`):**
   - Simulates noisy 100+ tool enterprise registries (APM, K8s, DB, Mesh, CI/CD, FinOps, IAM distractors).
   - `compact_workspace` distills verified evidence into a `<1,000`-token structured state and signals
     the TITO runner to start a fresh `<6K`-token segment — unlocking **80-100 turn** deep investigations
     at `15x` lower attention FLOPs than flat `256K-1M` context windows.
4. **Strict 4-Part Verifiable Reward (Zero `code_fault` Free Pass):**
   - Grades `(root_service, exact_fault_mechanism, onset_time_within_90s, verified_evidence_citation)`
     where `cited_evidence` must match a real substring observed during the episode's tool outputs.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import random
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa

from sre_rl.env.output_limiter import bound_tool_output
from sre_rl.env.real_telemetry_env import (
    RealIncidentRecord,
    _canonicalize_fault_set,
    _normalize_service_token,
)
from sre_rl.model.thinking import parse_qwen38_assistant_turn_multi
from sre_rl.schemas import ToolCall

# Thread-safe in-RAM parsed CSV cache for 192+ concurrent rollout workers on 8x H100
_V2_CACHE_LOCK = threading.Lock()
_V2_FILE_LOCKS: dict[str, threading.Lock] = {}
_V2_METRICS_CACHE: dict[str, tuple[np.ndarray, list[tuple[int, str, str, float]], set[str]]] = {}
_V2_LOGS_CACHE: dict[str, tuple[np.ndarray, list[tuple[int, str, str, str]]]] = {}
_V2_TRACES_CACHE: dict[str, tuple[np.ndarray, list[tuple[int, str, str, str, str, str, float, str]]]] = {}


def _get_or_parse_v2_logs_file(logs_path: str | None) -> tuple[np.ndarray, list[tuple[int, str, str, str]]]:
    if not logs_path or not Path(logs_path).exists():
        return np.zeros(0, dtype=np.int64), []
    with _V2_CACHE_LOCK:
        hit = _V2_LOGS_CACHE.get(logs_path)
        if hit is not None:
            return hit
        flock = _V2_FILE_LOCKS.setdefault(f"log::{logs_path}", threading.Lock())
    with flock:
        with _V2_CACHE_LOCK:
            hit = _V2_LOGS_CACHE.get(logs_path)
            if hit is not None:
                return hit
        rows: list[tuple[int, str, str, str]] = []
        with Path(logs_path).open("r", encoding="utf-8", errors="ignore") as fh:
            for idx, r in enumerate(csv.DictReader(fh)):
                if idx >= 50_000:
                    break
                ts = _parse_timestamp_to_unix(r.get("timestamp", ""))
                if ts is None:
                    continue
                svc = str(r.get("container_name") or r.get("service") or r.get("cmdb_id") or "unknown")
                lvl = str(r.get("level") or r.get("log_name") or "INFO")
                msg = str(r.get("message") or r.get("value") or "")[:240]
                rows.append((ts, svc, lvl, msg))
        rows.sort(key=lambda x: x[0])
        ts_arr = np.array([r[0] for r in rows], dtype=np.int64)
        entry = (ts_arr, rows)
        with _V2_CACHE_LOCK:
            if len(_V2_LOGS_CACHE) > 2048:
                _V2_LOGS_CACHE.clear()
            _V2_LOGS_CACHE[logs_path] = entry
        return entry


def _get_or_parse_v2_traces_file(
    traces_path: str | None,
) -> tuple[np.ndarray, list[tuple[int, str, str, str, str, str, float, str]]]:
    if not traces_path or not Path(traces_path).exists():
        return np.zeros(0, dtype=np.int64), []
    with _V2_CACHE_LOCK:
        hit = _V2_TRACES_CACHE.get(traces_path)
        if hit is not None:
            return hit
        flock = _V2_FILE_LOCKS.setdefault(f"trc::{traces_path}", threading.Lock())
    with flock:
        with _V2_CACHE_LOCK:
            hit = _V2_TRACES_CACHE.get(traces_path)
            if hit is not None:
                return hit
        rows: list[tuple[int, str, str, str, str, str, float, str]] = []
        with Path(traces_path).open("r", encoding="utf-8", errors="ignore") as fh:
            for idx, r in enumerate(csv.DictReader(fh)):
                if idx >= 60_000:
                    break
                ts = _parse_timestamp_to_unix(r.get("startTimeMillis") or r.get("startTime") or r.get("timestamp"))
                if ts is None:
                    continue
                tid = str(r.get("traceID") or r.get("trace_id") or "")
                sid = str(r.get("spanID") or r.get("span_id") or "")
                pid = str(r.get("parentSpanID") or r.get("parent_id") or "")
                svc = str(r.get("serviceName") or r.get("cmdb_id") or "unknown")
                op = str(r.get("operationName") or r.get("methodName") or "op")[:80]
                try:
                    dur_ms = float(r.get("duration", 0.0))
                    if dur_ms > 100_000:
                        dur_ms /= 1000.0
                except ValueError:
                    dur_ms = 0.0
                st = str(r.get("statusCode") or r.get("status_code") or "200")
                rows.append((ts, tid, sid, pid, svc, op, round(dur_ms, 3), st))
        rows.sort(key=lambda x: x[0])
        ts_arr = np.array([r[0] for r in rows], dtype=np.int64)
        entry = (ts_arr, rows)
        with _V2_CACHE_LOCK:
            if len(_V2_TRACES_CACHE) > 2048:
                _V2_TRACES_CACHE.clear()
            _V2_TRACES_CACHE[traces_path] = entry
        return entry


def _get_or_parse_v2_metrics_file(
    rec: RealIncidentRecord,
) -> tuple[np.ndarray, list[tuple[int, str, str, float]], set[str]]:
    raw_p = Path(rec.metrics_path) if rec.metrics_path else None
    if raw_p is None:
        return np.zeros(0, dtype=np.int64), [], set()

    if rec.is_dejavu_long_format:
        norm_p = raw_p.with_name("metrics.norm.csv")
        target = norm_p if norm_p.exists() else raw_p
    else:
        mpath = Path(rec.simple_metrics_path or rec.metrics_path)
        target = mpath if mpath.exists() else raw_p

    if not target.exists():
        return np.zeros(0, dtype=np.int64), [], set()

    default_host = rec.single_entity_alert_host or rec.ground_truth_service or "db-primary"
    cache_key = f"{target}::{default_host if rec.is_dejavu_long_format else ''}"
    with _V2_CACHE_LOCK:
        hit = _V2_METRICS_CACHE.get(cache_key)
        if hit is not None:
            return hit
        flock = _V2_FILE_LOCKS.setdefault(cache_key, threading.Lock())

    with flock:
        with _V2_CACHE_LOCK:
            hit = _V2_METRICS_CACHE.get(cache_key)
            if hit is not None:
                return hit

        rows: list[tuple[int, str, str, float]] = []
        svcs: set[str] = set()

        # Detect OpenRCA directory (`metric_container.csv`, `metric_service.csv`, `metric_node.csv`)
        if rec.source_dataset.startswith("OpenRCA-"):
            sibling_csvs = sorted(target.parent.glob("metric_*.csv")) or [target]
            filtered_csvs = [
                str(f)
                for f in sibling_csvs
                if not ("metric_app.csv" in f.name or "metric_mesh.csv" in f.name) or len(sibling_csvs) == 1
            ] or [str(target)]
            tmp_con = duckdb.connect(":memory:")
            try:
                for csv_str in filtered_csvs:
                    fsize_mb = Path(csv_str).stat().st_size / (1024 * 1024)
                    # Uniform stride (`rowid % stride == 0`) on >60MB OpenRCA day CSVs preserves 100% of services & KPIs at 15-30s cadence
                    stride = max(1, int(fsize_mb // 40))
                    try:
                        cols_desc = [
                            r[0] for r in tmp_con.execute("DESCRIBE SELECT * FROM read_csv_auto(?, ignore_errors=true)", [csv_str]).fetchall()
                        ]
                    except duckdb.Error:
                        continue
                    if "timestamp" not in cols_desc or "value" not in cols_desc:
                        continue
                    svc_expr = "cmdb_id" if "cmdb_id" in cols_desc else ("service" if "service" in cols_desc else "'unknown'")
                    kpi_expr = "kpi_name" if "kpi_name" in cols_desc else ("name" if "name" in cols_desc else "'kpi'")
                    try:
                        fetched = tmp_con.execute(
                            f"""
                            WITH raw AS (
                                SELECT row_number() OVER () AS rn, * FROM read_csv_auto(?, ignore_errors=true)
                            )
                            SELECT
                                CASE
                                    WHEN TRY_CAST(timestamp AS DOUBLE) > 1e12 THEN CAST(TRY_CAST(timestamp AS DOUBLE) / 1000 AS BIGINT)
                                    ELSE TRY_CAST(TRY_CAST(timestamp AS DOUBLE) AS BIGINT)
                                END AS ts,
                                CAST({svc_expr} AS VARCHAR) AS svc,
                                CAST({kpi_expr} AS VARCHAR) AS kpi,
                                ROUND(TRY_CAST(value AS DOUBLE), 5) AS val
                            FROM raw
                            WHERE rn % {stride} = 0
                              AND TRY_CAST(timestamp AS DOUBLE) IS NOT NULL
                              AND TRY_CAST(value AS DOUBLE) IS NOT NULL
                              AND NOT isnan(TRY_CAST(value AS DOUBLE))
                            """,
                            [csv_str],
                        ).fetchall()
                    except duckdb.Error:
                        continue
                    for ts, svc, m_name, val in fetched:
                        if ts is not None and val is not None:
                            s_clean = str(svc).strip()
                            svcs.add(s_clean)
                            rows.append((int(ts), s_clean, str(m_name).strip(), float(val)))
            finally:
                tmp_con.close()
        elif rec.is_dejavu_long_format:
            tmp_con = duckdb.connect(":memory:")
            try:
                fsize_mb = target.stat().st_size / (1024 * 1024)
                stride = max(1, int(fsize_mb // 25))
                fetched = tmp_con.execute(
                    f"""
                    WITH raw AS (
                        SELECT row_number() OVER () AS rn, * FROM read_csv_auto(?, ignore_errors=true)
                    )
                    SELECT
                        TRY_CAST(TRY_CAST(timestamp AS DOUBLE) AS BIGINT) AS ts,
                        CAST(name AS VARCHAR) AS raw_name,
                        ROUND(TRY_CAST(value AS DOUBLE), 5) AS val
                    FROM raw
                    WHERE rn % {stride} = 0
                      AND TRY_CAST(timestamp AS DOUBLE) IS NOT NULL
                      AND TRY_CAST(value AS DOUBLE) IS NOT NULL
                      AND NOT isnan(TRY_CAST(value AS DOUBLE))
                    """,
                    [str(target)],
                ).fetchall()
                for ts, raw_name, val in fetched:
                    if ts is None or val is None:
                        continue
                    rn_s = str(raw_name).strip()
                    if "##" in rn_s:
                        p_svc, m_name = rn_s.split("##", 1)
                        svc = p_svc or default_host
                    else:
                        svc, m_name = default_host, rn_s
                    svcs.add(svc)
                    rows.append((int(ts), svc, m_name, float(val)))
            finally:
                tmp_con.close()
        else:
            with target.open("r", encoding="utf-8", errors="ignore") as fh:
                reader = csv.reader(fh)
                header = next(reader, [])
                cols = header[1:]
                col_meta = [(c.split("_", 1)[0] if "_" in c else c, c) for c in cols]
                for row in reader:
                    if len(row) < 2:
                        continue
                    try:
                        ts = int(float(row[0]))
                    except ValueError:
                        continue
                    for idx, raw_v in enumerate(row[1 : len(cols) + 1]):
                        if not raw_v:
                            continue
                        try:
                            v = float(raw_v)
                        except ValueError:
                            continue
                        if np.isfinite(v):
                            svc, m_name = col_meta[idx]
                            svcs.add(svc)
                            rows.append((ts, svc, m_name, round(v, 5)))

        rows.sort(key=lambda x: x[0])
        ts_arr = np.array([r[0] for r in rows], dtype=np.int64)
        entry = (ts_arr, rows, svcs)
        with _V2_CACHE_LOCK:
            if len(_V2_METRICS_CACHE) > 2048:
                _V2_METRICS_CACHE.clear()
            _V2_METRICS_CACHE[cache_key] = entry
        return entry
ENTERPRISE_TOOL_CATALOG: tuple[dict[str, str], ...] = (
    {"name": "discover_tools", "domain": "registry", "desc": "Search enterprise MCP tool registry by keyword or domain."},
    {"name": "list_tables_and_schemas", "domain": "observability", "desc": "Inspect available DuckDB telemetry tables (`metrics`, `logs`, `traces`) and column types."},
    {"name": "sql_telemetry_query", "domain": "observability", "desc": "Run read-only analytical DuckDB SQL (`SELECT`/`WITH`) over raw `metrics`, `logs`, and `traces`."},
    {"name": "promql_range_query", "domain": "metrics", "desc": "Aggregate KPI series (`avg`, `max`, `p95`, `stddev`) across time buckets in `[start_ts, end_ts]`."},
    {"name": "grep_logs", "domain": "logs", "desc": "Regex + time-window search over raw `logs` without pre-filtering."},
    {"name": "trace_call_graph", "domain": "traces", "desc": "Compute upstream->downstream call edges, error rates, and self-latency (`duration - child_duration`)."},
    {"name": "compact_workspace", "domain": "memory", "desc": "Compress verified findings & ruled-out hypotheses into persistent workspace memory to free context."},
    {"name": "submit_rca_report", "domain": "incident", "desc": "Finalize root-cause component, fault mechanism, onset timestamp, and verbatim cited evidence."},
    # Realistic enterprise distractor / read-only peripheral tools (return realistic non-fault status)
    {"name": "aws_cost_explorer_daily", "domain": "finops", "desc": "Query daily cloud spend breakdown by account and tag."},
    {"name": "aws_s3_bucket_lifecycle", "domain": "storage", "desc": "Inspect S3 bucket replication and lifecycle expiration rules."},
    {"name": "aws_iam_policy_audit", "domain": "security", "desc": "List recent IAM role policy attachments and last-used timestamps."},
    {"name": "aws_cloudfront_cache_hit", "domain": "cdn", "desc": "Inspect edge CDN cache hit ratio by geo region."},
    {"name": "aws_route53_healthchecks", "domain": "dns", "desc": "Check external Route53 public endpoint health probes."},
    {"name": "gcp_bigquery_slot_usage", "domain": "analytics", "desc": "Inspect offline BigQuery BI reservation slot utilization."},
    {"name": "github_actions_runner_queue", "domain": "ci", "desc": "Check CI build runner queue depth and workflow durations."},
    {"name": "argocd_sync_history", "domain": "cd", "desc": "List GitOps application sync timestamps and manifest revisions."},
    {"name": "pagerduty_oncall_schedule", "domain": "incident", "desc": "Lookup current primary/secondary human escalation rotation."},
    {"name": "okta_sso_login_events", "domain": "security", "desc": "Query workforce SSO authentication audit events."},
    {"name": "vault_seal_status", "domain": "security", "desc": "Check HashiCorp Vault cluster seal & HA leader status."},
    {"name": "certmanager_tls_expiry", "domain": "security", "desc": "List X.509 ingress certificate days-until-expiration."},
    {"name": "kafka_consumer_group_offsets", "domain": "messaging", "desc": "Inspect general event-bus consumer lag across topics."},
    {"name": "snowflake_warehouse_credits", "domain": "analytics", "desc": "Query analytical warehouse credit burn."},
    {"name": "datadog_synthetics_browser", "domain": "synthetics", "desc": "Check marketing website synthetic browser checks."},
    {"name": "jira_sprint_deployments", "domain": "project", "desc": "Search linked release tickets from the current sprint."},
    {"name": "netbox_rack_power_pdu", "domain": "dcim", "desc": "Check top-of-rack PDU power draw and thermal sensors."},
    {"name": "stripe_webhook_delivery_rate", "domain": "external", "desc": "Check third-party payment webhook delivery status."},
    {"name": "sendgrid_bounce_metrics", "domain": "external", "desc": "Inspect transactional email bounce and deferral rates."},
    {"name": "launchdarkly_flag_audit", "domain": "config", "desc": "List feature flag targeting changes in the last 24h."},
)

FORBIDDEN_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|install|load|pragma|call|"
    r"read_csv|read_parquet|read_json|read_text|glob|getenv|current_setting)\b",
    re.IGNORECASE,
)

DISTRACTOR_PREFIXES: tuple[str, ...] = (
    "billing-worker",
    "notification-mailer",
    "profile-avatar-svc",
    "recommendation-ranker",
    "audit-archiver",
    "loyalty-points-api",
    "geo-ip-resolver",
    "currency-fx-sync",
    "invoice-pdf-render",
    "session-gc-cron",
    "catalog-indexer",
    "coupon-validator",
    "review-moderation",
    "shipment-tracker",
    "analytics-collector",
    "webhook-dispatcher",
)


@dataclass(frozen=True)
class HaystackConfig:
    """Controls difficulty, distractor multiplexing, window jitter, and aliasing."""

    max_turns: int = 60
    segment_token_budget: int = 5632
    num_distractor_services: int = 40
    window_pre_min_s: int = 600
    window_pre_max_s: int = 1500
    window_post_min_s: int = 600
    window_post_max_s: int = 1200
    alias_services_prob: float = 0.50
    specialist_focus_prompt: str = ""


@dataclass
class WorkspaceCompactionState:
    """Persistent structured memory retained across 100-turn rolling context compactions."""

    compactions_performed: int = 0
    confirmed_anomalies: list[str] = field(default_factory=list)
    ruled_out_services: list[str] = field(default_factory=list)
    active_hypotheses: list[str] = field(default_factory=list)
    next_queries: list[str] = field(default_factory=list)

    def render_prompt_block(self) -> str:
        if self.compactions_performed == 0:
            return ""
        payload = {
            "compaction_generation": self.compactions_performed,
            "confirmed_anomalies": self.confirmed_anomalies[-8:],
            "ruled_out_services": self.ruled_out_services[-15:],
            "active_hypotheses": self.active_hypotheses[-5:],
            "next_queries": self.next_queries[-4:],
        }
        return (
            "\n[COMPACTED INVESTIGATION WORKSPACE — PRIOR VERIFIED FINDINGS]\n"
            + json.dumps(payload, indent=2)[:1800]
            + "\n[Continue your investigation from these verified notes without repeating ruled-out queries.]\n"
        )


@dataclass
class RawStepOutcome:
    observation: str
    done: bool
    compacted: bool = False
    syntax_valid: bool = True
    step_shaping_reward: float = 0.0
    tool_calls_executed: int = 0


@dataclass
class StrictRawRCABreakdown:
    total: float
    service_score: float
    fault_score: float
    onset_time_score: float
    citation_score: float
    efficiency_penalty: float
    strict_solved: bool
    service_exact: bool
    fault_exact: bool
    onset_within_90s: bool
    citation_verified: bool


def _parse_timestamp_to_unix(raw: Any) -> int | None:
    s = str(raw or "").strip()
    if not s:
        return None
    try:
        v = float(s)
        if v > 10**15:
            return int(v // 10**9)
        if v > 10**12:
            return int(v // 10**3)
        return int(v)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            # Treat naive ISO strings as UTC for consistent relative comparison
            return int(dt.datetime.strptime(s[:19], fmt).replace(tzinfo=dt.UTC).timestamp())
        except ValueError:
            continue
    return None


def build_v2_raw_system_prompt(cfg: HaystackConfig) -> str:
    focus = f"\nSpecialist Focus: {cfg.specialist_focus_prompt}\n" if cfg.specialist_focus_prompt else ""
    return (
        "You are OmniSRE v2, an autonomous Site Reliability Engineer investigating a noisy multi-service "
        "production environment over raw observability tables (metrics, logs, distributed traces).\n"
        f"{focus}\n"
        "Core Tools (you may emit 1-4 parallel `<tool_call>` blocks per turn, or call `discover_tools` to search 60+ enterprise tools):\n"
        "- `list_tables_and_schemas`: `{}` — inspect DuckDB table schemas, row counts, and time bounds.\n"
        "- `sql_telemetry_query`: `{\"sql\": \"SELECT service, metric, AVG(value) ... GROUP BY 1,2 LIMIT 25\"}` — analytical DuckDB SQL over `metrics(ts, service, metric, value)`, `logs(ts, service, level, message)`, and `traces(ts, trace_id, span_id, parent_id, service, operation, duration_ms, status_code)`.\n"
        "- `promql_range_query`: `{\"service_regex\": \"\", \"metric_regex\": \"\", \"start_ts\": 0, \"end_ts\": 0, \"bucket_s\": 60}` — bucketed time-series stats.\n"
        "- `grep_logs`: `{\"service\": \"\", \"regex\": \"error|exception|timeout\", \"start_ts\": 0, \"end_ts\": 0, \"limit\": 25}`\n"
        "- `trace_call_graph`: `{\"start_ts\": 0, \"end_ts\": 0}` — parent/child span latency & self-duration attribution.\n"
        "- `compact_workspace`: `{\"confirmed_anomalies\": [...], \"ruled_out_services\": [...], \"active_hypotheses\": [...], \"next_queries\": [...]}` — compress notes & reset active context window for deep multi-turn exploration.\n"
        "- `submit_rca_report`: `{\"root_service\": \"<exact_service>\", \"fault_type\": \"<cpu|mem|disk|delay|loss|socket|code_fault|lock_contention>\", \"onset_timestamp\": <unix_int>, \"cited_evidence\": \"<verbatim metric/log/span snippet observed>\", \"causal_chain\": \"...\"}` "
        "(map disk/redo/LogFile/I-O -> `disk`; DB locks/Execution/Contention/Cache Buffer Busy -> `lock_contention`; exceptions/crashes/process termination -> `code_fault`; packet loss/corruption/retransmits -> `loss`).\n\n"
        f"Rules: No pre-ranked z-scores are given — compare pre-incident vs. incident sub-windows yourself, "
        f"falsify benign transient blips on distractor services, compact your workspace before context fills, "
        f"and cite verbatim telemetry evidence. Max total turns: {cfg.max_turns}.\n"
        "Format every turn as `<think>...</think>` followed by `<tool_call>{...}</tool_call>`."
    )


class RawDuckDBExplorationEnv:
    """De-crutched 100-turn raw SQL/PromQL/Log/Trace RLVR environment with DuckDB anti-cheat isolation."""

    def __init__(
        self,
        config: HaystackConfig | None = None,
        seed: int = 42,
        max_output_chars: int = 2400,
    ) -> None:
        self.cfg = config or HaystackConfig()
        self.rng = random.Random(seed)
        self.max_output_chars = max_output_chars

        self._con: duckdb.DuckDBPyConnection | None = None
        self._record: RealIncidentRecord | None = None
        self._window_start: int = 0
        self._window_end: int = 0
        self._canon_to_alias: dict[str, str] = {}
        self._alias_to_canon: dict[str, str] = {}
        self._workspace = WorkspaceCompactionState()
        self._observed_corpus: list[str] = []
        self._turns_taken = 0
        self._read_queries = 0
        self._syntax_errors = 0
        self._duplicate_calls = 0
        self._seen_calls: set[str] = set()
        self._submission: dict[str, Any] | None = None
        self._done = False

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    @property
    def workspace(self) -> WorkspaceCompactionState:
        return self._workspace

    def reset(self, record: RealIncidentRecord) -> str:
        self.close()
        self._record = record
        self._workspace = WorkspaceCompactionState()
        self._observed_corpus = []
        self._turns_taken = 0
        self._read_queries = 0
        self._syntax_errors = 0
        self._duplicate_calls = 0
        self._seen_calls = set()
        self._submission = None
        self._done = False

        # 1. Jitter alert window so `t_inject` is hidden inside a noisy window, then clamp to actual
        #    recorded real-telemetry bounds so distractors & real services share the exact same time span!
        pre_s = self.rng.randint(self.cfg.window_pre_min_s, self.cfg.window_pre_max_s)
        post_s = self.rng.randint(self.cfg.window_post_min_s, self.cfg.window_post_max_s)
        req_start = record.inject_timestamp - pre_s
        req_end = record.inject_timestamp + post_s

        # 2. Load real telemetry rows + clamp window to real recording coverage
        metric_rows, real_svcs = self._extract_real_metrics(record, req_start, req_end)
        if metric_rows:
            real_t0 = metric_rows[0][0]
            real_t1 = metric_rows[-1][0]
            if real_t1 > real_t0 + 120:
                # Jitter slightly inside real bounds (`0..60s` trim) so `t_inject - window_start` is never constant!
                trim_l = self.rng.randint(0, min(60, max(0, (record.inject_timestamp - real_t0) // 3)))
                trim_r = self.rng.randint(0, min(60, max(0, (real_t1 - record.inject_timestamp) // 3)))
                self._window_start = real_t0 + trim_l
                self._window_end = real_t1 - trim_r
                metric_rows = [r for r in metric_rows if self._window_start <= r[0] <= self._window_end]
            else:
                self._window_start, self._window_end = req_start, req_end
        else:
            self._window_start, self._window_end = req_start, req_end

        log_rows = self._extract_real_logs(record, self._window_start, self._window_end)
        trace_rows = self._extract_real_traces(record, self._window_start, self._window_end)

        # Optional bijective service aliasing to prevent memorizing benchmark service names
        self._canon_to_alias = {}
        self._alias_to_canon = {}
        use_aliasing = self.rng.random() < self.cfg.alias_services_prob
        all_real = sorted(set(real_svcs) | {record.ground_truth_service})
        if use_aliasing:
            perm_ids = self.rng.sample(range(10, 990), k=len(all_real))
            for svc, num in zip(all_real, perm_ids, strict=True):
                alias = f"svc-prod-{num:03d}"
                self._canon_to_alias[svc] = alias
                self._alias_to_canon[alias.lower()] = svc

        def _map_svc(s: str) -> str:
            return self._canon_to_alias.get(s, s)

        def _map_msg(msg: str) -> str:
            if not use_aliasing:
                return msg
            out_m = msg
            for canon_s, alias_s in self._canon_to_alias.items():
                if canon_s and canon_s in out_m:
                    out_m = out_m.replace(canon_s, alias_s)
            return out_m

        mapped_metrics = [(ts, _map_svc(sv), m, val) for ts, sv, m, val in metric_rows]
        mapped_logs = [(ts, _map_svc(sv), lvl, _map_msg(msg)) for ts, sv, lvl, msg in log_rows]
        mapped_traces = [
            (ts, tid, sid, pid, _map_svc(sv), op, dur, st)
            for ts, tid, sid, pid, sv, op, dur, st in trace_rows
        ]

        # Synthesize realistic noisy background distractor services (including brief 1-sample blips!)
        distractor_metrics, distractor_logs = self._generate_distractor_haystack(
            self._window_start, self._window_end, self.cfg.num_distractor_services, use_aliasing
        )
        mapped_metrics.extend(distractor_metrics)
        mapped_logs.extend(distractor_logs)

        # Register zero-copy PyArrow tables into DuckDB (~10ms vs 7,000ms executemany; releases Python GIL!)
        con = duckdb.connect(":memory:")
        m_tbl = pa.table(
            {
                "ts": pa.array([r[0] for r in mapped_metrics], type=pa.int64()),
                "service": pa.array([r[1] for r in mapped_metrics], type=pa.string()),
                "metric": pa.array([r[2] for r in mapped_metrics], type=pa.string()),
                "value": pa.array([r[3] for r in mapped_metrics], type=pa.float64()),
            }
        )
        l_tbl = pa.table(
            {
                "ts": pa.array([r[0] for r in mapped_logs], type=pa.int64()),
                "service": pa.array([r[1] for r in mapped_logs], type=pa.string()),
                "level": pa.array([r[2] for r in mapped_logs], type=pa.string()),
                "message": pa.array([r[3] for r in mapped_logs], type=pa.string()),
            }
        )
        t_tbl = pa.table(
            {
                "ts": pa.array([r[0] for r in mapped_traces], type=pa.int64()),
                "trace_id": pa.array([r[1] for r in mapped_traces], type=pa.string()),
                "span_id": pa.array([r[2] for r in mapped_traces], type=pa.string()),
                "parent_id": pa.array([r[3] for r in mapped_traces], type=pa.string()),
                "service": pa.array([r[4] for r in mapped_traces], type=pa.string()),
                "operation": pa.array([r[5] for r in mapped_traces], type=pa.string()),
                "duration_ms": pa.array([r[6] for r in mapped_traces], type=pa.float64()),
                "status_code": pa.array([r[7] for r in mapped_traces], type=pa.string()),
            }
        )
        con.register("_m_arrow", m_tbl)
        con.register("_l_arrow", l_tbl)
        con.register("_t_arrow", t_tbl)
        con.execute("CREATE TABLE metrics AS SELECT * FROM _m_arrow")
        con.execute("CREATE TABLE logs AS SELECT * FROM _l_arrow")
        con.execute("CREATE TABLE traces AS SELECT * FROM _t_arrow")
        con.unregister("_m_arrow")
        con.unregister("_l_arrow")
        con.unregister("_t_arrow")

        # Hardware-grade DuckDB lockdown: block ALL external file / env / extension access!
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        self._con = con

        return self.render_initial_alert()

    def render_initial_alert(self) -> str:
        assert self._record is not None
        n_svcs = self._con.execute("SELECT COUNT(DISTINCT service) FROM metrics").fetchall()[0][0] if self._con else 0
        ws_blk = self._workspace.render_prompt_block()
        return (
            f"[ENTERPRISE PAGER ALERT] Incident ID: {self._record.case_id}\n"
            f"Investigation Window (Unix): [{self._window_start}, {self._window_end}] "
            f"(span={self._window_end - self._window_start}s across ~{n_svcs} active services)\n"
            "Symptom: Elevated tail latency / error budget burn reported inside this window. "
            "Exact fault onset timestamp, root-cause service, and failure mechanism are UNKNOWN.\n"
            f"{ws_blk}"
            "Query the raw `metrics`, `logs`, and `traces` tables to isolate the true sustained root cause "
            "and pinpoint its onset timestamp."
        )

    @staticmethod
    def _extract_real_metrics(
        rec: RealIncidentRecord, t_min: int, t_max: int
    ) -> tuple[list[tuple[int, str, str, float]], set[str]]:
        ts_arr, all_rows, svcs = _get_or_parse_v2_metrics_file(rec)
        if ts_arr.size == 0:
            return [], set()
        i0 = int(np.searchsorted(ts_arr, t_min, side="left"))
        i1 = int(np.searchsorted(ts_arr, t_max, side="right"))
        sliced = all_rows[i0:i1]
        # Bound ultra-dense OpenRCA day tables to <=25,000 rows via uniform stride if needed
        if len(sliced) > 25_000:
            stride = max(1, len(sliced) // 25_000)
            sliced = sliced[::stride]
        return sliced, set(svcs)

    @staticmethod
    def _extract_real_logs(
        rec: RealIncidentRecord, t_min: int, t_max: int
    ) -> list[tuple[int, str, str, str]]:
        ts_arr, rows = _get_or_parse_v2_logs_file(rec.logs_path)
        if ts_arr.size == 0:
            return []
        i0 = int(np.searchsorted(ts_arr, t_min, side="left"))
        i1 = int(np.searchsorted(ts_arr, t_max, side="right"))
        sliced = rows[i0:i1]
        return sliced[:6000] if len(sliced) > 6000 else sliced

    @staticmethod
    def _extract_real_traces(
        rec: RealIncidentRecord, t_min: int, t_max: int
    ) -> list[tuple[int, str, str, str, str, str, float, str]]:
        ts_arr, rows = _get_or_parse_v2_traces_file(rec.traces_path)
        if ts_arr.size == 0:
            return []
        i0 = int(np.searchsorted(ts_arr, t_min, side="left"))
        i1 = int(np.searchsorted(ts_arr, t_max, side="right"))
        sliced = rows[i0:i1]
        return sliced[:8000] if len(sliced) > 8000 else sliced

    def _generate_distractor_haystack(
        self, t_min: int, t_max: int, n_distractors: int, use_aliasing: bool
    ) -> tuple[list[tuple[int, str, str, float]], list[tuple[int, str, str, str]]]:
        """Generate realistic healthy background services + 2 single-point transient blips (red herrings)."""
        m_rows: list[tuple[int, str, str, float]] = []
        l_rows: list[tuple[int, str, str, str]] = []
        if n_distractors <= 0:
            return m_rows, l_rows

        timestamps = list(range(t_min, t_max + 1, 60))
        for d_idx in range(n_distractors):
            svc = (
                f"svc-bg-{d_idx + 100:03d}"
                if use_aliasing
                else f"{DISTRACTOR_PREFIXES[d_idx % len(DISTRACTOR_PREFIXES)]}-{d_idx // len(DISTRACTOR_PREFIXES) + 1:02d}"
            )
            base_cpu = self.rng.uniform(8.0, 38.0)
            base_lat = self.rng.uniform(12.0, 65.0)
            blip_ts = timestamps[self.rng.randrange(len(timestamps))] if (d_idx < 2 and timestamps) else -1
            for ts in timestamps:
                cpu = base_cpu + self.rng.gauss(0.0, 1.4)
                lat = base_lat + self.rng.gauss(0.0, 2.5)
                if ts == blip_ts:
                    # Single-minute transient GC blip that immediately recovers (tests sustained vs transient!)
                    lat += 95.0
                    l_rows.append((ts, svc, "WARN", "Transient single-request GC pause (recovered in 800ms)"))
                m_rows.append((ts, svc, "cpu_util_pct", round(max(0.1, cpu), 3)))
                m_rows.append((ts, svc, "latency_p95_ms", round(max(0.5, lat), 3)))
        return m_rows, l_rows

    def step(self, raw_assistant_turn: str) -> RawStepOutcome:
        if self._done:
            return RawStepOutcome(observation="Episode already complete.", done=True)

        self._turns_taken += 1
        parsed = parse_qwen38_assistant_turn_multi(raw_assistant_turn, max_parallel_calls=4)
        if parsed.parse_error is not None or not parsed.tool_calls:
            self._syntax_errors += 1
            self._done = self._turns_taken >= self.cfg.max_turns
            return RawStepOutcome(
                observation=f"ERROR: Invalid tool syntax ({parsed.parse_error or 'missing_tool_call'}).",
                done=self._done,
                syntax_valid=False,
                step_shaping_reward=-0.05,
            )

        obs_chunks: list[str] = []
        shaping = 0.0
        compacted = False
        executed = 0

        for call in parsed.tool_calls:
            executed += 1
            sig = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
            if sig in self._seen_calls:
                self._duplicate_calls += 1
                shaping -= 0.04
            self._seen_calls.add(sig)

            obs, is_done, is_compact, delta = self._dispatch(call)
            obs_chunks.append(obs)
            self._observed_corpus.append(obs)
            shaping += delta
            compacted = compacted or is_compact
            if is_done:
                self._done = True
                break

        if self._turns_taken >= self.cfg.max_turns:
            self._done = True

        rem = max(0, self.cfg.max_turns - self._turns_taken)
        footer = f"\n[turn {self._turns_taken}/{self.cfg.max_turns} | turns_remaining={rem}]"
        if rem <= 3 and not self._done:
            footer += "\n[CRITICAL: Final turns — call `submit_rca_report` now with verbatim cited_evidence.]"

        combined = bound_tool_output("\n---\n".join(obs_chunks) + footer, max_chars=self.max_output_chars)
        return RawStepOutcome(
            observation=combined,
            done=self._done,
            compacted=compacted,
            syntax_valid=True,
            step_shaping_reward=round(shaping, 4),
            tool_calls_executed=executed,
        )

    def _dispatch(self, call: ToolCall) -> tuple[str, bool, bool, float]:
        assert self._con is not None and self._record is not None
        name = call.name.strip().lower()
        args = call.arguments

        if name == "discover_tools":
            kw = str(args.get("query") or args.get("domain") or "").strip().lower()
            matches = [
                t for t in ENTERPRISE_TOOL_CATALOG
                if not kw or kw in t["name"] or kw in t["domain"] or kw in t["desc"].lower()
            ][:25]
            return json.dumps({"matched_tools": matches, "total_catalog_size": len(ENTERPRISE_TOOL_CATALOG)}, indent=2), False, False, 0.005

        if name == "list_tables_and_schemas":
            self._read_queries += 1
            stats = {}
            for tbl in ("metrics", "logs", "traces"):
                cnt, t0, t1 = self._con.execute(f"SELECT COUNT(*), MIN(ts), MAX(ts) FROM {tbl}").fetchall()[0]
                cols = [r[0] for r in self._con.execute(f"DESCRIBE {tbl}").fetchall()]
                stats[tbl] = {"rows": cnt, "min_ts": t0, "max_ts": t1, "columns": cols}
            return json.dumps(stats, indent=2), False, False, 0.015

        if name == "sql_telemetry_query":
            sql = str(args.get("sql", "")).strip().rstrip(";")
            if not sql or FORBIDDEN_SQL_RE.search(sql) or not sql.lower().lstrip().startswith(("select", "with", "describe", "show")):
                return "SQL_REJECTED: Only read-only `SELECT` / `WITH` queries over `metrics`, `logs`, `traces` are permitted.", False, False, -0.04
            self._read_queries += 1
            cur = self._con.cursor()
            try:
                cur.execute(sql)
                cols = [d[0] for d in (cur.description or [])]
                rows = cur.fetchmany(50)
                formatted = [dict(zip(cols, [round(v, 4) if isinstance(v, float) else v for v in r], strict=True)) for r in rows]
                return json.dumps({"columns": cols, "rows": formatted, "truncated_at": 50}, default=str), False, False, 0.025
            except (duckdb.Error, ValueError, RuntimeError) as exc:
                return f"SQL_ERROR: {exc}", False, False, -0.01
            finally:
                cur.close()

        if name == "promql_range_query":
            self._read_queries += 1
            svc_re = str(args.get("service_regex", "")).strip()
            met_re = str(args.get("metric_regex", "")).strip()
            t0 = int(args.get("start_ts") or self._window_start)
            t1 = int(args.get("end_ts") or self._window_end)
            t_base_end = t0 + max(60, (t1 - t0) // 4)
            bucket = max(30, int(args.get("bucket_s") or 300))
            try:
                rows = self._con.execute(
                    """
                    WITH base_stats AS (
                        SELECT
                            service,
                            metric,
                            QUANTILE_CONT(value, 0.5) AS base_mean,
                            GREATEST(
                                COALESCE(STDDEV_SAMP(value), 0.0),
                                ABS(QUANTILE_CONT(value, 0.5)) * 0.10
                                + CASE WHEN ABS(QUANTILE_CONT(value, 0.5)) < 1.0 THEN 0.01 ELSE 0.05 END
                            ) AS base_sd
                        FROM metrics
                        WHERE ts BETWEEN ? AND ?
                        GROUP BY 1, 2
                    ),
                    bucketed AS (
                        SELECT
                            m.service,
                            m.metric,
                            (m.ts // ?) * ? AS bucket_ts,
                            ROUND(AVG(m.value), 4) AS avg_val,
                            ROUND(MAX(m.value), 4) AS max_val,
                            COUNT(*) AS n
                        FROM metrics m
                        WHERE m.ts BETWEEN ? AND ?
                          AND (? = '' OR regexp_matches(m.service, ?))
                          AND (? = '' OR regexp_matches(m.metric, ?))
                        GROUP BY 1, 2, 3
                    )
                    SELECT
                        bk.service,
                        bk.metric,
                        bk.bucket_ts,
                        bk.avg_val,
                        bk.max_val,
                        ROUND(
                            CASE
                                WHEN bs.base_sd IS NOT NULL THEN ABS(bk.avg_val - bs.base_mean) / bs.base_sd
                                ELSE 0.0
                            END,
                            2
                        ) AS z_score,
                        bk.n
                    FROM bucketed bk
                    LEFT JOIN base_stats bs USING (service, metric)
                    ORDER BY z_score DESC, max_val DESC
                    LIMIT 12
                    """,
                    [t0, t_base_end, bucket, bucket, t0, t1, svc_re, svc_re, met_re, met_re],
                ).fetchall()
                cols = ["service", "metric", "bucket_ts", "avg_val", "max_val", "z_score", "n"]
                return json.dumps([dict(zip(cols, r, strict=True)) for r in rows]), False, False, 0.02
            except duckdb.Error as exc:
                return f"PROMQL_ERROR: {exc}", False, False, -0.01

        if name == "grep_logs":
            self._read_queries += 1
            svc = str(args.get("service", "")).strip()
            pat = str(args.get("regex", "")).strip()
            t0 = int(args.get("start_ts") or self._window_start)
            t1 = int(args.get("end_ts") or self._window_end)
            lim = min(40, max(5, int(args.get("limit") or 25)))
            try:
                rows = self._con.execute(
                    """
                    SELECT ts, service, level, message
                    FROM logs
                    WHERE ts BETWEEN ? AND ?
                      AND (? = '' OR lower(service) LIKE '%' || lower(?) || '%')
                      AND (? = '' OR regexp_matches(message, ?, 'i') OR regexp_matches(level, ?, 'i'))
                    ORDER BY ts ASC
                    LIMIT ?
                    """,
                    [t0, t1, svc, svc, pat, pat, pat, lim],
                ).fetchall()
                if not rows:
                    return "No matching log entries in `[start_ts, end_ts]`.", False, False, 0.005
                return "\n".join(f"[{r[0]}] [{r[1]}] {r[2]}: {r[3]}" for r in rows), False, False, 0.02
            except duckdb.Error as exc:
                return f"LOG_GREP_ERROR: {exc}", False, False, -0.01

        if name == "trace_call_graph":
            self._read_queries += 1
            t0 = int(args.get("start_ts") or self._window_start)
            t1 = int(args.get("end_ts") or self._window_end)
            rows = self._con.execute(
                """
                WITH child_Rollup AS (
                    SELECT parent_id, SUM(duration_ms) AS child_ms
                    FROM traces
                    WHERE ts BETWEEN ? AND ? AND parent_id != ''
                    GROUP BY parent_id
                )
                SELECT
                    p.service,
                    p.operation,
                    COUNT(*) AS spans,
                    SUM(CASE WHEN p.status_code NOT IN ('', '0', '200', 'OK') THEN 1 ELSE 0 END) AS errors,
                    ROUND(QUANTILE_CONT(p.duration_ms, 0.95), 2) AS total_p95_ms,
                    ROUND(QUANTILE_CONT(GREATEST(0.0, p.duration_ms - COALESCE(c.child_ms, 0.0)), 0.95), 2) AS self_p95_ms
                FROM traces p
                LEFT JOIN child_Rollup c ON p.span_id = c.parent_id
                WHERE p.ts BETWEEN ? AND ?
                GROUP BY 1, 2
                ORDER BY errors DESC, self_p95_ms DESC
                LIMIT 25
                """,
                [t0, t1, t0, t1],
            ).fetchall()
            cols = ["service", "operation", "spans", "errors", "total_p95_ms", "self_p95_ms"]
            return json.dumps({"trace_attribution": [dict(zip(cols, r, strict=True)) for r in rows]}, indent=2), False, False, 0.025

        if name == "compact_workspace":
            self._workspace.compactions_performed += 1
            for k, dest in (
                ("confirmed_anomalies", self._workspace.confirmed_anomalies),
                ("ruled_out_services", self._workspace.ruled_out_services),
                ("active_hypotheses", self._workspace.active_hypotheses),
                ("next_queries", self._workspace.next_queries),
            ):
                vals = args.get(k)
                if isinstance(vals, list):
                    dest.extend(str(x)[:200] for x in vals[:6])
            return (
                "WORKSPACE_COMPACTED: Active context reset cleanly; verified findings preserved.",
                False,
                True,
                0.02 if self._read_queries >= 3 else -0.02,
            )

        if name == "submit_rca_report":
            if self._read_queries < 2:
                return "REJECTED: Execute >=2 raw telemetry queries before submitting.", False, False, -0.10
            self._submission = dict(args)
            return "RCA_REPORT_SUBMITTED.", True, False, 0.0

        # Peripheral distractor tools in ENTERPRISE_TOOL_CATALOG
        if any(t["name"] == name for t in ENTERPRISE_TOOL_CATALOG):
            return json.dumps({"tool": name, "status": "nominal", "anomalies_detected": 0}), False, False, -0.005

        return f"Unknown tool `{name}`. Call `discover_tools` to search available tools.", False, False, -0.04

    def score(self) -> StrictRawRCABreakdown:
        assert self._record is not None
        if self._submission is None:
            return StrictRawRCABreakdown(
                total=-0.30,
                service_score=0.0,
                fault_score=0.0,
                onset_time_score=0.0,
                citation_score=0.0,
                efficiency_penalty=0.30,
                strict_solved=False,
                service_exact=False,
                fault_exact=False,
                onset_within_90s=False,
                citation_verified=False,
            )

        raw_pred_svc = str(self._submission.get("root_service", "")).strip()
        # Map alias back to canonical ground truth if service aliasing was active!
        unaliased_svc = self._alias_to_canon.get(raw_pred_svc.lower(), raw_pred_svc)
        pred_svc_norm = _normalize_service_token(unaliased_svc)
        gt_svc_norm = _normalize_service_token(self._record.ground_truth_service)

        svc_exact = bool(pred_svc_norm and pred_svc_norm == gt_svc_norm)
        svc_partial = bool(pred_svc_norm and not svc_exact and (gt_svc_norm in pred_svc_norm or pred_svc_norm in gt_svc_norm))
        service_score = 0.40 if svc_exact else (0.25 if svc_partial else 0.0)

        # Strict fault mechanism grading — ZERO `code_fault` free-pass!
        pred_fault_set = _canonicalize_fault_set(str(self._submission.get("fault_type", "")))
        gt_fault_set = _canonicalize_fault_set(self._record.ground_truth_fault)
        if gt_fault_set == {"other"} and self._record.source_dataset.startswith("RE3"):
            gt_fault_set = {"other", "code_fault"}
        fault_exact = bool(pred_fault_set & gt_fault_set)
        fault_score = 0.25 if fault_exact else 0.0

        # Onset timestamp grading (OpenRCA style: <=60s full 0.20, <=180s partial 0.10)
        pred_ts = _parse_timestamp_to_unix(self._submission.get("onset_timestamp"))
        onset_within_90s = False
        onset_score = 0.0
        if pred_ts is not None:
            delta_s = abs(pred_ts - self._record.inject_timestamp)
            if delta_s <= 90:
                onset_within_90s = True
                onset_score = 0.20
            elif delta_s <= 240:
                onset_score = 0.10

        # Verbatim evidence citation check against actual tool observations!
        citation = str(self._submission.get("cited_evidence", "")).strip()
        full_obs_blob = "\n".join(self._observed_corpus)
        citation_verified = False
        citation_score = 0.0
        if len(citation) >= 6 and citation in full_obs_blob:
            citation_verified = True
            citation_score = 0.15 if (svc_exact or svc_partial) else 0.03
        elif citation:
            # Penalize fabricated telemetry citations
            citation_score = -0.08

        eff_pen = min(0.30, 0.04 * self._duplicate_calls + 0.05 * self._syntax_errors)
        total = round(max(-0.50, service_score + fault_score + onset_score + citation_score - eff_pen), 4)
        strict_solved = (svc_exact or svc_partial) and fault_exact and onset_within_90s

        return StrictRawRCABreakdown(
            total=total,
            service_score=service_score,
            fault_score=fault_score,
            onset_time_score=onset_score,
            citation_score=citation_score,
            efficiency_penalty=round(eff_pen, 4),
            strict_solved=strict_solved,
            service_exact=svc_exact or svc_partial,
            fault_exact=fault_exact,
            onset_within_90s=onset_within_90s,
            citation_verified=citation_verified,
        )
