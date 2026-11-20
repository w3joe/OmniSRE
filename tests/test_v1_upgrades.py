from __future__ import annotations

import random
from pathlib import Path

from sre_rl.env.live_k8s_env import (
    DEFAULT_LIVE_K8S_FAULTS,
    KindClusterLease,
    LiveK8sKindEnvironment,
    sample_live_k8s_harness,
)
from sre_rl.env.real_telemetry_env import (
    RealIncidentRecord,
    RealTelemetryReplayEnvironment,
    _canonicalize_fault_set,
    clear_real_telemetry_caches,
    sample_real_telemetry_harness,
)


def test_fault_taxonomy_handles_dejavu_oracle_and_compound_labels() -> None:
    assert "disk" in _canonicalize_fault_set("LogFile")
    assert {"disk", "lock_contention"} <= _canonicalize_fault_set("DBFile;LogFile")
    assert "lock_contention" in _canonicalize_fault_set("Global Cache Buffer Busy")
    assert "lock_contention" in _canonicalize_fault_set("Row Lock Contention")
    assert "disk" in _canonicalize_fault_set("ASM Control File")
    assert {"delay", "loss", "socket"} <= _canonicalize_fault_set("OS Network")
    assert {"cpu", "mem"} <= _canonicalize_fault_set("OS Memory;JVM Memory;JVM CPU")
    assert _canonicalize_fault_set("1") == {"other"}


def test_single_entity_dejavu_c_and_60pct_mcp_schema_randomization(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics.norm.csv"
    rows = ["name,timestamp,value"]
    for i in range(6):
        ts = 3600 - 300 + i * 50
        rows.append(f"##log file sync,{ts},0.05")
        rows.append(f"##CPU,{ts},0.10")
    for i in range(6):
        ts = 3600 + i * 30
        rows.append(f"##log file sync,{ts},18.50")
        rows.append(f"##CPU,{ts},0.12")
    metrics_csv.write_text("\n".join(rows), encoding="utf-8")

    rec = RealIncidentRecord(
        case_id="real-dejavu-c-00001",
        source_dataset="DejaVu-C",
        system_name="prod-dejavu-c",
        inject_timestamp=3600,
        ground_truth_service="by11ccpdadb1002",
        ground_truth_fault="LogFile",
        metrics_path=str(metrics_csv),
        is_dejavu_long_format=True,
        single_entity_alert_host="by11ccpdadb1002",
        candidate_services=["by11ccpdadb1002"],
    )

    clear_real_telemetry_caches()
    rng = random.Random(7)
    harness = sample_real_telemetry_harness(rng, default_max_turns=24, randomize_prob=1.0)
    assert harness.is_canonical is False
    assert harness.max_turns <= 24

    env = RealTelemetryReplayEnvironment(harness=harness)
    alert = env.reset(rec)
    assert "Alerted Host/Instance: by11ccpdadb1002" in alert

    t_list = harness.tool_name("list_services")
    t_anom = harness.tool_name("query_metric_anomalies")
    t_sub = harness.tool_name("submit_rca_report")
    sys_prompt = harness.build_system_prompt()
    assert f"`{t_anom}`" in sys_prompt and f"`{t_sub}`" in sys_prompt

    out = env.step(
        f"<think>Run aliased MCP tools</think>\n"
        f'<tool_call>{{"name": "{t_list}", "arguments": {{}}}}</tool_call>\n'
        f'<tool_call>{{"name": "{t_anom}", "arguments": {{"top_k": 5}}}}</tool_call>'
    )
    assert "by11ccpdadb1002" in out.observation
    assert "log file sync" in out.observation

    # Submitting WRONG fault on single-entity DB host MUST NOT solve the episode
    env_bad = RealTelemetryReplayEnvironment(harness=harness)
    env_bad.reset(rec)
    env_bad.step(
        f'<think>Inspect</think><tool_call>{{"name": "{t_list}", "arguments": {{}}}}</tool_call>'
        f'<tool_call>{{"name": "{t_anom}", "arguments": {{}}}}</tool_call>'
    )
    env_bad.step(
        f'<think>Wrong guess</think><tool_call>{{"name": "{t_sub}", "arguments": '
        '{"root_service": "by11ccpdadb1002", "fault_type": "cpu"}}</tool_call>'
    )
    assert env_bad.score().solved is False

    # Submitting disk (Oracle LogFile redo I/O) solves it cleanly!
    env.step(
        f'<think>LogFile sync spike = disk I/O</think><tool_call>{{"name": "{t_sub}", "arguments": '
        '{"root_service": "by11ccpdadb1002", "fault_type": "disk", "causal_chain": "Oracle log file sync I/O spike"}}</tool_call>'
    )
    sc = env.score()
    assert sc.solved is True
    assert sc.total >= 0.95


def test_live_k8s_env_guardrails_block_premature_writes_and_injector_snooping(tmp_path: Path) -> None:
    lease = KindClusterLease(
        name="sre-rl-kind-00",
        admin_kubeconfig=tmp_path / "admin.kubeconfig",
        agent_kubeconfig=tmp_path / "agent.kubeconfig",
    )
    harness = sample_live_k8s_harness(random.Random(3), default_max_turns=20, randomize_prob=1.0)
    env = LiveK8sKindEnvironment(lease=lease, harness=harness)
    env._spec = DEFAULT_LIVE_K8S_FAULTS[0]

    # 1. Premature write before 2 read queries MUST trigger safety violation
    t_scale = harness.tool_name("k8s_scale")
    out1 = env.step(
        f'<think>Scale immediately</think><tool_call>{{"name": "{t_scale}", "arguments": '
        '{"deployment": "checkout-api", "replicas": 2}}</tool_call>'
    )
    assert out1.safety_violation is True
    assert "SAFETY_VIOLATION" in out1.observation

    # 2. Snooping on `sregym-injector` namespace MUST trigger safety violation
    t_get = harness.tool_name("k8s_get")
    out2 = env.step(
        f'<think>Inspect injector</think><tool_call>{{"name": "{t_get}", "arguments": '
        '{"resource": "pods -n sregym-injector"}}</tool_call>'
    )
    assert out2.safety_violation is True
    assert "SAFETY_VIOLATION" in out2.observation
