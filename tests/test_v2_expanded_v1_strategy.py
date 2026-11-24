from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from sre_rl.data.scenario_factory import FAULT_TEMPLATES, TOPOLOGIES, generate_scenario
from sre_rl.env.live_k8s_env import (
    DEFAULT_LIVE_K8S_FAULTS,
    KindClusterLease,
    LiveK8sKindEnvironment,
)
from sre_rl.env.real_telemetry_env import (
    RealIncidentRecord,
    RealTelemetryReplayEnvironment,
    _canonicalize_fault_set,
    clear_real_telemetry_caches,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_phase1b_sft_dataset import synthesize_runbook_grounded_trajectory
from build_v2_expanded_dataset import (
    HELDOUT_TOPOLOGIES,
    build_v2_expanded_datasets,
    synthesize_structured_real_telemetry_sft_row,
)


def test_all_22_expanded_k8s_fault_templates_solve_with_zero_safety_violations() -> None:
    assert len(TOPOLOGIES) == 8
    assert len(FAULT_TEMPLATES) == 22

    rng = random.Random(123)
    for idx, tmpl in enumerate(FAULT_TEMPLATES):
        sc = generate_scenario(idx, rng, topology_name="hotel-booking-grid", template=tmpl)
        row = synthesize_runbook_grounded_trajectory(sc, include_runbook_lookup=True)
        assert row["mitigated"] is True, f"Template {tmpl['fault_id']} failed mitigation"
        assert float(row["reward"]) >= 0.85, f"Template {tmpl['fault_id']} low reward {row['reward']}"


def test_openrca_statistical_lens_in_v1_real_telemetry_env(tmp_path: Path) -> None:
    clear_real_telemetry_caches()
    mpath = tmp_path / "metric_container.csv"
    lines = ["timestamp,cmdb_id,kpi_name,value"]
    for step in range(24):
        ts = 50_000 - 360 + step * 30
        # Huge 8.5e9 byte counter on healthy os_021
        lines.append(f"{ts},os_021,mem_used_bytes,{8_500_000_000 + step * 500}")
        # True CPU spike on docker_003 at ts >= 50_000
        cpu = 0.98 if ts >= 50_000 else 0.07 + (step % 2) * 0.005
        lines.append(f"{ts},docker_003,container_cpu_used,{cpu}")
    mpath.write_text("\n".join(lines), encoding="utf-8")
    # Non-value sibling CSV (`metric_service.csv`) must be safely ignored
    (tmp_path / "metric_service.csv").write_text("timestamp,service,mrt,count\n50000,api,12.0,100\n", encoding="utf-8")

    rec = RealIncidentRecord(
        case_id="openrca-bank-test",
        source_dataset="OpenRCA-Bank",
        system_name="openrca-bank",
        inject_timestamp=50_000,
        ground_truth_service="docker_003",
        ground_truth_fault="high CPU usage",
        metrics_path=str(mpath),
        is_dejavu_long_format=False,
    )

    env = RealTelemetryReplayEnvironment()
    env.reset(rec)
    out = env.step('<think>Check top z-scores.</think>\n<tool_call>{"name": "query_metric_anomalies", "arguments": {"top_k": 5}}</tool_call>')
    top = json.loads(out.observation)["top_anomalies"]
    assert top[0]["service"] == "docker_003"
    assert top[0]["z_score"] > 20.0

    sft_row = synthesize_structured_real_telemetry_sft_row(rec, random.Random(7))
    assert sft_row is not None
    assert sft_row["reward"] >= 0.85


def test_expanded_dataset_builder_enforces_zero_heldout_topology_leakage(tmp_path: Path) -> None:
    manifest = build_v2_expanded_datasets(
        out_dir=tmp_path,
        existing_rl_train_jsonl=None,
        real_telemetry_train_jsonl=None,
        openrca_train_jsonl=None,
        num_new_k8s_scenarios=120,
        target_k8s_sft_rows=60,
        target_telemetry_sft_rows=0,
        seed=9,
    )
    assert manifest["sft_total_rows"] == 60
    assert manifest["unique_fault_templates"] == 22

    sft_blob = (tmp_path / "sft_v2_expanded_calibration.jsonl").read_text(encoding="utf-8")
    rl_tr_blob = (tmp_path / "rl_v2_train.jsonl").read_text(encoding="utf-8")
    for ho_topo in HELDOUT_TOPOLOGIES:
        assert ho_topo not in sft_blob
        assert ho_topo not in rl_tr_blob


def test_live_k8s_expanded_faults_and_delete_guardrails(tmp_path: Path) -> None:
    assert len(DEFAULT_LIVE_K8S_FAULTS) == 13
    assert "disk" in _canonicalize_fault_set("container read I/O load")
    assert _canonicalize_fault_set("Execution") == {"lock_contention"}

    lease = KindClusterLease("kind-test", tmp_path / "adm", tmp_path / "agt")
    env = LiveK8sKindEnvironment(lease)
    env._spec = DEFAULT_LIVE_K8S_FAULTS[-1]
    env._read_queries = 2
    # Deleting namespace or secret must be rejected as SAFETY_VIOLATION
    out_bad = env.step('<think>bad</think>\n<tool_call>{"name": "k8s_delete", "arguments": {"resource": "secret", "name": "db-creds"}}</tool_call>')
    assert "ERROR" in out_bad.observation or "SAFETY_VIOLATION" in out_bad.observation

    # Verify rollout_live_k8s_v1 AST signature matches exact positional call in train_rl_v2_omnisre_4xh100.py
    import ast

    v1_ast = ast.parse((Path(__file__).resolve().parents[1] / "scripts" / "train_rl_v1_omnisre_4xh100.py").read_text())
    fn_defs = {n.name: [a.arg for a in n.args.args] for n in v1_ast.body if isinstance(n, ast.FunctionDef)}
    assert fn_defs["rollout_live_k8s_v1"][:5] == ["sglang_base_url", "tokenizer", "lease_pool", "fault_idx", "rng"]
