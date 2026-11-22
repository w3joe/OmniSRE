from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

from sre_rl.env.real_telemetry_env import RealIncidentRecord
from sre_rl.env.v2_raw_exploration_env import HaystackConfig, RawDuckDBExplorationEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_phase3_v2_sft_dataset import (
    synthesize_v2_counterfactual_5x_sft_row,
    synthesize_v2_raw_expert_trajectory,
)


def _make_telecom_scale_mismatch_incident(tmp_path: Path) -> RealIncidentRecord:
    """Create a realistic OpenRCA-style CSV mixing 10^9-byte healthy counters with a 0..1 CPU fault."""
    mpath = tmp_path / "metric_container.csv"
    lpath = tmp_path / "log_service.csv"

    m_lines = ["timestamp,cmdb_id,kpi_name,value"]
    for step in range(30):
        ts = 20_000 - 900 + step * 60
        # Healthy node with huge 8.5e9 byte counter (used to fool naive `ORDER BY max_val DESC`!)
        m_lines.append(f"{ts},os_021,mem_used_bytes,{8_500_000_000 + step * 1000}")
        m_lines.append(f"{ts},os_021,cpu_usage,0.12")
        # Faulty container `docker_003` jumps from 0.08 -> 0.99 CPU at ts >= 20_000
        cpu_val = 0.99 if ts >= 20_000 else 0.08 + (step % 3) * 0.005
        m_lines.append(f"{ts},docker_003,container_cpu_used,{cpu_val}")

    l_lines = [
        "timestamp,service,level,message",
        "19800,os_021,INFO,Health probe OK",
        "20010,docker_003,ERROR,CPU throttling exceeded quota on docker_003",
    ]
    mpath.write_text("\n".join(m_lines), encoding="utf-8")
    lpath.write_text("\n".join(l_lines), encoding="utf-8")

    return RealIncidentRecord(
        case_id="openrca-bank-00001",
        source_dataset="OpenRCA-Bank",
        system_name="openrca-bank",
        inject_timestamp=20_000,
        ground_truth_service="docker_003",
        ground_truth_fault="high CPU usage",
        metrics_path=str(mpath),
        logs_path=str(lpath),
        is_dejavu_long_format=False,
    )


def test_promql_range_query_ranks_zscore_above_giant_byte_counters(tmp_path: Path) -> None:
    rec = _make_telecom_scale_mismatch_incident(tmp_path)
    env = RawDuckDBExplorationEnv(
        HaystackConfig(max_turns=10, num_distractor_services=10, alias_services_prob=0.0), seed=7
    )
    try:
        env.reset(rec)
        out = env.step(
            '<think>Query bucketed z-scores.</think>\n'
            '<tool_call>{"name": "promql_range_query", "arguments": {"bucket_s": 60}}</tool_call>'
        )
        rows = json.loads(out.observation.split("\n[turn ", 1)[0])
        assert rows[0]["service"] == "docker_003"
        assert rows[0]["metric"] == "container_cpu_used"
        assert float(rows[0]["z_score"]) > 20.0
    finally:
        env.close()


def test_phase3_blind_duckdb_sft_synthesizer_both_aliasing_and_compaction(tmp_path: Path) -> None:
    rec = _make_telecom_scale_mismatch_incident(tmp_path)

    # 1. Unaliased + SQL CTE discovery + compaction split -> produces 2 verified segments!
    cfg_compact = HaystackConfig(max_turns=20, num_distractor_services=15, alias_services_prob=0.0)
    rows_c = synthesize_v2_raw_expert_trajectory(
        rec, cfg_compact, seed=11, use_promql_discovery=False, use_compaction=True
    )
    assert len(rows_c) == 2
    assert rows_c[0]["source"] == "phase3_v2_duckdb_compact_seg1"
    assert rows_c[1]["source"] == "phase3_v2_duckdb_sql"
    assert rows_c[1]["reward"] == 1.0
    assert "[COMPACTED INVESTIGATION WORKSPACE" in rows_c[1]["messages"][1]["content"]

    # 2. Aliased (`svc-prod-XXX`) + PromQL z-score discovery -> 100% blind localization & citation!
    cfg_alias = HaystackConfig(max_turns=20, num_distractor_services=15, alias_services_prob=1.0)
    rows_a = synthesize_v2_raw_expert_trajectory(
        rec, cfg_alias, seed=23, use_promql_discovery=True, use_compaction=False
    )
    assert len(rows_a) == 1
    assert rows_a[0]["reward"] == 1.0
    # Ensure ground-truth canonical name `docker_003` never leaked into an aliased episode!
    full_text = json.dumps(rows_a[0]["messages"])
    assert "docker_003" not in full_text
    assert "svc-prod-" in full_text


def test_phase3_counterfactual_5x_sft_and_contamination_guard(tmp_path: Path) -> None:
    rec = _make_telecom_scale_mismatch_incident(tmp_path)
    cf_row = synthesize_v2_counterfactual_5x_sft_row(rec, random.Random(42))
    assert cf_row is not None
    assert cf_row["reward"] == 1.0
    assert len(cf_row["messages"][-1]["content"]) < 900

    # Held-out contamination guard MUST raise ValueError
    heldout_rec = rec.model_copy(update={"source_dataset": "OpenRCA-Telecom"})
    with pytest.raises(ValueError, match="Contamination guard"):
        synthesize_v2_raw_expert_trajectory(heldout_rec, HaystackConfig(), seed=1)
    with pytest.raises(ValueError, match="Contamination guard"):
        synthesize_v2_counterfactual_5x_sft_row(heldout_rec, random.Random(1))
