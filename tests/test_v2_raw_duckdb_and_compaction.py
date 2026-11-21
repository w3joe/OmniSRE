from __future__ import annotations

import json
import time
from pathlib import Path

from sre_rl.env.real_telemetry_env import RealIncidentRecord
from sre_rl.env.v2_raw_exploration_env import HaystackConfig, RawDuckDBExplorationEnv
from sre_rl.v2_tree_and_compaction import (
    CompactedSegmentRecord,
    LeafInvestigationArtifact,
    build_segmented_rollout_items,
    build_v2_role_diverse_synthesis_prompt,
    is_disagreeing_counterfactual_leaf_set,
    score_v2_counterfactual_synthesis_turn,
)


def _write_sample_incident(tmp_path: Path) -> RealIncidentRecord:
    metrics_csv = tmp_path / "metrics.csv"
    logs_csv = tmp_path / "logs.csv"
    traces_csv = tmp_path / "traces.csv"

    # Fault onset at t = 10_000 on `paymentservice` (code_fault `f3` with sustained latency + 500s)
    m_lines = ["timestamp,paymentservice_cpu,paymentservice_latency,frontend_latency"]
    for step in range(-12, 12):
        ts = 10_000 + step * 60
        if ts < 10_000:
            m_lines.append(f"{ts},14.2,28.5,35.0")
        else:
            m_lines.append(f"{ts},89.4,940.0,985.0")
    metrics_csv.write_text("\n".join(m_lines), encoding="utf-8")

    l_lines = [
        "timestamp,container_name,level,message",
        "9880,frontend,INFO,GET /checkout 200 OK",
        "10005,paymentservice,ERROR,NullPointerException in ChargeProcessor.execute(ChargeProcessor.java:142)",
        "10012,frontend,ERROR,Upstream RPC timeout calling paymentservice:50051",
    ]
    logs_csv.write_text("\n".join(l_lines), encoding="utf-8")

    t_lines = [
        "startTime,traceID,spanID,parentSpanID,serviceName,operationName,duration,statusCode",
        "10010,tr1,sp_root,,frontend,POST /checkout,960.0,500",
        "10010,tr1,sp_child,sp_root,paymentservice,ChargeProcessor/Charge,945.0,500",
    ]
    traces_csv.write_text("\n".join(t_lines), encoding="utf-8")

    return RealIncidentRecord(
        case_id="v2-test-0001",
        source_dataset="RE3-OB",
        system_name="online-boutique",
        inject_timestamp=10_000,
        ground_truth_service="paymentservice",
        ground_truth_fault="f3",
        metrics_path=str(metrics_csv),
        logs_path=str(logs_csv),
        traces_path=str(traces_csv),
    )


def test_v2_duckdb_haystack_isolation_sql_speed_and_strict_grading(tmp_path: Path) -> None:
    rec = _write_sample_incident(tmp_path)
    secret_file = tmp_path / "record.csv"
    secret_file.write_text("TOP_SECRET_GROUND_TRUTH", encoding="utf-8")

    cfg = HaystackConfig(
        max_turns=30,
        num_distractor_services=35,
        alias_services_prob=1.0,  # Force service aliasing (`svc-prod-XXX`)
    )
    env = RawDuckDBExplorationEnv(config=cfg, seed=7)
    try:
        alert = env.reset(rec)
        # Exact onset timestamp MUST NOT appear in the alert!
        assert "10000" not in alert
        assert "~37 active services" in alert

        # 1. Anti-cheat check: SQL attempting to read local files is blocked by regex + DuckDB lockdown
        cheat = env.step(
            '<think>Try reading disk</think>'
            f'<tool_call>{{"name": "sql_telemetry_query", "arguments": {{"sql": "SELECT * FROM read_csv(\'{secret_file}\')"}}}}</tool_call>'
        )
        assert "SQL_REJECTED" in cheat.observation

        # 2. Sub-5ms analytical SQL + trace DAG self-latency attribution + log grep
        t0 = time.perf_counter()
        step_out = env.step(
            "<think>Run analytical SQL, trace call graph & grep logs in parallel</think>\n"
            '<tool_call>{"name": "sql_telemetry_query", "arguments": {"sql": "SELECT service, metric, MAX(value) AS mx FROM metrics GROUP BY 1, 2 ORDER BY mx DESC LIMIT 5"}}</tool_call>\n'
            '<tool_call>{"name": "trace_call_graph", "arguments": {}}</tool_call>\n'
            '<tool_call>{"name": "grep_logs", "arguments": {"regex": "NullPointerException"}}</tool_call>'
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 150.0
        assert "columns" in step_out.observation and "NullPointerException in ChargeProcessor.execute" in step_out.observation
        # Trace self-latency isolates aliased paymentservice (~945ms self_p95_ms) over frontend (~15ms self_p95_ms)!
        aliased_payment = env._canon_to_alias["paymentservice"]
        assert aliased_payment in step_out.observation

        # 3. Test rolling `compact_workspace`
        comp_out = env.step(
            "<think>Compact findings before submitting</think>\n"
            '<tool_call>{"name": "compact_workspace", "arguments": {'
            f'"confirmed_anomalies": ["{aliased_payment} NullPointerException at ts=10005, self_p95=945ms"], '
            '"ruled_out_services": ["svc-bg-100"]}}</tool_call>'
        )
        assert comp_out.compacted is True
        assert aliased_payment in env.render_initial_alert()

        # 4. Verify NO `code_fault` free-pass: submitting wrong `fault_type="cpu"` MUST fail `strict_solved`!
        env_wrong = RawDuckDBExplorationEnv(config=cfg, seed=7)
        env_wrong.reset(rec)
        env_wrong.step(
            '<tool_call>{"name": "list_tables_and_schemas", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "grep_logs", "arguments": {"regex": "NullPointer"}}</tool_call>'
        )
        env_wrong.step(
            f'<tool_call>{{"name": "submit_rca_report", "arguments": {{"root_service": "{aliased_payment}", '
            '"fault_type": "cpu", "onset_timestamp": 10005, "cited_evidence": "FABRICATED_STRING_999"}}</tool_call>'
        )
        sc_wrong = env_wrong.score()
        assert sc_wrong.strict_solved is False
        assert sc_wrong.fault_exact is False
        assert sc_wrong.citation_score < 0.0
        env_wrong.close()

        # 5. Submitting exact aliased service + `code_fault` + onset within 90s + verbatim citation gets 1.0!
        env.step(
            '<think>Submit verified RCA</think>'
            f'<tool_call>{{"name": "submit_rca_report", "arguments": {{'
            f'"root_service": "{aliased_payment}", '
            '"fault_type": "code_fault", '
            '"onset_timestamp": 10005, '
            '"cited_evidence": "NullPointerException in ChargeProcessor.execute", '
            '"causal_chain": "NPE in ChargeProcessor caused 945ms self-latency and upstream 500s"'
            "}}</tool_call>"
        )
        sc = env.score()
        assert sc.strict_solved is True
        assert sc.service_exact is True
        assert sc.fault_exact is True
        assert sc.onset_within_90s is True
        assert sc.citation_verified is True
        assert sc.total >= 0.95
    finally:
        env.close()


def test_segmented_100_turn_gtpo_and_disagreeing_5x_synthesis(tmp_path: Path) -> None:
    rec = _write_sample_incident(tmp_path)
    # Episode 0: 2 compacted segments, solved (reward=1.0)
    ep0 = [
        CompactedSegmentRecord(0, 2, [10, 11, 12, 13, 14, 15], [(2, 5)], [0.02], 1.0),
        CompactedSegmentRecord(1, 2, [10, 21, 22, 23, 24, 25], [(2, 5)], [0.03], 1.0),
    ]
    # Episode 1: 1 segment, failed (reward=-0.30)
    ep1 = [
        CompactedSegmentRecord(0, 1, [10, 31, 32, 33, 34, 35], [(2, 5)], [-0.04], -0.30),
    ]
    samples = build_segmented_rollout_items([ep0, ep1])
    assert len(samples) == 3  # Both segments of ep0 + 1 segment of ep1 receive valid advantages
    assert samples[0][1][3] > 0.0  # Early pre-compaction segment of winning episode gets positive credit!
    assert samples[2][1][3] < 0.0  # Losing episode gets negative advantage!

    leaves = [
        LeafInvestigationArtifact("metrics_inflection_specialist", False, 0.2, "{}", {"root_service": "frontend"}, ["obs"]),
        LeafInvestigationArtifact("log_forensics_specialist", True, 1.0, "{}", {"root_service": "paymentservice"}, ["NullPointerException"]),
        LeafInvestigationArtifact("trace_dag_specialist", True, 0.95, "{}", {"root_service": "paymentservice"}, ["self_p95=945ms"]),
    ]
    assert is_disagreeing_counterfactual_leaf_set(leaves) is True
    msgs = build_v2_role_diverse_synthesis_prompt(rec, 9400, 10600, leaves)
    assert len(msgs) == 2
    assert "log_forensics_specialist" in msgs[1]["content"]
    assert "trace_dag_specialist" in msgs[1]["content"]
    assert json.loads("{}") == {}


def test_v2_counterfactual_synthesis_scoring_with_aliasing_and_citation_verification(
    tmp_path: Path,
) -> None:
    rec = _write_sample_incident(tmp_path)
    alias_map = {"svc-prod-412": "paymentservice", "svc-prod-108": "frontend"}
    leaves = [
        LeafInvestigationArtifact(
            role_name="metrics_inflection_specialist",
            strict_solved=False,
            outcome_reward=0.10,
            compacted_notes_json='{"confirmed_anomalies": ["svc-prod-108 latency spike at 10012"]}',
            final_submission={"root_service": "svc-prod-108", "fault_type": "delay", "onset_timestamp": 10012},
            key_observations=["[10012] [svc-prod-108] ERROR: Upstream RPC timeout calling svc-prod-412:50051"],
            window_start=9400,
            window_end=10600,
            alias_to_canon=alias_map,
        ),
        LeafInvestigationArtifact(
            role_name="log_forensics_specialist",
            strict_solved=True,
            outcome_reward=1.0,
            compacted_notes_json='{"confirmed_anomalies": ["svc-prod-412 NullPointerException at 10005"]}',
            final_submission={"root_service": "svc-prod-412", "fault_type": "code_fault", "onset_timestamp": 10005},
            key_observations=[
                "[10005] [svc-prod-412] ERROR: NullPointerException in ChargeProcessor.execute(ChargeProcessor.java:142)"
            ],
            window_start=9400,
            window_end=10600,
            alias_to_canon=alias_map,
        ),
    ]

    # 1. Valid counterfactual synthesis isolating aliased leaf root cause + verbatim citation -> 1.0
    good_turn = (
        "<think>svc-prod-108 timeout at 10012 is downstream of svc-prod-412 NPE at 10005.</think>\n"
        '<tool_call>{"name": "submit_rca_report", "arguments": {'
        '"root_service": "svc-prod-412", '
        '"fault_type": "code_fault", '
        '"onset_timestamp": 10005, '
        '"cited_evidence": "NullPointerException in ChargeProcessor.execute", '
        '"causal_chain": "svc-prod-412 threw NPE at 10005 causing svc-prod-108 RPC timeouts"'
        "}}</tool_call>"
    )
    sc_good = score_v2_counterfactual_synthesis_turn(good_turn, rec, leaves)
    assert sc_good.strict_solved is True
    assert sc_good.service_exact is True
    assert sc_good.fault_exact is True
    assert sc_good.onset_within_90s is True
    assert sc_good.citation_verified is True
    assert sc_good.total == 1.0

    # 2. Wrong fault mechanism (`cpu` instead of `code_fault`) + fabricated citation -> rejected
    bad_turn = (
        '<tool_call>{"name": "submit_rca_report", "arguments": {'
        '"root_service": "svc-prod-412", '
        '"fault_type": "cpu", '
        '"onset_timestamp": 10005, '
        '"cited_evidence": "NON_EXISTENT_HALLUCINATION_STRING"'
        "}}</tool_call>"
    )
    sc_bad = score_v2_counterfactual_synthesis_turn(bad_turn, rec, leaves)
    assert sc_bad.strict_solved is False
    assert sc_bad.fault_exact is False
    assert sc_bad.citation_verified is False
    assert sc_bad.citation_score < 0.0

    # 3. Malformed / missing tool call -> -0.30 penalty
    sc_missing = score_v2_counterfactual_synthesis_turn("<think>no tool call</think>", rec, leaves)
    assert sc_missing.strict_solved is False
    assert sc_missing.total == -0.30


def test_sregym_nested_cumulative_csv_parser(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from summarize_external_benchmarks import parse_sregym_run_dir

    run_dir = tmp_path / "1121_1738"
    p1_dir = run_dir / "stratus" / "network_policy_block"
    p2_dir = run_dir / "stratus" / "namespace_memory_limit"
    p1_dir.mkdir(parents=True)
    p2_dir.mkdir(parents=True)

    (p1_dir / "network_policy_block_stratus_results.csv").write_text(
        "problem_id,run_status,Diagnosis.success,Diagnosis.composite_score,Mitigation.success,timed_out\n"
        "network_policy_block,complete,True,1.0,True,\n",
        encoding="utf-8",
    )
    (p2_dir / "namespace_memory_limit_stratus_results.csv").write_text(
        "problem_id,run_status,Diagnosis.success,Diagnosis.composite_score,Mitigation.success,timed_out\n"
        "network_policy_block,complete,True,1.0,True,\n"
        "namespace_memory_limit,incomplete,True,0.89,,True\n",
        encoding="utf-8",
    )

    summary = parse_sregym_run_dir(run_dir)
    assert summary["problems"] == 2
    assert summary["diagnosis_pass_count"] == 2
    assert summary["diagnosis_pass_rate"] == 1.0
    assert summary["mitigation_pass_count"] == 1
    assert summary["mitigation_pass_rate"] == 0.5
    assert summary["diagnosis_composite_mean"] == 0.945
