"""Unit tests for RealTelemetryReplayEnvironment (RCAEval wide CSV & DejaVu long CSV)."""

from pathlib import Path

from sre_rl.env.real_telemetry_env import RealIncidentRecord, RealTelemetryReplayEnvironment


def test_real_telemetry_env_wide_prometheus_logs_traces(tmp_path: Path) -> None:
    t0 = 1_700_000_500
    metrics_csv = tmp_path / "simple_metrics.csv"
    lines = ["time,frontend_cpu,carts_cpu,payment_cpu"]
    for dt in range(-200, 120, 10):
        ts = t0 + dt
        c_cpu = 0.20 if dt < 0 else 4.95  # Strong real spike on carts_cpu after inject_time!
        lines.append(f"{ts},0.50,{c_cpu},0.15")
    metrics_csv.write_text("\n".join(lines), encoding="utf-8")

    logs_csv = tmp_path / "logs.csv"
    logs_csv.write_text(
        "time,timestamp,container_name,message,level,req_path,error\n"
        f"10:00,{t0 + 15},carts,ERROR CPU throttling timeout waiting for worker,ERROR,/carts,true\n"
        f"10:00,{t0 + 18},frontend,GET /catalogue 200 OK,INFO,/catalogue,\n",
        encoding="utf-8",
    )

    rec = RealIncidentRecord(
        case_id="real-test-0001",
        source_dataset="RE2-SS",
        system_name="sock-shop",
        inject_timestamp=t0,
        ground_truth_service="carts",
        ground_truth_fault="cpu",
        metrics_path=str(metrics_csv),
        simple_metrics_path=str(metrics_csv),
        logs_path=str(logs_csv),
    )

    env = RealTelemetryReplayEnvironment(max_turns=8)
    alert = env.reset(rec)
    assert "carts_cpu" not in alert  # Blind! No ground-truth filename leak!

    # Parallel read triage (anomalies + logs in one turn)
    out1 = env.step(
        "<think>Check top anomalies and error logs in parallel.</think>\n"
        '<tool_call>{"name": "query_metric_anomalies", "arguments": {"top_k": 5}}</tool_call>\n'
        '<tool_call>{"name": "search_logs", "arguments": {"errors_only": true}}</tool_call>'
    )
    assert "carts_cpu" in out1.observation
    assert "CPU throttling" in out1.observation

    out2 = env.step(
        "<think>Confirmed carts CPU saturation across metrics and logs. Submit RCA.</think>\n"
        '<tool_call>{"name": "submit_rca_report", "arguments": {"root_service": "carts", '
        '"fault_type": "cpu", "causal_chain": "carts CPU saturation caused timeouts", "confidence": 0.95}}</tool_call>'
    )
    assert out2.done is True
    score = env.score()
    assert score.solved is True
    assert score.total == 1.0


def test_real_telemetry_env_dejavu_long_format(tmp_path: Path) -> None:
    t0 = 10_800
    metrics_csv = tmp_path / "metrics.csv"
    rows = ["name,timestamp,value"]
    for dt in range(-240, 180, 60):
        val = 12.0 if dt < 0 else 480.0
        rows.append(f"bjclpdb02##RowLockWait,{t0 + dt},{val}")
        rows.append(f"bjclpdb01##RowLockWait,{t0 + dt},10.0")
    metrics_csv.write_text("\n".join(rows), encoding="utf-8")

    rec = RealIncidentRecord(
        case_id="real-dejavu-c-0001",
        source_dataset="DejaVu-C",
        system_name="prod-dejavu-c",
        inject_timestamp=t0,
        ground_truth_service="bjclpdb02",
        ground_truth_fault="Row Lock Contention",
        metrics_path=str(metrics_csv),
        is_dejavu_long_format=True,
    )
    env = RealTelemetryReplayEnvironment()
    env.reset(rec)
    out = env.step(
        "<think>Inspect anomalies & list services</think>\n"
        '<tool_call>{"name": "list_services", "arguments": {}}</tool_call>\n'
        '<tool_call>{"name": "query_metric_anomalies", "arguments": {"top_k": 5}}</tool_call>'
    )
    assert "bjclpdb02" in out.observation
    env.step(
        "<think>Submit lock contention on bjclpdb02</think>\n"
        '<tool_call>{"name": "submit_rca_report", "arguments": {"root_service": "bjclpdb02", '
        '"fault_type": "lock_contention", "causal_chain": "row lock contention spike on bjclpdb02"}}</tool_call>'
    )
    score = env.score()
    assert score.solved is True
    assert score.total >= 0.90
