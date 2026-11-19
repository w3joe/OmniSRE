"""Unit tests for the real-world SRE scraper normalizers & KB-to-RLVR grounding."""

from sre_rl.data.scraper import RealSREDatasetScraper, _normalize_tool_name
from sre_rl.data.trajectory_builder import synthesize_expert_trajectory


def test_tool_name_normalization() -> None:
    assert _normalize_tool_name("promql") == "query_metrics"
    assert _normalize_tool_name("k8s_get_pods") == "k8s_inspect"
    assert _normalize_tool_name("helm_rollback") == "rollback_release"
    assert _normalize_tool_name("kubectl_patch") == "apply_remediation"


def test_grounded_rlvr_scenario_from_real_runbook_and_promql_is_executable() -> None:
    scraper = RealSREDatasetScraper(seed=1)
    runbooks = [
        {
            "kb_type": "production_runbook",
            "source": "openshift/runbooks",
            "license": "Apache-2.0",
            "component": "etcd-operator",
            "alert_name": "EtcdHighFsyncLatency",
            "path": "alerts/etcd-operator/EtcdHighFsyncLatency.md",
            "body": "Etcd wal fsync latency exceeds 500ms; inspect disk storage retention and compaction.",
        }
    ]
    prom_rules = [
        {
            "kb_type": "promql_alert_rule",
            "source": "samber/awesome-prometheus-alerts",
            "group": "Databases",
            "service": "PostgreSQL",
            "exporter": "postgres_exporter",
            "alert_name": "PostgresqlTooManyConnections",
            "description": "PostgreSQL instance has too many active connections (>80% max_connections)",
            "promql": "sum by (instance) (pg_stat_activity_count) > 0.8 * pg_settings_max_connections",
            "severity": "critical",
            "for_duration": "2m",
        }
    ]

    scenarios = scraper.build_grounded_rlvr_scenarios_from_kb(
        runbooks=runbooks, promql_rules=prom_rules, max_scenarios=10
    )
    assert len(scenarios) == 2

    for sc in scenarios:
        traj = synthesize_expert_trajectory(sc)
        assert traj["mitigated"] is True
        assert traj["reward"] >= 0.90
