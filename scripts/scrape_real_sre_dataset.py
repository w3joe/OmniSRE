#!/usr/bin/env python3
"""Scrape, scrub, deduplicate & assemble the Real-World SRE SFT + RLVR Corpus (Dataset v2).

Combines:
- `rmems/incident-response-oncall-trajectories` (Apache-2.0 multi-turn OpenSRE traces)
- `miyad316/sre-agentic-trajectories-devops` (multi-turn K8s/Linux/DB SRE traces)
- `beatsprom/devops-kubernetes-iac-sft-dpo-2026` (CC-BY-4.0 K8s RCA + DPO pairs)
- `Snaseem2026/devops-incident-response` (real incident RCA & resolution traces)
- `nolabs/deepfabric-devops-reasoning-traces` (multi-step DevOps reasoning traces)
- `openshift/runbooks` + `prometheus-operator/runbooks` (Apache-2.0 K8s runbooks)
- `samber/awesome-prometheus-alerts` (1,217 real PromQL alert rules)
- `SREGym/SREGym` (MIT real-world Kubernetes/OS/eBPF fault specs)
- Plus adversarial log-injection & benign false-alarm balancing from `scenario_factory`
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sre_rl.data.scenario_factory import generate_scenario_suite
from sre_rl.data.scraper import RealSREDatasetScraper
from sre_rl.data.trajectory_builder import (
    export_jsonl,
    export_parquet_or_jsonl,
    purify_trajectory,
    synthesize_expert_trajectory,
    to_verl_rl_row,
)
from sre_rl.schemas import IncidentClassification


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape real open-source SRE corpora for Qwen3.8-27B RLVR")
    parser.add_argument("--output-dir", type=Path, default=Path("data/real_sre_v2"))
    parser.add_argument("--rmems-batches", type=int, default=350)
    parser.add_argument("--rmems-max", type=int, default=3500)
    parser.add_argument("--miyad-max", type=int, default=2000)
    parser.add_argument("--beatsprom-max", type=int, default=1500)
    parser.add_argument("--deepfabric-max", type=int, default=1000)
    parser.add_argument("--snaseem-max", type=int, default=1000)
    parser.add_argument("--grounded-rl-scenarios", type=int, default=900)
    parser.add_argument("--synthetic-safety-balance", type=int, default=400)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    scraper = RealSREDatasetScraper(max_workers=args.workers)

    print("[1/7] Scraping rmems/incident-response-oncall-trajectories (Apache-2.0)...", flush=True)
    rmems_rows = scraper.scrape_rmems_oncall(
        max_batches=args.rmems_batches, max_records=args.rmems_max
    )
    print(f"      -> collected {len(rmems_rows)} OpenSRE multi-turn trajectories", flush=True)

    print("[2/7] Scraping miyad316/sre-agentic-trajectories-devops...", flush=True)
    miyad_rows = scraper.scrape_miyad316_sre_trajectories(max_records=args.miyad_max)
    print(f"      -> collected {len(miyad_rows)} agentic SRE trajectories", flush=True)

    print("[3/7] Scraping beatsprom/devops-kubernetes-iac-sft-dpo-2026 (CC-BY-4.0)...", flush=True)
    beatsprom_rows = scraper.scrape_beatsprom_k8s_sre(max_records=args.beatsprom_max)
    print(f"      -> collected {len(beatsprom_rows)} K8s RCA + DPO traces", flush=True)

    print("[4/8] Scraping nolabs/deepfabric + Snaseem2026 + ibm-research/ITBench-Trajectories...", flush=True)
    deepfabric_rows = scraper.scrape_deepfabric_devops(max_records=args.deepfabric_max)
    snaseem_rows = scraper.scrape_snaseem_incidents(max_records=args.snaseem_max)
    itbench_rows = scraper.scrape_itbench_trajectories(max_sessions=105, min_judge_score=0.25)
    print(
        f"      -> collected {len(deepfabric_rows)} deepfabric + {len(snaseem_rows)} snaseem "
        f"+ {len(itbench_rows)} ITBench SRE traces",
        flush=True,
    )

    print("[5/8] Scraping production runbooks, PromQL rules, SREGym & postmortems...", flush=True)
    prom_rules = scraper.scrape_awesome_prometheus_alerts()
    ocp_runbooks = scraper.scrape_github_runbook_tarball(
        "openshift/runbooks", "master", "alerts/", license_id="Apache-2.0"
    )
    kprom_runbooks = scraper.scrape_github_runbook_tarball(
        "prometheus-operator/runbooks", "main", "content/runbooks/", license_id="Apache-2.0"
    )
    sregym_specs = scraper.scrape_sregym_fault_specs()
    postmortems = scraper.scrape_danluu_postmortems()
    all_runbooks = ocp_runbooks + kprom_runbooks
    print(
        f"      -> collected {len(prom_rules)} PromQL rules, {len(all_runbooks)} production runbooks, "
        f"{len(sregym_specs)} SREGym fault specs, {len(postmortems)} postmortems",
        flush=True,
    )

    print("[6/8] Building de-contaminated grounded RLVR splits + train-only SFT balance...", flush=True)
    grounded_scenarios = scraper.build_grounded_rlvr_scenarios_from_kb(
        runbooks=all_runbooks,
        promql_rules=prom_rules,
        max_scenarios=args.grounded_rl_scenarios,
    )
    synth_suite = generate_scenario_suite(num_scenarios=args.synthetic_safety_balance, seed=101)

    # Split ALL RL scenarios deterministically by topology_name FIRST so train SFT never leaks held-out!
    all_rl_scenarios = grounded_scenarios + synth_suite
    rl_train, rl_val, rl_heldout = [], [], []
    for sc in all_rl_scenarios:
        bucket = int(hashlib.sha256(sc.topology_name.encode("utf-8")).hexdigest()[:8], 16) % 100
        if bucket < 15 or sc.topology_name == "checkout-aurora-stack":
            rl_heldout.append(sc)
        elif bucket < 25:
            rl_val.append(sc)
        else:
            rl_train.append(sc)

    safety_sft_rows = []
    grounded_sft_rows = []
    for sc in rl_train:
        if sc.scenario_id.startswith("real-"):
            if len(grounded_sft_rows) < 400:
                traj = synthesize_expert_trajectory(sc)
                traj["source"] = "grounded_openshift_promql_runbook_trace"
                traj["license"] = "Apache-2.0"
                traj["messages"] = purify_trajectory(traj["messages"])
                grounded_sft_rows.append(traj)
        elif (
            sc.fault.classification != IncidentClassification.REAL_INCIDENT
            or sc.fault.contains_adversarial_log_injection
            or sc.fault.requires_secondary_drain_or_scale
        ):
            traj = synthesize_expert_trajectory(sc)
            traj["source"] = "qwen38-sre-rl/safety_and_metastable_balance"
            traj["license"] = "Apache-2.0"
            traj["messages"] = purify_trajectory(traj["messages"])
            safety_sft_rows.append(traj)

    print("[7/8] Writing scrubbed & deduplicated Dataset v2 artifacts...", flush=True)
    sft_corpus = (
        rmems_rows
        + miyad_rows
        + beatsprom_rows
        + deepfabric_rows
        + snaseem_rows
        + itbench_rows
        + grounded_sft_rows
        + safety_sft_rows
    )
    scraper.rng.shuffle(sft_corpus)

    sft_path = export_jsonl(sft_corpus, out / "sft_real_sre_corpus.jsonl")
    kb_path = export_jsonl(
        prom_rules + all_runbooks + sregym_specs + postmortems,
        out / "production_runbooks_promql_kb.jsonl",
    )
    rl_train_path = export_parquet_or_jsonl([to_verl_rl_row(s) for s in rl_train], out / "rl_real_train.parquet")
    rl_val_path = export_parquet_or_jsonl([to_verl_rl_row(s) for s in rl_val], out / "rl_real_val.parquet")
    rl_heldout_path = export_parquet_or_jsonl([to_verl_rl_row(s) for s in rl_heldout], out / "rl_real_heldout.parquet")
    eval_json_path = out / "eval_real_heldout_scenarios.json"
    eval_json_path.write_text(
        json.dumps([s.model_dump() for s in rl_heldout], indent=2),
        encoding="utf-8",
    )

    report = {
        "total_sft_trajectories": len(sft_corpus),
        "total_kb_entries": len(prom_rules) + len(all_runbooks) + len(sregym_specs) + len(postmortems),
        "total_rl_scenarios": len(all_rl_scenarios),
        "rl_train_scenarios": len(rl_train),
        "rl_val_scenarios": len(rl_val),
        "rl_heldout_scenarios": len(rl_heldout),
        "source_counts": scraper.stats.source_counts,
        "safety_balance_traces": len(safety_sft_rows),
        "grounded_runbook_traces": len(grounded_sft_rows),
        "dedup_skipped": scraper.stats.dedup_skipped,
        "quality_skipped": scraper.stats.quality_skipped,
        "pii_secret_redactions": scraper.scrubber.redaction_counts,
        "errors": scraper.stats.errors,
        "artifacts": {
            "sft_corpus": str(sft_path),
            "kb": str(kb_path),
            "rl_train": str(rl_train_path),
            "rl_val": str(rl_val_path),
            "rl_heldout": str(rl_heldout_path),
            "eval_scenarios": str(eval_json_path),
        },
    }
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    print("[8/8] Running 5-Gate Dataset Validator (PII, Tool JSON, Oracle, De-Contamination, Mix)...", flush=True)
    from dataclasses import asdict

    from sre_rl.data.validator import validate_dataset_directory

    val_report = validate_dataset_directory(
        dataset_dir=out,
        oracle_sample_size=150,
        auto_filter_invalid_sft=True,
    )
    print(json.dumps(asdict(val_report), indent=2), flush=True)


if __name__ == "__main__":
    main()
