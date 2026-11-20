#!/usr/bin/env python3
"""Extract & Index Real Multi-Modal Production Telemetry (`RCAEval` RE1/RE2/RE3 + `DejaVu` Bank/Cloud).

Creates a strictly zero-contamination Train / Held-Out Unseen-System index of real
recorded incident bundles:
- **Train Systems:** `RE1-OB`, `RE1-SS`, `RE2-OB`, `RE2-SS`, `RE3-OB`, `RE3-SS`,
  `DejaVu-A1`, `DejaVu-A2`, `DejaVu-B`
- **100% Held-Out Unseen Systems (Zero-Shot Eval):**
  * `RE3-TT` (41-service Train Ticket K8s cluster)
  * `DejaVu-C` (Commercial Oracle DB production lock/execution outages)
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

from sre_rl.env.real_telemetry_env import RealIncidentRecord

HELDOUT_DATASETS: frozenset[str] = frozenset({"RE3-TT", "RE2-TT", "DejaVu-C", "DejaVu-D"})


def _index_rcaeval_dir(ds_name: str, root: Path, start_idx: int) -> list[RealIncidentRecord]:
    records: list[RealIncidentRecord] = []
    sys_map = {"OB": "online-boutique", "SS": "sock-shop", "TT": "train-ticket"}
    suffix = ds_name.split("-")[-1]
    system_name = sys_map.get(suffix, ds_name.lower())

    for inject_file in sorted(root.rglob("inject_time.txt")):
        case_dir = inject_file.parent
        # Parent folder is `<service>_<fault>` (e.g. `carts_cpu`, `adservice_f3`)
        fault_dir_name = case_dir.parent.name
        if "_" not in fault_dir_name:
            continue
        root_svc, fault_type = fault_dir_name.rsplit("_", 1)
        try:
            inject_ts = int(float(inject_file.read_text(encoding="utf-8").strip()))
        except ValueError:
            continue

        metrics_p = case_dir / "metrics.csv"
        simple_p = case_dir / "simple_metrics.csv"
        data_p = case_dir / "data.csv"
        if not metrics_p.exists() and not simple_p.exists() and not data_p.exists():
            continue

        primary_m = metrics_p if metrics_p.exists() else (simple_p if simple_p.exists() else data_p)
        logs_p = case_dir / "logs.csv"
        traces_p = case_dir / "traces.csv"
        pod_nodes = sorted(case_dir.glob("pod-node-*.csv"))

        idx = start_idx + len(records)
        records.append(
            RealIncidentRecord(
                case_id=f"real-{ds_name.lower()}-{idx:05d}",
                source_dataset=ds_name,
                system_name=system_name,
                inject_timestamp=inject_ts,
                ground_truth_service=root_svc,
                ground_truth_fault=fault_type,
                metrics_path=str(primary_m),
                simple_metrics_path=str(simple_p) if simple_p.exists() else None,
                logs_path=str(logs_p) if logs_p.exists() else None,
                traces_path=str(traces_p) if traces_p.exists() else None,
                pod_node_path=str(pod_nodes[0]) if pod_nodes else None,
                is_dejavu_long_format=False,
            )
        )
    return records


def _detect_dejavu_single_entity_format(metrics_path: Path) -> bool:
    """Return True if DejaVu metrics.csv uses empty `##<metric>` single-host wait-event names (e.g. DejaVu-C)."""
    norm_p = metrics_path.with_name("metrics.norm.csv")
    target = norm_p if norm_p.exists() else metrics_path
    try:
        with target.open("r", encoding="utf-8", errors="ignore") as fh:
            reader = csv.DictReader(fh)
            checked = 0
            non_empty_prefixes = 0
            for row in reader:
                name = str(row.get("name", "")).strip()
                if not name:
                    continue
                checked += 1
                prefix = name.split("##", 1)[0].strip() if "##" in name else ""
                if prefix:
                    non_empty_prefixes += 1
                if checked >= 200:
                    break
            return checked > 0 and non_empty_prefixes == 0
    except OSError:
        return False


def _clean_dejavu_service_name(row: dict[str, str], is_single_entity: bool) -> str:
    # DejaVu-A1/A2 have clean `name` column (`docker_003`, `os_021`, `db_007`)
    direct_name = str(row.get("name") or "").strip()
    if direct_name:
        return direct_name

    # DejaVu-C has `entity` column (`metric.platform.bjclpdb01`)
    entity = str(row.get("entity") or "").strip()
    if entity:
        if entity.startswith("metric.platform."):
            return entity[len("metric.platform.") :]
        return entity

    # DejaVu-B has `kpi` like `Tomcat03##OSLinux-CPU...` and `root_cause_node` like `Tomcat03 OS CPU`
    kpi = str(row.get("kpi") or "").strip()
    if "##" in kpi:
        first_kpi = kpi.split(";", 1)[0].strip()
        prefix = first_kpi.split("##", 1)[0].strip()
        if prefix:
            return prefix

    rcn = str(row.get("root_cause_node") or "").strip()
    if rcn and not is_single_entity:
        # e.g. "Tomcat03 OS CPU" -> "Tomcat03"
        return rcn.split()[0].split(";")[0].strip()
    return rcn


def _index_dejavu_dir(ds_name: str, root: Path, start_idx: int) -> list[RealIncidentRecord]:
    records: list[RealIncidentRecord] = []
    fault_files = list(root.rglob("faults.csv"))
    for fpath in fault_files:
        base_dir = fpath.parent
        metrics_p = base_dir / "metrics.csv"
        if not metrics_p.exists():
            continue
        graph_p = base_dir / "graph.yml"
        system_name = f"prod-{ds_name.lower()}"
        is_single_entity = _detect_dejavu_single_entity_format(metrics_p)

        with fpath.open("r", encoding="utf-8", errors="ignore") as fh:
            for row in csv.DictReader(fh):
                raw_ts = row.get("timestamp") or ""
                if not raw_ts:
                    continue
                try:
                    ts = int(float(raw_ts))
                except ValueError:
                    continue
                root_svc = _clean_dejavu_service_name(row, is_single_entity=is_single_entity)
                fault_type = str(
                    row.get("fault_description")
                    or row.get("node_type")
                    or row.get("metric")
                    or "unknown"
                ).strip()
                if not root_svc:
                    continue

                idx = start_idx + len(records)
                records.append(
                    RealIncidentRecord(
                        case_id=f"real-{ds_name.lower()}-{idx:05d}",
                        source_dataset=ds_name,
                        system_name=system_name,
                        inject_timestamp=ts,
                        ground_truth_service=root_svc,
                        ground_truth_fault=fault_type,
                        metrics_path=str(metrics_p),
                        graph_path=str(graph_p) if graph_p.exists() else None,
                        is_dejavu_long_format=True,
                        single_entity_alert_host=root_svc if is_single_entity else None,
                        candidate_services=[root_svc] if is_single_entity else [],
                    )
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract & index real Zenodo telemetry archives")
    parser.add_argument("--raw-dir", type=Path, default=Path("/mnt/nvme/real_telemetry/raw"))
    parser.add_argument("--extracted-dir", type=Path, default=Path("/mnt/nvme/real_telemetry/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("/mnt/nvme/real_telemetry/index"))
    args = parser.parse_args()

    args.extracted_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[RealIncidentRecord] = []
    for zpath in sorted(args.raw_dir.glob("*.zip")):
        ds_name = zpath.stem
        target_dir = args.extracted_dir / ds_name
        if not target_dir.exists() or not any(target_dir.iterdir()):
            print(f"[index] Extracting {zpath.name} -> {target_dir}...", flush=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(target_dir)
            except zipfile.BadZipFile:
                print(f"[index] Skipping incomplete zip {zpath.name}", flush=True)
                continue

        if ds_name.startswith("DejaVu-"):
            recs = _index_dejavu_dir(ds_name, target_dir, len(all_records))
        else:
            recs = _index_rcaeval_dir(ds_name, target_dir, len(all_records))
        print(f"[index] Indexed {len(recs):4d} real incidents from {ds_name}", flush=True)
        all_records.extend(recs)

    train_recs = [r for r in all_records if r.source_dataset not in HELDOUT_DATASETS]
    heldout_recs = [r for r in all_records if r.source_dataset in HELDOUT_DATASETS]

    for fname, subset in (
        ("real_telemetry_train.jsonl", train_recs),
        ("real_telemetry_heldout.jsonl", heldout_recs),
    ):
        with (args.out_dir / fname).open("w", encoding="utf-8") as fh:
            for r in subset:
                fh.write(r.model_dump_json() + "\n")

    manifest = {
        "total_real_incidents": len(all_records),
        "train_incidents": len(train_recs),
        "heldout_unseen_system_incidents": len(heldout_recs),
        "by_dataset": dict(Counter(r.source_dataset for r in all_records)),
        "by_system": dict(Counter(r.system_name for r in all_records)),
        "with_real_logs": sum(1 for r in all_records if r.logs_path),
        "with_real_traces": sum(1 for r in all_records if r.traces_path),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
