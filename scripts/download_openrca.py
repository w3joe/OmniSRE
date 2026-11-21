#!/usr/bin/env python3
"""Download & Index `microsoft/OpenRCA` (335 Real Enterprise Bank / Telecom / Market Failures, 68.5 GB).

Strictly holds out `Telecom` and `Market/cloudbed-2` for zero-contamination external benchmarking,
while indexing `Bank` and `Market/cloudbed-1` for `OmniSRE v1.0` real-telemetry RLVR training.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

from sre_rl.env.real_telemetry_env import RealIncidentRecord

OPENRCA_GDRIVE_FOLDER_ID = "1wGiEnu4OkWrjPxfx5ZTROnU37-5UDoPM"
OPENRCA_HELDOUT_SYSTEMS: frozenset[str] = frozenset({"Telecom", "Market/cloudbed-2"})


def _parse_iso_or_unix(raw: str) -> int:
    s = raw.strip()
    try:
        return int(float(s))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return int(datetime.strptime(s[:19], fmt).timestamp())
        except ValueError:
            continue
    return 0


def index_openrca_directory(openrca_root: Path, out_dir: Path) -> dict[str, object]:
    """Index extracted OpenRCA `dataset/<System>/record.csv` + `telemetry/` directories."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_records: list[RealIncidentRecord] = []
    heldout_records: list[RealIncidentRecord] = []

    systems = ["Bank", "Market/cloudbed-1", "Market/cloudbed-2", "Telecom"]
    counter = 0

    for sys_rel in systems:
        sys_dir = openrca_root / "dataset" / sys_rel
        record_csv = sys_dir / "record.csv"
        if not record_csv.exists():
            record_csv = sys_dir / "query.csv"
        if not record_csv.exists():
            continue

        telemetry_dir = sys_dir / "telemetry"
        with record_csv.open("r", encoding="utf-8", errors="ignore") as fh:
            for row in csv.DictReader(fh):
                raw_ts = (
                    row.get("timestamp")
                    or row.get("datetime")
                    or row.get("start_time")
                    or ""
                )
                ts = _parse_iso_or_unix(raw_ts)
                root_svc = str(
                    row.get("component")
                    or row.get("root_cause_component")
                    or row.get("service")
                    or ""
                ).strip()
                fault_type = str(
                    row.get("reason")
                    or row.get("root_cause_reason")
                    or row.get("fault")
                    or "other"
                ).strip()
                if not root_svc:
                    continue

                # Locate matching date-partitioned telemetry folder if present
                date_str = raw_ts[:10].replace("/", "_") if len(raw_ts) >= 10 else ""
                day_dir = telemetry_dir / date_str if date_str and (telemetry_dir / date_str).exists() else telemetry_dir
                metric_files = sorted(day_dir.rglob("*metric*.csv")) if day_dir.exists() else []
                log_files = sorted(day_dir.rglob("*log*.csv")) if day_dir.exists() else []
                trace_files = sorted(day_dir.rglob("*trace*.csv")) if day_dir.exists() else []

                if not metric_files and not log_files and not trace_files:
                    continue

                rec = RealIncidentRecord(
                    case_id=f"openrca-{sys_rel.lower().replace('/', '-')}-{counter:05d}",
                    source_dataset=f"OpenRCA-{sys_rel}",
                    system_name=f"openrca-{sys_rel.lower().replace('/', '-')}",
                    inject_timestamp=ts,
                    ground_truth_service=root_svc,
                    ground_truth_fault=fault_type,
                    metrics_path=str(metric_files[0]) if metric_files else "",
                    logs_path=str(log_files[0]) if log_files else None,
                    traces_path=str(trace_files[0]) if trace_files else None,
                    is_dejavu_long_format=False,
                )
                counter += 1
                if sys_rel in OPENRCA_HELDOUT_SYSTEMS:
                    heldout_records.append(rec)
                else:
                    train_records.append(rec)

    for fname, subset in (
        ("openrca_train.jsonl", train_records),
        ("openrca_heldout.jsonl", heldout_records),
    ):
        with (out_dir / fname).open("w", encoding="utf-8") as fh:
            for r in subset:
                fh.write(r.model_dump_json() + "\n")

    manifest = {
        "total_openrca_indexed": len(train_records) + len(heldout_records),
        "train_incidents": len(train_records),
        "heldout_incidents": len(heldout_records),
        "by_dataset": dict(Counter(r.source_dataset for r in train_records + heldout_records)),
    }
    (out_dir / "openrca_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & index microsoft/OpenRCA real enterprise telemetry")
    parser.add_argument("--work-dir", type=Path, default=Path("/mnt/nvme/real_telemetry/openrca"))
    parser.add_argument("--out-dir", type=Path, default=Path("/mnt/nvme/real_telemetry/index"))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    repo_dir = args.work_dir / "OpenRCA"
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        if not repo_dir.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/microsoft/OpenRCA.git", str(repo_dir)],
                check=True,
            )
        print("[openrca] Downloading OpenRCA telemetry (Bank, Market, Telecom) from Hugging Face Hub (cdreetz/OpenRCA)...")
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

        snapshot_download("cdreetz/OpenRCA", repo_type="dataset", local_dir=str(repo_dir / "dataset"))

    manifest = index_openrca_directory(repo_dir, args.out_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
