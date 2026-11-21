#!/usr/bin/env python3
"""Parse and summarize official external benchmark artifacts (`microsoft/OpenRCA` + Live `SREGym-Lite`).

Works seamlessly against either:
- `final_external_benchmarks.json`
- Raw `sregym_runs_all/<run_ts>/stratus/<problem>/*_stratus_results.csv` + `openrca_summary.json`
  downloaded from `gs://onflow-30k-qwen38-sre-rl/omnisre-v1.0/external-benchmarks/`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_sregym_run_dir(run_dir: Path) -> dict[str, Any]:
    """Deduplicate cumulative per-problem SREGym CSVs inside a `<run_ts>` directory."""
    csv_files = sorted(run_dir.glob("*_stratus_results.csv")) + sorted(
        run_dir.glob("*/*/*_stratus_results.csv")
    )
    by_problem: dict[str, dict[str, str]] = {}
    ordered_problems: list[str] = []

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
            for row in csv.DictReader(fh):
                pid = (row.get("problem_id") or "").strip()
                if not pid:
                    continue
                if pid not in by_problem:
                    ordered_problems.append(pid)
                by_problem[pid] = row

    rows = [by_problem[pid] for pid in ordered_problems]
    n = len(rows)
    if n == 0:
        return {
            "run_dir": run_dir.name,
            "problems": 0,
            "completed_rate": 0.0,
            "diagnosis_pass_rate": 0.0,
            "diagnosis_composite_mean": 0.0,
            "mitigation_pass_rate": 0.0,
            "rows": [],
        }

    truthy = {"true", "1", "pass", "passed"}
    completed = sum(1 for r in rows if (r.get("run_status") or "").strip().lower() == "complete")
    diag_ok = sum(1 for r in rows if (r.get("Diagnosis.success") or "").strip().lower() in truthy)
    mit_ok = sum(1 for r in rows if (r.get("Mitigation.success") or "").strip().lower() in truthy)

    comp_vals: list[float] = []
    for r in rows:
        raw_c = (r.get("Diagnosis.composite_score") or "").strip()
        try:
            comp_vals.append(float(raw_c))
        except ValueError:
            comp_vals.append(0.0)

    return {
        "run_dir": run_dir.name,
        "problems": n,
        "completed_count": completed,
        "completed_rate": round(completed / n, 4),
        "diagnosis_pass_count": diag_ok,
        "diagnosis_pass_rate": round(diag_ok / n, 4),
        "diagnosis_composite_mean": round(sum(comp_vals) / n, 4),
        "mitigation_pass_count": mit_ok,
        "mitigation_pass_rate": round(mit_ok / n, 4),
        "per_problem": [
            {
                "problem_id": r.get("problem_id"),
                "run_status": r.get("run_status"),
                "diagnosis_success": (r.get("Diagnosis.success") or "").strip().lower() in truthy,
                "diagnosis_composite": r.get("Diagnosis.composite_score") or "0.0",
                "mitigation_success": (r.get("Mitigation.success") or "").strip().lower() in truthy,
                "timed_out": (r.get("timed_out") or "").strip().lower() in truthy,
            }
            for r in rows
        ],
    }


def summarize_external_dir(ext_dir: Path) -> dict[str, Any]:
    openrca_path = ext_dir / "openrca_summary.json"
    openrca = json.loads(openrca_path.read_text(encoding="utf-8")) if openrca_path.exists() else {}

    sregym_runs: list[dict[str, Any]] = []
    for sub_name in ("sregym_runs_all", "sregym_runs_after_omnisre", "results"):
        candidate_root = ext_dir / sub_name
        if candidate_root.is_dir():
            for child in sorted(candidate_root.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    parsed = parse_sregym_run_dir(child)
                    if parsed["problems"] > 0:
                        sregym_runs.append(parsed)
            if sregym_runs:
                break

    return {
        "openrca_official": openrca,
        "sregym_lite_live_k8s": sregym_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OpenRCA + Live SREGym-Lite external results")
    parser.add_argument("--ext-dir", type=Path, required=True, help="Local directory synced from external-benchmarks/")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    summary = summarize_external_dir(args.ext_dir)
    text = json.dumps(summary, indent=2)
    if args.out_json:
        args.out_json.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
