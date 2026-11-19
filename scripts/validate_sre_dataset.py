#!/usr/bin/env python3
"""Run the 5-Gate Production SRE Dataset Validator on a scraped dataset directory."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from sre_rl.data.validator import validate_dataset_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate scraped SRE SFT + RLVR dataset")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/real_sre_v2"))
    parser.add_argument("--oracle-sample-size", type=int, default=150)
    parser.add_argument("--auto-filter-invalid-sft", action="store_true")
    args = parser.parse_args()

    report = validate_dataset_directory(
        dataset_dir=args.dataset_dir,
        oracle_sample_size=args.oracle_sample_size,
        auto_filter_invalid_sft=args.auto_filter_invalid_sft,
    )
    print(json.dumps(asdict(report), indent=2))
    if not report.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
