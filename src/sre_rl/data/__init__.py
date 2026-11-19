"""Dataset A & Dataset B construction, scrubbing, trajectory repair, and splitting."""

from sre_rl.data.scenario_factory import (
    FAULT_TEMPLATES,
    TOPOLOGIES,
    generate_scenario,
    generate_scenario_suite,
)
from sre_rl.data.scrubber import TelemetryScrubber
from sre_rl.data.splits import DatasetSplits, split_scenarios_decontaminated
from sre_rl.data.trajectory_builder import (
    export_jsonl,
    export_parquet_or_jsonl,
    purify_trajectory,
    repair_failed_trajectory,
    synthesize_expert_trajectory,
    to_verl_rl_row,
)

__all__ = [
    "FAULT_TEMPLATES",
    "TOPOLOGIES",
    "DatasetSplits",
    "TelemetryScrubber",
    "export_jsonl",
    "export_parquet_or_jsonl",
    "generate_scenario",
    "generate_scenario_suite",
    "purify_trajectory",
    "repair_failed_trajectory",
    "split_scenarios_decontaminated",
    "synthesize_expert_trajectory",
    "to_verl_rl_row",
]
