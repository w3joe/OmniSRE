"""Anti-Contamination Train / Validation / Held-Out Split Strategy.

Random splits leak service names and fault signatures into evaluation. We enforce a
strict two-axis holdout:
1. Unseen Topology Holdout (`checkout-aurora-stack`)
2. Unseen Fault Mechanism Holdout (`k8s_configmap_dsn_port_typo`)
"""

from __future__ import annotations

from dataclasses import dataclass

from sre_rl.schemas import IncidentScenario

DEFAULT_HELDOUT_TOPOLOGIES: frozenset[str] = frozenset({"checkout-aurora-stack"})
DEFAULT_HELDOUT_FAULT_IDS: frozenset[str] = frozenset({"k8s_configmap_dsn_port_typo"})


@dataclass(frozen=True)
class DatasetSplits:
    train: list[IncidentScenario]
    val_in_domain: list[IncidentScenario]
    eval_heldout: list[IncidentScenario]


def split_scenarios_decontaminated(
    scenarios: list[IncidentScenario],
    heldout_topologies: frozenset[str] = DEFAULT_HELDOUT_TOPOLOGIES,
    heldout_fault_ids: frozenset[str] = DEFAULT_HELDOUT_FAULT_IDS,
    val_ratio: float = 0.10,
) -> DatasetSplits:
    heldout: list[IncidentScenario] = []
    in_domain: list[IncidentScenario] = []

    for sc in scenarios:
        if (
            sc.topology_name in heldout_topologies
            or sc.fault.fault_id in heldout_fault_ids
        ):
            heldout.append(sc)
        else:
            in_domain.append(sc)

    val_count = max(1, round(len(in_domain) * val_ratio)) if in_domain else 0
    val_in_domain = in_domain[:val_count]
    train = in_domain[val_count:]
    return DatasetSplits(
        train=train,
        val_in_domain=val_in_domain,
        eval_heldout=heldout,
    )
