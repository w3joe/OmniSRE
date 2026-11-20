"""Verifiable SRE rewards and Multi-Turn DAPO + GTPO advantage/loss estimators."""

from sre_rl.rewards.dapo_gtpo import (
    DAPOConfig,
    RolloutGroupItem,
    build_assistant_action_mask,
    compute_dapo_clipped_loss_numpy,
    compute_gtpo_hybrid_advantages,
    compute_masked_cispo_loss_torch,
    filter_zero_variance_groups,
)
from sre_rl.rewards.verifiable_reward import score_episode, shortest_service_hop_distance

__all__ = [
    "DAPOConfig",
    "RolloutGroupItem",
    "build_assistant_action_mask",
    "compute_dapo_clipped_loss_numpy",
    "compute_gtpo_hybrid_advantages",
    "compute_masked_cispo_loss_torch",
    "filter_zero_variance_groups",
    "score_episode",
    "shortest_service_hop_distance",
]
