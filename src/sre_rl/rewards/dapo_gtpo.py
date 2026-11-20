"""Multi-Turn DAPO + GTPO (Group Turn Policy Optimization) Estimator.

Solves three failure modes of vanilla trajectory-level GRPO on 25-turn SRE incidents:
1. Vanishing Advantages (DAPO Dynamic Filtering): Excludes prompt groups where all `G`
   sampled trajectories get identical rewards (`std < min_group_std`).
2. Entropy Collapse (DAPO Clip-Higher): Decouples `clip_eps_low=0.20` and
   `clip_eps_high=0.28` so low-probability exploratory diagnostic tools can increase
   in likelihood.
3. Multi-Turn Credit Assignment (GTPO Hybrid Advantage): Combines discounted per-turn
   shaping returns with dampened group-normalized outcome advantage, projected strictly
   onto assistant `<think>` + `<tool_call>` tokens (`action_mask = 1`, env tokens `= 0`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DAPOConfig:
    group_size: int = 8
    clip_eps_low: float = 0.20
    clip_eps_high: float = 0.28
    gamma: float = 0.92
    outcome_dampening_lambda: float = 0.30
    min_group_std: float = 1e-4
    min_group_max_reward: float = 0.50
    gate_positive_shaping_on_success: bool = True
    norm_eps: float = 1e-6


@dataclass(frozen=True)
class RolloutGroupItem:
    outcome_reward: float
    turn_rewards: list[float]
    turn_token_spans: list[tuple[int, int]]
    total_seq_len: int


def filter_zero_variance_groups(
    group_outcome_rewards: np.ndarray,
    min_std: float = 1e-4,
    min_max_reward: float | None = None,
) -> np.ndarray:
    """Return boolean mask of shape `(num_prompts,)` keeping informative groups.

    Excludes uniform zero-variance groups (`std <= min_std`) and, when `min_max_reward`
    is set (e.g. `0.50`), also excludes zero-winner groups where every rollout failed so
    DAPO cannot reinforce read-only timeout stalling over failed write attempts.
    """
    if group_outcome_rewards.ndim != 2:
        raise ValueError("`group_outcome_rewards` must have shape (num_prompts, group_size).")
    stds = np.std(group_outcome_rewards, axis=1)
    keep = stds > min_std
    if min_max_reward is not None:
        keep = keep & (np.max(group_outcome_rewards, axis=1) >= min_max_reward)
    return keep


def build_assistant_action_mask(
    total_seq_len: int,
    assistant_spans: list[tuple[int, int]],
) -> np.ndarray:
    """Create binary loss mask (`1.0` on assistant tokens, `0.0` on system/env observations)."""
    mask = np.zeros(total_seq_len, dtype=np.float32)
    for start, end in assistant_spans:
        s = max(0, min(total_seq_len, start))
        e = max(s, min(total_seq_len, end))
        mask[s:e] = 1.0
    return mask


def compute_gtpo_hybrid_advantages(
    group: list[RolloutGroupItem],
    config: DAPOConfig | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute token-level GTPO hybrid advantages and assistant action masks for one prompt group.

    Formula per rollout `i`, turn `k` (out of `K_i` turns):
        G_{i,k} = sum_{l=k}^{K_i - 1} gamma^{l-k} r_{i,l} + gamma^{K_i - k} o_i
        A^{hybrid}_{i,k} = GroupNorm(G_{i,k}) + lambda * A^{outcome}_i
    """
    cfg = config or DAPOConfig()
    if not group:
        return [], []

    outcomes = np.array([item.outcome_reward for item in group], dtype=np.float64)
    out_std = float(np.std(outcomes))
    if out_std < cfg.min_group_std:
        # Uniform group: zero advantage across all tokens (should be filtered by DAPO)
        zero_advs = [np.zeros(item.total_seq_len, dtype=np.float32) for item in group]
        masks = [
            build_assistant_action_mask(item.total_seq_len, item.turn_token_spans)
            for item in group
        ]
        return zero_advs, masks

    outcome_adv = (outcomes - float(np.mean(outcomes))) / (out_std + cfg.norm_eps)

    # Compute discounted turn returns G_{i,k} for every turn of every rollout in the group
    per_rollout_returns: list[np.ndarray] = []
    all_turn_returns: list[float] = []

    for item in group:
        num_turns = len(item.turn_rewards)
        if num_turns != len(item.turn_token_spans):
            raise ValueError(
                f"Mismatch between turn_rewards ({num_turns}) and "
                f"turn_token_spans ({len(item.turn_token_spans)})."
            )
        returns_k = np.zeros(num_turns, dtype=np.float64)
        running = item.outcome_reward
        allow_pos_shaping = (
            not cfg.gate_positive_shaping_on_success
            or item.outcome_reward >= cfg.min_group_max_reward
        )
        for k in range(num_turns - 1, -1, -1):
            r_k = item.turn_rewards[k]
            if not allow_pos_shaping and r_k > 0.0:
                r_k = 0.0
            running = r_k + cfg.gamma * running
            returns_k[k] = running
        per_rollout_returns.append(returns_k)
        all_turn_returns.extend(returns_k.tolist())

    flat_returns = np.array(all_turn_returns, dtype=np.float64)
    ret_mean = float(np.mean(flat_returns)) if flat_returns.size > 0 else 0.0
    ret_std = float(np.std(flat_returns)) if flat_returns.size > 0 else 0.0

    token_advantages: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []

    for idx, item in enumerate(group):
        mask = build_assistant_action_mask(item.total_seq_len, item.turn_token_spans)
        adv_seq = np.zeros(item.total_seq_len, dtype=np.float32)
        returns_k = per_rollout_returns[idx]

        for k, (start, end) in enumerate(item.turn_token_spans):
            norm_ret = (
                (returns_k[k] - ret_mean) / (ret_std + cfg.norm_eps)
                if ret_std >= cfg.min_group_std
                else 0.0
            )
            hybrid_adv = float(norm_ret + cfg.outcome_dampening_lambda * outcome_adv[idx])
            s = max(0, min(item.total_seq_len, start))
            e = max(s, min(item.total_seq_len, end))
            adv_seq[s:e] = hybrid_adv

        token_advantages.append(adv_seq * mask)
        action_masks.append(mask)

    return token_advantages, action_masks


def compute_dapo_clipped_loss_numpy(
    new_logprobs: np.ndarray,
    old_logprobs: np.ndarray,
    advantages: np.ndarray,
    action_mask: np.ndarray,
    config: DAPOConfig | None = None,
) -> float:
    """Compute token-level normalized DAPO surrogate loss (`-J_DAPO`, to minimize).

    Uses decoupled asymmetric clipping `[1 - clip_eps_low, 1 + clip_eps_high]` and
    normalizes strictly by total active assistant tokens across the group.
    """
    cfg = config or DAPOConfig()
    active_tokens = float(np.sum(action_mask))
    if active_tokens <= 0.0:
        return 0.0

    log_ratio = (new_logprobs - old_logprobs) * action_mask
    ratio = np.exp(np.clip(log_ratio, -20.0, 20.0))
    clipped_ratio = np.clip(ratio, 1.0 - cfg.clip_eps_low, 1.0 + cfg.clip_eps_high)

    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    token_obj = np.minimum(surr1, surr2) * action_mask
    return float(-np.sum(token_obj) / active_tokens)


def compute_masked_cispo_loss_torch(
    new_lp: Any,
    old_lp: Any,
    active_advantages: Any,
    ratio_min: float = 0.20,
    ratio_max: float = 5.00,
) -> tuple[Any, dict[str, float]]:
    """Compute ResolveAI SRE-1 / ScaleRL Masked CISPO policy objective (zero KL penalty).

    Formula:
        r_t = exp(new_lp_t - old_lp_t)
        M_t = 1[ratio_min <= r_t <= ratio_max]
        L_CISPO = - mean_t( M_t * stop_grad(min(r_t, ratio_max)) * A_t * new_lp_t )
    """
    import torch

    if new_lp.numel() == 0:
        zero = (new_lp * 0.0).sum()
        return zero, {"masked_fraction": 0.0, "mean_ratio": 1.0, "approx_k3_kl": 0.0}

    log_ratio = torch.clamp(new_lp - old_lp, -15.0, 15.0)
    ratio = torch.exp(log_ratio)
    in_band = (ratio >= ratio_min) & (ratio <= ratio_max)
    mask_f = in_band.to(dtype=torch.float32)

    # Stop-gradient importance weight inside [ratio_min, ratio_max], zero outside
    weight = torch.clamp(ratio.detach(), max=ratio_max) * mask_f
    denom = torch.clamp(mask_f.sum(), min=1.0)
    loss = -(weight * active_advantages.detach() * new_lp).sum() / denom

    # Schulman k3 KL diagnostic between trainer and sampler (after masking)
    inv_log_r = -log_ratio
    k3 = ((torch.exp(torch.clamp(inv_log_r, -10.0, 10.0)) - inv_log_r - 1.0) * mask_f).sum() / denom
    masked_frac = float(1.0 - (mask_f.sum() / max(1, ratio.numel())).item())
    mean_r = float(((ratio * mask_f).sum() / denom).item())
    return loss, {
        "masked_fraction": round(masked_frac, 4),
        "mean_ratio": round(mean_r, 4),
        "approx_k3_kl": round(float(k3.item()), 5),
    }
