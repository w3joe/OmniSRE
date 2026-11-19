"""Tests for Multi-Turn DAPO + GTPO advantage estimation and asymmetric surrogate loss."""

import numpy as np

from sre_rl.rewards.dapo_gtpo import (
    DAPOConfig,
    RolloutGroupItem,
    build_assistant_action_mask,
    compute_dapo_clipped_loss_numpy,
    compute_gtpo_hybrid_advantages,
    filter_zero_variance_groups,
)


def test_dapo_filters_all_pass_and_all_fail_zero_variance_groups() -> None:
    rewards = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],     # all fail -> drop
            [1.0, 1.0, 1.0, 1.0],     # all pass -> drop
            [-1.0, 0.2, 0.9, 1.0],    # informative variance -> keep
        ],
        dtype=np.float64,
    )
    keep = filter_zero_variance_groups(rewards)
    assert keep.tolist() == [False, False, True]


def test_gtpo_hybrid_advantage_masks_env_tokens_and_credits_good_early_turns() -> None:
    cfg = DAPOConfig(group_size=2, gamma=0.9, outcome_dampening_lambda=0.3)
    # Rollout 0: Strong early localization (+0.05) + full mitigation outcome (+1.0)
    # Rollout 1: Good early turn (+0.05), bad turn 2 (-0.10), failed outcome (0.0)
    group = [
        RolloutGroupItem(
            outcome_reward=1.0,
            turn_rewards=[0.05, 0.10],
            turn_token_spans=[(10, 20), (35, 45)],
            total_seq_len=60,
        ),
        RolloutGroupItem(
            outcome_reward=0.0,
            turn_rewards=[0.05, -0.10],
            turn_token_spans=[(10, 20), (35, 45)],
            total_seq_len=60,
        ),
    ]

    advs, masks = compute_gtpo_hybrid_advantages(group, config=cfg)
    assert len(advs) == 2

    # Environment / system observation tokens [0:10], [20:35], [45:60] MUST have 0 mask and 0 advantage
    assert np.all(masks[0][:10] == 0.0)
    assert np.all(advs[0][:10] == 0.0)
    assert np.all(advs[0][20:35] == 0.0)

    # Winning rollout assistant tokens get positive advantage; losing rollout gets negative
    assert float(np.mean(advs[0][10:20])) > 0.0
    assert float(np.mean(advs[1][35:45])) < 0.0
    # In Rollout 1, turn 0 (+0.05) should be penalized less harshly than turn 1 (-0.10)
    assert float(advs[1][10]) > float(advs[1][35])


def test_dapo_decoupled_clip_higher_surrogate_loss() -> None:
    cfg = DAPOConfig(clip_eps_low=0.20, clip_eps_high=0.28)
    mask = build_assistant_action_mask(10, [(2, 8)])
    old_lp = np.full(10, -2.0, dtype=np.float64)
    # Ratio = exp(0.24) ~= 1.271 -> above standard 1.20 clip, but INSIDE DAPO 1.28 clip-higher!
    new_lp = np.full(10, -1.76, dtype=np.float64)
    adv = np.full(10, 1.5, dtype=np.float64) * mask

    loss = compute_dapo_clipped_loss_numpy(new_lp, old_lp, adv, mask, config=cfg)
    expected_ratio = np.exp(0.24)
    assert np.isclose(loss, -(expected_ratio * 1.5), atol=1e-4)
