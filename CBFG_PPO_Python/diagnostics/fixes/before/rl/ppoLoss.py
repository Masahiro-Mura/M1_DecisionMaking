"""
Standard PPO loss terms from Main Paper Eq. (10)-(13).

For gradient descent, the paper's actor objective L_CLIP (to be maximized)
is returned as a negative loss.

The critic loss follows Eq. (12) using one-step TD targets.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from parameters.rlParameters import PPO_CLIP_EPSILON


def ppo_actor_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = PPO_CLIP_EPSILON,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Main Paper Eq. (10)-(11).

    Returns
    -------
    loss:
        Negative clipped surrogate objective for gradient descent.
    ratio:
        pi_theta(a|s) / pi_theta_old(a|s).
    """
    ratio = torch.exp(new_log_probs - old_log_probs)

    unclipped = ratio * advantages
    clipped = (
        torch.clamp(
            ratio,
            1.0 - clip_epsilon,
            1.0 + clip_epsilon,
        )
        * advantages
    )

    objective = torch.minimum(unclipped, clipped).mean()

    return -objective, ratio


def critic_td_loss(
    values: torch.Tensor,
    td_targets: torch.Tensor,
) -> torch.Tensor:
    """
    Main Paper Eq. (12):
        E[(r + gamma V(s') - V(s))^2]
    """
    return F.mse_loss(values, td_targets)
