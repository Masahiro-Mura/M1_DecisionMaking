"""
Advantage and return estimation.

The main paper uses A_hat in Eq. (8)-(10) and Algorithm 1 says
"Calculate Advantage", but it does not state the estimator.

For the first reproduction, Generalized Advantage Estimation (GAE) is used
with lambda from rlParameters.py, following the cited PPO literature [47].
This is a cited-reference-inspired implementation assumption.
"""

from __future__ import annotations

import numpy as np

from parameters.rlParameters import (
    DISCOUNT_FACTOR_GAMMA,
    GAE_LAMBDA,
)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    gamma: float = DISCOUNT_FACTOR_GAMMA,
    gae_lambda: float = GAE_LAMBDA,
    episode_ends: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE advantages and bootstrapped returns.

    terminated=1 blocks bootstrap. Gymnasium truncation is deliberately not
    treated as a terminal state, so a time-limit transition may bootstrap.
    episode_ends blocks the GAE recursion at BOTH termination and truncation.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=np.float64)
    episode_ends = (terminated.copy() if episode_ends is None
                    else np.asarray(episode_ends, dtype=np.float64))

    n = rewards.shape[0]

    if not (
        values.shape == (n,)
        and next_values.shape == (n,)
        and terminated.shape == (n,)
        and episode_ends.shape == (n,)
    ):
        raise ValueError("All GAE arrays must have shape (N,)")

    advantages = np.zeros(n, dtype=np.float64)
    gae = 0.0

    for t in reversed(range(n)):
        nonterminal = 1.0 - terminated[t]

        delta = (
            rewards[t]
            + gamma * next_values[t] * nonterminal
            - values[t]
        )

        gae = (
            delta
            + gamma * gae_lambda * (1.0 - max(episode_ends[t], terminated[t])) * gae
        )
        advantages[t] = gae

    returns = advantages + values

    return (
        advantages.astype(np.float32),
        returns.astype(np.float32),
    )


def one_step_td_targets(
    rewards: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    gamma: float = DISCOUNT_FACTOR_GAMMA,
) -> np.ndarray:
    """
    TD target matching Main Paper Eq. (12):

        r_t + gamma * V(s_{t+1})

    with bootstrap removed at true terminal states.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=np.float64)

    return (
        rewards
        + gamma * next_values * (1.0 - terminated)
    ).astype(np.float32)
