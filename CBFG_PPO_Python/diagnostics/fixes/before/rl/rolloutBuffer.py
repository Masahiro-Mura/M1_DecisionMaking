"""
On-policy rollout buffer for CBFG-PPO.

Main paper Algorithm 1 stores:
    (s_k, u_RL_k, u_CBF_k, r_k, s_{k+1})

Eq. (30) additionally distinguishes intervention and non-intervention samples.

The main paper calls D0 a replay buffer, but Algorithm 1 updates PPO from the
freshly collected trajectory. This implementation therefore uses an on-policy
rollout buffer rather than an off-policy replay buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from parameters.rlParameters import REPLAY_BUFFER_SIZE


@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    cbf_corrections: torch.Tensor
    cbf_interventions: torch.Tensor
    demonstration_actions: torch.Tensor


class RolloutBuffer:
    """Fixed-capacity on-policy buffer."""

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = int(capacity)
        self.clear()

    def clear(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.next_states: list[np.ndarray] = []
        self.terminated: list[bool] = []
        self.old_log_probs: list[float] = []
        self.values: list[float] = []
        self.cbf_corrections: list[np.ndarray] = []
        self.cbf_interventions: list[bool] = []
        self.demonstration_actions: list[np.ndarray] = []

        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.rewards)

    @property
    def full(self) -> bool:
        return len(self) >= self.capacity

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        old_log_prob: float,
        value: float,
        cbf_correction: np.ndarray,
        cbf_intervention: bool,
        demonstration_action: np.ndarray,
    ) -> None:
        """Append one transition."""
        if self.full:
            raise RuntimeError(
                "RolloutBuffer is full. Update the agent and clear the buffer."
            )

        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.rewards.append(float(reward))
        self.next_states.append(
            np.asarray(next_state, dtype=np.float32).copy()
        )
        self.terminated.append(bool(terminated))
        self.old_log_probs.append(float(old_log_prob))
        self.values.append(float(value))
        self.cbf_corrections.append(
            np.asarray(cbf_correction, dtype=np.float32).copy()
        )
        self.cbf_interventions.append(bool(cbf_intervention))
        self.demonstration_actions.append(
            np.asarray(demonstration_action, dtype=np.float32).copy()
        )

    def set_advantages_and_returns(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> None:
        advantages = np.asarray(advantages, dtype=np.float32)
        returns = np.asarray(returns, dtype=np.float32)

        if advantages.shape != (len(self),):
            raise ValueError("advantages must have shape (buffer_length,)")
        if returns.shape != (len(self),):
            raise ValueError("returns must have shape (buffer_length,)")

        self.advantages = advantages.copy()
        self.returns = returns.copy()

    def as_batch(
        self,
        device: torch.device,
    ) -> RolloutBatch:
        """Convert the entire rollout to tensors."""
        if self.advantages is None or self.returns is None:
            raise RuntimeError(
                "Advantages/returns are not set. "
                "Call set_advantages_and_returns() first."
            )

        return RolloutBatch(
            states=torch.as_tensor(
                np.asarray(self.states),
                dtype=torch.float32,
                device=device,
            ),
            actions=torch.as_tensor(
                np.asarray(self.actions),
                dtype=torch.float32,
                device=device,
            ),
            rewards=torch.as_tensor(
                self.rewards,
                dtype=torch.float32,
                device=device,
            ),
            next_states=torch.as_tensor(
                np.asarray(self.next_states),
                dtype=torch.float32,
                device=device,
            ),
            terminated=torch.as_tensor(
                self.terminated,
                dtype=torch.float32,
                device=device,
            ),
            old_log_probs=torch.as_tensor(
                self.old_log_probs,
                dtype=torch.float32,
                device=device,
            ),
            values=torch.as_tensor(
                self.values,
                dtype=torch.float32,
                device=device,
            ),
            advantages=torch.as_tensor(
                self.advantages,
                dtype=torch.float32,
                device=device,
            ),
            returns=torch.as_tensor(
                self.returns,
                dtype=torch.float32,
                device=device,
            ),
            cbf_corrections=torch.as_tensor(
                np.asarray(self.cbf_corrections),
                dtype=torch.float32,
                device=device,
            ),
            cbf_interventions=torch.as_tensor(
                self.cbf_interventions,
                dtype=torch.bool,
                device=device,
            ),
            demonstration_actions=torch.as_tensor(
                np.asarray(self.demonstration_actions),
                dtype=torch.float32,
                device=device,
            ),
        )

    @staticmethod
    def minibatch_indices(
        size: int,
        batch_size: int,
        rng: np.random.Generator,
    ):
        """Yield shuffled index arrays for one PPO epoch."""
        indices = np.arange(size)
        rng.shuffle(indices)

        for start in range(0, size, batch_size):
            yield indices[start : start + batch_size]
