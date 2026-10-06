"""
Stochastic Gaussian actor for the behavioral-planning layer.

Main paper:
    - policy pi_theta(a_t | s_t), Eq. (8)-(11)
    - action: [v_ref, theta_ref], Eq. (1)
    - Fig. 6: two Dense(256) hidden layers, ReLU
    - action components are normalized to [-1, 1]

Implementation detail
---------------------
The paper does not specify how Gaussian samples are constrained to [-1, 1].
This implementation uses a tanh-squashed Gaussian so sampled actions always
respect the paper's normalized action bounds.

The paper also does not report the initial Gaussian standard deviation.
A trainable, state-independent log standard deviation initialized from
rlParameters.INITIAL_LOG_STD is used as an implementation assumption.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from parameters.rlParameters import (
    ACTION_DIM,
    ACTOR_HIDDEN_DIMS,
    INITIAL_LOG_STD,
    OBSERVATION_DIM,
)


_LOG_PROB_EPS = 1e-6


class Actor(nn.Module):
    """Gaussian policy with tanh-bounded normalized actions."""

    def __init__(self) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        input_dim = OBSERVATION_DIM

        for hidden_dim in ACTOR_HIDDEN_DIMS:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(input_dim, ACTION_DIM)

        self.log_std = nn.Parameter(
            torch.full(
                (ACTION_DIM,),
                float(INITIAL_LOG_STD),
                dtype=torch.float32,
            )
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return the unsquashed Gaussian mean."""
        features = self.backbone(states)
        return self.mean_head(features)

    def distribution(self, states: torch.Tensor) -> Normal:
        """Return the underlying independent Gaussian distribution."""
        mean = self.forward(states)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def _squash_log_prob(
        distribution: Normal,
        pre_tanh_action: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Log probability after tanh transform.

        log pi(a|s) =
            log N(z; mu, sigma)
            - sum log(1 - tanh(z)^2)
        """
        base_log_prob = distribution.log_prob(pre_tanh_action).sum(dim=-1)

        correction = torch.log(
            1.0 - action.pow(2) + _LOG_PROB_EPS
        ).sum(dim=-1)

        return base_log_prob - correction

    def sample(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample normalized action and return (action, log_prob)."""
        distribution = self.distribution(states)
        pre_tanh_action = distribution.rsample()
        action = torch.tanh(pre_tanh_action)

        log_prob = self._squash_log_prob(
            distribution,
            pre_tanh_action,
            action,
        )

        return action, log_prob

    def deterministic_action(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Return tanh(mean), used for evaluation and BC regression."""
        return torch.tanh(self.forward(states))

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities of already-sampled bounded actions.

        Returns
        -------
        log_prob:
            Shape (batch,).
        entropy:
            Entropy of the underlying Gaussian, shape (batch,).
            The tanh-transformed distribution has no simple closed-form entropy.
            ENTROPY_COEFFICIENT is zero in the current reproduction settings.
        """
        clipped_actions = torch.clamp(
            actions,
            -1.0 + _LOG_PROB_EPS,
            1.0 - _LOG_PROB_EPS,
        )

        pre_tanh_action = torch.atanh(clipped_actions)
        distribution = self.distribution(states)

        log_prob = self._squash_log_prob(
            distribution,
            pre_tanh_action,
            clipped_actions,
        )

        entropy = distribution.entropy().sum(dim=-1)

        return log_prob, entropy
