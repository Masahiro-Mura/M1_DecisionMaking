"""
State-value critic V_omega(s).

Main paper:
    Eq. (12): critic TD-error loss
    Fig. 6: two Dense(256) hidden layers with ReLU and scalar value output
"""

from __future__ import annotations

import torch
from torch import nn

from parameters.rlParameters import (
    CRITIC_HIDDEN_DIMS,
    OBSERVATION_DIM,
)


class Critic(nn.Module):
    """Value network V_omega(s)."""

    def __init__(self) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        input_dim = OBSERVATION_DIM

        for hidden_dim in CRITIC_HIDDEN_DIMS:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return state values with shape (batch,)."""
        return self.network(states).squeeze(-1)
