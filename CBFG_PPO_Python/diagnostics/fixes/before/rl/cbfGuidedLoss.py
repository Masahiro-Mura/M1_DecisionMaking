"""
CBF-guided actor loss from Main Paper Eq. (27)-(31).

Paper-supported logic
---------------------
- Eq. (28): omega_i = 1 + exp(Delta a_i^CBF)
- Eq. (29): intervention when ||Delta a^CBF|| > delta
- Eq. (30): store intervention flag
- Eq. (31):
    non-intervention samples -> PPO clipped objective
    intervention samples     -> behavior-cloning objective

Important dimensional ambiguity in the main paper
-------------------------------------------------
The behavioral-planning action is [v_ref, theta_ref], while the CBF layer
corrects the MPC low-level input [acceleration, steering]. Nevertheless,
Eq. (27)/(31) compares a CBF-corrected behavior directly with pi_theta(.|s).

The paper does not specify the mapping from low-level CBF correction back into
the actor's normalized action space. To keep the first implementation
executable, this module uses an explicit implementation assumption:

    normalized_demo =
        clip(policy_action + normalized_low_level_correction, -1, 1)

where each low-level correction component is scaled by its corresponding
physical control range.

This bridge should be treated as a reproduction uncertainty and can later be
replaced without changing PPO internals.
"""

from __future__ import annotations

import numpy as np
import torch

from parameters.mpcParameters import (
    CONTROL_LOWER_BOUND,
    CONTROL_UPPER_BOUND,
)


def correction_amplitude(
    cbf_corrections: torch.Tensor,
) -> torch.Tensor:
    """Euclidean correction amplitude used in Eq. (28)-(29)."""
    return torch.linalg.vector_norm(
        cbf_corrections,
        ord=2,
        dim=-1,
    )


def intervention_weight(
    cbf_corrections: torch.Tensor,
) -> torch.Tensor:
    """Eq. (28): omega = 1 + exp(correction amplitude)."""
    amplitude = correction_amplitude(cbf_corrections)
    return 1.0 + torch.exp(amplitude)


def behavior_cloning_loss(
    predicted_actions: torch.Tensor,
    demonstration_actions: torch.Tensor,
    cbf_corrections: torch.Tensor,
) -> torch.Tensor:
    """
    Squared weighted BC term consistent with Main Paper Eq. (31).
    """
    weights = intervention_weight(cbf_corrections)

    squared_error = (
        predicted_actions - demonstration_actions
    ).pow(2).sum(dim=-1)

    return (weights * squared_error).mean()


def build_demonstration_action(
    policy_action: np.ndarray,
    cbf_correction: np.ndarray,
) -> np.ndarray:
    """
    Construct an actor-space demonstration under the documented v1 assumption.

    policy_action:
        normalized [v_ref, theta_ref] in [-1,1]^2

    cbf_correction:
        physical low-level [Delta acceleration, Delta steering]
    """
    policy_action = np.asarray(policy_action, dtype=np.float64)
    correction = np.asarray(cbf_correction, dtype=np.float64)

    if policy_action.shape != (2,) or correction.shape != (2,):
        raise ValueError(
            "policy_action and cbf_correction must both have shape (2,)"
        )

    control_span = (
        np.asarray(CONTROL_UPPER_BOUND, dtype=np.float64)
        - np.asarray(CONTROL_LOWER_BOUND, dtype=np.float64)
    )

    normalized_correction = 2.0 * correction / control_span

    return np.clip(
        policy_action + normalized_correction,
        -1.0,
        1.0,
    ).astype(np.float32)
