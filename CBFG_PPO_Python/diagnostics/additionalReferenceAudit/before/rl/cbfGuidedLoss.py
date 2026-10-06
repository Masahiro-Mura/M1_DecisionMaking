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

IMPLEMENTATION ASSUMPTION: The paper does not specify the inverse mapping
from corrected acceleration/steering to a reference action. For interventions
we use the speed/heading reached by one model step with the feasible corrected
control, then invert actionAdapter. This is a dimensionally consistent local
behavior teacher, NOT an exact inverse of the horizon MPC or a safety proof
for future references. CBF filtering remains active at every subsequent step.
"""

from __future__ import annotations

import numpy as np
import torch

from environment.actionAdapter import reference_to_action
from mpc.kinematicBicycleModel import discrete_step


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
    ego_state: np.ndarray | None = None,
    nominal_control: np.ndarray | None = None,
) -> np.ndarray:
    """
    Construct a model-realized reference teacher for a feasible intervention.

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

    if np.linalg.norm(correction) <= 1e-8:
        return np.clip(policy_action, -1., 1.).astype(np.float32)
    if ego_state is None or nominal_control is None:
        raise ValueError('A nonzero CBF teacher requires physical ego_state and nominal_control.')
    safe_state = discrete_step(ego_state, np.asarray(nominal_control) + correction)
    return reference_to_action(float(safe_state[2]), float(safe_state[3]))
