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

References [43] Sec. 4.1 and [44] Eq. (5)-(6) use demonstrated and
policy actions in the same action space. Neither reference specifies this
paper's low-level-control to reference-action conversion.

IMPLEMENTATION ASSUMPTION: add ONLY the one-step speed/heading difference
between corrected and nominal control to the original policy reference.
This retains the original intended reference under zero/tiny corrections and
does not replace it with the vehicle's next realized state. It is a local
correction transfer, NOT an inverse of the horizon MPC or a safety proof for
the learned references. Both BC error and Eq. (28)'s correction amplitude use
the same normalized reference-action coordinates. Raw acceleration/steering
corrections remain separate safety-controller quantities.

[44]'s maximum-likelihood BC and decaying DAPG weights and [43]'s off-policy
DDPGfD replay are NOT substituted for the main paper's Eq. (31) squared BC,
Eq. (28) exponential weights, or on-policy PPO.
"""

from __future__ import annotations

import numpy as np
import torch

from environment.actionAdapter import action_to_reference, reference_to_action
from mpc.kinematicBicycleModel import discrete_step


def correction_amplitude(
    behavior_corrections: torch.Tensor,
) -> torch.Tensor:
    """Dimensionless normalized reference-action amplitude for Eq. (28)."""
    return torch.linalg.vector_norm(
        behavior_corrections,
        ord=2,
        dim=-1,
    )


def intervention_weight(
    behavior_corrections: torch.Tensor,
) -> torch.Tensor:
    """Eq. (28), using rollout teacher - original normalized policy action."""
    amplitude = correction_amplitude(behavior_corrections)
    return 1.0 + torch.exp(amplitude)


def behavior_cloning_loss(
    predicted_actions: torch.Tensor,
    demonstration_actions: torch.Tensor,
    behavior_corrections: torch.Tensor,
) -> torch.Tensor:
    """
    Squared weighted BC term consistent with Main Paper Eq. (31).
    """
    weights = intervention_weight(behavior_corrections)

    squared_error = (
        predicted_actions - demonstration_actions
    ).pow(2).sum(dim=-1)

    return (weights * squared_error).mean()


def build_demonstration_action(
    policy_action: np.ndarray,
    cbf_correction: np.ndarray,
    ego_state: np.ndarray | None = None,
    nominal_control: np.ndarray | None = None,
    minimum_reference_speed: float = 0.,
) -> np.ndarray:
    """
    Transfer the model-predicted effect of a feasible correction to the reference.

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
    nominal_control = np.asarray(nominal_control, dtype=np.float64)
    nominal_state = discrete_step(ego_state, nominal_control)
    safe_state = discrete_step(ego_state, nominal_control + correction)
    reference_delta = safe_state[2:4] - nominal_state[2:4]
    original_reference = action_to_reference(policy_action, minimum_speed=minimum_reference_speed)
    return reference_to_action(
        original_reference.velocity + float(reference_delta[0]),
        original_reference.heading + float(reference_delta[1]),
        minimum_speed=minimum_reference_speed,
    )
