"""
MPC tracking cost from the main paper.

Main paper Eq. (17):

    J(z, u) = sum_k (
        ||z_ref - z_k||_Q^2
        + ||u_k||_R^2
    )

with

    z = [v, theta]^T.

The paper prints C = [0, 0, 1, 1] in Eq. (18c), but z is explicitly a
2-vector [v, theta]^T. Therefore this implementation extracts the third and
fourth state components directly, which is equivalent to the dimensionally
consistent matrix

    C = [[0, 0, 1, 0],
         [0, 0, 0, 1]].

This is treated as a correction of an apparent notation/printing issue in the
paper rather than as an additional model assumption.
"""

from __future__ import annotations

import numpy as np

from parameters.mpcParameters import Q, R


OUTPUT_DIM: int = 2
CONTROL_DIM: int = 2


def output_from_state(state: np.ndarray) -> np.ndarray:
    """Return z = [v, theta]^T from x = [x, y, v, theta]^T."""
    state = np.asarray(state, dtype=np.float64)

    if state.shape != (4,):
        raise ValueError(f"state must have shape (4,), got {state.shape}")

    return state[2:4].copy()


def quadratic_form(vector: np.ndarray, weight: np.ndarray) -> float:
    """Compute vector^T * weight * vector."""
    vector = np.asarray(vector, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)

    return float(vector.T @ weight @ vector)


def stage_tracking_cost(
    state: np.ndarray,
    control: np.ndarray,
    reference_output: np.ndarray,
    q_matrix: np.ndarray = Q,
    r_matrix: np.ndarray = R,
) -> float:
    """Cost of one state/control pair."""
    control = np.asarray(control, dtype=np.float64)
    reference_output = np.asarray(reference_output, dtype=np.float64)

    if control.shape != (CONTROL_DIM,):
        raise ValueError(f"control must have shape ({CONTROL_DIM},)")
    if reference_output.shape != (OUTPUT_DIM,):
        raise ValueError(f"reference_output must have shape ({OUTPUT_DIM},)")

    output = output_from_state(state)
    tracking_error = reference_output - output

    return (
        quadratic_form(tracking_error, q_matrix)
        + quadratic_form(control, r_matrix)
    )


def trajectory_cost(
    predicted_states: np.ndarray,
    control_sequence: np.ndarray,
    reference_output: np.ndarray,
    q_matrix: np.ndarray = Q,
    r_matrix: np.ndarray = R,
) -> float:
    """
    Compute the finite-horizon MPC cost.

    Implementation detail
    ---------------------
    Eq. (17) sums k=0,...,n_p-1. For N controls the rollout contains N+1
    states; only x_0,...,x_(N-1) enter the cost. No extra terminal penalty.
    """
    states = np.asarray(predicted_states, dtype=np.float64)
    controls = np.asarray(control_sequence, dtype=np.float64)
    reference = np.asarray(reference_output, dtype=np.float64)

    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError("predicted_states must have shape (N+1, 4)")
    if controls.ndim != 2 or controls.shape[1] != CONTROL_DIM:
        raise ValueError("control_sequence must have shape (N, 2)")
    if states.shape[0] != controls.shape[0] + 1:
        raise ValueError(
            "predicted_states must contain exactly one more state than controls."
        )
    if reference.shape != (OUTPUT_DIM,):
        raise ValueError("reference_output must have shape (2,)")

    cost = 0.0

    for state in states[:-1]:
        error = reference - output_from_state(state)
        cost += quadratic_form(error, q_matrix)

    for control in controls:
        cost += quadratic_form(control, r_matrix)

    return float(cost)


def tracking_objective_and_gradient(
    current_state: np.ndarray,
    control_sequence: np.ndarray,
    reference_output: np.ndarray,
    q_matrix: np.ndarray = Q,
    r_matrix: np.ndarray = R,
) -> tuple[float, np.ndarray]:
    """Exact Eq. (17) cost and adjoint gradient for the Eq. (16) model.

    Only speed/heading appear in the objective and their dynamics do not depend
    on x/y. Eliminating those unused states changes neither model nor cost.
    Supplying this derivative avoids 20 extra rollouts per numerical-gradient
    evaluation; horizon, bounds, SLSQP settings and terminal-cost rule are intact.
    """
    from parameters.mpcParameters import MPC_DT
    from parameters.vehicleParameters import WHEELBASE
    controls = np.asarray(control_sequence, dtype=np.float64)
    horizon = len(controls)
    outputs = np.empty((horizon + 1, 2), dtype=np.float64)
    outputs[0] = np.asarray(current_state, dtype=np.float64)[2:4]
    tangent = np.tan(controls[:, 1])
    sin_beta = tangent / np.sqrt(4. + tangent**2)
    d_sin_beta = 4. * (1. + tangent**2) / (4. + tangent**2)**1.5
    for k in range(horizon):
        v, heading = outputs[k]
        outputs[k+1] = (v + MPC_DT * controls[k, 0],
                        heading + MPC_DT * v / WHEELBASE * sin_beta[k])
    errors = outputs[:-1] - reference_output
    cost = float(np.einsum('ni,ij,nj->', errors, q_matrix, errors)
                 + np.einsum('ni,ij,nj->', controls, r_matrix, controls))
    state_gradient = errors @ (q_matrix + q_matrix.T)
    gradient = controls @ (r_matrix + r_matrix.T)
    adjoint = np.zeros(2)
    for k in range(horizon - 1, -1, -1):
        gradient[k, 0] += MPC_DT * adjoint[0]
        gradient[k, 1] += adjoint[1] * MPC_DT * outputs[k, 0] / WHEELBASE * d_sin_beta[k]
        adjoint = state_gradient[k] + np.array([
            adjoint[0] + adjoint[1] * MPC_DT / WHEELBASE * sin_beta[k], adjoint[1]])
    return cost, gradient.reshape(-1)
