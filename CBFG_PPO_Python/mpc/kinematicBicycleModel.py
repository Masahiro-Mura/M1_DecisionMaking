"""
Kinematic bicycle model used by the MPC motion-control layer.

Main paper:
    Eq. (14): continuous-time vehicle model
    Eq. (15): side-slip angle beta
    Eq. (16): forward-Euler discrete-time model

State:
    x = [X, Y, v, theta]^T

Control:
    u = [a_acc, delta]^T

Notes
-----
The paper explicitly defines

    beta = atan(0.5 * tan(delta))

rather than introducing separate front/rear axle distances. This module
therefore implements that expression directly.

The wheelbase value itself is not numerically reported in the paper and is
read from parameters/vehicleParameters.py, where it is marked as an
implementation assumption.
"""

from __future__ import annotations

import numpy as np

from parameters.mpcParameters import MPC_DT
from parameters.vehicleParameters import WHEELBASE


STATE_DIM: int = 4
CONTROL_DIM: int = 2


def slip_angle(steering_angle: float) -> float:
    """
    Compute beta from main-paper Eq. (15).

    Parameters
    ----------
    steering_angle:
        Front steering angle delta [rad].

    Returns
    -------
    float
        Side-slip angle beta [rad].
    """
    return float(np.arctan(0.5 * np.tan(float(steering_angle))))


def continuous_dynamics(
    state: np.ndarray,
    control: np.ndarray,
    wheelbase: float = WHEELBASE,
) -> np.ndarray:
    """
    Continuous-time kinematic bicycle model from Eq. (14).

    Parameters
    ----------
    state:
        [x, y, v, theta].
    control:
        [a_acc, delta].
    wheelbase:
        Vehicle wheelbase l [m].

    Returns
    -------
    np.ndarray
        [x_dot, y_dot, v_dot, theta_dot].
    """
    state = np.asarray(state, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)

    if state.shape != (STATE_DIM,):
        raise ValueError(f"state must have shape ({STATE_DIM},), got {state.shape}")
    if control.shape != (CONTROL_DIM,):
        raise ValueError(
            f"control must have shape ({CONTROL_DIM},), got {control.shape}"
        )
    if wheelbase <= 0.0:
        raise ValueError("wheelbase must be positive.")

    _, _, velocity, heading = state
    acceleration, steering_angle = control

    beta = slip_angle(steering_angle)

    return np.array(
        [
            velocity * np.cos(heading + beta),
            velocity * np.sin(heading + beta),
            acceleration,
            (velocity / wheelbase) * np.sin(beta),
        ],
        dtype=np.float64,
    )


def discrete_step(
    state: np.ndarray,
    control: np.ndarray,
    dt: float = MPC_DT,
    wheelbase: float = WHEELBASE,
) -> np.ndarray:
    """
    One-step forward-Euler update from main-paper Eq. (16).

    x(k+1) = x(k) + f(x(k), u(k)) * Ts
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    state = np.asarray(state, dtype=np.float64)
    derivative = continuous_dynamics(
        state=state,
        control=control,
        wheelbase=wheelbase,
    )

    return state + derivative * float(dt)


def rollout(
    initial_state: np.ndarray,
    control_sequence: np.ndarray,
    dt: float = MPC_DT,
    wheelbase: float = WHEELBASE,
) -> np.ndarray:
    """
    Predict a state trajectory for an MPC control sequence.

    Parameters
    ----------
    initial_state:
        Shape (4,).
    control_sequence:
        Shape (N, 2).

    Returns
    -------
    np.ndarray
        Predicted states with shape (N+1, 4), including x_0.
    """
    initial_state = np.asarray(initial_state, dtype=np.float64)
    controls = np.asarray(control_sequence, dtype=np.float64)

    if initial_state.shape != (STATE_DIM,):
        raise ValueError(
            f"initial_state must have shape ({STATE_DIM},), "
            f"got {initial_state.shape}"
        )

    if controls.ndim != 2 or controls.shape[1] != CONTROL_DIM:
        raise ValueError(
            f"control_sequence must have shape (N, {CONTROL_DIM}), "
            f"got {controls.shape}"
        )

    states = np.empty((controls.shape[0] + 1, STATE_DIM), dtype=np.float64)
    states[0] = initial_state

    for k, control in enumerate(controls):
        states[k + 1] = discrete_step(
            state=states[k],
            control=control,
            dt=dt,
            wheelbase=wheelbase,
        )

    return states
