"""
Convert the normalized RL action into physical MPC references.

Main paper Eq. (1):
    A = [v_ref, theta_ref]
    v_ref     in [-1, 1]
    theta_ref in [-1, 1]

Main Paper III-D specifies a maintained minimum EV speed of 5 or 10 m/s.
MergeEnvironment passes that scenario minimum to both conversion directions.
The maximum speed and heading range remain implementation assumptions.
The default zero minimum is retained for standalone diagnostic callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parameters.environmentParameters import (
    ACTION_NORMALIZED_MAX,
    ACTION_NORMALIZED_MIN,
    REFERENCE_HEADING_MAX,
    REFERENCE_HEADING_MIN,
    REFERENCE_SPEED_MAX,
    REFERENCE_SPEED_MIN,
)


@dataclass(frozen=True)
class PhysicalReference:
    velocity: float
    heading: float


def _linear_map(
    value: float,
    source_low: float,
    source_high: float,
    target_low: float,
    target_high: float,
) -> float:
    if source_high <= source_low:
        raise ValueError("source range must be increasing")

    alpha = (value - source_low) / (source_high - source_low)
    return float(target_low + alpha * (target_high - target_low))


def action_to_reference(action: np.ndarray, minimum_speed: float = REFERENCE_SPEED_MIN) -> PhysicalReference:
    """
    Map normalized policy output [-1,1]^2 to [v_ref, theta_ref].

    Input ordering is exactly the paper's Eq. (1):
        action[0] = normalized v_ref
        action[1] = normalized theta_ref
    """
    action = np.asarray(action, dtype=np.float64)

    if action.shape != (2,):
        raise ValueError(f"action must have shape (2,), got {action.shape}")

    clipped = np.clip(
        action,
        ACTION_NORMALIZED_MIN,
        ACTION_NORMALIZED_MAX,
    )

    if not np.isfinite(minimum_speed) or not 0. <= minimum_speed < REFERENCE_SPEED_MAX:
        raise ValueError("minimum_speed must be finite and in [0, reference maximum).")

    velocity = _linear_map(
        clipped[0],
        ACTION_NORMALIZED_MIN,
        ACTION_NORMALIZED_MAX,
        minimum_speed,
        REFERENCE_SPEED_MAX,
    )

    heading = _linear_map(
        clipped[1],
        ACTION_NORMALIZED_MIN,
        ACTION_NORMALIZED_MAX,
        REFERENCE_HEADING_MIN,
        REFERENCE_HEADING_MAX,
    )

    return PhysicalReference(
        velocity=velocity,
        heading=heading,
    )


def reference_to_action(velocity: float, heading: float, minimum_speed: float = REFERENCE_SPEED_MIN) -> np.ndarray:
    """Inverse of the existing physical reference mapping, with bounds."""
    if not np.isfinite(minimum_speed) or not 0. <= minimum_speed < REFERENCE_SPEED_MAX:
        raise ValueError("minimum_speed must be finite and in [0, reference maximum).")
    action = np.array([
        _linear_map(velocity, minimum_speed, REFERENCE_SPEED_MAX,
                    ACTION_NORMALIZED_MIN, ACTION_NORMALIZED_MAX),
        _linear_map(heading, REFERENCE_HEADING_MIN, REFERENCE_HEADING_MAX,
                    ACTION_NORMALIZED_MIN, ACTION_NORMALIZED_MAX),
    ])
    return np.clip(action, ACTION_NORMALIZED_MIN, ACTION_NORMALIZED_MAX).astype(np.float32)
