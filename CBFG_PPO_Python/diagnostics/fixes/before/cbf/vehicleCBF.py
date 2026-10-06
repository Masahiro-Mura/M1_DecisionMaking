"""
Vehicle-to-vehicle control barrier function.

Main paper:
    Eq. (21): ellipse boundary distance
    Eq. (22): vehicle clearance barrier h0

        h0(X_k) = ||x_ev - x_sv||_2 - l_ev - l_sv

The surrounding-vehicle one-step prediction required by the safe controller
is not specified in the paper. This module therefore permits the caller to
provide an explicit predicted next state. If none is supplied, a
constant-heading, constant-acceleration one-step prediction is used as an
implementation assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cbf.ellipseGeometry import pair_boundary_distances
from parameters.cbfParameters import (
    EGO_ELLIPSE_MAJOR_SEMI_AXIS,
    EGO_ELLIPSE_MINOR_SEMI_AXIS,
    EXTRA_COLLISION_SAFETY_MARGIN,
    SURROUNDING_ELLIPSE_MAJOR_SEMI_AXIS,
    SURROUNDING_ELLIPSE_MINOR_SEMI_AXIS,
)
from parameters.mpcParameters import MPC_DT


@dataclass
class SurroundingCBFState:
    """Current and optional predicted state of one surrounding vehicle."""

    state: np.ndarray
    acceleration: float = 0.0
    predicted_next_state: np.ndarray | None = None


def predict_surrounding_state(
    state: np.ndarray,
    acceleration: float = 0.0,
    dt: float = MPC_DT,
) -> np.ndarray:
    """
    One-step surrounding-vehicle prediction.

    State format:
        [x, y, v, theta]

    Assumption:
        heading is constant over one step; longitudinal acceleration is
        constant over one step.
    """
    x = np.asarray(state, dtype=np.float64)

    if x.shape != (4,):
        raise ValueError(f"state must have shape (4,), got {x.shape}")

    px, py, speed, heading = x
    next_speed = max(0.0, speed + float(acceleration) * float(dt))

    return np.array(
        [
            px + speed * np.cos(heading) * dt,
            py + speed * np.sin(heading) * dt,
            next_speed,
            heading,
        ],
        dtype=np.float64,
    )


def vehicle_barrier(
    ego_state: np.ndarray,
    surrounding_state: np.ndarray,
) -> float:
    """
    Compute the main-paper vehicle CBF h0 from Eq. (22).

    Positive:
        separated ellipses.
    Zero:
        ellipse boundaries touch.
    Negative:
        ellipse overlap / unsafe state.
    """
    ego = np.asarray(ego_state, dtype=np.float64)
    surrounding = np.asarray(surrounding_state, dtype=np.float64)

    if ego.shape != (4,) or surrounding.shape != (4,):
        raise ValueError("ego_state and surrounding_state must have shape (4,)")

    ego_center = ego[:2]
    surrounding_center = surrounding[:2]

    l_ego, l_surrounding = pair_boundary_distances(
        ego_center=ego_center,
        ego_heading=float(ego[3]),
        ego_semi_major=EGO_ELLIPSE_MAJOR_SEMI_AXIS,
        ego_semi_minor=EGO_ELLIPSE_MINOR_SEMI_AXIS,
        surrounding_center=surrounding_center,
        surrounding_heading=float(surrounding[3]),
        surrounding_semi_major=SURROUNDING_ELLIPSE_MAJOR_SEMI_AXIS,
        surrounding_semi_minor=SURROUNDING_ELLIPSE_MINOR_SEMI_AXIS,
    )

    center_distance = float(
        np.linalg.norm(ego_center - surrounding_center, ord=2)
    )

    return float(
        center_distance
        - l_ego
        - l_surrounding
        - EXTRA_COLLISION_SAFETY_MARGIN
    )
