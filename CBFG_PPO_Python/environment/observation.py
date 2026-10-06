"""
Build the 27-dimensional observation vector from Main Paper Table I.

Table I:
    ego velocity                        R^2
    ego position                        R^2
    lateral deviation from lane center  R
    left lane-change feasibility        R
    right lane-change feasibility       R
    velocities of four surrounding cars R^8
    relative positions of four cars     R^8
    accelerations of four cars          R^4
                                      ----
                                      R^27

The main paper states that all dynamic elements are normalized but does not
report the exact physical normalization ranges. Those ranges are therefore
kept in parameters/environmentParameters.py as implementation assumptions.

Surrounding-vehicle ordering is also not specified in the paper. This v1
implementation uses the four vehicles nearest to ego by Euclidean distance,
ordered nearest-first, and zero-fills missing slots.
"""

from __future__ import annotations

import numpy as np

from highway_env.road.road import Road
from highway_env.vehicle.kinematics import Vehicle

from parameters.environmentParameters import (
    LANE_WIDTH,
    NUM_SURROUNDING_VEHICLES,
    OBSERVATION_DIM,
    OBS_MISSING_VEHICLE_FILL,
    OBS_POSITION_X_MAX,
    OBS_POSITION_X_MIN,
    OBS_POSITION_Y_MAX,
    OBS_POSITION_Y_MIN,
    OBS_RELATIVE_X_ABS_MAX,
    OBS_RELATIVE_Y_ABS_MAX,
    OBS_SPEED_ABS_MAX,
)
from parameters.idmParameters import (
    MAX_ACCELERATION as IDM_MAX_ACCELERATION,
    MAX_DECELERATION as IDM_MAX_DECELERATION,
)


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _scale_to_minus_one_one(value: float, low: float, high: float) -> float:
    """Linearly map [low, high] to [-1, 1], with clipping."""
    if high <= low:
        raise ValueError("normalization upper bound must exceed lower bound")

    clipped = _clip(value, low, high)
    return float(2.0 * (clipped - low) / (high - low) - 1.0)


def _scale_symmetric(value: float, abs_max: float) -> float:
    """Normalize a symmetric physical range [-abs_max, abs_max]."""
    if abs_max <= 0.0:
        raise ValueError("abs_max must be positive")
    return float(np.clip(value / abs_max, -1.0, 1.0))


def vehicle_velocity_vector(vehicle: Vehicle) -> np.ndarray:
    """Return global [vx, vy] from speed and heading."""
    return np.array(
        [
            vehicle.speed * np.cos(vehicle.heading),
            vehicle.speed * np.sin(vehicle.heading),
        ],
        dtype=np.float64,
    )


def lateral_deviation(road: Road, vehicle: Vehicle) -> float:
    """
    Signed lateral deviation from the centerline of the vehicle's current lane.

    HighwayEnv local lane coordinates provide exactly this lateral coordinate.
    """
    lane = road.network.get_lane(vehicle.lane_index)
    _, lateral = lane.local_coordinates(vehicle.position)
    return float(lateral)


def lane_change_feasibility(
    road: Road,
    vehicle: Vehicle,
) -> tuple[float, float]:
    """
    Return (left_feasible, right_feasible) as 0/1 values.

    Main paper:
        lcf = feasibility of changing to the left lane
        rcf = feasibility of changing to the right lane

    HighwayEnv convention:
        left lane  -> smaller lane ID
        right lane -> larger lane ID

    A lane change is considered feasible when:
        1. the adjacent lane exists, and
        2. HighwayEnv judges that lane reachable from the current position.

    `is_reachable_from()` also rejects forbidden lanes.
    """

    current_lane_index = vehicle.lane_index
    current_lane_id = int(current_lane_index[2])

    left_feasible = 0.0
    right_feasible = 0.0

    # HighwayEnv itself uses side_lanes() when determining
    # whether LANE_LEFT / LANE_RIGHT actions are available.
    side_lane_indices = road.network.side_lanes(
        current_lane_index
    )

    for target_lane_index in side_lane_indices:
        target_lane = road.network.get_lane(
            target_lane_index
        )

        # The lane must actually be reachable from the
        # vehicle's current world position.
        if not target_lane.is_reachable_from(
            vehicle.position
        ):
            continue

        target_lane_id = int(target_lane_index[2])

        # HighwayEnv convention:
        # smaller lane ID = LEFT
        if target_lane_id < current_lane_id:
            left_feasible = 1.0

        # larger lane ID = RIGHT
        elif target_lane_id > current_lane_id:
            right_feasible = 1.0

    return left_feasible, right_feasible


def select_surrounding_vehicles(
    ego_vehicle: Vehicle,
    surrounding_vehicles: list[Vehicle],
    count: int = NUM_SURROUNDING_VEHICLES,
) -> list[Vehicle]:
    """
    Select the four nearest surrounding vehicles.

    The paper fixes four surrounding vehicles in the observation but does not
    specify their ordering. Nearest-first is the v1 implementation assumption.
    """
    ordered = sorted(
        surrounding_vehicles,
        key=lambda vehicle: float(
            np.linalg.norm(
                np.asarray(vehicle.position) - np.asarray(ego_vehicle.position)
            )
        ),
    )
    return ordered[:count]


def _surrounding_acceleration(vehicle: Vehicle) -> float:
    """Read longitudinal acceleration from our surrounding-vehicle model."""
    if hasattr(vehicle, "acceleration"):
        value = getattr(vehicle, "acceleration")
        if not callable(value):
            return float(value)

    if hasattr(vehicle, "longitudinal_acceleration"):
        return float(vehicle.longitudinal_acceleration)

    action = getattr(vehicle, "action", {})
    return float(action.get("acceleration", 0.0))


def build_observation(
    road: Road,
    ego_vehicle: Vehicle,
    surrounding_vehicles: list[Vehicle],
) -> np.ndarray:
    """Return the normalized observation with shape (27,)."""
    ego_velocity = vehicle_velocity_vector(ego_vehicle)
    ego_position = np.asarray(ego_vehicle.position, dtype=np.float64)

    d_cl = lateral_deviation(road, ego_vehicle)
    left_feasible, right_feasible = lane_change_feasibility(road, ego_vehicle)

    selected = select_surrounding_vehicles(
        ego_vehicle=ego_vehicle,
        surrounding_vehicles=surrounding_vehicles,
    )

    # ---- ego dynamic state -------------------------------------------------
    values: list[float] = [
        _scale_symmetric(ego_velocity[0], OBS_SPEED_ABS_MAX),
        _scale_symmetric(ego_velocity[1], OBS_SPEED_ABS_MAX),
        _scale_to_minus_one_one(
            ego_position[0], OBS_POSITION_X_MIN, OBS_POSITION_X_MAX
        ),
        _scale_to_minus_one_one(
            ego_position[1], OBS_POSITION_Y_MIN, OBS_POSITION_Y_MAX
        ),
        _scale_symmetric(d_cl, LANE_WIDTH / 2.0),
        left_feasible,
        right_feasible,
    ]

    # ---- four surrounding velocities: R^8 ---------------------------------
    velocity_block: list[float] = []
    relative_position_block: list[float] = []
    acceleration_block: list[float] = []

    for vehicle in selected:
        velocity = vehicle_velocity_vector(vehicle)
        relative_position = (
            np.asarray(vehicle.position, dtype=np.float64) - ego_position
        )

        velocity_block.extend(
            [
                _scale_symmetric(velocity[0], OBS_SPEED_ABS_MAX),
                _scale_symmetric(velocity[1], OBS_SPEED_ABS_MAX),
            ]
        )
        relative_position_block.extend(
            [
                _scale_symmetric(
                    relative_position[0], OBS_RELATIVE_X_ABS_MAX
                ),
                _scale_symmetric(
                    relative_position[1], OBS_RELATIVE_Y_ABS_MAX
                ),
            ]
        )
        acceleration_block.append(
            _scale_to_minus_one_one(
                _surrounding_acceleration(vehicle),
                -IDM_MAX_DECELERATION,
                IDM_MAX_ACCELERATION,
            )
        )

    missing = NUM_SURROUNDING_VEHICLES - len(selected)

    if missing > 0:
        velocity_block.extend(
            [OBS_MISSING_VEHICLE_FILL] * (2 * missing)
        )
        relative_position_block.extend(
            [OBS_MISSING_VEHICLE_FILL] * (2 * missing)
        )
        acceleration_block.extend(
            [OBS_MISSING_VEHICLE_FILL] * missing
        )

    values.extend(velocity_block)
    values.extend(relative_position_block)
    values.extend(acceleration_block)

    observation = np.asarray(values, dtype=np.float32)

    if observation.shape != (OBSERVATION_DIM,):
        raise RuntimeError(
            f"Observation must have shape ({OBSERVATION_DIM},), "
            f"got {observation.shape}"
        )

    return np.clip(observation, -1.0, 1.0)
