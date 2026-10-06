"""
Episode termination logic.

Main paper Section III-D:
    - maximum number of steps / time limit
    - collision with surrounding vehicles
    - deviation from road boundaries

Main paper Section IV-A/B:
    - success means accomplishing the merging task without collision
    - testing drives the ego vehicle from the initial position to a
      designated target location
    - merge point is the x-coordinate at which the ego first enters and
      becomes part of the main-road traffic flow

Important distinction in this implementation
--------------------------------------------
"Merge point" and "episode success" are intentionally different concepts.

A vehicle may first enter the main-road flow inside the merge section, but
the episode is not declared successful until it has completed the merge and
reached the downstream side of the merge section.

Gymnasium convention:
    terminated = success, collision, or road departure
    truncated  = time limit
"""

from __future__ import annotations

from dataclasses import dataclass

from highway_env.road.road import Road
from highway_env.vehicle.kinematics import Vehicle

from parameters.environmentParameters import (
    MAIN_ROAD_LANE_IDS,
    MAX_STEPS_BY_MIN_SPEED,
    SUCCESS_MAX_LATERAL_DEVIATION,
    SUCCESS_MIN_X,
    SUCCESS_STABILITY_DURATION,
    SUCCESS_TARGET_X,
    SIMULATION_DT,
    TERMINATE_ON_COLLISION,
    TERMINATE_ON_ROAD_DEPARTURE,
    TERMINATE_ON_SUCCESSFUL_MERGE,
)


@dataclass(frozen=True)
class TerminationResult:
    terminated: bool
    truncated: bool
    success: bool
    collision: bool
    road_departure: bool
    time_limit: bool
    reason: str | None
    control_failure: bool = False

    @property
    def failure(self) -> bool:
        return bool(self.collision or self.road_departure or self.time_limit or self.control_failure)


def check_collision(ego_vehicle: Vehicle) -> bool:
    """Return True when HighwayEnv reports a collision."""
    return bool(getattr(ego_vehicle, "crashed", False))


def check_road_departure(
    road: Road,
    ego_vehicle: Vehicle,
) -> bool:
    """
    Return True when the ego center is not on any scenario lane.

    The lane ``forbidden`` flag is ignored because a forbidden lane may still
    represent physically drivable ramp geometry.
    """
    for lane in road.network.lanes_list():
        if lane.on_lane(ego_vehicle.position, margin=0.0):
            return False

    return True


def _closest_lane_index(
    road: Road,
    ego_vehicle: Vehicle,
):
    """
    Return the HighwayEnv lane geometrically closest to the ego state.

    Using the closest lane is more robust during a lane transition than relying
    only on ``ego_vehicle.lane_index``, which can lag the geometric position.
    """
    return road.network.get_closest_lane_index(
        ego_vehicle.position,
        ego_vehicle.heading,
    )


def _lateral_deviation_from_lane(
    road: Road,
    ego_vehicle: Vehicle,
    lane_index,
) -> float:
    """Signed lateral distance from the selected lane centerline."""
    lane = road.network.get_lane(lane_index)
    _, lateral = lane.local_coordinates(ego_vehicle.position)
    return float(lateral)


def check_merge_completed(
    road: Road,
    ego_vehicle: Vehicle,
) -> bool:
    """
    Revised merge-completion criterion.

    Success requires all of the following:
        1. ego has reached/passed SUCCESS_MIN_X
           (= merge-section end in the revised parameters)
        2. the closest lane is a main-road lane
        3. ego is sufficiently close to that lane centerline
        4. ego has not collided

    The 1 m lateral tolerance is an explicit implementation assumption because
    the main paper does not provide a numerical completion threshold.
    """
    if check_collision(ego_vehicle):
        return False

    if float(ego_vehicle.position[0]) < SUCCESS_MIN_X:
        return False

    lane_index = _closest_lane_index(
        road=road,
        ego_vehicle=ego_vehicle,
    )
    lane_id = int(lane_index[2])

    if lane_id not in MAIN_ROAD_LANE_IDS:
        return False

    lateral_deviation = _lateral_deviation_from_lane(
        road=road,
        ego_vehicle=ego_vehicle,
        lane_index=lane_index,
    )

    return bool(
        abs(lateral_deviation) <= SUCCESS_MAX_LATERAL_DEVIATION
    )


def check_success(road: Road, ego_vehicle: Vehicle, stable_main_lane_steps: int = 0) -> bool:
    """IV-A/B: completed merge, sustained main-lane driving, target reached.

    IMPLEMENTATION ASSUMPTION: numerical duration/target/tolerance live in
    environmentParameters; the main paper does not give these thresholds.
    """
    return bool(check_merge_completed(road, ego_vehicle)
                and stable_main_lane_steps * SIMULATION_DT >= SUCCESS_STABILITY_DURATION
                and float(ego_vehicle.position[0]) >= SUCCESS_TARGET_X)


def check_time_limit(
    step_count: int,
    ego_initial_speed: float,
) -> bool:
    """Apply the 40 s / 20 s episode limits reported by the main paper."""
    if ego_initial_speed not in MAX_STEPS_BY_MIN_SPEED:
        raise ValueError(
            f"No episode limit configured for initial speed "
            f"{ego_initial_speed} m/s."
        )

    return bool(
        step_count >= MAX_STEPS_BY_MIN_SPEED[ego_initial_speed]
    )


def evaluate_termination(
    road: Road,
    ego_vehicle: Vehicle,
    step_count: int,
    ego_initial_speed: float,
    stable_main_lane_steps: int = 0,
) -> TerminationResult:
    """Evaluate all episode-ending conditions."""
    collision = (
        check_collision(ego_vehicle)
        if TERMINATE_ON_COLLISION
        else False
    )

    road_departure = (
        check_road_departure(road, ego_vehicle)
        if TERMINATE_ON_ROAD_DEPARTURE
        else False
    )

    success = (
        check_success(road, ego_vehicle, stable_main_lane_steps)
        if TERMINATE_ON_SUCCESSFUL_MERGE
        else False
    )

    time_limit = check_time_limit(
        step_count=step_count,
        ego_initial_speed=ego_initial_speed,
    )

    # Failure takes precedence over success in an ambiguous frame.
    if collision:
        return TerminationResult(
            terminated=True,
            truncated=False,
            success=False,
            collision=True,
            road_departure=road_departure,
            time_limit=False,
            reason="collision",
        )

    if road_departure:
        return TerminationResult(
            terminated=True,
            truncated=False,
            success=False,
            collision=False,
            road_departure=True,
            time_limit=False,
            reason="road_departure",
        )

    if success:
        return TerminationResult(
            terminated=True,
            truncated=False,
            success=True,
            collision=False,
            road_departure=False,
            time_limit=False,
            reason="success",
        )

    if time_limit:
        return TerminationResult(
            terminated=False,
            truncated=True,
            success=False,
            collision=False,
            road_departure=False,
            time_limit=True,
            reason="time_limit",
        )

    return TerminationResult(
        terminated=False,
        truncated=False,
        success=False,
        collision=False,
        road_departure=False,
        time_limit=False,
        reason=None,
    )
