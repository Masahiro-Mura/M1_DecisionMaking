"""
HighwayEnv-compatible surrounding vehicle driven by the paper's IDM model.

Responsibility:
    - obtain the leading vehicle from HighwayEnv's road representation
    - evaluate longitudinal acceleration with idmModel.py
    - keep the vehicle following its current main-road lane
    - expose acceleration/state information to the environment

The main paper states that surrounding vehicles on the main road are
controlled by IDM, but it does not specify an autonomous lane-change model.
Therefore, this implementation intentionally disables discretionary lane
changes and models surrounding vehicles as car-following vehicles.

Lateral lane following uses HighwayEnv's ControlledVehicle steering
controller. This is an implementation detail needed to remain on a lane and
is not an additional behavior-planning model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from highway_env.road.road import Road
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

from parameters.idmParameters import (
    DESIRED_SPEED_MAX,
    DESIRED_SPEED_MIN,
)
from surrounding_vehicle_model.idmModel import IDMModel, IDMParameters


class SurroundingVehicle(ControlledVehicle):
    """
    Main-road vehicle with paper-compatible IDM longitudinal behavior.

    Parameters are intentionally instance-specific so desired speed can be
    randomized for each surrounding vehicle, as stated in the main paper.
    """

    def __init__(
        self,
        road: Road,
        position: np.ndarray,
        heading: float = 0.0,
        speed: float = 0.0,
        target_lane_index=None,
        target_speed: float | None = None,
        route=None,
        idm_parameters: IDMParameters | None = None,
    ) -> None:
        if target_speed is None:
            target_speed = float(
                road.np_random.uniform(
                    DESIRED_SPEED_MIN,
                    DESIRED_SPEED_MAX,
                )
            )

        super().__init__(
            road=road,
            position=position,
            heading=heading,
            speed=speed,
            target_lane_index=target_lane_index,
            target_speed=target_speed,
            route=route,
        )

        self.idm_parameters = (
            idm_parameters
            if idm_parameters is not None
            else IDMModel.make_default_parameters(target_speed)
        )

        # Useful for Table-I observation construction later.
        self.longitudinal_acceleration: float = 0.0

    def act(self, action: dict | str | None = None) -> None:
        """
        Compute IDM acceleration and lane-following steering.

        External high-level lane-change actions are deliberately ignored:
        the main paper specifies IDM car-following for surrounding vehicles
        but does not provide a surrounding-vehicle lane-change policy.
        """
        if self.crashed:
            return

        # Allow HighwayEnv to update the lane graph segment at lane ends.
        self.follow_road()

        # Stay on the lane selected by road-following logic.
        self.target_lane_index = self.lane_index

        front_vehicle, _ = self.road.neighbour_vehicles(
            self,
            self.lane_index,
        )

        gap, front_speed = self._front_vehicle_state(front_vehicle)

        acceleration = IDMModel.acceleration(
            current_speed=self.speed,
            current_gap=gap,
            front_speed=front_speed,
            params=self.idm_parameters,
        )

        steering = self.steering_control(self.target_lane_index)

        self.longitudinal_acceleration = float(acceleration)

        self.action = {
            "acceleration": float(acceleration),
            "steering": float(steering),
        }

    def _front_vehicle_state(
        self,
        front_vehicle: Vehicle | None,
    ) -> tuple[float | None, float | None]:
        """
        Extract IDM gap and front speed from HighwayEnv.

        HighwayEnv's own IDM implementation uses lane_distance_to() as the
        longitudinal distance measure. We use the same road-coordinate measure
        so the model remains consistent with the simulator.
        """
        if front_vehicle is None:
            return None, None

        gap = float(self.lane_distance_to(front_vehicle))

        return gap, float(front_vehicle.speed)

    @property
    def acceleration(self) -> float:
        """Current longitudinal acceleration [m/s^2]."""
        return self.longitudinal_acceleration

    def set_desired_speed(self, desired_speed: float) -> None:
        """
        Change v0 while keeping all other IDM parameters unchanged.

        Useful for scenario randomization and future experiments.
        """
        self.idm_parameters = IDMParameters(
            desired_speed=float(desired_speed),
            jam_distance=self.idm_parameters.jam_distance,
            desired_time_gap=self.idm_parameters.desired_time_gap,
            max_acceleration=self.idm_parameters.max_acceleration,
            desired_deceleration=self.idm_parameters.desired_deceleration,
            acceleration_exponent=self.idm_parameters.acceleration_exponent,
        )
        self.target_speed = float(desired_speed)

    def state_dict(self) -> dict[str, Any]:
        """
        Return surrounding-vehicle state needed by future observation logic.

        The main paper's Table I requires surrounding positions, velocities,
        and accelerations. Position/velocity conversion into ego-relative
        quantities remains the responsibility of environment/observation.py.
        """
        velocity_vector = np.array(
            [
                self.speed * np.cos(self.heading),
                self.speed * np.sin(self.heading),
            ],
            dtype=np.float64,
        )

        return {
            "position": np.asarray(self.position, dtype=np.float64).copy(),
            "velocity": velocity_vector,
            "speed": float(self.speed),
            "heading": float(self.heading),
            "acceleration": float(self.longitudinal_acceleration),
            "desired_speed": float(self.idm_parameters.desired_speed),
            "lane_index": self.lane_index,
        }
