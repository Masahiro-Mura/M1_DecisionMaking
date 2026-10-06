"""
Initial vehicle placement and randomization for the merge scenario.

Responsibility:
    - create the Road using mergeRoad.py
    - spawn the ego vehicle on the parallel on-ramp section
    - randomly spawn surrounding main-road vehicles
    - randomize surrounding-vehicle initial/desired speeds

Vehicle behavior itself does NOT belong here. A surrounding vehicle class can
be injected, so this file can later use the project's dedicated
surrounding_vehicle_model implementation without changing scenario logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import numpy as np

from highway_env.road.road import Road
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle

from parameters.environmentParameters import (
    DEFAULT_EGO_INITIAL_SPEED,
    EGO_INITIAL_RAMP_X,
    MAX_SPAWN_ATTEMPTS_PER_VEHICLE,
    SURROUNDING_INITIAL_SPEED_MAX,
    SURROUNDING_INITIAL_SPEED_MIN,
    SURROUNDING_SPAWN_X_MAX,
    SURROUNDING_SPAWN_X_MIN,
    SURROUNDING_VEHICLE_COUNT_MAX,
    SURROUNDING_VEHICLE_COUNT_MIN,
    TRAINING_SPEED_CASES,
)
from parameters.idmParameters import (
    ACCELERATION_EXPONENT,
    DESIRED_SPEED_MAX,
    DESIRED_SPEED_MIN,
    DESIRED_TIME_GAP,
    JAM_DISTANCE,
    MAX_ACCELERATION as IDM_MAX_ACCELERATION,
    MAX_DECELERATION as IDM_MAX_DECELERATION,
)
from parameters.vehicleParameters import VEHICLE_LENGTH

from scenario.mergeRoad import MergeRoad


@dataclass
class MergeScenarioState:
    """Objects produced when a scenario is initialized."""

    road: Road
    ego_vehicle: Vehicle
    surrounding_vehicles: list[Vehicle]


class MergeScenario:
    """Create one randomized realization of the on-ramp merging world."""

    def __init__(
        self,
        seed: int | None = None,
        ego_initial_speed: float = DEFAULT_EGO_INITIAL_SPEED,
        surrounding_vehicle_cls: Type[Vehicle] = IDMVehicle,
        record_history: bool = False,
    ) -> None:
        if ego_initial_speed not in TRAINING_SPEED_CASES:
            raise ValueError(
                f"ego_initial_speed must be one of {TRAINING_SPEED_CASES}, "
                f"got {ego_initial_speed}."
            )

        self.rng = np.random.default_rng(seed)
        self.ego_initial_speed = float(ego_initial_speed)
        self.surrounding_vehicle_cls = surrounding_vehicle_cls
        self.record_history = record_history

    def create(self) -> MergeScenarioState:
        """Build a fresh road and populate it with ego + surrounding vehicles."""
        road = MergeRoad.build_road(
            np_random=self.rng,
            record_history=self.record_history,
        )

        ego = self._create_ego_vehicle(road)
        surrounding = self._create_surrounding_vehicles(road)

        road.vehicles = [ego, *surrounding]

        return MergeScenarioState(
            road=road,
            ego_vehicle=ego,
            surrounding_vehicles=surrounding,
        )

    def _create_ego_vehicle(self, road: Road) -> Vehicle:
        """
        Spawn ego at the beginning of the paper's x=230...310 m merge section.

        On the standard HighwayEnv merge road, ("b", "c", 2) is the parallel
        ramp lane. Its center is y=8 m, corresponding to the paper's ramp
        region y=6...10 m.
        """
        ramp_lane_index = ("b", "c", MergeRoad.RAMP_LANE_INDEX)
        ramp_lane = road.network.get_lane(ramp_lane_index)

        # EGO_INITIAL_RAMP_X is 230 m, the global start of this lane.
        lane_start_x = EGO_INITIAL_RAMP_X
        local_longitudinal = EGO_INITIAL_RAMP_X - lane_start_x

        position = ramp_lane.position(local_longitudinal, 0.0)
        heading = ramp_lane.heading_at(local_longitudinal)

        ego = Vehicle(
            road=road,
            position=position,
            heading=heading,
            speed=self.ego_initial_speed,
        )

        # HighwayEnv Vehicle initializes the low-level action at zero;
        # this is consistent with the paper's condensed case a_ego(0)=0.
        return ego

    def _create_surrounding_vehicles(self, road: Road) -> list[Vehicle]:
        """
        Randomize vehicle count, main-road positions and initial speeds.

        Main paper support:
        - number and speed are randomly generated during training;
        - in testing, surrounding positions are uniformly distributed;
        - condensed case uses 10 +/- 2 m/s initial surrounding speed.

        The count range is an explicit implementation assumption stored in
        environmentParameters.py.
        """
        count = int(
            self.rng.integers(
                SURROUNDING_VEHICLE_COUNT_MIN,
                SURROUNDING_VEHICLE_COUNT_MAX + 1,
            )
        )

        vehicles: list[Vehicle] = []

        for _ in range(count):
            vehicle = self._sample_collision_free_surrounding_vehicle(
                road=road,
                existing=vehicles,
            )
            vehicles.append(vehicle)

        return vehicles

    def _sample_collision_free_surrounding_vehicle(
        self,
        road: Road,
        existing: list[Vehicle],
    ) -> Vehicle:
        """Sample one main-road vehicle while avoiding overlapping spawns."""
        minimum_spacing = VEHICLE_LENGTH + JAM_DISTANCE

        for _ in range(MAX_SPAWN_ATTEMPTS_PER_VEHICLE):
            lane_id = int(self.rng.integers(0, MergeRoad.MAIN_LANE_COUNT))
            x_position = float(
                self.rng.uniform(
                    SURROUNDING_SPAWN_X_MIN,
                    SURROUNDING_SPAWN_X_MAX,
                )
            )

            lane_index = MergeRoad.main_lane_index_for_x(
                x_position=x_position,
                lane_id=lane_id,
            )
            lane = road.network.get_lane(lane_index)
            longitudinal = MergeRoad.lane_longitudinal_from_global_x(
                x_position=x_position,
                lane_index=lane_index,
            )

            position = lane.position(longitudinal, 0.0)

            if self._too_close_to_existing(
                position=position,
                lane_id=lane_id,
                existing=existing,
                minimum_spacing=minimum_spacing,
            ):
                continue

            speed = float(
                self.rng.uniform(
                    SURROUNDING_INITIAL_SPEED_MIN,
                    SURROUNDING_INITIAL_SPEED_MAX,
                )
            )
            desired_speed = float(
                self.rng.uniform(
                    DESIRED_SPEED_MIN,
                    DESIRED_SPEED_MAX,
                )
            )

            vehicle = self.surrounding_vehicle_cls(
                road=road,
                position=position,
                heading=lane.heading_at(longitudinal),
                speed=speed,
                target_lane_index=lane_index,
                target_speed=desired_speed,
            )

            self._configure_idm_parameters_if_supported(vehicle)
            return vehicle

        raise RuntimeError(
            "Could not place a surrounding vehicle without overlap. "
            "Reduce traffic count or widen the spawn interval."
        )

    @staticmethod
    def _too_close_to_existing(
        position: np.ndarray,
        lane_id: int,
        existing: list[Vehicle],
        minimum_spacing: float,
    ) -> bool:
        """
        Check longitudinal spawn spacing on the same main-road lane.

        Main-road lanes are horizontal, so global x difference is the
        longitudinal spacing.
        """
        for vehicle in existing:
            if vehicle.lane_index[2] != lane_id:
                continue

            if abs(float(vehicle.position[0] - position[0])) < minimum_spacing:
                return True

        return False

    @staticmethod
    def _configure_idm_parameters_if_supported(vehicle: Vehicle) -> None:
        """
        Apply the main-paper IDM parameters when the injected class supports them.

        This is only initialization. The actual IDM equations belong in
        surrounding_vehicle_model/, which can later replace HighwayEnv's
        IDMVehicle through constructor injection.
        """
        if hasattr(vehicle, "DISTANCE_WANTED"):
            vehicle.DISTANCE_WANTED = JAM_DISTANCE
        if hasattr(vehicle, "TIME_WANTED"):
            vehicle.TIME_WANTED = DESIRED_TIME_GAP
        if hasattr(vehicle, "COMFORT_ACC_MAX"):
            vehicle.COMFORT_ACC_MAX = IDM_MAX_ACCELERATION
        if hasattr(vehicle, "COMFORT_ACC_MIN"):
            vehicle.COMFORT_ACC_MIN = -IDM_MAX_DECELERATION
        if hasattr(vehicle, "DELTA"):
            vehicle.DELTA = ACCELERATION_EXPONENT
