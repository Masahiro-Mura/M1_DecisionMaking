"""
Pure Intelligent Driver Model (IDM) equations used by surrounding vehicles.

Main paper:
    Eq. (32):
        a_c = a_ma * [1 - (v_c / v_0)^4 - (d* / d_c)^2]

    Eq. (33):
        d* = d_0 + T_g * v + v * Delta_v / (2 * sqrt(a_ma * b_md))

Important convention used here:
    Delta_v = v_ego - v_front

With this convention, Delta_v > 0 means the ego/following vehicle is
approaching the front vehicle, so the desired gap grows.

The main paper does not explicitly define the sign convention of Delta_v.
This is an implementation assumption chosen to match the standard IDM
formulation from Treiber et al. [46].
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from parameters.idmParameters import (
    ACCELERATION_EXPONENT,
    DESIRED_SPEED_MAX,
    DESIRED_SPEED_MIN,
    DESIRED_TIME_GAP,
    JAM_DISTANCE,
    MAX_ACCELERATION,
    MAX_DECELERATION,
    MIN_GAP_EPSILON,
)


@dataclass(frozen=True)
class IDMParameters:
    """Container for one surrounding vehicle's IDM parameters."""

    desired_speed: float
    jam_distance: float = JAM_DISTANCE
    desired_time_gap: float = DESIRED_TIME_GAP
    max_acceleration: float = MAX_ACCELERATION
    desired_deceleration: float = MAX_DECELERATION
    acceleration_exponent: float = ACCELERATION_EXPONENT

    def __post_init__(self) -> None:
        if self.desired_speed <= 0.0:
            raise ValueError("desired_speed must be positive.")
        if self.jam_distance < 0.0:
            raise ValueError("jam_distance must be non-negative.")
        if self.desired_time_gap < 0.0:
            raise ValueError("desired_time_gap must be non-negative.")
        if self.max_acceleration <= 0.0:
            raise ValueError("max_acceleration must be positive.")
        if self.desired_deceleration <= 0.0:
            raise ValueError("desired_deceleration must be positive.")
        if self.acceleration_exponent <= 0.0:
            raise ValueError("acceleration_exponent must be positive.")


class IDMModel:
    """Stateless implementation of the paper's IDM equations."""

    @staticmethod
    def desired_gap(
        current_speed: float,
        relative_speed: float,
        params: IDMParameters,
    ) -> float:
        """
        Compute desired dynamic gap d* from Eq. (33).

        Parameters
        ----------
        current_speed:
            Speed of the following vehicle [m/s].
        relative_speed:
            current_speed - front_vehicle_speed [m/s].
            Positive means the following vehicle is closing in.
        params:
            IDM parameter set.

        Returns
        -------
        float
            Desired gap d* [m].
        """
        interaction_term = (
            current_speed * relative_speed
            / (
                2.0
                * sqrt(
                    params.max_acceleration
                    * params.desired_deceleration
                )
            )
        )

        desired_gap = (
            params.jam_distance
            + params.desired_time_gap * current_speed
            + interaction_term
        )

        # Standard IDM implementations prevent a negative dynamic-gap demand.
        # The paper does not specify this safeguard explicitly.
        return max(params.jam_distance, desired_gap)

    @staticmethod
    def acceleration(
        current_speed: float,
        current_gap: float | None,
        front_speed: float | None,
        params: IDMParameters,
    ) -> float:
        """
        Compute IDM acceleration according to Eq. (32).

        If no front vehicle is present, only the free-road term is used.

        Parameters
        ----------
        current_speed:
            Following vehicle speed [m/s].
        current_gap:
            Longitudinal gap to the front vehicle [m].
            None means that there is no relevant leading vehicle.
        front_speed:
            Leading vehicle speed [m/s].
            None when no leading vehicle exists.
        params:
            IDM parameter set.

        Returns
        -------
        float
            Longitudinal acceleration command [m/s^2].
        """
        speed = max(0.0, float(current_speed))

        free_road_term = (
            speed / params.desired_speed
        ) ** params.acceleration_exponent

        interaction_term = 0.0

        if current_gap is not None and front_speed is not None:
            safe_gap = max(float(current_gap), MIN_GAP_EPSILON)
            relative_speed = speed - max(0.0, float(front_speed))

            d_star = IDMModel.desired_gap(
                current_speed=speed,
                relative_speed=relative_speed,
                params=params,
            )

            interaction_term = (d_star / safe_gap) ** 2

        return params.max_acceleration * (
            1.0
            - free_road_term
            - interaction_term
        )

    @staticmethod
    def make_default_parameters(desired_speed: float) -> IDMParameters:
        """
        Create a parameter set while validating the paper's v0 range.

        Main paper Table II:
            desired speed v0 = 10...20 m/s
        """
        if not (DESIRED_SPEED_MIN <= desired_speed <= DESIRED_SPEED_MAX):
            raise ValueError(
                "desired_speed must lie inside the main-paper Table II "
                f"range [{DESIRED_SPEED_MIN}, {DESIRED_SPEED_MAX}] m/s."
            )

        return IDMParameters(desired_speed=float(desired_speed))
