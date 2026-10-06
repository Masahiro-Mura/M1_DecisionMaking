"""
Reward function from Main Paper Eq. (2)-(7).

    r = r_ld + r_merge + r_steer + r_speed + r_terminal

No extra reward terms are introduced in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

from parameters.environmentParameters import (
    LANE_CENTER_REWARD_COEFFICIENT,
    MERGE_DISTANCE_SCALE,
    MERGE_EXPONENT,
    MERGE_MISSING_VEHICLE_DISTANCE,
    REWARD_MAX_SPEED,
    TERMINAL_FAILURE_REWARD,
    TERMINAL_SUCCESS_REWARD,
)


@dataclass(frozen=True)
class RewardBreakdown:
    lane_center: float
    merge: float
    steering: float
    speed: float
    terminal: float

    @property
    def total(self) -> float:
        return float(
            self.lane_center
            + self.merge
            + self.steering
            + self.speed
            + self.terminal
        )


def lane_center_reward(lateral_deviation: float) -> float:
    """Eq. (3): r_ld = 1 / (1 + 4*d_c^2)."""
    d_c = float(lateral_deviation)
    return float(
        1.0
        / (
            1.0
            + LANE_CENTER_REWARD_COEFFICIENT * d_c**2
        )
    )


def merge_reward(
    distance_front: float | None,
    distance_rear: float | None,
) -> float:
    """
    Eq. (4).

    r_merge =
        1 - | exp(-(dP/40)^0.5) - exp(-(dF/40)^0.5) |

    The paper does not define the missing-front/rear case. For v1, missing
    distances are replaced by MERGE_MISSING_VEHICLE_DISTANCE from parameters.
    """
    d_p = (
        MERGE_MISSING_VEHICLE_DISTANCE
        if distance_front is None
        else max(0.0, float(distance_front))
    )
    d_f = (
        MERGE_MISSING_VEHICLE_DISTANCE
        if distance_rear is None
        else max(0.0, float(distance_rear))
    )

    front_term = exp(
        -((d_p / MERGE_DISTANCE_SCALE) ** MERGE_EXPONENT)
    )
    rear_term = exp(
        -((d_f / MERGE_DISTANCE_SCALE) ** MERGE_EXPONENT)
    )

    return float(1.0 - abs(front_term - rear_term))


def steering_reward(
    current_reference_heading: float,
    previous_reference_heading: float | None,
) -> float:
    """
    Eq. (5):
        r_steer = -|theta_ref(i+1) - theta_ref(i)|

    At the first step no previous reference exists, so the v1 implementation
    assigns zero steering-change penalty.
    """
    if previous_reference_heading is None:
        return 0.0

    return float(
        -abs(
            float(current_reference_heading)
            - float(previous_reference_heading)
        )
    )


def speed_reward(
    current_speed: float,
    maximum_speed: float = REWARD_MAX_SPEED,
) -> float:
    """Eq. (6): r_speed = v_c / v_max."""
    if maximum_speed <= 0.0:
        raise ValueError("maximum_speed must be positive")

    return float(float(current_speed) / maximum_speed)


def terminal_reward(
    success: bool,
    failure: bool,
) -> float:
    """
    Eq. (7):
        +1 success
        -1 fail
         0 otherwise
    """
    if success:
        return float(TERMINAL_SUCCESS_REWARD)
    if failure:
        return float(TERMINAL_FAILURE_REWARD)
    return 0.0


def compute_reward(
    lateral_deviation: float,
    distance_front: float | None,
    distance_rear: float | None,
    current_reference_heading: float,
    previous_reference_heading: float | None,
    current_speed: float,
    success: bool,
    failure: bool,
) -> RewardBreakdown:
    """Compute every paper reward term and preserve the decomposition."""
    return RewardBreakdown(
        lane_center=lane_center_reward(lateral_deviation),
        merge=merge_reward(distance_front, distance_rear),
        steering=steering_reward(
            current_reference_heading,
            previous_reference_heading,
        ),
        speed=speed_reward(current_speed),
        terminal=terminal_reward(success, failure),
    )
