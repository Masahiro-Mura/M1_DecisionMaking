from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project imports
# ============================================================

from environment.mergeEnvironment import MergeEnvironment
from environment.observation import lane_change_feasibility

from cbf.roadCBF import (
    RoadCBFContext,
    left_road_barrier,
    right_road_barrier,
)
from cbf.safeController import SafeController

from mpc.kinematicBicycleModel import discrete_step

from parameters.environmentParameters import (
    DEFAULT_EGO_INITIAL_SPEED,
    LANE_WIDTH,
)


# ============================================================
# Diagnostic settings
# ============================================================

TEST_STEERING_VALUES = (
    -0.30,
    0.00,
    +0.30,
)

TEST_ACCELERATION = 0.0


# ============================================================
# Helper functions
# ============================================================

def get_ego_state(env: MergeEnvironment) -> np.ndarray:
    """
    Return ego state in the format used by MPC / CBF:

        [x, y, v, heading]
    """

    return np.asarray(
        [
            env.vehicle.position[0],
            env.vehicle.position[1],
            env.vehicle.speed,
            env.vehicle.heading,
        ],
        dtype=np.float64,
    )


def get_lane_information(
    env: MergeEnvironment,
) -> tuple[
    tuple,
    object,
    float,
    float,
    float,
]:
    """
    Obtain the closest HighwayEnv lane and local geometry.

    Returns
    -------
    lane_index
    lane
    longitudinal
    lateral_deviation
    lane_heading
    """

    lane_index = env.road.network.get_closest_lane_index(
        env.vehicle.position,
        env.vehicle.heading,
    )

    lane = env.road.network.get_lane(lane_index)

    longitudinal, lateral_deviation = (
        lane.local_coordinates(env.vehicle.position)
    )

    lane_heading = lane.heading_at(longitudinal)

    return (
        lane_index,
        lane,
        float(longitudinal),
        float(lateral_deviation),
        float(lane_heading),
    )


def make_road_context(
    ego_state: np.ndarray,
    lateral_deviation: float,
    lane_heading: float,
) -> RoadCBFContext:
    """
    Construct RoadCBFContext using the current ego state.

    NOTE
    ----
    The main paper defines (xp, yp) and d_cl as prior values.

    In this diagnostic, the current point is intentionally used
    as the local reference point so that the sign relationship
    between steering direction and h1 / h2 can be isolated.

    This is a diagnostic assumption only.
    """

    lane_slope = float(np.tan(lane_heading))

    return RoadCBFContext(
        prior_x=float(ego_state[0]),
        prior_y=float(ego_state[1]),
        prior_lateral_deviation=float(lateral_deviation),
        lane_slope=lane_slope,
        lane_width=float(LANE_WIDTH),
    )


def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# Main diagnostic
# ============================================================

def main() -> None:

    env = MergeEnvironment(
        ego_initial_speed=DEFAULT_EGO_INITIAL_SPEED,
    )

    observation, info = env.reset(seed=10_000)

    try:
        # --------------------------------------------------------
        # 1. Current ego / lane state
        # --------------------------------------------------------

        ego_state = get_ego_state(env)

        (
            lane_index,
            lane,
            longitudinal,
            lateral_deviation,
            lane_heading,
        ) = get_lane_information(env)

        left_feasible, right_feasible = (
            lane_change_feasibility(
                env.road,
                env.vehicle,
            )
        )

        road_context = make_road_context(
            ego_state=ego_state,
            lateral_deviation=lateral_deviation,
            lane_heading=lane_heading,
        )

        current_h1 = left_road_barrier(
            ego_state,
            road_context,
        )

        current_h2 = right_road_barrier(
            ego_state,
            road_context,
        )

        print_header("CURRENT ROAD / EGO STATE")

        print(f"lane_index          = {lane_index}")
        print(f"x                   = {ego_state[0]: .6f} m")
        print(f"y                   = {ego_state[1]: .6f} m")
        print(f"speed               = {ego_state[2]: .6f} m/s")
        print(f"heading             = {ego_state[3]: .6f} rad")

        print()
        print(f"lane longitudinal   = {longitudinal: .6f} m")
        print(f"lateral deviation   = {lateral_deviation: .6f} m")
        print(f"lane heading        = {lane_heading: .6f} rad")
        print(
            f"lane slope          = "
            f"{road_context.lane_slope: .6f}"
        )

        print()
        print(f"lcf                 = {left_feasible}")
        print(f"rcf                 = {right_feasible}")

        print()
        print(f"h1 current          = {current_h1: .6f}")
        print(f"h2 current          = {current_h2: .6f}")

        # --------------------------------------------------------
        # 2. Steering sign test
        # --------------------------------------------------------

        print_header("STEERING SIGN TEST")

        print(
            "The same current state is propagated by one MPC/CBF "
            "model step."
        )
        print()
        print(
            "We compare negative / zero / positive steering and "
            "observe next_y, h1_next and h2_next."
        )

        print()

        print(
            f"{'steering':>12}"
            f"{'next_x':>14}"
            f"{'next_y':>14}"
            f"{'next_heading':>16}"
            f"{'h1_next':>14}"
            f"{'h2_next':>14}"
        )

        print("-" * 84)

        for steering in TEST_STEERING_VALUES:

            control = np.asarray(
                [
                    TEST_ACCELERATION,
                    steering,
                ],
                dtype=np.float64,
            )

            next_state = discrete_step(
                state=ego_state,
                control=control,
            )

            next_h1 = left_road_barrier(
                next_state,
                road_context,
            )

            next_h2 = right_road_barrier(
                next_state,
                road_context,
            )

            print(
                f"{steering:12.4f}"
                f"{next_state[0]:14.6f}"
                f"{next_state[1]:14.6f}"
                f"{next_state[3]:16.6f}"
                f"{next_h1:14.6f}"
                f"{next_h2:14.6f}"
            )

        # --------------------------------------------------------
        # 3. Direction interpretation
        # --------------------------------------------------------

        print_header("STEERING DIRECTION CHECK")

        state_negative = discrete_step(
            state=ego_state,
            control=np.asarray(
                [TEST_ACCELERATION, -0.30],
                dtype=np.float64,
            ),
        )

        state_positive = discrete_step(
            state=ego_state,
            control=np.asarray(
                [TEST_ACCELERATION, +0.30],
                dtype=np.float64,
            ),
        )

        delta_y_negative = (
            state_negative[1] - ego_state[1]
        )

        delta_y_positive = (
            state_positive[1] - ego_state[1]
        )

        print(
            f"steering -0.30 rad -> Δy = "
            f"{delta_y_negative:+.6f} m"
        )

        print(
            f"steering +0.30 rad -> Δy = "
            f"{delta_y_positive:+.6f} m"
        )

        print()

        if delta_y_positive > delta_y_negative:
            print(
                "Result: positive steering moves the ego "
                "toward larger global y."
            )
        else:
            print(
                "Result: positive steering moves the ego "
                "toward smaller global y."
            )

        print()
        print(
            "For the present merge geometry, compare this with:"
        )
        print(
            "    ramp center ≈ y = 8 m"
        )
        print(
            "    adjacent main lane center ≈ y = 4 m"
        )

        # --------------------------------------------------------
        # 4. Actual SafeController road-only solve
        # --------------------------------------------------------

        print_header("SAFE CONTROLLER ROAD-ONLY TEST")

        safe_controller = SafeController()

        nominal_control = np.asarray(
            [
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        safe_result = safe_controller.correct(
            ego_state=ego_state,
            nominal_control=nominal_control,

            # Isolate road CBFs only.
            surrounding_vehicles=[],

            road_context=road_context,

            left_change_feasible=bool(
                left_feasible
            ),
            right_change_feasible=bool(
                right_feasible
            ),
        )

        print(
            f"nominal control     = "
            f"{nominal_control}"
        )

        print(
            f"CBF correction      = "
            f"{safe_result.correction}"
        )

        print(
            f"corrected control   = "
            f"{safe_result.corrected_control}"
        )

        print(
            f"solver success      = "
            f"{safe_result.success}"
        )

        print(
            f"intervention        = "
            f"{safe_result.intervention}"
        )

        print(
            f"minimum constraint  = "
            f"{safe_result.minimum_constraint_value:.8f}"
        )

        print(
            f"solver message      = "
            f"{safe_result.message}"
        )

        # --------------------------------------------------------
        # 5. Propagate CBF-selected control
        # --------------------------------------------------------

        next_safe_state = discrete_step(
            state=ego_state,
            control=safe_result.corrected_control,
        )

        safe_h1_next = left_road_barrier(
            next_safe_state,
            road_context,
        )

        safe_h2_next = right_road_barrier(
            next_safe_state,
            road_context,
        )

        print()
        print("State after one CBF-corrected model step:")
        print(
            f"next x              = "
            f"{next_safe_state[0]: .6f} m"
        )
        print(
            f"next y              = "
            f"{next_safe_state[1]: .6f} m"
        )
        print(
            f"next heading        = "
            f"{next_safe_state[3]: .6f} rad"
        )

        print()
        print(
            f"h1 next             = "
            f"{safe_h1_next: .6f}"
        )
        print(
            f"h2 next             = "
            f"{safe_h2_next: .6f}"
        )

        # --------------------------------------------------------
        # 6. Diagnostic guidance
        # --------------------------------------------------------

        print_header("WHAT TO CHECK")

        print(
            "1. Does lcf / rcf match the physical road geometry?"
        )

        print(
            "2. Which steering sign moves y from the ramp "
            "(~8 m) toward the main lane (~4 m)?"
        )

        print(
            "3. Does the active barrier increase for that "
            "physically correct steering direction?"
        )

        print(
            "4. Does SafeController choose the same direction?"
        )

        print()
        print(
            "If SafeController chooses the opposite direction "
            "from the main lane, then the h1/h2 sign or their "
            "left/right correspondence is inconsistent with "
            "the HighwayEnv coordinate convention."
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()