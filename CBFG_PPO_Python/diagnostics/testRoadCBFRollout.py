"""
Diagnose road-CBF behavior over one complete rollout.

This script does NOT modify the environment, controller, or learned policy.

It runs one deterministic evaluation episode and logs:

    - ego state
    - closest HighwayEnv lane
    - signed lateral deviation
    - lcf / rcf
    - road CBF h1 / h2
    - RL reference action
    - MPC nominal control
    - CBF correction
    - final safe control
    - lane-index changes
    - termination reason

Purpose
-------
The goal is to identify exactly when:

    1. the ego changes its closest lane,
    2. lcf / rcf change,
    3. h1 / h2 approach or cross zero,
    4. the road CBF begins to intervene.

No production code is changed.
"""

from __future__ import annotations

from pathlib import Path
import sys


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Standard imports
# ============================================================

import argparse
import csv

import numpy as np


# ============================================================
# Project imports
# ============================================================

from cbf.roadCBF import (
    RoadCBFContext,
    left_road_barrier,
    right_road_barrier,
)

from environment.mergeEnvironment import MergeEnvironment
from environment.observation import lane_change_feasibility

from parameters.cbfParameters import ROAD_LANE_WIDTH
from parameters.environmentParameters import SIMULATION_DT

from rl.ppoAgent import PPOAgent


# ============================================================
# Paths
# ============================================================

DIAGNOSTICS_DIR = PROJECT_ROOT / "diagnostics"

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "bestPolicy.pt"
)

CSV_PATH = (
    DIAGNOSTICS_DIR
    / "roadCBFRollout.csv"
)


# ============================================================
# Diagnostic settings
# ============================================================

DEFAULT_SEED = 10_000
DEFAULT_EGO_SPEED = 5.0

# Print every simulation step.
PRINT_EVERY_STEP = True

# Treat a barrier smaller than this as "near boundary".
BARRIER_WARNING_THRESHOLD = 0.5


# ============================================================
# Helper functions
# ============================================================

def get_ego_state(
    env: MergeEnvironment,
) -> np.ndarray:
    """
    Return ego state as

        [x, y, v, heading].
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


def get_lane_diagnostics(
    env: MergeEnvironment,
) -> dict:
    """
    Obtain closest HighwayEnv lane and local lane coordinates.

    The road CBF itself is evaluated using exactly the current
    roadCBF.py implementation.

    Use the environment's actual physical road envelope, including the
    one-step corridor intersection at road-section changes.
    """

    ego_state = get_ego_state(env)

    lane_index = env.road.network.get_closest_lane_index(
        env.vehicle.position,
        env.vehicle.heading,
    )

    lane = env.road.network.get_lane(
        lane_index
    )

    (
        longitudinal,
        lateral_deviation,
    ) = lane.local_coordinates(
        env.vehicle.position
    )

    lane_heading = float(
        lane.heading_at(longitudinal)
    )

    lane_slope = float(
        np.tan(lane_heading)
    )

    left_feasible, right_feasible = (
        lane_change_feasibility(
            env.road,
            env.vehicle,
        )
    )

    road_context = env._road_cbf_context()

    h1 = left_road_barrier(
        ego_state,
        road_context,
    )

    h2 = right_road_barrier(
        ego_state,
        road_context,
    )

    return {
        "ego_state": ego_state,
        "lane_index": lane_index,
        "lane": lane,
        "longitudinal": float(longitudinal),
        "lateral_deviation": float(
            lateral_deviation
        ),
        "lane_heading": lane_heading,
        "lane_slope": lane_slope,
        "lcf": float(left_feasible),
        "rcf": float(right_feasible),
        "h1": float(h1),
        "h2": float(h2),
        "road_context": road_context,
    }


def lane_index_string(
    lane_index,
) -> str:
    """Convert HighwayEnv lane index to readable text."""

    if lane_index is None:
        return "None"

    try:
        return (
            f"{lane_index[0]}"
            f"->{lane_index[1]}"
            f":{lane_index[2]}"
        )
    except Exception:
        return str(lane_index)


def barrier_status(
    h1: float,
    h2: float,
) -> str:
    """
    Give a compact interpretation of road-CBF values.
    """

    if h1 < 0.0 or h2 < 0.0:
        return "OUTSIDE"

    if (
        h1 < BARRIER_WARNING_THRESHOLD
        or h2 < BARRIER_WARNING_THRESHOLD
    ):
        return "NEAR"

    return "SAFE"


def print_table_header() -> None:
    """Print rollout table header."""

    print()
    print("=" * 150)
    print("ROAD CBF ROLLOUT")
    print("=" * 150)

    print(
        f"{'t':>5} "
        f"{'x':>8} "
        f"{'y':>8} "
        f"{'lane':>13} "
        f"{'d_lat':>8} "
        f"{'lcf':>4} "
        f"{'rcf':>4} "
        f"{'h1':>8} "
        f"{'h2':>8} "
        f"{'MPC δ':>9} "
        f"{'CBF Δδ':>9} "
        f"{'safe δ':>9} "
        f"{'CBF':>5} "
        f"{'status':>8}"
    )

    print("-" * 150)


def print_row(
    row: dict,
) -> None:
    """Print one diagnostic step."""

    print(
        f"{row['time']:5.1f} "
        f"{row['ego_x']:8.2f} "
        f"{row['ego_y']:8.2f} "
        f"{row['lane']:>13} "
        f"{row['lateral_deviation']:8.3f} "
        f"{row['lcf']:4.0f} "
        f"{row['rcf']:4.0f} "
        f"{row['h1']:8.3f} "
        f"{row['h2']:8.3f} "
        f"{row['mpc_steering']:9.3f} "
        f"{row['cbf_delta_steering']:9.3f} "
        f"{row['safe_steering']:9.3f} "
        f"{row['cbf_intervention']:5d} "
        f"{row['barrier_status']:>8}"
    )


# ============================================================
# CSV
# ============================================================

def write_csv(
    rows: list[dict],
) -> None:
    """Write all diagnostic values to CSV."""

    if not rows:
        return

    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Rollout
# ============================================================

def run_diagnostic(
    model_path: str,
    ego_initial_speed: float,
    seed: int,
) -> None:
    """
    Run exactly one deterministic episode.
    """

    env = MergeEnvironment(
        ego_initial_speed=ego_initial_speed,
    )

    agent = PPOAgent()

    agent.load(
        model_path,
        load_optimizer=False,
    )

    observation, _ = env.reset(
        seed=seed,
    )

    rows: list[dict] = []

    terminated = False
    truncated = False

    step_index = 0

    previous_lane_index = None
    previous_lcf = None
    previous_rcf = None

    print()
    print(f"Model : {model_path}")
    print(f"Seed  : {seed}")
    print(
        f"Initial ego speed: "
        f"{ego_initial_speed:.1f} m/s"
    )

    print_table_header()

    try:
        while not terminated and not truncated:

            # ----------------------------------------------------
            # PRE-STEP road geometry
            # ----------------------------------------------------

            road = get_lane_diagnostics(
                env
            )

            ego_state = road["ego_state"]

            current_lane_index = (
                road["lane_index"]
            )

            # ----------------------------------------------------
            # Deterministic learned policy
            # ----------------------------------------------------

            action_result = agent.select_action(
                observation,
                deterministic=True,
            )

            rl_action = np.asarray(
                action_result.action,
                dtype=np.float64,
            )

            # ----------------------------------------------------
            # Run the real environment
            #
            # This executes the existing:
            #
            # RL -> MPC -> CBF -> HighwayEnv
            #
            # pipeline without modifying it.
            # ----------------------------------------------------

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                rl_action
            )

            # ----------------------------------------------------
            # Controls reported by the actual environment
            # ----------------------------------------------------

            safe_control = np.asarray(
                info["safe_control"],
                dtype=np.float64,
            )

            cbf_correction = np.asarray(
                info["cbf_correction"],
                dtype=np.float64,
            )

            # Since:
            #
            # safe_control
            #     = MPC_control + CBF_correction
            #
            # recover the actual MPC nominal command.
            mpc_control = (
                safe_control
                - cbf_correction
            )

            cbf_intervention = bool(
                info["cbf_intervention"]
            )

            # ----------------------------------------------------
            # POST-STEP state
            # ----------------------------------------------------

            next_state = get_ego_state(
                env
            )

            next_road = (
                get_lane_diagnostics(env)
            )

            # ----------------------------------------------------
            # Evaluate next h1/h2 using PRE-STEP tangent context.
            #
            # This is particularly important because it shows
            # what the current CBF geometry thinks happens over
            # one transition before the closest lane is changed.
            # ----------------------------------------------------

            h1_next_same_context = (
                left_road_barrier(
                    next_state,
                    road["road_context"],
                )
            )

            h2_next_same_context = (
                right_road_barrier(
                    next_state,
                    road["road_context"],
                )
            )

            status = barrier_status(
                road["h1"],
                road["h2"],
            )

            # ----------------------------------------------------
            # Detect changes
            # ----------------------------------------------------

            lane_changed = (
                previous_lane_index is not None
                and current_lane_index
                != previous_lane_index
            )

            feasibility_changed = (
                previous_lcf is not None
                and (
                    road["lcf"] != previous_lcf
                    or road["rcf"] != previous_rcf
                )
            )

            # ----------------------------------------------------
            # Store row
            # ----------------------------------------------------

            row = {
                "step": step_index,
                "time": (
                    step_index
                    * SIMULATION_DT
                ),

                # Ego state BEFORE env.step
                "ego_x": float(
                    ego_state[0]
                ),
                "ego_y": float(
                    ego_state[1]
                ),
                "ego_speed": float(
                    ego_state[2]
                ),
                "ego_heading": float(
                    ego_state[3]
                ),

                # Road geometry BEFORE env.step
                "lane": lane_index_string(
                    current_lane_index
                ),
                "longitudinal": road[
                    "longitudinal"
                ],
                "lateral_deviation": road[
                    "lateral_deviation"
                ],
                "lane_heading": road[
                    "lane_heading"
                ],
                "lane_slope": road[
                    "lane_slope"
                ],

                # Lane-change feasibility
                "lcf": road["lcf"],
                "rcf": road["rcf"],

                # Current road barriers
                "h1": road["h1"],
                "h2": road["h2"],

                # RL action
                "rl_action_0": float(
                    rl_action[0]
                ),
                "rl_action_1": float(
                    rl_action[1]
                ),

                # Physical references from action adapter
                "reference_velocity": float(
                    info[
                        "reference_velocity"
                    ]
                ),
                "reference_heading": float(
                    info[
                        "reference_heading"
                    ]
                ),

                # MPC nominal control
                "mpc_acceleration": float(
                    mpc_control[0]
                ),
                "mpc_steering": float(
                    mpc_control[1]
                ),

                # CBF correction
                "cbf_delta_acceleration": float(
                    cbf_correction[0]
                ),
                "cbf_delta_steering": float(
                    cbf_correction[1]
                ),

                # Final deployed control
                "safe_acceleration": float(
                    safe_control[0]
                ),
                "safe_steering": float(
                    safe_control[1]
                ),

                "cbf_intervention": int(
                    cbf_intervention
                ),
                "control_applied": bool(info.get('control_applied', True)),

                # Predicted barrier change over actual step,
                # evaluated in the PRE-STEP local road context.
                "h1_next_same_context": float(
                    h1_next_same_context
                ),
                "h2_next_same_context": float(
                    h2_next_same_context
                ),

                "delta_h1": float(
                    h1_next_same_context
                    - road["h1"]
                ),
                "delta_h2": float(
                    h2_next_same_context
                    - road["h2"]
                ),

                # State AFTER env.step
                "next_x": float(
                    next_state[0]
                ),
                "next_y": float(
                    next_state[1]
                ),
                "next_lane": (
                    lane_index_string(
                        next_road[
                            "lane_index"
                        ]
                    )
                ),
                "next_lateral_deviation": (
                    next_road[
                        "lateral_deviation"
                    ]
                ),
                "next_h1": next_road[
                    "h1"
                ],
                "next_h2": next_road[
                    "h2"
                ],
                "next_lcf": next_road[
                    "lcf"
                ],
                "next_rcf": next_road[
                    "rcf"
                ],

                # Events
                "lane_changed": int(
                    lane_changed
                ),
                "feasibility_changed": int(
                    feasibility_changed
                ),

                "barrier_status": status,

                # Environment result
                "reward": float(reward),
                "terminated": int(
                    terminated
                ),
                "truncated": int(
                    truncated
                ),
                "success": int(
                    info.get(
                        "success",
                        False,
                    )
                ),
                "collision": int(
                    info.get(
                        "collision",
                        False,
                    )
                ),
                "road_departure": int(
                    info.get(
                        "road_departure",
                        False,
                    )
                ),
                "termination_reason": str(
                    info.get(
                        "termination_reason",
                        "",
                    )
                ),
            }

            rows.append(row)

            # ----------------------------------------------------
            # Console output
            # ----------------------------------------------------

            if PRINT_EVERY_STEP:
                print_row(row)

            if lane_changed:
                print()
                print(
                    ">>> LANE INDEX CHANGED: "
                    f"{lane_index_string(previous_lane_index)}"
                    " -> "
                    f"{lane_index_string(current_lane_index)}"
                )
                print()

            if feasibility_changed:
                print()
                print(
                    ">>> lcf / rcf CHANGED: "
                    f"({previous_lcf:.0f}, "
                    f"{previous_rcf:.0f})"
                    " -> "
                    f"({road['lcf']:.0f}, "
                    f"{road['rcf']:.0f})"
                )
                print()

            if (
                road["h1"] < 0.0
                or road["h2"] < 0.0
            ):
                print(
                    ">>> WARNING: "
                    "road barrier is negative."
                )

            # ----------------------------------------------------
            # Update iteration
            # ----------------------------------------------------

            previous_lane_index = (
                current_lane_index
            )

            previous_lcf = road["lcf"]
            previous_rcf = road["rcf"]

            observation = next_observation

            step_index += 1

    finally:
        env.close()

    # ========================================================
    # Save result
    # ========================================================

    write_csv(rows)

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 72)
    print("ROLLOUT SUMMARY")
    print("=" * 72)

    if rows:
        last = rows[-1]

        intervention_count = sum(
            row["cbf_intervention"]
            for row in rows
        )

        lane_change_count = sum(
            row["lane_changed"]
            for row in rows
        )

        feasibility_change_count = sum(
            row["feasibility_changed"]
            for row in rows
        )

        minimum_h1 = min(
            row["h1"]
            for row in rows
        )

        minimum_h2 = min(
            row["h2"]
            for row in rows
        )

        print(
            f"Steps                : "
            f"{len(rows)}"
        )

        print(
            f"Duration             : "
            f"{len(rows) * SIMULATION_DT:.2f} s"
        )

        print(
            f"Final position       : "
            f"x={last['next_x']:.3f}, "
            f"y={last['next_y']:.3f}"
        )

        print(
            f"Final lane           : "
            f"{last['next_lane']}"
        )

        print(
            f"Minimum h1           : "
            f"{minimum_h1:.6f}"
        )

        print(
            f"Minimum h2           : "
            f"{minimum_h2:.6f}"
        )

        print(
            f"CBF interventions    : "
            f"{intervention_count}"
            f"/{len(rows)}"
        )

        print(
            f"Lane-index changes   : "
            f"{lane_change_count}"
        )

        print(
            f"lcf/rcf changes      : "
            f"{feasibility_change_count}"
        )

        print(
            f"Success              : "
            f"{last['success']}"
        )

        print(
            f"Collision            : "
            f"{last['collision']}"
        )

        print(
            f"Road departure       : "
            f"{last['road_departure']}"
        )

        print(
            f"Termination reason   : "
            f"{last['termination_reason']}"
        )

    print()
    print(
        f"CSV saved to:\n"
        f"{CSV_PATH}"
    )


# ============================================================
# Command-line arguments
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default=str(
            DEFAULT_MODEL_PATH
        ),
        help=(
            "Path to learned PPO model."
        ),
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_EGO_SPEED,
        choices=(5.0, 10.0),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    return parser.parse_args()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    run_diagnostic(
        model_path=args.model,
        ego_initial_speed=args.speed,
        seed=args.seed,
    )
