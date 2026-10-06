"""
Evaluate a learned CBFG-PPO policy, create an MP4, and save figures.

Revision purpose
----------------
This version uses HighwayEnv's native pygame RoadGraphics/VehicleGraphics
renderer through ``MergeEnvironment.render_mode='rgb_array'``.

It also makes it explicit whether the simulated ego vehicle:
    1. actually enters a main-road lane,
    2. continues for at least 10 seconds,
    3. completes the merge according to the revised termination criterion.

The evaluation is still performed over all episodes. For visualization,
a representative successful episode is selected preferentially:
    success + main-lane entry + duration >= 10 s
then success + main-lane entry,
otherwise the longest episode.

All outputs are written under results/.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from environment.mergeEnvironment import MergeEnvironment
from environment.observation import build_observation
from parameters.environmentParameters import (
    EVALUATION_EPISODES,
    MAIN_ROAD_LANE_IDS,
    SIMULATION_DT,
    VIDEO_FPS,
)
from rl.ppoAgent import PPOAgent


RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"
VIDEOS_DIR = RESULTS_DIR / "videos"

# Diagnostic target only. This does NOT alter episode termination.
MIN_DIAGNOSTIC_DURATION_S = 10.0


class EvaluationEnvironment(MergeEnvironment):
    """IV-B adds test-time velocity variations to the training scenarios.

    The amplitude is not reported. Extra independent uniform +/-1 m/s is an
    explicitly configurable reproduction assumption, not a paper-given value.
    """
    def __init__(self, *args, test_speed_jitter: float = 1., **kwargs):
        super().__init__(*args, **kwargs)
        if test_speed_jitter < 0.:
            raise ValueError("test_speed_jitter must be nonnegative")
        self.test_speed_jitter = float(test_speed_jitter)

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        rng = np.random.default_rng(None if seed is None else seed + 2000000)
        for vehicle in self.surrounding_vehicles:
            vehicle.speed = max(0., float(vehicle.speed + rng.uniform(
                -self.test_speed_jitter, self.test_speed_jitter)))
        observation = build_observation(self.road, self.vehicle, self.surrounding_vehicles)
        return observation, info


def configure_result_directory(output_dir: str | None) -> None:
    """Isolate experiment artifacts while preserving the default entry point."""
    global RESULTS_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR, VIDEOS_DIR
    if output_dir is None:
        return
    RESULTS_DIR = Path(output_dir).resolve()
    MODELS_DIR = RESULTS_DIR / "models"
    LOGS_DIR = RESULTS_DIR / "logs"
    FIGURES_DIR = RESULTS_DIR / "figures"
    VIDEOS_DIR = RESULTS_DIR / "videos"


def ensure_result_directories() -> None:
    for directory in (
        MODELS_DIR,
        LOGS_DIR,
        FIGURES_DIR,
        VIDEOS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def vehicle_snapshot(vehicle) -> dict:
    return {
        "x": float(vehicle.position[0]),
        "y": float(vehicle.position[1]),
        "speed": float(vehicle.speed),
        "heading": float(vehicle.heading),
    }


def main_lane_status(env: MergeEnvironment) -> tuple[bool, float, tuple]:
    """
    IV-B merge point: ego center first enters the main-road lane region.

    This is a diagnostic, not a termination/success rule. Main-road membership
    uses the physical half lane width (y=2..6 m), not the stricter 1 m centering
    tolerance used for sustained success. The paper does not specify a vehicle-
    footprint or sustained-flow threshold for the first-entry metric.
    """
    lane_index = env.road.network.get_closest_lane_index(
        env.vehicle.position,
        env.vehicle.heading,
    )
    lane = env.road.network.get_lane(lane_index)
    longitudinal, lateral = lane.local_coordinates(env.vehicle.position)

    entered = (
        lane_index[:2] in (("a", "b"), ("b", "c"), ("c", "d"))
        and int(lane_index[2]) in MAIN_ROAD_LANE_IDS
        and abs(float(lateral)) <= lane.width_at(longitudinal) / 2.
    )

    return bool(entered), float(lateral), lane_index



def write_trace_csv(trace: list[dict], vehicle_count: int) -> None:
    path = LOGS_DIR / "simulationLog.csv"

    fields = [
        "time",
        "ego_x",
        "ego_y",
        "ego_speed",
        "reference_velocity",
        "reference_heading",
        "acceleration",
        "steering",
        "control_applied",
        "control_failure",
        "cbf_delta_acceleration",
        "cbf_delta_steering",
        "cbf_intervention",
        "cbf_candidate_feasible",
        "cbf_candidate_status",
        "cbf_optimizer_reported_success",
        "cbf_recovery_used",
        "cbf_infeasibility_proven",
        "cbf_minimum_constraint",
        "cbf_minimum_current_barrier",
        "cbf_initial_safe_set_satisfied",
        "main_lane_entered",
        "closest_lane_lateral_deviation",
        "closest_lane_id",
        "reward",
        "reward_merge",
    ]

    for i in range(vehicle_count):
        fields.extend(
            [
                f"sv{i+1}_x",
                f"sv{i+1}_y",
                f"sv{i+1}_speed",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for item in trace:
            writer.writerow(
                {
                    key: item.get(key, "")
                    for key in fields
                }
            )


def save_simulation_figures(
    trace: list[dict],
    vehicle_count: int,
    merge_time: float | None,
) -> None:
    if not trace:
        return

    time = np.asarray([row["time"] for row in trace])

    def add_merge_marker(ax) -> None:
        if merge_time is not None:
            ax.axvline(
                merge_time,
                linestyle="--",
                linewidth=1.2,
                label="Main-lane entry",
            )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, [row["ego_x"] for row in trace], label="EV")
    for i in range(vehicle_count):
        ax.plot(
            time,
            [row[f"sv{i+1}_x"] for row in trace],
            label=f"SV{i+1}",
        )
    add_merge_marker(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("x [m]")
    ax.set_title("Vehicle x-coordinates")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "simulationXPosition.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, [row["ego_y"] for row in trace], label="EV")
    for i in range(vehicle_count):
        ax.plot(
            time,
            [row[f"sv{i+1}_y"] for row in trace],
            label=f"SV{i+1}",
        )
    add_merge_marker(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("y [m]")
    ax.axhspan(2., 6., alpha=.08, color="tab:blue", label="Main road (2-6 m)")
    ax.axhspan(6., 10., alpha=.08, color="tab:orange", label="Parallel ramp (6-10 m)")
    ax.axhline(6., color="gray", linestyle=":", linewidth=1.)
    ax.set_title("Vehicle y-coordinates: single-lane merging")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "simulationYPosition.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, [row["ego_speed"] for row in trace], label="EV")
    for i in range(vehicle_count):
        ax.plot(
            time,
            [row[f"sv{i+1}_speed"] for row in trace],
            label=f"SV{i+1}",
        )
    add_merge_marker(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title("Vehicle speeds")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "simulationVelocity.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        time,
        [row["acceleration"] for row in trace],
        label="Acceleration [m/s²]",
    )
    ax.plot(
        time,
        [row["steering"] for row in trace],
        label="Steering [rad]",
    )
    add_merge_marker(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Control value")
    ax.set_title("Safe low-level control")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "simulationControl.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        time,
        [row["cbf_delta_acceleration"] for row in trace],
        label="Delta acceleration",
    )
    ax.plot(
        time,
        [row["cbf_delta_steering"] for row in trace],
        label="Delta steering",
    )
    add_merge_marker(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("CBF correction")
    ax.set_title("CBF safety correction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "simulationCBFCorrection.png",
        dpi=160,
    )
    plt.close(fig)


def write_evaluation_csv(rows: list[dict]) -> None:
    if not rows:
        return

    path = LOGS_DIR / "evaluationLog.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_evaluation_summary(rows: list[dict]) -> None:
    if not rows:
        return

    success_rate = np.mean([row["success"] for row in rows])
    collision_rate = np.mean([row["collision"] for row in rows])
    road_departure_rate = np.mean(
        [row["road_departure"] for row in rows]
    )
    time_limit_rate = np.mean([row["time_limit"] for row in rows])
    control_failure_rate = np.mean([row.get('control_failure', 0) for row in rows])

    labels = [
        "Success",
        "Collision",
        "Road departure",
        "Time limit",
        "Control failure",
    ]
    values = [
        success_rate,
        collision_rate,
        road_departure_rate,
        time_limit_rate,
        control_failure_rate,
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title(f"Evaluation results ({len(rows)} episodes)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "evaluationSummary.png", dpi=160)
    plt.close(fig)


def run_episode(
    env: MergeEnvironment,
    agent: PPOAgent,
    seed: int,
    record_video: bool,
) -> tuple[dict, list[dict]]:
    observation, _ = env.reset(seed=seed)

    trace: list[dict] = []
    episode_reward = 0.0
    episode_merge_reward: list[float] = []
    terminated = False
    truncated = False
    info = {}

    first_main_lane_entry_time: float | None = None
    first_main_lane_entry_x: float | None = None

    writer = None
    if record_video:
        writer = imageio.get_writer(
            VIDEOS_DIR / "simulation.mp4",
            fps=VIDEO_FPS,
            codec="libx264",
        )
        frame = env.render()
        if frame is None:
            raise RuntimeError(
                "render_mode='rgb_array' did not return a frame."
            )
        writer.append_data(frame)

    step_index = 0

    try:
        while not terminated and not truncated:
            action_result = agent.select_action(
                observation,
                deterministic=True,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action_result.action)

            step_index += 1
            current_time = step_index * SIMULATION_DT

            entered_main_lane, lateral, lane_index = main_lane_status(env)

            if (
                entered_main_lane
                and first_main_lane_entry_time is None
            ):
                first_main_lane_entry_time = current_time
                first_main_lane_entry_x = float(
                    env.vehicle.position[0]
                )

            episode_reward += float(reward)
            episode_merge_reward.append(
                float(info.get("reward_merge", 0.0))
            )

            row = {
                "time": current_time,
                "ego_x": float(env.vehicle.position[0]),
                "ego_y": float(env.vehicle.position[1]),
                "ego_speed": float(env.vehicle.speed),
                "reference_velocity": float(
                    info["reference_velocity"]
                ),
                "reference_heading": float(
                    info["reference_heading"]
                ),
                "acceleration": float(info['applied_control'][0]) if info['control_applied'] else np.nan,
                "steering": float(info['applied_control'][1]) if info['control_applied'] else np.nan,
                "control_applied": int(info['control_applied']),
                "control_failure": int(info['control_failure']),
                "cbf_delta_acceleration": float(
                    info["cbf_correction"][0]
                ),
                "cbf_delta_steering": float(
                    info["cbf_correction"][1]
                ),
                "cbf_intervention": int(
                    info["cbf_intervention"]
                ),
                **{key: info[key] for key in (
                    "cbf_candidate_feasible", "cbf_candidate_status",
                    "cbf_optimizer_reported_success", "cbf_recovery_used",
                    "cbf_infeasibility_proven", "cbf_minimum_constraint",
                    "cbf_minimum_current_barrier", "cbf_initial_safe_set_satisfied")},
                "main_lane_entered": int(entered_main_lane),
                "closest_lane_lateral_deviation": lateral,
                "closest_lane_id": int(lane_index[2]),
                "reward": float(reward),
                "reward_merge": float(
                    info.get("reward_merge", 0.0)
                ),
            }

            for i, vehicle in enumerate(
                env.surrounding_vehicles
            ):
                snapshot = vehicle_snapshot(vehicle)
                row[f"sv{i+1}_x"] = snapshot["x"]
                row[f"sv{i+1}_y"] = snapshot["y"]
                row[f"sv{i+1}_speed"] = snapshot["speed"]

            trace.append(row)

            if writer is not None:
                frame = env.render()
                if frame is None:
                    raise RuntimeError(
                        "render_mode='rgb_array' did not return a frame."
                    )
                writer.append_data(frame)
    finally:
        if writer is not None:
            writer.close()

    duration_s = step_index * SIMULATION_DT
    main_lane_entered = (
        first_main_lane_entry_time is not None
    )

    summary = {
        "seed": seed,
        "reward": episode_reward,
        "episode_steps": step_index,
        "duration_s": duration_s,
        "success": int(info.get("success", False)),
        "collision": int(info.get("collision", False)),
        "road_departure": int(
            info.get("road_departure", False)
        ),
        "time_limit": int(info.get("time_limit", False)),
        "control_failure": int(info.get('control_failure', False)),
        "termination_reason": info.get(
            "termination_reason",
            "",
        ),
        "main_lane_entered": int(main_lane_entered),
        "main_lane_entry_time_s": (
            first_main_lane_entry_time
            if first_main_lane_entry_time is not None
            else np.nan
        ),
        # This now means the FIRST detected main-lane entry,
        # not the final x-position of a successful episode.
        "merge_point_x": (
            first_main_lane_entry_x
            if first_main_lane_entry_x is not None
            else np.nan
        ),
        "duration_at_least_10s": int(
            duration_s >= MIN_DIAGNOSTIC_DURATION_S
        ),
        "mean_r_merge": (
            float(np.mean(episode_merge_reward))
            if episode_merge_reward
            else 0.0
        ),
    }

    return summary, trace


def choose_visualization_episode(
    rows: list[dict],
) -> int:
    """
    Select the episode to visualize without changing aggregate evaluation.

    Priority:
      1) success + main-lane entry + duration >= 10 s
      2) success + main-lane entry
      3) main-lane entry
      4) longest episode
    """
    for i, row in enumerate(rows):
        if (
            row["success"]
            and row["main_lane_entered"]
            and row["duration_at_least_10s"]
        ):
            return i

    for i, row in enumerate(rows):
        if row["success"] and row["main_lane_entered"]:
            return i

    for i, row in enumerate(rows):
        if row["main_lane_entered"]:
            return i

    return int(
        np.argmax([row["duration_s"] for row in rows])
    )


def validate_diagnostic_source_update(model_path: str) -> list[str]:
    """Permit only the exact, audited diagnostic changes in [38]/[39] review.

    This is opt-in; the default evaluation keeps the original strict source
    guard. Do not rewrite historical checkpoint provenance as if retrained.
    """
    import hashlib
    import torch
    metadata = torch.load(model_path, map_location="cpu").get("metadata", {})
    audit = json.loads((PROJECT_ROOT / "diagnostics/controlReferenceAudit/source_compatibility.json").read_text(encoding="utf-8"))
    if metadata.get("implementation") != "cbfg-ppo-minimum-speed-v4" or not metadata.get("source_sha256"):
        raise ValueError("Only original trained v4 policies can use the diagnostic source update.")
    accepted = []
    for name, expected in metadata["source_sha256"].items():
        name = name.replace(chr(92), "/")
        actual = hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
        if expected == actual:
            continue
        change = audit.get(name, {})
        if name not in {"cbf/safeController.py", "environment/mergeEnvironment.py"} or expected != change.get("before") or actual != change.get("after"):
            raise ValueError(f"Unaudited source change: {name}. Diagnostic source update refused.")
        accepted.append(name)
    return accepted


def simulate(
    model_path: str,
    ego_initial_speed: float,
    episodes: int,
    seed: int,
    output_dir: str | None = None,
    test_speed_jitter: float = 1.,
    diagnostic_source_update: bool = False,
) -> None:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if diagnostic_source_update and output_dir is None:
        raise ValueError("Diagnostic evaluation requires a dedicated --output-dir.")
    accepted_updates = validate_diagnostic_source_update(model_path) if diagnostic_source_update else []
    configure_result_directory(output_dir)
    ensure_result_directories()

    env = EvaluationEnvironment(
        ego_initial_speed=ego_initial_speed,
        render_mode="rgb_array",
        test_speed_jitter=test_speed_jitter,
    )
    agent = PPOAgent()
    agent.load(model_path, load_optimizer=False, require_trained=True, require_compatible=not diagnostic_source_update)
    if agent.training_metadata.get('ego_initial_speed', ego_initial_speed) != ego_initial_speed:
        raise ValueError('Policy was trained for a different initial-speed scenario.')

    (LOGS_DIR / "evaluationManifest.json").write_text(json.dumps({
        "model": str(Path(model_path).resolve()), "ego_initial_speed": ego_initial_speed,
        "ego_minimum_speed": env.ego_minimum_speed,
        "episodes": episodes, "seed": seed, "test_speed_jitter": test_speed_jitter,
        "test_speed_jitter_source": "Implementation assumption: amplitude unreported in IV-B",
        "trained_policy_metadata": agent.training_metadata,
        "diagnostic_only": diagnostic_source_update,
        "audited_diagnostic_source_updates": accepted_updates,
        "source_compatibility_required": not diagnostic_source_update,
        "deterministic_action": "tanh of Gaussian mean (bounded policy convention)",
    }, indent=2), encoding="utf-8")
    evaluation_rows: list[dict] = []

    # ------------------------------------------------------------
    # First pass: evaluate all episodes without rendering overhead.
    # ------------------------------------------------------------
    for episode in range(episodes):
        summary, _ = run_episode(
            env=env,
            agent=agent,
            seed=seed + episode,
            record_video=False,
        )
        evaluation_rows.append(summary)

        print(
            f"[Evaluation {episode+1:3d}/{episodes}] "
            f"duration={summary['duration_s']:5.1f}s "
            f"main_lane={summary['main_lane_entered']} "
            f"merge_x={summary['merge_point_x']} "
            f"success={summary['success']} "
            f"reason={summary['termination_reason']}"
        )

    write_evaluation_csv(evaluation_rows)
    save_evaluation_summary(evaluation_rows)

    # ------------------------------------------------------------
    # Second pass: rerun a representative episode and record video.
    # ------------------------------------------------------------
    selected_index = choose_visualization_episode(
        evaluation_rows
    )
    selected_seed = seed + selected_index

    print(
        "\nVisualization episode: "
        f"{selected_index + 1}/{episodes}, seed={selected_seed}"
    )

    selected_summary, video_trace = run_episode(
        env=env,
        agent=agent,
        seed=selected_seed,
        record_video=True,
    )

    vehicle_count = len(env.surrounding_vehicles)

    write_trace_csv(
        video_trace,
        vehicle_count=vehicle_count,
    )
    save_simulation_figures(
        video_trace,
        vehicle_count=vehicle_count,
        merge_time=(
            None
            if np.isnan(
                selected_summary["main_lane_entry_time_s"]
            )
            else float(
                selected_summary["main_lane_entry_time_s"]
            )
        ),
    )

    env.close()

    success_rate = float(
        np.mean([row["success"] for row in evaluation_rows])
    )
    main_lane_entry_rate = float(
        np.mean(
            [
                row["main_lane_entered"]
                for row in evaluation_rows
            ]
        )
    )

    successful_merge_points = [
        row["merge_point_x"]
        for row in evaluation_rows
        if row["success"] and not np.isnan(row["merge_point_x"])
    ]
    mean_merge_point = (
        float(np.mean(successful_merge_points))
        if successful_merge_points
        else float("nan")
    )

    print("\nEvaluation complete.")
    print(f"Success rate       : {success_rate:.3f}")
    print(f"Main-lane entry rate: {main_lane_entry_rate:.3f}")
    print(f"Mean merge point   : {mean_merge_point:.3f} m")

    print("\nSelected simulation diagnostics:")
    print(
        f"Duration           : "
        f"{selected_summary['duration_s']:.1f} s"
    )
    print(
        f"Main lane entered  : "
        f"{bool(selected_summary['main_lane_entered'])}"
    )
    print(
        f"Main-lane entry t  : "
        f"{selected_summary['main_lane_entry_time_s']}"
    )
    print(
        f"Merge point x      : "
        f"{selected_summary['merge_point_x']}"
    )
    print(
        f"Success            : "
        f"{bool(selected_summary['success'])}"
    )

    if selected_summary["duration_s"] < MIN_DIAGNOSTIC_DURATION_S:
        print(
            "WARNING: selected simulation ended before 10 s. "
            "Check collision/road-departure/termination behavior."
        )

    if not selected_summary["main_lane_entered"]:
        print(
            "WARNING: ego never entered the main-lane center region."
        )

    print(f"\nVideo   : {VIDEOS_DIR / 'simulation.mp4'}")
    print(f"Logs    : {LOGS_DIR}")
    print(f"Figures : {FIGURES_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default=str(MODELS_DIR / "bestPolicy.pt"),
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=5.0,
        choices=(5.0, 10.0),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EVALUATION_EPISODES,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10_000,
    )

    parser.add_argument("--test-speed-jitter", type=float, default=1., help="Extra uniform test-speed variation [m/s]; unreported amplitude assumption.")
    parser.add_argument("--diagnostic-source-update", action="store_true", help="Evaluate a historical v4 policy only with the exact audited diagnostic changes; requires dedicated output-dir.")
    parser.add_argument("--output-dir", default=None, help="Dedicated result directory for this run.")
    parser.add_argument("--torch-threads", type=int, default=1, help="CPU threads for the small neural networks.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import torch
    if args.torch_threads < 1:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)

    simulate(
        model_path=args.model,
        ego_initial_speed=args.speed,
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        test_speed_jitter=args.test_speed_jitter,
        diagnostic_source_update=args.diagnostic_source_update,
    )
