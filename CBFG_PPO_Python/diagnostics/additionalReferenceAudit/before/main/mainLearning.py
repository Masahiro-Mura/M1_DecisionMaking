"""
Train CBFG-PPO and save all generated artifacts under results/.

Outputs
-------
results/
    models/
        bestPolicy.pt
        finalPolicy.pt
    checkpoints/
        checkpoint_<step>.pt
    logs/
        trainingLog.csv
        updateLog.csv
    figures/
        trainingReward.png
        successRate.png
        trainingLosses.png
        cbfInterventionRate.png

This file is an entry point only. PPO/MPC/CBF/IDM equations are implemented
in their dedicated modules.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from environment.mergeEnvironment import MergeEnvironment
from parameters.rlParameters import (
    CHECKPOINT_INTERVAL_STEPS,
    SUCCESS_SMOOTHING_WINDOW,
    SUCCESS_RATE_STEP_WINDOW,
    TOTAL_TIMESTEPS,
    BEST_POLICY_EVALUATION_EPISODES,
    BEST_POLICY_EVALUATION_SEED,
)
from rl.ppoAgent import PPOAgent, UpdateMetrics


RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"


def configure_result_directory(output_dir: str | None) -> None:
    """Isolate experiment artifacts while preserving the default entry point."""
    global RESULTS_DIR, MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR, FIGURES_DIR
    if output_dir is None:
        return
    RESULTS_DIR = Path(output_dir).resolve()
    MODELS_DIR = RESULTS_DIR / "models"
    CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
    LOGS_DIR = RESULTS_DIR / "logs"
    FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_result_directories() -> None:
    """Create output folders if they do not yet exist."""
    for directory in (
        MODELS_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def moving_average(values: list[float], window: int) -> np.ndarray:
    """Trailing moving average preserving the same number of samples."""
    array = np.asarray(values, dtype=np.float64)

    if array.size == 0:
        return array

    result = np.empty_like(array)

    for i in range(array.size):
        start = max(0, i - window + 1)
        result[i] = np.mean(array[start : i + 1])

    return result


def episode_is_complete(row: dict) -> bool:
    """Budget exhaustion is not a completed failure episode.

    For legacy logs, an empty termination_reason marks an unfinished episode.
    """
    if "episode_complete" in row:
        return bool(int(row["episode_complete"]))
    return bool(row.get("termination_reason"))


def recent_episode_success_rate(rows: list[dict], end_step: int) -> float:
    """IV-A: successful / completed episodes ending in (t-3000, t]."""
    completed = [row for row in rows if episode_is_complete(row)
                 and end_step - SUCCESS_RATE_STEP_WINDOW < int(row["global_step"]) <= end_step]
    return float(np.mean([float(row["success"]) for row in completed])) if completed else float("nan")


def paper_success_rates(rows: list[dict]) -> np.ndarray:
    """Raw IV-A rates evaluated at completed-episode endpoints.

    Endpoint sampling is an explicit choice; the paper gives no sampling cadence.
    Width-10 smoothing is applied to these rates by the plot function.
    """
    return np.array([recent_episode_success_rate(rows, int(row["global_step"]))
                     for row in rows], dtype=np.float64)


def save_training_figures(rows: list[dict]) -> None:
    """Generate training figures from episode-level log rows."""
    rows = [row for row in rows if episode_is_complete(row)]
    if not rows:
        return

    steps = np.asarray([row["global_step"] for row in rows], dtype=np.int64)
    rewards = np.asarray([row["episode_reward"] for row in rows], dtype=np.float64)
    raw_success_rates = paper_success_rates(rows)

    # ------------------------------------------------------------
    # Average episodic reward: main-paper Fig. 7 metric.
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, rewards)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Episode reward")
    ax.set_title("Training reward (single run)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "trainingReward.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------
    # IV-A: 3000-step completed-episode rate, then width-10 smoothing.
    # ------------------------------------------------------------
    success_rate = moving_average(
        raw_success_rates.tolist(),
        SUCCESS_SMOOTHING_WINDOW,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, success_rate)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        f"Success rate (last {SUCCESS_RATE_STEP_WINDOW} steps; smoothing = {SUCCESS_SMOOTHING_WINDOW})"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "successRate.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------
    # Latest optimization losses associated with each episode.
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        steps,
        [float(row["ppo_loss"]) for row in rows],
        label="PPO actor",
    )
    ax.plot(
        steps,
        [float(row["bc_loss"]) for row in rows],
        label="CBF-guided BC",
    )
    ax.plot(
        steps,
        [float(row["critic_loss"]) for row in rows],
        label="Critic",
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Training losses")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "trainingLosses.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        steps,
        [float(row["cbf_intervention_rate"]) for row in rows],
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Intervention fraction")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("CBF intervention rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "cbfInterventionRate.png",
        dpi=160,
    )
    plt.close(fig)


def write_episode_log(rows: list[dict]) -> None:
    """Rewrite the episode CSV so it is valid after every completed episode."""
    if not rows:
        return

    path = LOGS_DIR / "trainingLog.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_update_log(
    update_index: int,
    global_step: int,
    metrics: UpdateMetrics,
) -> None:
    """Append one PPO update result."""
    path = LOGS_DIR / "updateLog.csv"
    new_file = not path.exists() or path.stat().st_size == 0

    row = {
        "update": update_index,
        "global_step": global_step,
        "total_loss": metrics.total_loss,
        "actor_loss": metrics.actor_loss,
        "ppo_loss": metrics.ppo_loss,
        "bc_loss": metrics.bc_loss,
        "critic_loss": metrics.critic_loss,
        "entropy": metrics.entropy,
        "intervention_fraction": metrics.intervention_fraction,
    }

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if new_file:
            writer.writeheader()

        writer.writerow(row)


def evaluate_policy(agent: PPOAgent, ego_initial_speed: float, seed: int) -> tuple[float, float]:
    """Fixed held-out deterministic episodes; never use the training env.

    IMPLEMENTATION ASSUMPTION: rank by success fraction, then average return.
    Neither stochastic training scores nor unupdated networks are candidates.
    """
    env = MergeEnvironment(ego_initial_speed=ego_initial_speed)
    successes, returns = [], []
    try:
        for episode in range(BEST_POLICY_EVALUATION_EPISODES):
            observation, _ = env.reset(seed=BEST_POLICY_EVALUATION_SEED + seed + episode)
            total_reward = 0.
            while True:
                action = agent.select_action(observation, deterministic=True).action
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    successes.append(float(info['success']))
                    returns.append(total_reward)
                    break
    finally:
        env.close()
    return float(np.mean(successes)), float(np.mean(returns))


def select_best_policy(agent: PPOAgent, speed: float, seed: int,
                       best_score: tuple[float, float], global_step: int) -> tuple[float, float]:
    if agent.update_count <= 0:
        return best_score
    score = evaluate_policy(agent, speed, seed)
    selected = score >= best_score
    # Ties select the latest update, including the final partial rollout.
    if selected:
        agent.save(MODELS_DIR / 'bestPolicy.pt')
    path = LOGS_DIR / 'policySelectionLog.csv'
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow(['update', 'global_step', 'deterministic_success_rate',
                             'mean_return', 'selected_best'])
        writer.writerow([agent.update_count, global_step, *score, int(selected)])
    return score if selected else best_score


def train(
    ego_initial_speed: float,
    total_timesteps: int,
    seed: int,
    resume: str | None,
    output_dir: str | None = None,
) -> None:
    """Run CBFG-PPO training."""
    if total_timesteps <= 0:
        raise ValueError('total_timesteps must be positive')
    configure_result_directory(output_dir)
    ensure_result_directories()

    env = MergeEnvironment(
        ego_initial_speed=ego_initial_speed,
    )
    agent = PPOAgent(seed=seed)

    if resume is not None:
        agent.load(resume, load_optimizer=True, require_compatible=True)
        if agent.training_metadata.get('ego_initial_speed', ego_initial_speed) != ego_initial_speed:
            raise ValueError('Resume checkpoint was trained for a different initial-speed scenario.')
    agent.training_metadata = {'ego_initial_speed': ego_initial_speed,
                               'seed': seed, 'requested_timesteps': total_timesteps,
                               'resume': resume}

    episode_rows: list[dict] = []
    # Success statistics below use completed episodes in a 3000-step window.

    best_score = (-1., -np.inf)
    # Each training invocation has its own optimization/selection logs.
    for filename in ('updateLog.csv', 'policySelectionLog.csv'):
        (LOGS_DIR / filename).write_text('', encoding='utf-8')

    global_step = 0
    episode_index = 0
    update_index = 0
    next_checkpoint = CHECKPOINT_INTERVAL_STEPS

    zero_metrics = UpdateMetrics(
        total_loss=0.0,
        actor_loss=0.0,
        ppo_loss=0.0,
        bc_loss=0.0,
        critic_loss=0.0,
        entropy=0.0,
        intervention_fraction=0.0,
    )
    latest_metrics = zero_metrics

    while global_step < total_timesteps:
        observation, _ = env.reset(
            seed=seed + episode_index,
        )

        episode_reward = 0.0
        episode_length = 0
        episode_interventions = 0

        terminated = False
        truncated = False
        final_info = {}

        while (
            not terminated
            and not truncated
            and global_step < total_timesteps
        ):
            action_result = agent.select_action(
                observation,
                deterministic=False,
            )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action_result.action)

            agent.store_transition(
                state=observation,
                action_result=action_result,
                reward=reward,
                next_state=next_observation,
                # True terminal states stop value bootstrap.
                # Time-limit truncation is intentionally not passed as terminal.
                terminated=terminated,
                truncated=truncated,
                env_info=info,
            )

            observation = next_observation
            final_info = info

            episode_reward += float(reward)
            episode_length += 1
            episode_interventions += int(
                info.get("cbf_intervention", False)
            )
            global_step += 1
            agent.training_metadata['global_step'] = global_step

            if agent.buffer.full:
                latest_metrics = agent.update()
                update_index += 1
                append_update_log(
                    update_index,
                    global_step,
                    latest_metrics,
                )
                best_score = select_best_policy(agent, ego_initial_speed, seed,
                                                best_score, global_step)

            if global_step >= next_checkpoint:
                agent.save(
                    CHECKPOINTS_DIR
                    / f"checkpoint_{global_step}.pt"
                )
                next_checkpoint += CHECKPOINT_INTERVAL_STEPS

        success = bool(final_info.get("success", False))
        episode_complete = bool(terminated or truncated)

        episode_cbf_rate = (
            episode_interventions / max(episode_length, 1)
        )

        row = {
            "episode": episode_index,
            "global_step": global_step,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "success": int(success),
            "episode_complete": int(episode_complete),
            "collision": int(
                final_info.get("collision", False)
            ),
            "road_departure": int(
                final_info.get("road_departure", False)
            ),
            "time_limit": int(
                final_info.get("time_limit", False)
            ),
            "termination_reason": final_info.get(
                "termination_reason",
                "",
            ),
            "cbf_intervention_rate": episode_cbf_rate,
            "total_loss": latest_metrics.total_loss,
            "actor_loss": latest_metrics.actor_loss,
            "ppo_loss": latest_metrics.ppo_loss,
            "bc_loss": latest_metrics.bc_loss,
            "critic_loss": latest_metrics.critic_loss,
        }
        episode_rows.append(row)
        rolling_success_rate = recent_episode_success_rate(episode_rows, global_step)
        row["rolling_success_rate"] = rolling_success_rate

        write_episode_log(episode_rows)

        print(
            f"[Episode {episode_index:4d}] "
            f"step={global_step:7d} "
            f"reward={episode_reward:8.3f} "
            f"success={int(success)} "
            f"success_rate={rolling_success_rate:.3f} "
            f"cbf={episode_cbf_rate:.3f}"
        )

        episode_index += 1

    # Update from the final partial rollout so collected data are not discarded.
    if len(agent.buffer) > 0:
        latest_metrics = agent.update()
        update_index += 1
        append_update_log(
            update_index,
            global_step,
            latest_metrics,
        )
        best_score = select_best_policy(agent, ego_initial_speed, seed,
                                        best_score, global_step)

    agent.save(MODELS_DIR / "finalPolicy.pt")

    write_episode_log(episode_rows)
    save_training_figures(episode_rows)

    env.close()

    print("\nTraining complete.")
    print(f"Final model : {MODELS_DIR / 'finalPolicy.pt'}")
    print(f"Best model  : {MODELS_DIR / 'bestPolicy.pt'}")
    print(f"Logs        : {LOGS_DIR}")
    print(f"Figures     : {FIGURES_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--speed",
        type=float,
        default=5.0,
        choices=(5.0, 10.0),
        help="Ego initial-speed scenario [m/s].",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=TOTAL_TIMESTEPS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path.",
    )

    parser.add_argument("--output-dir", default=None, help="Dedicated result directory for this run.")
    parser.add_argument("--torch-threads", type=int, default=1, help="CPU threads for the small neural networks.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import torch
    if args.torch_threads < 1:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)

    train(
        ego_initial_speed=args.speed,
        total_timesteps=args.total_steps,
        seed=args.seed,
        resume=args.resume,
        output_dir=args.output_dir,
    )
