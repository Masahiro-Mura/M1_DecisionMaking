"""
CBFG-PPO agent orchestrating Actor, Critic, rollout buffer, PPO, and BC loss.

The mathematical details remain in the dedicated modules:
    actor.py
    critic.py
    advantage.py
    ppoLoss.py
    cbfGuidedLoss.py

This file only coordinates action selection, transition storage, and network
updates, matching the role of Algorithm 1 in the main paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import torch

from parameters.rlParameters import (
    ENTROPY_COEFFICIENT,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    NUM_OPTIMIZATION_EPOCHS,
    POLICY_LOSS_COEFFICIENT,
    TRAINING_BATCH_SIZE,
    VALUE_LOSS_COEFFICIENT,
)
from rl.actor import Actor
from rl.advantage import compute_gae, one_step_td_targets
from rl.cbfGuidedLoss import (
    behavior_cloning_loss,
    build_demonstration_action,
)
from rl.critic import Critic
from rl.ppoLoss import critic_td_loss, ppo_actor_loss
from rl.rolloutBuffer import RolloutBuffer


@dataclass(frozen=True)
class ActionResult:
    action: np.ndarray
    log_prob: float
    value: float
    pre_tanh_action: np.ndarray | None = None


@dataclass(frozen=True)
class UpdateMetrics:
    total_loss: float
    actor_loss: float
    ppo_loss: float
    bc_loss: float
    critic_loss: float
    entropy: float
    intervention_fraction: float


class PPOAgent:
    """CBFG-PPO agent for normalized 2-D behavioral actions."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        self.device = torch.device(device)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.rng = np.random.default_rng(seed)

        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)

        # Main paper states Adam is used. It does not specify whether actor and
        # critic use separate optimizers. A single Adam over both networks is
        # used so Eq. (13)'s weighted combined loss can be applied directly.
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters())
            + list(self.critic.parameters()),
            lr=LEARNING_RATE,
        )

        self.buffer = RolloutBuffer()
        self.update_count = 0
        self.training_metadata: dict = {}

    @torch.no_grad()
    def select_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> ActionResult:
        """Sample the policy during training or use its mean for evaluation."""
        state = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        value = self.critic(state)

        if deterministic:
            pre_tanh_tensor = self.actor.forward(state)
            action_tensor = torch.tanh(pre_tanh_tensor)
            log_prob_tensor, _ = self.actor.evaluate_actions(
                state,
                action_tensor,
                pre_tanh_actions=pre_tanh_tensor,
            )
        else:
            action_tensor, log_prob_tensor, pre_tanh_tensor = self.actor.sample(
                state, return_pre_tanh=True)

        return ActionResult(
            action=action_tensor.squeeze(0).cpu().numpy(),
            log_prob=float(log_prob_tensor.item()),
            value=float(value.item()),
            pre_tanh_action=pre_tanh_tensor.squeeze(0).cpu().numpy(),
        )

    def store_transition(
        self,
        state: np.ndarray,
        action_result: ActionResult,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        env_info: dict,
        truncated: bool = False,
    ) -> None:
        """
        Store one Algorithm-1 transition.

        env_info is expected from environment/MergeEnvironment and contains:
            cbf_correction
            cbf_intervention
        """
        cbf_correction = np.asarray(
            env_info.get("cbf_correction", np.zeros(2)),
            dtype=np.float32,
        )

        intervention = bool(
            env_info.get("cbf_intervention", False)
        )

        actor_valid = bool(env_info.get('control_applied', True))
        demonstration_action = build_demonstration_action(
            policy_action=action_result.action,
            cbf_correction=cbf_correction,
            ego_state=env_info.get('ego_state_before'),
            nominal_control=env_info.get('mpc_control'),
            minimum_reference_speed=float(env_info.get('ego_minimum_speed', 0.)),
        ) if actor_valid and intervention else action_result.action.copy()

        self.buffer.add(
            state=state,
            action=action_result.action,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            old_log_prob=action_result.log_prob,
            value=action_result.value,
            cbf_correction=cbf_correction,
            cbf_intervention=intervention,
            demonstration_action=demonstration_action,
            truncated=truncated,
            actor_valid=actor_valid,
            pre_tanh_action=action_result.pre_tanh_action,
        )

    @torch.no_grad()
    def _prepare_advantages(self) -> np.ndarray:
        """Compute next-state values, GAE and returns for the current rollout."""
        next_states = torch.as_tensor(
            np.asarray(self.buffer.next_states),
            dtype=torch.float32,
            device=self.device,
        )

        next_values = self.critic(next_states).cpu().numpy()

        advantages, returns = compute_gae(
            rewards=np.asarray(self.buffer.rewards, dtype=np.float32),
            values=np.asarray(self.buffer.values, dtype=np.float32),
            next_values=next_values,
            terminated=np.asarray(
                self.buffer.terminated,
                dtype=np.float32,
            ),
            episode_ends=np.asarray(self.buffer.episode_ends, dtype=np.float32),
        )

        self.buffer.set_advantages_and_returns(
            advantages=advantages,
            returns=returns,
        )

        return next_values

    def update(self) -> UpdateMetrics:
        """
        Update actor and critic from the current on-policy rollout.

        Actor:
            non-intervention -> PPO Eq. (10)
            intervention     -> BC Eq. (31)

        Critic:
            all samples -> Eq. (12)
        """
        if len(self.buffer) == 0:
            raise RuntimeError("Cannot update from an empty buffer.")

        next_values_np = self._prepare_advantages()
        batch = self.buffer.as_batch(self.device)

        td_targets = torch.as_tensor(
            one_step_td_targets(
                rewards=np.asarray(
                    self.buffer.rewards,
                    dtype=np.float32,
                ),
                next_values=next_values_np,
                terminated=np.asarray(
                    self.buffer.terminated,
                    dtype=np.float32,
                ),
            ),
            dtype=torch.float32,
            device=self.device,
        )

        metric_sums = {
            "total_loss": 0.0,
            "actor_loss": 0.0,
            "ppo_loss": 0.0,
            "bc_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
        }
        update_count = 0
        actor_valid = torch.as_tensor(self.buffer.actor_valid, dtype=torch.bool,
                                      device=self.device)

        for _ in range(NUM_OPTIMIZATION_EPOCHS):
            for indices_np in RolloutBuffer.minibatch_indices(
                size=len(self.buffer),
                batch_size=TRAINING_BATCH_SIZE,
                rng=self.rng,
            ):
                indices = torch.as_tensor(
                    indices_np,
                    dtype=torch.long,
                    device=self.device,
                )

                states = batch.states[indices]
                actions = batch.actions[indices]
                old_log_probs = batch.old_log_probs[indices]
                advantages = batch.advantages[indices]
                intervention_mask = batch.cbf_interventions[indices]
                valid = actor_valid[indices]
                demo_actions = batch.demonstration_actions[indices]

                new_log_probs, entropy_values = (
                    self.actor.evaluate_actions(states, actions,
                                                pre_tanh_actions=batch.pre_tanh_actions[indices])
                )

                # ----------------------------------------------------
                # Actor Eq. (31): disjoint PPO and intervention sets.
                # ----------------------------------------------------
                non_intervention_mask = ~intervention_mask & valid
                # IMPLEMENTATION ASSUMPTION: rejected solver outputs are not
                # demonstrations of applied safe behavior; exclude actor loss.
                intervention_mask = intervention_mask & valid

                zero_actor = self.actor.forward(states).sum() * 0.0

                if torch.any(non_intervention_mask):
                    ppo_loss, _ = ppo_actor_loss(
                        new_log_probs=new_log_probs[
                            non_intervention_mask
                        ],
                        old_log_probs=old_log_probs[
                            non_intervention_mask
                        ],
                        advantages=advantages[
                            non_intervention_mask
                        ],
                    )
                else:
                    ppo_loss = zero_actor

                if torch.any(intervention_mask):
                    predicted_demo_actions = (
                        self.actor.deterministic_action(
                            states[intervention_mask]
                        )
                    )

                    bc_loss = behavior_cloning_loss(
                        predicted_actions=predicted_demo_actions,
                        demonstration_actions=demo_actions[
                            intervention_mask
                        ],
                        # Fixed rollout correction in the SAME coordinates
                        # as BC error; do not recalculate against current mean.
                        behavior_corrections=(demo_actions - actions)[
                            intervention_mask
                        ],
                    )
                else:
                    bc_loss = zero_actor

                actor_loss = ppo_loss + bc_loss

                # ----------------------------------------------------
                # Critic Eq. (12): intervention does not modify critic.
                # ----------------------------------------------------
                values = self.critic(states)

                critic_loss = critic_td_loss(
                    values=values,
                    td_targets=td_targets[indices],
                )

                entropy = entropy_values.mean()

                total_loss = (
                    POLICY_LOSS_COEFFICIENT * actor_loss
                    + VALUE_LOSS_COEFFICIENT * critic_loss
                    - ENTROPY_COEFFICIENT * entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()

                if MAX_GRAD_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.actor.parameters())
                        + list(self.critic.parameters()),
                        max_norm=float(MAX_GRAD_NORM),
                    )

                self.optimizer.step()

                metric_sums["total_loss"] += float(total_loss.item())
                metric_sums["actor_loss"] += float(actor_loss.item())
                metric_sums["ppo_loss"] += float(ppo_loss.item())
                metric_sums["bc_loss"] += float(bc_loss.item())
                metric_sums["critic_loss"] += float(
                    critic_loss.item()
                )
                metric_sums["entropy"] += float(entropy.item())

                update_count += 1

        intervention_fraction = float(
            np.mean(self.buffer.cbf_interventions)
        )

        self.buffer.clear()
        self.update_count += 1

        denominator = max(update_count, 1)

        return UpdateMetrics(
            total_loss=metric_sums["total_loss"] / denominator,
            actor_loss=metric_sums["actor_loss"] / denominator,
            ppo_loss=metric_sums["ppo_loss"] / denominator,
            bc_loss=metric_sums["bc_loss"] / denominator,
            critic_loss=metric_sums["critic_loss"] / denominator,
            entropy=metric_sums["entropy"] / denominator,
            intervention_fraction=intervention_fraction,
        )

    def save(self, path: str | Path) -> None:
        """Save actor, critic, and optimizer states."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metadata": {
                    "schema_version": 5,
                    "update_count": self.update_count,
                    "implementation": "cbfg-ppo-episode-latent-v5",
                    "training": self.training_metadata.copy(),
                    "update_schedule": "episode end; final budget-limited partial episode also updated",
                    "buffer_interpretation": "3072 capacity, not a collection target",
                    "ppo_probability": "stored pre-tanh sample; stable tanh log-Jacobian",
                    "teacher_mapping": "original reference plus safe-minus-nominal one-step speed/heading",
                    "bc_weight_space": "normalized rollout teacher minus original normalized action",
                    "bc_objective": "main-paper Eq31 weighted squared error; not reference44 likelihood",
                    "intervention_threshold_space": "physical controller correction (unreported convention)",
                    "source_sha256": {
                        str(source.relative_to(Path(__file__).resolve().parents[1])):
                        hashlib.sha256(source.read_bytes()).hexdigest()
                        for folder in ('rl', 'mpc', 'cbf', 'environment', 'parameters', 'scenario', 'surrounding_vehicle_model')
                        for source in [*(Path(__file__).resolve().parents[1] / folder).glob('*.py'),
                                       *([Path(__file__).resolve().parents[1] / "main/mainLearning.py"] if folder == 'rl' else [])]
                    },
                },
            },
            path,
        )

    def load(
        self,
        path: str | Path,
        load_optimizer: bool = True,
        require_trained: bool = False,
        require_compatible: bool = False,
    ) -> None:
        """Load a saved checkpoint."""
        checkpoint = torch.load(
            Path(path),
            map_location=self.device,
        )

        metadata = checkpoint.get('metadata', {})
        trained = bool(metadata.get('update_count', 0) or checkpoint.get('optimizer', {}).get('state', {}))
        if require_trained and not trained:
            raise ValueError('Checkpoint has no training updates. Train a new policy or select a trained checkpoint.')
        if require_compatible:
            root = Path(__file__).resolve().parents[1]
            hashes = metadata.get('source_sha256', {})
            if metadata.get('implementation') != 'cbfg-ppo-episode-latent-v5' or not hashes or any(
                not (root / name).is_file() or hashlib.sha256((root / name).read_bytes()).hexdigest() != value
                for name, value in hashes.items()
            ):
                raise ValueError('Checkpoint environment/algorithm metadata is incompatible. Start fresh training; legacy checkpoints remain available for diagnostics.')

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.training_metadata = metadata.get('training', {}).copy()
        self.update_count = int(checkpoint.get('metadata', {}).get('update_count', 0))
        if not self.update_count:
            self.update_count = int(bool(checkpoint.get('optimizer', {}).get('state', {})))

        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )
