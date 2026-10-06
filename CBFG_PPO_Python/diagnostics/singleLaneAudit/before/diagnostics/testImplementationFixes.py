"""Regression tests for checkpoint selection, GAE, teacher units and safety.

Run: python -m unittest diagnostics.testImplementationFixes -v
Temporary training artifacts stay under diagnostics; existing results are intact.
"""
from __future__ import annotations
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from environment.actionAdapter import action_to_reference, reference_to_action
from environment.mergeEnvironment import MergeEnvironment
from environment.termination import check_success, evaluate_termination
from cbf.roadCBF import left_road_barrier, right_road_barrier, local_lateral_deviation
from cbf.safeController import SafeController
from mpc.kinematicBicycleModel import discrete_step
from mpc.mpcCost import trajectory_cost
from rl.advantage import compute_gae
from rl.cbfGuidedLoss import build_demonstration_action
from rl.ppoAgent import PPOAgent
from diagnostics.testMPCReference import isolated_checks
import main.mainLearning as learning
import main.mainSimulation as simulation


class ImplementationFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tests only: avoid tiny CPU batches spending time on thread scheduling.
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.env = MergeEnvironment()
        self.env.reset(seed=10000)

    def tearDown(self):
        self.env.close()

    def place(self, x, y, speed=5., heading=0.):
        self.env.vehicle.position = np.array([x, y], dtype=float)
        self.env.vehicle.speed = speed
        self.env.vehicle.heading = heading
        self.env.vehicle.on_state_update()

    def test_gae_truncation_bootstraps_but_does_not_cross_reset(self):
        advantages, _ = compute_gae(np.array([1., 100.]), np.zeros(2),
            np.array([3., 0.]), np.array([0., 1.]), episode_ends=np.ones(2))
        np.testing.assert_allclose(advantages, [3.97, 100.], rtol=1e-6)

    def test_gae_terminal_never_bootstraps(self):
        advantages, _ = compute_gae(np.array([1., 100.]), np.zeros(2),
            np.array([3., 0.]), np.ones(2), episode_ends=np.ones(2))
        np.testing.assert_allclose(advantages, [1., 100.])

    def test_gae_keeps_contiguous_episode_recursion(self):
        advantages, _ = compute_gae(np.array([1., 100.]), np.zeros(2),
                                   np.zeros(2), np.array([0., 1.]))
        np.testing.assert_allclose(advantages, [95.05, 100.], rtol=1e-6)

    def test_teacher_uses_safe_speed_heading_with_physical_units(self):
        state = np.array([230., 8., 5., .2])
        nominal, correction = np.array([1., -.1]), np.array([2., .15])
        teacher = build_demonstration_action(np.array([.8, -.8]), correction, state, nominal)
        ref = action_to_reference(teacher)
        expected = discrete_step(state, nominal + correction)
        np.testing.assert_allclose([ref.velocity, ref.heading], expected[2:], atol=1e-6)
        np.testing.assert_array_equal(build_demonstration_action(np.array([.3, -.2]), np.zeros(2)), np.array([.3, -.2], np.float32))
        with self.assertRaises(ValueError):
            build_demonstration_action(np.zeros(2), np.ones(2))

    def test_reference_adapter_round_trip(self):
        for v, heading in ((0., -.3), (5., 0.), (10., .1), (20., .3)):
            ref = action_to_reference(reference_to_action(v, heading))
            np.testing.assert_allclose([ref.velocity, ref.heading], [v, heading], atol=1e-6)

    def test_cost_has_no_extra_terminal_state_penalty(self):
        states = np.zeros((3, 4))
        states[-1, 2:] = [100., 100.]
        self.assertEqual(trajectory_cost(states, np.zeros((2, 2)), np.zeros(2)), 0.)
        states[1, 2:] = [2., 3.]
        self.assertEqual(trajectory_cost(states, np.zeros((2, 2)), np.zeros(2)), 13.)

    def test_mpc_signs_warm_start_and_highway_step(self):
        self.assertEqual(len(isolated_checks()), 21)

    def test_road_envelope_continuous_across_lane_change(self):
        values = []
        for y in (6.01, 5.99):
            self.place(250., y)
            state, context = self.env._ego_state(), self.env._road_cbf_context()
            values.append([left_road_barrier(state, context), right_road_barrier(state, context)])
            np.testing.assert_allclose(values[-1], [10.-y, y+2.], atol=1e-10)
        np.testing.assert_allclose(np.subtract(values[1], values[0]), [.02, -.02], atol=1e-10)

    def test_road_context_does_not_mix_previous_lane_frame(self):
        self.place(250., 4.)
        self.env._previous_ego_position = np.array([230., 8.])
        self.env._previous_lateral_deviation = 99.
        context = self.env._road_cbf_context()
        self.assertAlmostEqual(local_lateral_deviation(self.env._ego_state(), context), 0.)
        np.testing.assert_allclose(context.physical_barriers(self.env._ego_state()), [6., 6.])

    def test_road_envelope_ends_ramp_at_merge_end(self):
        self.place(309., 8.)
        context = self.env._road_cbf_context()
        self.assertGreater(left_road_barrier(self.env._ego_state(), context), 0.)
        self.assertLess(left_road_barrier(np.array([311., 8., 5., 0.]), context), 0.)

    def test_cbf_zero_when_safe_and_outer_edge_active_without_lane_flags(self):
        context = self.env._road_cbf_context()
        result = SafeController().correct(self.env._ego_state(), np.zeros(2), [], context)
        np.testing.assert_array_equal(result.correction, np.zeros(2))
        self.place(250., 9.99)
        context = self.env._road_cbf_context()
        result = SafeController().correct(self.env._ego_state(), np.array([0., .3]), [], context)
        self.assertTrue(result.success)
        self.assertLess(result.correction[1], -.1)
        self.assertGreaterEqual(result.minimum_constraint_value, -1e-6)

    def test_cbf_recovers_feasible_control_after_optimizer_failure(self):
        self.place(250., 9.99)
        failure = SimpleNamespace(x=np.zeros(2), success=False, message='synthetic solver failure')
        with patch('cbf.safeController.minimize', return_value=failure):
            result = SafeController().correct(self.env._ego_state(), np.array([0., .3]), [], self.env._road_cbf_context())
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.minimum_constraint_value, -1e-6)

    def test_infeasible_control_does_not_advance_world_or_become_teacher(self):
        self.place(309.5, 8., speed=10.)
        positions = [v.position.copy() for v in self.env.road.vehicles]
        observation, _, terminated, _, info = self.env.step(reference_to_action(10., 0.))
        self.assertTrue(terminated)
        self.assertEqual(info['termination_reason'], 'cbf_infeasible')
        self.assertFalse(info['control_applied'])
        self.assertIsNone(info['applied_control'])
        for vehicle, before in zip(self.env.road.vehicles, positions):
            np.testing.assert_array_equal(vehicle.position, before)
        agent = PPOAgent(device='cpu', seed=0)
        action = agent.select_action(observation, deterministic=True)
        agent.store_transition(observation, action, -1., observation, True, info)
        self.assertFalse(agent.buffer.actor_valid[0])
        self.assertTrue(agent.buffer.episode_ends[0])

    def test_success_requires_stability_target_and_no_collision(self):
        self.place(310., 4.)
        self.assertFalse(check_success(self.env.road, self.env.vehicle, 20))
        self.place(450., 4.)
        self.assertFalse(check_success(self.env.road, self.env.vehicle, 9))
        self.assertTrue(check_success(self.env.road, self.env.vehicle, 10))
        self.env.vehicle.crashed = True
        result = evaluate_termination(self.env.road, self.env.vehicle, 50, 5., 20)
        self.assertFalse(result.success)
        self.assertEqual(result.reason, 'collision')

    def test_stability_counter_accumulates_and_resets(self):
        self.env.surrounding_vehicles = []
        self.env.road.vehicles = [self.env.vehicle]
        self.place(310., 4., speed=10.)
        for _ in range(10):
            _, _, terminated, _, info = self.env.step(reference_to_action(10., 0.))
            self.assertFalse(terminated)
        self.assertEqual(info['stable_main_lane_steps'], 10)
        self.place(335., 2.)
        self.env.step(reference_to_action(10., 0.))
        self.assertEqual(self.env._stable_main_lane_steps, 0)
        self.env.reset(seed=10000)
        self.assertEqual(self.env._stable_main_lane_steps, 0)

    def test_simulation_records_rejected_control_as_failure_not_applied_input(self):
        failed = SimpleNamespace(control=np.zeros(2), success=False)
        agent = SimpleNamespace(select_action=lambda *args, **kwargs: SimpleNamespace(action=np.zeros(2)))
        with patch.object(self.env.mpc, 'solve', return_value=failed):
            summary, trace = simulation.run_episode(self.env, agent, 10000, False)
        self.assertEqual(summary['control_failure'], 1)
        self.assertEqual(trace[-1]['control_applied'], 0)
        self.assertTrue(np.isnan(trace[-1]['steering']))

    def test_manual_feedback_merge_reaches_target_without_departure(self):
        # IMPLEMENTATION ASSUMPTION: feedback is a diagnostic fixture only;
        # this proves controller/road integration, not learned-policy quality.
        self.env.surrounding_vehicles = []
        self.env.road.vehicles = [self.env.vehicle]
        for _ in range(200):
            heading = float(np.clip(.1 * (4. - self.env.vehicle.position[1]), -.1, .1))
            _, _, terminated, truncated, info = self.env.step(reference_to_action(10., heading))
            self.assertTrue(info['control_applied'])
            self.assertFalse(info['road_departure'])
            if terminated or truncated:
                break
        self.assertTrue(info['success'], info['termination_reason'])
        self.assertGreaterEqual(self.env.vehicle.position[0], 450.)

    def test_mpc_failure_is_rejected_without_world_step(self):
        failed = SimpleNamespace(control=np.zeros(2), success=False)
        position = self.env.vehicle.position.copy()
        with patch.object(self.env.mpc, 'solve', return_value=failed):
            _, _, terminated, _, info = self.env.step(np.zeros(2))
        self.assertTrue(terminated)
        self.assertEqual(info['termination_reason'], 'mpc_failed')
        self.assertFalse(info['control_applied'])
        np.testing.assert_array_equal(self.env.vehicle.position, position)
        self.env.reset(seed=10000)
        self.assertEqual(self.env._stable_main_lane_steps, 0)

    def test_best_selection_excludes_initial_policy(self):
        agent = PPOAgent(device='cpu', seed=0)
        with patch.object(learning, 'evaluate_policy') as evaluate:
            self.assertEqual(learning.select_best_policy(agent, 5., 0, (-1., -np.inf), 20), (-1., -np.inf))
            evaluate.assert_not_called()

    def test_best_selection_compares_updated_scores_and_replaces_ties(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            agent = SimpleNamespace(update_count=1)
            agent.save = lambda path: Path(path).write_text(str(agent.update_count))
            with patch.multiple(learning, MODELS_DIR=root, LOGS_DIR=root), patch.object(
                learning, 'evaluate_policy', side_effect=[(.5, 100.), (.4, 200.), (.5, 100.)]):
                best = learning.select_best_policy(agent, 5., 0, (-1., -np.inf), 100)
                self.assertEqual((root / 'bestPolicy.pt').read_text(), '1')
                agent.update_count = 2
                best = learning.select_best_policy(agent, 5., 0, best, 200)
                self.assertEqual((root / 'bestPolicy.pt').read_text(), '1')
                agent.update_count = 3
                learning.select_best_policy(agent, 5., 0, best, 300)
                self.assertEqual((root / 'bestPolicy.pt').read_text(), '3')

    def test_checkpoint_guards_reject_untrained_or_incompatible_models(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            model = Path(directory) / 'model.pt'
            agent = PPOAgent(device='cpu', seed=0)
            agent.save(model)
            with self.assertRaisesRegex(ValueError, 'no training updates'):
                agent.load(model, require_trained=True)
            checkpoint = torch.load(model, map_location='cpu', weights_only=True)
            checkpoint['metadata']['implementation'] = 'obsolete'
            torch.save(checkpoint, model)
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                agent.load(model, require_compatible=True)

    def test_short_training_selects_final_partial_update_and_preserves_parameters(self):
        # End-to-end real env, buffer, 40 epochs, metadata, evaluation and save.
        # Only output paths and held-out episode count are changed in this test.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            paths = {name: root / sub for name, sub in (
                ('RESULTS_DIR', ''), ('MODELS_DIR', 'models'),
                ('CHECKPOINTS_DIR', 'checkpoints'), ('LOGS_DIR', 'logs'), ('FIGURES_DIR', 'figures'))}
            with patch.multiple(learning, **paths), patch.object(learning, 'BEST_POLICY_EVALUATION_EPISODES', 1):
                learning.train(5., 8, 0, None)
                best = torch.load(root / 'models/bestPolicy.pt', map_location='cpu', weights_only=True)
                final = torch.load(root / 'models/finalPolicy.pt', map_location='cpu', weights_only=True)
                self.assertEqual(best['metadata']['update_count'], 1)
                self.assertEqual(best['metadata']['training']['global_step'], 8)
                self.assertTrue(best['optimizer']['state'])
                self.assertTrue(all(torch.equal(value, final['actor'][key]) for key, value in best['actor'].items()))
                self.assertTrue(best['metadata']['source_sha256'])
                loaded = PPOAgent(device='cpu', seed=0)
                loaded.load(root / 'models/bestPolicy.pt', require_trained=True, require_compatible=True)
                self.assertEqual((root / 'logs/policySelectionLog.csv').read_text().count('\n'), 2)
                initial = PPOAgent(device='cpu', seed=0)
                self.assertTrue(any(not torch.equal(value, best['actor'][key]) for key, value in initial.actor.state_dict().items()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
