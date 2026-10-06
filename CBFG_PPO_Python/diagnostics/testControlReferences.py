"""Regression tests for [38]/[39] feasibility diagnostics, not performance."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from cbf.safeController import SafeController
from cbf.roadCBF import RoadCBFContext
from environment.mergeEnvironment import MergeEnvironment
from pathlib import Path
import copy
import torch
from main.mainSimulation import validate_diagnostic_source_update
from rl.ppoAgent import PPOAgent


class ControlReferenceTests(unittest.TestCase):
    def solve_synthetic(self, current, constraint, candidate, optimizer_success):
        controller=SafeController()
        constraints=[{'current_barrier':current,'fun':constraint}]
        optimizer=SimpleNamespace(x=np.asarray(candidate),success=optimizer_success,message='test')
        with patch.object(controller,'_build_constraints',return_value=constraints), patch('cbf.safeController.minimize',return_value=optimizer):
            return controller.correct(np.array([250.,4.,5.,0.]),np.zeros(2),[])

    def test_optimized_result_retains_initial_safe_set_diagnostics(self):
        result=self.solve_synthetic(2.,lambda c:c[0]-.5,[.5,0.],True)
        self.assertTrue(result.success)
        self.assertEqual(result.minimum_current_barrier,2.)
        self.assertTrue(result.initial_safe_set_satisfied)
        self.assertTrue(result.optimizer_reported_success)
        self.assertFalse(result.used_feasibility_recovery)

    def test_feasible_solver_status_does_not_prove_initial_safety(self):
        result=self.solve_synthetic(-2.,lambda c:c[0]-.5,[.5,0.],True)
        self.assertTrue(result.success)
        self.assertFalse(result.initial_safe_set_satisfied)
        self.assertEqual(result.minimum_current_barrier,-2.)

    def test_failed_optimizer_may_supply_verified_feasible_candidate(self):
        result=self.solve_synthetic(2.,lambda c:c[0]-.5,[.5,0.],False)
        self.assertTrue(result.success)
        self.assertFalse(result.optimizer_reported_success)
        self.assertEqual(result.candidate_status,'verified_feasible_with_tolerance')

    def test_reported_optimizer_success_may_supply_infeasible_candidate(self):
        result=self.solve_synthetic(2.,lambda c:-1.,[0.,0.],True)
        self.assertFalse(result.success)
        self.assertTrue(result.optimizer_reported_success)
        self.assertEqual(result.minimum_current_barrier,2.)
        self.assertTrue(result.initial_safe_set_satisfied)
        self.assertEqual(result.candidate_status,'no_verified_feasible_candidate')

    def test_grid_recovery_is_reported_separately(self):
        result=self.solve_synthetic(2.,lambda c:c[0]-.5,[0.,0.],False)
        self.assertTrue(result.success)
        self.assertTrue(result.used_feasibility_recovery)
        self.assertFalse(result.optimizer_reported_success)

    def test_nominal_path_has_no_optimizer_status(self):
        result=SafeController().correct(np.array([250.,4.,5.,0.]),np.zeros(2),[],RoadCBFContext(250.,4.,0.,0.),True,True)
        self.assertTrue(result.initial_safe_set_satisfied)
        self.assertIsNone(result.optimizer_reported_success)

    def test_environment_exposes_feasibility_and_initial_state_separately(self):
        env=MergeEnvironment(ego_initial_speed=5.)
        try:
            env.reset(seed=0)
            _,_,_,_,info=env.step(np.zeros(2))
            self.assertEqual(info['cbf_candidate_feasible'],info['cbf_solver_success'])
            self.assertIn(info['cbf_candidate_status'],('verified_feasible_with_tolerance','no_verified_feasible_candidate'))
            self.assertFalse(info['cbf_infeasibility_proven'])
            self.assertTrue(np.isfinite(info['cbf_minimum_current_barrier']))
        finally:
            env.close()

    def test_real_road_intervention_preserves_current_barrier_diagnostics(self):
        env=MergeEnvironment(ego_initial_speed=5.)
        try:
            env.reset(seed=0)
            env.vehicle.position=np.array([250.,9.99])
            env.vehicle.heading=0.
            env.vehicle.speed=5.
            env.vehicle.lane_index=('b','c',1)
            env.vehicle.lane=env.road.network.get_lane(env.vehicle.lane_index)
            result=SafeController().correct(env._ego_state(),np.array([0.,.3]),[],env._road_cbf_context())
            self.assertTrue(result.success)
            self.assertLess(result.correction[1],-.1)
            self.assertTrue(result.initial_safe_set_satisfied)
            self.assertTrue(np.isfinite(result.minimum_current_barrier))
            self.assertAlmostEqual(result.minimum_current_barrier,.01,places=5)
        finally:
            env.close()

    def test_diagnostic_option_refuses_new_algorithm_changes(self):
        root=Path(__file__).resolve().parents[1]
        model=root/'results/runs/reference_v4_speed5_seed0/models/bestPolicy.pt'
        with self.assertRaisesRegex(ValueError,'Unaudited source change'):
            validate_diagnostic_source_update(str(model))

    def test_unaudited_checkpoint_hash_is_refused(self):
        root=Path(__file__).resolve().parents[1]
        model=root/'results/runs/reference_v4_speed5_seed0/models/bestPolicy.pt'
        checkpoint=copy.deepcopy(torch.load(model,map_location='cpu'))
        key=next(k for k in checkpoint['metadata']['source_sha256'] if k.replace(chr(92),'/')=='rl/actor.py')
        checkpoint['metadata']['source_sha256'][key]='0'*64
        with patch('torch.load',return_value=checkpoint):
            with self.assertRaisesRegex(ValueError,'Unaudited source change'):
                validate_diagnostic_source_update(str(model))

    def test_default_strict_policy_guard_is_preserved(self):
        root=Path(__file__).resolve().parents[1]
        agent=PPOAgent(device='cpu',seed=0)
        with self.assertRaisesRegex(ValueError,'metadata is incompatible'):
            agent.load(root/'results/runs/reference_v4_speed5_seed0/models/bestPolicy.pt',
                       load_optimizer=False,require_trained=True,require_compatible=True)


if __name__=='__main__':
    unittest.main(verbosity=2)
