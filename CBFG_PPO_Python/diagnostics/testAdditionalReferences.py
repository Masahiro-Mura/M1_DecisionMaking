"""Properties relevant to [37], [43], [44] and the local transfer assumption."""
from __future__ import annotations
import unittest
from unittest.mock import patch
import numpy as np
import torch
from cbf.safeController import SafeController
from cbf.roadCBF import RoadCBFContext
from cbf.vehicleCBF import SurroundingCBFState,vehicle_barrier
from environment.actionAdapter import action_to_reference,reference_to_action
from rl.cbfGuidedLoss import build_demonstration_action,behavior_cloning_loss,intervention_weight
from rl.ppoAgent import PPOAgent,ActionResult
from mpc.mpcController import MPCController
from environment.mergeEnvironment import MergeEnvironment


class AdditionalReferenceTests(unittest.TestCase):
    def test_pure_steering_intervention_does_not_change_reference_speed(self):
        # The reference is 18 m/s while actual speed is 5 m/s. Steering
        # correction alone must not invent a 13 m/s speed reduction.
        action=np.array([.8,0.])
        teacher=build_demonstration_action(action,np.array([0.,-.1]),
                                         np.array([250.,8.,5.,0.]),np.array([1.,.2]))
        self.assertAlmostEqual(action_to_reference(teacher).velocity,18.,places=5)
        self.assertLess(teacher[1],action[1])

    def test_teacher_continuity_at_zero_control_correction(self):
        action=np.array([.8,-.3]);state=np.array([250.,8.,5.,.2])
        zero=build_demonstration_action(action,np.zeros(2),state,np.array([1.,.1]))
        tiny=build_demonstration_action(action,np.array([1e-6,-1e-6]),state,np.array([1.,.1]))
        np.testing.assert_allclose(tiny,zero,atol=4e-7,rtol=0.)

    def test_acceleration_effect_retains_original_reference_and_physical_dt(self):
        teacher=build_demonstration_action(reference_to_action(12.,.1),np.array([2.,0.]),
                                         np.array([250.,8.,5.,0.]),np.array([1.,.1]))
        reference=action_to_reference(teacher)
        self.assertAlmostEqual(reference.velocity,12.4,places=5)
        self.assertAlmostEqual(reference.heading,.1,places=5)

    def test_teacher_is_bounded_and_zero_correction_is_identity(self):
        action=np.array([1.,1.])
        np.testing.assert_array_equal(build_demonstration_action(action,np.zeros(2)),action)
        teacher=build_demonstration_action(action,np.array([3.,.2]),
                                         np.array([250.,8.,10.,0.]),np.zeros(2))
        np.testing.assert_array_equal(teacher,action)

    def test_main_paper_squared_bc_and_fixed_rollout_weights(self):
        # [44] likelihood is intentionally not used: verify weighted MSE
        # and its descent direction with an independent hand calculation.
        predicted=torch.tensor([[0.,0.]],requires_grad=True)
        teacher=torch.tensor([[.3,.4]])
        rollout_action=torch.tensor([[.3,0.]])
        loss=behavior_cloning_loss(predicted,teacher,teacher-rollout_action)
        expected=(1.+np.exp(.4))*.25
        self.assertAlmostEqual(loss.item(),expected,places=6)
        loss.backward()
        np.testing.assert_allclose(predicted.grad.numpy(),
            [[-2*(1+np.exp(.4))*.3,-2*(1+np.exp(.4))*.4]],rtol=1e-6)
        self.assertFalse((teacher-rollout_action).requires_grad)

    def test_normalized_weights_have_finite_action_space_bound(self):
        corrections=torch.tensor([[0.,0.],[2.,2.]])
        weights=intervention_weight(corrections)
        self.assertEqual(weights[0].item(),2.)
        self.assertAlmostEqual(weights[1].item(),1+np.exp(np.sqrt(8)),places=4)

    def test_gamma_domain_from_reference37(self):
        for gamma in (0.,-0.1,1.1,float('nan'),float('inf')):
            with patch('cbf.safeController.CBF_GAMMA',gamma):
                with self.assertRaisesRegex(ValueError,'0 < gamma <= 1'):
                    SafeController()
        with patch('cbf.safeController.CBF_GAMMA',1.):SafeController()

    def test_exponential_barrier_decay_preserves_safe_sign(self):
        context=RoadCBFContext(250.,4.,0.,0.)
        controller=SafeController()
        result=controller.correct(np.array([250.,4.,5.,0.]),np.zeros(2),[],
                                  context,True,True)
        self.assertTrue(result.initial_safe_set_satisfied)
        self.assertAlmostEqual(result.minimum_current_barrier,2.)
        for c in controller._build_constraints(np.array([250.,4.,5.,0.]),np.zeros(2),[],context,True,True):
            self.assertAlmostEqual(c['fun'](np.zeros(2)),.1*c['current_barrier'])

    def test_feasible_residual_does_not_imply_initially_safe(self):
        # An overlapping SV moving away can satisfy the decay residual
        # while h is still negative. Feasibility alone is not the theorem.
        state=np.array([250.,4.,0.,0.])
        sv=SurroundingCBFState(np.array([251.,4.,3.,0.]))
        result=SafeController().correct(state,np.zeros(2),[sv])
        self.assertFalse(result.initial_safe_set_satisfied)
        self.assertLess(result.minimum_current_barrier,0.)
        self.assertTrue(result.success)

    def test_invalid_controller_candidate_never_creates_teacher(self):
        agent=PPOAgent(device='cpu',seed=0)
        observation=np.zeros(27,np.float32)
        action=ActionResult(np.zeros(2,np.float32),0.,0.)
        agent.store_transition(observation,action,-1.,observation,True,
            {'cbf_correction':np.ones(2),'cbf_intervention':True,'control_applied':False})
        self.assertFalse(agent.buffer.actor_valid[0])
        np.testing.assert_array_equal(agent.buffer.demonstration_actions[0],action.action)

    def test_scenario_minimum_speed_mapping_and_inverse(self):
        for minimum in (5.,10.):
            self.assertEqual(action_to_reference(np.array([-1.,0.]),minimum).velocity,minimum)
            self.assertEqual(action_to_reference(np.array([1.,0.]),minimum).velocity,20.)
            for speed in (minimum,(minimum+20)/2,20.):
                ref=action_to_reference(reference_to_action(speed,.2,minimum),minimum)
                np.testing.assert_allclose([ref.velocity,ref.heading],[speed,.2],atol=1e-6)

    def test_teacher_and_agent_use_scenario_speed_mapping(self):
        minimum=10.;state=np.array([250.,8.,10.,0.])
        action=reference_to_action(12.,0.,minimum)
        teacher=build_demonstration_action(action,np.array([2.,0.]),state,np.zeros(2),minimum)
        self.assertAlmostEqual(action_to_reference(teacher,minimum).velocity,12.4,places=5)
        agent=PPOAgent(device='cpu',seed=0);observation=np.zeros(27,np.float32)
        agent.store_transition(observation,ActionResult(action,0.,0.),1.,observation,False,
            {'ego_minimum_speed':minimum,'ego_state_before':state,'mpc_control':np.zeros(2),
             'cbf_correction':np.array([2.,0.]),'cbf_intervention':True,'control_applied':True})
        np.testing.assert_array_equal(agent.buffer.demonstration_actions[0],teacher)

    def test_mpc_respects_minimum_through_entire_horizon(self):
        result=MPCController().solve(np.array([250.,4.,10.2,0.]),0.,0.,minimum_speed=10.)
        self.assertTrue(result.success,result.message)
        self.assertGreaterEqual(float(result.predicted_states[1:,2].min()),10.-1e-6)
        self.assertLess(result.control[0],0.)

    def test_cbf_does_not_undo_minimum_speed_condition(self):
        state=np.array([250.,4.,10.,0.])
        result=SafeController().correct(state,np.array([-3.,0.]),[],minimum_speed=10.)
        self.assertTrue(result.success,result.message)
        self.assertGreaterEqual(result.corrected_control[0],-1e-6)
        self.assertGreaterEqual(state[2]+.2*result.corrected_control[0],10.-1e-6)

    def test_unreachable_minimum_rejected_instead_of_clipping_vehicle_speed(self):
        result=SafeController().correct(np.array([250.,4.,5.,0.]),np.zeros(2),[],minimum_speed=10.)
        self.assertFalse(result.success)
        self.assertIn('cannot be reached',result.message)

    def test_actual_world_speed_matches_minimum_under_lowest_speed_actions(self):
        for minimum in (5.,10.):
            env=MergeEnvironment(ego_initial_speed=minimum)
            try:
                env.reset(seed=10000);env.surrounding_vehicles=[];env.road.vehicles=[env.vehicle]
                for _ in range(12):
                    _,_,terminated,truncated,info=env.step(np.array([-1.,0.]))
                    self.assertTrue(info['control_applied'],info.get('cbf_solver_message'))
                    self.assertFalse(terminated or truncated)
                    self.assertGreaterEqual(env.vehicle.speed,minimum-1e-6)
                    self.assertAlmostEqual(info['reference_velocity'],minimum)
            finally:env.close()


if __name__=='__main__':unittest.main()
