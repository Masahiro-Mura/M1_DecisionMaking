"""Regression checks derived from supplied IDM [46] and PPO [47] papers."""
import unittest
from pathlib import Path
import numpy as np
import torch
from environment.mergeEnvironment import MergeEnvironment
from surrounding_vehicle_model.idmModel import IDMModel, IDMParameters
from surrounding_vehicle_model.surroundingVehicle import SurroundingVehicle
from rl.ppoLoss import ppo_actor_loss
from rl.advantage import compute_gae

class ReferencePaperTests(unittest.TestCase):
    def setUp(self):
        self.env=MergeEnvironment(); self.env.reset(seed=0)
    def tearDown(self): self.env.close()
    def vehicles(self, separation, front_speed=10., following_speed=10.):
        road=self.env.road
        follower=SurroundingVehicle(road,[250.,4.],speed=following_speed,target_speed=20.,target_lane_index=("b","c",0))
        leader=SurroundingVehicle(road,[250.+separation,4.],speed=front_speed,target_speed=20.,target_lane_index=("b","c",0))
        road.vehicles=[follower,leader]
        return follower,leader
    def test_idm_uses_net_distance_not_center_distance(self):
        follower,leader=self.vehicles(20.)
        gap,speed=follower._front_vehicle_state(leader)
        self.assertEqual(gap,15.); self.assertEqual(speed,10.)
        expected=3.*(1.-(10./20.)**4-(22./15.)**2)
        follower.act()
        self.assertAlmostEqual(follower.acceleration,expected)
    def test_stopped_queue_jam_distance_is_bumper_gap(self):
        follower,leader=self.vehicles(12.,front_speed=0.,following_speed=0.)
        follower.act()
        self.assertAlmostEqual(follower._front_vehicle_state(leader)[0],7.)
        self.assertAlmostEqual(follower.acceleration,0.)
    def test_gap_subtracts_half_lengths_for_unequal_vehicles(self):
        follower,leader=self.vehicles(20.)
        follower.LENGTH=4.;leader.LENGTH=6.
        self.assertEqual(follower._front_vehicle_state(leader)[0],15.)
    def test_approaching_vehicle_requires_larger_desired_gap(self):
        params=IDMParameters(20.)
        self.assertGreater(IDMModel.desired_gap(10.,5.,params),IDMModel.desired_gap(10.,0.,params))
    def test_desired_gap_matches_unclamped_printed_equation(self):
        params=IDMParameters(20.)
        expected=7.+1.5*10.+10.*(-20.)/(2.*np.sqrt(3.*5.))
        self.assertLess(expected,7.)
        self.assertAlmostEqual(IDMModel.desired_gap(10.,-20.,params),expected)
    def test_missing_leader_is_free_road_term(self):
        self.assertAlmostEqual(IDMModel.acceleration(10.,None,None,IDMParameters(20.)),3.*(1.-.5**4))
    def test_ppo_clip_limits_only_beneficial_probability_changes(self):
        old=torch.zeros(4)
        new=torch.log(torch.tensor([1.5,.5,.5,1.5]))
        advantage=torch.tensor([1.,-1.,1.,-1.])
        loss,_=ppo_actor_loss(new,old,advantage,.2)
        expected=-torch.tensor([1.2,-.8,.5,-1.5]).mean()
        self.assertAlmostEqual(loss.item(),expected.item(),places=6)
    def test_gae_matches_ppo_finite_sum_equations_11_12(self):
        rewards=np.array([1.,2.,3.]);values=np.array([.2,.4,.6]);next_values=np.array([.4,.6,.8])
        delta=rewards+.99*next_values-values
        expected=np.array([delta[0]+(.99*.95)*delta[1]+(.99*.95)**2*delta[2],delta[1]+(.99*.95)*delta[2],delta[2]])
        actual,_=compute_gae(rewards,values,next_values,np.zeros(3))
        np.testing.assert_allclose(actual,expected,rtol=1e-6)

    def test_exact_mpc_derivative_matches_original_cost_and_central_differences(self):
        from mpc.mpcCost import trajectory_cost, tracking_objective_and_gradient
        from mpc.kinematicBicycleModel import rollout
        rng=np.random.default_rng(5)
        for _ in range(12):
            state=np.array([250.,8.,rng.uniform(2.,20.),rng.uniform(-.5,.5)])
            controls=np.column_stack([rng.uniform(-3.,3.,10),rng.uniform(-.5,.5,10)])
            reference=np.array([rng.uniform(0.,20.),rng.uniform(-.6,.6)])
            cost,grad=tracking_objective_and_gradient(state,controls,reference)
            self.assertAlmostEqual(cost,trajectory_cost(rollout(state,controls),controls,reference),places=9)
            numerical=[]
            for i in range(20):
                plus=controls.copy().reshape(-1);minus=plus.copy()
                plus[i]+=1e-5;minus[i]-=1e-5
                jplus=trajectory_cost(rollout(state,plus.reshape(-1,2)),plus.reshape(-1,2),reference)
                jminus=trajectory_cost(rollout(state,minus.reshape(-1,2)),minus.reshape(-1,2),reference)
                numerical.append((jplus-jminus)/2e-5)
            np.testing.assert_allclose(grad,numerical,rtol=1e-5,atol=1e-6)

    def test_evaluation_speed_jitter_is_reproducible_bounded_and_updates_observation(self):
        from main.mainSimulation import EvaluationEnvironment
        base=EvaluationEnvironment(test_speed_jitter=0.)
        jittered=EvaluationEnvironment(test_speed_jitter=1.)
        try:
            before,_=base.reset(seed=12000);after,_=jittered.reset(seed=12000)
            expected_speeds=np.array([v.speed for v in jittered.surrounding_vehicles])
            differences=expected_speeds-np.array([v.speed for v in base.surrounding_vehicles])
            self.assertTrue(np.all(abs(differences)<=1.))
            self.assertGreater(np.max(abs(differences)),.01)
            self.assertFalse(np.array_equal(before,after))
            repeat,_=jittered.reset(seed=12000)
            np.testing.assert_array_equal(after,repeat)
            np.testing.assert_array_equal(expected_speeds,[v.speed for v in jittered.surrounding_vehicles])
        finally:base.close();jittered.close()

    def test_precreated_update_csv_has_header_and_preserves_first_update(self):
        import tempfile,csv
        from unittest.mock import patch
        import main.mainLearning as learning
        from rl.ppoAgent import UpdateMetrics
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            folder=Path(directory);(folder/"updateLog.csv").write_text("")
            with patch.object(learning,"LOGS_DIR",folder):
                learning.append_update_log(1,3072,UpdateMetrics(1.,2.,3.,4.,5.,6.,.7))
            with (folder/"updateLog.csv").open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]["global_step"],"3072")
            self.assertEqual(float(rows[0]["bc_loss"]),4.)

    def test_completed_csv_rows_generate_all_training_figures(self):
        import tempfile
        from unittest.mock import patch
        import main.mainLearning as learning
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            folder=Path(directory)
            rows=[dict(global_step="3000",success="1",episode_complete="1",episode_reward="25.5",ppo_loss="-1.2",bc_loss=".1",critic_loss=".8",cbf_intervention_rate=".2"),
                  dict(global_step="6000",success="0",episode_complete="1",episode_reward="12.0",ppo_loss="-.9",bc_loss=".2",critic_loss=".5",cbf_intervention_rate=".3")]
            with patch.object(learning,"FIGURES_DIR",folder):learning.save_training_figures(rows)
            for filename in ("trainingReward.png","successRate.png","trainingLosses.png","cbfInterventionRate.png"):
                with Image.open(folder/filename) as img:
                    self.assertGreater(img.width,500)

    def test_merge_point_detects_road_entry_before_success_center_tolerance(self):
        from main.mainSimulation import main_lane_status
        from environment.termination import check_merge_completed
        for y,expected in ((6.01,False),(5.99,True),(4.,True),(1.9,False)):
            self.env.vehicle.position=np.array([250.,y]);self.env.vehicle.heading=0.
            self.env.vehicle.on_state_update()
            self.assertEqual(main_lane_status(self.env)[0],expected)
        self.env.vehicle.position=np.array([305.,5.99]);self.env.vehicle.on_state_update()
        self.assertTrue(main_lane_status(self.env)[0])
        self.assertFalse(check_merge_completed(self.env.road,self.env.vehicle))

if __name__=="__main__":unittest.main(verbosity=2)
