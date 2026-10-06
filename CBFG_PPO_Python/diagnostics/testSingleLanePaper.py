"""Paper Fig.5 / IV-B geometry and IV-A metric regression checks."""
import unittest
import numpy as np
from environment.mergeEnvironment import MergeEnvironment
from environment.observation import lane_change_feasibility
from environment.termination import check_road_departure, check_merge_completed
from scenario.mergeRoad import MergeRoad
from main.mainLearning import recent_episode_success_rate, paper_success_rates, episode_is_complete
from rl.ppoAgent import PPOAgent
from pathlib import Path

class SingleLaneTests(unittest.TestCase):
    def test_network_has_one_main_lane_and_one_parallel_ramp(self):
        n=MergeRoad.build_network()
        for edge in [("a","b"),("b","c"),("c","d")]:
            self.assertEqual(len(n.graph[edge[0]][edge[1]]), 2 if edge == ("b","c") else 1)
            self.assertEqual(n.get_lane((*edge,0)).position(10,0)[1],4.)
        self.assertEqual(n.get_lane(("b","c",MergeRoad.RAMP_LANE_INDEX)).position(0,0).tolist(),[230.,8.])
        self.assertEqual(n.get_lane(("b","c",1)).position(80,0).tolist(),[310.,8.])
        with self.assertRaises(ValueError): MergeRoad.main_lane_index_for_x(250.,1)

    def test_scenarios_spawn_only_on_main_lane_and_ramp_can_merge_left(self):
        env=MergeEnvironment()
        try:
            for seed in range(20):
                observation,_=env.reset(seed=seed)
                self.assertEqual(env.vehicle.lane_index,("b","c",1))
                self.assertEqual(lane_change_feasibility(env.road,env.vehicle),(1.,0.))
                self.assertEqual(observation.shape,(27,))
                for sv in env.surrounding_vehicles:
                    self.assertEqual(sv.lane_index[2],0)
                    self.assertEqual(sv.position[1],4.)
            env.vehicle.position=np.array([330.,0.]); env.vehicle.on_state_update()
            self.assertTrue(check_road_departure(env.road,env.vehicle))
            self.assertFalse(check_merge_completed(env.road,env.vehicle))
            env.vehicle.position=np.array([330.,4.]); env.vehicle.on_state_update()
            self.assertFalse(check_road_departure(env.road,env.vehicle))
            self.assertTrue(check_merge_completed(env.road,env.vehicle))
            self.assertEqual(lane_change_feasibility(env.road,env.vehicle),(0.,0.))
        finally: env.close()

    def test_success_window_is_steps_not_episode_count_and_ignores_partial(self):
        rows=[dict(global_step=100,success=1,episode_complete=1),
              dict(global_step=2900,success=0,episode_complete=1),
              dict(global_step=3100,success=0,episode_complete=1),
              dict(global_step=3200,success=1,episode_complete=0)]
        self.assertEqual(recent_episode_success_rate(rows,2900),.5)
        self.assertEqual(recent_episode_success_rate(rows,3100),0.)
        self.assertEqual(recent_episode_success_rate(rows,3200),0.)
        np.testing.assert_allclose(paper_success_rates(rows[:3]),[1.,.5,0.])
        self.assertFalse(episode_is_complete(dict(termination_reason="")))
        self.assertTrue(episode_is_complete(dict(termination_reason="collision")))

    def test_two_lane_checkpoint_is_rejected_for_paper_evaluation(self):
        root=Path(__file__).resolve().parents[1]
        archived=root/"diagnostics/referenceAudit/legacyResults/models/bestPolicy.pt"
        checkpoint=archived if archived.exists() else root/"results/models/bestPolicy.pt"
        if not checkpoint.exists(): self.skipTest("No historical checkpoint")
        with self.assertRaisesRegex(ValueError,"incompatible"):
            PPOAgent(device="cpu").load(checkpoint,load_optimizer=False,require_compatible=True)

if __name__ == "__main__": unittest.main(verbosity=2)
