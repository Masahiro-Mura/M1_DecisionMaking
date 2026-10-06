import copy
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from main import mainLearning as learning
from rl.ppoAgent import PPOAgent
from rl.rolloutBuffer import RolloutBuffer


class LearningStabilityTests(unittest.TestCase):
    def test_rollout_pools_episodes_and_flushes_final_remainder(self):
        sizes, versions = [], []
        class SpyAgent(PPOAgent):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.buffer = RolloutBuffer(capacity=4)
            def select_action(self, *args, **kwargs):
                versions.append(self.update_count)
                return super().select_action(*args, **kwargs)
            def update(self):
                sizes.append(len(self.buffer))
                return super().update()
        class TwoStepEnv:
            def __init__(self, ego_initial_speed): self.ego_minimum_speed = ego_initial_speed
            def reset(self, seed): self.n=0; return np.zeros(27,np.float32), {}
            def step(self, action):
                self.n += 1
                return np.zeros(27,np.float32),1.,self.n==2,False,{'control_applied':True,'termination_reason':'failure' if self.n==2 else ''}
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            paths={name:getattr(learning,name) for name in ('RESULTS_DIR','MODELS_DIR','CHECKPOINTS_DIR','LOGS_DIR','FIGURES_DIR')}
            with patch.multiple(learning,**paths), patch.object(learning,'PPOAgent',SpyAgent), patch.object(learning,'MergeEnvironment',TwoStepEnv), patch.object(learning,'select_best_policy',side_effect=lambda a,s,se,b,g:b), patch.object(learning,'save_training_figures'):
                learning.train(5.,7,0,None,output_dir=directory,update_schedule='rollout')
            self.assertEqual(sizes,[4,3])
            self.assertEqual(versions,[0,0,0,0,1,1,1])
            with (Path(directory)/'logs/updateLog.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual([int(r['global_step']) for r in rows],[4,7])
            self.assertEqual([int(r['rollout_samples']) for r in rows],[4,3])

    def test_kl_rejection_rolls_back_bc_actor_critic_and_adam(self):
        agent=PPOAgent(device='cpu',seed=0)
        observation=np.zeros(27,np.float32)
        for _ in range(8):
            result=agent.select_action(observation)
            agent.buffer.add(observation,result.action,1.,observation,True,
                result.log_prob,result.value,np.ones(2),True,np.ones(2),
                pre_tanh_action=result.pre_tanh_action)
        actor=copy.deepcopy(agent.actor.state_dict())
        critic=copy.deepcopy(agent.critic.state_dict())
        for group in agent.optimizer.param_groups: group['lr']=10.
        metrics=agent.update()
        self.assertTrue(metrics.kl_stopped)
        self.assertEqual(metrics.optimizer_steps,0)
        self.assertEqual(metrics.policy_kl,0.)
        self.assertEqual(agent.optimizer.state, {})
        for name,value in actor.items(): torch.testing.assert_close(agent.actor.state_dict()[name],value,rtol=0,atol=0)
        for name,value in critic.items(): torch.testing.assert_close(agent.critic.state_dict()[name],value,rtol=0,atol=0)
        self.assertEqual(metrics.ppo_samples,0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rejected.pt'
            agent.save(path)
            with self.assertRaisesRegex(ValueError,'no training updates'):
                PPOAgent(device='cpu').load(path,require_trained=True)

    def test_normalization_uses_only_eligible_ppo_samples(self):
        agent=PPOAgent(device='cpu',seed=0)
        obs=np.zeros(27,np.float32)
        for intervention,valid,reward in [(False,True,1.),(False,True,3.),(True,True,1000.),(False,False,-1000.)]:
            a=agent.select_action(obs)
            agent.buffer.add(obs,a.action,reward,obs,True,a.log_prob,a.value,np.ones(2),intervention,a.action,actor_valid=valid,pre_tanh_action=a.pre_tanh_action)
        with patch('rl.ppoAgent.NUM_OPTIMIZATION_EPOCHS',0), patch('rl.ppoAgent.NORMALIZE_PPO_ADVANTAGES',True), patch.object(agent.buffer,'clear'):
            metrics=agent.update()
        np.testing.assert_allclose(agent.buffer.advantages[:2],[-1.,1.],atol=1e-6)
        self.assertGreater(agent.buffer.advantages[2],900.)
        self.assertLess(agent.buffer.advantages[3],-900.)
        self.assertEqual(metrics.ppo_samples,2)
        self.assertTrue(np.isfinite(metrics.raw_advantage_mean))

    def test_accepted_policy_respects_full_rollout_kl_bound(self):
        agent=PPOAgent(device='cpu',seed=0)
        for i in range(128):
            obs=np.full(27,i/128,np.float32)
            a=agent.select_action(obs)
            agent.store_transition(obs,a,float(i%7),obs,True,{'control_applied':True})
        states=torch.tensor(np.array(agent.buffer.states))
        with torch.no_grad():
            old=agent.actor.distribution(states)
            old=torch.distributions.Normal(old.loc.clone(),old.scale.clone())
        metrics=agent.update()
        with torch.no_grad():
            measured=float(torch.distributions.kl_divergence(old,agent.actor.distribution(states)).sum(-1).mean())
        self.assertLessEqual(measured,0.03)
        self.assertAlmostEqual(measured,metrics.policy_kl,places=6)
        self.assertGreater(metrics.optimizer_steps,0)
        self.assertTrue(np.isfinite([metrics.std_speed,metrics.std_heading]).all())


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
