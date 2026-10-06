import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from rl.actor import Actor
from rl.ppoAgent import PPOAgent
from rl.rolloutBuffer import RolloutBuffer
from main import mainLearning as learning


class TrainingV5Tests(unittest.TestCase):
    def test_unchanged_policy_ratio_one_even_at_saturated_actions(self):
        torch.manual_seed(0)
        actor=Actor()
        with torch.no_grad():
            for p in actor.parameters():p.zero_()
            actor.mean_head.bias.copy_(torch.tensor([20.,-20.]))
        states=torch.zeros(32,27)
        with torch.no_grad():action,old,latent=actor.sample(states,return_pre_tanh=True)
        self.assertTrue(torch.all(action.abs()==1.))
        new,_=actor.evaluate_actions(states,action,pre_tanh_actions=latent)
        self.assertTrue(torch.isfinite(new).all())
        torch.testing.assert_close(torch.exp(new-old),torch.ones(32),rtol=0,atol=0)
        (-new.mean()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters()))

    def test_probability_ratio_matches_independent_latent_gaussian_ratio(self):
        actor=Actor();states=torch.zeros(3,27)
        with torch.no_grad():
            for p in actor.parameters():p.zero_()
        latent=torch.tensor([[20.,-20.],[.2,-.3],[8.,-9.]])
        actions=torch.tanh(latent)
        old_dist=actor.distribution(states)
        old,_=actor.evaluate_actions(states,actions,pre_tanh_actions=latent)
        with torch.no_grad():actor.mean_head.bias.add_(.1)
        new,_=actor.evaluate_actions(states,actions,pre_tanh_actions=latent)
        expected=actor.distribution(states).log_prob(latent).sum(-1)-old_dist.log_prob(latent).sum(-1)
        torch.testing.assert_close(new-old,expected,atol=1e-4,rtol=1e-5)

    def test_agent_buffer_preserves_latent_sample_and_old_log_probability(self):
        agent=PPOAgent(device='cpu',seed=0)
        with torch.no_grad():agent.actor.mean_head.bias.fill_(12.)
        obs=np.zeros(27,np.float32)
        action=agent.select_action(obs)
        agent.store_transition(obs,action,1.,obs,True,{'control_applied':True})
        agent.buffer.set_advantages_and_returns(np.ones(1),np.ones(1))
        batch=agent.buffer.as_batch('cpu')
        new,_=agent.actor.evaluate_actions(batch.states,batch.actions,batch.pre_tanh_actions)
        torch.testing.assert_close(new,batch.old_log_probs,atol=0,rtol=0)
        np.testing.assert_array_equal(batch.pre_tanh_actions.numpy()[0],action.pre_tanh_action)
        agent.buffer.clear()
        self.assertEqual(agent.buffer.pre_tanh_actions,[])

    def test_missing_latent_for_saturated_action_is_refused(self):
        buffer=RolloutBuffer()
        with self.assertRaisesRegex(ValueError,'require original'):
            buffer.add(np.zeros(27),np.ones(2),1.,np.zeros(27),True,0.,0.,np.zeros(2),False,np.ones(2))

    def test_backward_compatible_sample_api(self):
        action,log_prob=Actor().sample(torch.zeros(2,27))
        self.assertEqual(action.shape,(2,2))
        self.assertEqual(log_prob.shape,(2,))

    def test_episode_updates_and_final_partial_episode_exactly_once(self):
        sizes=[];versions=[]
        class SpyAgent(PPOAgent):
            def select_action(self,*args,**kwargs):
                versions.append(self.update_count)
                return super().select_action(*args,**kwargs)
            def update(self):
                sizes.append(len(self.buffer))
                return super().update()
        class TwoStepEnv:
            def __init__(self,ego_initial_speed):self.ego_minimum_speed=ego_initial_speed
            def reset(self,seed):self.n=0;return np.zeros(27,np.float32),{}
            def step(self,action):
                self.n+=1
                return np.zeros(27,np.float32),1.,self.n==2,False,{'control_applied':True,'success':self.n==2,'termination_reason':'success' if self.n==2 else ''}
            def close(self):pass
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root=Path(directory)
            def select(agent,speed,seed,score,step):
                agent.save(root/'models/bestPolicy.pt');return score
            original_paths={name:getattr(learning,name) for name in ('RESULTS_DIR','MODELS_DIR','CHECKPOINTS_DIR','LOGS_DIR','FIGURES_DIR')}
            with patch.multiple(learning,**original_paths),patch.object(learning,'PPOAgent',SpyAgent),patch.object(learning,'MergeEnvironment',TwoStepEnv),patch.object(learning,'select_best_policy',side_effect=select),patch.object(learning,'save_training_figures'):
                learning.train(5.,7,0,None,output_dir=str(root),update_schedule='episode')
            self.assertEqual(sizes,[2,2,2,1])
            self.assertEqual(versions,[0,0,1,1,2,2,3])
            with (root/'logs/updateLog.csv').open() as f:updates=list(csv.DictReader(f))
            self.assertEqual([int(r['global_step']) for r in updates],[2,4,6,7])
            with (root/'logs/trainingLog.csv').open() as f:episodes=list(csv.DictReader(f))
            self.assertEqual([int(r['episode_complete']) for r in episodes],[1,1,1,0])
            loaded=PPOAgent(device='cpu',seed=0)
            loaded.load(root/'models/finalPolicy.pt',require_trained=True,require_compatible=True)
            self.assertEqual(loaded.update_count,4)
            metadata=torch.load(root/'models/finalPolicy.pt',map_location='cpu')['metadata']
            self.assertEqual(metadata['implementation'],'cbfg-ppo-stability-v6')
            self.assertIn('main/mainLearning.py',{k.replace(chr(92),'/') for k in metadata['source_sha256']})

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
