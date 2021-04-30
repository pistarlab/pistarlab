import json
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from gym import spaces
from PIL import Image
import pprint

from ..models import reinforce_mlp_v2
from ..models.predictors import PolicyEstimator, ValueEstimator
from ..models.transformers import (IntToOneHotTransformer,
                                              MirrorTransformer,
                                              NpArrayFlattenTransformer,
                                              Transformer,
                                              build_action_transformer,
                                              build_observation_transformer,
                                              prep_frames)

from ..config import AgentConfig
from ..core import AgentTasker,  SysContext
from ..common import Observation, Action

from ..utils.misc import gen_uid
import logging
import random
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from modelinspector.inspector import Inspector


import copy

import os
from ..core import SessionEnv

def writer(store,data,uid,session_id,filepath):
    data_root = os.path.join("model_inspector","session_data", uid)
    store.delayed_write_by_path(
        copy.deepcopy(data),
        os.path.join(data_root, session_id , filepath))


class PolicyGradientAgentConfig(AgentConfig):



    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        discount_factor=0.99,
        learning_rate = 0.001,
        hidden_nodes=128,
        training_batch_size=1,
        entropy_beta = 0.01,
        name="PG"
    ):
        super(PolicyGradientAgentConfig, self).__init__(
            observation_space=observation_space, 
            action_space=action_space)
        self.discount_factor = discount_factor
        self.hidden_nodes = hidden_nodes
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.training_batch_size = training_batch_size
        self.entropy_beta = entropy_beta
        


class PGN(nn.Module):
    def __init__(self,hidden_nodes, input_size, n_actions):
        super(PGN,self).__init__()
        self.net= nn.Sequential(
            nn.Linear(input_size,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,n_actions))
        
    def forward(self, x):
        return self.net(x)

def calc_qvals(rewards:List[float], GAMMA:float):
    res:List[float] = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)

    return list(reversed(res))

def calc_qvals_accross_eps(reward_done_tuples,GAMMA):
    
    result_list = []
    rlist = []
    for r,d in reward_done_tuples:
        rlist.append(r)
        if d is True:
            result_list.extend(calc_qvals(rlist,GAMMA))
            rlist = []

    if len(rlist) >0:
        result_list.extend(calc_qvals(rlist,GAMMA))
    return result_list


def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)

class PolicyGradientAgent(AgentTasker):
    """
    """
    metadata={
        'logged_attributes':[('episode_data','loss', float),('episode_data','policy_loss', float)],
        'description':
        """
Policy Gradient with Entropy Bonus

Source: https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter11/04_cartpole_pg.py
        """
    }

    @classmethod
    def instance_from_config(
        cls, config: PolicyGradientAgentConfig, ctx: SysContext, uid=None
    ):
        if uid is None:
            uid = gen_uid("agent")


        return PolicyGradientAgent(
            config,
            uid=uid,
            ctx=ctx)

    def __init__(
        self,
        config: PolicyGradientAgentConfig,
        uid: str,
        ctx: SysContext,
    ):
        super(PolicyGradientAgent, self).__init__(uid, config=config, ctx=ctx)


        self.net = PGN(self.config.hidden_nodes,self.config.observation_space.shape[0], self.config.action_space.n)

        self.optimizer = optim.Adam(self.net.parameters(), lr = self.config.learning_rate)
        self.inspector = Inspector(session_id = 'policy', writer=lambda data, session_id, filepath: writer(ctx.get_store(), data,uid, session_id, filepath))
    

        self.update_counter = 0
        self.reward_sum = 0.0
        self.reward_sum_counter = 0

        self.ep_counter = 0
        self.reward_rollout=10
        

    def start_episode(
        self, ob_data: Observation,env:SessionEnv=None
    ) :

        self.ep_history=[]
        self.ob_prev = None
        self.action_prev = None
        return self.step(ob_data,env = env)

    def step(self,  ob_data: Observation,env:SessionEnv=None) ->Action:
        
        ob = ob_data.value
        done = ob_data.done
        action_scores = self.net(float32_preprocessor([ob]))
        action_probs_p = F.softmax(action_scores,dim=1)
        action_probs = action_probs_p.data.cpu().numpy()
        actions = []
        for prob in action_probs:
            actions.append(np.random.choice(len(prob), p=prob))
        action = np.array(actions)[0]

        if self.ob_prev is not None:
            self.ep_history.append({
                'ob':self.ob_prev,
                'action':int(self.action_prev),
                'reward':ob_data.reward,
                'done':done})

            self.update_policy(env)

        self.ob_prev = ob
        self.action_prev = action
        
        return Action(action)


    def update_policy(self,env):
        
              
        if (self.config.training_batch_size + self.reward_rollout )> len(self.ep_history):
            return

        # Train
        self.optimizer.zero_grad()

        
        batch_ob = [v['ob'] for v in self.ep_history[:self.config.training_batch_size]]
        batch_actions = [v['action'] for v in self.ep_history[:self.config.training_batch_size]]

        #reward calc
        reward_done_tuples = [(v['reward'],v['done']) for v in self.ep_history]
        batch_qvals = calc_qvals_accross_eps(reward_done_tuples,self.config.discount_factor)
        
        assert(len(batch_qvals)== (self.config.training_batch_size + self.reward_rollout))
        
        batch_weights = []
        for qv in batch_qvals[:self.config.training_batch_size]:
            self.reward_sum += qv
            baseline = self.reward_sum / (1 + self.reward_sum_counter)
            weight = qv - baseline
            batch_weights.append(weight)
            self.reward_sum_counter+=1

        batch_ob_t = torch.FloatTensor(batch_ob)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_weights_t = torch.FloatTensor(batch_weights)

        logits_t = self.net(batch_ob_t)
        log_prob_t:Tensor = F.log_softmax(logits_t, dim=1)      
        
        log_prob_actions_t:Tensor = batch_weights_t * log_prob_t[range(self.config.training_batch_size), batch_actions_t]
        loss_policy_t = -log_prob_actions_t.mean()

        prob_t = F.softmax(logits_t, dim=1)
        entropy_t = -(prob_t * log_prob_t).sum(dim=1).mean()
        entropy_loss_t = -self.config.entropy_beta * entropy_t
        loss_t = loss_policy_t + entropy_loss_t

        loss_t.backward()
        self.optimizer.step()

        if self.update_counter % 1000 == 0:
            self.inspector.log_state(epoch=0,
                itr=self.update_counter, 
                model=self.net,
                input_dict={"input.1":batch_ob_t},
                output_dict={"output":logits_t},
                loss_dict={'loss':loss_t},
                label_dict={})
            self.inspector.log_metrics(
                    epoch=0,
                    itr=self.update_counter, 
                    metrics={'loss':loss_t.item()})
            self.update_counter+=1

        self.log('episode_data',data={'loss':loss_t.item(),'policy_loss':loss_policy_t.item()},env=env)

        self.ep_history=self.ep_history[self.config.training_batch_size:]

            
    def end_episode(self, ob_data: Observation,env:SessionEnv=None):
        _ = self.step(ob_data)

        self.ep_counter+=1