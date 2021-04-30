import json
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from gym import spaces
from PIL import Image
import pprint


from ..config import AgentConfig,  RewardConfig
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

from modelinspector.inspector import Inspector

from ..core import SessionEnv
import copy

import os

def writer(store,data,uid,session_id,filepath):
    data_root = os.path.join("model_inspector","session_data", uid)
    store.delayed_write_by_path(
        copy.deepcopy(data),
        os.path.join(data_root, session_id , filepath))


class ReinforceV3AgentConfig(AgentConfig):


    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        reward_config: RewardConfig,
        discount_factor=0.99,
        learning_rate = 0.0001,
        hidden_nodes=128,
        training_batch_size=1,
        use_baseline=False,
        name="Reinforce_v3"
    ):
        super(ReinforceV3AgentConfig, self).__init__(name=name,
            observation_space=observation_space, 
            action_space=action_space,
            reward_config=reward_config)
        self.discount_factor = discount_factor
        self.hidden_nodes = hidden_nodes
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_config = reward_config
        self.learning_rate = learning_rate
        self.training_batch_size = training_batch_size
        self.use_baseline = use_baseline


class PGN(nn.Module):
    def __init__(self,hidden_nodes, input_size, n_actions):
        super(PGN,self).__init__()
        self.net= nn.Sequential(
            nn.Linear(input_size,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,n_actions))
        
    def forward(self, x):
        return self.net(x)



def calc_qvals(rewards, GAMMA):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)

    return list(reversed(res))

def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)

class ReinforceV3Agent(AgentTasker):
    """
    https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter11/02_cartpole_reinforce.py
    """
    

    metadata={'logged_attributes':[('episode_data','loss', float)]}


    @classmethod
    def instance_from_config(
        cls, config: ReinforceV3AgentConfig, ctx: SysContext, uid=None
    ):
        if uid is None:
            uid = gen_uid("agent")


        return ReinforceV3Agent(
            config,
            uid=uid,
            ctx=ctx)

    def __init__(
        self,
        config: ReinforceV3AgentConfig,
        uid: str,
        ctx: SysContext,
    ):
        super(ReinforceV3Agent, self).__init__(uid, config=config, ctx=ctx)


        self.net = PGN(self.config.hidden_nodes,self.config.observation_space.shape[0], self.config.action_space.n)

        self.optimizer = optim.Adam(self.net.parameters(), lr = self.config.learning_rate)
        self.inspector = Inspector(session_id = 'policy', writer=lambda data, session_id, filepath: writer(ctx.get_store(), data,uid, session_id, filepath))
    

        self.history_batch= []

        self.update_counter = 0
        self.ep_counter =0


    def start_episode(
        self, ob_data: Observation,env:SessionEnv=None
    ) :

        self.ep_history=[]
        return self.step(ob_data,env = env)

    def step(self,  ob_data: Observation,env:SessionEnv=None) ->Action:
        
        ob = ob_data.value
        done = ob_data.done
        reward = ob_data.reward


        action_scores = self.net(float32_preprocessor([ob]))
        action_probs_p = F.softmax(action_scores,dim=1)
        action_probs = action_probs_p.data.cpu().numpy()
        actions = []
        for prob in action_probs:
            actions.append(np.random.choice(len(prob), p=prob))
        action = np.array(actions)[0]

        self.ep_history.append({'ob':ob,'action':int(action),'reward':reward,'done':done})


        return Action(action)
            
    def end_episode(self, ob_data: Observation,env:SessionEnv=None):
        _ = self.step(ob_data)

        self.history_batch.append(self.ep_history)
              
        if self.config.training_batch_size > len(self.history_batch):
            return
        
        # Train
        self.optimizer.zero_grad()

        # align such that ob  a => r + (ob':optionaly). s.t. p(r|ob,a)
        batch_ob = []
        batch_actions = []
        batch_qvals = []

        for history in self.history_batch:
            batch_ob.extend([v['ob'] for v in history][:-1])
            batch_actions.extend([v['action'] for v in history][:-1])
            qvals = calc_qvals( [v['reward'] for v in history][1:],self.config.discount_factor)
            if self.config.use_baseline:
                qvals = qvals - np.mean(qvals)
            batch_qvals.extend(qvals)

        batch_ob_t = torch.FloatTensor(batch_ob)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_t = torch.FloatTensor(batch_qvals)

        logits_t = self.net(batch_ob_t)
        log_prob_t = F.log_softmax(logits_t, dim=1)      
        
        log_prob_actions_t = batch_qvals_t * log_prob_t[range(len(batch_ob_t)), batch_actions_t]
        loss_t = -log_prob_actions_t.mean()
        loss_t.backward()
        self.optimizer.step()

        self.history_batch= []

        if self.update_counter % 300 == 0:
            self.inspector.log_state(epoch=0,
                itr=self.update_counter, 
                model=self.net,
                input_dict={"input.1":batch_ob_t},
                output_dict={"output":logits_t},
                loss_dict={'loss':loss_t},
                name_dict={})
            self.inspector.log_metrics(
                    epoch=0,
                    itr=self.update_counter, 
                    metrics={'loss':loss_t.item()})
            self.update_counter+=1

        self.log('episode_data',self.ep_counter,data={'loss':loss_t.item()},env=env)
        self.ep_counter+=1





