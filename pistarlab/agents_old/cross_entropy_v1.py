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
from ..config import AgentConfig, EnvironmentConfig, RewardConfig
from ..core import SysContext
from ..common import Action,Observation
from ..utils.misc import gen_uid
from ..utils.timer import Timer
from ..env import Environment
from ..agent import StepperAgent

class CrossEntropyAgentConfig(AgentConfig):


    def __init__(
        self,
        learning_rate = 0.0001,
        hidden_nodes=128,
        print_step_freq=1000,
        batch_size=16,
        train_percentile = 70,
        **kwargs):
        super().__init__(**kwargs)

        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_percentile = train_percentile
        self.notes = """
Derived from : https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter04/01_cartpole.py
        """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):
    def __init__(self,input_size,action_size,hidden_nodes):
        super(Model,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,action_size))
    
    def forward(self,x):
        return self.net(x)


#converts to box
class Box1DSpaceTransformer:

    def __init__(self, source_space):
        self.source_space = source_space
        self.space = source_space
        self.transformer_func = lambda x: x

        if isinstance(source_space,spaces.Discrete):
            l = self.source_space.n                          
            def func(x):
                a = np.zeros(l)
                a[x] = 1
                return a
            self.space = spaces.Box(0,1,shape=(l,), dtype=np.float32)
            self.transformer_func = func
        elif isinstance(source_space,spaces.Box) and len(source_space.shape) >1:
            self.transformer_func = lambda x: x.ravel()

        
        self.transformer_batch_func = np.vectorize(self.transformer_func)
        

    def get_size(self):
        return self.space.shape[0]
    
    def transform(self,x):
        return self.transformer_func(x)
    
    def transform_batch(self,xbatch):
        return self.transformer_batch_func(xbatch)

class CrossEntropyAgent(StepperAgent):

    @staticmethod
    def instance_from_config(**kwargs):
        return CrossEntropyAgent(**kwargs)

    def __init__(
        self,**kwargs):
        super().__init__(**kwargs)

        self.ob_transformer = Box1DSpaceTransformer(self.config.observation_space)

        action_size = self.config.action_space.n

        # Service Objects (probably should not serialized as is)
        self.model = Model(
            input_size = self.ob_transformer.get_size(),
            action_size = action_size,
            hidden_nodes = self.config.hidden_nodes)

        self.objective = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.config.learning_rate)
        self.sm = sm = nn.Softmax(dim=1)

        self.ep_data_log = []
        self.episode_batch = []

    def start_episode(self, ob: Observation,env:Environment=None) -> Action:

        self.ep_data_log=[]
        self.episode_reward = 0

        return self.step(ob)

    def step(self, ob: Observation,env:Environment=None) -> Action:
        done = ob.done
        reward = ob.reward
        if reward:
            self.episode_reward += reward
        obv = self.ob_transformer.transform(ob.value)
        action_probs = self.sm(self.model(torch.FloatTensor([obv])))
        action_probs_np = action_probs.data.numpy()[0]

        action = np.random.choice(len(action_probs_np), p=action_probs_np)

        self.ep_data_log.append({'ob':obv,'action':action,'reward':reward,'done':done})

        return Action(action)
            
    def end_episode(self, ob: Observation,env:Environment=None):
        _ = self.step(ob,env)
        
        # Add episode record to list
        self.episode_batch.append((self.episode_reward,self.ep_data_log))

       
        rewards = list(map(lambda s: s[0], self.episode_batch))
        reward_bound = np.percentile(rewards, self.config.train_percentile)
        reward_mean = float(np.mean(rewards))
        
        train_obs = []
        train_act = []
        for reward, steps in self.episode_batch:
            if reward < reward_bound:
                continue
            train_obs.extend(map(lambda step: step['ob'], steps))
            train_act.extend(map(lambda step: step['action'], steps))

        train_obs_v = torch.FloatTensor(train_obs)
        train_act_v = torch.LongTensor(train_act)

        self.optimizer.zero_grad()
        action_scores_v = self.model(train_obs_v)
        loss_v = self.objective(action_scores_v, train_act_v)
        loss_v.backward()
        self.optimizer.step()
        self.logger.info("Loss:{}".format(loss_v))

        return Action(agent_step_log={'loss':loss_v.item()})


