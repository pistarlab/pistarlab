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

from ..config import AgentConfig, RewardConfig
from ..core import SysContext
from ..agent import Agent, StepperAgent
from ..common import Observation, Action

from ..utils.misc import gen_uid
from ..env import Environment
class ReinforceV2AgentConfig(AgentConfig):


    def __init__(
        self,
        discount_factor=0.99,
        learning_rate = 0.000001,
        hidden_nodes=128,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.discount_factor = discount_factor
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate


class ReinforceV2Agent(StepperAgent):

   
    @staticmethod
    def instance_from_config( config: ReinforceV2AgentConfig, uid=None, **kwargs):
        if uid is None:
            uid = gen_uid()

        ob_transformer = build_observation_transformer(config.observation_space,for_cnn=False)
        action_transformer = build_action_transformer(config.action_space)

        policy_estimator = reinforce_mlp_v2.ReinforcePolicyEstimator(            
            input_size= ob_transformer.get_shape(),
            action_num=action_transformer.get_shape(),
            learning_rate= config.learning_rate,
            hidden_nodes=config.hidden_nodes,
            uid = uid)

        return ReinforceV2Agent(
            config=config,
            ob_transformer=ob_transformer,
            action_transformer=action_transformer,
            policy_estimator=policy_estimator,
            uid=uid,
            **kwargs)

    def __init__(
        self,
        ob_transformer:Transformer,
        action_transformer:Transformer,
        policy_estimator: PolicyEstimator,
       **kwargs
    ):
        super().__init__(**kwargs)

        self.ob_transformer = ob_transformer
        self.action_transformer = action_transformer
        self.policy_estimator = policy_estimator

        self.ep_data_log = []
        self.ep_counter = 0

    def start_episode(
        self, ob_data: Observation,env:Environment=None
    ) :


        self.ep_data_log=[]

        ob = self.ob_transformer.transform(ob_data.value)
        done = ob_data.done
        reward = ob_data.reward
        action, action_probs = self.policy_estimator.predict_choice(ob)

        self.ep_data_log.append({'ob':ob,'action':action,'reward':reward,'done':done})

        return Action(action)

    def step(self,  ob_data: Observation,env:Environment=None) ->Action:
        
        ob = self.ob_transformer.transform(ob_data.value)
        done = ob_data.done
        reward = ob_data.reward
        action, action_probs = self.policy_estimator.predict_choice(ob)

        self.ep_data_log.append({'ob':ob,'action':action,'reward':reward,'done':done})

        return Action(action)
            
    def end_episode(self, ob_data: Observation,env:Environment=None):
        _ = self.step(ob_data)

        T =  len(self.ep_data_log)
        
        loss_total = 0
        rewards = []
        for t in range(0,T):
            g = 0
            for k in range(t,T-1):
                g += pow(self.config.discount_factor,k-t) * self.ep_data_log[k+1]['reward']
            rewards.append(g)

        discounted_rewards = np.array(rewards) -  np.mean(rewards)
        for t in range(0,T):
            ob_in = self.ob_transformer.transform(self.ep_data_log[t]['ob'])
            action_in = self.action_transformer.transform(self.ep_data_log[t]['action'])
            loss_total  += self.policy_estimator.update(ob_in,action_in,discounted_rewards[t])
        
        loss_avg = loss_total/T

        self.log('episode_data',self.ep_counter,data={'loss':loss_avg},env=env)
        self.ep_counter+=1
        return Action(agent_step_log={'loss':loss_avg})
