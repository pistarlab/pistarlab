import json
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from gym import spaces

from ..models import predictors_mlp
from ..models.predictors import PolicyEstimator, ValueEstimator
from ..models.transformers import (IntToOneHotTransformer,
                                              MirrorTransformer,
                                              NpArrayFlattenTransformer,
                                              Transformer,
                                              build_action_transformer,
                                              build_observation_transformer,
                                              prep_frames)

from ..config import AgentConfig,  
from ..core import Action, AgentTasker,  Observation, SysContext
from ..utils.misc import gen_uid
from ..utils.timer import Timer
from ..core import SessionEnv
class A2CAgentSimpleConfig(AgentConfig):



    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        discount_factor=0.99,
        hidden_nodes=50,
        learning_rate=0.00000001,
        name="A2CAgentSimple",
        log_state_freq = 1000
    ):
        super(A2CAgentSimpleConfig, self).__init__(
            observation_space=observation_space, 
            action_space=action_space)
            
        self.discount_factor = discount_factor
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate
        self.log_state_freq = log_state_freq

class A2CAgentSimple(AgentTasker):

    metadata = {
        'supported_observation_spaces':['Discrete'],
        'supported_action_spaces':['Discrete']}
    
    description = """

    """
    
    @classmethod
    def instance_from_config(
        cls, config: A2CAgentSimpleConfig, ctx: SysContext, uid=None
    ):
        uid = uid or gen_uid()

        ob_transformer = build_observation_transformer(config.observation_space,for_cnn=False)
        action_transformer = build_action_transformer(config.action_space)

        model_p = predictors_mlp.PolicyModel(            
            input_size= ob_transformer.get_shape(),
            action_num=action_transformer.get_shape(),
            hidden_nodes=config.hidden_nodes,
            learning_rate = config.learning_rate,
            uid = uid,
            store = ctx.get_store(),
            log_state_freq = config.log_state_freq)

        model_v = predictors_mlp.ValueModel(            
            input_size= ob_transformer.get_shape(),
            action_num=action_transformer.get_shape(),
            hidden_nodes=config.hidden_nodes,
            learning_rate = config.learning_rate,
            uid = uid,
            store = ctx.get_store(),
            log_state_freq = config.log_state_freq)
            
        policy_estimator = predictors_mlp.PolicyEstimatorMLP(model_p)
        value_estimator = predictors_mlp.ValueEstimatorMLP(model_v)


        return A2CAgentSimple(
            config,
            ob_transformer=ob_transformer,
            action_transformer=action_transformer,
            policy_estimator=policy_estimator,
            value_estimator=value_estimator,
            uid=uid,
            ctx=ctx)

    def __init__(
        self,
        config: A2CAgentSimpleConfig,
        ob_transformer:Transformer,
        action_transformer:Transformer,
        policy_estimator: PolicyEstimator,
        value_estimator: ValueEstimator,
        uid: str,
        ctx: SysContext,
    ):
        super(A2CAgentSimple, self).__init__(uid, config=config, ctx=ctx)

        # Service Objects (probably should not serialized as is)
        self.ob_transformer = ob_transformer
        self.action_transformer = action_transformer
        self.policy_estimator = policy_estimator
        self.value_estimator = value_estimator

        # internal state
        self.ob_prev = None

        # CONFIGURE
        self.p_loss = None
        self.v_loss = None

    def start_episode(self, ob: Observation,env:SessionEnv=None) -> Action:

        # reset these
        self.td_target = None
        self.td_error= None
        self.p_loss = None
        self.v_loss = None
        self.reward_prev= None
        self.value_pred=None
        self.value_pred_prev=None
        self.done = None
        self.action_prepped=None
        self.action = None
        self.obv_prev = None

        return self.step(ob,env)

    def step(self, ob: Observation,env:SessionEnv=None) -> Action:

        self.obv = self.ob_transformer.transform(ob.value)
        self.reward_prev = ob.reward
        self.done = ob.done
       
        self.value_pred = self.value_estimator.predict(self.obv)

        # skip for first step
        if self.obv_prev is not None:
            self.value_pred_prev = self.value_estimator.predict(self.obv_prev) # or below

            # Calculate Advantage and target reward for training
            if self.done:
                # Reward
                self.td_target = self.reward_prev
                # Advantage
                self.td_error = self.reward_prev + self.value_pred_prev
            else:
                self.td_target = self.reward_prev + (self.config.discount_factor * self.value_pred)

                self.td_error = self.reward_prev + (self.config.discount_factor * self.value_pred) - self.value_pred_prev

            # Perform Value and Policy Function Update, TODO: does this order matter for shared model?
            self.v_loss = self.value_estimator.update(self.obv_prev, self.td_target)

            self.p_loss = self.policy_estimator.update(
                self.obv_prev, self.td_error, self.action_transformer.transform(self.action))
                
        # Select an action
        self.action, self.action_probs = self.policy_estimator.predict_choice(self.obv)

        # Prep for next iteration
        self.obv_prev = self.obv

        return Action(self.action)


    def end_episode(self,ob: Observation,env:SessionEnv=None):
        _ = self.step(ob,env)
        #disregard Action

