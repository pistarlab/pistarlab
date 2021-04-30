import json
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from gym import spaces
from PIL import Image
import pprint

from ..agents.models import (predictors_shared, predictors_shared_pyt,
                                 predictors_tf)
from ..agents.models.predictors import PolicyEstimator, ValueEstimator
from ..agents.models.transformers import (IntToOneHotTransformer,
                                              MirrorTransformer,
                                              NpArrayFlattenTransformer,
                                              Transformer,
                                              build_action_transformer,
                                              build_observation_transformer,
                                              prep_frames)
from ..common import (DataTypeCategory, EpisodeStatus, GoalType,
                             RewardType)
from ..config import BaseAgentConfig, EnvironmentConfig, RewardConfig
from ..core import Action, BaseAgent, Environment, Observation
from .memory import Memory
from ..utils.misc import gen_uid
from ..utils.timer import Timer

class A2CAgentConfig(BaseAgentConfig):

    @classmethod
    def instance_from_env(cls, env: Environment):
        return A2CAgentConfig(
            observation_space=env.config.observation_space,
            action_space=env.config.action_space,
            reward_config=env.config.reward_config
        )

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        reward_config: RewardConfig,
        discount_factor=0.99,
        hidden_nodes=50,
        smoothing_window_size=100,
        print_step_freq=1000,
        ob_frames=1,
        learn=True
    ):
        super(A2CAgentConfig, self).__init__(name="A2C_CNN")
        self.ob_frames = ob_frames

        self.discount_factor = discount_factor
        self.hidden_nodes = hidden_nodes
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_config = RewardConfig


        # TODO, add multiple options for sampling, should this be session specific?  probably, yes!
        self.learn = learn


class A2CAgent(BaseAgent):
    
    @classmethod
    def instance_from_config(
        cls, config: A2CAgentConfig, store: Storage, uid=None
    ):
        if uid is None:
            uid = gen_uid("agent")

        # TODO:  Confirm configuraiton here (is it compatable)

        ob_transformer = build_observation_transformer(config.observation_space,for_cnn=True)
        action_transformer = build_action_transformer(config.action_space)

        ob_input_shape = ob_transformer.get_shape()

        if len(ob_input_shape) != 3:
            raise Exception("Error expected 3 dimentions, actual {}".format(len(ob_input_shape)))

        ob_input_shape = (ob_input_shape[0], ob_input_shape[1] * config.ob_frames,ob_input_shape[2])

        model = predictors_shared_pyt.SharedPolicyModel(
            ob_input_shape=ob_input_shape,
            action_output_shape=action_transformer.get_shape(),
            hidden_nodes=config.hidden_nodes,
            uid=uid,
            store = store,
            log_state_freq = 2000)

        policy_estimator = predictors_shared_pyt.PolicyEstimatorShared(model)

        value_estimator = predictors_shared_pyt.ValueEstimatorShared(model)



        return A2CAgent(
            config,
            ob_transformer=ob_transformer,
            action_transformer=action_transformer,
            policy_estimator=policy_estimator,
            value_estimator=value_estimator,
            uid=uid,
            store=store)

    def __init__(
        self,
        config: A2CAgentConfig,
        ob_transformer:Transformer,
        action_transformer:Transformer,
        policy_estimator: PolicyEstimator,
        value_estimator: ValueEstimator,
        uid: str,
        store: Storage,
    ):
        super(A2CAgent, self).__init__(uid, config=config, store=store)
        """

        """
        # Service Objects (probably should not serialized as is)
        self.ob_transformer = ob_transformer
        self.action_transformer = action_transformer
        self.policy_estimator = policy_estimator
        self.value_estimator = value_estimator

        # internal state
        self.ob_prev = None

        # One Entry Per Epiosde
        self.reward_log = []
        self.reward_last_log = []

        self.__learn = self.config.learn
        self.__checkpoint_counter = 0
        self.ep_data_log = []

        # CONFIGURE
        self.p_loss = None
        self.v_loss = None

    def set_learn(self, learn):
        self.__learn = learn

    def get_ob_input(self):
        return prep_frames(
            input_frames = self.memory.get_last_n_steps_in_episode( self.config.ob_frames, list_name='obs'),
            shape = self.ob_transformer.get_shape(),
            expected_frame_count = self.config.ob_frames,
            transformer=self.ob_transformer)

    def start_episode(
        self, session: Session, input_data: Observation
    ) -> Tuple[Session, Action]:
        """
        Initialize Episode
        :param session:
        :param input_data:
        :return:
        """

        self.ep_data_log={}
        self.memory = Memory()

        self.memory.add_entry(
            action_value=None,
            ob_value=self.ob_transformer.transform(input_data.ob),
            reward_value=input_data.reward.value,
            terminal_value=input_data.done)
        
        self.ob = self.get_ob_input()
        self.action, self.action_probs = self.policy_estimator.predict_choice(self.ob)
        self.ob_prev = self.ob


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

        self.log_step("Start",session)

        return session, Action(self.action)

    def step(self, session: Session, input_data: Observation) -> Tuple[Session, Action]:

        self.memory.add_entry(
            action_value=self.action,
            ob_value=self.ob_transformer.transform(input_data.ob),
            reward_value=input_data.reward.value,
            terminal_value=input_data.done)
        
        self.ob = self.get_ob_input()

        self.reward_prev = input_data.reward.value
        self.done = input_data.done
        self.value_pred = self.value_estimator.predict(self.ob)

        self.value_pred_prev = self.value_estimator.predict(self.ob_prev)

        # Calculate Advantage and target reward for training
        if self.done:
            # Reward
            self.td_target = self.reward_prev
            # Advantage
            self.td_error = self.reward_prev + self.value_pred_prev
        else:
            self.td_target = self.reward_prev + (self.config.discount_factor * self.value_pred)

            self.td_error = self.reward_prev + (self.config.discount_factor * self.value_pred) - self.value_pred_prev

        # Perform Value and Policy Function Update, TODO: does this order matter?
        if self.__learn:
            self.v_loss = self.value_estimator.update(self.ob_prev, self.td_target)

            self.p_loss = self.policy_estimator.update(
                self.ob_prev, self.td_error, self.action_transformer.transform(self.action))
            
        # Select an action
        try:
            self.action, self.action_probs = self.policy_estimator.predict_choice(self.ob)
        except Exception as e:
            self.log_step("Exception Occurred, printing last log",session,force_log=True)
            raise e

        self.ob_prev = self.ob

        self.log_step(" - ",session)

        return session, Action(self.action)

    def log_step(self,console_message,session,force_log=False):
        ep_data = {}
        ep_data["step_reward_used"] = self.reward_prev
        ep_data["step_value_pred"] = self.value_pred
        ep_data["step_value_pred_prev"] = self.value_pred_prev
        ep_data["step_action"] = self.action
        ep_data["step_td_error"] = self.td_error
        ep_data["step_td_target"] = self.td_target
        ep_data["step_policy_loss"] = self.p_loss
        ep_data["step_value_loss"] = self.v_loss

        # Load into ep_data_log
        for k,v in ep_data.items():
            self.ep_data_log[k] = self.ep_data_log.get(k,[]) + [v]

        # Log on episode step
        if force_log or ((session.episode_step_counter % self.config.print_step_freq) == 0):
            logging.info(
                "{}, Epsiode:{} at step: {}, Action:{}, Reward: {}, Pred Value: {}, PLoss: {}, VLoss: {}, SessionStep: {}".format(
                    console_message,
                    session.episode_counter,
                    session.episode_step_counter,
                    self.action,
                    self.reward_prev,
                    self.value_pred,
                    self.p_loss if self.__learn else "NA",
                    self.v_loss if self.__learn else "NA",
                    session.session_step_counter))                    

        # Log everything for episode
        if session.episode_counter % session.config.episode_record_freq == 0:            
            self.store.log_episode_data(
                session_id=session.get_uid(),
                episode_id=session.episode_counter,
                step_id=session.episode_step_counter,
                value=ep_data)
            
    def end_episode(self, session: Session) -> Session:
        """
        Wrap up episode
        :param session:
        :return:
        """

        rewards = self.memory.get_last_n_steps_in_episode(list_name='rewards')[1:]

        episode_len = len(rewards)

        # Mean reward over **current** episode
        reward_mean = (sum(rewards) - 1) / (episode_len + 1)
        self.reward_log.append(reward_mean)

        # Mean of reward means from multiple episodes (above)
        head_rewards = self.reward_log[-self.config.smoothing_window_size:]

        # Mean of last reward over **muliple** episodes
        self.reward_last_log.append(self.reward_prev)
        head_rewards_last = self.reward_last_log[-self.config.smoothing_window_size:]

        # Log
        data = {}
        data["episode_length"] = episode_len
        data["last_reward"] = self.reward_prev
        data["reward_mean"] = reward_mean
        data["head_rewards_mean"] = np.mean(head_rewards)
        data["head_rewards_last_mean"] = np.mean(head_rewards_last)
        
        for k,v in self.ep_data_log.items():
            v = [d for d in v if d is not None]
            data["{}__{}".format(k,'mean')]=np.mean(v)
            data["{}__{}".format(k,'median')]=np.median(v)
            data["{}__{}".format(k,'min')]=np.min(v)
            data["{}__{}".format(k,'max')]=np.max(v)
            data["{}__{}".format(k,'sum')]=np.sum(v)
            data["{}__{}".format(k,'last_first_diff')]=v[-1] - v[0]
 
        self.store.log_session_data(
            session_id = session.get_uid(), 
            step_id = session.episode_counter, 
            value=data)

        logging.info("Episode Complete")
            
        return session

