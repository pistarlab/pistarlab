from ray import tune
import ray
from ray.tune.schedulers import AsyncHyperBandScheduler
from pistarlab import ctx, Agent, Task
from pistarlab.session import RLSession
from pistarlab.session_config import RLSessionConfig
import time
import logging
import numpy as np
from pistarlab.task import Task, AgentTask
from pistarlab.task_runner import TaskRunner
from pistarlab.utils.pyson import pyson
from pistarlab.dbmodels import TaskModel, EnvSpecModel
import json
from pistarlab.utils.env_helpers import get_wrapped_env_instance
from ray.rllib.env import EnvContext

from ray.tune.registry import register_env
import traceback

import copy
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents import trainer
from ray.rllib.agents.trainer import Trainer, COMMON_CONFIG
from pistarlab.utils.env_helpers import render_env_frame_as_image

from pistarlab.meta import *
from pistarlab.util_funcs import *

import gym

from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.a3c import a3c
from ray.rllib.agents.a3c import a2c

TRAINER_LOOKUP = {
    "PPO": {
        'class': ppo.PPOTrainer,
        'config': {}},
    "A3C": {
        'class': a3c.A3CTrainer,
        'config': {}},
    "A2C": {
        'class': a2c.A2CTrainer,
        'config': {}},
}

class DummyEnv(gym.Env):

    def __init__(self, observation_space, action_space, render_func):
        self.observation_space = observation_space
        self.action_space = action_space
        self.render_func = render_func

    def reset(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def render(self, *args):
        return self.render_func(*args)

# TODO: Should this be an actual session so we can provide a consistant view of a multiagent session.
#  furthermore,  there are different ways a multi agent session can be created (RLLIb vs other)
class MultiAgentSessionEnvWrapper(MultiAgentEnv):
    """
    Intended to be compatible with RLLib MultiAgentEnv
    see: https://github.com/ray-project/ray/blob/master/rllib/env/multi_agent_env.py
       and https://github.com/ray-project/ray/blob/master/rllib/examples/env/multi_agent.py
    """

    def __init__(
            self,
            env_spec_id,
            env_kwargs={},
            config: RLSessionConfig = None,
            task_id=None,
            vector_index=None,
            worker_index=None,
            agent_data_map={},
            player_to_agent_map={},
            env=None,):
        super().__init__()
        self.env_spec_id = env_spec_id
        self.env_kwargs = env_kwargs
        self.config = config or RLSessionConfig()
        self.running = True
        self.task_id = task_id
        self.task = Task.load(task_id)

        env_spec_model = self.get_spec_env_dbmodel()

        env_entry_point = env_spec_model.entry_point

        self.vector_index = vector_index
        self.worker_index = worker_index

        if env is None:
            env = get_wrapped_env_instance(env_entry_point, self.env_kwargs, self.config.wrappers)

        
        self.env = env

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.player_to_agent_map = player_to_agent_map
        self.agent_data_map = agent_data_map

        self.step_counter = 0
        self.episode_counter = 0
        self.steps_in_episode_counter = 0

        self.sessions = {}


        # TODO: Should wrap separately for each agent

        for agent_idx in self.env.agents:
            self.new_session(agent_idx)

    def get_spec_env_dbmodel(self) -> EnvSpecModel:
        return ctx.get_dbsession().query(EnvSpecModel).get(self.env_spec_id)

    def get_task(self):
        return self.task

    def new_session(self, player_id):
        agent_id = self.player_to_agent_map[player_id]
        agent_data = self.agent_data_map[agent_id]

        session = RLSession(
            config=RLSessionConfig(**agent_data['session_config']),
            task_id=self.task_id,
            agent_id=agent_id,
            agent_run_config=agent_data['run_config'],
            vector_index=self.vector_index,
            worker_index=self.worker_index,
            env_spec_id=self.env_spec_id,
            session_type=RL_MULTIPLAYER_MULTIAGENT_SESS,
            env=DummyEnv(self.env.observation_space, self.env.action_space, render_func=lambda *args: self.env.render(*args)))
        self.sessions[player_id] = session

    def reset(self) -> DataDict:
        for session in self.sessions.values():
            session.before_reset()

        ob_dict = self.env.reset()
        for id_, ob in ob_dict.items():
            self.sessions[id_].after_reset(ob)
        if self.step_counter > 0 :
            self.episode_counter +=1

        self.steps_in_episode_counter = 0
        
        return ob_dict

    def step(self, action_dict: DataDict) -> Tuple[DataDict, DataDict, DataDict, DataDict]:
        for k, action in action_dict.items():
            self.sessions[k].before_step(action)

        obs, rews, dones, infos = self.env.step(action_dict)

        for k, action in action_dict.items():
            self.sessions[k].after_step(obs[k], rews[k], dones[k], infos[k], action)
        self.step_counter += 1
        self.steps_in_episode_counter+=1
        if self.config.max_steps_in_episode is not None and self.steps_in_episode_counter >= self.config.max_steps_in_episode:
            dones={k:True for k,v in dones.items()}
        return obs, rews, dones, infos

    def close(self):
        for session in self.sessions.values():
            session.close()

    def with_agent_groups(self, **kwargs):
        return self.env.with_agent_groups(**kwargs)


def multi_agent_env_creator(env_ctx: EnvContext):
    ctx.connect()
    task_id = env_ctx['task_id']
    session_config = env_ctx['session_config']
    env_spec_id = env_ctx['env_spec_id']
    env_kwargs = env_ctx['env_kwargs']
    player_to_agent_map = env_ctx['player_to_agent_map']
    agent_data_map = env_ctx['agent_data_map']

    worker_index = env_ctx.worker_index
    vector_index = env_ctx.vector_index
    session_config = RLSessionConfig(**session_config)

    try:
        return MultiAgentSessionEnvWrapper(
            env_spec_id=env_spec_id,
            env_kwargs=env_kwargs,
            config=session_config,
            player_to_agent_map=player_to_agent_map,
            agent_data_map=agent_data_map,
            env=None,
            task_id=task_id,
            vector_index=vector_index,
            worker_index=worker_index)

    except Exception as e:
        logging.error("Exception when starting session session uid {}, exception: {}\n {}".format("??", e, traceback.format_exc()))


class RLLibMultiAgentRunner(TaskRunner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        task = self.get_task()
        config = task.get_config()
        var_lookup = {}
        self.config = pyson(config, var_lookup=var_lookup)

        env_spec_id = self.config['env_spec_id']

        trainer_cls: type = TRAINER_LOOKUP[self.config['trainer_class']]['class']
        trainer_config = self.config['trainer_config']
        agent_assignments = self.config['agent_assignments']
        session_config = RLSessionConfig().get_dict()
        session_config.update(self.config['session_config'])

        agent_data_map = self.config['agent_data_map']

        env_spec = ctx.get_env_spec(env_spec_id)

        env_kwargs = self.config.get('env_kwargs', {})

        # TODO: check env spec type  (SOLO,AEC,RLLIB_MULTI) and apply necessary wrappers
        # Get Spaces from agent
        wrappers = config['session_config']['wrappers']
        env_instance = get_wrapped_env_instance(
            env_spec.entry_point,
            env_kwargs,
            wrappers=wrappers)
        env_instance.reset()
        # all agents must have same space due to limitation in RLLIB
        observation_space = env_instance.observation_space
        action_space = env_instance.action_space

        policies_config = {}
        player_to_agent_map = {}
        from .agent import RLlibAgent
        agent_states = {}
        self.agents = {}
        for agent_id, agent_data in agent_data_map.items():
            agent: RLlibAgent = Agent.load(agent_id)
            self.agents[agent_id] = agent
            agent_state = agent.get_state()
            if agent_state is not None:
                state = agent.get_state()
                if state is not None:
                    agent_states[agent_id] = state
            agent_run_config = agent.get_config(agent_data['run_config'])

            if 'session_config' not in agent_data or len(agent_data['session_config']) == 0:
                agent_data['session_config'] = RLSessionConfig().get_dict()

            player_id = agent_assignments[agent_id]
            player_to_agent_map[player_id] = agent_id

            policies_config[agent_id] = (agent.get_policy_class(), observation_space, action_space, agent_run_config)

        def policy_mapping_fn(x): return player_to_agent_map[x]

        multiagent_config = {
            'policies': policies_config,
            'policy_mapping_fn': policy_mapping_fn,
            # Optional list of policies to train, or None for all policies.
            "policies_to_train": None,
            # Optional function that can be used to enhance the local agent
            # observations to include more state.
            # See rllib/evaluation/observation_function.py for more info.
            "observation_fn": None,
            # When replay_mode=lockstep, RLlib will replay all the agent
            # transitions at a particular timestep together in a batch. This allows
            # the policy to implement differentiable shared computations between
            # agents it controls at that timestep. When replay_mode=independent,
            # transitions are replayed independently per policy.
            "replay_mode": "independent",
        }

        register_env("RLLibMultaAgentSession", multi_agent_env_creator)
        # env_config will be passed to the session
        env_config = {}
        env_config['task_id'] = task.get_id()
        env_config['session_config'] = session_config
        env_config['env_spec_id'] = env_spec_id
        env_config['env_kwargs'] = env_kwargs
        env_config['agent_data_map'] = agent_data_map
        env_config['player_to_agent_map'] = player_to_agent_map

        trainer_config = {}
        trainer_config['env'] = "RLLibMultaAgentSession"
        trainer_config['multiagent'] = multiagent_config
        trainer_config['env_config'] = env_config
        # trainer_config['callbacks'] = make_callbacks(task_id)
        trainer_config['num_workers'] = 0
        trainer_config['num_envs_per_worker'] = 1

        # task.store_data('trainer_config',trainer_config)

        self.trainer: Trainer = trainer_cls(trainer_config)

        for agent_id, agent in self.agents.items():
            agent.update_space_config(self.trainer,policy_id = agent_id)


        # TODO: fix
        if (len(agent_states)) > 0:
            self.trainer.set_weights(agent_states)

        self.max_episodes = session_config['max_episodes']
        self.max_steps = session_config['max_steps']
        self.status_check_freq_secs = 2

    def run(self):

        task = self.get_task()
        task_id = task.get_id()
        max_timesteps = self.max_steps

        last_check = 0
        train_summary = None
        checkpoint_freq = 2000
        status_check_freq_secs = 2
        train_summary = None
        exit_state = None
        try:
            done = False
            terminated = False
            i = 0
            while not done:
                task.get_logger().info("Training Iteration {} - {}".format(task_id, i))

                # GET TRAINING SUMMARY
                train_summary = copy.copy(self.trainer.train())
                timestep_count = 0
                episode_count = 0
                done = train_summary.get('done', False)
                if train_summary is None:
                    done = True
                    task.get_logger().info("Training Summary is Null, Terminating work for {}".format(task_id))
                else:
                    train_summary.pop('config')
                    ctx.get_store().save(key=(TASK_ENTITY, task_id), name="train_summary", value=train_summary)
                    timestep_count = train_summary.get('timesteps_total', 0)
                    episode_count = train_summary.get('episodes_total', 0)
                    task.get_logger().info(
                        f"""\t
                        Training iteration complete {task_id} - {i}, 
                        total_timesteps: {timestep_count}, 
                        episodes_total:{episode_count}""")

                # Check for terminination request
                if not done and (time.time() - last_check > status_check_freq_secs):
                    if task.get_status() != STATE_RUNNING:
                        done = True
                        terminated = True
                        task.get_logger().info("Terminating work for {} as requested".format(task_id))
                    last_check = time.time()

                # Check timestep and episode limits
                if not done and timestep_count and max_timesteps is not None and timestep_count >= max_timesteps:
                    done = True
                    task.get_logger().info("Max Steps Reached for {}".format(task_id))
                elif not done and self.max_episodes and episode_count >= self.max_episodes:
                    done = True
                    task.get_logger().info("Max Episodes Reached for {}".format(task_id))

                # make checkpoint if needed
                if done or (i + 1) % checkpoint_freq == 0:
                    checkpoint_data = {}
                    checkpoint_data['timesteps_total'] = timestep_count
                    checkpoint_data['episodes_total'] = episode_count
                    checkpoint_data[F_TIMESTAMP] = time.time()
                    for agent_id, weights in self.trainer.get_weights().items():
                        self.agents[agent_id].save_state(weights, checkpoint_data)

                    self.task.get_logger().info("Saved agent state: {}".format(checkpoint_data))
                task.update_summary(train_summary)
                i += 1

            if terminated:
                exit_state= STATE_TERMINATED
            else:
                exit_state = STATE_COMPLETED


        finally:
            self.trainer.stop()

        task.update_summary(train_summary)


        if terminated:
            exit_state = STATE_TERMINATED
        else:
            exit_state = STATE_COMPLETED
            
        return train_summary, exit_state
