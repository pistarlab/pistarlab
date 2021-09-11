import copy
import logging

import time
import traceback
from typing import Any, Callable, Dict, List, Tuple, Type

import ray
from pistarlab import ctx
from pistarlab.agent import Agent
from pistarlab.meta import *
from pistarlab.session_config import RLSessionConfig
from pistarlab.session_env import RLMultiSessionEnv
from pistarlab.task import AgentTask, Task
from pistarlab.task_runner import AgentTaskRunner
from pistarlab.util_funcs import merged_dict
from pistarlab.utils.env_helpers import get_wrapped_env_instance
from pistarlab.utils.param_helpers import create_params_from_dict
from pistarlab.utils.pyson import pyson
from pistarlab.utils.serialize import space_to_pyson
from ray.rllib.agents import trainer
from ray.rllib.agents.a3c import a2c, a3c
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTorchPolicy, ppo
from ray.rllib.agents.sac import SACTorchPolicy, sac
from ray.rllib.agents.trainer import COMMON_CONFIG, Trainer
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.env import BaseEnv, EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.tune.registry import register_env

from .ac_policy import PistarACTorchPolicy
from .random_policy import RandomPolicy

filter_list = ['callbacks']

randomPolicyTrainer = build_trainer(
    name="RandomPolicyTrainer",
    default_policy=RandomPolicy)


DEFAULT_AGENT_PARAMS = {
    "trainer_config.num_workers": {
        "displayed_name": "Number of Workers",
        "description": "Number of rollout worker actors to create for parallel sampling. Setting this to 0 will force rollouts to be done in the trainer actor.",
        "path": "trainer_config.num_workers",
        "data_type": "int",
        "mode": "default",
        "interfaces": ["run"],
        "type_info": {

        }
    },
    "trainer_config.num_envs_per_worker": {
        "displayed_name": "Number of Environments per Worker",
        "description": "When num_workers > 0, the driver (local_worker; worker-idx=0) does not need an environment. This is because it doesn't have to sample (done by remote_workers; worker_indices > 0) nor evaluate (done by evaluation workers; see below).",
        "path": "trainer_config.num_envs_per_worker",
        "data_type": "int",
        "mode": "default",
        "interfaces": ["run"],
        "type_info": {

        }
    },
    "trainer_config.gamma": {
        "displayed_name": "Gamma",
        "description": "",
        "path": "trainer_config.gamma",
        "data_type": "float",
        "mode": "default",
        "interfaces": ["run"],
        "type_info": {
            "min": 0.1,
            "max": 1.0
        }
    },
    "trainer_config.lr": {
        "displayed_name": "Learning Rate",
        "description": "",
        "path": "trainer_config.lr",
        "data_type": "float",
        "mode": "default",
        "interfaces": ["run"],
        "type_info": {
            "min": 0.1,
            "max": 1.0,
            'use_range': True
        }
    },
    "trainer_config.rollout_fragment_length": {
        "displayed_name": "Rollout Fragment Length",
        "description": "",
        "path": "trainer_config.rollout_fragment_length",
        "data_type": "int",
        "mode": "default",
        "interfaces": ["run"],
        "type_info": {
            "min": 5,
            "max": 100000,
            'use_range': True
        }
    },
    "trainer_config.soft_horizon": {
        "displayed_name": "Soft Horizon",
        "description": "",
        "path": "trainer_config.soft_horizon",
        "data_type": "bool",
        "mode": "default",
        "interfaces": ["run"],
        "type_info": {
        }
    },
    "trainer_config.model.dim": {
        "displayed_name": "Final resized frame dimension",
        "description": "Final resized frame dimension 84 or 42",
        "path": "trainer_config.model.dim",
        "data_type": "int",
        "mode": "default",
        "default": 84,
        "interfaces": [],
        "type_info": {
            "min": 1,
            "max": 1000
        }
    },
    "trainer_config.model.conv_filters": {
        "displayed_name": "Convolutional Filters",
        "description": "Convolutional Filters",
        "path": "trainer_config.model.conv_filters",
        "data_type": "list",
        "mode": "default",
        "default": "[[16, [4, 4], 2], [32, [4, 4], 2], [256, [11, 11], 1]]",
        "interfaces": [],
        "type_info": {
            "sub_type": "int"
        }
    },
    "trainer_config.model.fcnet_hiddens": {
        "displayed_name": "Fully Connected Hiddens",
        "description": "Size and number of hidden layers to be used. These are used if no custom model is specified and the input space is 1D",
        "path": "trainer_config.model.fcnet_hiddens",
        "data_type": "list",
        "mode": "default",
        "interfaces": [],
        "type_info": {
            "sub_type": "int"
        }
    },
    "trainer_config.model.fcnet_activation": {
        "displayed_name": "Activation function descriptor",
        "description": "Activation following each fully connected layer",
        "path": "trainer_config.model.fcnet_activation",
        "data_type": "category",
        "mode": "default",
        "interfaces": [],
        "type_info": {
            "options": ['tanh', 'relu', 'swish', 'silo', 'linear', 'None']
        }
    },
    "trainer_config.model.use_lstm": {
        "displayed_name": "Use LSTM",
        "description": "Include LSTM for state",
        "path": "trainer_config.model.use_lstm",
        "data_type": "bool",
        "mode": "default",
        "interfaces": [],
        "type_info": {
        }
    }
}
# "default": "[[16, [4, 4], 2], [32, [4, 4], 2], [256, [11, 11], 1]]"
MODEL_COMPONENT_PARAMS = {
    "components.model.__component_id": {
        "displayed_name": "Model Component",
        "description": "Select a Model Component",
        "path": "components.model.__component_id",
        "data_type": "component",
        "component_type": "ray.rllib.models.modelv2:ModelV2",
        "mode": "default",
        "type_info": {
        }
    }
}
# 'components': {'model': {'type': 'model', 'default': None}},


def get_default_trainer_config(trainer_config, overrides={}):
    new_trainer_config = copy.deepcopy(trainer_config)
    del new_trainer_config['callbacks']
    return merged_dict(new_trainer_config, overrides)


def agent_reg_entry(
        policy_class,
        trainer_class,
        trainer_config,
        assigned_params=DEFAULT_AGENT_PARAMS,
        auto_config_spaces=True,
        algo_type_ids=[],
        config_overrides={}):
    prepped_trainer_config = get_default_trainer_config(
        trainer_config, config_overrides)
    generated_params = create_params_from_dict(
        prepped_trainer_config, 'trainer_config', interfaces=[])
    params = merged_dict(generated_params, assigned_params)

    return {
        'policy_class': policy_class,
        "trainer_class": trainer_class,
        "trainer_config": prepped_trainer_config,
        "auto_config_spaces": auto_config_spaces,
        "default_trainer_config": trainer_config,
        "algo_type_ids": algo_type_ids,
        "params": params
    }


# NOTE: Using lambdas to speed up loading i.e. lazy loading
AGENT_REG = {
    "RANDOM": lambda: agent_reg_entry(
        RandomPolicy,
        randomPolicyTrainer,
        trainer.COMMON_CONFIG,
        config_overrides={"num_workers": 0},
        algo_type_ids=['RANDOM']),
    "PPO": lambda: agent_reg_entry(
        PPOTorchPolicy,
        ppo.PPOTrainer,
        ppo.DEFAULT_CONFIG,
        config_overrides={"num_workers": 0},
        algo_type_ids=['PPO']),
    "A3C": lambda: agent_reg_entry(
        PistarACTorchPolicy,
        a3c.A3CTrainer,
        a3c.DEFAULT_CONFIG,
        algo_type_ids=['A3C']),
    "A2C": lambda: agent_reg_entry(
        A3CTFPolicy,
        a2c.A2CTrainer,
        a2c.A2C_DEFAULT_CONFIG,
        config_overrides={'framework': 'tf', "num_workers": 0},
        algo_type_ids=['A2C']),
    "A2C_torch": lambda: agent_reg_entry(
        A3CTorchPolicy,
        a2c.A2CTrainer,
        a2c.A2C_DEFAULT_CONFIG,
        config_overrides={'framework': 'torch', "num_workers": 0},
        algo_type_ids=['A2C']),
    "SAC_torch": lambda: agent_reg_entry(
        SACTorchPolicy,
        sac.SACTrainer,
        sac.DEFAULT_CONFIG,
        config_overrides={'framework': 'torch', "num_workers": 0},
        algo_type_ids=['SAC']),
    "A2C_torch_v2": lambda: agent_reg_entry(
        PistarACTorchPolicy,
        a2c.A2CTrainer,
        a2c.A2C_DEFAULT_CONFIG,
        assigned_params=merged_dict(
            DEFAULT_AGENT_PARAMS, MODEL_COMPONENT_PARAMS),
        config_overrides={'framework': 'torch', "num_workers": 0},
        algo_type_ids=['A2C'])
}


class SessionEnvWrapper(RLMultiSessionEnv, MultiAgentEnv):

    def __init__(self, *args, **kwargs):
        RLMultiSessionEnv.__init__(self, *args, **kwargs)
        self.agents = self.players
        self.num_agents = len(self.players)


def env_creator(env_ctx: EnvContext):
    ctx.connect()
    task_id = env_ctx['task_id']
    session_config = env_ctx['session_config']
    env_spec_id = env_ctx['env_spec_id']
    env_kwargs = env_ctx['env_kwargs']
    agent_run_config = env_ctx['agent_run_config']
    batch_size = env_ctx.get('batch_size', 1)
    agent_id = env_ctx['agent_id']
    use_remote_client = env_ctx['use_remote_client']

    task: Task = Task.load(task_id)

    run_meta = {
        'worker_index': env_ctx.worker_index,
        'vector_index': env_ctx.vector_index
    }
    try:
        return SessionEnvWrapper(
            task_id=task_id,
            agent_id=agent_id,
            env_spec_id=env_spec_id,
            env_kwargs=env_kwargs,
            run_meta=run_meta,
            config=RLSessionConfig(**session_config),
            agent_run_config=agent_run_config,
            batch_size=batch_size,
            use_remote_client=use_remote_client,
            timeout_abort_check_callback=lambda: task.get_status() != STATE_RUNNING)

    except Exception as e:
        task.get_logger().error("Exception when starting session session uid {}, exception: {}\n {}".format(
            "??", e, traceback.format_exc()))


class RLLibCallbacks(DefaultCallbacks):

    def __init__(self, task_id, **kwargs):
        super().__init__(task_id, **kwargs)
        self.task_id = task_id

    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        pass

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        pass

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        pass

    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        pass

    def on_train_result(self, trainer, result: dict, **kwargs):
        pass

    def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass

# TODO add logger
# from ray.tune.logger import Logger
# class PistarlabLogger(Logger):

#     def _init(self):
#         self.update_config(self.config)

#     def on_result(self, result):
#         self.write("\n")
#         self.local_out.flush()

#     def write(self, b):
#         #LOGS
#         self.local_out.write(b)

#     def flush(self):
#         self.local_out.flush()

#     def close(self):
#         self.local_out.close()

#     def update_config(self, config):
#         pass


def make_callbacks(task_id):
    def cb(**kwargs):
        return RLLibCallbacks(task_id, **kwargs)
    return cb


class RLlibAgent(Agent):

    def initialize(self):
        pass

    def get_config(self, run_config={}):
        agent_config = self.get_dbmodel().config
        agent_config['trainer_config']['model'] = self.get_model_component_config()
        merged_config = merged_dict(agent_config, run_config)
        return merged_config

    def get_rllib_full_config(self):
        """
        Creates an approximation of the actual config that rllib would\n
        when including default values.
        """
        return merged_dict(self.get_default_config(), self.get_config())

    def get_model_component_config(self):
        agent_config = self.get_dbmodel().config
        model_config = agent_config['trainer_config'].get('model', {})
        component_def = agent_config.get('components')
        if component_def is None or 'model' not in component_def:
            return model_config
        else:

            model_component = self.get_component_by_name('model')
            if model_component is None:
                raise Exception(
                    "Component with name 'model' required for agent {}".format(self.get_id()))
            config = model_component.get_config()
            model_config['custom_model'] = model_component.get_spec_id()
            model_config['custom_model_config'] = config

            return model_config

    def get_policy_name(self):
        return self.get_config()['policy_name']

    def get_policy_class(self):
        return AGENT_REG[self.get_policy_name()]()['policy_class']

    def get_trainer_class(self):
        return AGENT_REG[self.get_policy_name()]()['trainer_class']

    def get_default_config(self):
        return {'trainer_config': AGENT_REG[self.get_policy_name()]['default_trainer_config']}

    def get_trainer_config(self,
                           env_spec_id=None,
                           env_kwargs={},
                           env_meta=None,
                           task_id=None,
                           batch_size=1,
                           session_config={},
                           agent_run_config={},
                           num_workers=0,
                           num_envs_per_worker=1,
                           num_gpus_per_worker=0,
                           num_gpus=0,
                           use_remote_client=False,
                           trainer_config_misc={}):
        agent_run_config = self.get_config(agent_run_config)

        env_spec = ctx.get_env_spec(env_spec_id)

        # TODO: Would be better if we created an instance of the env instead of depending on metadata
        # for example, need update metadata when using a wrapper
        if not env_meta:
            if use_remote_client:
                env_meta = copy.copy(env_kwargs)
                env_meta['players'] = list(
                    env_kwargs['observation_spaces'].keys())
                env_meta = pyson(env_meta)
            else:

                if len(session_config.get("wrappers", [])) > 0:
                    wrappers = session_config.get(
                        "wrappers") + env_spec.config['default_wrappers']
                    logging.error(
                        "WRAPPERS DEFINED AT RUNTIME ARE CURRENTLY NOT SUPPORTED.  not aborting, but probably will throw error")
                    env = get_wrapped_env_instance(
                        entry_point=env_spec.entry_point,
                        kwargs=env_kwargs,
                        wrappers=wrappers)
                    env_player_name = env.players[0]
                    env_meta['players'] = [
                        f'player_{i}' for i in range(batch_size)]
                    env_meta['observation_spaces'] = {
                        f'player_{i}': env.observation_spaces[env_player_name] for i in range(batch_size)}
                    env_meta['action_spaces'] = {
                        f'player_{i}': env.action_spaces[env_player_name] for i in range(batch_size)}
                else:
                    env_meta = copy.copy(env_spec.meta)

                    # TODO: Not clean or resusable
                    if env_spec.env_type == RL_SINGLEPLAYER_ENV:
                        env_meta['players'] = [
                            f'player_{i}' for i in range(batch_size)]
                        env_meta['observation_spaces'] = {
                            f'player_{i}': env_meta['observation_spaces']['default'] for i in range(batch_size)}
                        env_meta['action_spaces'] = {
                            f'player_{i}': env_meta['action_spaces']['default'] for i in range(batch_size)}
                    env_meta = pyson(env_meta)

        player_to_agent_map = {
            player: DEFAULT_POLICY_ID for player in env_meta['players']}

        # Config Policies - in this case, just one policy.
        default_observation_space = list(
            env_meta['observation_spaces'].values())[0]
        default_action_space = list(env_meta['action_spaces'].values())[0]
        policies_config = {}
        policies_config[DEFAULT_POLICY_ID] = (
            None, default_observation_space, default_action_space, agent_run_config)

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

        env_config = {}
        env_config['task_id'] = task_id
        env_config['session_config'] = session_config
        env_config['env_spec_id'] = env_spec_id
        env_config['env_kwargs'] = env_kwargs
        env_config['batch_size'] = batch_size
        env_config['agent_id'] = self.get_id()
        env_config['agent_run_config'] = agent_run_config
        env_config['use_remote_client'] = use_remote_client

        # Prepare trainer config only using valid keys
        trainer_config = copy.deepcopy(agent_run_config['trainer_config'])
        trainer_config['callbacks'] = make_callbacks(task_id)
        trainer_config['env'] = "RLLibSession"
        trainer_config['env_config'] = env_config
        trainer_config['multiagent'] = multiagent_config

        # TODO: Need to handle number of workers as standard parameter
        # trainer_config['num_workers'] = num_workers
        # trainer_config['num_envs_per_worker'] = num_envs_per_worker
        # trainer_config['num_gpus_per_worker'] = num_gpus_per_worker
        trainer_config['num_gpus'] = num_gpus
        trainer_config['seed'] = self.get_seed_as_int()

        # Merge with default config to create full trainer config
        trainer_config = merged_dict(
            AGENT_REG[self.get_policy_name()]()['default_trainer_config'],
            trainer_config)
        trainer_config = merged_dict(trainer_config, trainer_config_misc)
        return trainer_config

    def get_trainer(self, trainer_config={}):
        trainer_clss = self.get_trainer_class()
        trainer: Trainer = trainer_clss(config=trainer_config)
        agent_state = self.get_state()
        agent_states = {}
        if agent_state is not None:
            agent_states = {DEFAULT_POLICY_ID: self.get_state()}
            # if state is not None:
            #     trainer.set_weights({'default_policy':state})
        if (len(agent_states)) > 0:
            trainer.set_weights(agent_states)
        return trainer

    def get_policy(self):
        if self.get_config_key('auto_config_spaces', False):
            raise Exception(
                "auto_config_spaces is True. Cannot instantiate policy without knowing observation and action space.")

        # TODO: only needed if using TF
        ctx.tf_reset_graph()
        policy_class: Type[Policy] = self.get_policy_class()
        observation_space = pyson(self.get_config_key('observation_space'))
        action_space = pyson(self.get_config_key('action_space'))
        policy = policy_class(
            obs_space=observation_space,
            action_space=action_space,
            config={})
        state = self.get_state()
        if state:
            policy.set_weights(self.get_state())
        return policy

    def update_interfaces(self, trainer: Trainer, interface_id="run", policy_id=DEFAULT_POLICY_ID):

        if self.is_interface_auto_config(interface_id):
            policy: Policy = trainer.get_policy(policy_id)
            self.update_interface(
                interface_id,
                policy.observation_space,
                policy.action_space)

        # policy: Policy = trainer.get_policy(policy_id)

    def update_space_config(self, trainer: Trainer, interface_id="run", policy_id=DEFAULT_POLICY_ID):
        # TODO: Add to parent class

        interfaces = self.get_config_key("interfaces")
        interface = interfaces.get(interface_id, {})

        if interface.get('auto_config_spaces', False):
            policy: Policy = trainer.get_policy(policy_id)

            interface['observation_space'] = space_to_pyson(
                policy.observation_space)
            interface['action_space'] = space_to_pyson(policy.action_space)
            interface['auto_config_spaces'] = False
            interfaces[interface_id] = interface

            self.update_config_key('interfaces', interfaces)


class RLlibAgentRunner(AgentTaskRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        register_env("RLLibSession", env_creator)

    def run(self):
        task = self.get_task()

        task_id = task.get_id()
        agent: RLlibAgent = task.get_agent()

        # Get Task Config
        task_config = task.get_config()
        interface_id = task_config['interface_id']

        # Prepare trainer config only using valid keys
        # num_workers = task_config.get('num_workers', 0)
        checkpoint_freq = task_config['checkpoint_freq']

        session_count = 1
        total_gpus = 0
        # num_envs_per_worker = 1
        num_gpus = 0  # 0.0001
        # num_gpus_per_worker = 0 if num_workers == 0 else (total_gpus - num_gpus / num_workers)

        # Create trainer class
        trainer_config = agent.get_trainer_config(
            env_spec_id=task_config['env_spec_id'],
            env_kwargs=task_config['env_kwargs'],
            task_id=task_id,
            session_config=task_config['session_config'],
            agent_run_config=task_config['agent_run_config'],
            # num_workers=num_workers,
            batch_size=task_config['batch_size'],
            # num_envs_per_worker=num_envs_per_worker,
            # num_gpus_per_worker=num_gpus_per_worker,
            num_gpus=num_gpus,
            use_remote_client=task_config.get('use_remote_client', False))
        if task_id is not None:
            trainer_config_copy = copy.deepcopy(trainer_config)
            trainer_config_copy['callbacks'] = "NOT_SERIALIZED"
            trainer_config_copy['multiagent']['policy_mapping_fn'] = "NOT_SERIALIZED"
            ctx.get_store().save(
                key=(TASK_ENTITY, task_id),
                name="trainer_config",
                value=trainer_config_copy, flush=True)
        trainer = agent.get_trainer(trainer_config)
        agent.update_space_config(trainer, interface_id)
        num_workers = trainer_config['num_workers']
        num_envs_per_worker = trainer_config['num_envs_per_worker']
        if num_workers == 0:
            session_count = num_envs_per_worker
        else:
            session_count = num_workers * num_envs_per_worker

        max_timesteps = task_config['session_config'].get('max_steps')
        if max_timesteps is not None:
            max_timesteps *= session_count
        max_episodes = task_config['session_config'].get('max_episodes')
        if max_episodes is not None:
            max_episodes *= session_count

        train_summary = {}
        exit_state = None
        try:
            done = False
            terminated = False
            i = 0
            while not done:
                self.get_logger().info("Training Iteration {} - {}".format(task_id, i))

                # GET TRAINING SUMMARY
                train_summary = copy.copy(trainer.train())
                timestep_count = 0
                episode_count = None
                done = train_summary.get('done', False)
                if train_summary is None:
                    done = True
                    self.get_logger().info("Training Summary is Null, Terminating work for {}".format(task_id))
                else:
                    train_summary.pop('config')
                    ctx.get_store().save(key=(TASK_ENTITY, task_id),
                                         name="train_summary", value=train_summary)
                    timestep_count = train_summary.get('timesteps_total', 0)
                    episode_count = train_summary.get('episodes_total', 0)
                    self.get_logger().info(
                        f"\tTraining iteration complete {task_id} - {i}, total_timesteps: {timestep_count}, episodes_total:{episode_count}")

                # Check for terminination request
                if not self.is_running():
                    done = True

                # Check timestep and episode limits
                if not done and timestep_count and max_timesteps and timestep_count >= max_timesteps:
                    done = True
                    self.get_logger().info("Max Steps Reached for {}".format(task_id))
                elif not done and max_episodes and episode_count >= max_episodes:
                    done = True
                    self.get_logger().info("Max Episodes Reached for {}".format(task_id))

                task.update_summary(train_summary)
                agent.log_stat_dict(
                    task_id=task.get_id(),
                    data=train_summary['info']['learner'].get(DEFAULT_POLICY_ID, {}))

                # make checkpoint if needed
                if done or (i + 1) % checkpoint_freq == 0:
                    checkpoint_data = {}
                    checkpoint_data[F_TIMESTAMP] = time.time()
                    agent.save_state(state=trainer.get_weights()[
                                     DEFAULT_POLICY_ID], meta=checkpoint_data)
                    agent.flush_stats()

                i += 1
        finally:
            trainer.stop()
            agent.close()

        if terminated:
            exit_state = STATE_TERMINATED
        else:
            exit_state = STATE_COMPLETED

        return train_summary, exit_state
