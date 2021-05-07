

import copy
import os
import time
from collections.abc import Iterable


from . import ctx
from .dbmodels import EnvSpecModel
from .remote_env import (RemoteMultiAgentEnvClient,
                                  RemoteSingleAgentEnvClient)
from .session import DataSession, RLSession
from .util_funcs import merged_dict
from .utils.env_helpers import  get_wrapped_env_instance
from .utils.pyson import pyson

from .common import MAEnv
from .env_base import DataEnvironment
from .meta import *
from .session_config import DataSessionConfig, RLSessionConfig
from .task import Task


class IterableMonitor:

    def __init__(self, name, iterable: Iterable):
        self.name = name
        self.iterable: Iterable = iterable
        self.epoch_id = None
        self.total_batches = 0
        self.batch_id = 0
        self.last_inputs = None
        self.last_targets = None

    def __iter__(self):
        if self.epoch_id is None:
            self.epoch_id = 0
        self.epoch_id += 1
        self.batch_id = 0
        self.last_inputs = None
        self.last_targets = None

        self.iter = self.iterable.__iter__()
        self.next = self.__next__
        return self

    def __next__(self):
        self.last_inputs, self.last_targets = self.iter.__next__()
        self.total_batches += 1
        self.batch_id += 1
        return self.last_inputs, self.last_targets


class DataSessionEnv:

    def __init__(self,
                 task_id=None,
                 agent_id=None,
                 agent_run_config=None,
                 env_spec_id=None,
                 env_kwargs={},
                 run_meta={},
                 config: DataSessionConfig = None):
        self.metadata = {}  # TODO: just place holder, should put something here (required in stable baselines "dummy_vec_env.py")

        self.task_id = task_id
        self.agent_id = agent_id
        self.env_spec_id = env_spec_id
        self.agent_id = agent_id
        self.agent_run_config = agent_run_config

        self.run_meta = run_meta

        env_spec_model = self.get_spec_env_dbmodel()
        default_env_kwargs = env_spec_model.config.get('env_kwargs', {})
        self.env_kwargs = pyson(merged_dict(default_env_kwargs, env_kwargs))

        env_entry_point = env_spec_model.entry_point

        self.config = config or DataSessionConfig()

        self.flush_interval_secs = 1.0
        self.last_summary_flush = 0
        self._logger = None

        wrappers = self.config.wrappers + env_spec_model.config['default_wrappers']

        self.env: DataEnvironment = get_wrapped_env_instance(
            entry_point=env_entry_point,
            kwargs=self.env_kwargs,
            wrappers=wrappers)

        self.session = DataSession(
            task_id=task_id,
            agent_id=agent_id,
            agent_run_config=agent_run_config,
            env_spec_id=env_spec_id)

        self.data_iterable = IterableMonitor(
            'data',
            self.env.get_data_loader())

        self.eval_iterable = IterableMonitor(
            'eval',
            self.env.get_eval_loader())

        self.eval_running_results = []

    def get_data_loader(self):
        return self.data_iterable

    def get_eval_loader(self):
        self.eval_running_results = []
        return self.eval_iterable

    def log_metrics(self, **metrics):
        epoch_id = self.data_iterable.epoch_id
        batch_id = self.data_iterable.batch_id
        total_batches = self.data_iterable.total_batches
        metrics['timestamp'] = time.time()
        metrics['epoch_id'] = epoch_id
        metrics['batch_id'] = batch_id
        metrics['total_batches'] = total_batches
        return self.session.log_data_metrics(metrics)

    #TODO: mucho code overlap with above
    def log_evaluation_metrics(self, **metrics):
        epoch_id = self.data_iterable.epoch_id
        batch_id = self.data_iterable.batch_id
        total_batches = self.data_iterable.total_batches
        metrics['timestamp'] = time.time()
        metrics['epoch_id'] = epoch_id
        metrics['batch_id'] = batch_id
        metrics['total_batches'] = total_batches
        self.session.log_evaluation_metrics(metrics)
        return metrics

    # API for prediction only output
    def submit_predictions(self, name, outputs, predictions, **metrics):
        inputs = self.data_iterable.last_inputs
        metrics = self.log_metrics(**metrics)
        self.session.log_predictions(name, inputs, outputs, predictions, metrics)
        self.env.submit_predictions(outputs, predictions)

    def get_logger(self):
        if self._logger is None:
            self._logger = ctx.get_entity_logger(TASK_ENTITY, self.task_id)
        return self._logger

    def get_sess_config(self) -> DataSessionConfig:
        return self.config

    def get_spec_env_dbmodel(self) -> EnvSpecModel:
        return ctx.get_dbsession().query(EnvSpecModel).get(self.env_spec_id)

    def get_task(self) -> Task:
        return Task.load(self.task_id)

    def close(self):
        self.session.close()
        self.env.close()

    def evaluate(self, expected, predictions):
        return self.env.evaluate(expected, predictions)

    def is_complete(self):
        # TODO: support termination criteria
        return False


class BatchedEnv:

    def __init__(self, env_entry_point, env_kwargs, batch_size=1, wrappers=[]):
        self.env_entry_point = env_entry_point
        self.env_kwargs = env_kwargs
        self.batch_size = batch_size
        self.num_players = batch_size
        self.wrappers = wrappers
        self.max_num_players = batch_size
        self.players = [f"player_{i}" for i in range(self.num_players)]
        self.possible_players = self.players

        self.envs = {}
        self.observation_spaces = {}
        self.action_spaces = {}
        for k in self.players:
            ctx.get_logger().info(f"Loading env: {self.env_entry_point}, with {self.env_kwargs}")
            env = get_wrapped_env_instance(self.env_entry_point, self.env_kwargs, wrappers)
            self.observation_spaces[k] = env.observation_space
            self.action_spaces[k] = env.action_space
            self.envs[k] = env

        self.awaiting_reset = [k for k in self.players]
        self.enable_auto_reset = False

    def reset(self):
        # Forced reset
        obs = {}
        for k, env in self.envs.items():
            obs[k] = env.reset()
        self.awaiting_reset = []
        return obs

    def auto_reset(self):
        obs = {}
        for k in self.awaiting_reset:
            obs[k] = self.envs[k].reset()
        self.awaiting_reset = []
        return obs

    def step(self, action_dict):

        rews, dones, infos = {}, {}, {}

        # Auto reset if enabled
        if self.enable_auto_reset and len(self.awaiting_reset) > 0:
            obs = self.auto_reset()
            for k in obs.keys():
                rews[k] = None
                dones[k] = False
                infos[k] = {'start': True}
        else:
            obs = {}

        # Take steps for all actions provided
        for k, action in action_dict.items():
            ob, rew, done, info = self.envs[k].step(action)
            if done:
                # Used for auto reset
                self.awaiting_reset.append(k)
            obs[k] = ob
            rews[k] = rew
            dones[k] = done
            infos[k] = info

        # Set __all__ done key
        if len(dones) == self.num_players:
            dones["__all__"] = all(dones.values())

        return obs, rews, dones, infos

    def render(self, player_id, *args, **kwargs):
        return self.envs[player_id].render(*args, **kwargs)

    def close(self):
        for env in self.envs.values():
            env.close()


class RLSessionEnvBase:

    def __init__(self,
                 task_id=None,
                 agent_id=None,
                 env_spec_id=None,
                 env_kwargs={},
                 config: RLSessionConfig = None):
        self.metadata = {}  # TODO: just place holder, should put something here (required in stable baselines "dummy_vec_env.py")

        self.task_id = task_id
        self.agent_id = agent_id
        self.env_spec_id = env_spec_id
        env_spec_model = self.get_spec_env_dbmodel()
        self.env_entry_point = env_spec_model.entry_point
        self.env_type = env_spec_model.env_type


        self.env_config = copy.deepcopy(env_spec_model.config)

        self.env_kwargs = self.env_config['env_kwargs']
        self.env_kwargs.update(env_kwargs)

        self.config = config or RLSessionConfig()

        self.flush_interval_secs = 1.0
        self.last_summary_flush = 0
        self._logger = None

    def get_logger(self):
        if self._logger is None:
            self._logger = ctx.get_entity_logger(TASK_ENTITY, self.task_id)
        return self._logger

    def get_sess_config(self) -> RLSessionConfig:
        return self.config

    def get_spec_env_dbmodel(self) -> EnvSpecModel:
        return ctx.get_dbsession().query(EnvSpecModel).get(self.env_spec_id)

    def get_task(self) -> Task:
        return Task.load(self.task_id)


class RLMultiSessionEnv(RLSessionEnvBase):
    """
    TODO: This class serves too many purposes and the logic gets complicated. Consider switching to Factory for building objects that share an interface
    TODO: Consider adding Non-Logging Remote Session for consistancy
    TODO: FIX limitation by introducing batched multiplayer interface

    Options:
    NonBatched/Batched,
    Single/Multiplayer
    """

    def __init__(
            self,
            agent_id=None,
            agent_run_config={},
            use_remote_client=False,
            auto_reset=False,
            batch_size=1,
            run_meta={},
            child_session_configs={},
            timeout_abort_check_callback=lambda: False,
            *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.agent_id = agent_id
        self.agent_run_config = agent_run_config

        self.use_remote_client = use_remote_client
        self.run_meta = run_meta
        self.batch_size = batch_size
        self.child_session_configs = child_session_configs



        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = ""

        self.is_multiplayer = self.env_type == RL_MULTIPLAYER_ENV

        if self.is_multiplayer and self.batch_size > 1:
            raise Exception("batchsize must be '1' for multiagent environments")

        self.player_list = None

        self.task = self.get_task()

        # Only local sessions have a primary session
        self.session = None

        self.parent_session_id = None

        # Configure Environemnt (make sure to assign self.player_list)
        if self.use_remote_client:
            # Remote Env
            self.get_logger().info("Is RemoteMultiAgentClient with {}".format(self.env_kwargs))
            self.env = RemoteMultiAgentEnvClient(
                agent_id=agent_id,
                timeout_abort_check_callback=timeout_abort_check_callback,
                **self.env_kwargs)
            self.parent_session_id = self.env_kwargs['session_id']
            self.player_list = self.env.players
        elif self.is_multiplayer:
            # Local Env and Multi Agent
            self.get_logger().info("Is Local Multi Agent")
            wrappers = self.config.wrappers + self.env_config['default_wrappers']
            self.env: MAEnv = get_wrapped_env_instance(
                entry_point=self.env_entry_point,
                kwargs=self.env_kwargs,
                wrappers=wrappers)
            self.env.reset()
            self.player_list = self.env.players

            self.session = RLSession(
                config=self.config,
                task_id=self.task_id,
                agent_id=self.agent_id,
                agent_run_config=self.agent_run_config,
                env_spec_id=self.env_spec_id,
                session_type=RL_MULTIPLAYER_SINGLEAGENT_SESS,
                run_meta=run_meta,
                render_fn=lambda mode: self.env.render(mode=mode),
                persists_at_creation=True)
            self.parent_session_id = self.session.get_id()
            self.task.set_primary_session_by_id(self.parent_session_id)
        else:
            # Local Env and Batched
            self.get_logger().info(f"Is Batched Single Agent Env batch_size = {self.batch_size}")
            wrappers = self.config.wrappers + self.env_config['default_wrappers']
            self.get_logger().info(f"Wrappers {wrappers}")
            self.env = BatchedEnv(
                env_entry_point=self.env_entry_point,
                env_kwargs=self.env_kwargs,
                batch_size=self.batch_size,
                wrappers=wrappers)
            self.player_list = self.env.players

            if self.batch_size > 1:

                self.session = RLSession(
                    config=self.config,
                    task_id=self.task_id,
                    agent_id=self.agent_id,
                    agent_run_config=self.agent_run_config,
                    env_spec_id=self.env_spec_id,
                    session_type=RL_SINGLEPLAYER_BATCHED_SESS,
                    run_meta=run_meta,
                    render_fn=None,
                    persists_at_creation=True)
                self.parent_session_id = self.session.get_id()
                self.task.set_primary_session_by_id(self.parent_session_id)

        self.players = self.env.players
        self.num_players = self.env.num_players

        self.get_logger().info("Players {}".format(self.player_list))
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        self.get_logger().info(f"Obs Spaces {self.observation_spaces}")
        self.reset_all = True
        self.auto_reset = auto_reset

        self.child_sessions: Dict[str, RLSession] = {}
        for player_id in self.player_list:
            self.new_child_session(player_id)

        self.done_players = set()
        self.valid_action_dict = {}

    def get_child_summary_list(self):
        return {s.get_id(): s.get_summary() for s in self.child_sessions.values()}

    def get_child_session_player_ids(self):
        return {s.get_id(): player_id for player_id, s in self.child_sessions.items()}

    def new_child_session(self, player_id):

        env_run_meta = {
            'players': [player_id],
            'observation_spaces': {player_id: self.env.observation_spaces[player_id]},
            'action_spaces': {player_id: self.env.action_spaces[player_id]}
        }

        render_fn = lambda *args, **kwargs: self.env.render(player_id=player_id, *args, **kwargs)
        if player_id in self.child_session_configs:
            sess_config = RLSessionConfig(**self.child_session_configs[player_id])
        else:
            sess_config = copy.copy(self.get_sess_config())
            # Child sessions do not terminate on their own.  Make sure session_configuration reflects this.
            if self.session:
                sess_config.max_episodes = 0
                sess_config.max_steps = 0
                sess_config.max_steps_in_episode = 0

        session = RLSession(
            config=sess_config,
            task_id=self.task_id,
            agent_id=self.agent_id,
            parent_session_id=self.parent_session_id,
            agent_run_config=self.agent_run_config,
            env_spec_id=self.env_spec_id,
            env_run_meta=env_run_meta,
            render_fn=render_fn,
            session_type=RL_SINGLEPLAYER_SESS,
            persists_at_creation=True)
        if self.parent_session_id is None:
            self.task.set_primary_session_by_id(session.get_id())
        self.child_sessions[player_id] = session

    def reset(self):

        for k, session in self.child_sessions.items():
            session.before_reset()

        if self.session:
            self.session.before_reset()

        obs = self.env.reset()

        if self.session:
            self.session.after_reset(obs)

        for k, ob in obs.items():
            self.child_sessions[k].after_reset(ob)

        self.done_players = set()
        self.reset_all = False
        return obs

    def step(self, action_dict) -> Tuple[Dict[str,Any],Dict[str,Any],Dict[str,Any],Dict[str,Any]]:

        if self.auto_reset and self.reset_all:
            obs = self.reset()
            rews, dones, infos = {}, {}, {}
            for k, ob in obs.items():
                rews[k] = None
                dones[k] = False
                infos[k] = {'start': True}
        else:
            self.valid_action_dict = {}
            for player_id, action in action_dict.items():
                if player_id not in self.done_players:
                    self.valid_action_dict[player_id] = action
                    self.child_sessions[player_id].before_step(action)
                else:
                    self.get_logger().debug("Warning: agent id {} is complete. Action is invalid and will not be used".format(player_id))

            if self.session:
                self.session.before_step(self.valid_action_dict)

            obs, rews, dones, infos = self.env.step(self.valid_action_dict)

            if self.session:
                self.session.after_step(obs, rews, dones, infos, self.valid_action_dict)

            for player_id in obs.keys():
                action = self.valid_action_dict.get(player_id)
                self.child_sessions[player_id].after_step(
                    obs[player_id],
                    rews[player_id],
                    dones[player_id],
                    infos[player_id],
                    action)

        for player_id, done in dones.items():
            if done:
                self.done_players.add(player_id)

        if len(self.done_players) == len(self.child_sessions):
            dones['__all__'] = True

        # Check session limits, if reached - mark all as done
        if self.is_complete():
            dones = {player_id: True for player_id, v in dones.items()}
            dones["__all__"] = True

        # If done, reset on next step
        dones["__all__"] = dones.get("__all__", False)
        if dones["__all__"]:
            self.reset_all = True

        # save session stats
        cur_time = time.time()
        if self.reset_all or (((cur_time - self.last_summary_flush) > self.flush_interval_secs)):
            for session in self.child_sessions.values():
                session.save_summary()
            if self.session:
                self.session.save_summary()
            self.last_summary_flush = cur_time
        return obs, rews, dones, infos

    def is_complete(self):
        if self.session:
            return self.session.is_complete()
        else:
            # TODO, remote session?
            False

    def render(self, *args, **kwargs):
        return self.env(*args, **kwargs)

    def close(self):
        self.env.close()
        close_state = STATE_TERMINATED
        if self.session:
            close_state = self.session.close()

        for session in self.child_sessions.values():
            session.close(state=close_state)


class RLSingleSessionEnv(RLSessionEnvBase):
    """
    TODO: NOT CURRENTLY WORKING
    """

    def __init__(
            self,
            agent_id=None,
            agent_run_config={},
            use_remote_client=False,
            run_meta={},
            timeout_abort_check_callback=lambda: False,
            *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.agent_id = agent_id
        self.agent_run_config = agent_run_config

        self.use_remote_client = use_remote_client
        self.run_meta = run_meta

        env_spec_model = self.get_spec_env_dbmodel()
        env_entry_point = env_spec_model.entry_point

        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = ""

        self.player_id = None
        parent_session_id = None

        if self.use_remote_client:
            # Remote Env
            self.get_logger().info("Is RemoteMultiAgentClient with {}".format(self.env_kwargs))
            self.env = RemoteSingleAgentEnvClient(
                agent_id=agent_id,
                timeout_abort_check_callback=timeout_abort_check_callback,
                **self.env_kwargs)
            parent_session_id = self.env_kwargs['session_id']
        else:
            # Local Env
            assert(env_spec_model.env_type == RL_SINGLEPLAYER_ENV), "Cannot use RLSingleSessionEnv with Local Mode MultiPlayer Environments"
            self.get_logger().info("Is Local Single Agent Env")
            wrappers = self.config.wrappers + env_spec_model.config['default_wrappers']
            self.env = get_wrapped_env_instance(
                entry_point=env_entry_point,
                kwargs=self.env_kwargs,
                wrappers=wrappers)
            self.env.reset()

        self.session = RLSession(
            config=self.config,
            task_id=self.task_id,
            agent_id=self.agent_id,
            agent_run_config=self.agent_run_config,
            parent_session_id=parent_session_id,
            env_spec_id=self.env_spec_id,
            session_type=RL_SINGLEPLAYER_SESS,
            run_meta=run_meta,
            render_fn=lambda mode: self.env.render(mode=mode))

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        self.session.before_reset()
        ob = self.env.reset()
        self.session.after_reset(ob)
        return ob

    def step(self, action):
        self.session.before_step(action)
        ob, rew, done, info = self.env.step(action)
        self.session.after_step(ob, rew, done, info, action)
        # save session stats
        cur_time = time.time()
        if self.done or (((cur_time - self.last_summary_flush) > self.flush_interval_secs)):
            for session in self.child_sessions.values():
                session.save_summary()
            if self.session:
                self.session.save_summary()
            self.last_summary_flush = cur_time

        return ob, rew, done, info

    def is_complete(self):
        return self.session.is_complete()

    def close(self):
        self.env.close()
        self.session.close()
