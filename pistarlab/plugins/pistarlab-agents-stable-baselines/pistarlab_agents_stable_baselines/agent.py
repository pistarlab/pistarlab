from pistarlab.task import AgentTask
from pistarlab.session_env import RLSingleSessionEnv
from pistarlab.session_config import  RLSessionConfig
from stable_baselines3 import A2C, DDPG, DQN, HER, TD3, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm

from pistarlab.agent import Agent
from pistarlab.task_runner import AgentTaskRunner
from pistarlab.meta import *

import time
import inspect
import logging
from pistarlab import ctx

import threading

# SEE: https://github.com/hill-a/stable-baselines


class AgentCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, env, task, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.step_time = time.time()
        self.running = True
        self.last_check = 0
        self.check_freq_secs = 0.2
        self.task = task
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:


        # Check for abort
        if time.time() - self.last_check > self.check_freq_secs:
            if self.task.get_status() != STATE_RUNNING:
                self.running = False
            self.last_check = time.time()

        if self.running:
            self.running = not self.env.is_complete()
        
        if not self.running:
            logging.info("No longer running")

        return self.running

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


AGENT_REG = {
    "A2C": {'class': A2C, 'default_policy': "MlpPolicy"},
    "DDPG": {'class': DDPG, 'default_policy': "MlpPolicy"},
    "DQN": {'class': DQN, 'default_policy': "MlpPolicy"},
    "HER": {'class': HER, 'default_policy': "MlpPolicy"},
    "PPO": {'class': PPO, 'default_policy': "MlpPolicy"},
    "TD3": {'class': TD3, 'default_policy': "MlpPolicy"},
    "SAC": {'class': SAC, 'default_policy': "MlpPolicy"},
}

excluded_params = {'env', 'self', 'policy_kwargs', 'policy', 'full_tensorboard_log', 'tensorboard_log', '_init_setup_model', 'verbose'}

def import_agents(plugin_id):
    for data in list(AGENT_REG.values()):
        clss = data['class']
        clss_name = clss.__name__
        doc = inspect.getdoc(clss)
        model_args = {}

        # Extract Params
        # sig = inspect.signature(clss.__init__)
        # for name in sig.parameters:
        #     if name in excluded_params:
        #         continue
        #     param = sig.parameters[name]
        #     model_args[name] = "NO_DEFAULT" if param.default == inspect._empty else param.default

        ctx.register_agent_spec(
            spec_id='stable_baselines_{}'.format(clss_name),
            runner_entry_point='pistarlab_agents_stable_baselines.agent:StableBaselineAgentTaskRunner',
            description='StableBaseline\n\nhttps://stable-baselines.readthedocs.io/en/master/index.html\n\n{}'.format(doc),
            auto_config_spaces=True,
            plugin_id=plugin_id,
            config={
                'model_args': model_args,
                'policy': data['default_policy'],
                'policy_kwargs':{},
                'agent_cls_name': clss_name
            },
        )


class StableBaselineAgentTaskRunner(AgentTaskRunner):


    def run(self):
        
        task = self.get_task()
        agent = task.get_agent()

        task_id = task.get_id()
        task_config = task.get_config()

        # checkpoint_freq = task_config['checkpoint_freq']
        # status_check_freq_secs = task_config['status_check_freq_secs']
        env_spec_id = task_config['env_spec_id']
        env_kwargs = task_config['env_kwargs']
        use_remote_client = task_config.get('use_remote_client',False)

        session_config = task_config['session_config']
        agent_run_config = agent.get_config(task_config['agent_run_config'])


        env = RLSingleSessionEnv(
            env_spec_id=env_spec_id,
            env_kwargs=env_kwargs,
            config=RLSessionConfig(**session_config),
            agent_id=agent.get_id(),
            agent_run_config=agent_run_config,
            task_id=task_id,
            use_remote_client = use_remote_client,
            timeout_abort_check_callback = lambda: task.get_status() != STATE_RUNNING )

        agent_cls_name = agent_run_config['agent_cls_name']
        policy = agent_run_config['policy']
        policy_kwargs = agent_run_config['policy_kwargs']
        model_args = agent_run_config['model_args']
        

        modelcls = AGENT_REG[agent_cls_name]['class']
        model:BaseAlgorithm = modelcls(
            policy=policy,
            env=env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            **model_args)
        callback = AgentCallback(
            env=env,
            task=task,
            verbose=1)
        total_timesteps = env.config.max_steps
        if not total_timesteps:
            total_timesteps = 10000000  # TODO: make absolute max configurable

        learn_results = model.learn(total_timesteps=total_timesteps,callback=callback)
        self.get_logger().info("Learning compplete {}".format(learn_results))
        env.close()
        exit_state = STATE_COMPLETED

        return {}, exit_state
