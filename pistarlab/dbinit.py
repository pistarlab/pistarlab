from sqlalchemy import orm as db_orm

from . import ctx
from .agents.random import RandomAgentRunner
from .dbmodels import *
from .tasks.matask import MultiAgentRunner
from .utils import env_helpers
from .utils.agent_helpers import get_agent_spec_interface_dict

BUILTIN_EXTENSION_ID = "builtin"


def add_wrapper_components(sess: db_orm.Session):
    # TODO: finish this
    pass


def load_default_data():
    sess = ctx.get_dbsession()
    try:
        for entity_name in [TASK_ENTITY, SESSION_ENTITY, AGENT_ENTITY, COMPONENT_ENTITY]:
            sess.add(SystemCounter(id=entity_name, value=0))
        sess.commit()
    except:
        sess.rollback()

    ctx.register_task_spec(
        spec_id="agent_task",
        displayed_name="Agent Task",
        description="built-in",
        extension_id=BUILTIN_EXTENSION_ID,
        version="0.0.1-dev",
        extension_version="0.0.1-dev",
        entry_point="pistarlab.task:AgentTask",
        runner_entry_point=None,
        config={
            'agent_id': None,
            'session_config': {},
            'agent_run_config': {},
            'env_spec_id': None,
            'env_kwargs': {},

        })
    ctx.register_task_spec(
        spec_id="hyperparam_exp",
        displayed_name="Hyper Parameter Experiment",
        description='Uses Ray Tune for Hyper Parameter Experiments: https://docs.ray.io/en/master/tune/',
        version="0.0.1-dev",
        disabled=False,
        entry_point='pistarlab.task:Task',
        extension_id=BUILTIN_EXTENSION_ID,
        extension_version="0.0.1-dev",
        runner_entry_point="pistarlab.tasks.tune_default:TuneDefaultRunner",
        config={
            'agent_spec_id': 'REQUIRED',
            'agent_config': {
                'trainer_config': {'lr': {'func_name': 'tune.uniform', 'args': [0.001, 0.1]}}},
            'env_spec_id': 'REQUIRED',
            'env_kwargs': {},
            'session_config': {
                'max_episodes': None,
                'max_steps': None,
                'max_steps_in_episode': 500,
                'episode_record_freq': 500,
                'step_log_freq': 50,
                'episode_record_preview_interval': 1,
                'episode_log_freq': 100,
                'preview_rendering': False,
                'meta': {},
                'wrappers': []},
            'stop_config': {'episode_count': 30},
            'run_config': {'num_samples': 3},
            'sched_config': {
                'time_attr': 'training_iteration',
                'metric': 'episode_reward_mean',
                'mode': 'max',
                'max_t': 400,
                'grace_period': 20}}
    )

    ctx.register_agent_spec_from_classes(
        runner_cls=RandomAgentRunner,
        spec_id="RandomAgent",
        config={
                'interfaces': {'run': get_agent_spec_interface_dict()}
        },
        extension_id=BUILTIN_EXTENSION_ID,
        algo_type_id="RANDOM")


    # MULTI AGENT RUNNER (OVER NETWORK)
    ctx.register_task_spec_from_class(
        MultiAgentRunner)

    spec_data = env_helpers.get_env_spec_data(
        spec_id="test_multiplayer_parallel",
        displayed_name="Test Environment: Multi-Agent Parallel",
        spec_displayed_name="Multi-Agent Parallel",
        env_type=RL_MULTIPLAYER_ENV,
        entry_point="pistarlab.envs.ma_test_envs:MultiAgentTestParallelEnv")
    
    spec_data['metadata'] = env_helpers.probe_env_metadata(spec_data)

    env_data = env_helpers.get_environment_data(
        environment_id="pistarlab_tests",
        displayed_name="Builtin Tests",
        collection="piSTAR Lab",
        env_specs=[spec_data])

    ctx.register_environment(
        extension_id=BUILTIN_EXTENSION_ID,
        **env_data)
