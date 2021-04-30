import logging

from pistarlab.utils.pyson import pyson
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler

from .. import Agent, Task, ctx
from ..task_runner import AgentTask, TaskRunner


class TuneTrainable(tune.Trainable):

    def setup(self, config):
        ctx.connect()
        env_spec_id = config['env_spec_id']
        env_kwargs = config['env_kwargs']
        parent_task_id = config.get('parent_task_id')
        agent_config = config.get('agent_config')
        agent_spec_id = config.get('agent_spec_id')
        session_config = config.get('session_config')

        self.task = AgentTask.create(
            env_spec_id=env_spec_id,
            env_kwargs=env_kwargs,
            session_config=session_config,
            agent=Agent.create(agent_spec_id, config=agent_config),
            parent_task_id=parent_task_id)

        self.run_future = self.task.run()

    def step(self):
        stats = self.task.wait_for_next_summary_update()
        logging.info("TASK STATS: {}".format(stats))
        reward_total = stats.get('reward_total__task_sum', 0)
        episode_count = stats.get('episode_count__task_sum', 0)

        episode_reward_mean = 0
        if episode_count > 0:
            episode_reward_mean = reward_total / episode_count

        return dict(episode_reward_mean=episode_reward_mean,
                    episode_count=episode_count)

    def cleanup(self):
        logging.info("cleaning up")
        self.task._shutdown()
        result = self.run_future.get()
        print("FINAL RESULT: {}".format(result))


class TuneDefaultRunner(TaskRunner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        task = self.get_task()
        config = task.get_config()
        var_lookup = {'tune': tune}
        self.config = pyson(config, var_lookup=var_lookup)

        self.agent_spec_id = self.config['agent_spec_id']
        self.agent_config = self.config['agent_config']

        self.env_spec_id = self.config['env_spec_id']
        self.env_kwargs = self.config['env_kwargs']
        self.session_config = self.config['session_config']

        self.stop_config = self.config['stop_config']
        self.run_config = self.config['run_config']
        self.sched_config = self.config['sched_config']

    def run(self):

        task = self.get_task()
        sched = AsyncHyperBandScheduler(**self.sched_config)

        analysis = tune.run(
            TuneTrainable,
            name="TASK_ID__{}".format(task.get_id()),
            scheduler=sched,
            stop=self.stop_config,
            num_samples=self.run_config['num_samples'],
            config={
                'parent_task_id': task.get_id(),
                'agent_config': self.agent_config,
                'agent_spec_id': self.agent_spec_id,
                'env_spec_id': self.env_spec_id,
                'env_kwargs': self.env_kwargs,
                'session_config': self.session_config
            })

        summary = analysis.results_df.to_dict()
        task.update_summary(summary)
        return summary
