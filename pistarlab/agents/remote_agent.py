import time

from .. import ctx
from ..meta import *
from ..session_config import RLSessionConfig
from ..session_env import RLMultiSessionEnv
from ..task_runner import AgentTaskRunner


class RemoteAgentRunner(AgentTaskRunner):

    def run(self) -> Tuple[Any, Any]:
        task = self.get_task()
        agent = task.get_agent()

        task_id = task.get_id()
        task_config = task.get_config()

        env_spec_id = task_config['env_spec_id']
        env_kwargs = task_config['env_kwargs']

        session_config = task_config['session_config']
        agent_run_config = agent.get_config(task_config['agent_run_config'])

        batch_size = task_config.get('batch_size', 1)
        use_remote_client = task_config.get('use_remote_client', False)

        start_time = time.time()

        # init data
        ep_count = 0
        step_count = 0
        action_dict = {}
        reset = True
        obs, rews, dones, infos = {}, {}, {}, {}

        # init run state
        running = True

        env = RLMultiSessionEnv(
            env_spec_id=env_spec_id,
            env_kwargs=env_kwargs,
            config=RLSessionConfig(**session_config),
            agent_id=agent.get_id(),
            agent_run_config=agent_run_config,
            batch_size=batch_size,
            task_id=task_id,
            use_remote_client=use_remote_client,
            timeout_abort_check_callback=lambda: task.get_status() != STATE_RUNNING)

        while running:
            if reset:
                obs = env.reset()
                rews, dones, infos = {}, {}, {}
                for k in obs.keys():
                    rews[k] = None
                    dones[k] = False
                    infos[k] = {}
                reset = False
                if step_count > 0:
                    ep_count += 1
            else:
                action_dict = {player_id: env.action_spaces[player_id].sample() for player_id in obs.keys()}
                #TODO: get action from redis
                obs, rews, dones, infos = env.step(action_dict)
                step_count += len(action_dict)

            if dones.get("__all__", False):
                reset = True
            if step_count % 10000 == 0:
                self.get_logger().info("Step Count: {}".format(step_count))

            if env.is_complete():
                running = False

            # Check for terminination request
            if not self.is_running():
                running = False

        pre_close_delta = time.time() - start_time
        delta = time.time() - start_time
        self.get_logger().info("Runtime in seconds: {}, Step_Count:{}, steps_per_second: {} preclose: {}".format(delta, step_count, step_count / delta, step_count / pre_close_delta))

        env.close()
        exit_state = STATE_COMPLETED

        return {}, exit_state
